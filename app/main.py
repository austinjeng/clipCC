from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.config import Settings
from app.errors.handlers import (
    DuplicateTokensError,
    InferenceConcurrencyError,
    InferenceTimeoutError,
    InvalidAggregationError,
    InvalidFpsError,
    InvalidLabelsError,
    InvalidPromptTemplateError,
    TokenTruncationError,
    UnsupportedFormatError,
)
from app.inference_runner import InferenceRunner
from app.middleware import RequestGateMiddleware
from app.models.clip_model import ClipModel
from app.models.model_spec import ModelSpec
from app.resource_gates import ResourceGates
from app.schemas.response import (
    ClassifyMetadata,
    ClassifyResponse,
    HealthResponse,
    ReadyResponse,
)
from app.services.scoring import build_response_scores, compute_frame_scores
from app.services.video import FrameExtractor, probe_video, validate_video_constraints
from app.temp_store import TempStore

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
BATCH_SIZE = 8

DISCLAIMER_MEAN = (
    "Scores are relative to the supplied labels, not calibrated probabilities. "
    "Not suitable for safety-critical decisions."
)
DISCLAIMER_MAX = (
    "Scores are relative to the supplied labels, not calibrated probabilities. "
    "Max-mode scores are independent peaks per label and do not sum to 1. "
    "Not suitable for safety-critical decisions."
)


def create_app(settings: Optional[Settings] = None) -> RequestGateMiddleware:
    if settings is None:
        settings = Settings()

    settings.validate_auth_config()

    # State holders — populated during lifespan startup
    state: dict = {"clip_model": None, "temp_store": None, "gates": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load model spec
        baked_path = Path("/app/.baked_model")
        if baked_path.exists():
            spec = ModelSpec.from_baked_metadata(baked_path)
        else:
            spec = ModelSpec(
                model_name="ViT-B-32",
                pretrained="laion2b_s34b_b79k",
                cache_dir=settings.clip_cache_dir,
            )

        clip_model = ClipModel(spec)
        temp_store = TempStore(settings.temp_dir)
        temp_store.run_janitor()
        gates = ResourceGates(
            max_upload_concurrency=settings.effective_upload_concurrency,
            max_inference_concurrency=settings.max_concurrent_requests,
        )

        state["clip_model"] = clip_model
        state["temp_store"] = temp_store
        state["gates"] = gates

        yield

        # No explicit cleanup needed

    app = FastAPI(lifespan=lifespan)

    @app.get("/live", response_model=HealthResponse)
    async def live():
        return HealthResponse(status="ok")

    @app.get("/ready")
    async def ready():
        clip_model: Optional[ClipModel] = state.get("clip_model")
        if clip_model is None:
            return JSONResponse(status_code=503, content={"detail": "Model not loaded"})
        return ReadyResponse(
            status="ready",
            model=clip_model.spec.model_name,
            pretrained=clip_model.spec.pretrained,
            device=clip_model.device,
        )

    @app.post("/api/v1/classify", response_model=ClassifyResponse)
    async def classify(
        video: UploadFile,
        labels: str = Form(...),
        prompt_template: str = Form(default="a video of {}"),
        fps: float = Form(default=1.0),
        aggregation: str = Form(default="mean"),
    ):
        clip_model: Optional[ClipModel] = state.get("clip_model")
        temp_store: Optional[TempStore] = state.get("temp_store")
        gates: Optional[ResourceGates] = state.get("gates")

        # --- Validation ---

        # Check file extension
        filename = video.filename or ""
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(ext if ext else filename)

        # Check fps range
        if fps < 0.1 or fps > 5.0:
            raise InvalidFpsError(fps)

        # Check aggregation
        if aggregation not in ("mean", "max"):
            raise InvalidAggregationError(aggregation)

        # Parse labels JSON
        try:
            parsed_labels = json.loads(labels)
        except (json.JSONDecodeError, ValueError):
            raise InvalidLabelsError("labels must be a valid JSON array of strings.")

        if not isinstance(parsed_labels, list) or not all(
            isinstance(lb, str) for lb in parsed_labels
        ):
            raise InvalidLabelsError("labels must be a valid JSON array of strings.")

        # Check label count 3-10
        if len(parsed_labels) < 3 or len(parsed_labels) > 10:
            raise InvalidLabelsError(
                "Number of labels must be between 3 and 10 (inclusive)."
            )

        # Check each label non-empty and ≤200 chars
        for lb in parsed_labels:
            if not lb.strip():
                raise InvalidLabelsError("Each label must be a non-empty string.")
            if len(lb) > 200:
                raise InvalidLabelsError(
                    f"Label '{lb[:50]}...' exceeds the maximum length of 200 characters."
                )

        # Check no duplicates
        seen: set[str] = set()
        for lb in parsed_labels:
            if lb in seen:
                raise InvalidLabelsError(f"Duplicate label: '{lb}'.")
            seen.add(lb)

        # Validate prompt template
        brace_count = prompt_template.count("{}")
        if brace_count != 1:
            raise InvalidPromptTemplateError(
                "prompt_template must contain exactly one '{}' placeholder."
            )
        if len(prompt_template) > 500:
            raise InvalidPromptTemplateError(
                "prompt_template must be 500 characters or fewer."
            )

        # Build prompts using replace (not .format())
        prompts = [prompt_template.replace("{}", lb) for lb in parsed_labels]

        # Token truncation check
        token_counts = clip_model.tokenize_and_check(prompts, max_tokens=77)
        for prompt, count in zip(prompts, token_counts):
            if count > 77:
                raise TokenTruncationError(prompt, count)

        # Duplicate token sequence check
        raw_tokens = clip_model.tokenize_raw(prompts)
        for i in range(len(raw_tokens)):
            for j in range(i + 1, len(raw_tokens)):
                if torch.equal(raw_tokens[i], raw_tokens[j]):
                    raise DuplicateTokensError(parsed_labels[i], parsed_labels[j])

        # --- Processing ---
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()

        try:
            # Save upload
            stored = temp_store.save_upload(request_id, video.file)

            # Probe and validate video
            video_info = probe_video(stored.path, timeout=settings.ffmpeg_timeout_seconds)
            validate_video_constraints(video_info, settings, fps)

            async with gates.inference_admission():
                runner = InferenceRunner(timeout_seconds=settings.request_timeout_seconds)

                # Pre-compute text features once (outside the pipeline thread is fine
                # since ClipModel methods are thread-safe under GIL)
                text_features = clip_model.encode_text(prompts)
                logit_scale = clip_model.model.logit_scale.exp().item()

                def pipeline(cancel_event, runner_ref):
                    frame_dir = temp_store.create_frame_dir(request_id)
                    extractor = FrameExtractor(ffmpeg_timeout=settings.ffmpeg_timeout_seconds)
                    frame_samples = extractor.extract(
                        video_path=stored.path,
                        fps=fps,
                        max_frames=settings.max_frames,
                        frame_dir=frame_dir,
                        cancel_event=cancel_event,
                    )

                    all_confidence: list[torch.Tensor] = []
                    all_raw_sim: list[torch.Tensor] = []
                    all_frames = []

                    for batch_start in range(0, len(frame_samples), BATCH_SIZE):
                        if cancel_event.is_set():
                            break
                        batch = frame_samples[batch_start: batch_start + BATCH_SIZE]
                        images = [Image.open(fs.path).convert("RGB") for fs in batch]

                        image_features = clip_model.encode_images(images)
                        cosine_sim = image_features @ text_features.T
                        confidence, raw_sim = compute_frame_scores(cosine_sim, logit_scale)

                        all_confidence.append(confidence)
                        all_raw_sim.append(raw_sim)
                        all_frames.extend(batch)

                        # Delete consumed frames
                        for fs in batch:
                            try:
                                fs.path.unlink(missing_ok=True)
                            except Exception:
                                pass

                    return (
                        torch.cat(all_confidence, dim=0),
                        torch.cat(all_raw_sim, dim=0),
                        all_frames,
                    )

                result = await runner.run(pipeline)

                if result is None:
                    raise InferenceTimeoutError(settings.request_timeout_seconds)

                all_confidence, all_raw_sim, all_frames = result

            scores, best_match = build_response_scores(
                all_confidence, all_raw_sim, parsed_labels, all_frames, aggregation
            )

            processing_time = time.monotonic() - start_time
            disclaimer = DISCLAIMER_MAX if aggregation == "max" else DISCLAIMER_MEAN

            return ClassifyResponse(
                best_match=best_match,
                scores=scores,
                metadata=ClassifyMetadata(
                    frames_analyzed=len(all_frames),
                    video_duration_seconds=video_info.duration,
                    model=clip_model.spec.model_name,
                    device=clip_model.device,
                    aggregation=aggregation,
                    processing_time_seconds=round(processing_time, 3),
                    disclaimer=disclaimer,
                ),
            )

        except InferenceConcurrencyError:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many inference requests in progress. Please retry in a moment."},
            )
        finally:
            temp_store.cleanup(request_id)

    # Wrap the FastAPI app in RequestGateMiddleware
    return RequestGateMiddleware(
        app=app,
        gates=ResourceGates(
            max_upload_concurrency=settings.effective_upload_concurrency,
            max_inference_concurrency=settings.max_concurrent_requests,
        ),
        api_key=settings.api_key,
        max_body_bytes=settings.max_file_size_bytes,
    )
