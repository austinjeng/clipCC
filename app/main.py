from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Optional

import anyio
import torch
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel as PydanticBaseModel

from app.config import Settings
from app.errors.handlers import (
    DuplicateTokensError,
    InferenceConcurrencyError,
    InferenceTimeoutError,
    InvalidAggregationError,
    InvalidContrastParamsError,
    InvalidFpsError,
    InvalidLabelsError,
    InvalidPromptTemplateError,
    InvalidTemporalParamsError,
    NoFramesExtractedError,
    TokenTruncationError,
    UnsupportedFormatError,
)
from app.inference_runner import InferenceRunner
from app.middleware import RequestGateMiddleware
from app.models.model_manager import (
    ModelManager,
    NoModelLoadedError,
    ModelNotCachedError,
    InsufficientResourcesError,
)
from app.resource_gates import ResourceGates
from app.schemas.response import (
    ClassifyMetadata,
    ClassifyResponse,
    HealthResponse,
    RawContrastParams,
    RawTemporalParams,
    ReadyResponse,
    ResolvedContrastOptions,
    ResolvedTemporalOptions,
)
from app.services.frame_timeline import FrameTimeline
from app.services.scoring import aggregate_frame_scores, VALID_CONTRAST_REDUCTIONS
from app.services.temporal_policy import get_policy
from app.services.video import FrameExtractor, probe_video, validate_video_constraints
from app.temp_store import TempStore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

DISCLAIMER_MEAN = (
    "Scores are relative to the supplied labels, not calibrated probabilities. "
    "Not suitable for safety-critical decisions."
)
DISCLAIMER_MAX = (
    "Scores are relative to the supplied labels, not calibrated probabilities. "
    "Max-mode scores are independent peaks per label and do not sum to 1. "
    "Not suitable for safety-critical decisions."
)
DISCLAIMER_CONTRAST = (
    "Contrast verdict is based on group score difference vs threshold. "
    "Model policy defaults are heuristic, not calibrated. "
    "Not suitable for safety-critical decisions."
)

STATIC_DIR = Path(__file__).parent / "static"


class LoadModelRequest(PydanticBaseModel):
    model_id: str


def _parse_label_array(raw: str, field_name: str) -> list[str]:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        raise InvalidLabelsError(f"{field_name} must be a valid JSON array of strings.")
    if not isinstance(parsed, list) or not all(isinstance(lb, str) for lb in parsed):
        raise InvalidLabelsError(f"{field_name} must be a valid JSON array of strings.")
    return parsed


def _validate_label_group(label_list: list[str], field_name: str, max_count: int = 50) -> None:
    if len(label_list) < 1 or len(label_list) > max_count:
        raise InvalidLabelsError(
            f"Number of {field_name} must be between 1 and {max_count} (inclusive)."
        )
    seen: set[str] = set()
    for lb in label_list:
        if not lb.strip():
            raise InvalidLabelsError("Each label must be a non-empty string.")
        if len(lb) > 200:
            raise InvalidLabelsError(
                f"Label '{lb[:50]}...' exceeds the maximum length of 200 characters."
            )
        if lb in seen:
            raise InvalidLabelsError(f"Duplicate label: '{lb}'.")
        seen.add(lb)


def create_app(settings: Optional[Settings] = None) -> RequestGateMiddleware:
    if settings is None:
        settings = Settings()

    settings.validate_auth_config()

    state: dict = {"manager": None, "temp_store": None, "gates": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager = ModelManager(
            cache_dir=settings.clip_cache_dir,
            offline=settings.clipcc_offline,
        )
        temp_store = TempStore(settings.temp_dir)
        temp_store.run_janitor()
        gates = ResourceGates(
            max_upload_concurrency=settings.effective_upload_concurrency,
            max_inference_concurrency=settings.max_concurrent_requests,
        )

        state["manager"] = manager
        state["temp_store"] = temp_store
        state["gates"] = gates

        async def _auto_load():
            try:
                await manager.load_model(settings.default_model_id)
                logger.info(f"Auto-loaded model: {settings.default_model_id}")
            except Exception as e:
                logger.error(f"Failed to auto-load model: {e}")

        if not settings.skip_model_autoload:
            asyncio.create_task(_auto_load())

        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/live", response_model=HealthResponse)
    async def live():
        return HealthResponse(status="ok")

    @app.get("/ready")
    async def ready():
        manager: Optional[ModelManager] = state.get("manager")
        if manager is None or manager.active_model is None:
            return JSONResponse(status_code=503, content={"detail": "Model not loaded"})
        return ReadyResponse(
            status="ready",
            model=manager.active_model_id or "",
            pretrained=manager.active_model_id or "",
            device=manager.active_model.device,
        )

    @app.get("/api/v1/models")
    async def list_models():
        manager: ModelManager = state["manager"]
        return manager.list_models()

    @app.post("/api/v1/models/load")
    async def load_model_endpoint(request: LoadModelRequest):
        manager: ModelManager = state["manager"]
        if request.model_id not in manager.registry:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": f"Unknown model_id: {request.model_id}",
                    "error_type": "ValueError",
                },
            )
        try:
            await manager.load_model(request.model_id)
        except ModelNotCachedError as e:
            return JSONResponse(
                status_code=409,
                content={
                    "detail": str(e),
                    "error_type": "ModelNotCachedError",
                },
            )
        except InsufficientResourcesError as e:
            return JSONResponse(
                status_code=422,
                content={
                    "detail": str(e),
                    "error_type": "InsufficientResourcesError",
                },
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "detail": f"Failed to load model: {str(e)}",
                    "error_type": type(e).__name__,
                },
            )
        return {"status": "loaded", "model_id": request.model_id}

    @app.get("/api/v1/models/active")
    async def active_model():
        manager: ModelManager = state["manager"]
        if manager.active_model is None:
            return JSONResponse(status_code=404, content={"detail": "No model loaded"})
        config = manager.registry[manager.active_model_id]

        from app.services.temporal_policy import ScoreSemantics

        model = manager.active_model
        semantics_str = ""
        if model.model_type == "siglip2":
            semantics_str = ScoreSemantics.SIGLIP2_SIGMOID
        elif model.model_type == "clip":
            semantics_str = ScoreSemantics.CLIP_RELATIVE_SOFTMAX

        temporal_defaults = None
        if semantics_str:
            try:
                policy = get_policy(semantics_str)
                temporal_defaults = {
                    "threshold": policy.default_threshold(),
                    "threshold_mode": policy.threshold_mode(),
                    "gap_tolerance": 2.0,
                    "min_duration": 1.0,
                }
            except ValueError:
                pass

        contrast_defaults = None
        if semantics_str:
            try:
                policy = get_policy(semantics_str)
                contrast_defaults = {
                    "threshold": policy.contrast_default_threshold(),
                    "contrast_reduce": policy.contrast_default_reduction(),
                    "label_pooling": policy.contrast_label_pooling(),
                }
            except ValueError:
                pass

        return {
            "model_id": config.model_id,
            "display_name": config.display_name,
            "model_type": config.model_type,
            "params": config.params,
            "resolution": config.resolution,
            "device": manager.active_model.device,
            "temporal_defaults": temporal_defaults,
            "contrast_defaults": contrast_defaults,
        }

    @app.get("/api/v1/labels/defaults")
    async def label_defaults():
        return {"labels": settings.default_labels}

    @app.get("/")
    async def serve_ui():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/static/vendor/{filename}")
    async def serve_vendor(filename: str):
        path = STATIC_DIR / "vendor" / filename
        if not path.exists():
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        return FileResponse(path)

    @app.post("/api/v1/classify", response_model=ClassifyResponse)
    async def classify(
        video: UploadFile,
        labels: str | None = Form(default=None),
        positive_labels: str | None = Form(default=None),
        negative_labels: str | None = Form(default=None),
        prompt_template: str = Form(default="This is a photo of {}."),
        fps: float = Form(default=1.0),
        aggregation: str = Form(default="mean"),
        threshold: float | None = Form(default=None, ge=0.0, le=1.0),
        gap_tolerance: float | None = Form(default=None, ge=0.0, le=10.0),
        min_duration: float | None = Form(default=None, ge=0.0, le=10.0),
        contrast_reduce: str | None = Form(default=None),
    ):
        manager: Optional[ModelManager] = state.get("manager")
        temp_store: Optional[TempStore] = state.get("temp_store")
        gates: Optional[ResourceGates] = state.get("gates")

        filename = video.filename or ""
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(ext if ext else filename)

        if fps < 0.1 or fps > 5.0:
            raise InvalidFpsError(fps)

        if aggregation not in ("mean", "max", "temporal", "contrast"):
            raise InvalidAggregationError(aggregation)

        # Mutual exclusivity: contrast vs standard labels
        if aggregation == "contrast":
            if labels is not None:
                raise InvalidContrastParamsError(
                    "Use 'positive_labels' and 'negative_labels' with aggregation='contrast', not 'labels'."
                )
            if positive_labels is None or negative_labels is None:
                raise InvalidContrastParamsError(
                    "Both 'positive_labels' and 'negative_labels' are required with aggregation='contrast'."
                )
            if contrast_reduce is not None and contrast_reduce not in VALID_CONTRAST_REDUCTIONS:
                raise InvalidContrastParamsError(
                    f"Invalid contrast_reduce '{contrast_reduce}'. "
                    f"Valid: {', '.join(sorted(VALID_CONTRAST_REDUCTIONS))}."
                )
        else:
            if positive_labels is not None or negative_labels is not None:
                raise InvalidContrastParamsError(
                    "'positive_labels' and 'negative_labels' are only valid with aggregation='contrast'."
                )
            if contrast_reduce is not None:
                raise InvalidContrastParamsError(
                    "'contrast_reduce' is only valid with aggregation='contrast'."
                )
            if labels is None:
                raise InvalidLabelsError("labels must be a valid JSON array of strings.")

        raw_temporal = RawTemporalParams(
            threshold=threshold if aggregation == "temporal" else None,
            gap_tolerance=gap_tolerance,
            min_duration=min_duration,
        )
        if aggregation not in ("temporal", "contrast") and raw_temporal.has_any():
            raise InvalidTemporalParamsError()

        # Parse labels based on mode
        pos_count = 0
        if aggregation == "contrast":
            parsed_pos = _parse_label_array(positive_labels, "positive_labels")
            parsed_neg = _parse_label_array(negative_labels, "negative_labels")
            _validate_label_group(parsed_pos, "positive_labels", max_count=50)
            _validate_label_group(parsed_neg, "negative_labels", max_count=50)
            # Cross-group uniqueness
            all_labels_set: set[str] = set()
            for lb in parsed_pos + parsed_neg:
                if lb in all_labels_set:
                    raise InvalidLabelsError(f"Duplicate label across groups: '{lb}'.")
                all_labels_set.add(lb)
            parsed_labels = parsed_pos + parsed_neg
            pos_count = len(parsed_pos)
        else:
            parsed_labels = _parse_label_array(labels, "labels")
            _validate_label_group(parsed_labels, "labels", max_count=50)

        brace_count = prompt_template.count("{}")
        if brace_count != 1:
            raise InvalidPromptTemplateError(
                "prompt_template must contain exactly one '{}' placeholder."
            )
        if len(prompt_template) > 500:
            raise InvalidPromptTemplateError(
                "prompt_template must be 500 characters or fewer."
            )

        prompts = [prompt_template.replace("{}", lb) for lb in parsed_labels]

        try:
            async with manager.acquire(timeout=settings.request_timeout_seconds) as lease:
                model = lease.model

                token_counts = model.validate_prompts(prompts)
                for prompt, count in zip(prompts, token_counts):
                    if count > model.max_token_length:
                        raise TokenTruncationError(prompt, count)

                raw_tokens = model.tokenize_raw(prompts)
                for i in range(len(raw_tokens)):
                    for j in range(i + 1, len(raw_tokens)):
                        if torch.equal(raw_tokens[i], raw_tokens[j]):
                            raise DuplicateTokensError(parsed_labels[i], parsed_labels[j])

                request_id = str(uuid.uuid4())
                start_time = time.monotonic()

                try:
                    stored = await anyio.to_thread.run_sync(
                        temp_store.save_upload, request_id, video.file
                    )

                    video_info = await anyio.to_thread.run_sync(
                        partial(
                            probe_video,
                            stored.path,
                            timeout=settings.ffmpeg_timeout_seconds,
                        )
                    )
                    validate_video_constraints(video_info, settings, fps)

                    temporal_opts = None
                    timeline = None
                    policy = None

                    async with gates.inference_admission():
                        runner = InferenceRunner(timeout_seconds=settings.request_timeout_seconds)

                        def pipeline(cancel_event, runner_ref):
                            frame_dir = temp_store.create_frame_dir(request_id)
                            extractor = FrameExtractor(ffmpeg_timeout=settings.ffmpeg_timeout_seconds)
                            frame_samples = extractor.extract(
                                video_path=stored.path,
                                fps=fps,
                                max_frames=settings.max_frames,
                                frame_dir=frame_dir,
                                cancel_event=cancel_event,
                                runner=runner_ref,
                            )

                            all_batches = []
                            all_frames = []

                            for batch_start in range(0, len(frame_samples), settings.batch_size):
                                if cancel_event.is_set():
                                    break
                                batch = frame_samples[batch_start: batch_start + settings.batch_size]
                                images = [Image.open(fs.path).convert("RGB") for fs in batch]

                                score_batch_result = model.score_batch(images, prompts)
                                all_batches.append(score_batch_result)
                                all_frames.extend(batch)

                                for fs in batch:
                                    try:
                                        fs.path.unlink(missing_ok=True)
                                    except Exception:
                                        pass

                            return all_batches, all_frames

                        result = await runner.run(pipeline)

                        if result is None:
                            raise InferenceTimeoutError(settings.request_timeout_seconds)

                        all_batches, all_frames = result

                        if not all_frames:
                            raise NoFramesExtractedError()

                        contrast_opts = None
                        if aggregation == "contrast":
                            batch_semantics = all_batches[0].semantics if all_batches else ""
                            policy = get_policy(batch_semantics)
                            raw_contrast = RawContrastParams(
                                threshold=threshold,
                                contrast_reduce=contrast_reduce,
                            )
                            contrast_opts = ResolvedContrastOptions.resolve(
                                raw_contrast,
                                policy.contrast_default_threshold(),
                                policy.contrast_default_reduction(),
                            )
                        elif aggregation == "temporal":
                            batch_semantics = all_batches[0].semantics if all_batches else ""
                            policy = get_policy(batch_semantics)
                            temporal_opts = ResolvedTemporalOptions.resolve(
                                raw_temporal, policy.default_threshold()
                            )
                            timeline = FrameTimeline(all_frames, fps, video_info.duration)

                    agg_result = aggregate_frame_scores(
                        all_batches, parsed_labels, all_frames, aggregation,
                        temporal_options=temporal_opts,
                        timeline=timeline,
                        policy=policy,
                        contrast_options=contrast_opts,
                        pos_count=pos_count,
                    )

                    processing_time = time.monotonic() - start_time
                    if aggregation == "contrast":
                        disclaimer = DISCLAIMER_CONTRAST
                    elif aggregation == "max":
                        disclaimer = DISCLAIMER_MAX
                    else:
                        disclaimer = DISCLAIMER_MEAN
                    semantics = all_batches[0].semantics if all_batches else ""

                    return ClassifyResponse(
                        best_match=agg_result.best_match,
                        scores=agg_result.scores,
                        metadata=ClassifyMetadata(
                            frames_analyzed=len(all_frames),
                            video_duration_seconds=video_info.duration,
                            model=manager.active_model_id or "",
                            device=model.device,
                            aggregation=aggregation,
                            processing_time_seconds=round(processing_time, 3),
                            disclaimer=disclaimer,
                            model_type=model.model_type,
                            score_semantics=semantics,
                        ),
                        temporal=agg_result.temporal,
                        contrast=agg_result.contrast,
                    )

                except InferenceConcurrencyError:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Too many inference requests in progress. Please retry in a moment."},
                    )
                finally:
                    temp_store.cleanup(request_id)

        except NoModelLoadedError:
            return JSONResponse(
                status_code=503,
                content={"detail": "No model loaded. Load a model first via POST /api/v1/models/load."},
            )
        except TimeoutError:
            raise InferenceTimeoutError(settings.request_timeout_seconds)

    return RequestGateMiddleware(
        app=app,
        gates=ResourceGates(
            max_upload_concurrency=settings.effective_upload_concurrency,
            max_inference_concurrency=settings.max_concurrent_requests,
        ),
        api_key=settings.api_key,
        max_body_bytes=settings.max_file_size_bytes,
    )
