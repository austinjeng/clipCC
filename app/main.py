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
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel as PydanticBaseModel

from app.config import Settings
from app.errors.handlers import (
    DuplicateTokensError,
    GemmaOutputParseError,
    InferenceConcurrencyError,
    InferenceTimeoutError,
    InvalidAggregationError,
    InvalidContrastParamsError,
    InvalidFpsError,
    InvalidGemmaParamsError,
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
from app.models.residency import ResidencyLedger
from app.models.vlm_slot import VlmSlot, VlmState
from app.schemas.gemma import (
    GemmaLabelScoresResponse,
    GemmaLatency,
    GemmaMetadata,
    GemmaQAResponse,
    GemmaStatusResponse,
)
from app.services.gemma_prompts import (
    build_label_scores_prompt,
    build_qa_prompt,
    label_scores_token_budget,
    parse_label_scores,
)
from app.services.gemma_sampler import (
    extract_frames as gemma_extract_frames,
    plan_timestamps,
    resolve_window,
    validate_gemma_video,
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

JANITOR_INTERVAL_SECONDS = 600

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

    ledger = ResidencyLedger(headroom_gb=settings.residency_headroom_gb)

    def _gemma_loader():
        # Lazy import: unit tests and SigLIP2-only deployments never touch
        # the multimodal transformers stack.
        from app.models.gemma_vlm import GemmaVLM
        return GemmaVLM(
            hf_repo=settings.gemma_model_id,
            cache_dir=settings.clip_cache_dir,
            image_token_budget=settings.gemma_image_token_budget,
        )

    def _gemma_device() -> str:
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"  # mps draws from host RAM; ledger treats both as host memory

    vlm_slot = VlmSlot(
        loader=_gemma_loader,
        ledger=ledger,
        device=_gemma_device(),
        reserve_bytes=int(settings.gemma_reserve_gb * 1e9),
        enabled=settings.gemma_enabled,
    )

    _temp_store = TempStore(settings.temp_dir)
    _gates = ResourceGates(
        max_upload_concurrency=settings.effective_upload_concurrency,
        max_inference_concurrency=settings.max_concurrent_requests,
    )
    state: dict = {"manager": None, "temp_store": _temp_store, "gates": _gates}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager = ModelManager(
            cache_dir=settings.clip_cache_dir,
            offline=settings.clipcc_offline,
            ledger=ledger,
        )
        _temp_store.run_janitor()

        state["manager"] = manager

        async def _auto_load():
            try:
                await manager.load_model(settings.default_model_id)
                logger.info(f"Auto-loaded model: {settings.default_model_id}")
            except Exception as e:
                logger.error(f"Failed to auto-load model: {e}")

        async def _janitor_loop():
            # Periodic cleanup of orphaned temp dirs (e.g. from a hard crash).
            # Offloaded so the filesystem walk never blocks the event loop.
            while True:
                await asyncio.sleep(JANITOR_INTERVAL_SECONDS)
                try:
                    await anyio.to_thread.run_sync(_temp_store.run_janitor)
                except Exception as e:
                    logger.warning(f"Janitor sweep failed: {e}")

        # Retain references: asyncio holds only a weak ref to tasks, so an
        # unreferenced task can be garbage-collected mid-run.
        background_tasks: list[asyncio.Task] = []
        if not settings.skip_model_autoload:
            background_tasks.append(asyncio.create_task(_auto_load()))
        background_tasks.append(asyncio.create_task(_janitor_loop()))
        state["background_tasks"] = background_tasks

        try:
            yield
        finally:
            # Graceful shutdown: cancel and drain the background tasks instead
            # of abandoning them when the event loop tears down.
            for task in background_tasks:
                task.cancel()
            for task in background_tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.warning(f"Background task error during shutdown: {e}")

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

    @app.get("/gemma")
    async def serve_gemma_ui():
        return FileResponse(STATIC_DIR / "gemma.html")

    @app.get("/api/v1/gemma/status", response_model=GemmaStatusResponse)
    async def gemma_status():
        return GemmaStatusResponse(
            enabled=vlm_slot.enabled,
            state=vlm_slot.state.value,
            error=vlm_slot.error,
            model_id=settings.gemma_model_id,
            device=vlm_slot.device,
        )

    @app.post("/api/v1/gemma/warm", status_code=202)
    async def gemma_warm():
        try:
            state_now = await vlm_slot.warm()
        except InsufficientResourcesError as e:
            return JSONResponse(status_code=503, content={"detail": str(e)})
        except RuntimeError as e:
            return JSONResponse(status_code=409, content={"detail": str(e)})
        return JSONResponse(status_code=202, content={"state": state_now.value})

    def _check_gemma_upload(video: UploadFile) -> None:
        filename = video.filename or ""
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(ext if ext else filename)
        if vlm_slot.state != VlmState.LOADED:
            # Defensive double-check; the middleware pre-body gate (503) is primary.
            raise HTTPException(
                status_code=503,
                detail="Gemma model is not loaded. Trigger loading via POST /api/v1/gemma/warm.",
            )

    async def _run_gemma_pipeline(
        video: UploadFile,
        window_start: float,
        prompt_for_model: str,
        max_new_tokens: int,
        parse_fn,  # (text) -> parsed | raises ValueError; None = return raw text
    ):
        """Shared label_scores/qa pipeline. Returns (payload, window, n_frames,
        video_info, latency, parse_retries)."""
        temp_store: TempStore = state["temp_store"]
        gates: ResourceGates = state["gates"]
        model = vlm_slot.model

        request_id = str(uuid.uuid4())
        try:
            stored = await anyio.to_thread.run_sync(
                temp_store.save_upload, request_id, video.file
            )
            video_info = await anyio.to_thread.run_sync(
                partial(probe_video, stored.path, timeout=settings.ffmpeg_timeout_seconds)
            )
            validate_gemma_video(video_info, settings)
            span_start, span_end = resolve_window(
                video_info.duration, window_start, settings.gemma_analysis_window_seconds
            )
            timestamps = plan_timestamps(span_start, span_end, settings.effective_gemma_max_frames)

            async with gates.vlm_admission():
                runner = InferenceRunner(timeout_seconds=settings.request_timeout_seconds)
                deadline = time.monotonic() + settings.request_timeout_seconds

                def pipeline(cancel_event, runner_ref):
                    t0 = time.monotonic()
                    frame_dir = temp_store.create_frame_dir(request_id)
                    frames = gemma_extract_frames(
                        stored.path, timestamps, frame_dir, cancel_event,
                        ffmpeg_timeout=settings.ffmpeg_timeout_seconds, runner=runner_ref,
                    )
                    if not frames:
                        raise NoFramesExtractedError()
                    images = [Image.open(f.path).convert("RGB") for f in frames]
                    t1 = time.monotonic()

                    text = ""
                    parsed = None
                    parse_err = None
                    attempts = 2 if parse_fn is not None else 1
                    used_attempts = 0
                    for attempt in range(attempts):
                        if cancel_event.is_set():
                            break
                        used_attempts = attempt + 1
                        text = model.generate(images, prompt_for_model, max_new_tokens, cancel_event)
                        if parse_fn is None:
                            break
                        try:
                            parsed = parse_fn(text)
                            parse_err = None
                            break
                        except ValueError as e:
                            parse_err = e
                    t2 = time.monotonic()
                    if parse_fn is not None and parsed is None:
                        raise GemmaOutputParseError(str(parse_err))
                    latency = GemmaLatency(
                        extract_seconds=round(t1 - t0, 3),
                        generate_seconds=round(t2 - t1, 3),
                        parse_seconds=0.0,  # parse happens inside the generate loop; negligible
                    )
                    n_retries = used_attempts - 1 if parse_fn is not None else 0
                    return (parsed if parse_fn is not None else text), len(frames), latency, n_retries

                result = await runner.run(pipeline)

                if result is None:
                    # Soft timeout: runner already waited for the worker to
                    # unwind; measure and log the overrun (spec §3 metric).
                    overrun = time.monotonic() - deadline
                    logger.warning(
                        f"Gemma request timed out; worker unwind overran the deadline by {overrun:.1f}s"
                    )
                    raise InferenceTimeoutError(settings.request_timeout_seconds)

            payload, n_frames, latency, parse_retries = result
            return payload, (span_start, span_end), n_frames, video_info, latency, parse_retries
        finally:
            temp_store.cleanup(request_id)

    def _gemma_metadata(window, n_frames, video_info, latency, parse_retries) -> GemmaMetadata:
        return GemmaMetadata(
            model=settings.gemma_model_id,
            device=vlm_slot.model.device if vlm_slot.model else vlm_slot.device,
            frames_analyzed=n_frames,
            window_start_seconds=window[0],
            window_end_seconds=window[1],
            video_duration_seconds=video_info.duration,
            latency=latency,
            parse_retries=parse_retries,
        )

    @app.post("/api/v1/gemma/label_scores", response_model=GemmaLabelScoresResponse)
    async def gemma_label_scores(
        video: UploadFile,
        labels: str = Form(),
        window_start: float = Form(default=0.0, ge=0.0),
    ):
        _check_gemma_upload(video)
        parsed_labels = _parse_label_array(labels, "labels")
        _validate_label_group(parsed_labels, "labels", max_count=settings.gemma_max_labels)

        prompt = build_label_scores_prompt(parsed_labels, settings.gemma_evidence_top_k)
        budget = label_scores_token_budget(len(parsed_labels))

        payload, window, n_frames, video_info, latency, retries = await _run_gemma_pipeline(
            video, window_start, prompt, budget,
            parse_fn=lambda text: parse_label_scores(text, parsed_labels),
        )
        return GemmaLabelScoresResponse(
            scores=payload,
            metadata=_gemma_metadata(window, n_frames, video_info, latency, retries),
        )

    @app.post("/api/v1/gemma/qa", response_model=GemmaQAResponse)
    async def gemma_qa(
        video: UploadFile,
        prompt: str = Form(),
        window_start: float = Form(default=0.0, ge=0.0),
    ):
        _check_gemma_upload(video)
        if not prompt.strip():
            raise InvalidGemmaParamsError("prompt must be a non-empty string.")
        if len(prompt) > 2000:
            raise InvalidGemmaParamsError("prompt must be 2000 characters or fewer.")

        payload, window, n_frames, video_info, latency, retries = await _run_gemma_pipeline(
            video, window_start, build_qa_prompt(prompt),
            settings.gemma_max_new_tokens_qa, parse_fn=None,
        )
        return GemmaQAResponse(
            answer=payload,
            metadata=_gemma_metadata(window, n_frames, video_info, latency, retries),
        )

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
            if gap_tolerance is not None or min_duration is not None:
                raise InvalidContrastParamsError(
                    "'gap_tolerance' and 'min_duration' are only valid with aggregation='temporal'."
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

    middleware = RequestGateMiddleware(
        app=app,
        gates=_gates,  # single shared instance — middleware upload gate and route limiters share one pool
        api_key=settings.api_key,
        max_body_bytes=settings.max_file_size_bytes,
        vlm_state=lambda: vlm_slot.state.value,
    )
    # Test seam: lets tests inject a fake loader and inspect slot state.
    middleware.vlm_slot_for_tests = vlm_slot
    return middleware
