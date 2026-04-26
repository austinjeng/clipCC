# ClipCC: CLIP-Based Video Classification API

## Overview

A Dockerized FastAPI application that accepts a video file and a set of text labels, then uses OpenCLIP to score how well each label describes the video content. Returns confidence scores, raw similarities, and the best-matching label.

**Use case example:** Upload dashcam footage with labels like "drunk driving", "normal driving", "distracted driving" — the API returns which label best matches the video.

**Important:** CLIP produces relative confidence scores, not calibrated probabilities. Scores indicate how well each label matches relative to the other supplied labels. These scores are not suitable for safety-critical or legal decisions without additional validation.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Frame extraction | Uniform sampling at configurable fps (default 1.0) | Simple, predictable, effective for most classification |
| CLIP variant | OpenCLIP `ViT-L-14` / `laion2b_s32b_b82k` | Best accuracy/size tradeoff; better zero-shot than original CLIP |
| Output format | Raw similarities + relative confidence + best_match | Maximum flexibility for API consumers |
| Video limits | 5 min / 500MB / 300 max frames | Keeps synchronous API responsive and bounded |
| GPU strategy | Auto-detect, separate build profiles for CPU/CUDA | Reliable builds per target |
| Text prompts | Auto-template with optional custom prompts | Easy for casual users, flexible for power users |
| Architecture | Monolith FastAPI with concurrency controls | Simplest for v1; bounded by semaphore and frame cap |
| Security | Fail-closed API key + concurrency semaphore | Auth required by default; explicit opt-out for dev |
| Workers | Single uvicorn worker, pinned | Semaphore and model memory are process-local |

## API Design

### Endpoint

```
POST /api/v1/classify
```

### Request

`multipart/form-data` with fields:

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `video` | file | yes | — | Video file (mp4, avi, mov, mkv) |
| `labels` | JSON string | yes | — | Array of 3–10 text labels (non-empty, no duplicates, max 200 chars each) |
| `prompt_template` | string | no | `"a video of {}"` | Template for text prompts; must contain exactly one `{}` placeholder, max 500 chars |
| `fps` | float | no | `1.0` | Frame sampling rate (0.1–5.0) |
| `aggregation` | string | no | `"mean"` | Score aggregation: `"mean"` (dominant content) or `"max"` (any occurrence) |

### Response (200 OK) — Mean Aggregation

Confidences sum to 1.0 (distribution over labels).

```json
{
  "best_match": {
    "label": "normal driving",
    "confidence": 0.45
  },
  "scores": [
    {
      "label": "drunk driving",
      "confidence": 0.15,
      "raw_similarity": 0.27
    },
    {
      "label": "normal driving",
      "confidence": 0.45,
      "raw_similarity": 0.31
    }
  ],
  "metadata": {
    "frames_analyzed": 297,
    "video_duration_seconds": 297.4,
    "model": "ViT-L-14",
    "device": "cuda",
    "aggregation": "mean",
    "processing_time_seconds": 12.3,
    "disclaimer": "Scores are relative to the supplied labels, not calibrated probabilities. Not suitable for safety-critical decisions."
  }
}
```

### Response (200 OK) — Max Aggregation

Each label's confidence is its independent peak across frames. Scores do NOT sum to 1.

```json
{
  "best_match": {
    "label": "normal driving",
    "confidence": 0.72
  },
  "scores": [
    {
      "label": "drunk driving",
      "confidence": 0.38,
      "raw_similarity": 0.29,
      "peak_frame_index": 142,
      "approx_timestamp_seconds": 142.0
    },
    {
      "label": "normal driving",
      "confidence": 0.72,
      "raw_similarity": 0.34,
      "peak_frame_index": 85,
      "approx_timestamp_seconds": 85.0
    }
  ],
  "metadata": {
    "frames_analyzed": 297,
    "video_duration_seconds": 297.4,
    "model": "ViT-L-14",
    "device": "cuda",
    "aggregation": "max",
    "processing_time_seconds": 12.3,
    "disclaimer": "Scores are relative to the supplied labels, not calibrated probabilities. Max-mode scores are independent peaks per label and do not sum to 1. Not suitable for safety-critical decisions."
  }
}
```

### Error Responses

All errors return a JSON body with a human-friendly `detail` message.

| Status | Condition | Example message |
|---|---|---|
| 401 | Missing/invalid API key | "Invalid or missing API key. Provide a valid key in the X-API-Key header." |
| 413 | File too large | "Your video is 620MB, which exceeds the 500MB limit. Try trimming or compressing it." |
| 422 | Duration too long | "Video duration is 8m32s, which exceeds the 5-minute limit. Please upload a shorter clip." |
| 422 | Too many frames | "Your video would produce 1,500 frames (5m at 5fps), which exceeds the 300-frame limit. Lower the fps or use a shorter clip." |
| 422 | Resolution too high | "Video resolution is 7680x4320, which exceeds the 3840x2160 limit. Please use a lower-resolution source." |
| 422 | Multiple video streams | "Video contains 3 video streams. Only single-stream videos are supported." |
| 422 | Label count invalid | "Please provide between 3 and 10 labels. You provided 2." |
| 422 | Empty/duplicate label | "Labels must be non-empty, unique, and under 200 characters each." |
| 422 | Prompt exceeds token limit | "Prompt 'a video of extremely long label...' exceeds CLIP's 77-token context window (got 83 tokens). Shorten the label or template." |
| 422 | Duplicate after tokenization | "Labels 'driving slowly' and 'driving  slowly' produce identical token sequences. Use more distinct labels." |
| 422 | Invalid prompt template | "Prompt template must contain exactly one '{}' placeholder." |
| 422 | Invalid fps | "FPS must be between 0.1 and 5.0. You provided 10.0." |
| 422 | Invalid aggregation | "Aggregation must be 'mean' or 'max'. You provided 'median'." |
| 415 | Unsupported format | "Unsupported format '.webm'. Supported: mp4, avi, mov, mkv." |
| 429 | Upload concurrency limit | "Too many uploads in progress. Please retry in a moment." |
| 429 | Inference concurrency limit | "Server is processing the maximum number of videos (2). Please retry in a moment." |
| 500 | Processing failure | "An error occurred while processing your video. Please try again or use a different file." |
| 504 | Request timeout | "Processing exceeded the 300-second time limit and was cancelled. Try a shorter video or lower fps." |

## Processing Pipeline

```
Upload -> Validate -> Extract Frames -> Score -> Respond
```

### 0. Pre-Parse Gate (ASGI Middleware)

ASGI middleware runs **before** FastAPI/Starlette multipart parsing. Uses an app-scoped `ResourceGates` service that owns both limiters and exposes context-manager methods.

Path routing:
- `GET /live` — exempt from all gates (unauthenticated liveness probe)
- `GET /ready` — exempt from concurrency gates, respects auth config (readiness probe with model/device details)
- `POST /api/v1/classify` — all gates apply

For `POST /api/v1/classify`, `ResourceGates.upload_admission(scope)` runs these steps:

1. **Auth gate:** Read `X-API-Key` from `scope["headers"]`. If `API_KEY` is configured and the header doesn't match, respond 401 — zero body bytes consumed.
2. **Upload concurrency gate:** `anyio.CapacityLimiter(MAX_UPLOAD_CONCURRENCY)` with `acquire_nowait()`. If full, respond 429 ("Too many uploads in progress") — zero body bytes consumed.
3. **Body size gate:** Wrap the ASGI `receive` channel, counting bytes. If cumulative bytes exceed `MAX_FILE_SIZE_MB`, abort with 413 — body never fully materializes.

**Ownership:** The middleware is the sole owner of the upload limiter. It acquires on entry and releases in its own `finally` block after the downstream app returns. The route never touches the upload limiter directly. This prevents double-release or leaked slots.

The **inference limiter is NOT acquired here** — acquiring it before body parsing means slow uploads hold inference slots without doing inference. The upload concurrency gate is separate and higher-limit (default `MAX_CONCURRENT_REQUESTS + 2`) to allow a small upload queue beyond inference capacity.

**Temp disk budget:** During multipart parsing, Starlette spools the upload to disk. Then `TempStore.save_upload()` streams it to a controlled temp file. Briefly, both copies exist. Honest worst-case budget: `2 * MAX_UPLOAD_CONCURRENCY * MAX_FILE_SIZE_MB + frame_overhead`. Recommend mounting the temp directory on a size-limited tmpfs or volume in production. A future optimization can bypass `UploadFile` entirely by streaming the video part directly to disk in the ASGI middleware.

### 0.1 Inference Gate (Route-Level, Post-Parse)

After FastAPI parses the multipart request (upload slot still held by middleware's `finally`):

The route uses `ResourceGates.inference_admission()` to acquire the inference limiter, then delegates to `InferenceRunner`.

1. **Inference capacity:** `anyio.CapacityLimiter(MAX_CONCURRENT_REQUESTS)` with `acquire_nowait()`. If full, raises `WouldBlock` → return 429 immediately. Middleware's `finally` handles upload slot release and `TempStore` cleanup.

2. **InferenceRunner:** Encapsulates the entire blocking pipeline lifecycle. The route calls `runner.run(request) -> Result | TimeoutError`. Internally:
   - Owns the worker thread, `cancel_event: threading.Event`, active subprocess handle, and temp cleanup.
   - Blocking pipeline runs via `anyio.to_thread.run_sync`.
   - An async timer (`REQUEST_TIMEOUT_SECONDS`) runs concurrently. On timeout:
     - Sets `cancel_event` (pipeline checks between batches and before subprocess calls)
     - Kills any active ffmpeg subprocess via `subprocess_handle.kill()`
     - **Waits for the worker thread to actually exit** before releasing the inference limiter. No abandoned threads.
   - Returns `504` only after the worker has exited and `TempStore` cleanup is confirmed. Client may wait up to one extra batch.
   - On success/error: returns result, releases inference limiter, cleanup handled by `TempStore`.

3. **Limiter ownership:** Upload slot: owned by middleware `finally`. Inference slot: owned by `InferenceRunner`, released only after the worker thread exits (success, error, or post-timeout). Neither is released by the route directly.

**Trade-off:** A client whose upload succeeds but arrives when inference is full gets 429 after uploading — worse UX than pre-upload rejection, but correctly protects the scarce resource (GPU/CPU inference time).

### 1. Validate

- **Upload size:** Primary enforcement is the ASGI middleware (see step 0). Endpoint also checks actual file size via `TempStore` as a second gate.
- **Temp file handling via `TempStore`:** `TempStore.save_upload(upload) -> StoredUpload` streams `upload.file` content into a controlled temp directory in chunks (64KB). Returns a `StoredUpload` with the file path and size. Do NOT attempt to reuse Starlette's `SpooledTemporaryFile` path — its `.name` can be a file descriptor int, and renaming can confuse Starlette's cleanup. `TempStore` also owns frame extraction temp dirs, tracks aggregate disk usage, and provides `cleanup(request_id)` for guaranteed teardown.
- Probe with `ffprobe` (subprocess with argv list, 30s timeout) for duration, format, and stream validation:
  - Reject resolution > 3840x2160 (4K) or pixel count > 8.3MP (prevents ffmpeg decode cost explosion on 8K+ video)
  - Reject files with multiple video streams
  - Reject non-finite or non-positive duration
- Compute expected frame count: `duration * fps`. Reject if > `MAX_FRAMES` (default 300).
- Validate labels: count 3–10, non-empty, no duplicates, max 200 chars each
- Validate prompt template contains exactly one `{}` placeholder, max 500 chars. Template application uses `str.replace('{}', label)` (not `.format()`) to avoid crashes on stray braces in labels.
- **Token truncation check:** After applying the template, tokenize each prompt using OpenCLIP's tokenizer. Reject if any prompt exceeds CLIP's 77-token context window (would be silently truncated). Reject if any two prompts produce identical token sequences post-tokenization (labels that look different in text but become the same to the model).
- Fail fast with specific error at each step

### 2. Extract Frames

`FrameExtractor.extract(video_path, fps, max_frames, cancel_event) -> list[FrameSample]`

- Use `ffmpeg` via `subprocess.run` with argv list (never `shell=True`). Subprocess handle registered with `InferenceRunner` for timeout kill.
- Flags: `-nostdin`, `-v error`, subprocess timeout = `FFMPEG_TIMEOUT_SECONDS` (default 120)
- Command: `["ffmpeg", "-nostdin", "-i", input_path, "-vf", f"fps={fps},scale='min(512,iw)':'min(512,ih)':force_original_aspect_ratio=decrease", "-q:v", "2", "-frames:v", str(max_frames), output_pattern]`
- `-frames:v` hard-caps output frame count as a safety net
- Scale filter caps long side at 512px during extraction (CLIP resizes to 224px anyway; extracting 4K/8K frames is pure waste of disk and memory)
- Output as JPEG to `TempStore`-managed directory
- Returns `list[FrameSample]` where `FrameSample` is a dataclass: `path: Path`, `sample_index: int`, `approx_timestamp_seconds: float` (computed as `sample_index / fps`). **Note:** approximate — may not match source timestamps for VFR content, non-zero start times, or ffmpeg rounding. Accurate PTS-based timestamps via ffmpeg showinfo filter are a v2 enhancement.

### 3. Score

- Process frames in batches of `BATCH_SIZE` (default 32)
- Per batch: load frames as PIL images, run inference, delete consumed frame files immediately
- Text encoding computed once and cached for the request
- Compute cosine similarity per frame per label
- Compute scaled logits: `logit_scale.exp() * cosine_similarity` (using OpenCLIP's learned `logit_scale` parameter, typically ~100). This is required — softmax over raw cosines produces near-uniform, meaningless distributions.
- `confidence` = softmax over scaled logits. `raw_similarity` = unscaled cosine value.
- Aggregate based on `aggregation` parameter:
  - `mean`: mean of per-frame confidence scores (best for "what is this video mostly showing?"). `raw_similarity` is also the mean across frames. Confidences sum to 1.0.
  - `max`: for each label, return the highest confidence score across all frames independently (best for "did this happen at any point?"). `raw_similarity` comes from the same frame that produced the peak confidence. Each score entry includes `peak_frame_index` (0-based) and `approx_timestamp_seconds` (approximate — see Extract Frames). **Note:** max-mode confidences do NOT sum to 1 and tend to rise with more sampled frames, since each label's peak is selected independently.

### 4. Cleanup

- Delete temp video file and remaining frame files in `finally` block
- Guaranteed cleanup on success or error

### Startup

- **Auth gate:** If neither `API_KEY` nor `ALLOW_UNAUTHENTICATED=true` is set, refuse to start with a clear error message.
- **Model config:** Load `ModelSpec` from `/app/.baked_model` (single source of truth for model name, pretrained tag, cache path). No runtime env vars for model selection — the baked image defines the model.
- Load OpenCLIP model once at FastAPI startup (`@app.on_event("startup")`) using `ModelSpec` values with explicit `cache_dir` kwarg to ensure baked weights are found.
- Set model to evaluation mode immediately after loading (disables dropout/batchnorm training behavior).
- All inference runs under `torch.inference_mode()` context (disables gradient tracking, reduces memory). GPU inference additionally uses `torch.autocast("cuda")` for FP16 throughput.
- Keep model in memory for all requests
- Log device selection (cuda/cpu)
- Run `TempStore` janitor: delete any files older than 1 hour (handles crash leftovers)
- Expose two health endpoints:
  - `GET /live` — unauthenticated, exempt from all gates. Returns `200 {"status": "ok"}` if the process is running. For container liveness probes.
  - `GET /ready` — exempt from concurrency gates, respects auth config. Returns model name, device, and readiness status. For detailed readiness probes and monitoring.

## Docker & Infrastructure

### Dockerfile

- Base: `python:3.11-slim`
- System dependencies: `ffmpeg`
- Python dependencies: `fastapi`, `uvicorn`, `open_clip_torch`, `torch`, `Pillow`, `python-multipart`
- Build args:
  - `TORCH_VARIANT` selects torch wheel index:
    - `cpu`: `--index-url https://download.pytorch.org/whl/cpu` (~800MB image)
    - `cu121`: `--index-url https://download.pytorch.org/whl/cu121` (~4GB image)
  - `MODEL_NAME` (default `ViT-L-14`) and `PRETRAINED` (default `laion2b_s32b_b82k`) control which model is baked in
- `ENV CLIP_CACHE_DIR=/app/models` set explicitly in Dockerfile for both build and runtime. Prevents HOME-relative cache mismatches between build and runtime users.
- **Model weights baked into image** via `RUN python -c "import open_clip; open_clip.create_model_and_transforms('${MODEL_NAME}', pretrained='${PRETRAINED}', cache_dir='/app/models')"` during build. The `cache_dir` kwarg is required — OpenCLIP's internal downloader ignores env vars and defaults to `~/.cache/clip` without it. No runtime download.
- Build args are written to `/app/.baked_model` metadata file (model name, pretrained tag, cache path) so startup can validate.
- Uvicorn runs with `--workers 1` (semaphore and model memory are process-local; more workers would duplicate both).

### docker-compose.yml

Two profiles: `cpu` and `gpu`.

```yaml
services:
  clipcc-cpu:
    build:
      context: .
      args:
        TORCH_VARIANT: cpu
    ports:
      - "8000:8000"
    environment:
      - MAX_FILE_SIZE_MB=500
      - MAX_DURATION_SECONDS=300
      - MAX_FRAMES=300
      - DEFAULT_FPS=1.0
      - MAX_CONCURRENT_REQUESTS=2
      - ALLOW_UNAUTHENTICATED=true  # remove and set API_KEY for production
      # - API_KEY=your-secret-key
    profiles: ["cpu"]

  clipcc-gpu:
    build:
      context: .
      args:
        TORCH_VARIANT: cu121
    ports:
      - "8000:8000"
    environment:
      - MAX_FILE_SIZE_MB=500
      - MAX_DURATION_SECONDS=300
      - MAX_FRAMES=300
      - DEFAULT_FPS=1.0
      - MAX_CONCURRENT_REQUESTS=1  # default 1 for GPU to avoid VRAM overcommit; increase based on available VRAM
      - ALLOW_UNAUTHENTICATED=true  # remove and set API_KEY for production
      # - API_KEY=your-secret-key
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    profiles: ["gpu"]
```

Usage: `docker compose --profile cpu up` or `docker compose --profile gpu up`

### Configuration

All limits and model settings via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MAX_FILE_SIZE_MB` | 500 | Maximum upload size in MB |
| `MAX_DURATION_SECONDS` | 300 | Maximum video duration in seconds |
| `MAX_FRAMES` | 300 | Hard cap on frames extracted per request |
| `DEFAULT_FPS` | 1.0 | Default frame sampling rate |
| `BATCH_SIZE` | 32 | Frames per inference batch |
| `MAX_CONCURRENT_REQUESTS` | 2 (CPU) / 1 (GPU) | Max simultaneous inference requests. GPU default is 1 to avoid VRAM overcommit. |
| `MAX_UPLOAD_CONCURRENCY` | `MAX_CONCURRENT_REQUESTS + 2` | Max simultaneous uploads in the parse phase. Worst-case temp disk: `2 * MAX_UPLOAD_CONCURRENCY * MAX_FILE_SIZE_MB + frame_overhead`. |
| `API_KEY` | (unset) | If set, requires `X-API-Key` header |
| `ALLOW_UNAUTHENTICATED` | false | Must be `true` if `API_KEY` is unset; otherwise server refuses to start |
| `FFMPEG_TIMEOUT_SECONDS` | 120 | Subprocess timeout for ffmpeg/ffprobe |
| `REQUEST_TIMEOUT_SECONDS` | 300 | End-to-end inference timeout per request. Cooperative cancellation between batches. |
| `CLIP_CACHE_DIR` | /app/models | Explicit model cache directory. Must match between build and runtime. |

### Model Weights

- Baked into the Docker image at build time (no runtime download, no volume needed)
- Model name, pretrained tag, and cache path stored in `/app/.baked_model` — single source of truth at runtime (loaded as `ModelSpec` value object)
- `GET /ready` confirms model is loaded and reports details; `GET /live` is a minimal liveness check

### Trust Boundary

This is a compute-intensive endpoint. The v1 security posture:
- **Fail-closed auth:** Server refuses to start unless `API_KEY` is set or `ALLOW_UNAUTHENTICATED=true` is explicit
- ASGI middleware enforces auth + upload concurrency + body size before multipart parsing; `/live` exempt from all gates, `/ready` exempt from concurrency
- Upload concurrency limiter bounds aggregate temp disk: `2 * MAX_UPLOAD_CONCURRENCY * MAX_FILE_SIZE_MB + frame_overhead` (accounts for Starlette spool + controlled copy)
- Inference `CapacityLimiter` prevents GPU/CPU exhaustion (acquired post-parse, held until worker thread exits)
- Manual timer + cooperative cancellation for request timeout (limiter not released until thread exits, no abandoned threads)
- Frame cap + resolution cap + body-size cap + token truncation check bounds worst-case compute, memory, and semantic validity per request
- Model runs in inference mode with no gradient tracking; GPU uses FP16 autocast
- Single uvicorn worker prevents limiter bypass and model memory duplication
- For production: deploy behind a reverse proxy with rate limiting, request buffering, and network restrictions

### Image Size

- ~2GB CPU variant (slim torch + baked model weights)
- ~5GB CUDA variant (full torch + CUDA + baked model weights)

## Project Structure

```
clipCC/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── .env.example
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, startup, endpoint
│   ├── config.py             # Settings from env vars (pydantic BaseSettings)
│   ├── middleware.py          # ASGI request gate (delegates to ResourceGates)
│   ├── resource_gates.py      # ResourceGates: upload + inference CapacityLimiters with context managers
│   ├── temp_store.py          # TempStore: temp file lifecycle, aggregate tracking, cleanup
│   ├── inference_runner.py    # InferenceRunner: worker thread, cancel event, subprocess handle, timeout, limiter hold
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clip_model.py     # OpenCLIP loading and inference
│   │   └── model_spec.py     # ModelSpec: value object loaded from /app/.baked_model
│   ├── services/
│   │   ├── __init__.py
│   │   ├── video.py          # FrameExtractor: frame extraction via ffmpeg, returns list[FrameSample]
│   │   └── scoring.py        # Aggregation logic (mean/max, confidence scores)
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── response.py       # Pydantic response/error models
│   └── errors/
│       ├── __init__.py
│       └── handlers.py        # Custom exceptions + friendly error responses
└── tests/
    ├── __init__.py
    ├── test_api.py            # Endpoint integration tests
    ├── test_video.py          # Frame extraction tests
    └── test_scoring.py        # Scoring logic tests
```

### Module Boundaries

- **`resource_gates.py`** — `ResourceGates` service. Owns both `CapacityLimiter` instances. Exposes `upload_admission(scope)` (used by middleware) and `inference_admission()` (used by route). Middleware is sole owner of upload limiter lifecycle.
- **`temp_store.py`** — `TempStore` service. Owns all temp directories and files. Provides `save_upload(upload) -> StoredUpload`, manages frame extraction dirs, tracks aggregate disk usage, and `cleanup(request_id)` for guaranteed teardown. Startup janitor deletes stale files.
- **`inference_runner.py`** — `InferenceRunner` service. Encapsulates worker thread, `cancel_event`, subprocess handle, timeout timer, and inference limiter hold. Route calls `runner.run(request) -> Result`. Returns 504 only after worker exits and cleanup confirms. No limiter released until thread is done.
- **`model_spec.py`** — `ModelSpec` value object. Loaded once from `/app/.baked_model` at startup. Single source of truth for model name, pretrained tag, cache path. Used by config, health, and model loading.
- **`clip_model.py`** — Owns model loading and raw inference. Loads model using `ModelSpec` with explicit `cache_dir`. Model set to inference mode at load. Input: images + texts. Output: similarity matrix.
- **`video.py`** — `FrameExtractor`. Owns ffmpeg interaction. Input: file path, fps, max_frames, cancel_event. Output: `list[FrameSample]` (path, sample_index, approx_timestamp_seconds). Registers subprocess handle with `InferenceRunner` for timeout kill.
- **`scoring.py`** — Owns aggregation. Input: per-frame similarities + `FrameSample` metadata. Output: final confidence scores + best match (with `peak_frame_index` and `approx_timestamp_seconds` for max mode).
- **`main.py`** — Orchestrates the pipeline, handles HTTP layer only. Creates `InferenceRunner`, delegates all lifecycle management.

## Out of Scope for v1

- Async job queue / background processing
- WebSocket progress streaming
- Multiple model support at runtime
- Full authentication / OAuth / user management (optional API key is included)
- Rate limiting beyond concurrency semaphore (use reverse proxy for production)
- User-facing video pre-processing options (resize, crop — internal scale-down during extraction is included)
- Per-frame results (only aggregated scores returned)
