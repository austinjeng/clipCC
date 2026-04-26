# ClipCC

A cross-platform, Dockerized API that classifies video content against user-provided text labels using [OpenCLIP](https://github.com/mlfoundations/open_clip) (ViT-L-14). Upload a video file, provide a set of descriptive labels, and get back confidence scores for how well each label describes the video.

**Example use case:** Upload dashcam footage with labels like `"drunk driving"`, `"normal driving"`, `"distracted driving"` and get back which label best matches the video content.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [POST /api/v1/classify](#post-apiv1classify)
  - [GET /live](#get-live)
  - [GET /ready](#get-ready)
- [Configuration](#configuration)
- [Authentication](#authentication)
- [GPU Support](#gpu-support)
- [Building a Custom Model](#building-a-custom-model)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Q&A](#qa)

---

## Prerequisites

You only need **Docker** and **Docker Compose**. Everything else (Python, ffmpeg, PyTorch, the CLIP model) is bundled inside the Docker image.

| Requirement | Minimum Version | How to Check |
|---|---|---|
| Docker | 20.10+ | `docker --version` |
| Docker Compose | 2.0+ (V2) | `docker compose version` |

**Install Docker:**
- **macOS:** [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
- **Windows:** [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
- **Linux:** [Docker Engine](https://docs.docker.com/engine/install/) + [Docker Compose plugin](https://docs.docker.com/compose/install/linux/)

---

## Quick Start

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd clipCC
```

### 2. Build the Docker image (CPU)

```bash
docker compose --profile cpu build
```

This takes **5-15 minutes** on the first build. It downloads:
- Python 3.11 slim base image (~50 MB)
- ffmpeg and system libraries (~100 MB)
- PyTorch CPU (~200 MB)
- OpenCLIP and dependencies (~100 MB)
- ViT-L-14 model weights (~900 MB) — **baked into the image, no runtime download**

Subsequent builds with code-only changes are fast (cached layers).

### 3. Start the server

```bash
docker compose --profile cpu up
```

Wait for the log line:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

The model loads into memory on startup. This takes **10-30 seconds** on CPU.

### 4. Test it

Open a new terminal:

```bash
# Check the server is alive
curl http://localhost:8000/live
# {"status":"ok"}

# Check the model is loaded
curl http://localhost:8000/ready
# {"status":"ready","model":"ViT-L-14","pretrained":"laion2b_s32b_b82k","device":"cpu"}
```

### 5. Classify a video

Create a test video (or use your own .mp4):

```bash
# Generate a 5-second test video with ffmpeg
ffmpeg -y -f lavfi -i testsrc=duration=5:size=320x240:rate=10 \
  -c:v libx264 -pix_fmt yuv420p test_video.mp4
```

Send it to the API:

```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -F "video=@test_video.mp4" \
  -F 'labels=["test pattern","outdoor scene","person walking"]' \
  -F "fps=1.0" \
  -F "aggregation=mean"
```

Response:

```json
{
  "best_match": {
    "label": "test pattern",
    "confidence": 1.0
  },
  "scores": [
    {
      "label": "test pattern",
      "confidence": 1.0,
      "raw_similarity": 0.3496
    },
    {
      "label": "outdoor scene",
      "confidence": 0.0,
      "raw_similarity": 0.1736
    },
    {
      "label": "person walking",
      "confidence": 0.0,
      "raw_similarity": 0.1737
    }
  ],
  "metadata": {
    "frames_analyzed": 5,
    "video_duration_seconds": 5.0,
    "model": "ViT-L-14",
    "device": "cpu",
    "aggregation": "mean",
    "processing_time_seconds": 1.4,
    "disclaimer": "Scores are relative to the supplied labels, not calibrated probabilities. Not suitable for safety-critical decisions."
  }
}
```

### 6. Stop the server

```bash
docker compose --profile cpu down
```

---

## API Reference

### POST /api/v1/classify

Classify a video against a set of text labels.

**Content-Type:** `multipart/form-data`

#### Request Fields

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `video` | file | Yes | - | Video file. Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv` |
| `labels` | string (JSON) | Yes | - | JSON array of 3-10 text labels. Example: `'["driving","parking","crash"]'` |
| `prompt_template` | string | No | `"a video of {}"` | Template for CLIP text prompts. `{}` is replaced with each label. Max 500 characters. |
| `fps` | float | No | `1.0` | Frame sampling rate. Range: 0.1-5.0. Higher = more frames = slower but potentially more accurate. |
| `aggregation` | string | No | `"mean"` | Score aggregation method: `"mean"` or `"max"` |

#### Aggregation Methods

**`mean`** (default) — Averages confidence scores across all sampled frames. Best for answering "What is this video mostly showing?" Confidence scores sum to 1.0.

**`max`** — Returns the peak confidence score for each label across all frames independently. Best for answering "Did this happen at any point in the video?" Scores do NOT sum to 1.0. Each score includes `peak_frame_index` and `approx_timestamp_seconds` indicating which frame produced the peak.

#### Prompt Template

CLIP scores video frames against text prompts. By default, each label is wrapped as `"a video of {label}"`. You can customize this:

```bash
# Default: "a video of {label}"
-F 'labels=["driving","parking"]'

# Custom template for specific context:
-F 'labels=["driving","parking"]'
-F 'prompt_template=a dashcam video showing {}'
# Produces: "a dashcam video showing driving", "a dashcam video showing parking"
```

Better prompt templates can improve classification accuracy. The template must contain exactly one `{}` placeholder.

#### Response — Mean Mode

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
    },
    {
      "label": "distracted driving",
      "confidence": 0.40,
      "raw_similarity": 0.30
    }
  ],
  "metadata": {
    "frames_analyzed": 297,
    "video_duration_seconds": 297.4,
    "model": "ViT-L-14",
    "device": "cpu",
    "aggregation": "mean",
    "processing_time_seconds": 12.3,
    "disclaimer": "Scores are relative to the supplied labels, not calibrated probabilities. Not suitable for safety-critical decisions."
  }
}
```

#### Response — Max Mode

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
    "processing_time_seconds": 2.1,
    "disclaimer": "Scores are relative to the supplied labels, not calibrated probabilities. Max-mode scores are independent peaks per label and do not sum to 1. Not suitable for safety-critical decisions."
  }
}
```

#### Response Fields

| Field | Description |
|---|---|
| `best_match.label` | The label with the highest confidence score |
| `best_match.confidence` | The confidence value of the best match |
| `scores[].label` | The original label text |
| `scores[].confidence` | Relative confidence score (softmax over CLIP's logit-scaled cosine similarities) |
| `scores[].raw_similarity` | Unscaled cosine similarity between frame and text embeddings |
| `scores[].peak_frame_index` | (Max mode only) 0-based index of the frame that produced the peak score |
| `scores[].approx_timestamp_seconds` | (Max mode only) Approximate timestamp of the peak frame (`frame_index / fps`) |
| `metadata.frames_analyzed` | Total number of frames sampled from the video |
| `metadata.video_duration_seconds` | Duration of the input video |
| `metadata.model` | OpenCLIP model architecture used |
| `metadata.device` | Compute device (`cpu` or `cuda`) |
| `metadata.aggregation` | Aggregation method used |
| `metadata.processing_time_seconds` | Wall-clock time for the inference pipeline |
| `metadata.disclaimer` | Reminder that scores are relative, not absolute |

#### Error Responses

All errors return JSON with a human-friendly `detail` message.

| Status | Condition | Example |
|---|---|---|
| 401 | Missing or invalid API key | `"Invalid or missing API key..."` |
| 413 | File exceeds size limit | `"File size 620.0 MB exceeds the maximum allowed size of 500.0 MB."` |
| 415 | Unsupported video format | `"File format '.webm' is not supported."` |
| 422 | Video too long | `"Video duration 512.0s exceeds the maximum..."` |
| 422 | Too many frames | `"Extracting 1500 frames...exceeds the maximum of 300 frames."` |
| 422 | Resolution too high | `"Video resolution 7680x4320 exceeds the maximum supported resolution of 3840x2160."` |
| 422 | Label validation failure | `"Number of labels must be between 3 and 10 (inclusive)."` |
| 422 | Prompt too long for CLIP | `"Prompt '...' has 83 tokens and will be truncated..."` |
| 422 | Invalid fps | `"FPS value 10.0 is invalid. Must be between 0.1 and 5.0."` |
| 429 | Too many concurrent uploads | `"Too many uploads in progress. Please retry in a moment."` |
| 429 | Too many concurrent inferences | `"Server is processing the maximum number of videos..."` |
| 504 | Processing timeout | `"Inference timed out after 300.0s."` |

### GET /live

Liveness probe. Returns `200` if the process is running. No authentication required. Not affected by concurrency limits.

```json
{"status": "ok"}
```

### GET /ready

Readiness probe. Returns `200` with model details if the model is loaded and ready to serve. Returns `503` if the model is still loading. Respects authentication if configured.

```json
{
  "status": "ready",
  "model": "ViT-L-14",
  "pretrained": "laion2b_s32b_b82k",
  "device": "cpu"
}
```

---

## Configuration

All settings are controlled via environment variables. Set them in `docker-compose.yml`, a `.env` file, or pass them with `docker run -e`.

Copy the example:
```bash
cp .env.example .env
# Edit .env as needed
```

| Variable | Default | Description |
|---|---|---|
| `MAX_FILE_SIZE_MB` | `500` | Maximum upload file size in megabytes |
| `MAX_DURATION_SECONDS` | `300` | Maximum video duration (5 minutes) |
| `MAX_FRAMES` | `300` | Hard cap on total frames extracted per request |
| `DEFAULT_FPS` | `1.0` | Default frame sampling rate when not specified in the request |
| `BATCH_SIZE` | `32` | Number of frames processed per inference batch |
| `MAX_CONCURRENT_REQUESTS` | `2` (CPU) / `1` (GPU) | Max simultaneous inference requests. GPU defaults to 1 to avoid VRAM exhaustion. |
| `MAX_UPLOAD_CONCURRENCY` | `MAX_CONCURRENT_REQUESTS + 2` | Max simultaneous uploads being parsed. Bounds temp disk usage. |
| `API_KEY` | (unset) | If set, all requests to `/api/v1/classify` and `/ready` must include an `X-API-Key` header with this value |
| `ALLOW_UNAUTHENTICATED` | `false` | Must be `true` if `API_KEY` is not set. Server refuses to start without explicit auth configuration. |
| `FFMPEG_TIMEOUT_SECONDS` | `120` | Timeout for ffmpeg/ffprobe subprocess calls |
| `REQUEST_TIMEOUT_SECONDS` | `300` | End-to-end timeout for the entire inference pipeline per request |
| `CLIP_CACHE_DIR` | `/app/models` | Directory where model weights are stored. Must match the build-time cache. |

---

## Authentication

ClipCC uses **fail-closed** authentication. The server will not start unless you either:

1. Set an API key:
   ```yaml
   environment:
     - API_KEY=your-secret-key-here
   ```
   Then include the key in every request:
   ```bash
   curl -X POST http://localhost:8000/api/v1/classify \
     -H "X-API-Key: your-secret-key-here" \
     -F "video=@video.mp4" \
     -F 'labels=["a","b","c"]'
   ```

2. Explicitly opt out (for local development only):
   ```yaml
   environment:
     - ALLOW_UNAUTHENTICATED=true
   ```

The `/live` endpoint is always unauthenticated (for container health checks). The `/ready` endpoint respects authentication settings.

---

## GPU Support

GPU acceleration dramatically improves inference speed (10-30x faster than CPU).

### Requirements

- NVIDIA GPU with CUDA support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- Docker configured with the `nvidia` runtime

### Build and run with GPU

```bash
# Build with CUDA support (larger image, ~5 GB)
docker compose --profile gpu build

# Start with GPU
docker compose --profile gpu up
```

The GPU profile:
- Uses `cu121` (CUDA 12.1) PyTorch wheels
- Defaults to `MAX_CONCURRENT_REQUESTS=1` to avoid VRAM overcommit
- Automatically reserves all available GPUs

### Verify GPU is being used

```bash
curl http://localhost:8000/ready
# {"status":"ready","model":"ViT-L-14",...,"device":"cuda"}
```

If `device` says `"cuda"`, GPU acceleration is active.

---

## Building a Custom Model

The default image bakes in `ViT-L-14` with `laion2b_s32b_b82k` weights. To use a different OpenCLIP model, pass build args:

```bash
# Example: use smaller, faster ViT-B-32
docker build \
  --build-arg MODEL_NAME=ViT-B-32 \
  --build-arg PRETRAINED=laion2b_s34b_b79k \
  --build-arg TORCH_VARIANT=cpu \
  -t clipcc-custom .
```

Available models: see [OpenCLIP model list](https://github.com/mlfoundations/open_clip#pretrained-model-interface).

The model choice is baked into the image at build time and cannot be changed at runtime. This is by design: it prevents accidental runtime downloads and ensures the image is fully self-contained.

---

## Running Tests

### Without Docker (local development)

```bash
# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx anyio trio

# Run unit tests (no ffmpeg or model needed)
python -m pytest tests/test_config.py tests/test_temp_store.py tests/test_scoring.py tests/test_resource_gates.py -v

# Run all tests (requires ffmpeg in PATH and ~400 MB model download on first run)
python -m pytest tests/ -v
```

### Test summary

| Test File | Tests | What It Covers |
|---|---|---|
| `test_config.py` | 6 | Settings validation, auth config |
| `test_temp_store.py` | 5 | File upload, cleanup, janitor |
| `test_video.py` | 9 | ffprobe, validation, frame extraction |
| `test_clip_model.py` | 7 | Model loading, encoding, tokenization |
| `test_scoring.py` | 7 | Mean/max aggregation, logit scaling |
| `test_resource_gates.py` | 10 | Concurrency limiters |
| `test_middleware.py` | 12 | Auth, upload gates, body size |
| `test_inference_runner.py` | 3 | Timeout, cancellation |
| `test_api.py` | 8 | Full integration (health, validation, classify) |
| **Total** | **67** | |

---

## Project Structure

```
clipCC/
├── Dockerfile                  # Multi-stage build with baked model weights
├── docker-compose.yml          # CPU and GPU profiles
├── requirements.txt            # Python dependencies
├── .env.example                # Configuration reference
├── app/
│   ├── main.py                 # FastAPI app factory, routes, pipeline orchestration
│   ├── config.py               # Pydantic settings from environment variables
│   ├── middleware.py            # ASGI middleware: auth, upload concurrency, body size
│   ├── resource_gates.py       # anyio CapacityLimiters for upload and inference
│   ├── temp_store.py           # Temp file lifecycle and cleanup
│   ├── inference_runner.py     # Threaded pipeline runner with cooperative timeout
│   ├── models/
│   │   ├── clip_model.py       # OpenCLIP model loading and inference
│   │   └── model_spec.py       # Model metadata from baked .baked_model file
│   ├── services/
│   │   ├── video.py            # ffprobe validation and ffmpeg frame extraction
│   │   └── scoring.py          # Mean/max aggregation over frame scores
│   ├── schemas/
│   │   └── response.py         # Pydantic response models
│   └── errors/
│       └── handlers.py         # Custom HTTP exceptions with friendly messages
└── tests/                      # 67 tests across 9 test files
```

---

## How It Works

### Processing Pipeline

```
Client uploads video + labels
        |
        v
[ASGI Middleware] ── Auth check (X-API-Key header)
        |              Upload concurrency limit
        |              Body size limit (streaming)
        v
[Route Handler] ── Input validation (format, fps, labels, tokens)
        |
        v
[Inference Gate] ── Acquire inference slot (or 429)
        |
        v
[InferenceRunner thread]
        |
        ├── Save upload to temp file
        ├── ffprobe: validate duration, resolution, stream count
        ├── ffmpeg: extract frames at requested fps (scaled to 512px max)
        ├── For each batch of frames:
        │     ├── Load as PIL images
        │     ├── Encode images + text through OpenCLIP
        │     ├── Compute cosine similarities
        │     ├── Apply logit scale + softmax
        │     └── Delete consumed frame files
        ├── Aggregate scores (mean or max)
        └── Return response
        |
        v
[Cleanup] ── Delete all temp files (guaranteed via finally)
```

### Key Design Decisions

- **Synchronous pipeline in a thread:** The inference pipeline (ffmpeg + PyTorch) is blocking. It runs in a worker thread via `anyio.to_thread.run_sync` so the async event loop stays responsive for health checks and concurrent gate decisions.

- **Two-level concurrency control:** Upload concurrency and inference concurrency are controlled by separate `anyio.CapacityLimiter` instances. Uploads are bounded but higher-limit; inference slots are scarce (especially on GPU).

- **Cooperative timeout:** The `InferenceRunner` uses a `threading.Event` flag checked between batches. On timeout, it sets the flag, kills any active ffmpeg subprocess, and waits for the worker thread to exit before releasing the inference slot. No abandoned threads.

- **Model baked into Docker image:** Model weights are downloaded during `docker build` and stored in the image. No runtime downloads, no volume mounts needed. The model config is a single source of truth in `/app/.baked_model`.

- **Logit scaling:** CLIP's learned `logit_scale` parameter (~100) is applied before softmax. Without it, softmax over raw cosine similarities (0.2-0.35 range) produces near-uniform, meaningless distributions.

---

## Q&A

### General

**Q: What video formats are supported?**
A: `.mp4`, `.avi`, `.mov`, and `.mkv`. The video must have a single video stream, resolution at most 3840x2160, and duration at most 5 minutes (configurable).

**Q: How long does classification take?**
A: Depends on video length, fps, and hardware. A 5-minute video at 1fps = 300 frames. On CPU, expect 1-5 minutes. On GPU, expect 10-30 seconds. The `processing_time_seconds` field in the response tells you the exact time.

**Q: What does "confidence" mean?**
A: Confidence scores are **relative**, not absolute. They indicate how well each label matches compared to the other labels you provided. The same video scored against different label sets will produce different confidence values. They are NOT calibrated probabilities and should NOT be used for safety-critical decisions.

**Q: What does `raw_similarity` mean?**
A: It's the unscaled cosine similarity between the video frame embedding and the text label embedding, before the model's learned temperature scaling and softmax. Useful for advanced users who want to do their own scoring.

**Q: Can I use this for real-time video?**
A: No. ClipCC is designed for offline video classification. It processes uploaded video files, not live streams.

### Setup and Docker

**Q: The build is taking a very long time. Is that normal?**
A: Yes, the first build downloads ~1.3 GB of dependencies including the ViT-L-14 model weights. Subsequent builds are much faster because Docker caches the layers. Only code changes (in `app/`) trigger a rebuild of the final layers.

**Q: Can I run this without Docker?**
A: Yes, but Docker is strongly recommended for reproducibility. For local development:
```bash
pip install -r requirements.txt
ALLOW_UNAUTHENTICATED=true uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```
You'll need Python 3.11+, ffmpeg in your PATH, and the model will be downloaded on first startup.

**Q: How big is the Docker image?**
A: ~2 GB for the CPU variant, ~5 GB for the GPU (CUDA) variant. Most of the size is PyTorch and the model weights.

**Q: I get "Cannot connect to the Docker daemon" — what do I do?**
A: Start Docker Desktop (macOS/Windows) or the Docker service (Linux: `sudo systemctl start docker`). Wait a few seconds for it to initialize.

**Q: I get a `torchvision` error during build. What happened?**
A: Make sure the Dockerfile installs `torch torchvision` from the same PyTorch wheel index. The included Dockerfile already handles this correctly.

### GPU

**Q: How do I know if the GPU is being used?**
A: Check the `/ready` endpoint. If `"device": "cuda"`, GPU is active. If `"device": "cpu"`, it's using CPU even though you built with the GPU profile — check that the NVIDIA Container Toolkit is installed and Docker can see your GPU (`docker run --gpus all nvidia-smi`).

**Q: Can I use an AMD GPU?**
A: Not directly. The CUDA build profile targets NVIDIA GPUs. AMD GPU support via ROCm would require a different PyTorch build and Dockerfile modifications.

**Q: Why is `MAX_CONCURRENT_REQUESTS` set to 1 for GPU?**
A: Two concurrent ViT-L-14 inference calls with 32-frame batches can exhaust VRAM on GPUs with less than 16 GB. If you have a high-VRAM GPU (24+ GB), you can safely increase this to 2.

### API Usage

**Q: How do I choose good labels?**
A: Labels should be distinct, descriptive, and cover the range of content you expect. More specific labels work better than vague ones. For example, `"person running on a track"` is better than `"movement"`.

**Q: What's the best `fps` value?**
A: `1.0` (default) works well for most use cases. Increase to 2-5 for short videos where you need finer temporal resolution. Lower to 0.5 for long videos where the content is mostly static. Higher fps = more frames = slower processing but potentially more accurate.

**Q: When should I use `max` instead of `mean` aggregation?**
A: Use `mean` when you want to know the dominant content of the video ("What is this video mostly about?"). Use `max` when you want to detect if something appeared at any point ("Did this happen at all in the video?"). For example, detecting a brief traffic violation in a long driving video benefits from `max`.

**Q: Can I send more than 10 labels?**
A: No. The limit is 3-10 labels per request. If you need more categories, make multiple requests with different label sets.

**Q: My labels produce "identical token sequences" error. Why?**
A: CLIP's tokenizer may produce the same token sequence for labels that look different as text (e.g., extra spaces, Unicode variants). The API rejects these because the model literally cannot distinguish between them. Use more distinct wording.

**Q: What is the prompt template for?**
A: CLIP was trained on image-text pairs. Wrapping labels in a natural sentence (like "a video of {label}") often improves accuracy over bare labels. The default `"a video of {}"` works well for general use. For domain-specific content, try templates like `"a surveillance camera recording of {}"` or `"a dashcam video showing {}"`.

### Performance and Limits

**Q: I'm getting 429 "Too many uploads" — what do I do?**
A: The server limits concurrent uploads to prevent disk exhaustion. Wait a moment and retry. If this happens frequently, increase `MAX_UPLOAD_CONCURRENCY`.

**Q: I'm getting 504 timeout — what do I do?**
A: The inference pipeline exceeded `REQUEST_TIMEOUT_SECONDS` (default 300s). Options: use a shorter video, lower the `fps`, or increase the timeout. On CPU, 300 frames can take several minutes.

**Q: What's the maximum video file size?**
A: 500 MB by default (configurable via `MAX_FILE_SIZE_MB`). The limit is enforced at the ASGI middleware level before the file is fully parsed, so oversized uploads are rejected quickly.

**Q: Can multiple users call the API at the same time?**
A: Yes, with limits. Upload concurrency defaults to `MAX_CONCURRENT_REQUESTS + 2` (allowing a small queue). Inference concurrency defaults to 2 on CPU, 1 on GPU. Requests beyond the limit receive a `429` response. For higher throughput, deploy behind a load balancer with multiple instances.

### Security

**Q: Is this safe to deploy on the public internet?**
A: For production deployment, you should: (1) set `API_KEY` instead of `ALLOW_UNAUTHENTICATED`, (2) deploy behind a reverse proxy (nginx, Caddy) with rate limiting and TLS, (3) restrict network access. The built-in concurrency limits provide basic DoS protection but are not a substitute for proper infrastructure.

**Q: Does the API store my videos?**
A: No. Uploaded videos are saved to a temporary directory, processed, and immediately deleted. A startup janitor also cleans up any stale files from previous crashes. No video data persists after the request completes.
