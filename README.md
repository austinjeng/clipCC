# ClipCC

English | [繁體中文](README.zh-TW.md)


A cross-platform, Dockerized API that classifies video content against user-provided text labels. Upload a video file, provide a set of descriptive labels, and get back confidence scores for how well each label describes the video.

**Example use case:** Upload dashcam footage with labels like `"drunk driving"`, `"normal driving"`, `"distracted driving"` and get back which label best matches the video content.

---

## Two Versions Available

This repository has two branches, each offering a different model backend:

| | `master` (CLIP) | `SigLip2` (SigLIP2) |
|---|---|---|
| **Model** | [OpenCLIP](https://github.com/mlfoundations/open_clip) ViT-L-14 | [SigLIP2](https://huggingface.co/docs/transformers/model_doc/siglip2) (multiple sizes) |
| **Model switching** | Fixed at build time | Hot-swap from web UI or API — no restart |
| **Web UI** | None (API only) | Built-in UI at `http://localhost:8000/` |
| **Model storage** | Baked into Docker image | Downloaded on demand, cached in Docker volume |
| **Scoring** | Softmax (scores sum to 1) | Sigmoid (independent per-label scores) |
| **Docker image size** | ~2 GB (weights included) | ~1 GB (weights downloaded separately) |

**Which should I use?**
- Use **`master`** if you want a simple, self-contained setup with no runtime downloads.
- Use **`SigLip2`** if you want a web UI, the ability to switch between models without restarting, or access to newer/larger SigLIP2 models.

### Getting the source code

```bash
# Clone the repository (lands on master by default)
git clone https://github.com/austinjeng/clipCC.git
cd clipCC

# To use the SigLip2 version instead:
git checkout SigLip2
```

Or clone the SigLip2 branch directly:

```bash
git clone -b SigLip2 https://github.com/austinjeng/clipCC.git
cd clipCC
```

> The setup instructions below cover **both versions**. The `master` branch guide is first, followed by the [SigLip2 Branch Guide](#siglip2-branch-guide).

---

## Table of Contents

- [Two Versions Available](#two-versions-available)
- [SigLip2 Branch Guide](#siglip2-branch-guide) (web UI + hot-swap models)
- [Setup by Platform](#setup-by-platform) (master branch)
  - [Windows](#windows)
  - [macOS](#macos)
  - [Linux](#linux)
- [Your First Classification](#your-first-classification)
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

## Setup by Platform

Pick your operating system and follow the steps. All three paths end at the same place: a running ClipCC server on `http://localhost:8000`.

### Getting the source code (master)

See [Getting the source code](#getting-the-source-code) above. Make sure you are on the `master` branch:

```bash
git checkout master
```

---

### Windows

You have two options: **Docker** (recommended) or **native** (no Docker, no WSL2).

#### Option A: Docker (Recommended)

**Prerequisites:**
- Windows 10/11 (Home, Pro, or Enterprise)
- [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/) (4.0+)

Docker Desktop will enable WSL2 automatically during installation. If you're on Pro/Enterprise and prefer not to use WSL2, you can switch to the Hyper-V backend in Docker Desktop Settings > General > uncheck "Use the WSL 2 based engine."

**Step 1: Install Docker Desktop**

Download and run the installer from [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/). Restart your computer when prompted. After restart, open Docker Desktop and wait for the engine to start (the whale icon in the system tray should stop animating).

Verify installation in PowerShell:
```powershell
docker --version
# Docker version 28.x.x
docker compose version
# Docker Compose version v2.x.x
```

**Step 2: Build the image**

```powershell
docker compose --profile cpu build
```

The first build takes **5-15 minutes**. It downloads Python, ffmpeg, PyTorch, OpenCLIP, and the ViT-L-14 model weights (~900 MB) — all baked into the image so there are no runtime downloads.

**Step 3: Start the server**

```powershell
docker compose --profile cpu up
```

Wait for the log line:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Model loading takes **10-30 seconds** on CPU. Once you see the Uvicorn line, the server is ready.

**Step 4: Verify**

Open a new PowerShell window:
```powershell
curl.exe http://localhost:8000/live
# {"status":"ok"}

curl.exe http://localhost:8000/ready
# {"status":"ready","model":"ViT-L-14","pretrained":"laion2b_s32b_b82k","device":"cpu"}
```

**Step 5: Stop the server**

```powershell
docker compose --profile cpu down
```

You're set. Jump to [Your First Classification](#your-first-classification).

---

#### Option B: Native Windows (No Docker, No WSL2)

Run ClipCC directly on Windows without Docker or WSL2. Suitable for development or if Docker isn't an option.

**Prerequisites:**
- Windows 10/11
- Python 3.11+ ([python.org installer](https://www.python.org/downloads/) — check "Add Python to PATH" during install)
- ffmpeg in your PATH

**Step 1: Install ffmpeg**

Via winget (built into Windows 10 1709+ and Windows 11):
```powershell
winget install ffmpeg
```

Or download manually from [ffmpeg.org/download](https://ffmpeg.org/download.html#build-windows), extract, and add the `bin` folder to your system PATH.

Verify:
```powershell
ffmpeg -version
ffprobe -version
```

**Step 2: Install dependencies**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **Note:** The virtual environment (`.venv`) keeps ClipCC's dependencies isolated from your system Python. You must activate it each time you open a new terminal before running the server.

**Step 3: Start the server**

Set the required environment variables and launch:
```powershell
$env:ALLOW_UNAUTHENTICATED = "true"
$env:TEMP_DIR = "C:\temp\clipcc"
$env:CLIP_CACHE_DIR = "C:\temp\clipcc_models"

uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```

On first startup, the ViT-B-32 model weights (~400 MB) will be downloaded automatically. This is the development default when running outside Docker. Subsequent starts use the cached model.

> **Docker vs. native model:** The Docker image bakes in the larger ViT-L-14 model (~900 MB, higher accuracy). Native setup falls back to ViT-B-32 (~400 MB, faster download, slightly lower accuracy). Both work for development and testing.

**Step 4: Verify**

Open a new PowerShell window (activate the venv first: `.\.venv\Scripts\Activate.ps1`):
```powershell
curl.exe http://localhost:8000/live
curl.exe http://localhost:8000/ready
```

> **Important:** Always use `curl.exe` (not `curl`) in PowerShell. The bare `curl` command in PowerShell is an alias for `Invoke-WebRequest`, which has different syntax and will not work with the examples in this guide.

**Step 5: Stop the server**

Press `Ctrl+C` in the terminal running uvicorn.

You're set. Jump to [Your First Classification](#your-first-classification).

> **Note:** The native setup does not create a `.baked_model` metadata file (Docker creates this during build). The app falls back to ViT-B-32, a smaller model that downloads faster (~400 MB vs. ~900 MB for ViT-L-14). The default temp and cache paths (`/tmp/clipcc`, `/app/models`) are Linux paths — you **must** override them with Windows paths as shown above.

---

### macOS

You have two options: **Docker** (recommended) or **native**.

#### Option A: Docker (Recommended)

**Prerequisites:**
- macOS 12 (Monterey) or later
- [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)

**Step 1: Install Docker Desktop**

Download and install from [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/). Choose the correct chip variant:
- **Apple Silicon (M1/M2/M3/M4):** "Mac with Apple chip"
- **Intel:** "Mac with Intel chip"

After installation, open Docker Desktop and wait for the engine to start (the whale icon in the menu bar should stop animating).

Verify in Terminal:
```bash
docker --version
# Docker version 28.x.x
docker compose version
# Docker Compose version v2.x.x
```

**Step 2: Build the image**

```bash
docker compose --profile cpu build
```

The first build takes **5-15 minutes**. It downloads Python, ffmpeg, PyTorch, OpenCLIP, and the ViT-L-14 model weights (~900 MB) — all baked into the image so there are no runtime downloads.

> **Apple Silicon note:** The image builds for `linux/arm64` and runs via Docker's Linux VM. PyTorch CPU inference works well on Apple Silicon through this layer. Native Metal/MPS acceleration is not available inside Docker — use the native option below if you want MPS.

**Step 3: Start the server**

```bash
docker compose --profile cpu up
```

Wait for the log line:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Model loading takes **10-30 seconds**. Once you see the Uvicorn line, the server is ready.

**Step 4: Verify**

Open a new Terminal tab:
```bash
curl http://localhost:8000/live
# {"status":"ok"}

curl http://localhost:8000/ready
# {"status":"ready","model":"ViT-L-14","pretrained":"laion2b_s32b_b82k","device":"cpu"}
```

**Step 5: Stop the server**

```bash
docker compose --profile cpu down
```

You're set. Jump to [Your First Classification](#your-first-classification).

---

#### Option B: Native macOS (No Docker)

Run ClipCC directly. Useful for development or to use Apple Silicon MPS acceleration.

**Prerequisites:**
- macOS 12+
- Python 3.11+ (via Homebrew or [python.org](https://www.python.org/downloads/))
- ffmpeg

**Step 1: Install dependencies**

Using [Homebrew](https://brew.sh/) (recommended):
```bash
brew install python@3.11 ffmpeg
```

Or if you already have Python 3.11+ installed:
```bash
brew install ffmpeg
```

Verify:
```bash
python3 --version
# Python 3.11.x or higher
ffmpeg -version
ffprobe -version
```

**Step 2: Install Python packages**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** The virtual environment (`.venv`) keeps ClipCC's dependencies isolated from your system Python. Modern macOS and Homebrew Python reject global `pip install` with an "externally managed environment" error — using a venv avoids this entirely. Activate it each time you open a new terminal: `source .venv/bin/activate`

**Step 3: Start the server**

```bash
ALLOW_UNAUTHENTICATED=true \
TEMP_DIR=/tmp/clipcc \
CLIP_CACHE_DIR=$HOME/.cache/clipcc_models \
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```

On first startup, the ViT-B-32 model weights (~400 MB) will be downloaded automatically. This is the development default when running outside Docker. Subsequent starts use the cached model.

> **Docker vs. native model:** The Docker image bakes in the larger ViT-L-14 model (~900 MB, higher accuracy). Native setup falls back to ViT-B-32 (~400 MB, faster download, slightly lower accuracy). Both work for development and testing.

**Step 4: Verify**

Open a new Terminal tab (activate the venv first: `source .venv/bin/activate`):
```bash
curl http://localhost:8000/live
curl http://localhost:8000/ready
```

**Step 5: Stop the server**

Press `Ctrl+C` in the terminal running uvicorn.

You're set. Jump to [Your First Classification](#your-first-classification).

---

### Linux

You have two options: **Docker** (recommended) or **native**.

#### Option A: Docker (Recommended)

**Prerequisites:**
- A 64-bit Linux distribution (Ubuntu 20.04+, Debian 11+, Fedora 36+, etc.)
- [Docker Engine](https://docs.docker.com/engine/install/) (20.10+)
- [Docker Compose plugin](https://docs.docker.com/compose/install/linux/) (V2)

**Step 1: Install Docker Engine**

Follow the official guide for your distribution: [Install Docker Engine](https://docs.docker.com/engine/install/).

For Ubuntu/Debian:
```bash
# Add Docker's official GPG key and repository (see Docker docs for latest commands)
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Allow running Docker without sudo (requires re-login)
sudo usermod -aG docker $USER
```

Verify (after re-login):
```bash
docker --version
# Docker version 28.x.x
docker compose version
# Docker Compose version v2.x.x
```

**Step 2: Build the image**

```bash
docker compose --profile cpu build
```

The first build takes **5-15 minutes**. It downloads Python, ffmpeg, PyTorch, OpenCLIP, and the ViT-L-14 model weights (~900 MB) — all baked into the image so there are no runtime downloads.

**Step 3: Start the server**

```bash
docker compose --profile cpu up
```

Wait for the log line:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Model loading takes **10-30 seconds**. Once you see the Uvicorn line, the server is ready.

**Step 4: Verify**

Open a new terminal:
```bash
curl http://localhost:8000/live
# {"status":"ok"}

curl http://localhost:8000/ready
# {"status":"ready","model":"ViT-L-14","pretrained":"laion2b_s32b_b82k","device":"cpu"}
```

**Step 5: Stop the server**

```bash
docker compose --profile cpu down
```

You're set. Jump to [Your First Classification](#your-first-classification).

---

#### Option B: Native Linux (No Docker)

Run ClipCC directly. Useful for development or when Docker isn't available.

**Prerequisites:**
- Python 3.11+
- ffmpeg and ffprobe

**Step 1: Install system dependencies**

Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv python3-pip ffmpeg
```

Fedora:
```bash
sudo dnf install python3.11 ffmpeg
```

Arch Linux:
```bash
sudo pacman -S python ffmpeg
```

Verify:
```bash
python3 --version
# Python 3.11.x or higher
ffmpeg -version
ffprobe -version
```

**Step 2: Install Python packages**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** The virtual environment (`.venv`) keeps ClipCC's dependencies isolated from your system Python. Debian 12+, Ubuntu 23.04+, and Fedora 38+ reject global `pip install` with an "externally managed environment" error — using a venv avoids this entirely. Activate it each time you open a new terminal: `source .venv/bin/activate`

**Step 3: Start the server**

```bash
ALLOW_UNAUTHENTICATED=true \
TEMP_DIR=/tmp/clipcc \
CLIP_CACHE_DIR=$HOME/.cache/clipcc_models \
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```

On first startup, the ViT-B-32 model weights (~400 MB) will be downloaded automatically. This is the development default when running outside Docker. Subsequent starts use the cached model.

> **Docker vs. native model:** The Docker image bakes in the larger ViT-L-14 model (~900 MB, higher accuracy). Native setup falls back to ViT-B-32 (~400 MB, faster download, slightly lower accuracy). Both work for development and testing.

**Step 4: Verify**

Open a new terminal (activate the venv first: `source .venv/bin/activate`):
```bash
curl http://localhost:8000/live
curl http://localhost:8000/ready
```

**Step 5: Stop the server**

Press `Ctrl+C` in the terminal running uvicorn.

You're set. Continue to the next section.

---

## Your First Classification

These commands work the same on all platforms once the server is running.

### Create a test video

If you have ffmpeg installed locally (native setup), or use any `.mp4` you already have:

```bash
ffmpeg -y -f lavfi -i testsrc=duration=5:size=320x240:rate=10 \
  -c:v libx264 -pix_fmt yuv420p test_video.mp4
```

On Windows PowerShell, use backticks for line continuation:
```powershell
ffmpeg -y -f lavfi -i testsrc=duration=5:size=320x240:rate=10 `
  -c:v libx264 -pix_fmt yuv420p test_video.mp4
```

If you don't have ffmpeg locally (Docker-only setup), use any short `.mp4` file you have on your machine.

### Classify with mean aggregation

```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -F "video=@test_video.mp4" \
  -F 'labels=["test pattern","outdoor scene","person walking"]' \
  -F "fps=1.0" \
  -F "aggregation=mean"
```

Windows PowerShell (note: must use `curl.exe`, not `curl`):
```powershell
curl.exe -X POST http://localhost:8000/api/v1/classify `
  -F "video=@test_video.mp4" `
  -F "labels=[""test pattern"",""outdoor scene"",""person walking""]" `
  -F "fps=1.0" `
  -F "aggregation=mean"
```

> **PowerShell quoting:** PowerShell uses `""` to escape double quotes inside double-quoted strings. Single quotes do not support variable interpolation in PowerShell, but they also don't escape inner quotes the way bash does. The safest approach is double-quoted strings with `""` for inner quotes as shown above.

Example response (illustrative — exact scores vary by model, platform, and library version):

```json
{
  "best_match": {
    "label": "test pattern",
    "confidence": 0.98
  },
  "scores": [
    {
      "label": "test pattern",
      "confidence": 0.98,
      "raw_similarity": 0.35
    },
    {
      "label": "outdoor scene",
      "confidence": 0.01,
      "raw_similarity": 0.17
    },
    {
      "label": "person walking",
      "confidence": 0.01,
      "raw_similarity": 0.17
    }
  ],
  "metadata": {
    "frames_analyzed": 5,
    "video_duration_seconds": 5.0,
    "model": "ViT-L-14",
    "device": "cpu",
    "aggregation": "mean",
    "processing_time_seconds": 1.4,
    "disclaimer": "Scores are relative to the supplied labels, ..."
  }
}
```

> **Note:** Docker users will see `"model": "ViT-L-14"`. Native users will see `"model": "ViT-B-32"` (the development fallback). The `"test pattern"` label should score highest in both cases since the test video is literally an ffmpeg test pattern.

### Classify with max aggregation

```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -F "video=@test_video.mp4" \
  -F 'labels=["test pattern","outdoor scene","person walking"]' \
  -F "aggregation=max"
```

Max mode returns `peak_frame_index` and `approx_timestamp_seconds` for each label, indicating which frame produced the peak confidence.

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

All settings are controlled via environment variables. There are three ways to set them depending on your setup:

**Docker users:** Set variables in the `environment:` section of `docker-compose.yml` (already pre-configured with sensible defaults).

**Native users (`.env` file):** Copy the example and edit. The app automatically reads `.env` from the working directory on startup:
```bash
cp .env.example .env
# Edit .env as needed
```

> **Important for native users:** The `.env.example` file ships with Docker-friendly defaults (e.g., `CLIP_CACHE_DIR=/app/models`). If you are running natively, you **must** change the path variables to local paths. See the comments inside `.env.example` for platform-specific examples.

**Native users (inline):** Pass variables directly when launching uvicorn (as shown in the platform setup sections).

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
| `TEMP_DIR` | `/tmp/clipcc` | Directory for temporary upload and frame files. **Windows native users:** override to a Windows path like `C:\temp\clipcc`. |

---

## Authentication

ClipCC uses **fail-closed** authentication. The server will not start unless you either:

1. Set an API key:

   Docker (`docker-compose.yml`):
   ```yaml
   environment:
     - API_KEY=your-secret-key-here
   ```

   Native (bash):
   ```bash
   API_KEY=your-secret-key-here uvicorn app.main:create_app --factory ...
   ```

   Native (PowerShell):
   ```powershell
   $env:API_KEY = "your-secret-key-here"
   uvicorn app.main:create_app --factory ...
   ```

   Then include the key in every request:
   ```bash
   curl -X POST http://localhost:8000/api/v1/classify \
     -H "X-API-Key: your-secret-key-here" \
     -F "video=@video.mp4" \
     -F 'labels=["a","b","c"]'
   ```

2. Explicitly opt out (for local development only):

   Docker: `ALLOW_UNAUTHENTICATED=true` in environment
   Native bash: `ALLOW_UNAUTHENTICATED=true uvicorn ...`
   Native PowerShell: `$env:ALLOW_UNAUTHENTICATED = "true"`

The `/live` endpoint is always unauthenticated (for container health checks). The `/ready` endpoint respects authentication settings.

---

## GPU Support

GPU acceleration dramatically improves inference speed (10-30x faster than CPU).

### Requirements

- **NVIDIA GPU** with CUDA support (RTX 20xx series or newer recommended)
- **NVIDIA drivers** installed on the host
- **NVIDIA Container Toolkit** — required for Docker GPU passthrough

| Platform | GPU Support |
|---|---|
| **Linux (Docker)** | Full support. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). |
| **Windows (Docker)** | Supported via WSL2 backend. Docker Desktop automatically passes through the GPU. |
| **Windows (Native)** | Works if CUDA-compatible PyTorch is installed: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| **macOS (Docker)** | Not supported. Docker on macOS runs a Linux VM without GPU passthrough. |
| **macOS (Native)** | PyTorch MPS (Apple Silicon) may work but is untested with OpenCLIP. CPU is the safe default. |

### Build and run with GPU (Docker)

```bash
# Build with CUDA support (larger image, ~5 GB)
docker compose --profile gpu build

# Start with GPU
docker compose --profile gpu up
```

### Verify GPU is being used

```bash
curl http://localhost:8000/ready
# {"status":"ready","model":"ViT-L-14",...,"device":"cuda"}
```

If `device` says `"cuda"`, GPU acceleration is active. If it says `"cpu"`, check that NVIDIA Container Toolkit is installed and Docker can see your GPU: `docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`.

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

### With Docker setup

There is no in-container test runner. Tests are designed to run on your local machine.

### Local test execution

Install test dependencies (inside your virtual environment):
```bash
source .venv/bin/activate  # Linux/macOS
# or: .\.venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -r requirements.txt
pip install pytest pytest-asyncio httpx anyio trio
```

Run unit tests (no ffmpeg or model download needed):
```bash
python -m pytest tests/test_config.py tests/test_temp_store.py tests/test_scoring.py tests/test_resource_gates.py -v
```

Run the full suite (requires ffmpeg in PATH; first run downloads ViT-B-32 ~400 MB for test model):
```bash
python -m pytest tests/ -v
```

Windows PowerShell:
```powershell
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

## SigLip2 Branch Guide

This section covers the `SigLip2` branch, which uses Google's [SigLIP2](https://huggingface.co/docs/transformers/model_doc/siglip2) models via HuggingFace transformers. It includes a built-in web UI, hot-swappable models, and on-demand model downloads.

### Switch to the SigLip2 branch

```bash
git checkout SigLip2
```

### Available SigLIP2 models

| Model | Parameters | Resolution | Best for |
|---|---|---|---|
| `siglip2-base-patch16-256` | 0.4B | 256px | Fast inference, low memory (default for CPU) |
| `siglip2-base-patch16-384` | 0.4B | 384px | Better accuracy, still lightweight |
| `siglip2-large-patch16-256` | 0.9B | 256px | Higher quality, moderate speed |
| `siglip2-large-patch16-384` | 0.9B | 384px | High quality, moderate memory |
| `siglip2-so400m-patch14-384` | 1B | 384px | Best quality (default for GPU) |
| `siglip2-so400m-patch16-512` | 1B | 512px | Highest resolution, most memory |

### Setup — Docker (Recommended)

Prerequisites are the same as the master branch (Docker Desktop or Docker Engine).

**Step 1: Build the image**

```bash
docker compose --profile cpu build
```

The build is faster than master (~1 GB, no model weights baked in).

**Step 2: Start the server**

```bash
docker compose --profile cpu up
```

On first startup, the default model (`siglip2-base-patch16-256`, ~800 MB) downloads automatically. This happens once — the Docker volume caches it for future runs.

Wait for:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Auto-loaded model: siglip2-base-patch16-256
```

**Step 3: Open the web UI**

Open your browser to **http://localhost:8000/**

You'll see the clipCC web interface with:
- A model dropdown (6 SigLIP2 models to choose from)
- Status indicator (green = model loaded and ready)
- Video upload, labels input, and classify button

**Step 4: Classify a video from the web UI**

1. The default model auto-loads at startup. Wait for the green status dot.
2. Upload a `.mp4`, `.avi`, `.mov`, or `.mkv` video file
3. Enter labels separated by commas (3-10 labels), e.g.: `driving, parking, reversing`
4. Click **Classify**
5. Results appear as horizontal confidence bars

**Step 5: Switch models (optional)**

1. Select a different model from the dropdown
2. Click **Load Model**
3. Wait for the spinner to finish (downloads the model if not cached, then loads it)
4. The status dot turns green when ready
5. Classify again — the new model is now active

**Step 6: Stop the server**

```bash
docker compose --profile cpu down
```

Model weights persist in the Docker volume — they won't need to re-download next time.

### Setup — Native (No Docker)

**Prerequisites:** Same as master branch (Python 3.11+, ffmpeg). Plus install the additional dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -r requirements.txt
```

**Start the server:**

```bash
ALLOW_UNAUTHENTICATED=true \
CLIP_CACHE_DIR=$HOME/.cache/clipcc_models \
TEMP_DIR=/tmp/clipcc \
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```

Windows PowerShell:
```powershell
$env:ALLOW_UNAUTHENTICATED = "true"
$env:CLIP_CACHE_DIR = "C:\temp\clipcc_models"
$env:TEMP_DIR = "C:\temp\clipcc"
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```

On first startup, the default model (~800 MB) downloads automatically. Open **http://localhost:8000/** in your browser.

### SigLip2 API — Classify via command line

The web UI is optional. You can also use the API directly:

```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -F "video=@test_video.mp4" \
  -F 'labels=["driving","parking","reversing"]' \
  -F "prompt_template=This is a photo of {}." \
  -F "fps=1.0" \
  -F "aggregation=mean"
```

> **Note:** The default prompt template on the SigLip2 branch is `"This is a photo of {}."` (matching SigLIP2's training data), not `"a video of {}"`.

### SigLip2 API — Model management endpoints

```bash
# List available models
curl http://localhost:8000/api/v1/models

# Load a specific model
curl -X POST http://localhost:8000/api/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "siglip2-large-patch16-384"}'

# Check which model is active
curl http://localhost:8000/api/v1/models/active
```

### SigLip2 — Understanding scores

SigLIP2 uses **sigmoid** scoring instead of softmax. This means:

- Each label gets an **independent** confidence score between 0 and 1
- Scores do **NOT** sum to 1 (unlike the master branch)
- A score of 0.8 means the model is 80% confident that label applies, regardless of other labels
- Multiple labels can score high simultaneously if the video matches several descriptions

The response includes `score_semantics: "siglip2_pairwise_sigmoid"` so you know which scoring method was used.

### SigLip2 — Configuration

All settings from the master branch still apply, plus one new variable:

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_MODEL_ID` | `siglip2-base-patch16-256` | Model to auto-load at startup. Set to a larger model if you have GPU/more memory. |

### SigLip2 — GPU support

```bash
# Build and run with GPU
docker compose --profile gpu build
docker compose --profile gpu up
```

The GPU profile defaults to `siglip2-so400m-patch14-384` (1B parameters) — the highest quality model that fits comfortably in most GPUs. You can switch to other models from the web UI at any time.

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

### Windows-Specific

**Q: Do I need WSL2?**
A: No. Docker Desktop can use the Hyper-V backend on Windows Pro/Enterprise (uncheck "Use the WSL 2 based engine" in Settings). On Windows Home, WSL2 is required for Docker but is installed automatically. You can also run ClipCC natively without Docker at all — see [Windows Option B](#option-b-native-windows-no-docker-no-wsl2).

**Q: `curl` doesn't work in PowerShell. What do I do?**
A: PowerShell aliases `curl` to `Invoke-WebRequest`, which has completely different syntax. Always use `curl.exe` (the real curl, included in Windows 10+) in all commands from this guide:
```powershell
curl.exe http://localhost:8000/live
```
If you see errors about parameters or missing flags, check that you typed `curl.exe`, not `curl`.

**Q: What Windows paths should I use for native setup?**
A: Override these environment variables in PowerShell:
```powershell
$env:TEMP_DIR = "C:\temp\clipcc"
$env:CLIP_CACHE_DIR = "C:\temp\clipcc_models"
```
The defaults (`/tmp/clipcc`, `/app/models`) are Linux paths and won't work on Windows.

### macOS-Specific

**Q: Does this work on Apple Silicon (M1/M2/M3/M4)?**
A: Yes. Docker builds a `linux/arm64` image that runs well on Apple Silicon via Docker's Linux VM. For native setup, PyTorch CPU works natively on ARM. MPS (Metal) GPU acceleration is untested with OpenCLIP.

**Q: Can I use GPU acceleration on Mac?**
A: Not via Docker (macOS Docker runs a Linux VM without GPU passthrough). Native setup may work with PyTorch MPS on Apple Silicon, but this is untested with OpenCLIP. CPU performance on Apple Silicon is quite good regardless.

### Linux-Specific

**Q: I get "permission denied" when running Docker commands.**
A: Add your user to the `docker` group: `sudo usermod -aG docker $USER`, then log out and back in. Or prefix commands with `sudo`.

**Q: How do I set up GPU passthrough on Linux?**
A: Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), then restart Docker: `sudo systemctl restart docker`. Verify with: `docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`.

### Docker (All Platforms)

**Q: The build is taking a very long time. Is that normal?**
A: Yes, the first build downloads ~1.3 GB of dependencies including the ViT-L-14 model weights. Subsequent builds are much faster because Docker caches the layers. Only code changes (in `app/`) trigger a rebuild of the final layers.

**Q: How big is the Docker image?**
A: ~2 GB for the CPU variant, ~5 GB for the GPU (CUDA) variant. Most of the size is PyTorch and the model weights.

**Q: I get "Cannot connect to the Docker daemon" — what do I do?**
A: Start Docker Desktop (macOS/Windows) or the Docker service (Linux: `sudo systemctl start docker`). Wait a few seconds for it to initialize.

**Q: I get a `torchvision` error during build. What happened?**
A: Make sure the Dockerfile installs `torch torchvision` from the same PyTorch wheel index. The included Dockerfile already handles this correctly.

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
