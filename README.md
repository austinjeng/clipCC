# ClipCC

English | [繁體中文](README.zh-TW.md)


A cross-platform, Dockerized API that classifies video content against user-provided text labels. Upload a video file, provide a set of descriptive labels, and get back confidence scores for how well each label describes the video.

**Example use case:** Upload dashcam footage with labels like `"drunk driving"`, `"normal driving"`, `"distracted driving"` and get back which label best matches the video content.

---

## Features

- **6 SigLIP2 models** — from 0.4B to 1B parameters, 256px to 512px resolution. Hot-swap between models from the web UI or API without restarting.
- **Built-in web UI** at `http://localhost:8000/` — model selector, video upload, label input, and results visualization.
- **Three aggregation modes** — `mean` (average across frames), `max` (peak per label with timestamp), `temporal` (frame-by-frame timeline with segment detection).
- **Sigmoid scoring** — each label gets an independent confidence score between 0 and 1. Multiple labels can score high simultaneously.
- **Gemma 4 E2B exploration** *(optional, GPU-oriented)* — a vision-language model for open-ended Q&A and label scoring over sampled frames at the `/gemma` page. Loads on demand, off by default on CPU.
- **Hybrid mode** *(optional)* — SigLIP2 scores every frame, then Gemma 4 verifies the top-k labels at the `/hybrid` page.
- **Lightweight Docker image** — ~1 GB base image. Models download on demand and are cached in a Docker volume.
- **Pre-built images** — `docker pull ghcr.io/austinjeng/clipcc:latest` for instant setup on amd64 and arm64.
- **GPU support** — NVIDIA CUDA acceleration (10-30x faster than CPU).
- **Fail-closed authentication** — API key or explicit opt-out required.

### Quick Start (Docker)

Pull and run the pre-built image — no cloning or building required:

```bash
docker pull ghcr.io/austinjeng/clipcc:latest
docker run -p 8000:8000 -e ALLOW_UNAUTHENTICATED=true \
  -v clipcc-models:/app/models ghcr.io/austinjeng/clipcc:latest
```

Open `http://localhost:8000` in your browser. The default model (~1.5 GB) downloads automatically on first start.

**Production deployment** (use `--env-file` to keep secrets out of shell history):

```bash
# Create .env file
echo "API_KEY=your-secret-key" > .env

# Run
docker run -p 8000:8000 --env-file .env \
  -v clipcc-models:/app/models ghcr.io/austinjeng/clipcc:latest
```

### Building from Source

```bash
git clone https://github.com/austinjeng/clipCC.git
cd clipCC
```

---

## Table of Contents

- [Features](#features)
  - [Quick Start (Docker)](#quick-start-docker)
  - [Building from Source](#building-from-source)
- [Available Models](#available-models)
- [Setup by Platform](#setup-by-platform)
  - [Windows](#windows)
  - [macOS](#macos)
  - [Linux](#linux)
- [Your First Classification](#your-first-classification)
- [Web UI](#web-ui)
- [API Reference](#api-reference)
  - [POST /api/v1/classify](#post-apiv1classify)
  - [GET /api/v1/models](#get-apiv1models)
  - [POST /api/v1/models/load](#post-apiv1modelsload)
  - [GET /api/v1/models/active](#get-apiv1modelsactive)
  - [GET /live](#get-live)
  - [GET /ready](#get-ready)
  - [Gemma 4 & Hybrid Mode](#gemma-4--hybrid-mode)
- [Configuration](#configuration)
- [Authentication](#authentication)
- [GPU Support](#gpu-support)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Q&A](#qa)

---

## Available Models

ClipCC ships with 6 SigLIP2 models. The default model auto-loads at startup and can be changed via the `DEFAULT_MODEL_ID` environment variable.

| Model | Parameters | Resolution | Best for |
|---|---|---|---|
| `siglip2-base-patch16-256` | 0.4B | 256px | Fast inference, low memory (default for CPU) |
| `siglip2-base-patch16-384` | 0.4B | 384px | Better accuracy, still lightweight |
| `siglip2-large-patch16-256` | 0.9B | 256px | Higher quality, moderate speed |
| `siglip2-large-patch16-384` | 0.9B | 384px | High quality, moderate memory |
| `siglip2-so400m-patch14-384` | 1B | 384px | Best quality (default for GPU) |
| `siglip2-so400m-patch16-512` | 1B | 512px | Highest resolution, most memory |

**Switching models:**
- **Web UI:** Select from the dropdown and click **Load Model**
- **API:** `POST /api/v1/models/load` with `{"model_id": "siglip2-large-patch16-384"}`
- **Startup default:** Set `DEFAULT_MODEL_ID` environment variable

Models are downloaded from HuggingFace on first use (~1.5 GB for base models, ~3.5-4.5 GB for large/SO400M models) and cached for future runs.

---

## Setup by Platform

Pick your operating system and follow the steps. All three paths end at the same place: a running ClipCC server on `http://localhost:8000`.

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
docker compose --profile cpu-build build
```

The first build takes **5-10 minutes**. It downloads Python, ffmpeg, PyTorch, and supporting libraries. Model weights are downloaded separately on first startup, not during the build.

> **Skip the build:** to run the prebuilt image instead, use `docker compose --profile cpu up` — it pulls `ghcr.io/austinjeng/clipcc:latest`. Note that the prebuilt image does **not** include any local source changes; use the `cpu-build` profile above to run your own code.

**Step 3: Start the server**

```powershell
docker compose --profile cpu-build up
```

On first startup, the default model (`siglip2-base-patch16-256`, ~1.5 GB) downloads automatically. This happens once — the Docker volume caches it for future runs.

Wait for:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Auto-loaded model: siglip2-base-patch16-256
```

**Step 4: Verify**

Open a new PowerShell window:
```powershell
curl.exe http://localhost:8000/live
# {"status":"ok"}

curl.exe http://localhost:8000/ready
# {"status":"ready","model":"siglip2-base-patch16-256","pretrained":"siglip2-base-patch16-256","device":"cpu"}
```

Open **http://localhost:8000/** in your browser to access the web UI.

**Step 5: Stop the server**

```powershell
docker compose --profile cpu-build down
```

Model weights persist in the Docker volume — they won't need to re-download next time.

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

On first startup, the default model (`siglip2-base-patch16-256`, ~1.5 GB) downloads automatically. Subsequent starts use the cached model.

**Step 4: Verify**

Open a new PowerShell window (activate the venv first: `.\.venv\Scripts\Activate.ps1`):
```powershell
curl.exe http://localhost:8000/live
curl.exe http://localhost:8000/ready
```

> **Important:** Always use `curl.exe` (not `curl`) in PowerShell. The bare `curl` command in PowerShell is an alias for `Invoke-WebRequest`, which has different syntax and will not work with the examples in this guide.

Open **http://localhost:8000/** in your browser to access the web UI.

**Step 5: Stop the server**

Press `Ctrl+C` in the terminal running uvicorn.

You're set. Jump to [Your First Classification](#your-first-classification).

> **Note:** The default temp and cache paths (`/tmp/clipcc`, `/app/models`) are Linux paths — you **must** override them with Windows paths as shown above.

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
docker compose --profile cpu-build build
```

The first build takes **5-10 minutes**. It downloads Python, ffmpeg, PyTorch, and supporting libraries. Model weights are downloaded separately on first startup, not during the build.

> **Skip the build:** to run the prebuilt image instead, use `docker compose --profile cpu up` — it pulls `ghcr.io/austinjeng/clipcc:latest`. Note that the prebuilt image does **not** include any local source changes; use the `cpu-build` profile above to run your own code.

> **Apple Silicon note:** The image builds for `linux/arm64` and runs via Docker's Linux VM. PyTorch CPU inference works well on Apple Silicon through this layer. Native Metal/MPS acceleration is not available inside Docker — use the native option below if you want MPS.

**Step 3: Start the server**

```bash
docker compose --profile cpu-build up
```

On first startup, the default model (`siglip2-base-patch16-256`, ~1.5 GB) downloads automatically. This happens once — the Docker volume caches it for future runs.

Wait for:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Auto-loaded model: siglip2-base-patch16-256
```

**Step 4: Verify**

Open a new Terminal tab:
```bash
curl http://localhost:8000/live
# {"status":"ok"}

curl http://localhost:8000/ready
# {"status":"ready","model":"siglip2-base-patch16-256","pretrained":"siglip2-base-patch16-256","device":"cpu"}
```

Open **http://localhost:8000/** in your browser to access the web UI.

**Step 5: Stop the server**

```bash
docker compose --profile cpu-build down
```

Model weights persist in the Docker volume — they won't need to re-download next time.

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

On first startup, the default model (`siglip2-base-patch16-256`, ~1.5 GB) downloads automatically. Subsequent starts use the cached model.

**Step 4: Verify**

Open a new Terminal tab (activate the venv first: `source .venv/bin/activate`):
```bash
curl http://localhost:8000/live
curl http://localhost:8000/ready
```

Open **http://localhost:8000/** in your browser to access the web UI.

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
docker compose --profile cpu-build build
```

The first build takes **5-10 minutes**. It downloads Python, ffmpeg, PyTorch, and supporting libraries. Model weights are downloaded separately on first startup, not during the build.

> **Skip the build:** to run the prebuilt image instead, use `docker compose --profile cpu up` — it pulls `ghcr.io/austinjeng/clipcc:latest`. Note that the prebuilt image does **not** include any local source changes; use the `cpu-build` profile above to run your own code.

**Step 3: Start the server**

```bash
docker compose --profile cpu-build up
```

On first startup, the default model (`siglip2-base-patch16-256`, ~1.5 GB) downloads automatically. This happens once — the Docker volume caches it for future runs.

Wait for:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Auto-loaded model: siglip2-base-patch16-256
```

**Step 4: Verify**

Open a new terminal:
```bash
curl http://localhost:8000/live
# {"status":"ok"}

curl http://localhost:8000/ready
# {"status":"ready","model":"siglip2-base-patch16-256","pretrained":"siglip2-base-patch16-256","device":"cpu"}
```

Open **http://localhost:8000/** in your browser to access the web UI.

**Step 5: Stop the server**

```bash
docker compose --profile cpu-build down
```

Model weights persist in the Docker volume — they won't need to re-download next time.

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

On first startup, the default model (`siglip2-base-patch16-256`, ~1.5 GB) downloads automatically. Subsequent starts use the cached model.

**Step 4: Verify**

Open a new terminal (activate the venv first: `source .venv/bin/activate`):
```bash
curl http://localhost:8000/live
curl http://localhost:8000/ready
```

Open **http://localhost:8000/** in your browser to access the web UI.

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
    "confidence": 0.92
  },
  "scores": [
    {
      "label": "test pattern",
      "confidence": 0.92,
      "raw_similarity": 0.35
    },
    {
      "label": "outdoor scene",
      "confidence": 0.12,
      "raw_similarity": 0.17
    },
    {
      "label": "person walking",
      "confidence": 0.08,
      "raw_similarity": 0.15
    }
  ],
  "metadata": {
    "frames_analyzed": 5,
    "video_duration_seconds": 5.0,
    "model": "siglip2-base-patch16-256",
    "device": "cpu",
    "aggregation": "mean",
    "processing_time_seconds": 1.4,
    "disclaimer": "Scores are relative to the supplied labels, ...",
    "model_type": "siglip2",
    "score_semantics": "siglip2_pairwise_sigmoid"
  }
}
```

> **Note:** SigLIP2 uses sigmoid scoring — each label gets an independent confidence between 0 and 1. Scores do **not** sum to 1.0. The `"test pattern"` label should score highest since the test video is literally an ffmpeg test pattern.

### Classify with max aggregation

```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -F "video=@test_video.mp4" \
  -F 'labels=["test pattern","outdoor scene","person walking"]' \
  -F "aggregation=max"
```

Max mode returns `peak_frame_index` and `approx_timestamp_seconds` for each label, indicating which frame produced the peak confidence.

### Classify with temporal aggregation

```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -F "video=@test_video.mp4" \
  -F 'labels=["test pattern","outdoor scene","person walking"]' \
  -F "fps=1.0" \
  -F "aggregation=temporal"
```

Temporal mode returns a frame-by-frame timeline, detected segments where labels exceed the confidence threshold, and label summaries with duration-weighted statistics. See [API Reference](#post-apiv1classify) for the full response format.

---

## Web UI

ClipCC includes a built-in web interface at **http://localhost:8000/**. The web UI is optional — all features are also available via the API.

Two optional exploration pages are linked from the top nav: **`/gemma`** (Gemma 4 Q&A + label scoring) and **`/hybrid`** (SigLIP2 → Gemma verification). Both require the Gemma slot — set `GEMMA_ENABLED=true` (GPU recommended) and warm it first. See [Gemma 4 & Hybrid Mode](#gemma-4--hybrid-mode).

### What you'll see

- **Model selector** — dropdown with all 6 SigLIP2 models and a status indicator (green = loaded and ready)
- **Video upload** — drag or select `.mp4`, `.avi`, `.mov`, or `.mkv` files
- **Labels input** — enter labels separated by commas (3-10 labels)
- **Aggregation mode** — choose between mean, max, or temporal
- **Temporal controls** — when temporal mode is selected, sliders appear for threshold, gap tolerance, and minimum duration

### Walkthrough

1. The default model auto-loads at startup. Wait for the green status dot.
2. Upload a video file.
3. Enter labels separated by commas, e.g.: `driving, parking, reversing`
4. Select an aggregation mode.
5. Click **Classify**.
6. Results appear as horizontal confidence bars. In temporal mode, a timeline chart and segment table are also displayed.

### Switching models

1. Select a different model from the dropdown.
2. Click **Load Model**.
3. Wait for the spinner to finish (downloads the model if not cached, then loads it).
4. The status dot turns green when ready.
5. Classify again — the new model is now active.

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
| `prompt_template` | string | No | `"This is a photo of {}."` | Template for text prompts. `{}` is replaced with each label. Max 500 characters. |
| `fps` | float | No | `1.0` | Frame sampling rate. Range: 0.1-5.0. Higher = more frames = slower but potentially more accurate. |
| `aggregation` | string | No | `"mean"` | Score aggregation method: `"mean"`, `"max"`, or `"temporal"` |
| `threshold` | float | No | model default | Confidence threshold for temporal segment detection. Range: 0.0-1.0. Only valid when `aggregation=temporal`. |
| `gap_tolerance` | float | No | `2.0` | Maximum gap in seconds between frames to merge into one segment. Range: 0.0-10.0. Only valid when `aggregation=temporal`. |
| `min_duration` | float | No | `1.0` | Minimum segment duration in seconds to include in results. Range: 0.0-10.0. Only valid when `aggregation=temporal`. |

#### Aggregation Methods

**`mean`** (default) — Averages confidence scores across all sampled frames. Best for answering "What is this video mostly showing?"

**`max`** — Returns the peak confidence score for each label across all frames independently. Best for answering "Did this happen at any point in the video?" Each score includes `peak_frame_index` and `approx_timestamp_seconds` indicating which frame produced the peak.

**`temporal`** — Returns a frame-by-frame timeline with per-label confidence scores, plus detected segments where labels exceed the threshold. Best for answering "When and for how long did each event occur?" Supports configurable threshold, gap tolerance (to merge nearby segments), and minimum duration (to filter noise). Returns segment statistics including active average, coverage ratio, and duration-weighted confidence.

#### Prompt Template

The model scores video frames against text prompts. By default, each label is wrapped as `"This is a photo of {label}."`. You can customize this:

```bash
# Default: "This is a photo of {label}."
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
    "confidence": 0.72
  },
  "scores": [
    {
      "label": "drunk driving",
      "confidence": 0.15,
      "raw_similarity": 0.27
    },
    {
      "label": "normal driving",
      "confidence": 0.72,
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
    "model": "siglip2-base-patch16-256",
    "device": "cpu",
    "aggregation": "mean",
    "processing_time_seconds": 12.3,
    "disclaimer": "Scores are relative to the supplied labels, not calibrated probabilities. Not suitable for safety-critical decisions.",
    "model_type": "siglip2",
    "score_semantics": "siglip2_pairwise_sigmoid"
  }
}
```

#### Response — Max Mode

```json
{
  "best_match": {
    "label": "normal driving",
    "confidence": 0.85
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
      "confidence": 0.85,
      "raw_similarity": 0.34,
      "peak_frame_index": 85,
      "approx_timestamp_seconds": 85.0
    }
  ],
  "metadata": {
    "frames_analyzed": 297,
    "video_duration_seconds": 297.4,
    "model": "siglip2-base-patch16-256",
    "device": "cuda",
    "aggregation": "max",
    "processing_time_seconds": 2.1,
    "disclaimer": "Scores are relative to the supplied labels, not calibrated probabilities. Max-mode scores are independent peaks per label and do not sum to 1. Not suitable for safety-critical decisions.",
    "model_type": "siglip2",
    "score_semantics": "siglip2_pairwise_sigmoid"
  }
}
```

#### Response — Temporal Mode

```json
{
  "best_match": {
    "label": "normal driving",
    "confidence": 0.85
  },
  "scores": [
    {
      "label": "drunk driving",
      "confidence": 0.38,
      "raw_similarity": 0.29
    },
    {
      "label": "normal driving",
      "confidence": 0.85,
      "raw_similarity": 0.34
    }
  ],
  "metadata": {
    "frames_analyzed": 297,
    "video_duration_seconds": 297.4,
    "model": "siglip2-base-patch16-256",
    "device": "cpu",
    "aggregation": "temporal",
    "processing_time_seconds": 14.2,
    "disclaimer": "...",
    "model_type": "siglip2",
    "score_semantics": "siglip2_pairwise_sigmoid"
  },
  "temporal": {
    "timeline": [
      {"timestamp": 0.0, "frame_index": 0, "scores": {"drunk driving": 0.12, "normal driving": 0.78}},
      {"timestamp": 1.0, "frame_index": 1, "scores": {"drunk driving": 0.10, "normal driving": 0.81}}
    ],
    "segments": [
      {
        "label": "normal driving",
        "start_time": 0.0,
        "end_time": 120.5,
        "duration": 120.5,
        "stats": {
          "active_avg": 0.76,
          "interval_avg": 0.74,
          "coverage_ratio": 0.95,
          "active_duration": 114.5
        },
        "peak_confidence": 0.85,
        "peak_timestamp": 42.0
      }
    ],
    "label_summaries": [
      {
        "label": "normal driving",
        "segment_count": 2,
        "total_active_duration": 180.0,
        "total_segment_duration": 200.0,
        "peak_confidence": 0.85,
        "duration_weighted_confidence": 0.74
      }
    ],
    "best_segment": {
      "label": "normal driving",
      "start_time": 0.0,
      "end_time": 120.5,
      "duration": 120.5,
      "stats": {"active_avg": 0.76, "interval_avg": 0.74, "coverage_ratio": 0.95, "active_duration": 114.5},
      "peak_confidence": 0.85,
      "peak_timestamp": 42.0
    },
    "threshold_mode": "absolute",
    "effective_threshold": 0.5,
    "threshold_was_defaulted": true
  }
}
```

#### Response Fields

| Field | Description |
|---|---|
| `best_match.label` | The label with the highest confidence score |
| `best_match.confidence` | The confidence value of the best match |
| `scores[].label` | The original label text |
| `scores[].confidence` | Confidence score. SigLIP2: independent sigmoid (0-1), scores do not sum to 1. See `score_semantics`. |
| `scores[].raw_similarity` | Unscaled cosine similarity between frame and text embeddings |
| `scores[].peak_frame_index` | (Max mode only) 0-based index of the frame that produced the peak score |
| `scores[].approx_timestamp_seconds` | (Max mode only) Approximate timestamp of the peak frame (`frame_index / fps`) |
| `metadata.model` | Model ID used for classification |
| `metadata.device` | Compute device (`cpu` or `cuda`) |
| `metadata.aggregation` | Aggregation method used |
| `metadata.model_type` | Model backend type (`siglip2`) |
| `metadata.score_semantics` | Scoring method identifier (`siglip2_pairwise_sigmoid`) |
| `metadata.processing_time_seconds` | Wall-clock time for the inference pipeline |
| `metadata.disclaimer` | Reminder that scores are relative, not absolute |
| `temporal.timeline` | (Temporal only) Frame-by-frame scores for each label |
| `temporal.segments` | (Temporal only) Detected time segments where a label exceeded the threshold |
| `temporal.segments[].stats` | Segment statistics: `active_avg`, `interval_avg`, `coverage_ratio`, `active_duration` |
| `temporal.label_summaries` | (Temporal only) Per-label aggregate statistics across all segments |
| `temporal.best_segment` | (Temporal only) The segment with the highest peak confidence, or `null` |
| `temporal.threshold_mode` | (Temporal only) `"absolute"` (SigLIP2) |
| `temporal.effective_threshold` | (Temporal only) The threshold value used (explicit or model default) |
| `temporal.threshold_was_defaulted` | (Temporal only) `true` if the model's default threshold was used |

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
| 422 | Prompt too long for model | `"Prompt '...' has 83 tokens and will be truncated..."` |
| 422 | Invalid fps | `"FPS value 10.0 is invalid. Must be between 0.1 and 5.0."` |
| 422 | Temporal params with non-temporal mode | Temporal parameters are only valid when `aggregation=temporal` |
| 429 | Too many concurrent uploads | `"Too many uploads in progress. Please retry in a moment."` |
| 429 | Too many concurrent inferences | `"Too many inference requests in progress. Please retry in a moment."` |
| 503 | No model loaded | `"Model not loaded"` |
| 504 | Processing timeout | `"Inference timed out after 300.0s."` |

---

### GET /api/v1/models

List all available models with their cache and load status.

```bash
curl http://localhost:8000/api/v1/models
```

```json
[
  {
    "model_id": "siglip2-base-patch16-256",
    "display_name": "SigLIP2 Base (256px)",
    "model_type": "siglip2",
    "params": "0.4B",
    "resolution": 256,
    "loaded": true,
    "cached": true
  },
  {
    "model_id": "siglip2-large-patch16-384",
    "display_name": "SigLIP2 Large (384px)",
    "model_type": "siglip2",
    "params": "0.9B",
    "resolution": 384,
    "loaded": false,
    "cached": false
  }
]
```

| Field | Description |
|---|---|
| `loaded` | `true` if this model is currently active and serving requests |
| `cached` | `true` if model weights are already downloaded locally |

---

### POST /api/v1/models/load

Load a model by ID. Downloads the model if not cached. If another model is currently active, it will be replaced after in-flight requests complete.

**Content-Type:** `application/json`

```bash
curl -X POST http://localhost:8000/api/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "siglip2-large-patch16-384"}'
```

**Success response:**
```json
{"status": "loaded", "model_id": "siglip2-large-patch16-384"}
```

**Error responses:**

| Status | Condition |
|---|---|
| 400 | Unknown `model_id` |
| 500 | Model failed to load |

---

### GET /api/v1/models/active

Get metadata for the currently loaded model, including temporal mode defaults.

```bash
curl http://localhost:8000/api/v1/models/active
```

```json
{
  "model_id": "siglip2-base-patch16-256",
  "display_name": "SigLIP2 Base (256px)",
  "model_type": "siglip2",
  "params": "0.4B",
  "resolution": 256,
  "device": "cpu",
  "temporal_defaults": {
    "threshold": 0.5,
    "threshold_mode": "absolute",
    "gap_tolerance": 2.0,
    "min_duration": 1.0
  }
}
```

Returns `404` if no model is loaded.

---

### Gemma 4 & Hybrid Mode

> **Optional feature, off by default on CPU.** The Gemma 4 E2B vision-language model loads **only when warmed** (never at startup) and is memory-heavy: **~11.4 GB VRAM at bf16 on GPU**, **~22 GB RAM at fp32 on CPU** (demo-only). It is disabled on the CPU Docker profiles; enable with `GEMMA_ENABLED=true` on a GPU host, then call `/api/v1/gemma/warm`. Web pages: `/gemma` and `/hybrid`.

#### GET /api/v1/gemma/status

Slot lifecycle and capability info — `enabled`, `state` (`idle` / `loading` / `loaded` / `failed`), `device`, and the active `model_id`.

```bash
curl http://localhost:8000/api/v1/gemma/status
```

#### POST /api/v1/gemma/warm

Trigger the one-time model load into the slot. Returns `202` with the new state, `503` if there isn't enough free memory, or `409` if a load is already in progress. Gemma and hybrid requests return `503` until the slot is `loaded`.

```bash
curl -X POST http://localhost:8000/api/v1/gemma/warm
```

#### POST /api/v1/gemma/label_scores

Score labels against frames sampled from an analysis window using Gemma (multipart form).

| Field | Required | Description |
|---|---|---|
| `video` | yes | Video file |
| `labels` | yes | JSON array of label strings (max `GEMMA_MAX_LABELS`) |
| `window_start` | no | Start of the analysis window in seconds (default `0`) |
| `instruction` | no | Extra instruction prepended to the prompt (≤2000 chars) |
| `max_frames` | no | Frames to sample, clamped to `[1, GEMMA_MAX_FRAMES_CAP]` |

#### POST /api/v1/gemma/qa

Open-ended question answering over sampled frames (multipart form): `video`, `prompt` (required, ≤2000 chars), optional `window_start` and `max_frames`.

#### POST /api/v1/hybrid

Two-phase pipeline: SigLIP2 scores every frame, then Gemma 4 verifies the top labels (multipart form).

| Field | Required | Description |
|---|---|---|
| `video` | yes | Video file |
| `labels` | yes | JSON array of label strings |
| `fps` | no | SigLIP2 sampling rate, `0.1`–`5.0` (default `1.0`) |
| `aggregation` | no | `max` or `mean` (default `max`) |
| `threshold` | no | Score threshold `0`–`1` (default `0.5`) |
| `top_k` | no | Top SigLIP2 labels passed to Gemma, `1`–`GEMMA_MAX_FRAMES_CAP` (default `3`) |
| `max_verified_labels` | no | Cap on Gemma-verified labels (default `HYBRID_MAX_VERIFIED_LABELS`) |
| `instruction` | no | Extra instruction for the Gemma verification prompt |

---

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
  "model": "siglip2-base-patch16-256",
  "pretrained": "siglip2-base-patch16-256",
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
| `DEFAULT_MODEL_ID` | `siglip2-base-patch16-256` | Model to auto-load at startup. Set to a larger model (e.g. `siglip2-so400m-patch14-384`) if you have GPU/more memory. |
| `MAX_FILE_SIZE_MB` | `500` | Maximum upload file size in megabytes |
| `MAX_DURATION_SECONDS` | `300` | Maximum video duration (5 minutes) |
| `MAX_FRAMES` | `300` | Hard cap on total frames extracted per request |
| `DEFAULT_FPS` | `1.0` | Default frame sampling rate when not specified in the request |
| `BATCH_SIZE` | `32` | Number of frames processed per inference batch |
| `MAX_CONCURRENT_REQUESTS` | `2` (CPU) / `1` (GPU) | Max simultaneous inference requests. GPU defaults to 1 to avoid VRAM exhaustion. |
| `MAX_UPLOAD_CONCURRENCY` | `MAX_CONCURRENT_REQUESTS + 2` | Max simultaneous uploads being parsed. Bounds temp disk usage. |
| `API_KEY` | (unset) | If set, all requests to `/api/v1/classify` and `/ready` must include an `X-API-Key` header with this value |
| `ALLOW_UNAUTHENTICATED` | `false` | Must be `true` if `API_KEY` is not set. Server refuses to start without explicit auth configuration. |
| `SKIP_MODEL_AUTOLOAD` | `false` | Skip automatic model loading at startup. The server starts and `/live` works, but `/ready` returns 503 until a model is loaded via `/api/v1/models/load`. Useful for CI smoke tests or deferred-load deployments. |
| `FFMPEG_TIMEOUT_SECONDS` | `120` | Timeout for ffmpeg/ffprobe subprocess calls |
| `REQUEST_TIMEOUT_SECONDS` | `300` | End-to-end timeout for the entire inference pipeline per request |
| `CLIP_CACHE_DIR` | `/app/models` | Directory where model weights are downloaded and cached. |
| `TEMP_DIR` | `/tmp/clipcc` | Directory for temporary upload and frame files. **Windows native users:** override to a Windows path like `C:\temp\clipcc`. |
| `GEMMA_ENABLED` | `true` (`false` on CPU Docker profiles) | Enables the Gemma 4 / hybrid slot. Loads on warm only, never at startup. ~11.4 GB VRAM (GPU bf16) / ~22 GB RAM (CPU fp32) when warmed. See `.env.example` for the full set of `GEMMA_*` / `HYBRID_*` tuning vars. |

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
| **Windows (Native)** | Works if CUDA-compatible PyTorch is installed: `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126` (run after `pip install -r requirements.txt`, which installs the CPU build) |
| **macOS (Docker)** | Not supported. Docker on macOS runs a Linux VM without GPU passthrough. |
| **macOS (Native)** | PyTorch MPS (Apple Silicon) may work but is untested with SigLIP2 models. CPU is the safe default. |

### Build and run with GPU (Docker)

```bash
# Build with CUDA support (larger image, ~5 GB)
docker compose --profile gpu build

# Start with GPU
docker compose --profile gpu up
```

The GPU profile defaults to `siglip2-so400m-patch14-384` (1B parameters) — the highest quality model that fits comfortably in most GPUs. You can switch to other models from the web UI or API at any time.

### Verify GPU is being used

```bash
curl http://localhost:8000/ready
# {"status":"ready","model":"siglip2-so400m-patch14-384",...,"device":"cuda"}
```

If `device` says `"cuda"`, GPU acceleration is active. If it says `"cpu"`, check that NVIDIA Container Toolkit is installed and Docker can see your GPU: `docker run --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi`.

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
python -m pytest tests/test_config.py tests/test_temp_store.py tests/test_scoring.py tests/test_resource_gates.py tests/test_frame_timeline.py tests/test_temporal_policy.py -v
```

Run the full suite (requires ffmpeg in PATH; downloads the default model, ~1.5 GB, into a pytest temp dir once per run):
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
| `test_api.py` | 35 | Full integration (health, validation, classify, contrast) |
| `test_base_model.py` | 2 | BaseModel abstraction contract |
| `test_config.py` | 24 | Settings validation, auth config, path expansion |
| `test_download_script.py` | 7 | Model download script |
| `test_frame_timeline.py` | 9 | Frame intervals, timestamps, gap/duration math |
| `test_gemma_api.py` | 21 | Gemma endpoints (status, warm, label_scores, qa), memory reserve |
| `test_gemma_integration.py` | 4 | Gated real-model Gemma tests (`GEMMA_INTEGRATION=1`) |
| `test_gemma_prompts.py` | 22 | Gemma prompt build, strict ID-keyed parsing |
| `test_gemma_sampler.py` | 14 | Analysis window, timestamp planning, frame seeks |
| `test_gemma_verdict.py` | 8 | Hybrid verdict prompt and parse |
| `test_hybrid_api.py` | 6 | Hybrid endpoint validation and flow |
| `test_hybrid_integration.py` | 1 | Gated hybrid end-to-end test |
| `test_hybrid_middleware.py` | 4 | Hybrid route gating |
| `test_hybrid_select.py` | 7 | Label gating/ranking, top-k frame spread selection |
| `test_inference_runner.py` | 5 | Timeout, cancellation, worker teardown |
| `test_integration.py` | 1 | End-to-end integration flow |
| `test_middleware.py` | 21 | Auth, upload gates, body size |
| `test_model_manager.py` | 44 | Model registry, hot-swap, lease concurrency, non-blocking load |
| `test_residency.py` | 11 | Residency ledger reserve/commit/rollback |
| `test_resource_gates.py` | 7 | Concurrency limiters |
| `test_scoring.py` | 37 | Mean/max/temporal/contrast aggregation, scoring context |
| `test_siglip2_model.py` | 10 | SigLIP2 model loading, sigmoid scoring |
| `test_temp_store.py` | 6 | File upload, cleanup, janitor |
| `test_temporal_policy.py` | 6 | Scoring policy, threshold mode |
| `test_video.py` | 12 | ffprobe, validation, frame extraction, missing-ffmpeg errors |
| `test_vlm_slot.py` | 14 | VLM slot state machine, warm lifecycle |
| **Total** | **338** | |

---

## Project Structure

```
clipCC/
├── Dockerfile                  # Single-stage build (~1 GB, no baked weights)
├── docker-compose.yml          # CPU and GPU profiles with model volume
├── requirements.txt            # Python dependencies
├── .env.example                # Configuration reference
├── pytest.ini                  # Pytest configuration
├── app/
│   ├── main.py                 # FastAPI app factory, routes, pipeline orchestration
│   ├── config.py               # Pydantic settings from environment variables
│   ├── middleware.py            # ASGI middleware: auth, upload concurrency, body size
│   ├── resource_gates.py       # anyio CapacityLimiters for upload and inference
│   ├── temp_store.py           # Temp file lifecycle and cleanup
│   ├── inference_runner.py     # Threaded pipeline runner with cooperative timeout
│   ├── models/
│   │   ├── base_model.py       # Abstract base class for all model backends
│   │   ├── siglip2_model.py    # SigLIP2 model via HuggingFace transformers (sigmoid scoring)
│   │   ├── model_manager.py    # Model registry, hot-swap with lease-based concurrency
│   │   ├── residency.py        # Per-device atomic model-memory ledger
│   │   ├── vlm_slot.py         # Gemma load-once state machine (warm-only)
│   │   └── gemma_vlm.py        # Gemma 4 E2B wrapper
│   ├── services/
│   │   ├── video.py            # ffprobe validation and ffmpeg frame extraction
│   │   ├── scoring.py          # Mean/max/temporal/contrast aggregation over frame scores
│   │   ├── frame_timeline.py   # Frame interval math for temporal mode
│   │   ├── temporal_policy.py  # SigLIP2 scoring policy (threshold behavior)
│   │   ├── gemma_sampler.py    # Timestamp-seek frame sampling for Gemma
│   │   ├── gemma_prompts.py    # Gemma prompt build + strict ID-keyed parse
│   │   └── hybrid_select.py    # Hybrid label gating and top-k frame selection
│   ├── schemas/
│   │   ├── response.py         # Pydantic response models
│   │   ├── gemma.py            # Gemma response models
│   │   └── hybrid.py           # Hybrid response models
│   ├── errors/
│   │   └── handlers.py         # Custom HTTP exceptions
│   └── static/
│       ├── index.html          # Web UI (model selector, temporal controls, chart visualization)
│       ├── gemma.html          # Gemma 4 exploration UI
│       ├── hybrid.html         # Hybrid mode UI
│       └── vendor/
│           └── chart.min.js    # Chart.js 4.4.9 for temporal timeline rendering
└── tests/                      # 338 tests across 26 files
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
[ModelManager] ── Acquire model lease (or 503 if no model loaded)
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
        │     ├── model.score_batch() → ScoreBatch (confidence, similarity, logits)
        │     └── Delete consumed frame files
        ├── Aggregate scores:
        │     ├── mean: average confidence across frames
        │     ├── max: peak confidence per label with timestamp
        │     └── temporal: frame timeline → threshold detection
        │           → segment merging → label summaries
        └── Return ClassifyResponse
        |
        v
[Cleanup] ── Delete all temp files (guaranteed via finally)
```

### Key Design Decisions

- **Synchronous pipeline in a thread:** The inference pipeline (ffmpeg + PyTorch) is blocking. It runs in a worker thread via `anyio.to_thread.run_sync` so the async event loop stays responsive for health checks and concurrent gate decisions.

- **Two-level concurrency control:** Upload concurrency and inference concurrency are controlled by separate `anyio.CapacityLimiter` instances. Uploads are bounded but higher-limit; inference slots are scarce (especially on GPU).

- **Cooperative timeout:** The `InferenceRunner` uses a `threading.Event` flag checked between batches. On timeout, it sets the flag, kills any active ffmpeg subprocess, and waits for the worker thread to exit before releasing the inference slot. No abandoned threads.

- **On-demand model loading:** Models are downloaded from HuggingFace on first use and cached locally (Docker volume or local directory). The `ModelManager` coordinates hot-swap with lease-based concurrency — in-flight requests complete on the current model before a new model is loaded.

- **Scoring semantics:** SigLIP2 models use pairwise sigmoid scoring (independent per-label, scores between 0 and 1, which do not sum to 1). The `score_semantics` field in the response identifies the method used, and the temporal aggregation pipeline selects the matching threshold policy.

---

## Q&A

### General

**Q: What video formats are supported?**
A: `.mp4`, `.avi`, `.mov`, and `.mkv`. The video must have a single video stream, resolution at most 3840x2160, and duration at most 5 minutes (configurable).

**Q: How long does classification take?**
A: Depends on video length, fps, and hardware. A 5-minute video at 1fps = 300 frames. On CPU, expect 1-5 minutes. On GPU, expect 10-30 seconds. The `processing_time_seconds` field in the response tells you the exact time.

**Q: What does "confidence" mean?**
A: Confidence is a **sigmoid** score between 0 and 1 — a score of 0.8 means the model is 80% confident that label applies, independent of other labels. Scores do **not** sum to 1. The `score_semantics` field in the response identifies the scoring method (`siglip2_pairwise_sigmoid`).

**Q: What does `raw_similarity` mean?**
A: The unscaled cosine similarity between the video frame embedding and the text label embedding, before any model-specific scoring transformation. Useful for advanced users who want to do their own scoring.

**Q: Can I use this for real-time video?**
A: No. ClipCC is designed for offline video classification. It processes uploaded video files, not live streams.

**Q: How do I switch models?**
A: From the web UI, select a model from the dropdown and click **Load Model**. Via the API, send `POST /api/v1/models/load` with `{"model_id": "siglip2-large-patch16-384"}`. The new model downloads if not cached, then loads. In-flight requests complete on the old model first.

**Q: What is temporal aggregation?**
A: Temporal mode (`aggregation=temporal`) analyzes confidence scores frame by frame, detects time segments where labels exceed a threshold, and returns a timeline with segment statistics. Use it to answer "When and for how long did each event occur?" — for example, finding the exact timestamps of a traffic violation in dashcam footage.

**Q: Do models persist across restarts?**
A: Yes. Docker caches downloaded models in a named volume (`clipcc-models`). Native setups cache in `CLIP_CACHE_DIR`. Models only download once.

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
A: Yes. Docker builds a `linux/arm64` image that runs well on Apple Silicon via Docker's Linux VM. For native setup, PyTorch CPU works natively on ARM.

**Q: Can I use GPU acceleration on Mac?**
A: Not via Docker (macOS Docker runs a Linux VM without GPU passthrough). Native setup may work with PyTorch MPS on Apple Silicon, but this is untested with SigLIP2 models. CPU performance on Apple Silicon is quite good regardless.

### Linux-Specific

**Q: I get "permission denied" when running Docker commands.**
A: Add your user to the `docker` group: `sudo usermod -aG docker $USER`, then log out and back in. Or prefix commands with `sudo`.

**Q: How do I set up GPU passthrough on Linux?**
A: Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), then restart Docker: `sudo systemctl restart docker`. Verify with: `docker run --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi`.

### Docker (All Platforms)

**Q: The build is taking a long time. Is that normal?**
A: Yes, the first build downloads ~1 GB of dependencies including Python, PyTorch, and transformers. Subsequent builds are much faster because Docker caches the layers. Model weights are downloaded separately on first startup, not during the build.

**Q: How big is the Docker image?**
A: ~1 GB for the CPU variant, ~5 GB for the GPU (CUDA) variant. Model weights are stored separately in a Docker volume.

**Q: I get "Cannot connect to the Docker daemon" — what do I do?**
A: Start Docker Desktop (macOS/Windows) or the Docker service (Linux: `sudo systemctl start docker`). Wait a few seconds for it to initialize.

**Q: I get a `torchvision` error during build. What happened?**
A: Make sure the Dockerfile installs `torch torchvision` from the same PyTorch wheel index. The included Dockerfile already handles this correctly.

### API Usage

**Q: How do I choose good labels?**
A: Labels should be distinct, descriptive, and cover the range of content you expect. More specific labels work better than vague ones. For example, `"person running on a track"` is better than `"movement"`.

**Q: What's the best `fps` value?**
A: `1.0` (default) works well for most use cases. Increase to 2-5 for short videos where you need finer temporal resolution. Lower to 0.5 for long videos where the content is mostly static. Higher fps = more frames = slower processing but potentially more accurate.

**Q: When should I use `mean`, `max`, or `temporal` aggregation?**
A: Use `mean` when you want to know the dominant content of the video ("What is this video mostly about?"). Use `max` when you want to detect if something appeared at any point ("Did this happen at all?"). Use `temporal` when you need to know when and for how long each label appeared, with segment detection and timeline visualization.

**Q: Can I send more than 10 labels?**
A: No. The limit is 3-10 labels per request. If you need more categories, make multiple requests with different label sets.

**Q: My labels produce "identical token sequences" error. Why?**
A: The model's tokenizer may produce the same token sequence for labels that look different as text (e.g., extra spaces, Unicode variants). The API rejects these because the model literally cannot distinguish between them. Use more distinct wording.

**Q: What is the prompt template for?**
A: The model scores video frames against text prompts. Wrapping labels in a natural sentence often improves accuracy over bare labels. The default `"This is a photo of {}."` works well for general use. For domain-specific content, try templates like `"a surveillance camera recording of {}"` or `"a dashcam video showing {}"`.

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
