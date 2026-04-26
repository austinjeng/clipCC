# ClipCC: CLIP-Based Video Classification API

## Overview

A Dockerized FastAPI application that accepts a video file and a set of text labels, then uses OpenCLIP to score how well each label describes the video content. Returns probability scores, raw similarities, and the best-matching label.

**Use case example:** Upload dashcam footage with labels like "drunk driving", "normal driving", "distracted driving" — the API returns which label best matches the video.

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Frame extraction | Uniform sampling at configurable fps (default 1.0) | Simple, predictable, effective for most classification |
| CLIP variant | OpenCLIP `ViT-L-14` / `laion2b_s32b_b82k` | Best accuracy/size tradeoff; better zero-shot than original CLIP |
| Output format | Raw similarities + softmax probabilities + best_match | Maximum flexibility for API consumers |
| Video limits | 5 min / 500MB | Keeps synchronous API responsive |
| GPU strategy | Auto-detect, single image for both | Maximum portability |
| Text prompts | Auto-template with optional custom prompts | Easy for casual users, flexible for power users |
| Architecture | Monolith FastAPI | Simplest for v1; inference fits within HTTP timeout |

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
| `labels` | JSON string | yes | — | Array of 3–10 text labels |
| `prompt_template` | string | no | `"a video of {}"` | Template for text prompts; `{}` replaced with each label |
| `fps` | float | no | `1.0` | Frame sampling rate (0.1–5.0) |

### Response (200 OK)

```json
{
  "best_match": {
    "label": "normal_driving",
    "probability": 0.45
  },
  "scores": [
    {
      "label": "drunk driving",
      "probability": 0.15,
      "raw_similarity": 0.27
    },
    {
      "label": "normal driving",
      "probability": 0.45,
      "raw_similarity": 0.31
    }
  ],
  "metadata": {
    "frames_analyzed": 297,
    "video_duration_seconds": 297.4,
    "model": "ViT-L-14",
    "device": "cuda",
    "processing_time_seconds": 12.3
  }
}
```

### Error Responses

All errors return a JSON body with a human-friendly `detail` message.

| Status | Condition | Example message |
|---|---|---|
| 413 | File too large | "Your video is 620MB, which exceeds the 500MB limit. Try trimming or compressing it." |
| 422 | Duration too long | "Video duration is 8m32s, which exceeds the 5-minute limit. Please upload a shorter clip." |
| 422 | Label count invalid | "Please provide between 3 and 10 labels. You provided 2." |
| 422 | Invalid prompt template | "Prompt template must contain '{}' as a placeholder for labels." |
| 422 | Invalid fps | "FPS must be between 0.1 and 5.0. You provided 10.0." |
| 415 | Unsupported format | "Unsupported format '.webm'. Supported: mp4, avi, mov, mkv." |
| 500 | Processing failure | "An error occurred while processing your video. Please try again or use a different file." |

## Processing Pipeline

```
Upload -> Validate -> Extract Frames -> Score -> Respond
```

### 1. Validate

- Check file size from content-length header before reading the full body
- Save uploaded file to temp directory
- Probe with `ffprobe` for duration and format validation
- Validate label count (3–10), prompt template contains `{}`
- Fail fast with specific error at each step

### 2. Extract Frames

- Use `ffmpeg` subprocess to extract frames at requested fps
- Output as JPEG to temp directory (fast decode, sufficient quality for CLIP)
- Command: `ffmpeg -i input.mp4 -vf fps={fps} -q:v 2 frame_%05d.jpg`

### 3. Score

- Load frames as PIL images
- Batch process through OpenCLIP (batch size 32)
- Text encoding cached per request (same labels across all frames)
- Compute cosine similarity per frame per label
- Aggregate via mean of per-frame softmax probabilities

### 4. Cleanup

- Delete temp video file and extracted frames in `finally` block
- Guaranteed cleanup on success or error

### Model Loading

- Load OpenCLIP model once at FastAPI startup event
- Keep in memory for all requests
- Log device selection (cuda/cpu) at startup

## Docker & Infrastructure

### Dockerfile

- Base: `python:3.11-slim`
- System dependencies: `ffmpeg`
- Python dependencies: `fastapi`, `uvicorn`, `open_clip_torch`, `torch`, `Pillow`, `python-multipart`
- Single image works for both GPU and CPU

### docker-compose.yml

```yaml
services:
  clipcc:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MAX_FILE_SIZE_MB=500
      - MAX_DURATION_SECONDS=300
      - DEFAULT_FPS=1.0
      - MODEL_NAME=ViT-L-14
      - PRETRAINED=laion2b_s32b_b82k
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### Configuration

All limits and model settings via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MAX_FILE_SIZE_MB` | 500 | Maximum upload size in MB |
| `MAX_DURATION_SECONDS` | 300 | Maximum video duration in seconds |
| `DEFAULT_FPS` | 1.0 | Default frame sampling rate |
| `MODEL_NAME` | ViT-L-14 | OpenCLIP model architecture |
| `PRETRAINED` | laion2b_s32b_b82k | OpenCLIP pretrained weights |
| `BATCH_SIZE` | 32 | Frames per inference batch |

### Model Caching

- Volume mount at `/app/models` persists downloaded weights across restarts
- `OPENCLIP_CACHE_DIR` env var points to this volume

### Image Size

- ~3–4GB with CUDA PyTorch
- ~900MB model weights on first run (cached to volume)

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
│   ├── models/
│   │   ├── __init__.py
│   │   └── clip_model.py     # OpenCLIP loading and inference
│   ├── services/
│   │   ├── __init__.py
│   │   ├── video.py          # Frame extraction via ffmpeg
│   │   └── scoring.py        # Aggregation logic (mean, softmax)
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

- **`clip_model.py`** — Owns model loading and raw inference. Input: images + texts. Output: similarity matrix.
- **`video.py`** — Owns ffmpeg interaction. Input: file path + fps. Output: list of frame paths.
- **`scoring.py`** — Owns aggregation. Input: per-frame similarities. Output: final probabilities + best match.
- **`main.py`** — Orchestrates the pipeline, handles HTTP layer only.

## Out of Scope for v1

- Async job queue / background processing
- WebSocket progress streaming
- Multiple model support at runtime
- Authentication / rate limiting
- Video pre-processing (resize, crop)
- Per-frame results (only aggregated scores returned)
