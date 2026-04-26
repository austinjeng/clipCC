# ClipCC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Dockerized FastAPI API that scores video content against user-provided text labels using OpenCLIP ViT-L-14.

**Architecture:** Monolith FastAPI with ASGI middleware for pre-parse gates (auth, upload concurrency, body size), route-level inference limiter, and an InferenceRunner that owns the blocking pipeline thread lifecycle. All temp files managed by TempStore. Model config from baked metadata file.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, open_clip_torch, torch, Pillow, ffmpeg, anyio, pydantic, pytest, Docker

**Spec:** `docs/superpowers/specs/2026-04-26-clip-video-scoring-api-design.md`

---

## File Map

```
clipCC/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI app, startup, routes, pipeline orchestration
│   ├── config.py              # Settings from env vars (pydantic BaseSettings)
│   ├── middleware.py           # ASGI request gate (delegates to ResourceGates)
│   ├── resource_gates.py      # Upload + inference CapacityLimiters with context managers
│   ├── temp_store.py          # Temp file lifecycle, aggregate tracking, cleanup
│   ├── inference_runner.py    # Worker thread, cancel event, subprocess handle, timeout
│   ├── models/
│   │   ├── __init__.py
│   │   ├── clip_model.py      # OpenCLIP loading and inference
│   │   └── model_spec.py      # ModelSpec value object from /app/.baked_model
│   ├── services/
│   │   ├── __init__.py
│   │   ├── video.py           # FrameExtractor + FrameSample + ffprobe validation
│   │   └── scoring.py         # Aggregation (mean/max), confidence, raw_similarity
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── response.py        # Pydantic response/error models
│   └── errors/
│       ├── __init__.py
│       └── handlers.py         # Custom exceptions + friendly error formatting
└── tests/
    ├── __init__.py
    ├── conftest.py             # Shared fixtures (temp dirs, mock model, test videos)
    ├── test_config.py
    ├── test_temp_store.py
    ├── test_video.py
    ├── test_scoring.py
    ├── test_resource_gates.py
    ├── test_middleware.py
    ├── test_inference_runner.py
    └── test_api.py
```

---

## Task 1: Project Scaffold + Config + Errors

**Files:**
- Create: `requirements.txt`
- Create: `app/__init__.py`, `app/models/__init__.py`, `app/services/__init__.py`, `app/schemas/__init__.py`, `app/errors/__init__.py`
- Create: `app/config.py`
- Create: `app/models/model_spec.py`
- Create: `app/errors/handlers.py`
- Create: `app/schemas/response.py`
- Create: `tests/__init__.py`, `tests/conftest.py`, `tests/test_config.py`

- [ ] **Step 1: Create requirements.txt**

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
open-clip-torch>=2.26.0
torch>=2.2.0
Pillow>=10.2.0
pydantic>=2.6.0
pydantic-settings>=2.2.0
anyio>=4.3.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
```

- [ ] **Step 2: Create app/config.py**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    max_file_size_mb: int = 500
    max_duration_seconds: int = 300
    max_frames: int = 300
    default_fps: float = 1.0
    batch_size: int = 32
    max_concurrent_requests: int = 2
    max_upload_concurrency: int | None = None
    api_key: str | None = None
    allow_unauthenticated: bool = False
    ffmpeg_timeout_seconds: int = 120
    request_timeout_seconds: int = 300
    clip_cache_dir: str = "/app/models"
    temp_dir: str = "/tmp/clipcc"

    @property
    def effective_upload_concurrency(self) -> int:
        if self.max_upload_concurrency is not None:
            return self.max_upload_concurrency
        return self.max_concurrent_requests + 2

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    def validate_auth_config(self) -> None:
        if not self.api_key and not self.allow_unauthenticated:
            raise RuntimeError(
                "Server requires API_KEY to be set, or ALLOW_UNAUTHENTICATED=true "
                "for development use. Refusing to start without explicit auth config."
            )

    model_config = {"env_prefix": ""}
```

- [ ] **Step 3: Create app/models/model_spec.py**

```python
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    pretrained: str
    cache_dir: str

    @classmethod
    def from_baked_metadata(cls, path: Path = Path("/app/.baked_model")) -> "ModelSpec":
        data = json.loads(path.read_text())
        return cls(
            model_name=data["model_name"],
            pretrained=data["pretrained"],
            cache_dir=data["cache_dir"],
        )

    @classmethod
    def default(cls) -> "ModelSpec":
        return cls(
            model_name="ViT-L-14",
            pretrained="laion2b_s32b_b82k",
            cache_dir="/app/models",
        )

    def to_json(self) -> str:
        return json.dumps({
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "cache_dir": self.cache_dir,
        })
```

- [ ] **Step 4: Create app/errors/handlers.py**

```python
from fastapi import HTTPException


class FileTooLargeError(HTTPException):
    def __init__(self, size_mb: float, max_mb: int):
        super().__init__(
            status_code=413,
            detail=f"Your video is {size_mb:.0f}MB, which exceeds the {max_mb}MB limit. "
                   f"Try trimming or compressing it.",
        )


class DurationTooLongError(HTTPException):
    def __init__(self, duration_seconds: float, max_seconds: int):
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        max_min = max_seconds // 60
        super().__init__(
            status_code=422,
            detail=f"Video duration is {minutes}m{seconds:02d}s, which exceeds the "
                   f"{max_min}-minute limit. Please upload a shorter clip.",
        )


class TooManyFramesError(HTTPException):
    def __init__(self, frame_count: int, max_frames: int, duration: float, fps: float):
        super().__init__(
            status_code=422,
            detail=f"Your video would produce {frame_count:,} frames "
                   f"({duration:.0f}s at {fps}fps), which exceeds the {max_frames}-frame "
                   f"limit. Lower the fps or use a shorter clip.",
        )


class ResolutionTooHighError(HTTPException):
    def __init__(self, width: int, height: int):
        super().__init__(
            status_code=422,
            detail=f"Video resolution is {width}x{height}, which exceeds the "
                   f"3840x2160 limit. Please use a lower-resolution source.",
        )


class MultipleVideoStreamsError(HTTPException):
    def __init__(self, stream_count: int):
        super().__init__(
            status_code=422,
            detail=f"Video contains {stream_count} video streams. "
                   f"Only single-stream videos are supported.",
        )


class InvalidLabelsError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=422, detail=detail)


class InvalidPromptTemplateError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=422, detail=detail)


class InvalidFpsError(HTTPException):
    def __init__(self, fps: float):
        super().__init__(
            status_code=422,
            detail=f"FPS must be between 0.1 and 5.0. You provided {fps}.",
        )


class InvalidAggregationError(HTTPException):
    def __init__(self, aggregation: str):
        super().__init__(
            status_code=422,
            detail=f"Aggregation must be 'mean' or 'max'. You provided '{aggregation}'.",
        )


class UnsupportedFormatError(HTTPException):
    def __init__(self, extension: str):
        super().__init__(
            status_code=415,
            detail=f"Unsupported format '{extension}'. Supported: mp4, avi, mov, mkv.",
        )


class TokenTruncationError(HTTPException):
    def __init__(self, prompt: str, token_count: int):
        truncated = prompt[:50] + "..." if len(prompt) > 50 else prompt
        super().__init__(
            status_code=422,
            detail=f"Prompt '{truncated}' exceeds CLIP's 77-token context window "
                   f"(got {token_count} tokens). Shorten the label or template.",
        )


class DuplicateTokensError(HTTPException):
    def __init__(self, label_a: str, label_b: str):
        super().__init__(
            status_code=422,
            detail=f"Labels '{label_a}' and '{label_b}' produce identical token "
                   f"sequences. Use more distinct labels.",
        )


class InferenceTimeoutError(HTTPException):
    def __init__(self, timeout_seconds: int):
        super().__init__(
            status_code=504,
            detail=f"Processing exceeded the {timeout_seconds}-second time limit and "
                   f"was cancelled. Try a shorter video or lower fps.",
        )


class UploadConcurrencyError(Exception):
    pass


class InferenceConcurrencyError(Exception):
    pass
```

- [ ] **Step 5: Create app/schemas/response.py**

```python
from pydantic import BaseModel


class ScoreItem(BaseModel):
    label: str
    confidence: float
    raw_similarity: float
    peak_frame_index: int | None = None
    approx_timestamp_seconds: float | None = None


class BestMatch(BaseModel):
    label: str
    confidence: float


class ClassifyMetadata(BaseModel):
    frames_analyzed: int
    video_duration_seconds: float
    model: str
    device: str
    aggregation: str
    processing_time_seconds: float
    disclaimer: str


class ClassifyResponse(BaseModel):
    best_match: BestMatch
    scores: list[ScoreItem]
    metadata: ClassifyMetadata


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    model: str
    pretrained: str
    device: str


class ErrorResponse(BaseModel):
    detail: str
```

- [ ] **Step 6: Create test fixtures in tests/conftest.py**

```python
import os
from pathlib import Path

import pytest

from app.config import Settings


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def settings(temp_dir):
    return Settings(
        max_file_size_mb=10,
        max_duration_seconds=30,
        max_frames=30,
        default_fps=1.0,
        batch_size=4,
        max_concurrent_requests=2,
        allow_unauthenticated=True,
        ffmpeg_timeout_seconds=30,
        request_timeout_seconds=30,
        clip_cache_dir=str(temp_dir / "models"),
        temp_dir=str(temp_dir / "temp"),
    )


@pytest.fixture
def small_video(temp_dir):
    video_path = temp_dir / "test.mp4"
    os.system(
        f"ffmpeg -y -f lavfi -i testsrc=duration=3:size=320x240:rate=1 "
        f"-c:v libx264 -pix_fmt yuv420p {video_path} 2>/dev/null"
    )
    if not video_path.exists():
        pytest.skip("ffmpeg not available")
    return video_path
```

- [ ] **Step 7: Write config tests in tests/test_config.py**

```python
import pytest
from app.config import Settings


def test_default_settings():
    s = Settings(allow_unauthenticated=True)
    assert s.max_file_size_mb == 500
    assert s.max_frames == 300
    assert s.default_fps == 1.0
    assert s.effective_upload_concurrency == 4


def test_custom_upload_concurrency():
    s = Settings(allow_unauthenticated=True, max_upload_concurrency=10)
    assert s.effective_upload_concurrency == 10


def test_max_file_size_bytes():
    s = Settings(allow_unauthenticated=True, max_file_size_mb=100)
    assert s.max_file_size_bytes == 100 * 1024 * 1024


def test_validate_auth_config_fails_without_key_or_flag():
    s = Settings(allow_unauthenticated=False)
    with pytest.raises(RuntimeError, match="API_KEY"):
        s.validate_auth_config()


def test_validate_auth_config_passes_with_key():
    s = Settings(api_key="secret")
    s.validate_auth_config()


def test_validate_auth_config_passes_with_flag():
    s = Settings(allow_unauthenticated=True)
    s.validate_auth_config()
```

- [ ] **Step 8: Run tests**

Run: `cd /Users/austin/MITAC/clipCC && pip install pydantic-settings pytest httpx 2>/dev/null; python -m pytest tests/test_config.py -v`
Expected: All 6 tests PASS

- [ ] **Step 9: Commit**

```bash
git add requirements.txt app/ tests/
git commit -m "feat: project scaffold with config, errors, schemas, model spec"
```

---

## Task 2: TempStore Service

**Files:**
- Create: `app/temp_store.py`
- Create: `tests/test_temp_store.py`

- [ ] **Step 1: Write TempStore tests in tests/test_temp_store.py**

```python
import io
from pathlib import Path

import pytest

from app.temp_store import TempStore, StoredUpload


@pytest.fixture
def store(temp_dir):
    return TempStore(base_dir=temp_dir / "temp")


def test_save_upload_creates_file(store):
    content = b"fake video content" * 1000
    upload_file = io.BytesIO(content)
    result = store.save_upload("req-1", upload_file)
    assert isinstance(result, StoredUpload)
    assert result.path.exists()
    assert result.size == len(content)


def test_save_upload_streams_in_chunks(store):
    content = b"x" * (128 * 1024)
    upload_file = io.BytesIO(content)
    result = store.save_upload("req-1", upload_file, chunk_size=64 * 1024)
    assert result.path.read_bytes() == content


def test_create_frame_dir(store):
    frame_dir = store.create_frame_dir("req-1")
    assert frame_dir.is_dir()


def test_cleanup_removes_all_request_files(store):
    content = b"video"
    store.save_upload("req-1", io.BytesIO(content))
    frame_dir = store.create_frame_dir("req-1")
    (frame_dir / "frame_00001.jpg").write_bytes(b"jpeg")
    store.cleanup("req-1")
    assert not any(store.base_dir.glob("req-1*"))


def test_janitor_removes_old_files(store):
    import os
    import time
    old_dir = store.base_dir / "old-req"
    old_dir.mkdir(parents=True)
    (old_dir / "video.mp4").write_bytes(b"old")
    one_hour_ago = time.time() - 3601
    os.utime(old_dir, (one_hour_ago, one_hour_ago))
    store.run_janitor(max_age_seconds=3600)
    assert not old_dir.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_temp_store.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement TempStore in app/temp_store.py**

```python
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass
class StoredUpload:
    path: Path
    size: int


class TempStore:
    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(
        self, request_id: str, file: BinaryIO, chunk_size: int = 64 * 1024
    ) -> StoredUpload:
        request_dir = self.base_dir / request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        dest = request_dir / f"upload_{uuid.uuid4().hex[:8]}.tmp"
        total = 0
        with open(dest, "wb") as f:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
        return StoredUpload(path=dest, size=total)

    def create_frame_dir(self, request_id: str) -> Path:
        frame_dir = self.base_dir / request_id / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        return frame_dir

    def cleanup(self, request_id: str) -> None:
        request_dir = self.base_dir / request_id
        if request_dir.exists():
            shutil.rmtree(request_dir, ignore_errors=True)

    def run_janitor(self, max_age_seconds: int = 3600) -> None:
        if not self.base_dir.exists():
            return
        cutoff = time.time() - max_age_seconds
        for child in self.base_dir.iterdir():
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_temp_store.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/temp_store.py tests/test_temp_store.py
git commit -m "feat: TempStore service for temp file lifecycle"
```

---

## Task 3: Video Service (ffprobe + FrameExtractor)

**Files:**
- Create: `app/services/video.py`
- Create: `tests/test_video.py`

- [ ] **Step 1: Write tests in tests/test_video.py**

```python
import threading

import pytest

from app.services.video import (
    FrameExtractor,
    FrameSample,
    VideoInfo,
    probe_video,
    validate_video_constraints,
)


class TestProbeVideo:
    def test_probe_valid_video(self, small_video):
        info = probe_video(small_video, timeout=30)
        assert info.duration > 0
        assert info.width > 0
        assert info.height > 0
        assert info.video_stream_count == 1

    def test_probe_nonexistent_file(self, temp_dir):
        with pytest.raises(RuntimeError, match="ffprobe"):
            probe_video(temp_dir / "nonexistent.mp4", timeout=5)


class TestValidateVideoConstraints:
    def test_rejects_too_long(self, settings):
        info = VideoInfo(duration=999, width=320, height=240, video_stream_count=1, format_name="mov,mp4")
        with pytest.raises(Exception, match="minute"):
            validate_video_constraints(info, settings, fps=1.0)

    def test_rejects_too_many_frames(self, settings):
        info = VideoInfo(duration=29, width=320, height=240, video_stream_count=1, format_name="mov,mp4")
        with pytest.raises(Exception, match="frame"):
            validate_video_constraints(info, settings, fps=5.0)

    def test_rejects_high_resolution(self, settings):
        info = VideoInfo(duration=5, width=7680, height=4320, video_stream_count=1, format_name="mov,mp4")
        with pytest.raises(Exception, match="3840x2160"):
            validate_video_constraints(info, settings, fps=1.0)

    def test_rejects_multiple_streams(self, settings):
        info = VideoInfo(duration=5, width=320, height=240, video_stream_count=2, format_name="mov,mp4")
        with pytest.raises(Exception, match="video streams"):
            validate_video_constraints(info, settings, fps=1.0)

    def test_accepts_valid_video(self, settings):
        info = VideoInfo(duration=5, width=320, height=240, video_stream_count=1, format_name="mov,mp4")
        validate_video_constraints(info, settings, fps=1.0)


class TestFrameExtractor:
    def test_extract_frames(self, small_video, temp_dir):
        extractor = FrameExtractor(ffmpeg_timeout=30)
        frame_dir = temp_dir / "frames"
        frame_dir.mkdir()
        cancel = threading.Event()
        frames = extractor.extract(
            video_path=small_video, fps=1.0, max_frames=30,
            frame_dir=frame_dir, cancel_event=cancel,
        )
        assert len(frames) > 0
        assert all(isinstance(f, FrameSample) for f in frames)
        assert all(f.path.exists() for f in frames)
        assert frames[0].sample_index == 0
        assert frames[0].approx_timestamp_seconds == 0.0

    def test_extract_respects_cancel(self, small_video, temp_dir):
        extractor = FrameExtractor(ffmpeg_timeout=30)
        frame_dir = temp_dir / "frames"
        frame_dir.mkdir()
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(RuntimeError, match="cancel"):
            extractor.extract(
                video_path=small_video, fps=1.0, max_frames=30,
                frame_dir=frame_dir, cancel_event=cancel,
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_video.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement in app/services/video.py**

```python
import json
import math
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config import Settings
from app.errors.handlers import (
    DurationTooLongError,
    MultipleVideoStreamsError,
    ResolutionTooHighError,
    TooManyFramesError,
)


@dataclass
class VideoInfo:
    duration: float
    width: int
    height: int
    video_stream_count: int
    format_name: str


@dataclass
class FrameSample:
    path: Path
    sample_index: int
    approx_timestamp_seconds: float


def probe_video(video_path: Path, timeout: int = 30) -> VideoInfo:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v",
        "-show_entries", "stream=width,height",
        "-show_entries", "format=duration,format_name",
        "-of", "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffprobe timed out after {timeout}s") from e

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    if not streams:
        raise RuntimeError("No video streams found")

    duration = float(fmt.get("duration", 0))
    if duration <= 0 or not math.isfinite(duration):
        raise RuntimeError("Video has non-finite or non-positive duration")

    return VideoInfo(
        duration=duration,
        width=int(streams[0].get("width", 0)),
        height=int(streams[0].get("height", 0)),
        video_stream_count=len(streams),
        format_name=fmt.get("format_name", ""),
    )


def validate_video_constraints(info: VideoInfo, settings: Settings, fps: float) -> None:
    if info.duration > settings.max_duration_seconds:
        raise DurationTooLongError(info.duration, settings.max_duration_seconds)

    expected_frames = int(info.duration * fps)
    if expected_frames > settings.max_frames:
        raise TooManyFramesError(expected_frames, settings.max_frames, info.duration, fps)

    if info.width > 3840 or info.height > 2160:
        raise ResolutionTooHighError(info.width, info.height)

    if info.width * info.height > 8_300_000:
        raise ResolutionTooHighError(info.width, info.height)

    if info.video_stream_count > 1:
        raise MultipleVideoStreamsError(info.video_stream_count)


class FrameExtractor:
    def __init__(self, ffmpeg_timeout: int = 120):
        self.ffmpeg_timeout = ffmpeg_timeout
        self.active_process: Optional[subprocess.Popen] = None

    def extract(
        self,
        video_path: Path,
        fps: float,
        max_frames: int,
        frame_dir: Path,
        cancel_event: threading.Event,
    ) -> list[FrameSample]:
        if cancel_event.is_set():
            raise RuntimeError("Extraction cancelled before start")

        output_pattern = str(frame_dir / "frame_%05d.jpg")
        vf = (
            f"fps={fps},"
            f"scale='min(512,iw)':'min(512,ih)':force_original_aspect_ratio=decrease"
        )
        cmd = [
            "ffmpeg", "-nostdin", "-v", "error",
            "-i", str(video_path),
            "-vf", vf,
            "-q:v", "2",
            "-frames:v", str(max_frames),
            output_pattern,
        ]

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.active_process = proc
            _, stderr = proc.communicate(timeout=self.ffmpeg_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise RuntimeError(f"ffmpeg timed out after {self.ffmpeg_timeout}s")
        finally:
            self.active_process = None

        if cancel_event.is_set():
            raise RuntimeError("Extraction cancelled")

        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode().strip()}")

        frame_files = sorted(frame_dir.glob("frame_*.jpg"))
        return [
            FrameSample(
                path=f,
                sample_index=i,
                approx_timestamp_seconds=i / fps,
            )
            for i, f in enumerate(frame_files)
        ]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_video.py -v`
Expected: All tests PASS (skip if ffmpeg unavailable)

- [ ] **Step 5: Commit**

```bash
git add app/services/video.py tests/test_video.py
git commit -m "feat: video probing, validation, and frame extraction"
```

---

## Task 4: CLIP Model Wrapper

**Files:**
- Create: `app/models/clip_model.py`
- Create: `tests/test_clip_model.py`

- [ ] **Step 1: Write tests in tests/test_clip_model.py**

```python
import pytest
import torch
from PIL import Image

from app.models.clip_model import ClipModel
from app.models.model_spec import ModelSpec


@pytest.fixture
def dummy_images():
    return [Image.new("RGB", (224, 224), color=(i * 30, 100, 200)) for i in range(3)]


@pytest.fixture
def model(temp_dir):
    spec = ModelSpec(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        cache_dir=str(temp_dir / "models"),
    )
    return ClipModel(spec)


def test_load_model(model):
    assert model.model is not None
    assert model.preprocess is not None
    assert model.tokenizer is not None
    assert model.device in ("cpu", "cuda")


def test_encode_text(model):
    features = model.encode_text(["a photo of a cat", "a photo of a dog"])
    assert features.shape[0] == 2
    assert features.shape[1] > 0


def test_encode_images(model, dummy_images):
    features = model.encode_images(dummy_images)
    assert features.shape[0] == 3


def test_compute_similarities(model, dummy_images):
    texts = ["red image", "green image", "blue image"]
    similarities, logit_scale = model.compute_similarities(dummy_images, texts)
    assert similarities.shape == (3, 3)
    assert logit_scale > 0


def test_tokenize_and_check(model):
    prompts = ["a video of driving", "a video of parking"]
    token_counts = model.tokenize_and_check(prompts, max_tokens=77)
    assert len(token_counts) == 2
    assert all(0 < c <= 77 for c in token_counts)


def test_tokenize_detects_long_prompt(model):
    long_prompt = "a video of " + "very " * 100 + "long description"
    token_counts = model.tokenize_and_check([long_prompt], max_tokens=77)
    assert token_counts[0] > 77


def test_tokenize_raw(model):
    prompts = ["a video of driving", "a video of parking"]
    result = model.tokenize_raw(prompts)
    assert len(result) == 2
    assert all(isinstance(t, torch.Tensor) for t in result)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_clip_model.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement in app/models/clip_model.py**

```python
import open_clip
import torch
from PIL import Image

from app.models.model_spec import ModelSpec


class ClipModel:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            spec.model_name,
            pretrained=spec.pretrained,
            cache_dir=spec.cache_dir,
        )
        self.model = self.model.to(self.device)
        # Set to inference mode: disables dropout and batchnorm training behavior
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(spec.model_name)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda"):
                    return self.model.encode_text(tokens, normalize=True)
            return self.model.encode_text(tokens, normalize=True)

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda"):
                    return self.model.encode_image(batch, normalize=True)
            return self.model.encode_image(batch, normalize=True)

    def compute_similarities(
        self, images: list[Image.Image], texts: list[str]
    ) -> tuple[torch.Tensor, float]:
        text_features = self.encode_text(texts)
        image_features = self.encode_images(images)
        logit_scale = self.model.logit_scale.exp().item()
        cosine_sim = image_features @ text_features.T
        return cosine_sim, logit_scale

    def tokenize_and_check(self, prompts: list[str], max_tokens: int = 77) -> list[int]:
        counts = []
        for prompt in prompts:
            tokens = self.tokenizer([prompt])[0]
            nonzero = (tokens != 0).sum().item()
            counts.append(nonzero)
        return counts

    def tokenize_raw(self, prompts: list[str]) -> list[torch.Tensor]:
        return [self.tokenizer([p])[0] for p in prompts]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_clip_model.py -v`
Expected: All tests PASS (first run downloads ViT-B-32 ~400MB)

- [ ] **Step 5: Commit**

```bash
git add app/models/clip_model.py tests/test_clip_model.py
git commit -m "feat: ClipModel wrapper with text/image encoding and token checking"
```

---

## Task 5: Scoring Service

**Files:**
- Create: `app/services/scoring.py`
- Create: `tests/test_scoring.py`

- [ ] **Step 1: Write tests in tests/test_scoring.py**

```python
import torch
import pytest
from pathlib import Path

from app.services.scoring import (
    compute_frame_scores,
    aggregate_mean,
    aggregate_max,
    build_response_scores,
)
from app.services.video import FrameSample


def make_frame(index: int, fps: float = 1.0) -> FrameSample:
    return FrameSample(
        path=Path(f"/tmp/frame_{index:05d}.jpg"),
        sample_index=index,
        approx_timestamp_seconds=index / fps,
    )


def test_compute_frame_scores_shape():
    cosine = torch.tensor([[0.25, 0.30], [0.28, 0.32]])
    conf, raw = compute_frame_scores(cosine, logit_scale=100.0)
    assert conf.shape == (2, 2)
    assert raw.shape == (2, 2)


def test_confidence_sums_to_one():
    cosine = torch.tensor([[0.25, 0.30, 0.20]])
    conf, _ = compute_frame_scores(cosine, logit_scale=100.0)
    assert abs(conf[0].sum().item() - 1.0) < 1e-5


def test_raw_similarity_is_unscaled():
    cosine = torch.tensor([[0.25, 0.30]])
    _, raw = compute_frame_scores(cosine, logit_scale=100.0)
    assert torch.allclose(raw, cosine, atol=1e-6)


def test_aggregate_mean():
    conf = torch.tensor([[0.3, 0.7], [0.4, 0.6]])
    raw = torch.tensor([[0.25, 0.30], [0.27, 0.28]])
    labels = ["driving", "parking"]
    frames = [make_frame(0), make_frame(1)]
    result = aggregate_mean(conf, raw, labels, frames)
    assert len(result) == 2
    assert abs(result[0].confidence - 0.35) < 1e-5
    assert result[0].peak_frame_index is None


def test_aggregate_max():
    conf = torch.tensor([[0.3, 0.7], [0.8, 0.2]])
    raw = torch.tensor([[0.25, 0.30], [0.31, 0.22]])
    labels = ["driving", "parking"]
    frames = [make_frame(0), make_frame(1)]
    result = aggregate_max(conf, raw, labels, frames)
    assert len(result) == 2
    assert abs(result[0].confidence - 0.8) < 1e-5
    assert result[0].peak_frame_index == 1
    assert abs(result[1].confidence - 0.7) < 1e-5
    assert result[1].peak_frame_index == 0


def test_build_response_mean():
    conf = torch.tensor([[0.3, 0.7], [0.4, 0.6]])
    raw = torch.tensor([[0.25, 0.30], [0.27, 0.28]])
    frames = [make_frame(0), make_frame(1)]
    scores, best = build_response_scores(conf, raw, ["a", "b"], frames, "mean")
    assert best.label == "b"
    assert len(scores) == 2


def test_build_response_max():
    conf = torch.tensor([[0.3, 0.7], [0.8, 0.2]])
    raw = torch.tensor([[0.25, 0.30], [0.31, 0.22]])
    frames = [make_frame(0), make_frame(1)]
    scores, best = build_response_scores(conf, raw, ["a", "b"], frames, "max")
    assert best.label == "a"
    assert scores[0].peak_frame_index == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement in app/services/scoring.py**

```python
import torch

from app.schemas.response import BestMatch, ScoreItem
from app.services.video import FrameSample


def compute_frame_scores(
    cosine_sim: torch.Tensor, logit_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_similarity = cosine_sim.clone()
    scaled_logits = cosine_sim * logit_scale
    confidence = torch.softmax(scaled_logits, dim=-1)
    return confidence, raw_similarity


def aggregate_mean(
    confidence: torch.Tensor,
    raw_sim: torch.Tensor,
    labels: list[str],
    frames: list[FrameSample],
) -> list[ScoreItem]:
    mean_conf = confidence.mean(dim=0)
    mean_raw = raw_sim.mean(dim=0)
    return [
        ScoreItem(
            label=labels[i],
            confidence=round(mean_conf[i].item(), 6),
            raw_similarity=round(mean_raw[i].item(), 6),
        )
        for i in range(len(labels))
    ]


def aggregate_max(
    confidence: torch.Tensor,
    raw_sim: torch.Tensor,
    labels: list[str],
    frames: list[FrameSample],
) -> list[ScoreItem]:
    max_conf, max_indices = confidence.max(dim=0)
    return [
        ScoreItem(
            label=labels[i],
            confidence=round(max_conf[i].item(), 6),
            raw_similarity=round(raw_sim[max_indices[i].item(), i].item(), 6),
            peak_frame_index=max_indices[i].item(),
            approx_timestamp_seconds=frames[max_indices[i].item()].approx_timestamp_seconds,
        )
        for i in range(len(labels))
    ]


def build_response_scores(
    confidence: torch.Tensor,
    raw_sim: torch.Tensor,
    labels: list[str],
    frames: list[FrameSample],
    aggregation: str,
) -> tuple[list[ScoreItem], BestMatch]:
    if aggregation == "max":
        scores = aggregate_max(confidence, raw_sim, labels, frames)
    else:
        scores = aggregate_mean(confidence, raw_sim, labels, frames)
    best = max(scores, key=lambda s: s.confidence)
    best_match = BestMatch(label=best.label, confidence=best.confidence)
    return scores, best_match
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_scoring.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/scoring.py tests/test_scoring.py
git commit -m "feat: scoring service with mean/max aggregation and logit scaling"
```

---

## Task 6: ResourceGates

**Files:**
- Create: `app/resource_gates.py`
- Create: `tests/test_resource_gates.py`

- [ ] **Step 1: Write tests in tests/test_resource_gates.py**

```python
import pytest
import anyio

from app.resource_gates import ResourceGates
from app.errors.handlers import UploadConcurrencyError, InferenceConcurrencyError


@pytest.fixture
def gates():
    return ResourceGates(max_upload_concurrency=2, max_inference_concurrency=1)


@pytest.mark.anyio
async def test_upload_admission_allows_within_limit(gates):
    async with gates.upload_admission():
        pass


@pytest.mark.anyio
async def test_upload_admission_rejects_over_limit(gates):
    async with gates.upload_admission():
        async with gates.upload_admission():
            with pytest.raises(UploadConcurrencyError):
                async with gates.upload_admission():
                    pass


@pytest.mark.anyio
async def test_inference_admission_allows_within_limit(gates):
    async with gates.inference_admission():
        pass


@pytest.mark.anyio
async def test_inference_admission_rejects_over_limit(gates):
    async with gates.inference_admission():
        with pytest.raises(InferenceConcurrencyError):
            async with gates.inference_admission():
                pass


@pytest.mark.anyio
async def test_upload_slot_released_after_exception(gates):
    with pytest.raises(ValueError):
        async with gates.upload_admission():
            raise ValueError("boom")
    async with gates.upload_admission():
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_resource_gates.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement in app/resource_gates.py**

```python
from contextlib import asynccontextmanager

import anyio

from app.errors.handlers import InferenceConcurrencyError, UploadConcurrencyError


class ResourceGates:
    def __init__(
        self,
        max_upload_concurrency: int = 4,
        max_inference_concurrency: int = 2,
    ):
        self._upload_limiter = anyio.CapacityLimiter(max_upload_concurrency)
        self._inference_limiter = anyio.CapacityLimiter(max_inference_concurrency)

    @asynccontextmanager
    async def upload_admission(self):
        token = object()
        try:
            self._upload_limiter.acquire_nowait_on_behalf_of(token)
        except anyio.WouldBlock:
            raise UploadConcurrencyError()
        try:
            yield
        finally:
            self._upload_limiter.release_on_behalf_of(token)

    @asynccontextmanager
    async def inference_admission(self):
        token = object()
        try:
            self._inference_limiter.acquire_nowait_on_behalf_of(token)
        except anyio.WouldBlock:
            raise InferenceConcurrencyError()
        try:
            yield
        finally:
            self._inference_limiter.release_on_behalf_of(token)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_resource_gates.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/resource_gates.py tests/test_resource_gates.py
git commit -m "feat: ResourceGates with non-blocking upload and inference limiters"
```

---

## Task 7: ASGI Middleware

**Files:**
- Create: `app/middleware.py`
- Create: `tests/test_middleware.py`

- [ ] **Step 1: Write tests in tests/test_middleware.py**

```python
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from app.middleware import RequestGateMiddleware
from app.resource_gates import ResourceGates


async def echo_app(request):
    body = await request.body()
    return JSONResponse({"size": len(body)})


async def health_app(request):
    return JSONResponse({"status": "ok"})


def make_test_app(api_key=None, max_body=1024, max_upload=2):
    gates = ResourceGates(max_upload_concurrency=max_upload, max_inference_concurrency=1)
    app = Starlette(routes=[
        Route("/api/v1/classify", echo_app, methods=["POST"]),
        Route("/live", health_app, methods=["GET"]),
        Route("/ready", health_app, methods=["GET"]),
    ])
    return RequestGateMiddleware(
        app, gates=gates, api_key=api_key, max_body_bytes=max_body
    )


@pytest.mark.anyio
async def test_live_bypasses_all_gates():
    app = make_test_app(api_key="secret")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/live")
        assert r.status_code == 200


@pytest.mark.anyio
async def test_ready_bypasses_concurrency_but_checks_auth():
    app = make_test_app(api_key="secret")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/ready")
        assert r.status_code == 401
        r = await c.get("/ready", headers={"X-API-Key": "secret"})
        assert r.status_code == 200


@pytest.mark.anyio
async def test_auth_rejects_missing_key():
    app = make_test_app(api_key="secret")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/classify", content=b"data")
        assert r.status_code == 401


@pytest.mark.anyio
async def test_auth_accepts_valid_key():
    app = make_test_app(api_key="secret")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/classify", content=b"data", headers={"X-API-Key": "secret"})
        assert r.status_code == 200


@pytest.mark.anyio
async def test_no_auth_when_key_not_configured():
    app = make_test_app(api_key=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/classify", content=b"data")
        assert r.status_code == 200


@pytest.mark.anyio
async def test_body_size_rejection():
    app = make_test_app(max_body=100)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/classify", content=b"x" * 200)
        assert r.status_code == 413
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_middleware.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement in app/middleware.py**

```python
import json
from typing import Optional

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.resource_gates import ResourceGates
from app.errors.handlers import UploadConcurrencyError


class RequestGateMiddleware:
    EXEMPT_PATHS = {"/live"}
    CONCURRENCY_EXEMPT_PATHS = {"/live", "/ready"}
    GATED_PATHS = {"/api/v1/classify"}

    def __init__(
        self,
        app: ASGIApp,
        gates: ResourceGates,
        api_key: Optional[str] = None,
        max_body_bytes: int = 500 * 1024 * 1024,
    ):
        self.app = app
        self.gates = gates
        self.api_key = api_key
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        if path in self.EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        if path in self.CONCURRENCY_EXEMPT_PATHS:
            if self.api_key and not self._check_auth(scope):
                await self._send_error(send, 401, "Invalid or missing API key. Provide a valid key in the X-API-Key header.")
                return
            await self.app(scope, receive, send)
            return

        if path not in self.GATED_PATHS:
            await self.app(scope, receive, send)
            return

        if self.api_key and not self._check_auth(scope):
            await self._send_error(send, 401, "Invalid or missing API key. Provide a valid key in the X-API-Key header.")
            return

        try:
            async with self.gates.upload_admission():
                checker = _SizeChecker(receive, self.max_body_bytes)
                try:
                    await self.app(scope, checker.receive, send)
                except _BodyTooLargeSignal:
                    size_mb = checker.total / (1024 * 1024)
                    max_mb = self.max_body_bytes / (1024 * 1024)
                    await self._send_error(
                        send, 413,
                        f"Your video is {size_mb:.0f}MB, which exceeds the "
                        f"{max_mb:.0f}MB limit. Try trimming or compressing it.",
                    )
        except UploadConcurrencyError:
            await self._send_error(send, 429, "Too many uploads in progress. Please retry in a moment.")

    def _check_auth(self, scope: Scope) -> bool:
        headers = dict(scope.get("headers", []))
        key = headers.get(b"x-api-key", b"").decode()
        return key == self.api_key

    async def _send_error(self, send: Send, status: int, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode()
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(body)).encode()],
            ],
        })
        await send({"type": "http.response.body", "body": body})


class _SizeChecker:
    def __init__(self, original_receive: Receive, max_bytes: int):
        self._receive = original_receive
        self._max_bytes = max_bytes
        self.total = 0

    async def receive(self) -> Message:
        message = await self._receive()
        if message.get("type") == "http.request":
            self.total += len(message.get("body", b""))
            if self.total > self._max_bytes:
                raise _BodyTooLargeSignal()
        return message


class _BodyTooLargeSignal(Exception):
    pass
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_middleware.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/middleware.py tests/test_middleware.py
git commit -m "feat: ASGI middleware with auth, upload concurrency, and body size gates"
```

---

## Task 8: InferenceRunner

**Files:**
- Create: `app/inference_runner.py`
- Create: `tests/test_inference_runner.py`

- [ ] **Step 1: Write tests in tests/test_inference_runner.py**

```python
import time
import pytest

from app.inference_runner import InferenceRunner


def slow_pipeline(cancel_event, runner_ref) -> str:
    for _ in range(50):
        if cancel_event.is_set():
            return "cancelled"
        time.sleep(0.05)
    return "done"


def fast_pipeline(cancel_event, runner_ref) -> str:
    return "done"


def failing_pipeline(cancel_event, runner_ref) -> str:
    raise ValueError("pipeline error")


@pytest.mark.anyio
async def test_successful_run():
    runner = InferenceRunner(timeout_seconds=10)
    result = await runner.run(fast_pipeline)
    assert result == "done"


@pytest.mark.anyio
async def test_timeout_cancels_and_returns_none():
    runner = InferenceRunner(timeout_seconds=0.3)
    result = await runner.run(slow_pipeline)
    assert result is None
    assert runner.timed_out


@pytest.mark.anyio
async def test_pipeline_error_propagates():
    runner = InferenceRunner(timeout_seconds=10)
    with pytest.raises(ValueError, match="pipeline error"):
        await runner.run(failing_pipeline)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_inference_runner.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement in app/inference_runner.py**

```python
import asyncio
import subprocess
import threading
from typing import Any, Callable, Optional

import anyio


class InferenceRunner:
    def __init__(self, timeout_seconds: float = 300):
        self.timeout_seconds = timeout_seconds
        self.cancel_event = threading.Event()
        self.active_process: Optional[subprocess.Popen] = None
        self.timed_out = False
        self._lock = threading.Lock()

    def register_process(self, proc: Optional[subprocess.Popen]) -> None:
        with self._lock:
            self.active_process = proc

    def unregister_process(self) -> None:
        with self._lock:
            self.active_process = None

    def _kill_active_process(self) -> None:
        with self._lock:
            if self.active_process is not None:
                try:
                    self.active_process.kill()
                except OSError:
                    pass

    async def run(
        self,
        pipeline: Callable[[threading.Event, "InferenceRunner"], Any],
    ) -> Optional[Any]:
        result_holder: dict[str, Any] = {}
        error_holder: dict[str, BaseException] = {}
        done = asyncio.Event()
        loop = asyncio.get_running_loop()

        def _worker():
            try:
                result_holder["value"] = pipeline(self.cancel_event, self)
            except BaseException as e:
                error_holder["value"] = e
            finally:
                loop.call_soon_threadsafe(done.set)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        try:
            with anyio.fail_after(self.timeout_seconds):
                await done.wait()
        except TimeoutError:
            self.timed_out = True
            self.cancel_event.set()
            self._kill_active_process()
            thread.join(timeout=self.timeout_seconds)
            return None

        thread.join(timeout=5)

        if "value" in error_holder:
            raise error_holder["value"]

        return result_holder.get("value")
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_inference_runner.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/inference_runner.py tests/test_inference_runner.py
git commit -m "feat: InferenceRunner with cooperative timeout and subprocess kill"
```

---

## Task 9: Main App — Startup, Health, and Classify Route

**Files:**
- Create: `app/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write tests in tests/test_api.py**

```python
import json
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.config import Settings


@pytest.fixture
def test_settings(temp_dir):
    return Settings(
        max_file_size_mb=10,
        max_duration_seconds=30,
        max_frames=30,
        default_fps=1.0,
        batch_size=4,
        max_concurrent_requests=2,
        allow_unauthenticated=True,
        ffmpeg_timeout_seconds=30,
        request_timeout_seconds=30,
        clip_cache_dir=str(temp_dir / "models"),
        temp_dir=str(temp_dir / "temp"),
    )


@pytest.fixture
async def client(test_settings):
    app = create_app(test_settings)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_live(client):
    r = await client.get("/live")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.anyio
async def test_ready(client):
    r = await client.get("/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ready"
    assert "model" in data


@pytest.mark.anyio
async def test_missing_video(client):
    r = await client.post("/api/v1/classify", data={"labels": json.dumps(["a", "b", "c"])})
    assert r.status_code == 422


@pytest.mark.anyio
async def test_invalid_labels_count(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={"labels": json.dumps(["a"])},
    )
    assert r.status_code == 422
    assert "3 and 10" in r.json()["detail"]


@pytest.mark.anyio
async def test_invalid_fps(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={"labels": json.dumps(["a", "b", "c"]), "fps": "10.0"},
    )
    assert r.status_code == 422
    assert "FPS" in r.json()["detail"]


@pytest.mark.anyio
async def test_unsupported_format(client):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.webm", b"fake", "video/webm")},
        data={"labels": json.dumps(["a", "b", "c"])},
    )
    assert r.status_code == 415


@pytest.mark.anyio
async def test_classify_mean(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={"labels": json.dumps(["outdoor scene", "indoor scene", "vehicle"])},
    )
    assert r.status_code == 200
    data = r.json()
    assert "best_match" in data
    assert len(data["scores"]) == 3
    assert data["metadata"]["aggregation"] == "mean"
    total = sum(s["confidence"] for s in data["scores"])
    assert abs(total - 1.0) < 0.01


@pytest.mark.anyio
async def test_classify_max(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={
            "labels": json.dumps(["outdoor scene", "indoor scene", "vehicle"]),
            "aggregation": "max",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["metadata"]["aggregation"] == "max"
    for score in data["scores"]:
        assert "peak_frame_index" in score
        assert "approx_timestamp_seconds" in score
```

- [ ] **Step 2: Implement in app/main.py**

```python
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from app.config import Settings
from app.errors.handlers import (
    InferenceConcurrencyError,
    InferenceTimeoutError,
    InvalidAggregationError,
    InvalidFpsError,
    InvalidLabelsError,
    InvalidPromptTemplateError,
    TokenTruncationError,
    DuplicateTokensError,
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

logger = logging.getLogger("clipcc")

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MEAN_DISCLAIMER = (
    "Scores are relative to the supplied labels, not calibrated probabilities. "
    "Not suitable for safety-critical decisions."
)
MAX_DISCLAIMER = (
    "Scores are relative to the supplied labels, not calibrated probabilities. "
    "Max-mode scores are independent peaks per label and do not sum to 1. "
    "Not suitable for safety-critical decisions."
)


def create_app(settings: Optional[Settings] = None) -> RequestGateMiddleware:
    if settings is None:
        settings = Settings()
    settings.validate_auth_config()

    clip_model: Optional[ClipModel] = None
    model_spec: Optional[ModelSpec] = None
    gates = ResourceGates(
        max_upload_concurrency=settings.effective_upload_concurrency,
        max_inference_concurrency=settings.max_concurrent_requests,
    )
    temp_store = TempStore(base_dir=settings.temp_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal clip_model, model_spec
        baked_path = Path("/app/.baked_model")
        if baked_path.exists():
            model_spec = ModelSpec.from_baked_metadata(baked_path)
        else:
            model_spec = ModelSpec(
                model_name="ViT-B-32",
                pretrained="laion2b_s34b_b79k",
                cache_dir=settings.clip_cache_dir,
            )
        logger.info("Loading model: %s / %s", model_spec.model_name, model_spec.pretrained)
        clip_model = ClipModel(model_spec)
        logger.info("Model loaded on device: %s", clip_model.device)
        temp_store.run_janitor()
        yield

    app = FastAPI(title="ClipCC", lifespan=lifespan)

    @app.get("/live", response_model=HealthResponse)
    async def live():
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse)
    async def ready():
        if clip_model is None or model_spec is None:
            return JSONResponse(status_code=503, content={"status": "loading"})
        return ReadyResponse(
            status="ready",
            model=model_spec.model_name,
            pretrained=model_spec.pretrained,
            device=clip_model.device,
        )

    @app.post("/api/v1/classify", response_model=ClassifyResponse)
    async def classify(
        video: UploadFile = File(...),
        labels: str = Form(...),
        prompt_template: str = Form("a video of {}"),
        fps: float = Form(1.0),
        aggregation: str = Form("mean"),
    ):
        request_id = uuid.uuid4().hex[:12]
        ext = Path(video.filename or "").suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(ext)
        if fps < 0.1 or fps > 5.0:
            raise InvalidFpsError(fps)
        if aggregation not in ("mean", "max"):
            raise InvalidAggregationError(aggregation)

        try:
            label_list = json.loads(labels)
        except json.JSONDecodeError:
            raise InvalidLabelsError("Labels must be a valid JSON array of strings.")
        if not isinstance(label_list, list) or not all(isinstance(l, str) for l in label_list):
            raise InvalidLabelsError("Labels must be a JSON array of strings.")
        if len(label_list) < 3 or len(label_list) > 10:
            raise InvalidLabelsError(
                f"Please provide between 3 and 10 labels. You provided {len(label_list)}."
            )
        for label in label_list:
            if not label.strip() or len(label) > 200:
                raise InvalidLabelsError(
                    "Labels must be non-empty, unique, and under 200 characters each."
                )
        if len(set(label_list)) != len(label_list):
            raise InvalidLabelsError(
                "Labels must be non-empty, unique, and under 200 characters each."
            )

        if "{}" not in prompt_template or prompt_template.count("{}") > 1:
            raise InvalidPromptTemplateError(
                "Prompt template must contain exactly one '{}' placeholder."
            )
        if len(prompt_template) > 500:
            raise InvalidPromptTemplateError("Prompt template must be under 500 characters.")

        prompts = [prompt_template.replace("{}", label) for label in label_list]

        token_counts = clip_model.tokenize_and_check(prompts, max_tokens=77)
        for i, count in enumerate(token_counts):
            if count > 77:
                raise TokenTruncationError(prompts[i], count)

        token_seqs = clip_model.tokenize_raw(prompts)
        seen: dict[tuple, int] = {}
        for i, seq in enumerate(token_seqs):
            key = tuple(seq.tolist())
            if key in seen:
                raise DuplicateTokensError(label_list[seen[key]], label_list[i])
            seen[key] = i

        stored = temp_store.save_upload(request_id, video.file)
        try:
            video_info = probe_video(stored.path, timeout=settings.ffmpeg_timeout_seconds)
            validate_video_constraints(video_info, settings, fps)

            try:
                async with gates.inference_admission():
                    runner = InferenceRunner(timeout_seconds=settings.request_timeout_seconds)
                    start_time = time.monotonic()

                    def pipeline(cancel_event, runner_ref):
                        frame_dir = temp_store.create_frame_dir(request_id)
                        extractor = FrameExtractor(ffmpeg_timeout=settings.ffmpeg_timeout_seconds)
                        frames = extractor.extract(
                            video_path=stored.path, fps=fps,
                            max_frames=settings.max_frames,
                            frame_dir=frame_dir, cancel_event=cancel_event,
                        )

                        import torch
                        from PIL import Image
                        all_conf, all_raw = [], []
                        for batch_start in range(0, len(frames), settings.batch_size):
                            if cancel_event.is_set():
                                raise RuntimeError("Inference cancelled")
                            batch = frames[batch_start:batch_start + settings.batch_size]
                            images = [Image.open(f.path) for f in batch]
                            cosine_sim, logit_scale = clip_model.compute_similarities(images, prompts)
                            conf, raw = compute_frame_scores(cosine_sim, logit_scale)
                            all_conf.append(conf)
                            all_raw.append(raw)
                            for f in batch:
                                f.path.unlink(missing_ok=True)
                            for img in images:
                                img.close()

                        return torch.cat(all_conf, dim=0), torch.cat(all_raw, dim=0), frames

                    result = await runner.run(pipeline)
                    if result is None:
                        raise InferenceTimeoutError(settings.request_timeout_seconds)

                    full_conf, full_raw, all_frames = result
                    elapsed = time.monotonic() - start_time
                    scores, best_match = build_response_scores(
                        full_conf, full_raw, label_list, all_frames, aggregation
                    )
                    disclaimer = MAX_DISCLAIMER if aggregation == "max" else MEAN_DISCLAIMER

                    return ClassifyResponse(
                        best_match=best_match,
                        scores=scores,
                        metadata=ClassifyMetadata(
                            frames_analyzed=full_conf.shape[0],
                            video_duration_seconds=round(video_info.duration, 1),
                            model=model_spec.model_name,
                            device=clip_model.device,
                            aggregation=aggregation,
                            processing_time_seconds=round(elapsed, 1),
                            disclaimer=disclaimer,
                        ),
                    )
            except InferenceConcurrencyError:
                return JSONResponse(
                    status_code=429,
                    content={"detail": f"Server is processing the maximum number of videos ({settings.max_concurrent_requests}). Please retry in a moment."},
                )
        finally:
            temp_store.cleanup(request_id)

    return RequestGateMiddleware(
        app, gates=gates,
        api_key=settings.api_key,
        max_body_bytes=settings.max_file_size_bytes,
    )
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_api.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: main app with classify route, health endpoints, and full pipeline"
```

---

## Task 10: Docker

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.env.example`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

ARG TORCH_VARIANT=cpu
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ app/

ARG MODEL_NAME=ViT-L-14
ARG PRETRAINED=laion2b_s32b_b82k

ENV CLIP_CACHE_DIR=/app/models

RUN python -c "import json, open_clip; \
open_clip.create_model_and_transforms('${MODEL_NAME}', pretrained='${PRETRAINED}', cache_dir='/app/models'); \
json.dump({'model_name': '${MODEL_NAME}', 'pretrained': '${PRETRAINED}', 'cache_dir': '/app/models'}, open('/app/.baked_model', 'w'))"

EXPOSE 8000

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

- [ ] **Step 2: Create docker-compose.yml**

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
      - ALLOW_UNAUTHENTICATED=true
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
      - MAX_CONCURRENT_REQUESTS=1
      - ALLOW_UNAUTHENTICATED=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    profiles: ["gpu"]
```

- [ ] **Step 3: Create .env.example**

```bash
# ClipCC Configuration

# Upload and processing limits
MAX_FILE_SIZE_MB=500
MAX_DURATION_SECONDS=300
MAX_FRAMES=300
DEFAULT_FPS=1.0
BATCH_SIZE=32

# Concurrency
MAX_CONCURRENT_REQUESTS=2

# Authentication (fail-closed: one of these must be set)
# API_KEY=your-secret-key-here
ALLOW_UNAUTHENTICATED=true

# Timeouts
FFMPEG_TIMEOUT_SECONDS=120
REQUEST_TIMEOUT_SECONDS=300

# Model cache (must match Dockerfile build)
CLIP_CACHE_DIR=/app/models
```

- [ ] **Step 4: Build CPU image**

Run: `docker compose --profile cpu build`
Expected: Successful build (several minutes for model download)

- [ ] **Step 5: Smoke test**

Run: `docker compose --profile cpu up -d && sleep 30 && curl -s http://localhost:8000/live && docker compose --profile cpu down`
Expected: `{"status":"ok"}`

- [ ] **Step 6: Commit**

```bash
git add Dockerfile docker-compose.yml .env.example
git commit -m "feat: Docker setup with CPU/GPU profiles and baked model weights"
```

---

## Task 11: End-to-End Smoke Test

- [ ] **Step 1: Start container**

Run: `docker compose --profile cpu up -d`

- [ ] **Step 2: Create test video**

Run: `ffmpeg -y -f lavfi -i testsrc=duration=5:size=320x240:rate=10 -c:v libx264 -pix_fmt yuv420p /tmp/test_clipcc.mp4`

- [ ] **Step 3: Test classify (mean)**

Run:
```bash
curl -s -X POST http://localhost:8000/api/v1/classify \
  -F "video=@/tmp/test_clipcc.mp4" \
  -F 'labels=["test pattern","outdoor scene","person walking"]' \
  -F "fps=1.0" | python3 -m json.tool
```
Expected: 200 with best_match, 3 scores summing to ~1.0, metadata with aggregation=mean

- [ ] **Step 4: Test classify (max)**

Run:
```bash
curl -s -X POST http://localhost:8000/api/v1/classify \
  -F "video=@/tmp/test_clipcc.mp4" \
  -F 'labels=["test pattern","outdoor scene","person walking"]' \
  -F "aggregation=max" | python3 -m json.tool
```
Expected: 200 with peak_frame_index and approx_timestamp_seconds in each score

- [ ] **Step 5: Test error handling**

Run:
```bash
curl -s -X POST http://localhost:8000/api/v1/classify \
  -F "video=@/tmp/test_clipcc.mp4" \
  -F 'labels=["a"]' | python3 -m json.tool
```
Expected: 422 with "Please provide between 3 and 10 labels"

- [ ] **Step 6: Stop container and commit**

```bash
docker compose --profile cpu down
```

---

## Self-Review

**1. Spec coverage:**
- [x] POST /api/v1/classify with multipart — Task 9
- [x] Mean and max aggregation — Task 5 + 9
- [x] All error codes (401, 413, 415, 422, 429, 500, 504) — Tasks 1, 7, 8, 9
- [x] ASGI middleware (auth + upload concurrency + body size) — Task 7
- [x] ResourceGates (upload + inference limiters) — Task 6
- [x] InferenceRunner (timeout, cancel, subprocess kill) — Task 8
- [x] TempStore (save, cleanup, janitor) — Task 2
- [x] FrameExtractor + FrameSample — Task 3
- [x] ffprobe validation (resolution, streams, duration) — Task 3
- [x] ClipModel (load, encode, tokenize, logit_scale) — Task 4
- [x] Scoring (compute_frame_scores, aggregate_mean/max) — Task 5
- [x] Token truncation + duplicate check — Task 9
- [x] ModelSpec from .baked_model — Task 1
- [x] GET /live and GET /ready — Task 9
- [x] Fail-closed auth — Task 1
- [x] Docker CPU/GPU — Task 10
- [x] Baked model weights — Task 10
- [x] prompt_template with str.replace — Task 9
- [x] approx_timestamp_seconds — Tasks 3, 5

**2. Placeholder scan:** No TBD/TODO/placeholders found.

**3. Type consistency:**
- `FrameSample` — consistent across video.py, scoring.py, main.py
- `StoredUpload` — consistent across temp_store.py and main.py
- `ModelSpec` — consistent across model_spec.py, clip_model.py, main.py
- `ScoreItem`/`BestMatch`/`ClassifyResponse` — consistent across schemas and main
- `ResourceGates.upload_admission()`/`.inference_admission()` — consistent
- `InferenceRunner.run()` signature `(cancel_event, runner_ref) -> Any` — consistent
- `compute_frame_scores(cosine_sim, logit_scale)` returns `(confidence, raw_similarity)` — consistent
