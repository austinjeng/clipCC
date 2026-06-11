from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.inference_runner import InferenceRunner

from app.config import Settings
from app.errors.handlers import (
    DurationTooLongError,
    InvalidGemmaParamsError,
    MultipleVideoStreamsError,
    ResolutionTooHighError,
)
from app.services.video import VideoInfo

# Spacing floor: two frames closer than this are visually redundant for a VLM
_MIN_SPACING_SECONDS = 0.5


@dataclass
class GemmaFrame:
    path: Path
    timestamp_seconds: float


def validate_gemma_video(info: VideoInfo, settings: Settings) -> None:
    """Gemma-specific constraints: duration/resolution/stream rules apply,
    but NOT the frames-vs-duration rule (Gemma samples a fixed frame count
    from a window regardless of total duration)."""
    if info.duration > settings.max_duration_seconds:
        raise DurationTooLongError(info.duration, settings.max_duration_seconds)
    if info.width > 3840 or info.height > 2160:
        raise ResolutionTooHighError(info.width, info.height)
    if info.width * info.height > 8_300_000:
        raise ResolutionTooHighError(info.width, info.height)
    if info.video_stream_count > 1:
        raise MultipleVideoStreamsError(info.video_stream_count)


def resolve_window(duration: float, window_start: float, window_seconds: float) -> tuple[float, float]:
    if window_start < 0:
        raise InvalidGemmaParamsError("window_start must be >= 0.")
    if window_start >= duration:
        raise InvalidGemmaParamsError(
            f"window_start {window_start:.1f}s is beyond the video duration {duration:.1f}s."
        )
    return window_start, min(duration, window_start + window_seconds)


def plan_timestamps(span_start: float, span_end: float, n_frames: int) -> list[float]:
    """Uniform midpoint sampling: frame i sits at the center of slice i.
    These exact values are recorded with the frames and enumerated to the
    model — the closed timestamp set for events mode (spec §4/§6)."""
    if n_frames <= 0 or span_end - span_start <= 0:
        return []
    span = span_end - span_start
    n = min(n_frames, max(1, int(span / _MIN_SPACING_SECONDS)))
    slice_len = span / n
    return [round(span_start + (i + 0.5) * slice_len, 3) for i in range(n)]


def extract_frames(
    video_path: Path,
    timestamps: list[float],
    frame_dir: Path,
    cancel_event: threading.Event,
    ffmpeg_timeout: int = 120,
    runner: "Optional[InferenceRunner]" = None,
) -> list[GemmaFrame]:
    """One ffmpeg seek per timestamp. N is small (<=16) so per-seek process
    overhead is acceptable and -ss before -i makes each seek fast.
    Returns a partial list if cancel_event fires mid-loop — short output is
    a valid result, not an error."""
    frames: list[GemmaFrame] = []
    for i, ts in enumerate(timestamps):
        if cancel_event.is_set():
            break
        out_path = frame_dir / f"gemma_{i:03d}.jpg"
        cmd = [
            "ffmpeg", "-nostdin", "-v", "error",
            "-ss", f"{ts:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-vf", "scale='min(896,iw)':'min(896,ih)':force_original_aspect_ratio=decrease",
            "-q:v", "2",
            str(out_path),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if runner is not None:
            runner.register_process(proc)
        try:
            _, stderr = proc.communicate(timeout=ffmpeg_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise RuntimeError(f"ffmpeg seek at {ts:.1f}s timed out after {ffmpeg_timeout}s")
        finally:
            if runner is not None:
                runner.unregister_process()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg seek at {ts:.1f}s failed: {stderr.decode(errors='replace').strip()}")
        if out_path.exists():
            frames.append(GemmaFrame(path=out_path, timestamp_seconds=ts))
    return frames
