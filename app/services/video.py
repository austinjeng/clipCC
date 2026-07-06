import json
import math
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
    FFmpegMissingError,
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
    except FileNotFoundError as e:
        raise FFmpegMissingError("ffprobe") from e
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
        runner: "Optional[InferenceRunner]" = None,
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
            # Register with the runner so a request timeout can kill ffmpeg
            # instead of letting it run to its own ffmpeg_timeout deadline.
            if runner is not None:
                runner.register_process(proc)
            _, stderr = proc.communicate(timeout=self.ffmpeg_timeout)
        except FileNotFoundError as e:
            raise FFmpegMissingError("ffmpeg") from e
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise RuntimeError(f"ffmpeg timed out after {self.ffmpeg_timeout}s")
        finally:
            self.active_process = None
            if runner is not None:
                runner.unregister_process()

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
