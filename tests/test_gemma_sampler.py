import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from app.config import Settings
from app.errors.handlers import DurationTooLongError, InvalidGemmaParamsError
from app.services.gemma_sampler import plan_timestamps, resolve_window, validate_gemma_video
from app.services.video import VideoInfo


def make_info(duration=120.0, width=1280, height=720, streams=1):
    return VideoInfo(duration=duration, width=width, height=height,
                     video_stream_count=streams, format_name="mp4")


def settings():
    return Settings(allow_unauthenticated=True)


# --- resolve_window ---

def test_window_clamps_to_video_end():
    start, end = resolve_window(duration=45.0, window_start=0.0, window_seconds=60.0)
    assert (start, end) == (0.0, 45.0)


def test_window_with_offset():
    start, end = resolve_window(duration=180.0, window_start=30.0, window_seconds=60.0)
    assert (start, end) == (30.0, 90.0)


def test_window_start_beyond_duration_raises():
    with pytest.raises(InvalidGemmaParamsError):
        resolve_window(duration=50.0, window_start=60.0, window_seconds=60.0)


def test_negative_window_start_raises():
    with pytest.raises(InvalidGemmaParamsError):
        resolve_window(duration=50.0, window_start=-1.0, window_seconds=60.0)


# --- plan_timestamps ---

def test_timestamps_are_uniform_midpoints():
    ts = plan_timestamps(span_start=0.0, span_end=80.0, n_frames=8)
    assert ts == [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]


def test_timestamps_respect_offset_window():
    ts = plan_timestamps(span_start=30.0, span_end=90.0, n_frames=4)
    assert ts == [37.5, 52.5, 67.5, 82.5]


def test_short_video_fewer_frames_than_requested():
    # 1.5s video at most 1 frame per 0.5s spacing floor → fewer frames, no dupes
    ts = plan_timestamps(span_start=0.0, span_end=1.5, n_frames=8)
    assert len(ts) == len(set(ts))
    assert all(0.0 <= t <= 1.5 for t in ts)
    assert len(ts) <= 8


def test_zero_frames_returns_empty():
    assert plan_timestamps(span_start=0.0, span_end=10.0, n_frames=0) == []


def test_zero_span_returns_empty():
    assert plan_timestamps(span_start=5.0, span_end=5.0, n_frames=8) == []


# --- validate_gemma_video ---

def test_duration_limit_still_applies():
    with pytest.raises(DurationTooLongError):
        validate_gemma_video(make_info(duration=999.0), settings())


def test_long_video_within_limit_passes():
    # 290s exceeds 60s window but is under max_duration_seconds=300 — must NOT
    # raise (Gemma analyzes a window; the frames-vs-duration rule is SigLIP2's)
    validate_gemma_video(make_info(duration=290.0), settings())


def test_resolution_limit_applies():
    from app.errors.handlers import ResolutionTooHighError
    with pytest.raises(ResolutionTooHighError):
        validate_gemma_video(make_info(width=4000, height=2200), settings())


def test_multi_stream_rejected():
    from app.errors.handlers import MultipleVideoStreamsError
    with pytest.raises(MultipleVideoStreamsError):
        validate_gemma_video(make_info(streams=2), settings())


# --- extract_frames (real ffmpeg) ---

@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="requires ffmpeg")
def test_extract_frames_real_ffmpeg(tmp_path):
    from app.services.gemma_sampler import extract_frames

    video = tmp_path / "smoke.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         "testsrc=duration=5:size=320x240:rate=10", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    frames = extract_frames(video, [1.0, 2.5, 4.0], tmp_path, threading.Event())
    assert [f.timestamp_seconds for f in frames] == [1.0, 2.5, 4.0]
    assert all(f.path.exists() and f.path.stat().st_size > 0 for f in frames)
