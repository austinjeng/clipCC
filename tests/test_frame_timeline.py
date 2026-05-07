import pytest
from pathlib import Path
from app.services.frame_timeline import FrameTimeline, FrameInterval
from app.services.video import FrameSample

def make_frames(count: int, fps: float = 1.0) -> list[FrameSample]:
    return [
        FrameSample(
            path=Path(f"/tmp/f{i}.jpg"),
            sample_index=i,
            approx_timestamp_seconds=i / fps,
        )
        for i in range(count)
    ]

def test_intervals_basic():
    frames = make_frames(3, fps=1.0)
    tl = FrameTimeline(frames, fps=1.0, video_duration=3.0)
    assert len(tl.intervals) == 3
    assert tl.intervals[0] == FrameInterval(index=0, start=0.0, end=1.0)
    assert tl.intervals[1] == FrameInterval(index=1, start=1.0, end=2.0)
    assert tl.intervals[2] == FrameInterval(index=2, start=2.0, end=3.0)

def test_final_frame_clamped_to_duration():
    frames = make_frames(3, fps=1.0)
    tl = FrameTimeline(frames, fps=1.0, video_duration=2.5)
    assert tl.intervals[2].end == 2.5

def test_gap_seconds_adjacent():
    frames = make_frames(5, fps=1.0)
    tl = FrameTimeline(frames, fps=1.0, video_duration=5.0)
    assert tl.gap_seconds(0, 1) == 0.0

def test_gap_seconds_with_skip():
    frames = make_frames(5, fps=1.0)
    tl = FrameTimeline(frames, fps=1.0, video_duration=5.0)
    assert tl.gap_seconds(1, 3) == 1.0

def test_segment_duration():
    frames = make_frames(5, fps=1.0)
    tl = FrameTimeline(frames, fps=1.0, video_duration=5.0)
    assert tl.segment_duration(1, 3) == 3.0

def test_segment_duration_single_frame():
    frames = make_frames(5, fps=1.0)
    tl = FrameTimeline(frames, fps=1.0, video_duration=5.0)
    assert tl.segment_duration(2, 2) == 1.0

def test_timestamp():
    frames = make_frames(3, fps=2.0)
    tl = FrameTimeline(frames, fps=2.0, video_duration=1.5)
    assert tl.timestamp(0) == 0.0
    assert tl.timestamp(1) == 0.5
    assert tl.timestamp(2) == 1.0

def test_high_fps():
    frames = make_frames(10, fps=5.0)
    tl = FrameTimeline(frames, fps=5.0, video_duration=2.0)
    assert tl.frame_interval == 0.2
    assert abs(tl.intervals[0].end - 0.2) < 1e-9
    assert abs(tl.segment_duration(0, 4) - 1.0) < 1e-9

def test_single_frame():
    frames = make_frames(1, fps=1.0)
    tl = FrameTimeline(frames, fps=1.0, video_duration=0.5)
    assert len(tl.intervals) == 1
    assert tl.intervals[0].start == 0.0
    assert tl.intervals[0].end == 0.5
    assert tl.segment_duration(0, 0) == 0.5
