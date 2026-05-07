from __future__ import annotations
from dataclasses import dataclass
from app.services.video import FrameSample

@dataclass(frozen=True, eq=True)
class FrameInterval:
    index: int
    start: float
    end: float

class FrameTimeline:
    def __init__(self, frames: list[FrameSample], fps: float, video_duration: float):
        self.frames = frames
        self.fps = fps
        self.video_duration = video_duration
        self.frame_interval = 1.0 / fps
        self.intervals: list[FrameInterval] = self._build_intervals()

    def _build_intervals(self) -> list[FrameInterval]:
        intervals = []
        for i, frame in enumerate(self.frames):
            start = frame.approx_timestamp_seconds
            end = min(start + self.frame_interval, self.video_duration)
            intervals.append(FrameInterval(index=i, start=start, end=end))
        return intervals

    def gap_seconds(self, seg_a_end_idx: int, seg_b_start_idx: int) -> float:
        return self.intervals[seg_b_start_idx].start - self.intervals[seg_a_end_idx].end

    def segment_duration(self, start_idx: int, end_idx: int) -> float:
        return self.intervals[end_idx].end - self.intervals[start_idx].start

    def timestamp(self, idx: int) -> float:
        return self.intervals[idx].start
