from __future__ import annotations

from pydantic import BaseModel, Field


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
    model_type: str = ""
    score_semantics: str = ""


class RawTemporalParams(BaseModel):
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    gap_tolerance: float | None = Field(None, ge=0.0, le=10.0)
    min_duration: float | None = Field(None, ge=0.0, le=10.0)

    def has_any(self) -> bool:
        return any(v is not None for v in (self.threshold, self.gap_tolerance, self.min_duration))


class ResolvedTemporalOptions(BaseModel):
    threshold: float
    gap_tolerance: float
    min_duration: float
    threshold_was_defaulted: bool

    @classmethod
    def resolve(cls, raw: RawTemporalParams, default_threshold: float) -> "ResolvedTemporalOptions":
        return cls(
            threshold=raw.threshold if raw.threshold is not None else default_threshold,
            gap_tolerance=raw.gap_tolerance if raw.gap_tolerance is not None else 2.0,
            min_duration=raw.min_duration if raw.min_duration is not None else 1.0,
            threshold_was_defaulted=(raw.threshold is None),
        )


class RawContrastParams(BaseModel):
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    contrast_reduce: str | None = None

    def has_any(self) -> bool:
        return self.threshold is not None or self.contrast_reduce is not None


class ResolvedContrastOptions(BaseModel):
    threshold: float
    threshold_was_defaulted: bool
    threshold_source: str
    calibration_status: str
    contrast_reduce: str

    @classmethod
    def resolve(
        cls,
        raw: RawContrastParams,
        policy_threshold: float,
        policy_reduce: str,
    ) -> "ResolvedContrastOptions":
        return cls(
            threshold=raw.threshold if raw.threshold is not None else policy_threshold,
            threshold_was_defaulted=(raw.threshold is None),
            threshold_source="user" if raw.threshold is not None else "model_policy",
            calibration_status="uncalibrated",
            contrast_reduce=raw.contrast_reduce if raw.contrast_reduce is not None else policy_reduce,
        )


class FrameScore(BaseModel):
    timestamp: float
    frame_index: int
    scores: dict[str, float]


class SegmentStats(BaseModel):
    active_avg: float
    interval_avg: float
    coverage_ratio: float
    active_duration: float


class Segment(BaseModel):
    label: str
    start_time: float
    end_time: float
    duration: float
    stats: SegmentStats
    peak_confidence: float
    peak_timestamp: float


class LabelSummary(BaseModel):
    label: str
    segment_count: int
    total_active_duration: float
    total_segment_duration: float
    peak_confidence: float
    duration_weighted_confidence: float


class TemporalResult(BaseModel):
    timeline: list[FrameScore]
    segments: list[Segment]
    label_summaries: list[LabelSummary]
    best_segment: Segment | None
    threshold_mode: str
    effective_threshold: float
    threshold_was_defaulted: bool


class ContrastLabelScore(BaseModel):
    label: str
    score: float


class ContrastGroupResult(BaseModel):
    group: str
    mean_group_score: float
    labels: list[ContrastLabelScore]


class ContrastResult(BaseModel):
    verdict: str
    difference: float
    threshold: float
    threshold_was_defaulted: bool
    threshold_source: str
    calibration_status: str
    contrast_reduce: str
    positive: ContrastGroupResult
    negative: ContrastGroupResult
    score_semantics: str
    label_pooling: str
    dominant_label: str | None


class ClassifyResponse(BaseModel):
    best_match: BestMatch
    scores: list[ScoreItem]
    metadata: ClassifyMetadata
    temporal: TemporalResult | None = None
    contrast: ContrastResult | None = None


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    model: str
    pretrained: str
    device: str


class ErrorResponse(BaseModel):
    detail: str
