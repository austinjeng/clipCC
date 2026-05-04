from __future__ import annotations

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
    model_type: str = ""
    score_semantics: str = ""


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
