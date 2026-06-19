from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.schemas.gemma import GEMMA_SCORE_SEMANTICS

SIGLIP2_SCORE_SEMANTICS = "siglip2_pairwise_sigmoid"

HYBRID_DISCLAIMER = (
    "SigLIP2 scores are pairwise sigmoid; Gemma verdicts are an independent "
    "generative second opinion (uncalibrated). They are not the same scale and "
    "must not be compared as numbers. Not suitable for safety-critical decisions."
)


class HybridFrameRef(BaseModel):
    frame_index: int
    timestamp_seconds: float
    score: float
    thumbnail: str  # data:image/jpeg;base64,...


class HybridLabelResult(BaseModel):
    label: str
    siglip2_score: float
    gemma_evaluated: bool  # Gemma was run on this label — NOT "confirmed present"
    verdict: Literal["present", "not_present", "uncertain"] | None = None
    explanation: str | None = None
    parse_failed: bool = False
    frames_shown: list[HybridFrameRef] = []


class HybridLatency(BaseModel):
    siglip2_seconds: float
    gemma_seconds: float


class HybridMetadata(BaseModel):
    siglip2_model: str
    gemma_model: str
    device: str
    frames_analyzed: int
    video_duration_seconds: float
    aggregation: str
    threshold: float
    top_k: int
    max_verified_labels: int
    labels_above_threshold: int
    labels_truncated: int
    gemma_calls: int
    siglip2_score_semantics: str = SIGLIP2_SCORE_SEMANTICS
    gemma_score_semantics: str = GEMMA_SCORE_SEMANTICS
    disclaimer: str = HYBRID_DISCLAIMER
    latency: HybridLatency


class HybridResponse(BaseModel):
    results: list[HybridLabelResult]
    metadata: HybridMetadata
