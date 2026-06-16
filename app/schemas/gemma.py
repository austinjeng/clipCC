from __future__ import annotations

from pydantic import BaseModel

GEMMA_SCORE_SEMANTICS = "gemma4_verbalized_uncalibrated"

GEMMA_DISCLAIMER = (
    "Scores are verbalized self-reports from a generative model: uncalibrated, "
    "quantized to round numbers, and implicitly contrastive across labels. "
    "Treat as ordinal. NOT comparable in magnitude to SigLIP2 sigmoid scores. "
    "Not suitable for safety-critical decisions."
)


class GemmaScoreItem(BaseModel):
    label: str
    score: float | None  # None = model omitted this label id
    evidence: str | None = None


class GemmaLatency(BaseModel):
    extract_seconds: float
    generate_seconds: float
    parse_seconds: float


class GemmaMetadata(BaseModel):
    model: str
    device: str
    frames_analyzed: int
    window_start_seconds: float
    window_end_seconds: float
    video_duration_seconds: float
    score_semantics: str = GEMMA_SCORE_SEMANTICS
    disclaimer: str = GEMMA_DISCLAIMER
    latency: GemmaLatency
    parse_retries: int = 0


class GemmaLabelScoresResponse(BaseModel):
    scores: list[GemmaScoreItem]
    metadata: GemmaMetadata


class GemmaQAResponse(BaseModel):
    answer: str
    metadata: GemmaMetadata


class GemmaStatusResponse(BaseModel):
    enabled: bool
    state: str  # idle | loading | loaded | failed
    error: str | None = None
    model_id: str
    device: str
    default_label_instruction: str = ""  # default editable prompt for label-scores mode
    label_scores_contract: str = ""  # locked JSON-output contract appended to every label-scores prompt
