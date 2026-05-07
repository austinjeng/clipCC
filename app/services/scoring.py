from __future__ import annotations

from dataclasses import dataclass

import torch

from app.models.base_model import ScoreBatch
from app.schemas.response import BestMatch, ScoreItem
from app.services.video import FrameSample


@dataclass
class ScoringContext:
    confidence: torch.Tensor
    raw_similarity: torch.Tensor
    logits: torch.Tensor
    semantics: str
    labels: list[str]
    frames: list[FrameSample]

    @classmethod
    def from_batches(
        cls,
        batches: list[ScoreBatch],
        labels: list[str],
        frames: list[FrameSample],
    ) -> ScoringContext:
        if not batches:
            raise ValueError("Cannot build ScoringContext from empty batch list")
        semantics_set = {b.semantics for b in batches}
        if len(semantics_set) != 1:
            raise ValueError(
                f"Mixed model semantics in single request: {semantics_set}. "
                "All batches must come from the same model."
            )
        return cls(
            confidence=torch.cat([b.confidence for b in batches], dim=0),
            raw_similarity=torch.cat([b.raw_similarity for b in batches], dim=0),
            logits=torch.cat([b.logits for b in batches], dim=0),
            semantics=semantics_set.pop(),
            labels=labels,
            frames=frames,
        )


@dataclass
class TemporalScoringContext(ScoringContext):
    timeline: "FrameTimeline"  # type: ignore[name-defined]

    @classmethod
    def from_base(
        cls, ctx: ScoringContext, timeline: "FrameTimeline",  # type: ignore[name-defined]
    ) -> TemporalScoringContext:
        return cls(
            confidence=ctx.confidence,
            raw_similarity=ctx.raw_similarity,
            logits=ctx.logits,
            semantics=ctx.semantics,
            labels=ctx.labels,
            frames=ctx.frames,
            timeline=timeline,
        )


@dataclass
class AggregationResult:
    scores: list[ScoreItem]
    best_match: BestMatch
    temporal: "TemporalResult | None" = None  # type: ignore[name-defined]


def compute_frame_scores(
    cosine_sim: torch.Tensor, logit_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_similarity = cosine_sim.clone()
    scaled_logits = cosine_sim * logit_scale
    confidence = torch.softmax(scaled_logits, dim=-1)
    return confidence, raw_similarity


def aggregate_mean(ctx: ScoringContext) -> AggregationResult:
    mean_conf = ctx.confidence.mean(dim=0)
    mean_raw = ctx.raw_similarity.mean(dim=0)
    scores = [
        ScoreItem(
            label=ctx.labels[i],
            confidence=round(mean_conf[i].item(), 6),
            raw_similarity=round(mean_raw[i].item(), 6),
        )
        for i in range(len(ctx.labels))
    ]
    best = max(scores, key=lambda s: s.confidence)
    return AggregationResult(
        scores=scores,
        best_match=BestMatch(label=best.label, confidence=best.confidence),
    )


def aggregate_max(ctx: ScoringContext) -> AggregationResult:
    max_conf, max_indices = ctx.confidence.max(dim=0)
    scores = [
        ScoreItem(
            label=ctx.labels[i],
            confidence=round(max_conf[i].item(), 6),
            raw_similarity=round(
                ctx.raw_similarity[max_indices[i].item(), i].item(), 6
            ),
            peak_frame_index=max_indices[i].item(),
            approx_timestamp_seconds=ctx.frames[
                max_indices[i].item()
            ].approx_timestamp_seconds,
        )
        for i in range(len(ctx.labels))
    ]
    best = max(scores, key=lambda s: s.confidence)
    return AggregationResult(
        scores=scores,
        best_match=BestMatch(label=best.label, confidence=best.confidence),
    )


def aggregate_temporal(
    ctx: TemporalScoringContext,
    temporal_options: "ResolvedTemporalOptions",  # type: ignore[name-defined]
    policy: "TemporalScoringPolicy",  # type: ignore[name-defined]
) -> AggregationResult:
    raise NotImplementedError("aggregate_temporal will be implemented in Task 6")


def aggregate_frame_scores(
    batches: list[ScoreBatch],
    labels: list[str],
    frames: list[FrameSample],
    aggregation: str,
    temporal_options: "ResolvedTemporalOptions | None" = None,  # type: ignore[name-defined]
    timeline: "FrameTimeline | None" = None,  # type: ignore[name-defined]
    policy: "TemporalScoringPolicy | None" = None,  # type: ignore[name-defined]
) -> AggregationResult:
    ctx = ScoringContext.from_batches(batches, labels, frames)

    if aggregation == "temporal":
        if timeline is None or policy is None or temporal_options is None:
            raise ValueError(
                "Temporal aggregation requires timeline, policy, and temporal_options"
            )
        temporal_ctx = TemporalScoringContext.from_base(ctx, timeline)
        return aggregate_temporal(temporal_ctx, temporal_options, policy)
    elif aggregation == "max":
        return aggregate_max(ctx)
    else:
        return aggregate_mean(ctx)
