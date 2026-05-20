from __future__ import annotations

import math
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


VALID_CONTRAST_REDUCTIONS = {"mean", "top_k_mean", "max", "quantile"}


def contrast_reduce(margins: torch.Tensor, mode: str) -> float:
    if mode == "mean":
        return margins.mean().item()
    elif mode == "top_k_mean":
        k = max(1, math.ceil(len(margins) * 0.10))
        _, indices = margins.abs().topk(k)
        return margins[indices].mean().item()
    elif mode == "max":
        idx = margins.abs().argmax()
        return margins[idx].item()
    elif mode == "quantile":
        pos = torch.quantile(margins.float(), 0.90).item()
        neg = torch.quantile(margins.float(), 0.10).item()
        return pos if abs(pos) >= abs(neg) else neg
    else:
        raise ValueError(f"Unknown contrast reduction mode: {mode}")


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
    from app.schemas.response import (
        FrameScore,
        LabelSummary,
        Segment,
        SegmentStats,
        TemporalResult,
    )

    detection = policy.detection_scores(ctx)
    threshold = temporal_options.threshold
    gap_tol = temporal_options.gap_tolerance
    min_dur = temporal_options.min_duration
    tl = ctx.timeline

    timeline_entries = []
    for i in range(detection.shape[0]):
        scores_dict = {
            ctx.labels[j]: round(detection[i, j].item(), 6)
            for j in range(len(ctx.labels))
        }
        timeline_entries.append(
            FrameScore(timestamp=tl.timestamp(i), frame_index=i, scores=scores_dict)
        )

    all_segments: list[Segment] = []
    for j, label in enumerate(ctx.labels):
        label_scores = detection[:, j]
        mask = label_scores >= threshold

        raw_segments: list[tuple[int, int]] = []
        in_segment = False
        start_idx = 0
        for i in range(len(mask)):
            if mask[i] and not in_segment:
                start_idx = i
                in_segment = True
            elif not mask[i] and in_segment:
                raw_segments.append((start_idx, i - 1))
                in_segment = False
        if in_segment:
            raw_segments.append((start_idx, len(mask) - 1))

        merged: list[tuple[int, int]] = []
        for seg in raw_segments:
            if merged and tl.gap_seconds(merged[-1][1], seg[0]) <= gap_tol:
                merged[-1] = (merged[-1][0], seg[1])
            else:
                merged.append(seg)

        for start, end in merged:
            duration = tl.segment_duration(start, end)
            if duration < min_dur:
                continue

            interval_scores = label_scores[start : end + 1]
            active_mask = interval_scores >= threshold
            active_scores = interval_scores[active_mask]

            active_dur = sum(
                tl.intervals[start + k].end - tl.intervals[start + k].start
                for k in range(end - start + 1)
                if active_mask[k]
            )

            peak_val, peak_rel = interval_scores.max(dim=0)
            peak_idx = start + peak_rel.item()

            all_segments.append(
                Segment(
                    label=label,
                    start_time=tl.timestamp(start),
                    end_time=tl.intervals[end].end,
                    duration=round(duration, 6),
                    stats=SegmentStats(
                        active_avg=round(active_scores.mean().item(), 6),
                        interval_avg=round(interval_scores.mean().item(), 6),
                        coverage_ratio=round(
                            active_mask.sum().item() / len(interval_scores), 6
                        ),
                        active_duration=round(active_dur, 6),
                    ),
                    peak_confidence=round(peak_val.item(), 6),
                    peak_timestamp=tl.timestamp(peak_idx),
                )
            )

    label_summaries: list[LabelSummary] = []
    for label in ctx.labels:
        label_segs = [s for s in all_segments if s.label == label]
        total_active = sum(s.stats.active_duration for s in label_segs)
        total_segment = sum(s.duration for s in label_segs)
        peak = max((s.peak_confidence for s in label_segs), default=0.0)

        if total_active > 0:
            dwc = sum(
                s.stats.active_duration * s.stats.active_avg for s in label_segs
            ) / total_active
        else:
            dwc = 0.0

        label_summaries.append(
            LabelSummary(
                label=label,
                segment_count=len(label_segs),
                total_active_duration=round(total_active, 6),
                total_segment_duration=round(total_segment, 6),
                peak_confidence=round(peak, 6),
                duration_weighted_confidence=round(dwc, 6),
            )
        )

    best_segment = (
        max(all_segments, key=lambda s: s.peak_confidence) if all_segments else None
    )

    mean_result = aggregate_mean(ctx)

    return AggregationResult(
        scores=mean_result.scores,
        best_match=mean_result.best_match,
        temporal=TemporalResult(
            timeline=timeline_entries,
            segments=all_segments,
            label_summaries=label_summaries,
            best_segment=best_segment,
            threshold_mode=policy.threshold_mode(),
            effective_threshold=temporal_options.threshold,
            threshold_was_defaulted=temporal_options.threshold_was_defaulted,
        ),
    )


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
