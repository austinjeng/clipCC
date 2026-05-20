import torch
import pytest
from pathlib import Path
from app.services.scoring import compute_frame_scores, aggregate_mean, aggregate_max, aggregate_frame_scores, ScoringContext, AggregationResult
from app.services.scoring import aggregate_temporal, TemporalScoringContext
from app.services.frame_timeline import FrameTimeline
from app.services.temporal_policy import SigLip2Policy
from app.schemas.response import BestMatch, ResolvedTemporalOptions
from app.models.base_model import ScoreBatch
from app.services.video import FrameSample

def make_frame(index: int, fps: float = 1.0) -> FrameSample:
    return FrameSample(path=Path(f"/tmp/frame_{index:05d}.jpg"), sample_index=index, approx_timestamp_seconds=index / fps)

def test_scoring_context_from_batches_single():
    batch = ScoreBatch(
        confidence=torch.tensor([[0.8, 0.2]]),
        raw_similarity=torch.tensor([[0.5, 0.3]]),
        logits=torch.tensor([[1.5, -1.0]]),
        semantics="siglip2_pairwise_sigmoid",
    )
    frames = [make_frame(0)]
    ctx = ScoringContext.from_batches([batch], ["a", "b"], frames)
    assert ctx.confidence.shape == (1, 2)
    assert ctx.semantics == "siglip2_pairwise_sigmoid"
    assert ctx.labels == ["a", "b"]

def test_scoring_context_from_batches_multi():
    batch1 = ScoreBatch(
        confidence=torch.tensor([[0.8, 0.2]]),
        raw_similarity=torch.tensor([[0.5, 0.3]]),
        logits=torch.tensor([[1.5, -1.0]]),
        semantics="siglip2_pairwise_sigmoid",
    )
    batch2 = ScoreBatch(
        confidence=torch.tensor([[0.3, 0.9]]),
        raw_similarity=torch.tensor([[0.2, 0.7]]),
        logits=torch.tensor([[-0.5, 2.0]]),
        semantics="siglip2_pairwise_sigmoid",
    )
    frames = [make_frame(0), make_frame(1)]
    ctx = ScoringContext.from_batches([batch1, batch2], ["a", "b"], frames)
    assert ctx.confidence.shape == (2, 2)
    assert ctx.logits.shape == (2, 2)

def test_scoring_context_empty_batches_raises():
    with pytest.raises(ValueError, match="empty batch list"):
        ScoringContext.from_batches([], ["a"], [])

def test_scoring_context_mixed_semantics_raises():
    batch1 = ScoreBatch(
        confidence=torch.tensor([[0.8]]),
        raw_similarity=torch.tensor([[0.5]]),
        logits=torch.tensor([[1.5]]),
        semantics="siglip2_pairwise_sigmoid",
    )
    batch2 = ScoreBatch(
        confidence=torch.tensor([[0.3]]),
        raw_similarity=torch.tensor([[0.2]]),
        logits=torch.tensor([[-0.5]]),
        semantics="clip_relative_softmax",
    )
    with pytest.raises(ValueError, match="Mixed model semantics"):
        ScoringContext.from_batches([batch1, batch2], ["a"], [make_frame(0), make_frame(1)])

def test_compute_frame_scores_shape():
    cosine = torch.tensor([[0.25, 0.30], [0.28, 0.32]])
    conf, raw = compute_frame_scores(cosine, logit_scale=100.0)
    assert conf.shape == (2, 2)
    assert raw.shape == (2, 2)

def test_confidence_sums_to_one():
    cosine = torch.tensor([[0.25, 0.30, 0.20]])
    conf, _ = compute_frame_scores(cosine, logit_scale=100.0)
    assert abs(conf[0].sum().item() - 1.0) < 1e-5

def test_raw_similarity_is_unscaled():
    cosine = torch.tensor([[0.25, 0.30]])
    _, raw = compute_frame_scores(cosine, logit_scale=100.0)
    assert torch.allclose(raw, cosine, atol=1e-6)

def test_aggregate_mean():
    conf = torch.tensor([[0.3, 0.7], [0.4, 0.6]])
    raw = torch.tensor([[0.25, 0.30], [0.27, 0.28]])
    labels = ["driving", "parking"]
    frames = [make_frame(0), make_frame(1)]
    ctx = ScoringContext(
        confidence=conf, raw_similarity=raw,
        logits=torch.zeros_like(conf), semantics="test",
        labels=labels, frames=frames,
    )
    result = aggregate_mean(ctx)
    assert len(result.scores) == 2
    assert abs(result.scores[0].confidence - 0.35) < 1e-5
    assert result.scores[0].peak_frame_index is None

def test_aggregate_max():
    conf = torch.tensor([[0.3, 0.7], [0.8, 0.2]])
    raw = torch.tensor([[0.25, 0.30], [0.31, 0.22]])
    labels = ["driving", "parking"]
    frames = [make_frame(0), make_frame(1)]
    ctx = ScoringContext(
        confidence=conf, raw_similarity=raw,
        logits=torch.zeros_like(conf), semantics="test",
        labels=labels, frames=frames,
    )
    result = aggregate_max(ctx)
    assert len(result.scores) == 2
    assert abs(result.scores[0].confidence - 0.8) < 1e-5
    assert result.scores[0].peak_frame_index == 1
    assert abs(result.scores[1].confidence - 0.7) < 1e-5
    assert result.scores[1].peak_frame_index == 0

def test_aggregate_frame_scores_mean():
    batch1 = ScoreBatch(
        confidence=torch.tensor([[0.8, 0.1, 0.1], [0.6, 0.2, 0.2]]),
        raw_similarity=torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.3, 0.3]]),
        logits=torch.tensor([[2.0, -1.0, -1.0], [1.0, -0.5, -0.5]]),
        semantics="clip_relative_softmax",
    )
    frames = [make_frame(0), make_frame(1)]
    labels = ["cat", "dog", "bird"]
    result = aggregate_frame_scores([batch1], labels, frames, "mean")
    assert len(result.scores) == 3
    assert result.best_match.label == "cat"
    assert result.best_match.confidence > 0

def test_aggregate_frame_scores_max():
    batch1 = ScoreBatch(
        confidence=torch.tensor([[0.3, 0.9, 0.1]]),
        raw_similarity=torch.tensor([[0.2, 0.7, 0.1]]),
        logits=torch.tensor([[0.0, 2.0, -1.0]]),
        semantics="siglip2_pairwise_sigmoid",
    )
    batch2 = ScoreBatch(
        confidence=torch.tensor([[0.8, 0.2, 0.1]]),
        raw_similarity=torch.tensor([[0.6, 0.1, 0.05]]),
        logits=torch.tensor([[1.5, -0.5, -1.0]]),
        semantics="siglip2_pairwise_sigmoid",
    )
    frames = [make_frame(0), make_frame(1)]
    labels = ["cat", "dog", "bird"]
    result = aggregate_frame_scores([batch1, batch2], labels, frames, "max")
    assert len(result.scores) == 3
    assert result.best_match.label == "dog"


def make_temporal_ctx(
    confidence: torch.Tensor,
    labels: list[str],
    fps: float = 1.0,
    video_duration: float | None = None,
) -> TemporalScoringContext:
    n_frames = confidence.shape[0]
    if video_duration is None:
        video_duration = float(n_frames) / fps
    frames = [make_frame(i, fps) for i in range(n_frames)]
    timeline = FrameTimeline(frames, fps, video_duration)
    ctx = ScoringContext(
        confidence=confidence,
        raw_similarity=confidence.clone(),
        logits=torch.zeros_like(confidence),
        semantics="siglip2_pairwise_sigmoid",
        labels=labels,
        frames=frames,
    )
    return TemporalScoringContext.from_base(ctx, timeline)


def default_options(threshold: float = 0.5) -> ResolvedTemporalOptions:
    return ResolvedTemporalOptions(
        threshold=threshold,
        gap_tolerance=2.0,
        min_duration=1.0,
        threshold_was_defaulted=True,
    )


def test_temporal_basic_segment_detection():
    scores = torch.tensor([
        [0.1], [0.2], [0.1],
        [0.7], [0.8], [0.9], [0.6], [0.7],
        [0.1], [0.2],
    ])
    ctx = make_temporal_ctx(scores, ["sleepy"])
    opts = default_options(threshold=0.5)
    policy = SigLip2Policy()
    result = aggregate_temporal(ctx, opts, policy)
    assert result.temporal is not None
    assert len(result.temporal.timeline) == 10
    assert len(result.temporal.segments) == 1
    seg = result.temporal.segments[0]
    assert seg.label == "sleepy"
    assert seg.start_time == 3.0
    assert seg.end_time == 8.0
    assert abs(seg.duration - 5.0) < 1e-6
    assert seg.peak_confidence == pytest.approx(0.9, abs=1e-6)
    assert seg.peak_timestamp == 5.0


def test_temporal_gap_bridging():
    scores = torch.tensor([[0.8], [0.7], [0.3], [0.8], [0.9]])
    ctx = make_temporal_ctx(scores, ["sleepy"])
    opts = ResolvedTemporalOptions(
        threshold=0.5, gap_tolerance=2.0, min_duration=0.0, threshold_was_defaulted=True,
    )
    result = aggregate_temporal(ctx, opts, SigLip2Policy())
    assert len(result.temporal.segments) == 1
    seg = result.temporal.segments[0]
    assert seg.start_time == 0.0
    assert seg.end_time == 5.0


def test_temporal_gap_not_bridged():
    scores = torch.tensor([[0.8], [0.7], [0.3], [0.8], [0.9]])
    ctx = make_temporal_ctx(scores, ["sleepy"])
    opts = ResolvedTemporalOptions(
        threshold=0.5, gap_tolerance=0.0, min_duration=0.0, threshold_was_defaulted=True,
    )
    result = aggregate_temporal(ctx, opts, SigLip2Policy())
    assert len(result.temporal.segments) == 2


def test_temporal_min_duration_filter():
    scores = torch.tensor([[0.1], [0.8], [0.1], [0.1], [0.1]])
    ctx = make_temporal_ctx(scores, ["sleepy"])
    opts = ResolvedTemporalOptions(
        threshold=0.5, gap_tolerance=0.0, min_duration=2.0, threshold_was_defaulted=True,
    )
    result = aggregate_temporal(ctx, opts, SigLip2Policy())
    assert len(result.temporal.segments) == 0


def test_temporal_all_below_threshold():
    scores = torch.tensor([[0.1], [0.2], [0.3]])
    ctx = make_temporal_ctx(scores, ["sleepy"])
    result = aggregate_temporal(ctx, default_options(), SigLip2Policy())
    assert len(result.temporal.segments) == 0
    assert result.temporal.best_segment is None
    assert len(result.temporal.timeline) == 3
    assert len(result.temporal.label_summaries) == 1
    assert result.temporal.label_summaries[0].segment_count == 0


def test_temporal_all_above_threshold():
    scores = torch.tensor([[0.8], [0.9], [0.7]])
    ctx = make_temporal_ctx(scores, ["sleepy"])
    opts = ResolvedTemporalOptions(
        threshold=0.5, gap_tolerance=0.0, min_duration=0.0, threshold_was_defaulted=True,
    )
    result = aggregate_temporal(ctx, opts, SigLip2Policy())
    assert len(result.temporal.segments) == 1
    seg = result.temporal.segments[0]
    assert seg.stats.coverage_ratio == 1.0
    assert seg.stats.active_avg == seg.stats.interval_avg


def test_temporal_segment_stats_with_gap():
    scores = torch.tensor([[0.8], [0.7], [0.3], [0.9]])
    ctx = make_temporal_ctx(scores, ["sleepy"])
    opts = ResolvedTemporalOptions(
        threshold=0.5, gap_tolerance=2.0, min_duration=0.0, threshold_was_defaulted=True,
    )
    result = aggregate_temporal(ctx, opts, SigLip2Policy())
    seg = result.temporal.segments[0]
    assert seg.stats.active_avg == pytest.approx(0.8, abs=1e-6)
    assert seg.stats.interval_avg == pytest.approx(0.675, abs=1e-6)
    assert seg.stats.coverage_ratio == pytest.approx(0.75, abs=1e-6)
    assert seg.stats.active_duration == pytest.approx(3.0, abs=1e-6)


def test_temporal_label_summaries():
    scores = torch.tensor([[0.8, 0.1], [0.9, 0.2], [0.7, 0.1]])
    ctx = make_temporal_ctx(scores, ["sleepy", "awake"])
    opts = ResolvedTemporalOptions(
        threshold=0.5, gap_tolerance=0.0, min_duration=0.0, threshold_was_defaulted=True,
    )
    result = aggregate_temporal(ctx, opts, SigLip2Policy())
    summaries = {s.label: s for s in result.temporal.label_summaries}
    assert summaries["sleepy"].segment_count == 1
    assert summaries["sleepy"].total_active_duration == pytest.approx(3.0, abs=1e-6)
    assert summaries["sleepy"].peak_confidence == pytest.approx(0.9, abs=1e-6)
    assert summaries["awake"].segment_count == 0
    assert summaries["awake"].total_active_duration == 0.0


def test_temporal_best_segment():
    scores = torch.tensor([[0.7, 0.95], [0.6, 0.1]])
    ctx = make_temporal_ctx(scores, ["a", "b"])
    opts = ResolvedTemporalOptions(
        threshold=0.5, gap_tolerance=0.0, min_duration=0.0, threshold_was_defaulted=True,
    )
    result = aggregate_temporal(ctx, opts, SigLip2Policy())
    assert result.temporal.best_segment is not None
    assert result.temporal.best_segment.label == "b"
    assert result.temporal.best_segment.peak_confidence == pytest.approx(0.95, abs=1e-6)


def test_temporal_effective_threshold_in_result():
    scores = torch.tensor([[0.8], [0.2]])
    ctx = make_temporal_ctx(scores, ["x"])
    opts = ResolvedTemporalOptions(
        threshold=0.6, gap_tolerance=2.0, min_duration=1.0, threshold_was_defaulted=False,
    )
    result = aggregate_temporal(ctx, opts, SigLip2Policy())
    assert result.temporal.effective_threshold == 0.6
    assert result.temporal.threshold_was_defaulted is False
    assert result.temporal.threshold_mode == "absolute"


def test_contrast_result_serialization():
    from app.schemas.response import ContrastLabelScore, ContrastGroupResult, ContrastResult
    result = ContrastResult(
        verdict="positive",
        difference=0.27,
        threshold=0.15,
        threshold_was_defaulted=True,
        threshold_source="model_policy",
        calibration_status="uncalibrated",
        contrast_reduce="mean",
        positive=ContrastGroupResult(
            group="positive",
            mean_group_score=0.72,
            labels=[ContrastLabelScore(label="safe driving", score=0.72)],
        ),
        negative=ContrastGroupResult(
            group="negative",
            mean_group_score=0.45,
            labels=[ContrastLabelScore(label="dangerous driving", score=0.45)],
        ),
        score_semantics="siglip2_pairwise_sigmoid",
        label_pooling="mean",
        dominant_label="safe driving",
    )
    d = result.model_dump()
    assert d["verdict"] == "positive"
    assert d["dominant_label"] == "safe driving"
    assert d["positive"]["mean_group_score"] == 0.72
    assert len(d["positive"]["labels"]) == 1


def test_resolved_contrast_options_defaults():
    from app.schemas.response import RawContrastParams, ResolvedContrastOptions
    raw = RawContrastParams()
    opts = ResolvedContrastOptions.resolve(raw, policy_threshold=0.15, policy_reduce="mean")
    assert opts.threshold == 0.15
    assert opts.threshold_was_defaulted is True
    assert opts.threshold_source == "model_policy"
    assert opts.contrast_reduce == "mean"


def test_resolved_contrast_options_user_override():
    from app.schemas.response import RawContrastParams, ResolvedContrastOptions
    raw = RawContrastParams(threshold=0.25, contrast_reduce="top_k_mean")
    opts = ResolvedContrastOptions.resolve(raw, policy_threshold=0.15, policy_reduce="mean")
    assert opts.threshold == 0.25
    assert opts.threshold_was_defaulted is False
    assert opts.threshold_source == "user"
    assert opts.contrast_reduce == "top_k_mean"


def test_reduce_mean():
    from app.services.scoring import contrast_reduce
    margins = torch.tensor([0.1, 0.2, 0.3, -0.1, -0.2])
    result = contrast_reduce(margins, "mean")
    assert abs(result - 0.06) < 1e-5


def test_reduce_top_k_mean_positive_event():
    from app.services.scoring import contrast_reduce
    margins = torch.tensor([0.01, 0.02, -0.01, 0.8, 0.9, 0.01, -0.02, 0.01, 0.03, 0.02])
    result = contrast_reduce(margins, "top_k_mean")
    # k = max(1, ceil(10*0.10)) = 1, picks largest abs = 0.9
    assert abs(result - 0.9) < 1e-5


def test_reduce_top_k_mean_negative_event():
    from app.services.scoring import contrast_reduce
    margins = torch.tensor([0.01, 0.02, -0.01, -0.8, -0.9, 0.01, -0.02, 0.01, 0.03, 0.02])
    result = contrast_reduce(margins, "top_k_mean")
    assert abs(result - (-0.9)) < 1e-5


def test_reduce_top_k_mean_single_frame():
    from app.services.scoring import contrast_reduce
    margins = torch.tensor([0.5])
    result = contrast_reduce(margins, "top_k_mean")
    assert abs(result - 0.5) < 1e-5


def test_reduce_max_positive():
    from app.services.scoring import contrast_reduce
    margins = torch.tensor([0.1, -0.3, 0.5, -0.2])
    result = contrast_reduce(margins, "max")
    assert abs(result - 0.5) < 1e-5


def test_reduce_max_negative_stronger():
    from app.services.scoring import contrast_reduce
    margins = torch.tensor([0.1, -0.8, 0.3, -0.2])
    result = contrast_reduce(margins, "max")
    assert abs(result - (-0.8)) < 1e-5


def test_reduce_quantile_positive_tail():
    from app.services.scoring import contrast_reduce
    margins = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7])
    result = contrast_reduce(margins, "quantile")
    assert result > 0


def test_reduce_quantile_negative_tail():
    from app.services.scoring import contrast_reduce
    margins = torch.tensor([-0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = contrast_reduce(margins, "quantile")
    assert result < 0
