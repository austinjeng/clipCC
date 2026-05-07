import torch
import pytest
from pathlib import Path
from app.services.scoring import compute_frame_scores, aggregate_mean, aggregate_max, aggregate_frame_scores, ScoringContext, AggregationResult
from app.models.base_model import ScoreBatch
from app.services.video import FrameSample
from app.schemas.response import BestMatch

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
