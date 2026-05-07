import torch
import pytest
from dataclasses import dataclass
from app.services.temporal_policy import (
    ScoreSemantics,
    TemporalScoringPolicy,
    SigLip2Policy,
    SoftmaxPolicy,
    get_policy,
)

def test_score_semantics_constants():
    assert ScoreSemantics.SIGLIP2_SIGMOID == "siglip2_pairwise_sigmoid"
    assert ScoreSemantics.CLIP_RELATIVE_SOFTMAX == "clip_relative_softmax"

def test_siglip2_policy_defaults():
    policy = SigLip2Policy()
    assert policy.default_threshold() == 0.5
    assert policy.threshold_mode() == "absolute"

def test_softmax_policy_defaults():
    policy = SoftmaxPolicy()
    assert policy.default_threshold() == 0.3
    assert policy.threshold_mode() == "relative"

def test_siglip2_detection_scores_returns_confidence():
    policy = SigLip2Policy()
    @dataclass
    class FakeCtx:
        confidence: torch.Tensor
        logits: torch.Tensor
    ctx = FakeCtx(
        confidence=torch.tensor([[0.8, 0.2], [0.3, 0.9]]),
        logits=torch.tensor([[1.5, -1.0], [-0.5, 2.0]]),
    )
    result = policy.detection_scores(ctx)
    assert torch.equal(result, ctx.confidence)

def test_softmax_detection_scores_returns_confidence():
    policy = SoftmaxPolicy()
    @dataclass
    class FakeCtx:
        confidence: torch.Tensor
        logits: torch.Tensor
    ctx = FakeCtx(
        confidence=torch.tensor([[0.7, 0.3]]),
        logits=torch.tensor([[1.0, -1.0]]),
    )
    result = policy.detection_scores(ctx)
    assert torch.equal(result, ctx.confidence)

def test_get_policy_siglip2():
    policy = get_policy(ScoreSemantics.SIGLIP2_SIGMOID)
    assert isinstance(policy, SigLip2Policy)

def test_get_policy_clip():
    policy = get_policy(ScoreSemantics.CLIP_RELATIVE_SOFTMAX)
    assert isinstance(policy, SoftmaxPolicy)

def test_get_policy_unknown_raises():
    with pytest.raises(ValueError, match="No temporal scoring policy"):
        get_policy("unknown_semantics")
