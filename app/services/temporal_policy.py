from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import torch

class ScoreSemantics:
    SIGLIP2_SIGMOID = "siglip2_pairwise_sigmoid"
    CLIP_RELATIVE_SOFTMAX = "clip_relative_softmax"

class TemporalScoringPolicy(ABC):
    @abstractmethod
    def detection_scores(self, ctx: Any) -> torch.Tensor: ...
    @abstractmethod
    def default_threshold(self) -> float: ...
    @abstractmethod
    def threshold_mode(self) -> str: ...
    @abstractmethod
    def contrast_label_pooling(self) -> str: ...
    @abstractmethod
    def contrast_default_threshold(self) -> float: ...
    @abstractmethod
    def contrast_default_reduction(self) -> str: ...

class SigLip2Policy(TemporalScoringPolicy):
    def detection_scores(self, ctx: Any) -> torch.Tensor:
        return ctx.confidence
    def default_threshold(self) -> float:
        return 0.5
    def threshold_mode(self) -> str:
        return "absolute"
    def contrast_label_pooling(self) -> str:
        return "mean"
    def contrast_default_threshold(self) -> float:
        return 0.15
    def contrast_default_reduction(self) -> str:
        return "mean"

class SoftmaxPolicy(TemporalScoringPolicy):
    def detection_scores(self, ctx: Any) -> torch.Tensor:
        return ctx.confidence
    def default_threshold(self) -> float:
        return 0.3
    def threshold_mode(self) -> str:
        return "relative"
    def contrast_label_pooling(self) -> str:
        return "logsumexp_normalized"
    def contrast_default_threshold(self) -> float:
        return 0.10
    def contrast_default_reduction(self) -> str:
        return "mean"

_POLICY_REGISTRY: dict[str, type[TemporalScoringPolicy]] = {
    ScoreSemantics.SIGLIP2_SIGMOID: SigLip2Policy,
    ScoreSemantics.CLIP_RELATIVE_SOFTMAX: SoftmaxPolicy,
}

def get_policy(semantics: str) -> TemporalScoringPolicy:
    cls = _POLICY_REGISTRY.get(semantics)
    if cls is None:
        raise ValueError(
            f"No temporal scoring policy registered for semantics '{semantics}'. "
            f"Known: {list(_POLICY_REGISTRY.keys())}"
        )
    return cls()
