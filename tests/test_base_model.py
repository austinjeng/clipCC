import pytest
import torch
from app.models.base_model import BaseModel, ScoreBatch


def test_score_batch_dataclass():
    batch = ScoreBatch(
        confidence=torch.tensor([[0.8, 0.2]]),
        raw_similarity=torch.tensor([[0.6, 0.3]]),
        logits=torch.tensor([[1.5, -0.5]]),
        semantics="clip_relative_softmax",
    )
    assert batch.confidence.shape == (1, 2)
    assert batch.semantics == "clip_relative_softmax"


def test_base_model_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseModel()
