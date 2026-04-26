import torch
import pytest
from pathlib import Path
from app.services.scoring import compute_frame_scores, aggregate_mean, aggregate_max, build_response_scores
from app.services.video import FrameSample

def make_frame(index: int, fps: float = 1.0) -> FrameSample:
    return FrameSample(path=Path(f"/tmp/frame_{index:05d}.jpg"), sample_index=index, approx_timestamp_seconds=index / fps)

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
    result = aggregate_mean(conf, raw, labels, frames)
    assert len(result) == 2
    assert abs(result[0].confidence - 0.35) < 1e-5
    assert result[0].peak_frame_index is None

def test_aggregate_max():
    conf = torch.tensor([[0.3, 0.7], [0.8, 0.2]])
    raw = torch.tensor([[0.25, 0.30], [0.31, 0.22]])
    labels = ["driving", "parking"]
    frames = [make_frame(0), make_frame(1)]
    result = aggregate_max(conf, raw, labels, frames)
    assert len(result) == 2
    assert abs(result[0].confidence - 0.8) < 1e-5
    assert result[0].peak_frame_index == 1
    assert abs(result[1].confidence - 0.7) < 1e-5
    assert result[1].peak_frame_index == 0

def test_build_response_mean():
    conf = torch.tensor([[0.3, 0.7], [0.4, 0.6]])
    raw = torch.tensor([[0.25, 0.30], [0.27, 0.28]])
    frames = [make_frame(0), make_frame(1)]
    scores, best = build_response_scores(conf, raw, ["a", "b"], frames, "mean")
    assert best.label == "b"
    assert len(scores) == 2

def test_build_response_max():
    conf = torch.tensor([[0.3, 0.7], [0.8, 0.2]])
    raw = torch.tensor([[0.25, 0.30], [0.31, 0.22]])
    frames = [make_frame(0), make_frame(1)]
    scores, best = build_response_scores(conf, raw, ["a", "b"], frames, "max")
    assert best.label == "a"
    assert scores[0].peak_frame_index == 1
