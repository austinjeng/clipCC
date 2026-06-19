import torch
from dataclasses import dataclass

from app.config import Settings
from app.schemas.hybrid import (
    HybridFrameRef, HybridLabelResult, HybridResponse, HybridMetadata, HybridLatency,
)
from app.services.hybrid_select import (
    per_label_scores, gate_and_rank_labels, select_topk_spread, FrameRef,
)


def test_config_has_hybrid_defaults():
    s = Settings(allow_unauthenticated=True)
    assert s.gemma_max_new_tokens_verdict == 160
    assert s.hybrid_max_verified_labels == 6
    assert s.hybrid_thumbnail_px == 160


def test_label_result_defaults_for_skipped_label():
    r = HybridLabelResult(label="x", siglip2_score=0.4, gemma_evaluated=False)
    assert r.verdict is None
    assert r.explanation is None
    assert r.parse_failed is False
    assert r.frames_shown == []


def test_response_round_trips():
    resp = HybridResponse(
        results=[HybridLabelResult(
            label="texting", siglip2_score=0.91, gemma_evaluated=True,
            verdict="present", explanation="phone in hand",
            frames_shown=[HybridFrameRef(frame_index=4, timestamp_seconds=4.0, score=0.93, thumbnail="data:image/jpeg;base64,AAAA")],
        )],
        metadata=HybridMetadata(
            siglip2_model="siglip2-base", gemma_model="google/gemma-4-E2B-it", device="cpu",
            frames_analyzed=60, video_duration_seconds=60.0, aggregation="max", threshold=0.5,
            top_k=3, max_verified_labels=6, labels_above_threshold=1, labels_truncated=0,
            gemma_calls=1, latency=HybridLatency(siglip2_seconds=1.0, gemma_seconds=2.0),
        ),
    )
    assert resp.results[0].verdict == "present"
    assert resp.metadata.gemma_score_semantics == "gemma4_verbalized_uncalibrated"
    assert "not the same scale" in resp.metadata.disclaimer


@dataclass
class _FakeFrame:
    approx_timestamp_seconds: float


@dataclass
class _FakeCtx:
    confidence: torch.Tensor
    labels: list
    frames: list


def _ctx(conf_rows, labels, timestamps):
    return _FakeCtx(
        confidence=torch.tensor(conf_rows, dtype=torch.float32),
        labels=labels,
        frames=[_FakeFrame(t) for t in timestamps],
    )


def test_per_label_scores_max_vs_mean():
    ctx = _ctx([[0.1, 0.9], [0.8, 0.2]], ["a", "b"], [0.0, 1.0])
    assert per_label_scores(ctx, "max") == [0.8, 0.9]
    assert per_label_scores(ctx, "mean") == [round(0.45, 6), round(0.55, 6)]


def test_gate_filters_by_threshold_and_caps():
    ctx = _ctx([[0.9, 0.7, 0.6, 0.2]], ["a", "b", "c", "d"], [0.0])
    scores = per_label_scores(ctx, "max")  # [0.9, 0.7, 0.6, 0.2]
    selected, n_above, n_trunc = gate_and_rank_labels(ctx, scores, threshold=0.5, max_verified_labels=2)
    assert [g.label for g in selected] == ["a", "b"]   # top-2 by score
    assert n_above == 3                                  # a, b, c clear 0.5
    assert n_trunc == 1                                  # c dropped by the cap


def test_select_topk_spread_enforces_min_gap():
    # frames clustered at 4.0/4.1/4.2 then one at 22.0; k=2 must skip near-dups
    conf = [[0.95], [0.94], [0.93], [0.80]]
    ctx = _ctx(conf, ["a"], [4.0, 4.1, 4.2, 22.0])
    refs = select_topk_spread(ctx, label_idx=0, k=2, min_gap_seconds=0.5)
    times = sorted(r.timestamp_seconds for r in refs)
    assert times == [4.0, 22.0]


def test_select_topk_spread_truncates_to_k():
    conf = [[0.9], [0.8], [0.7]]
    ctx = _ctx(conf, ["a"], [0.0, 1.0, 2.0])
    refs = select_topk_spread(ctx, label_idx=0, k=2)
    assert len(refs) == 2
    assert isinstance(refs[0], FrameRef)
    assert refs[0].timestamp_seconds == 0.0  # highest score first (ranked)
