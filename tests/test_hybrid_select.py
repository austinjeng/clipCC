from app.config import Settings
from app.schemas.hybrid import (
    HybridFrameRef, HybridLabelResult, HybridResponse, HybridMetadata, HybridLatency,
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
