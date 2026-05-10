"""Integration test: load model -> classify video -> verify response."""
import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
import torch
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import create_app
from app.models.base_model import ScoreBatch


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _make_mock_siglip2(*args, **kwargs):
    """Create a mock that satisfies the BaseModel interface."""
    mock = MagicMock()
    mock.model_type = "siglip2"
    mock.device = "cpu"
    mock.max_token_length = 64

    def _validate_prompts(prompts):
        return [10] * len(prompts)

    mock.validate_prompts = _validate_prompts

    def _tokenize_raw(prompts):
        return [torch.tensor([i + 1, i + 2, i + 3]) for i in range(len(prompts))]

    mock.tokenize_raw = _tokenize_raw

    def _score_batch(images, texts):
        n_images = len(images)
        n_texts = len(texts)
        raw_logits = torch.randn(n_images, n_texts)
        confidence = torch.softmax(raw_logits, dim=-1)
        raw_similarity = torch.randn(n_images, n_texts) * 0.3
        return ScoreBatch(
            confidence=confidence,
            raw_similarity=raw_similarity,
            logits=raw_logits,
            semantics="siglip2_pairwise_sigmoid",
        )

    mock.score_batch = _score_batch
    return mock


@pytest.fixture
def test_settings(temp_dir):
    return Settings(
        max_file_size_mb=10,
        max_duration_seconds=30,
        max_frames=30,
        default_fps=1.0,
        batch_size=4,
        max_concurrent_requests=2,
        allow_unauthenticated=True,
        ffmpeg_timeout_seconds=30,
        request_timeout_seconds=30,
        clip_cache_dir=str(temp_dir / "models"),
        temp_dir=str(temp_dir / "temp"),
    )


@pytest.fixture
async def client(test_settings):
    with patch("app.models.model_manager.SigLip2Model", side_effect=_make_mock_siglip2):
        app = create_app(test_settings)
        inner_app = app._app
        async with inner_app.router.lifespan_context(inner_app):
            await asyncio.sleep(0.1)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                yield c


LABELS = ["outdoor scene", "indoor scene", "vehicle"]


@pytest.mark.anyio
async def test_full_load_then_classify(client, small_video):
    """End-to-end: load a model, verify it is active, classify, check response."""

    # 1. Load a model explicitly
    r = await client.post(
        "/api/v1/models/load",
        json={"model_id": "siglip2-base-patch16-384"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "loaded"
    assert data["model_id"] == "siglip2-base-patch16-384"

    # 2. Verify it is the active model
    r = await client.get("/api/v1/models/active")
    assert r.status_code == 200
    active = r.json()
    assert active["model_id"] == "siglip2-base-patch16-384"
    assert active["model_type"] == "siglip2"
    assert "device" in active

    # 3. Classify a video
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={"labels": json.dumps(LABELS)},
    )
    assert r.status_code == 200
    result = r.json()

    # 4. Verify response structure
    assert "best_match" in result
    assert result["best_match"]["label"] in LABELS
    assert 0.0 <= result["best_match"]["confidence"] <= 1.0

    assert "scores" in result
    assert len(result["scores"]) == len(LABELS)
    returned_labels = {s["label"] for s in result["scores"]}
    assert returned_labels == set(LABELS)
    for score in result["scores"]:
        assert 0.0 <= score["confidence"] <= 1.0

    assert "metadata" in result
    meta = result["metadata"]
    assert meta["aggregation"] == "mean"
    assert meta["model"] == "siglip2-base-patch16-384"
    assert "frames_analyzed" in meta
    assert meta["frames_analyzed"] > 0
    assert "processing_time_seconds" in meta
    assert "device" in meta

    # Confidences should sum to ~1.0 (softmax mock)
    total = sum(s["confidence"] for s in result["scores"])
    assert abs(total - 1.0) < 0.01
