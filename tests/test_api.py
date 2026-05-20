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
        # Use softmax-like values so mean confidences sum to ~1.0
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
        max_file_size_mb=10, max_duration_seconds=30, max_frames=30,
        default_fps=1.0, batch_size=4, max_concurrent_requests=2,
        allow_unauthenticated=True, ffmpeg_timeout_seconds=30,
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
            # Give the background auto-load task time to complete
            await asyncio.sleep(0.1)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                yield c


# ── Health / Ready ──────────────────────────────────────────────────


@pytest.mark.anyio
async def test_live(client):
    r = await client.get("/live")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.anyio
async def test_ready(client):
    r = await client.get("/ready")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ready"
    assert "model" in data


# ── Validation errors ──────────────────────────────────────────────


@pytest.mark.anyio
async def test_missing_video(client):
    r = await client.post("/api/v1/classify", data={"labels": json.dumps(["a", "b", "c"])})
    assert r.status_code == 422


@pytest.mark.anyio
async def test_label_defaults(client):
    r = await client.get("/api/v1/labels/defaults")
    assert r.status_code == 200
    data = r.json()
    assert "labels" in data
    assert isinstance(data["labels"], list)
    assert len(data["labels"]) >= 1


@pytest.mark.anyio
async def test_invalid_labels_count(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={"labels": json.dumps([])},
    )
    assert r.status_code == 422
    assert "1 and 50" in r.json()["detail"]


@pytest.mark.anyio
async def test_invalid_fps(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={"labels": json.dumps(["a", "b", "c"]), "fps": "10.0"},
    )
    assert r.status_code == 422
    assert "FPS" in r.json()["detail"]


@pytest.mark.anyio
async def test_unsupported_format(client):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.webm", b"fake", "video/webm")},
        data={"labels": json.dumps(["a", "b", "c"])},
    )
    assert r.status_code == 415


# ── Classify (mean / max) ─────────────────────────────────────────


@pytest.mark.anyio
async def test_classify_mean(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={"labels": json.dumps(["outdoor scene", "indoor scene", "vehicle"])},
    )
    assert r.status_code == 200
    data = r.json()
    assert "best_match" in data
    assert len(data["scores"]) == 3
    assert data["metadata"]["aggregation"] == "mean"
    # Mock uses softmax so confidences sum to ~1.0
    total = sum(s["confidence"] for s in data["scores"])
    assert abs(total - 1.0) < 0.01


@pytest.mark.anyio
async def test_classify_max(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={
            "labels": json.dumps(["outdoor scene", "indoor scene", "vehicle"]),
            "aggregation": "max",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["metadata"]["aggregation"] == "max"
    for score in data["scores"]:
        assert "peak_frame_index" in score
        assert "approx_timestamp_seconds" in score


# ── Model endpoints ───────────────────────────────────────────────


@pytest.mark.anyio
async def test_list_models(client):
    r = await client.get("/api/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) >= 6
    ids = [m["model_id"] for m in data]
    assert "siglip2-base-patch16-256" in ids
    # The default model should already be loaded
    loaded = [m for m in data if m["loaded"]]
    assert len(loaded) == 1


@pytest.mark.anyio
async def test_active_model(client):
    r = await client.get("/api/v1/models/active")
    assert r.status_code == 200
    data = r.json()
    assert data["model_id"] == "siglip2-base-patch16-256"
    assert data["model_type"] == "siglip2"
    assert "device" in data


@pytest.mark.anyio
async def test_load_model(client):
    r = await client.post(
        "/api/v1/models/load",
        json={"model_id": "siglip2-base-patch16-384"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "loaded"
    assert data["model_id"] == "siglip2-base-patch16-384"


@pytest.mark.anyio
async def test_load_unknown_model(client):
    r = await client.post(
        "/api/v1/models/load",
        json={"model_id": "nonexistent-model"},
    )
    assert r.status_code == 400
    assert "Unknown model_id" in r.json()["detail"]


@pytest.mark.anyio
async def test_active_model_after_swap(client):
    # Load a different model then check /active
    await client.post(
        "/api/v1/models/load",
        json={"model_id": "siglip2-base-patch16-384"},
    )
    r = await client.get("/api/v1/models/active")
    assert r.status_code == 200
    assert r.json()["model_id"] == "siglip2-base-patch16-384"


# ── Load model typed error responses ─────────────────────────────


@pytest.fixture
def isolated_settings(temp_dir):
    return Settings(
        allow_unauthenticated=True,
        clip_cache_dir=str(temp_dir / "models"),
        temp_dir=str(temp_dir / "tmp"),
        skip_model_autoload=True,
    )


class TestLoadModelErrorHandling:
    @pytest.mark.asyncio
    async def test_unknown_model_returns_400(self, isolated_settings):
        outer_app = create_app(isolated_settings)
        inner_app = outer_app._app
        async with inner_app.router.lifespan_context(inner_app):
            async with AsyncClient(transport=ASGITransport(app=outer_app), base_url="http://test") as client:
                resp = await client.post("/api/v1/models/load", json={"model_id": "nonexistent"})
                assert resp.status_code == 400
                assert "error_type" in resp.json()

    @pytest.mark.asyncio
    async def test_uncached_offline_returns_409(self, isolated_settings):
        isolated_settings.clipcc_offline = True
        outer_app = create_app(isolated_settings)
        inner_app = outer_app._app
        async with inner_app.router.lifespan_context(inner_app):
            async with AsyncClient(transport=ASGITransport(app=outer_app), base_url="http://test") as client:
                resp = await client.post(
                    "/api/v1/models/load",
                    json={"model_id": "siglip2-base-patch16-256"},
                )
                assert resp.status_code == 409
                body = resp.json()
                assert body["error_type"] == "ModelNotCachedError"

    @pytest.mark.asyncio
    async def test_insufficient_resources_returns_422(self, isolated_settings):
        outer_app = create_app(isolated_settings)
        inner_app = outer_app._app
        async with inner_app.router.lifespan_context(inner_app):
            async with AsyncClient(transport=ASGITransport(app=outer_app), base_url="http://test") as client:
                with patch("app.models.model_manager.psutil") as mock_psutil:
                    mock_psutil.virtual_memory.return_value = MagicMock(
                        total=4 * 1e9, available=4 * 1e9
                    )
                    resp = await client.post(
                        "/api/v1/models/load",
                        json={"model_id": "siglip2-giant-opt-patch16-384"},
                    )
                assert resp.status_code == 422
                body = resp.json()
                assert body["error_type"] == "InsufficientResourcesError"


# ── Contrast validation ───────────────────────────────────────────


@pytest.mark.anyio
async def test_contrast_requires_positive_and_negative_labels(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={
            "aggregation": "contrast",
            "positive_labels": json.dumps(["safe driving"]),
        },
    )
    assert r.status_code == 422
    assert "negative_labels" in r.json()["detail"]


@pytest.mark.anyio
async def test_contrast_rejects_labels_field(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={
            "aggregation": "contrast",
            "labels": json.dumps(["a", "b"]),
            "positive_labels": json.dumps(["safe"]),
            "negative_labels": json.dumps(["dangerous"]),
        },
    )
    assert r.status_code == 422


@pytest.mark.anyio
async def test_non_contrast_rejects_positive_labels(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={
            "aggregation": "mean",
            "labels": json.dumps(["a", "b"]),
            "positive_labels": json.dumps(["safe"]),
        },
    )
    assert r.status_code == 422


@pytest.mark.anyio
async def test_contrast_invalid_reduce_mode(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={
            "aggregation": "contrast",
            "positive_labels": json.dumps(["safe"]),
            "negative_labels": json.dumps(["dangerous"]),
            "contrast_reduce": "invalid_mode",
        },
    )
    assert r.status_code == 422


@pytest.mark.anyio
async def test_contrast_cross_group_duplicate_rejected(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={
            "aggregation": "contrast",
            "positive_labels": json.dumps(["driving"]),
            "negative_labels": json.dumps(["driving"]),
        },
    )
    assert r.status_code == 422
    assert "Duplicate" in r.json()["detail"] or "duplicate" in r.json()["detail"]


@pytest.mark.anyio
async def test_contrast_label_count_max_50_per_group(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={
            "aggregation": "contrast",
            "positive_labels": json.dumps([f"pos_{i}" for i in range(51)]),
            "negative_labels": json.dumps(["neg"]),
        },
    )
    assert r.status_code == 422
    assert "50" in r.json()["detail"]


@pytest.mark.anyio
async def test_classify_contrast(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={
            "aggregation": "contrast",
            "positive_labels": json.dumps(["outdoor scene", "nature"]),
            "negative_labels": json.dumps(["indoor scene"]),
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["metadata"]["aggregation"] == "contrast"
    assert "contrast" in data
    c = data["contrast"]
    assert c["verdict"] in ("positive", "negative", "uncertain")
    assert "difference" in c
    assert "threshold" in c
    assert c["threshold_was_defaulted"] is True
    assert c["threshold_source"] == "model_policy"
    assert c["calibration_status"] == "uncalibrated"
    assert c["contrast_reduce"] == "mean"
    assert c["positive"]["group"] == "positive"
    assert c["negative"]["group"] == "negative"
    assert len(c["positive"]["labels"]) == 2
    assert len(c["negative"]["labels"]) == 1
    assert c["score_semantics"] == "siglip2_pairwise_sigmoid"
    assert "best_match" in data
    assert "scores" in data
    assert len(data["scores"]) == 3


@pytest.mark.anyio
async def test_classify_contrast_with_user_threshold(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={
            "aggregation": "contrast",
            "positive_labels": json.dumps(["outdoor"]),
            "negative_labels": json.dumps(["indoor"]),
            "threshold": "0.30",
            "contrast_reduce": "top_k_mean",
        },
    )
    assert r.status_code == 200
    c = r.json()["contrast"]
    assert c["threshold"] == 0.30
    assert c["threshold_was_defaulted"] is False
    assert c["threshold_source"] == "user"
    assert c["contrast_reduce"] == "top_k_mean"
