import asyncio
import io
from unittest.mock import patch

import pytest
import torch
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import create_app
from app.models.base_model import ScoreBatch
from app.models.vlm_slot import VlmState


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _make_mock_siglip2(*args, **kwargs):
    from unittest.mock import MagicMock
    m = MagicMock()
    m.model_type = "siglip2"
    m.device = "cpu"
    m.max_token_length = 64
    m.validate_prompts = lambda prompts: [10] * len(prompts)
    m.tokenize_raw = lambda prompts: [torch.tensor([i + 1, i + 2, i + 3]) for i in range(len(prompts))]

    def _score_batch(images, texts):
        n = len(images)
        t = len(texts)
        # High, deterministic confidence so every label clears any low threshold
        conf = torch.full((n, t), 0.9)
        return ScoreBatch(confidence=conf, raw_similarity=torch.zeros(n, t),
                          logits=torch.zeros(n, t), semantics="siglip2_pairwise_sigmoid")

    m.score_batch = _score_batch
    return m


class FakeGemma:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def generate(self, frames, prompt, max_new_tokens, cancel_event):
        self.calls.append(prompt)
        return '{"verdict": "present", "explanation": "seen in the frames"}'


def make_settings(temp_dir):
    return Settings(
        allow_unauthenticated=True, max_file_size_mb=10, max_duration_seconds=30,
        max_frames=30, batch_size=4, max_concurrent_requests=2,
        ffmpeg_timeout_seconds=30, request_timeout_seconds=30,
        clip_cache_dir=str(temp_dir / "models"), temp_dir=str(temp_dir / "tmp"),
    )


async def _force_loaded(slot, fake):
    slot._ledger._device_free = lambda device: 10**12
    slot._loader = lambda: fake
    await slot.warm()
    await slot.wait_settled()
    assert slot.state == VlmState.LOADED


@pytest.fixture
async def hybrid_app(temp_dir, small_video):
    settings = make_settings(temp_dir)
    with patch("app.models.model_manager.SigLip2Model", side_effect=_make_mock_siglip2):
        app = create_app(settings)
        inner = app._app
        async with inner.router.lifespan_context(inner):
            await asyncio.sleep(0.1)  # let the SigLIP2 auto-load finish
            await _force_loaded(app.vlm_slot_for_tests, FakeGemma())
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                yield c, app


def _upload(small_video):
    return {"video": ("clip.mp4", small_video.read_bytes(), "video/mp4")}


@pytest.mark.anyio
async def test_hybrid_happy_path(hybrid_app, small_video):
    c, app = hybrid_app
    r = await c.post("/api/v1/hybrid", files=_upload(small_video),
                     data={"labels": '["texting", "eating"]', "threshold": "0.0",
                           "top_k": "2", "aggregation": "max"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["results"]) == 2
    evaluated = [x for x in body["results"] if x["gemma_evaluated"]]
    assert len(evaluated) == 2
    assert evaluated[0]["verdict"] == "present"
    assert evaluated[0]["frames_shown"]
    assert evaluated[0]["frames_shown"][0]["thumbnail"].startswith("data:image/jpeg;base64,")
    assert body["metadata"]["gemma_calls"] == 2
    assert body["metadata"]["aggregation"] == "max"


@pytest.mark.anyio
async def test_hybrid_cap_truncates_calls(hybrid_app, small_video):
    c, app = hybrid_app
    labels = '["a", "b", "c", "d"]'
    r = await c.post("/api/v1/hybrid", files=_upload(small_video),
                     data={"labels": labels, "threshold": "0.0", "max_verified_labels": "2"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["metadata"]["gemma_calls"] == 2
    assert body["metadata"]["labels_above_threshold"] == 4
    assert body["metadata"]["labels_truncated"] == 2
    assert sum(1 for x in body["results"] if x["gemma_evaluated"]) == 2


@pytest.mark.anyio
async def test_hybrid_no_label_above_threshold(hybrid_app, small_video):
    c, app = hybrid_app
    r = await c.post("/api/v1/hybrid", files=_upload(small_video),
                     data={"labels": '["x"]', "threshold": "1.0"})  # 0.9 < 1.0
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["metadata"]["gemma_calls"] == 0
    assert body["results"][0]["gemma_evaluated"] is False
    assert body["results"][0]["verdict"] is None


@pytest.mark.anyio
async def test_hybrid_rejects_bad_top_k(hybrid_app, small_video):
    c, app = hybrid_app
    r = await c.post("/api/v1/hybrid", files=_upload(small_video),
                     data={"labels": '["x"]', "top_k": "0"})
    assert r.status_code == 422
    assert "top_k" in r.json()["detail"]


@pytest.mark.anyio
async def test_hybrid_503_when_no_siglip2_model(temp_dir, small_video):
    # Gemma loaded, but SigLIP2 autoload skipped → manager has no active model.
    settings = make_settings(temp_dir)
    settings.skip_model_autoload = True
    with patch("app.models.model_manager.SigLip2Model", side_effect=_make_mock_siglip2):
        app = create_app(settings)
        inner = app._app
        async with inner.router.lifespan_context(inner):
            await _force_loaded(app.vlm_slot_for_tests, FakeGemma())
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                r = await c.post("/api/v1/hybrid", files=_upload(small_video),
                                 data={"labels": '["x"]', "threshold": "0.0"})
                assert r.status_code == 503


@pytest.mark.anyio
async def test_hybrid_page_served(hybrid_app):
    c, app = hybrid_app
    r = await c.get("/hybrid")
    assert r.status_code == 200
    assert "Hybrid" in r.text
    assert "/api/v1/hybrid" in r.text  # the page posts to the endpoint
