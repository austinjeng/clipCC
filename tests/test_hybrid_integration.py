"""Real-model hybrid smoke test. Run explicitly:

    GEMMA_INTEGRATION=1 CLIP_CACHE_DIR=./models ALLOW_UNAUTHENTICATED=true \
        python -m pytest tests/test_hybrid_integration.py -v -s

Requires ~12GB free memory, ~12GB disk for Gemma weights, the SigLIP2 weights,
and ffmpeg. Skipped unless GEMMA_INTEGRATION=1.
"""
import asyncio
import os
import subprocess

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import create_app
from app.models.vlm_slot import VlmState

pytestmark = pytest.mark.skipif(
    os.environ.get("GEMMA_INTEGRATION") != "1",
    reason="set GEMMA_INTEGRATION=1 to run the real-model hybrid smoke test",
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("vid") / "test.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi",
         "-i", "testsrc=duration=10:size=640x480:rate=5", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@pytest.mark.anyio
async def test_hybrid_end_to_end_real_models(synthetic_video):
    settings = Settings(allow_unauthenticated=True)
    app = create_app(settings)
    inner = app._app
    async with inner.router.lifespan_context(inner):
        # Wait for SigLIP2 auto-load.
        for _ in range(60):
            r = await _get(app, "/ready")
            if r.status_code == 200:
                break
            await asyncio.sleep(1)
        # Warm Gemma.
        slot = app.vlm_slot_for_tests
        await slot.warm()
        await slot.wait_settled()
        assert slot.state == VlmState.LOADED

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post(
                "/api/v1/hybrid",
                files={"video": ("t.mp4", synthetic_video.read_bytes(), "video/mp4")},
                data={"labels": '["a test pattern", "a person"]', "threshold": "0.0",
                      "top_k": "2", "aggregation": "max"},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        print("\n[hybrid-integration] metadata:", body["metadata"])
        assert len(body["results"]) == 2
        # gemma_calls equals the number of evaluated labels.
        evaluated = [x for x in body["results"] if x["gemma_evaluated"]]
        assert body["metadata"]["gemma_calls"] == len(evaluated)
        assert len(evaluated) >= 1, f"expected >=1 evaluated label with threshold=0.0, got 0; results={body['results']}"
        for x in evaluated:
            assert x["verdict"] in ("present", "not_present", "uncertain")
            assert x["frames_shown"]  # real frames were re-extracted
            assert x["frames_shown"][0]["thumbnail"].startswith("data:image/jpeg;base64,")


async def _get(app, path):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        return await c.get(path)
