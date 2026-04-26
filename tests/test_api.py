import json
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import create_app
from app.config import Settings


@pytest.fixture
def anyio_backend():
    return "asyncio"


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
    app = create_app(test_settings)
    # app is a RequestGateMiddleware; the FastAPI inner app owns the lifespan.
    # httpx 0.27 ASGITransport does not emit lifespan events, so we trigger the
    # lifespan manually via the inner app's router context.
    inner_app = app._app
    async with inner_app.router.lifespan_context(inner_app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c


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


@pytest.mark.anyio
async def test_missing_video(client):
    r = await client.post("/api/v1/classify", data={"labels": json.dumps(["a", "b", "c"])})
    assert r.status_code == 422


@pytest.mark.anyio
async def test_invalid_labels_count(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={"labels": json.dumps(["a"])},
    )
    assert r.status_code == 422
    assert "3 and 10" in r.json()["detail"]


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
