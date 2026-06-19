import io

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import create_app
from app.models.vlm_slot import VlmState


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_settings(**overrides):
    return Settings(
        allow_unauthenticated=True,
        skip_model_autoload=True,
        temp_dir="/tmp/clipcc-gemma-test",
        **overrides,
    )


class FakeGemma:
    """Stands in for GemmaVLM: returns canned text per prompt content."""
    device = "cpu"

    def __init__(self, label_scores_reply='[{"id": 1, "score": 0.9, "evidence": "seen"}]',
                 qa_reply="The driver is texting."):
        self.label_scores_reply = label_scores_reply
        self.qa_reply = qa_reply
        self.calls: list[str] = []

    def generate(self, frames, prompt, max_new_tokens, cancel_event):
        self.calls.append(prompt)
        if "JSON array" in prompt:
            return self.label_scores_reply
        return self.qa_reply


@pytest.fixture
def app_and_slot():
    settings = make_settings()
    app = create_app(settings)
    # create_app returns the middleware; the slot is attached as a test seam
    slot = app.vlm_slot_for_tests
    return app, slot


async def force_loaded(slot, fake=None):
    slot._ledger._device_free = lambda device: 10**12  # plenty
    slot._loader = lambda: (fake or FakeGemma())
    await slot.warm()
    await slot.wait_settled()
    assert slot.state == VlmState.LOADED
    return slot.model


def tiny_upload():
    # Not a real video: probe failures are exercised separately; the happy-path
    # tests stub probe+extract below.
    return {"video": ("clip.mp4", io.BytesIO(b"\x00" * 64), "video/mp4")}


@pytest.mark.anyio
async def test_status_idle_initially(app_and_slot):
    app, slot = app_and_slot
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/v1/gemma/status")
        assert r.status_code == 200
        body = r.json()
        assert body["state"] == "idle"
        assert body["enabled"] is True
        assert body["model_id"] == "google/gemma-4-E2B-it"


@pytest.mark.anyio
async def test_warm_kicks_load_and_returns_202(app_and_slot):
    app, slot = app_and_slot
    slot._ledger._device_free = lambda device: 10**12
    slot._loader = lambda: FakeGemma()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/warm")
        assert r.status_code == 202
        await slot.wait_settled()
        r = await c.get("/api/v1/gemma/status")
        assert r.json()["state"] == "loaded"


@pytest.mark.anyio
async def test_warm_insufficient_memory_503(app_and_slot):
    app, slot = app_and_slot
    slot._ledger._device_free = lambda device: 1_000_000  # ~1MB
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/warm")
        assert r.status_code == 503
        assert "GB" in r.json()["detail"]


@pytest.mark.anyio
async def test_cold_label_scores_503_with_retry_after(app_and_slot):
    app, slot = app_and_slot
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/label_scores", files=tiny_upload(),
                         data={"labels": '["texting"]'})
        assert r.status_code == 503
        assert r.headers.get("retry-after") == "10"


@pytest.mark.anyio
async def test_label_scores_validates_label_count(app_and_slot):
    app, slot = app_and_slot
    await force_loaded(slot)
    too_many = "[" + ",".join(f'"label {i}"' for i in range(51)) + "]"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/label_scores", files=tiny_upload(),
                         data={"labels": too_many})
        assert r.status_code == 422
        assert "50" in r.json()["detail"]


@pytest.mark.anyio
async def test_qa_requires_prompt(app_and_slot):
    app, slot = app_and_slot
    await force_loaded(slot)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/qa", files=tiny_upload(), data={})
        assert r.status_code == 422


@pytest.mark.anyio
async def test_qa_prompt_length_capped(app_and_slot):
    app, slot = app_and_slot
    await force_loaded(slot)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/qa", files=tiny_upload(),
                         data={"prompt": "x" * 2001})
        assert r.status_code == 422


@pytest.mark.anyio
async def test_label_scores_happy_path_with_stubbed_video(app_and_slot, monkeypatch, tmp_path):
    app, slot = app_and_slot
    fake = FakeGemma(label_scores_reply='[{"id": 1, "score": 0.8, "evidence": "phone"}, {"id": 2, "score": 0.1}]')
    await force_loaded(slot, fake)

    import app.main as main_mod
    from app.services.video import VideoInfo
    from app.services.gemma_sampler import GemmaFrame
    from PIL import Image

    monkeypatch.setattr(main_mod, "probe_video",
                        lambda path, timeout=30: VideoInfo(duration=30.0, width=640, height=480,
                                                           video_stream_count=1, format_name="mp4"))

    def fake_extract(video_path, timestamps, frame_dir, cancel_event, ffmpeg_timeout=120, runner=None):
        frames = []
        for i, ts in enumerate(timestamps):
            p = tmp_path / f"f{i}.jpg"
            Image.new("RGB", (8, 8)).save(p)
            frames.append(GemmaFrame(path=p, timestamp_seconds=ts))
        return frames

    monkeypatch.setattr(main_mod, "gemma_extract_frames", fake_extract)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/label_scores", files=tiny_upload(),
                         data={"labels": '["texting", "sleeping"]'})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["scores"][0] == {"label": "texting", "score": 0.8, "evidence": "phone"}
        assert body["scores"][1]["score"] == 0.1
        assert body["raw_output"] == '[{"id": 1, "score": 0.8, "evidence": "phone"}, {"id": 2, "score": 0.1}]'
        md = body["metadata"]
        assert md["score_semantics"] == "gemma4_verbalized_uncalibrated"
        assert md["window_start_seconds"] == 0.0
        assert md["window_end_seconds"] == 30.0
        assert md["frames_analyzed"] == 8
        assert set(md["latency"]) == {"extract_seconds", "generate_seconds", "parse_seconds"}


@pytest.mark.anyio
@pytest.mark.parametrize("requested,expected", [(12, 12), (99, 16), (0, 1)])
async def test_label_scores_max_frames_clamped(app_and_slot, monkeypatch, tmp_path, requested, expected):
    """Per-request max_frames is honored and clamped to [1, cap]. 30s window
    keeps the 0.5s spacing floor (60) out of the way, so frames_analyzed
    reflects the clamp only."""
    app, slot = app_and_slot
    fake = FakeGemma(label_scores_reply='[{"id": 1, "score": 0.5}]')
    await force_loaded(slot, fake)

    import app.main as main_mod
    from app.services.video import VideoInfo
    from app.services.gemma_sampler import GemmaFrame
    from PIL import Image

    monkeypatch.setattr(main_mod, "probe_video",
                        lambda path, timeout=30: VideoInfo(duration=30.0, width=640, height=480,
                                                           video_stream_count=1, format_name="mp4"))

    def fake_extract(video_path, timestamps, frame_dir, cancel_event, ffmpeg_timeout=120, runner=None):
        frames = []
        for i, ts in enumerate(timestamps):
            p = tmp_path / f"f{i}.jpg"
            Image.new("RGB", (8, 8)).save(p)
            frames.append(GemmaFrame(path=p, timestamp_seconds=ts))
        return frames

    monkeypatch.setattr(main_mod, "gemma_extract_frames", fake_extract)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/label_scores", files=tiny_upload(),
                         data={"labels": '["texting"]', "max_frames": str(requested)})
        assert r.status_code == 200, r.text
        assert r.json()["metadata"]["frames_analyzed"] == expected


@pytest.mark.anyio
async def test_label_scores_retries_once_then_502(app_and_slot, monkeypatch, tmp_path):
    app, slot = app_and_slot
    fake = FakeGemma(label_scores_reply="this is not json")
    await force_loaded(slot, fake)

    import app.main as main_mod
    from app.services.video import VideoInfo
    from app.services.gemma_sampler import GemmaFrame
    from PIL import Image

    monkeypatch.setattr(main_mod, "probe_video",
                        lambda path, timeout=30: VideoInfo(duration=10.0, width=640, height=480,
                                                           video_stream_count=1, format_name="mp4"))

    def fake_extract(video_path, timestamps, frame_dir, cancel_event, ffmpeg_timeout=120, runner=None):
        p = tmp_path / "f.jpg"
        Image.new("RGB", (8, 8)).save(p)
        return [GemmaFrame(path=p, timestamp_seconds=t) for t in timestamps]

    monkeypatch.setattr(main_mod, "gemma_extract_frames", fake_extract)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/label_scores", files=tiny_upload(),
                         data={"labels": '["texting"]'})
        assert r.status_code == 502
        # generate called twice: initial + one bounded retry
        assert len([p for p in fake.calls if "JSON array" in p]) == 2


@pytest.mark.anyio
async def test_qa_happy_path(app_and_slot, monkeypatch, tmp_path):
    app, slot = app_and_slot
    await force_loaded(slot)

    import app.main as main_mod
    from app.services.video import VideoInfo
    from app.services.gemma_sampler import GemmaFrame
    from PIL import Image

    monkeypatch.setattr(main_mod, "probe_video",
                        lambda path, timeout=30: VideoInfo(duration=10.0, width=640, height=480,
                                                           video_stream_count=1, format_name="mp4"))

    def fake_extract(video_path, timestamps, frame_dir, cancel_event, ffmpeg_timeout=120, runner=None):
        p = tmp_path / "f.jpg"
        Image.new("RGB", (8, 8)).save(p)
        return [GemmaFrame(path=p, timestamp_seconds=t) for t in timestamps]

    monkeypatch.setattr(main_mod, "gemma_extract_frames", fake_extract)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/qa", files=tiny_upload(),
                         data={"prompt": "what is happening?"})
        assert r.status_code == 200, r.text
        assert r.json()["answer"] == "The driver is texting."


@pytest.mark.anyio
async def test_gemma_page_served(app_and_slot):
    app, slot = app_and_slot
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/gemma")
        assert r.status_code == 200
        assert "Gemma" in r.text


@pytest.mark.anyio
async def test_status_exposes_default_label_instruction(app_and_slot):
    app, slot = app_and_slot
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/v1/gemma/status")
        assert "You are analyzing frames" in r.json()["default_label_instruction"]


@pytest.mark.anyio
async def test_status_exposes_label_scores_contract(app_and_slot):
    app, slot = app_and_slot
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/v1/gemma/status")
        contract = r.json()["label_scores_contract"]
        assert "Respond with ONLY a JSON array" in contract
        assert "Include every id exactly once." in contract


@pytest.mark.anyio
async def test_label_scores_instruction_length_capped(app_and_slot):
    app, slot = app_and_slot
    await force_loaded(slot)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/label_scores", files=tiny_upload(),
                         data={"labels": '["texting"]', "instruction": "x" * 2001})
        assert r.status_code == 422


@pytest.mark.anyio
async def test_label_scores_custom_instruction_reaches_model(app_and_slot, monkeypatch, tmp_path):
    app, slot = app_and_slot
    fake = FakeGemma(label_scores_reply='[{"id": 1, "score": 0.5}]')
    await force_loaded(slot, fake)

    import app.main as main_mod
    from app.services.video import VideoInfo
    from app.services.gemma_sampler import GemmaFrame
    from PIL import Image

    monkeypatch.setattr(main_mod, "probe_video",
                        lambda path, timeout=30: VideoInfo(duration=10.0, width=640, height=480,
                                                           video_stream_count=1, format_name="mp4"))

    def fake_extract(video_path, timestamps, frame_dir, cancel_event, ffmpeg_timeout=120, runner=None):
        p = tmp_path / "f.jpg"
        Image.new("RGB", (8, 8)).save(p)
        return [GemmaFrame(path=p, timestamp_seconds=t) for t in timestamps]

    monkeypatch.setattr(main_mod, "gemma_extract_frames", fake_extract)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/label_scores", files=tiny_upload(),
                         data={"labels": '["texting"]', "instruction": "Custom: rate harshly."})
        assert r.status_code == 200, r.text
    assert any("Custom: rate harshly." in p for p in fake.calls)
