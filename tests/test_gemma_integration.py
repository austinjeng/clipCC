"""Real-model smoke test. Run explicitly:

    GEMMA_INTEGRATION=1 CLIP_CACHE_DIR=./models ALLOW_UNAUTHENTICATED=true \
        python -m pytest tests/test_gemma_integration.py -v -s

Requires ~12GB free memory, ~12GB disk for weights, and ffmpeg.
Exploration deliverables printed: load time, per-stage latency, parse success.
"""
import os
import subprocess
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GEMMA_INTEGRATION") != "1",
    reason="set GEMMA_INTEGRATION=1 to run the real-model smoke test",
)


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """30s test pattern video via ffmpeg."""
    path = tmp_path_factory.mktemp("vid") / "test.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=30:size=640x480:rate=10",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


@pytest.fixture(scope="module")
def gemma():
    from app.config import Settings
    from app.models.gemma_vlm import GemmaVLM

    settings = Settings(allow_unauthenticated=True)
    t0 = time.monotonic()
    model = GemmaVLM(
        hf_repo=settings.gemma_model_id,
        cache_dir=settings.clip_cache_dir,
        image_token_budget=settings.gemma_image_token_budget,
    )
    print(f"\n[gemma-integration] load time: {time.monotonic() - t0:.1f}s, device: {model.device}")
    return model


def test_librosa_not_required(gemma):
    # Audio is out of scope; the image/video path must not import librosa (spec §10)
    import sys
    assert "librosa" not in sys.modules


def test_label_scores_end_to_end(gemma, synthetic_video, tmp_path):
    from app.services.gemma_sampler import extract_frames, plan_timestamps
    from app.services.gemma_prompts import (
        build_label_scores_prompt, label_scores_token_budget, parse_label_scores,
    )
    from PIL import Image

    labels = ["colorful test pattern", "a person driving a car"]
    timestamps = plan_timestamps(0.0, 30.0, 4)
    frames = extract_frames(synthetic_video, timestamps, tmp_path, threading.Event())
    assert len(frames) == 4
    images = [Image.open(f.path).convert("RGB") for f in frames]

    t0 = time.monotonic()
    text = gemma.generate(
        images,
        build_label_scores_prompt(labels, evidence_top_k=2),
        label_scores_token_budget(len(labels)),
        threading.Event(),
    )
    gen_s = time.monotonic() - t0
    print(f"[gemma-integration] generate: {gen_s:.1f}s\n[gemma-integration] raw output: {text!r}")

    items = parse_label_scores(text, labels)  # raises ValueError if template leaks tokens
    by_label = {i.label: i.score for i in items}
    # A test pattern IS a colorful test pattern and is NOT a person driving:
    assert by_label["colorful test pattern"] is not None
    assert by_label["a person driving a car"] is not None
    assert by_label["colorful test pattern"] > by_label["a person driving a car"]


def test_qa_end_to_end(gemma, synthetic_video, tmp_path):
    from app.services.gemma_sampler import extract_frames, plan_timestamps
    from app.services.gemma_prompts import build_qa_prompt
    from PIL import Image

    frames = extract_frames(synthetic_video, plan_timestamps(0.0, 30.0, 2), tmp_path, threading.Event())
    images = [Image.open(f.path).convert("RGB") for f in frames]
    answer = gemma.generate(images, build_qa_prompt("What do these frames show?"), 200, threading.Event())
    print(f"[gemma-integration] qa answer: {answer!r}")
    assert len(answer) > 0
    assert "<|channel" not in answer  # ghost thought-channel leak check (spec §5)


def test_cancel_event_stops_generation(gemma, synthetic_video, tmp_path):
    from app.services.gemma_sampler import extract_frames, plan_timestamps
    from PIL import Image

    frames = extract_frames(synthetic_video, plan_timestamps(0.0, 30.0, 1), tmp_path, threading.Event())
    images = [Image.open(f.path).convert("RGB") for f in frames]
    ev = threading.Event()
    ev.set()  # pre-cancelled: generation must stop at the first decode-step check
    t0 = time.monotonic()
    gemma.generate(images, "Describe this in extreme detail.", 400, ev)
    elapsed = time.monotonic() - t0
    print(f"[gemma-integration] pre-cancelled generate returned in {elapsed:.1f}s (≈ prefill only)")
