# Hybrid Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Hybrid mode that scores frames with SigLIP2, gates labels by peak score, then re-extracts each gated label's top-k frames at Gemma resolution and asks Gemma 4 E2B for a `present`/`not_present`/`uncertain` verdict + explanation.

**Architecture:** New `POST /api/v1/hybrid` route runs two sequential phases against one shared deadline — SigLIP2 scoring (under the inference gate + model lease) then per-label Gemma verdicts (under the VLM gate). Frame files from the SigLIP2 phase are deleted as scored (existing behavior); Gemma frames are freshly re-extracted at 896px from the selected timestamps and inlined as base64 thumbnails. A new standalone `/hybrid` page drives it.

**Tech Stack:** FastAPI, PyTorch, transformers, Pillow, anyio, pytest + pytest-asyncio, ffmpeg. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-06-17-hybrid-mode-design.md`

## Global Constraints

- No new dependencies — Pillow, torch, transformers, anyio are already present.
- Error responses use the `{"detail": "<string>"}` shape (the app flattens FastAPI validation errors to this).
- Labels: 1..`gemma_max_labels` (50), each non-empty, ≤200 chars, unique (reuse `_validate_label_group`).
- Gemma frame cap: `gemma_max_frames_cap` = 16. `top_k` validated to `1..16`.
- Gemma wrapper already sets `enable_thinking=False`; do not change it.
- Frame spacing floor for spread selection: 0.5s (matches `gemma_sampler._MIN_SPACING_SECONDS`).
- Two phases never hold both resource gates at once; each phase's `InferenceRunner` gets the *remaining* time to the shared deadline.

## File Structure

- Create `app/schemas/hybrid.py` — response models (`HybridFrameRef`, `HybridLabelResult`, `HybridLatency`, `HybridMetadata`, `HybridResponse`) + semantics/disclaimer constants.
- Modify `app/config.py` — add `gemma_max_new_tokens_verdict`, `hybrid_max_verified_labels`, `hybrid_thumbnail_px`.
- Create `app/services/hybrid_select.py` — pure selection + gating + thumbnail helper (`per_label_scores`, `gate_and_rank_labels`, `select_topk_spread`, `thumbnail_data_uri`).
- Modify `app/services/gemma_prompts.py` — add `build_verdict_prompt`, `parse_verdict`, `VERDICT_LITERALS`, `DEFAULT_VERDICT_INSTRUCTION`.
- Modify `app/middleware.py` — gate `/api/v1/hybrid` like a VLM upload route (auth → VLM-state → upload concurrency → body size).
- Modify `app/main.py` — imports, `_run_hybrid_pipeline` helper, `POST /api/v1/hybrid` route, `GET /hybrid` static route.
- Create `app/static/hybrid.html` — standalone page.
- Modify `app/static/index.html` + `app/static/gemma.html` — add `Hybrid` nav link.
- Tests: `tests/test_hybrid_select.py`, `tests/test_gemma_verdict.py`, `tests/test_hybrid_middleware.py`, `tests/test_hybrid_api.py`, `tests/test_hybrid_integration.py` (gated).

---

### Task 1: Schemas + config

**Files:**
- Create: `app/schemas/hybrid.py`
- Modify: `app/config.py:38-41` (add three settings near the other gemma settings)
- Test: `tests/test_hybrid_select.py` (schema defaults checked here too — first file in the suite)

**Interfaces:**
- Produces: `HybridResponse`, `HybridLabelResult`, `HybridFrameRef`, `HybridLatency`, `HybridMetadata`; constants `SIGLIP2_SCORE_SEMANTICS`, `HYBRID_DISCLAIMER`. Settings: `gemma_max_new_tokens_verdict: int = 160`, `hybrid_max_verified_labels: int = 6`, `hybrid_thumbnail_px: int = 160`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hybrid_select.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hybrid_select.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.schemas.hybrid'`

- [ ] **Step 3: Create the schema module**

Create `app/schemas/hybrid.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.schemas.gemma import GEMMA_DISCLAIMER, GEMMA_SCORE_SEMANTICS

SIGLIP2_SCORE_SEMANTICS = "siglip2_pairwise_sigmoid"

HYBRID_DISCLAIMER = (
    "SigLIP2 scores are pairwise sigmoid; Gemma verdicts are an independent "
    "generative second opinion (uncalibrated). They are not the same scale and "
    "must not be compared as numbers. Not suitable for safety-critical decisions."
)


class HybridFrameRef(BaseModel):
    frame_index: int
    timestamp_seconds: float
    score: float
    thumbnail: str  # data:image/jpeg;base64,...


class HybridLabelResult(BaseModel):
    label: str
    siglip2_score: float
    gemma_evaluated: bool  # Gemma was run on this label — NOT "confirmed present"
    verdict: Literal["present", "not_present", "uncertain"] | None = None
    explanation: str | None = None
    parse_failed: bool = False
    frames_shown: list[HybridFrameRef] = []


class HybridLatency(BaseModel):
    siglip2_seconds: float
    gemma_seconds: float


class HybridMetadata(BaseModel):
    siglip2_model: str
    gemma_model: str
    device: str
    frames_analyzed: int
    video_duration_seconds: float
    aggregation: str
    threshold: float
    top_k: int
    max_verified_labels: int
    labels_above_threshold: int
    labels_truncated: int
    gemma_calls: int
    siglip2_score_semantics: str = SIGLIP2_SCORE_SEMANTICS
    gemma_score_semantics: str = GEMMA_SCORE_SEMANTICS
    disclaimer: str = HYBRID_DISCLAIMER
    latency: HybridLatency


class HybridResponse(BaseModel):
    results: list[HybridLabelResult]
    metadata: HybridMetadata
```

- [ ] **Step 4: Add config settings**

In `app/config.py`, after line 38 (`gemma_evidence_top_k: int = 3`), add:

```python
    gemma_max_new_tokens_verdict: int = 160
    hybrid_max_verified_labels: int = 6
    hybrid_thumbnail_px: int = 160
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_hybrid_select.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add app/schemas/hybrid.py app/config.py tests/test_hybrid_select.py
git commit -m "feat(hybrid): response schemas + config settings"
```

---

### Task 2: Pure selection + gating + thumbnail helpers

**Files:**
- Create: `app/services/hybrid_select.py`
- Test: `tests/test_hybrid_select.py` (append)

**Interfaces:**
- Consumes: `ScoringContext` (`app/services/scoring.py`) with `.confidence` tensor `(num_frames, num_labels)`, `.labels`, `.frames[i].approx_timestamp_seconds`.
- Produces:
  - `FrameRef(frame_index: int, timestamp_seconds: float, score: float)` dataclass
  - `GatedLabel(label: str, label_idx: int, score: float)` dataclass
  - `per_label_scores(ctx, aggregation: str) -> list[float]`
  - `gate_and_rank_labels(ctx, scores: list[float], threshold: float, max_verified_labels: int) -> tuple[list[GatedLabel], int, int]` → (selected, n_above_threshold, n_truncated)
  - `select_topk_spread(ctx, label_idx: int, k: int, min_gap_seconds: float = 0.5) -> list[FrameRef]` (ranked best-first)
  - `thumbnail_data_uri(path, max_px: int) -> str`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hybrid_select.py`:

```python
import torch
from dataclasses import dataclass

from app.services.hybrid_select import (
    per_label_scores, gate_and_rank_labels, select_topk_spread, FrameRef,
)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hybrid_select.py -k "spread or gate or per_label" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.hybrid_select'`

- [ ] **Step 3: Create the module**

Create `app/services/hybrid_select.py`:

```python
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.services.scoring import ScoringContext


@dataclass
class FrameRef:
    frame_index: int
    timestamp_seconds: float
    score: float


@dataclass
class GatedLabel:
    label: str
    label_idx: int
    score: float


def per_label_scores(ctx: ScoringContext, aggregation: str) -> list[float]:
    """One score per label used for gating and display."""
    if aggregation == "max":
        vals = ctx.confidence.max(dim=0).values
    elif aggregation == "mean":
        vals = ctx.confidence.mean(dim=0)
    else:
        raise ValueError(f"hybrid aggregation must be 'max' or 'mean', got {aggregation!r}")
    return [round(v.item(), 6) for v in vals]


def gate_and_rank_labels(
    ctx: ScoringContext, scores: list[float], threshold: float, max_verified_labels: int
) -> tuple[list[GatedLabel], int, int]:
    """Keep labels at/above threshold, ranked by score, capped to max_verified_labels.
    Returns (selected, n_above_threshold, n_truncated)."""
    above = [
        GatedLabel(label=ctx.labels[i], label_idx=i, score=scores[i])
        for i in range(len(ctx.labels))
        if scores[i] >= threshold
    ]
    above.sort(key=lambda g: g.score, reverse=True)
    selected = above[:max_verified_labels]
    return selected, len(above), len(above) - len(selected)


def select_topk_spread(
    ctx: ScoringContext, label_idx: int, k: int, min_gap_seconds: float = 0.5
) -> list[FrameRef]:
    """Top-k frames for one label by score, rejecting any frame within
    min_gap_seconds of an already-picked frame. Returned ranked best-first;
    the caller sorts chronologically before sending to Gemma."""
    col = ctx.confidence[:, label_idx]
    order = col.argsort(descending=True).tolist()
    picked: list[FrameRef] = []
    for fi in order:
        if len(picked) >= k:
            break
        ts = ctx.frames[fi].approx_timestamp_seconds
        if all(abs(ts - p.timestamp_seconds) >= min_gap_seconds for p in picked):
            picked.append(FrameRef(frame_index=fi, timestamp_seconds=ts, score=round(col[fi].item(), 6)))
    return picked


def thumbnail_data_uri(path: Path, max_px: int) -> str:
    """Downscale a frame to a small base64 JPEG data URI for inline display."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hybrid_select.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add app/services/hybrid_select.py tests/test_hybrid_select.py
git commit -m "feat(hybrid): pure top-k-spread selection, gating, thumbnail helper"
```

---

### Task 3: Gemma verdict prompt + parser

**Files:**
- Modify: `app/services/gemma_prompts.py` (append; reuses module-level `_FENCE_RE`, `json`)
- Test: `tests/test_gemma_verdict.py`

**Interfaces:**
- Produces: `VERDICT_LITERALS: tuple[str, str, str]`, `DEFAULT_VERDICT_INSTRUCTION: str`, `build_verdict_prompt(label, instruction=None) -> str`, `parse_verdict(text) -> dict` (keys `verdict`, `explanation`; raises `ValueError` on any violation).

- [ ] **Step 1: Write the failing test**

Create `tests/test_gemma_verdict.py`:

```python
import pytest

from app.services.gemma_prompts import build_verdict_prompt, parse_verdict, VERDICT_LITERALS


def test_prompt_names_label_and_demands_json_object():
    p = build_verdict_prompt("texting while driving")
    assert "texting while driving" in p
    assert "JSON object" in p
    assert '"verdict"' in p and '"explanation"' in p


def test_prompt_uses_custom_instruction():
    p = build_verdict_prompt("eating", instruction="Look only at the hands.")
    assert "Look only at the hands." in p


def test_parse_valid_object():
    out = parse_verdict('{"verdict": "present", "explanation": "phone in hand"}')
    assert out == {"verdict": "present", "explanation": "phone in hand"}


def test_parse_tolerates_prose_and_fences():
    out = parse_verdict('Sure!\n```json\n{"verdict":"not_present","explanation":"nothing seen"}\n```')
    assert out["verdict"] == "not_present"


def test_parse_rejects_unknown_verdict():
    with pytest.raises(ValueError):
        parse_verdict('{"verdict": "maybe", "explanation": "x"}')


def test_parse_rejects_missing_explanation():
    with pytest.raises(ValueError):
        parse_verdict('{"verdict": "present"}')


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_verdict("not json at all")


def test_literals_are_the_three_values():
    assert VERDICT_LITERALS == ("present", "not_present", "uncertain")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gemma_verdict.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_verdict_prompt'`

- [ ] **Step 3: Append to `app/services/gemma_prompts.py`**

Add at the end of the file:

```python
VERDICT_LITERALS = ("present", "not_present", "uncertain")

DEFAULT_VERDICT_INSTRUCTION = (
    "You are analyzing frames sampled from a video, in chronological order.\n"
    "Decide whether the following behavior is visibly happening in these frames."
)


def build_verdict_prompt(label: str, instruction: str | None = None) -> str:
    instr = (instruction or "").strip() or DEFAULT_VERDICT_INSTRUCTION
    return (
        f"{instr}\n\n"
        f"Behavior: {label}\n\n"
        "Respond with ONLY a JSON object, no other text:\n"
        '{"verdict": "present" | "not_present" | "uncertain", "explanation": "<one short sentence>"}\n'
        'Use "present" only if you clearly see it, "not_present" if you clearly do not, '
        '"uncertain" if the frames are ambiguous. The explanation must be one short sentence.'
    )


def parse_verdict(text: str) -> dict:
    """Strict parse: tolerate prose/fences, require a JSON object with a valid
    verdict literal and a non-empty explanation string. Raises ValueError otherwise."""
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    verdict = data.get("verdict")
    if verdict not in VERDICT_LITERALS:
        raise ValueError(f"verdict must be one of {VERDICT_LITERALS}, got {verdict!r}")
    explanation = data.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("explanation must be a non-empty string")
    return {"verdict": verdict, "explanation": explanation.strip()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gemma_verdict.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add app/services/gemma_prompts.py tests/test_gemma_verdict.py
git commit -m "feat(hybrid): gemma verdict prompt builder + strict parser"
```

---

### Task 4: Middleware route policy for `/api/v1/hybrid`

**Files:**
- Modify: `app/middleware.py:120-152` (extend the classify/gemma-upload branch)
- Test: `tests/test_hybrid_middleware.py`

**Interfaces:**
- Consumes: existing `RequestGateMiddleware(app, gates, api_key, max_body_bytes, vlm_state)`.
- Produces: `/api/v1/hybrid` now requires auth, fails fast `503` when `vlm_state() != "loaded"` (before body drain), and passes through the upload-concurrency + body-size gates.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hybrid_middleware.py`:

```python
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from app.middleware import RequestGateMiddleware
from app.resource_gates import ResourceGates


async def echo_app(request):
    body = await request.body()
    return JSONResponse({"size": len(body)})


def make_app(api_key=None, max_body=1024, vlm="loaded"):
    gates = ResourceGates(max_upload_concurrency=2, max_inference_concurrency=1)
    app = Starlette(routes=[Route("/api/v1/hybrid", echo_app, methods=["POST"])])
    return RequestGateMiddleware(
        app, gates=gates, api_key=api_key, max_body_bytes=max_body, vlm_state=lambda: vlm
    )


@pytest.mark.anyio
async def test_hybrid_requires_auth():
    app = make_app(api_key="secret")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/hybrid", content=b"data")
        assert r.status_code == 401


@pytest.mark.anyio
async def test_hybrid_cold_gemma_503_before_body():
    app = make_app(vlm="idle")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/hybrid", content=b"data")
        assert r.status_code == 503
        assert r.headers.get("retry-after") == "10"


@pytest.mark.anyio
async def test_hybrid_oversized_body_413():
    app = make_app(max_body=100, vlm="loaded")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/hybrid", content=b"x" * 200)
        assert r.status_code == 413


@pytest.mark.anyio
async def test_hybrid_loaded_passes_through():
    app = make_app(vlm="loaded")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/hybrid", content=b"hello")
        assert r.status_code == 200
        assert r.json()["size"] == 5
```

This test file needs `anyio_backend`. Add to the top of the file:

```python
@pytest.fixture
def anyio_backend():
    return "asyncio"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hybrid_middleware.py -v`
Expected: FAIL — `test_hybrid_cold_gemma_503_before_body` returns 200 (path falls through the `:235` pass-through), `test_hybrid_oversized_body_413` returns 200.

- [ ] **Step 3: Extend the middleware branch**

In `app/middleware.py`, find line 121:

```python
        is_gemma_upload = path in GEMMA_UPLOAD_PATHS
```

Replace with:

```python
        # /api/v1/hybrid runs SigLIP2 then Gemma — gate it like a VLM upload route
        # (it needs Gemma loaded before we drain a 500MB body).
        is_vlm_upload = path in GEMMA_UPLOAD_PATHS or path == "/api/v1/hybrid"
```

Then on line 123 change:

```python
        if path == "/api/v1/classify" or is_gemma_upload:
```

to:

```python
        if path == "/api/v1/classify" or is_vlm_upload:
```

And on line 132 change:

```python
            if is_gemma_upload:
```

to:

```python
            if is_vlm_upload:
```

(There are exactly two `is_gemma_upload` references after the assignment; both become `is_vlm_upload`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hybrid_middleware.py tests/test_middleware.py -v`
Expected: PASS (new file passes; existing middleware tests still pass)

- [ ] **Step 5: Commit**

```bash
git add app/middleware.py tests/test_hybrid_middleware.py
git commit -m "feat(hybrid): gate /api/v1/hybrid like a VLM upload route"
```

---

### Task 5: `/api/v1/hybrid` route + pipeline + `/hybrid` static route

**Files:**
- Modify: `app/main.py` (imports near lines 48-84; new helper + routes after the gemma routes, before `classify` at line 573; `GET /hybrid` near the `GET /gemma` route at line 374)
- Test: `tests/test_hybrid_api.py`

**Interfaces:**
- Consumes: `select_topk_spread`, `gate_and_rank_labels`, `per_label_scores`, `thumbnail_data_uri` (Task 2); `build_verdict_prompt`, `parse_verdict` (Task 3); `HybridResponse` etc. (Task 1); existing `ScoringContext`, `FrameExtractor`, `gemma_extract_frames`, `InferenceRunner`, `manager.acquire`, `gates`, `vlm_slot`, `temp_store`.
- Produces: `POST /api/v1/hybrid` → `HybridResponse`; `GET /hybrid` → `hybrid.html`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hybrid_api.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hybrid_api.py -v`
Expected: FAIL with `404 Not Found` (route not registered yet).

- [ ] **Step 3: Add imports to `app/main.py`**

After the existing `from app.schemas.gemma import (...)` block (ends line 54), add:

```python
from app.schemas.hybrid import (
    HybridFrameRef,
    HybridLabelResult,
    HybridLatency,
    HybridMetadata,
    HybridResponse,
)
```

After the `from app.services.gemma_prompts import (...)` block (ends line 62), add:

```python
from app.services.gemma_prompts import build_verdict_prompt, parse_verdict
```

After `from app.services.scoring import aggregate_frame_scores, VALID_CONTRAST_REDUCTIONS` (line 81), add:

```python
from app.services.scoring import ScoringContext
from app.services.hybrid_select import (
    gate_and_rank_labels,
    per_label_scores,
    select_topk_spread,
    thumbnail_data_uri,
)
```

- [ ] **Step 4: Add the `GET /hybrid` static route**

In `app/main.py`, after the `serve_gemma_ui` route (lines 374-376), add:

```python
    @app.get("/hybrid")
    async def serve_hybrid_ui():
        return FileResponse(STATIC_DIR / "hybrid.html")
```

- [ ] **Step 5: Add the hybrid route**

In `app/main.py`, immediately before `@app.post("/api/v1/classify", ...)` (line 573), add:

```python
    @app.post("/api/v1/hybrid", response_model=HybridResponse)
    async def hybrid(
        video: UploadFile,
        labels: str = Form(),
        fps: float = Form(default=1.0),
        aggregation: str = Form(default="max"),
        threshold: float = Form(default=0.5, ge=0.0, le=1.0),
        top_k: int = Form(default=3),
        max_verified_labels: int | None = Form(default=None),
        instruction: str = Form(default=""),
    ):
        manager: Optional[ModelManager] = state.get("manager")
        temp_store: TempStore = state["temp_store"]
        gates: ResourceGates = state["gates"]

        # --- validation (before any work) ---
        ext = Path(video.filename or "").suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(ext if ext else (video.filename or ""))
        if vlm_slot.state != VlmState.LOADED:
            raise HTTPException(
                status_code=503,
                detail="Gemma model is not loaded. Trigger loading via POST /api/v1/gemma/warm.",
            )
        if fps < 0.1 or fps > 5.0:
            raise InvalidFpsError(fps)
        if aggregation not in ("max", "mean"):
            raise InvalidAggregationError(aggregation)
        if top_k < 1 or top_k > settings.gemma_max_frames_cap:
            raise InvalidGemmaParamsError(
                f"top_k must be between 1 and {settings.gemma_max_frames_cap}."
            )
        cap = settings.hybrid_max_verified_labels if max_verified_labels is None else max_verified_labels
        if cap < 1:
            raise InvalidGemmaParamsError("max_verified_labels must be >= 1.")
        parsed_labels = _parse_label_array(labels, "labels")
        _validate_label_group(parsed_labels, "labels", max_count=settings.gemma_max_labels)
        if len(instruction) > 2000:
            raise InvalidGemmaParamsError("instruction must be 2000 characters or fewer.")

        prompts = [f"This is a photo of {lb}." for lb in parsed_labels]
        request_id = str(uuid.uuid4())
        deadline = time.monotonic() + settings.request_timeout_seconds
        t_start = time.monotonic()

        def _remaining() -> int:
            return max(1, int(deadline - time.monotonic()))

        try:
            async with manager.acquire(timeout=settings.request_timeout_seconds) as lease:
                model = lease.model
                siglip2_model_id = manager.active_model_id or ""
                device = model.device

                token_counts = model.validate_prompts(prompts)
                for p, count in zip(prompts, token_counts):
                    if count > model.max_token_length:
                        raise TokenTruncationError(p, count)

                stored = await anyio.to_thread.run_sync(
                    temp_store.save_upload, request_id, video.file
                )
                video_info = await anyio.to_thread.run_sync(
                    partial(probe_video, stored.path, timeout=settings.ffmpeg_timeout_seconds)
                )
                validate_video_constraints(video_info, settings, fps)

                # --- Phase 1: SigLIP2 scores every frame (frames deleted as scored) ---
                async with gates.inference_admission():
                    runner_s = InferenceRunner(timeout_seconds=_remaining())

                    def score_pipeline(cancel_event, runner_ref):
                        frame_dir = temp_store.create_frame_dir(request_id)
                        extractor = FrameExtractor(ffmpeg_timeout=settings.ffmpeg_timeout_seconds)
                        frame_samples = extractor.extract(
                            video_path=stored.path, fps=fps, max_frames=settings.max_frames,
                            frame_dir=frame_dir, cancel_event=cancel_event, runner=runner_ref,
                        )
                        all_batches = []
                        all_frames = []
                        for bs in range(0, len(frame_samples), settings.batch_size):
                            if cancel_event.is_set():
                                break
                            batch = frame_samples[bs: bs + settings.batch_size]
                            images = [Image.open(fs.path).convert("RGB") for fs in batch]
                            all_batches.append(model.score_batch(images, prompts))
                            all_frames.extend(batch)
                            for fs in batch:
                                try:
                                    fs.path.unlink(missing_ok=True)
                                except Exception:
                                    pass
                        return all_batches, all_frames

                    score_result = await runner_s.run(score_pipeline)
                    if score_result is None:
                        raise InferenceTimeoutError(settings.request_timeout_seconds)
                    all_batches, all_frames = score_result
                    if not all_frames:
                        raise NoFramesExtractedError()
                siglip2_seconds = round(time.monotonic() - t_start, 3)

            # SigLIP2 lease released. Build context + selection (cheap, main thread).
            ctx = ScoringContext.from_batches(all_batches, parsed_labels, all_frames)
            scores = per_label_scores(ctx, aggregation)
            selected, n_above, n_trunc = gate_and_rank_labels(ctx, scores, threshold, cap)
            label_refs: dict[str, list] = {}
            for g in selected:
                refs = select_topk_spread(ctx, g.label_idx, top_k)
                refs.sort(key=lambda r: r.timestamp_seconds)  # chronological for Gemma
                label_refs[g.label] = refs

            # --- Phase 2: per-label Gemma verdicts (re-extract frames at 896px) ---
            gemma_out: dict = {}
            t_gemma = time.monotonic()
            if selected:
                async with gates.vlm_admission():
                    runner_g = InferenceRunner(timeout_seconds=_remaining())
                    gmodel = vlm_slot.model

                    def verdict_pipeline(cancel_event, runner_ref):
                        out: dict = {}
                        frame_dir = temp_store.create_frame_dir(request_id)
                        for g in selected:
                            if cancel_event.is_set():
                                break
                            refs = label_refs[g.label]
                            timestamps = [r.timestamp_seconds for r in refs]
                            gframes = gemma_extract_frames(
                                stored.path, timestamps, frame_dir, cancel_event,
                                ffmpeg_timeout=settings.ffmpeg_timeout_seconds, runner=runner_ref,
                            )
                            images = [Image.open(f.path).convert("RGB") for f in gframes]
                            shown = [
                                HybridFrameRef(
                                    frame_index=ref.frame_index,
                                    timestamp_seconds=ref.timestamp_seconds,
                                    score=ref.score,
                                    thumbnail=thumbnail_data_uri(f.path, settings.hybrid_thumbnail_px),
                                )
                                for ref, f in zip(refs, gframes)
                            ]
                            prompt = build_verdict_prompt(g.label, instruction=instruction or None)
                            verdict = explanation = None
                            parse_failed = False
                            for _ in range(2):
                                if cancel_event.is_set():
                                    break
                                text = gmodel.generate(
                                    images, prompt, settings.gemma_max_new_tokens_verdict, cancel_event
                                )
                                try:
                                    parsed = parse_verdict(text)
                                    verdict, explanation = parsed["verdict"], parsed["explanation"]
                                    parse_failed = False
                                    break
                                except ValueError:
                                    parse_failed = True
                            if verdict is None:
                                verdict, explanation, parse_failed = (
                                    "uncertain", "(could not parse model output)", True
                                )
                            out[g.label] = (verdict, explanation, parse_failed, shown)
                        return out

                    gemma_out = await runner_g.run(verdict_pipeline)
                    if gemma_out is None:
                        raise InferenceTimeoutError(settings.request_timeout_seconds)
            gemma_seconds = round(time.monotonic() - t_gemma, 3)

            results = []
            for i, label in enumerate(parsed_labels):
                if label in gemma_out:
                    verdict, explanation, parse_failed, shown = gemma_out[label]
                    results.append(HybridLabelResult(
                        label=label, siglip2_score=scores[i], gemma_evaluated=True,
                        verdict=verdict, explanation=explanation, parse_failed=parse_failed,
                        frames_shown=shown,
                    ))
                else:
                    results.append(HybridLabelResult(
                        label=label, siglip2_score=scores[i], gemma_evaluated=False,
                    ))

            return HybridResponse(
                results=results,
                metadata=HybridMetadata(
                    siglip2_model=siglip2_model_id, gemma_model=settings.gemma_model_id,
                    device=device, frames_analyzed=len(all_frames),
                    video_duration_seconds=video_info.duration, aggregation=aggregation,
                    threshold=threshold, top_k=top_k, max_verified_labels=cap,
                    labels_above_threshold=n_above, labels_truncated=n_trunc,
                    gemma_calls=len(gemma_out),
                    latency=HybridLatency(siglip2_seconds=siglip2_seconds, gemma_seconds=gemma_seconds),
                ),
            )
        except NoModelLoadedError:
            return JSONResponse(
                status_code=503,
                content={"detail": "No model loaded. Load a model first via POST /api/v1/models/load."},
            )
        finally:
            temp_store.cleanup(request_id)
```

- [ ] **Step 6: Create a placeholder `hybrid.html` so `GET /hybrid` resolves**

The route test does not hit `/hybrid`, but create a minimal file now so the app never 500s on it (Task 6 replaces it):

```bash
printf '<!doctype html><title>Hybrid</title><h1>Hybrid (placeholder)</h1>\n' > app/static/hybrid.html
```

- [ ] **Step 7: Run the route tests**

Run: `python -m pytest tests/test_hybrid_api.py -v`
Expected: PASS (5 passed). If `test_hybrid_happy_path` fails on frame extraction, confirm `small_video` fixture exists in `tests/conftest.py` (it does) and ffmpeg is installed.

- [ ] **Step 8: Run the regression-sensitive suites**

Run: `python -m pytest tests/test_api.py tests/test_gemma_api.py tests/test_middleware.py -v`
Expected: PASS (no regressions from the new imports/route).

- [ ] **Step 9: Commit**

```bash
git add app/main.py app/static/hybrid.html tests/test_hybrid_api.py
git commit -m "feat(hybrid): /api/v1/hybrid route + two-phase pipeline + /hybrid page route"
```

---

### Task 6: Hybrid web UI + nav links

**Files:**
- Create (overwrite placeholder): `app/static/hybrid.html`
- Modify: `app/static/index.html:358` (add nav link)
- Modify: `app/static/gemma.html:97` (add nav link)
- Test: `tests/test_hybrid_api.py` (append a UI smoke test)

**Interfaces:**
- Consumes: `GET /api/v1/models`, `POST /api/v1/models/load`, `GET /api/v1/gemma/status`, `POST /api/v1/gemma/warm`, `POST /api/v1/hybrid`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hybrid_api.py`:

```python
@pytest.mark.anyio
async def test_hybrid_page_served(hybrid_app):
    c, app = hybrid_app
    r = await c.get("/hybrid")
    assert r.status_code == 200
    assert "Hybrid" in r.text
    assert "/api/v1/hybrid" in r.text  # the page posts to the endpoint
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hybrid_api.py::test_hybrid_page_served -v`
Expected: FAIL (placeholder lacks `/api/v1/hybrid`).

- [ ] **Step 3: Write `app/static/hybrid.html`**

Overwrite `app/static/hybrid.html` with:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>clipCC — Hybrid</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 880px; margin: 24px auto; padding: 0 16px; color: #1a1a1a; }
    nav { display: flex; align-items: center; margin-bottom: 16px; border-bottom: 1px solid #ddd; padding-bottom: 8px; }
    nav a { margin-right: 16px; text-decoration: none; color: #0366d6; }
    nav a.active { font-weight: bold; color: #1a1a1a; }
    .card { border: 1px solid #e1e4e8; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
    label { display: block; font-size: 13px; font-weight: 600; margin: 8px 0 4px; }
    input, select, textarea { width: 100%; box-sizing: border-box; padding: 6px; font-size: 14px; }
    .row { display: flex; gap: 12px; } .row > div { flex: 1; }
    button { padding: 8px 14px; font-size: 14px; cursor: pointer; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; color: #fff; }
    .pill.loaded { background: #28a745; } .pill.idle, .pill.loading { background: #6a737d; } .pill.failed { background: #d73a49; }
    .result { border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin-bottom: 10px; }
    .result.skipped { opacity: 0.55; }
    .bar { height: 8px; background: #eee; border-radius: 4px; overflow: hidden; }
    .bar > span { display: block; height: 100%; background: #0366d6; }
    .verdict { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; color: #fff; }
    .verdict.present { background: #28a745; } .verdict.not_present { background: #d73a49; } .verdict.uncertain { background: #6a737d; }
    .thumbs img { height: 72px; margin: 4px 4px 0 0; border-radius: 4px; vertical-align: top; }
    .muted { color: #6a737d; font-size: 12px; }
    #error { color: #d73a49; }
  </style>
</head>
<body>
  <nav>
    <a href="/">SigLIP2</a>
    <a href="/gemma">Gemma 4</a>
    <a href="/hybrid" class="active">Hybrid</a>
  </nav>

  <h1>Hybrid: SigLIP2 → top-k → Gemma</h1>
  <p class="muted">SigLIP2 scores every frame; labels above the threshold get their top-k frames verified by Gemma.</p>

  <div class="card">
    <strong>SigLIP2 model:</strong>
    <div class="row">
      <div><select id="modelSelect"></select></div>
      <div style="flex:0"><button id="loadBtn">Load</button></div>
    </div>
    <div class="muted" id="modelStatus">—</div>
  </div>

  <div class="card">
    <strong>Gemma:</strong> <span id="gemmaPill" class="pill idle">idle</span>
    <button id="warmBtn">Warm model</button>
  </div>

  <div class="card">
    <label>Video</label>
    <input type="file" id="video" accept="video/*" />
    <label>Labels (one per line)</label>
    <textarea id="labels" rows="3">texting while driving
eating while driving
sleeping while driving</textarea>
    <div class="row">
      <div><label>Aggregation</label>
        <select id="aggregation"><option value="max">max (did it occur?)</option><option value="mean">mean</option></select></div>
      <div><label>Threshold</label><input type="number" id="threshold" value="0.5" min="0" max="1" step="0.05" /></div>
    </div>
    <div class="row">
      <div><label>Top-k frames / label</label><input type="number" id="topk" value="3" min="1" max="16" /></div>
      <div><label>Max verified labels</label><input type="number" id="maxlabels" value="6" min="1" /></div>
    </div>
    <label>Gemma instruction (optional)</label>
    <input type="text" id="instruction" placeholder="Leave blank for the default verdict instruction" />
    <p><button id="runBtn">Run Hybrid</button> <span id="status" class="muted"></span></p>
    <p id="error"></p>
  </div>

  <div id="results"></div>

  <script>
    const $ = (id) => document.getElementById(id);

    async function refreshModels() {
      try {
        const r = await fetch('/api/v1/models');
        const data = await r.json();
        const models = Array.isArray(data) ? data : (data.models || []);
        $('modelSelect').innerHTML = models
          .map((m) => `<option value="${m.model_id}">${m.display_name || m.model_id}</option>`).join('');
        const active = await (await fetch('/api/v1/models/active')).json();
        if (active && active.model_id) {
          $('modelSelect').value = active.model_id;
          $('modelStatus').textContent = 'Loaded: ' + (active.display_name || active.model_id);
        }
      } catch (e) { $('modelStatus').textContent = 'Could not list models.'; }
    }

    $('loadBtn').onclick = async () => {
      $('modelStatus').textContent = 'Loading…';
      const r = await fetch('/api/v1/models/load', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: $('modelSelect').value }),
      });
      const body = await r.json();
      $('modelStatus').textContent = r.ok ? ('Loaded: ' + body.model_id) : ('Error: ' + body.detail);
    };

    async function refreshGemma() {
      try {
        const s = await (await fetch('/api/v1/gemma/status')).json();
        const pill = $('gemmaPill');
        pill.textContent = s.state;
        pill.className = 'pill ' + s.state;
      } catch (e) {}
    }
    $('warmBtn').onclick = async () => { await fetch('/api/v1/gemma/warm', { method: 'POST' }); setTimeout(refreshGemma, 500); };

    function renderResults(body) {
      const wrap = $('results');
      wrap.innerHTML = '<h2>Results</h2>' +
        `<p class="muted">${body.metadata.gemma_calls} Gemma call(s); ` +
        `${body.metadata.labels_truncated} label(s) truncated by the cap. ` +
        `SigLIP2 ${body.metadata.latency.siglip2_seconds}s, Gemma ${body.metadata.latency.gemma_seconds}s.</p>` +
        `<p class="muted">${body.metadata.disclaimer}</p>`;
      body.results.forEach((r) => {
        const pct = Math.round(r.siglip2_score * 100);
        let html = `<div class="result ${r.gemma_evaluated ? '' : 'skipped'}">`;
        html += `<strong>${r.label}</strong> `;
        if (r.gemma_evaluated) {
          html += `<span class="verdict ${r.verdict}">${r.verdict}</span>`;
          if (r.parse_failed) html += ' <span class="muted">(parse failed)</span>';
        } else {
          html += '<span class="muted">not verified (below threshold or capped)</span>';
        }
        html += `<div class="bar"><span style="width:${pct}%"></span></div>`;
        html += `<div class="muted">SigLIP2 ${r.siglip2_score.toFixed(3)}</div>`;
        if (r.explanation) html += `<div>${r.explanation}</div>`;
        if (r.frames_shown && r.frames_shown.length) {
          html += '<div class="thumbs">' +
            r.frames_shown.map((f) => `<img src="${f.thumbnail}" title="t=${f.timestamp_seconds}s score=${f.score}" />`).join('') +
            '</div>';
        }
        html += '</div>';
        wrap.innerHTML += html;
      });
    }

    $('runBtn').onclick = async () => {
      $('error').textContent = '';
      const file = $('video').files[0];
      if (!file) { $('error').textContent = 'Choose a video first.'; return; }
      const labels = $('labels').value.split('\n').map((s) => s.trim()).filter(Boolean);
      const form = new FormData();
      form.append('video', file);
      form.append('labels', JSON.stringify(labels));
      form.append('aggregation', $('aggregation').value);
      form.append('threshold', $('threshold').value);
      form.append('top_k', $('topk').value);
      form.append('max_verified_labels', $('maxlabels').value);
      if ($('instruction').value.trim()) form.append('instruction', $('instruction').value.trim());
      $('status').textContent = 'Running…';
      $('runBtn').disabled = true;
      try {
        const r = await fetch('/api/v1/hybrid', { method: 'POST', body: form });
        const body = await r.json();
        if (!r.ok) { $('error').textContent = body.detail || 'Request failed.'; $('results').innerHTML = ''; }
        else renderResults(body);
      } catch (e) { $('error').textContent = String(e); }
      finally { $('status').textContent = ''; $('runBtn').disabled = false; }
    };

    refreshModels(); refreshGemma(); setInterval(refreshGemma, 4000);
  </script>
</body>
</html>
```

- [ ] **Step 4: Add nav links to the other two pages**

In `app/static/index.html`, line 358 is:

```html
    <a href="/gemma" style="margin-right:16px;">Gemma 4</a>
```

Add immediately after it:

```html
    <a href="/hybrid" style="margin-right:16px;">Hybrid</a>
```

In `app/static/gemma.html`, line 97 is:

```html
    <a href="/gemma" class="active">Gemma 4</a>
```

Add immediately after it:

```html
    <a href="/hybrid">Hybrid</a>
```

- [ ] **Step 5: Run the UI smoke test**

Run: `python -m pytest tests/test_hybrid_api.py::test_hybrid_page_served -v`
Expected: PASS

- [ ] **Step 6: Manual visual check (optional but recommended)**

```bash
GEMMA_ENABLED=true ALLOW_UNAUTHENTICATED=true SKIP_MODEL_AUTOLOAD=true \
  uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```
Open `http://127.0.0.1:8000/hybrid` — confirm nav (SigLIP2 | Gemma 4 | **Hybrid**), the form renders, and the SigLIP2/Gemma cards populate. (A full run needs both models loaded.)

- [ ] **Step 7: Commit**

```bash
git add app/static/hybrid.html app/static/index.html app/static/gemma.html tests/test_hybrid_api.py
git commit -m "feat(hybrid): standalone /hybrid page + nav links"
```

---

### Task 7: Gated real-model integration test

**Files:**
- Create: `tests/test_hybrid_integration.py`

**Interfaces:**
- Consumes: the live `/api/v1/hybrid` route with a real SigLIP2 model and a real Gemma model.

- [ ] **Step 1: Write the gated test**

Create `tests/test_hybrid_integration.py`:

```python
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
        for x in evaluated:
            assert x["verdict"] in ("present", "not_present", "uncertain")
            assert x["frames_shown"]  # real frames were re-extracted
            assert x["frames_shown"][0]["thumbnail"].startswith("data:image/jpeg;base64,")


async def _get(app, path):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        return await c.get(path)
```

- [ ] **Step 2: Verify it is skipped by default**

Run: `python -m pytest tests/test_hybrid_integration.py -v`
Expected: SKIPPED (1 skipped) — no models downloaded in normal runs.

- [ ] **Step 3: Commit**

```bash
git add tests/test_hybrid_integration.py
git commit -m "test(hybrid): gated real-model end-to-end smoke test"
```

- [ ] **Step 4: Full unit suite gate**

Run: `python -m pytest tests/ -v --ignore=tests/test_gemma_integration.py --ignore=tests/test_hybrid_integration.py`
Expected: PASS (entire non-gated suite green, including all new hybrid tests).

---

## Self-Review

**Spec coverage:**
- Topology one-call-per-label → Task 5 `verdict_pipeline` loop. ✓
- Verdict + explanation → Task 3 prompt/parse, Task 1 schema. ✓
- Gate: score ≥ threshold, top-N capped at `max_verified_labels` → Task 2 `gate_and_rank_labels`, Task 5 wiring + metadata. ✓
- Gate default `max` → Task 5 `aggregation="max"` default + Task 2 `per_label_scores`. ✓
- Standalone `/hybrid` page + nav → Task 5 route, Task 6 page + links. ✓
- Frame selection spread (≥0.5s) → Task 2 `select_topk_spread`. ✓
- Re-extract at 896px (not reuse 512px deleted frames) → Task 5 uses `gemma_extract_frames` on selected timestamps. ✓
- Inline base64 thumbnails → Task 2 `thumbnail_data_uri`, Task 5 `HybridFrameRef`. ✓
- Middleware route policy (H1) → Task 4. ✓
- `max_verified_labels` cap (H2) → Task 2 + Task 5 + metadata `labels_truncated`. ✓
- Single shared deadline (M3) → Task 5 `deadline` + `_remaining()` per phase. ✓
- Per-label parse isolation (M4) → Task 5 per-label try/except, degrade to `uncertain`. ✓
- `gemma_evaluated` rename (M5) → Task 1 schema. ✓
- `top_k` 1..16 validation (L2) → Task 5. ✓
- Chronological frame order before Gemma (L3) → Task 5 `refs.sort(...)`. ✓
- Real model id (L1) → Task 1 uses `settings.gemma_model_id`. ✓
- Error/edge cases (no-hits 200, parse fail, timeout, cold/no-model 503) → Task 5 + tests in Task 5. ✓
- Tests: unit (Tasks 1-3), middleware (Task 4), route non-gated (Task 5), UI smoke (Task 6), gated E2E (Task 7). ✓

**Placeholder scan:** No TBD/TODO; every code step has full code. The Task 5 Step 6 placeholder `hybrid.html` is intentional scaffolding, replaced in Task 6 Step 3. ✓

**Type consistency:** `FrameRef`/`GatedLabel` (Task 2) consumed in Task 5; `HybridFrameRef`/`HybridLabelResult`/`HybridMetadata`/`HybridLatency`/`HybridResponse` (Task 1) consumed in Task 5; `build_verdict_prompt`/`parse_verdict` (Task 3) consumed in Task 5; settings names (`gemma_max_new_tokens_verdict`, `hybrid_max_verified_labels`, `hybrid_thumbnail_px`) consistent across Tasks 1 and 5. ✓
