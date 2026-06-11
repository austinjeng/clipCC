# Gemma 4 E2B Exploration — Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Gemma 4 E2B generative-VLM slot to clipCC with `/api/v1/gemma/{label_scores,qa,status,warm}` endpoints and a separate web UI page, per `docs/superpowers/specs/2026-06-12-gemma4-e2b-exploration-design.md`.

**Architecture:** A purpose-built `VlmSlot` (idle→loading→loaded|failed state machine, warm-only loading, threaded `from_pretrained`) holds a `GemmaVLM` wrapper, gated by a dedicated capacity-1 limiter and a per-device atomic `ResidencyLedger` shared with the SigLIP2 `ModelManager`. Three new typed routes reuse the existing `InferenceRunner`/`TempStore` discipline; a dedicated timestamp-seek sampler replaces `FrameExtractor` for Gemma. Middleware gains an explicit route policy table with a pre-body slot-state gate.

**Tech Stack:** FastAPI, transformers (`AutoModelForMultimodalLM`), torch, anyio, pytest + pytest-asyncio (anyio mode), ffmpeg.

**Spec:** `docs/superpowers/specs/2026-06-12-gemma4-e2b-exploration-design.md` — read it before starting.

**Conventions you must follow (from this codebase):**
- Tests run with `python -m pytest tests/<file> -v` from repo root. Async tests use `@pytest.mark.anyio`.
- Unit tests must not import/load real models. Heavy model work goes in the env-gated integration test (Task 14).
- Match existing style: `from __future__ import annotations`, typed errors in `app/errors/handlers.py`, settings in `app/config.py` (pydantic-settings, env vars are UPPER_SNAKE of field names).
- Commit after every task with the message given.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `app/config.py` | modify | Gemma settings block |
| `app/errors/handlers.py` | modify | `GemmaOutputParseError` (502), `InvalidGemmaParamsError` (422) |
| `app/resource_gates.py` | modify | `vlm_admission()` capacity-1 limiter |
| `app/models/residency.py` | create | Per-device atomic reserve/commit/rollback ledger |
| `app/models/model_manager.py` | modify | Consult ledger in `_check_resources`; register SigLIP2 reservation |
| `app/models/vlm_slot.py` | create | Load state machine, warm-only, threaded load |
| `app/models/gemma_vlm.py` | create | transformers wrapper + cancel `StoppingCriteria` |
| `app/services/gemma_sampler.py` | create | Window resolve, timestamp plan, per-timestamp ffmpeg seek |
| `app/services/gemma_prompts.py` | create | Prompt builders + strict ID-keyed parse layer |
| `app/schemas/gemma.py` | create | Response models (`GemmaScoreItem`, metadata, status) |
| `app/middleware.py` | modify | Route policy table + pre-body slot gate |
| `app/main.py` | modify | Slot/ledger wiring, 4 routes + `/gemma` page route |
| `app/static/gemma.html` | create | Gemma exploration page |
| `app/static/index.html` | modify | Top nav |
| `requirements.txt`, `Dockerfile` | modify | accelerate, torchvision; variant-index install |
| `tests/test_residency.py` | create | Ledger units |
| `tests/test_vlm_slot.py` | create | State machine units (fake loader) |
| `tests/test_gemma_sampler.py` | create | Timestamp math + validation units |
| `tests/test_gemma_prompts.py` | create | Prompt/parse units |
| `tests/test_middleware.py` | modify | Policy table tests incl. negatives |
| `tests/test_gemma_api.py` | create | Route tests with fake slot |
| `tests/test_gemma_integration.py` | create | Env-gated real-model smoke |

---

### Task 1: Config — Gemma settings

**Files:**
- Modify: `app/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_config.py`:

```python
def test_gemma_defaults():
    from app.config import Settings
    s = Settings(allow_unauthenticated=True)
    assert s.gemma_model_id == "google/gemma-4-E2B-it"
    assert s.gemma_enabled is True
    assert s.gemma_max_frames == 8
    assert s.gemma_max_frames_cap == 16
    assert s.gemma_max_labels == 16
    assert s.gemma_analysis_window_seconds == 60.0
    assert s.gemma_max_new_tokens_qa == 400
    assert s.gemma_image_token_budget == 280
    assert s.gemma_evidence_top_k == 3
    assert s.gemma_reserve_gb == 12.0
    assert s.residency_headroom_gb == 2.0


def test_gemma_max_frames_clamped_to_cap():
    from app.config import Settings
    s = Settings(allow_unauthenticated=True, gemma_max_frames=99)
    assert s.effective_gemma_max_frames == 16
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_config.py -v -k gemma`
Expected: FAIL (`AttributeError: 'Settings' object has no attribute 'gemma_model_id'`)

- [ ] **Step 3: Implement** — in `app/config.py`, add fields inside `Settings` after `default_labels` (before `model_config`):

```python
    # --- Gemma 4 E2B exploration (spec: docs/superpowers/specs/2026-06-12-gemma4-e2b-exploration-design.md) ---
    gemma_model_id: str = "google/gemma-4-E2B-it"
    gemma_enabled: bool = True
    gemma_max_frames: int = 8
    gemma_max_frames_cap: int = 16
    gemma_max_labels: int = 16
    gemma_analysis_window_seconds: float = 60.0
    gemma_max_new_tokens_qa: int = 400
    gemma_image_token_budget: int = 280
    gemma_evidence_top_k: int = 3
    # 11.4 GB bf16 weights + KV/activations margin; reserved in the residency ledger
    gemma_reserve_gb: float = 12.0
    residency_headroom_gb: float = 2.0
```

and add a property next to `effective_upload_concurrency`:

```python
    @property
    def effective_gemma_max_frames(self) -> int:
        return min(self.gemma_max_frames, self.gemma_max_frames_cap)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: all PASS (existing tests unaffected)

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat(gemma): config block for Gemma 4 E2B exploration"
```

---

### Task 2: Error types

**Files:**
- Modify: `app/errors/handlers.py`
- Test: `tests/test_gemma_prompts.py` (created here, extended in Task 8)

- [ ] **Step 1: Write the failing test** — create `tests/test_gemma_prompts.py`:

```python
from app.errors.handlers import GemmaOutputParseError, InvalidGemmaParamsError


def test_gemma_output_parse_error_is_502():
    err = GemmaOutputParseError("bad json")
    assert err.status_code == 502
    assert "bad json" in err.detail


def test_invalid_gemma_params_error_is_422():
    err = InvalidGemmaParamsError("prompt too long")
    assert err.status_code == 422
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_gemma_prompts.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement** — append to `app/errors/handlers.py`:

```python
class GemmaOutputParseError(HTTPException):
    """The VLM produced output that failed strict parsing after retry (HTTP 502:
    upstream model produced unusable output)."""

    def __init__(self, detail: str) -> None:
        super().__init__(status_code=502, detail=f"Gemma output unusable: {detail}")


class InvalidGemmaParamsError(HTTPException):
    def __init__(self, detail: str) -> None:
        super().__init__(status_code=422, detail=detail)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_gemma_prompts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/errors/handlers.py tests/test_gemma_prompts.py
git commit -m "feat(gemma): typed errors for parse failure and param validation"
```

---

### Task 3: VLM capacity limiter

**Files:**
- Modify: `app/resource_gates.py`
- Test: `tests/test_resource_gates.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_resource_gates.py` (match the file's existing style; it already tests `inference_admission` rejection):

```python
@pytest.mark.anyio
async def test_vlm_admission_rejects_when_full():
    from app.errors.handlers import InferenceConcurrencyError
    gates = ResourceGates(max_upload_concurrency=4, max_inference_concurrency=2, max_vlm_concurrency=1)
    async with gates.vlm_admission():
        with pytest.raises(InferenceConcurrencyError):
            async with gates.vlm_admission():
                pass


@pytest.mark.anyio
async def test_vlm_admission_independent_of_inference():
    gates = ResourceGates(max_upload_concurrency=4, max_inference_concurrency=1, max_vlm_concurrency=1)
    # Holding the vlm slot must not consume the inference slot
    async with gates.vlm_admission():
        async with gates.inference_admission():
            pass
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_resource_gates.py -v -k vlm`
Expected: FAIL (`TypeError: unexpected keyword argument 'max_vlm_concurrency'`)

- [ ] **Step 3: Implement** — in `app/resource_gates.py`, change `__init__` and add the contextmanager:

```python
    def __init__(self, max_upload_concurrency: int = 4, max_inference_concurrency: int = 2,
                 max_vlm_concurrency: int = 1):
        self._upload_limiter = anyio.CapacityLimiter(max_upload_concurrency)
        self._inference_limiter = anyio.CapacityLimiter(max_inference_concurrency)
        self._vlm_limiter = anyio.CapacityLimiter(max_vlm_concurrency)

    @asynccontextmanager
    async def vlm_admission(self):
        # Dedicated limiter: a tens-of-seconds Gemma generation must never
        # starve SigLIP2 classify traffic (which shares inference_admission).
        token = object()
        try:
            self._vlm_limiter.acquire_on_behalf_of_nowait(token)
        except _WOULD_BLOCK:
            raise InferenceConcurrencyError()
        try:
            yield
        finally:
            self._vlm_limiter.release_on_behalf_of(token)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_resource_gates.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/resource_gates.py tests/test_resource_gates.py
git commit -m "feat(gemma): dedicated capacity-1 vlm_admission limiter"
```

---

### Task 4: Residency ledger

**Files:**
- Create: `app/models/residency.py`
- Test: `tests/test_residency.py`

- [ ] **Step 1: Write the failing test** — create `tests/test_residency.py`:

```python
import pytest

from app.models.residency import ResidencyLedger
from app.models.model_manager import InsufficientResourcesError


def make_ledger(free_bytes_by_device, headroom_gb=0.0):
    ledger = ResidencyLedger(headroom_gb=headroom_gb)
    # Test seam: replace the device probe with a fake
    ledger._device_free = lambda device: free_bytes_by_device[device]
    return ledger


def test_reserve_within_free_succeeds():
    ledger = make_ledger({"cpu": 10_000_000_000})
    ledger.reserve("vlm", "cpu", 8_000_000_000)
    assert ledger.reserved_bytes("cpu") == 8_000_000_000


def test_reserve_beyond_free_raises():
    ledger = make_ledger({"cpu": 10_000_000_000})
    with pytest.raises(InsufficientResourcesError):
        ledger.reserve("vlm", "cpu", 11_000_000_000)


def test_headroom_subtracted():
    ledger = make_ledger({"cpu": 10_000_000_000}, headroom_gb=4.0)
    with pytest.raises(InsufficientResourcesError):
        ledger.reserve("vlm", "cpu", 7_000_000_000)  # 10 - 4 headroom = 6 available


def test_second_reservation_sees_first():
    ledger = make_ledger({"cpu": 10_000_000_000})
    ledger.reserve("vlm", "cpu", 6_000_000_000)
    with pytest.raises(InsufficientResourcesError):
        ledger.reserve("siglip2", "cpu", 5_000_000_000)


def test_rollback_frees_reservation():
    ledger = make_ledger({"cpu": 10_000_000_000})
    ledger.reserve("vlm", "cpu", 8_000_000_000)
    ledger.rollback("vlm")
    assert ledger.reserved_bytes("cpu") == 0
    ledger.reserve("siglip2", "cpu", 8_000_000_000)  # now fits


def test_commit_keeps_reservation_and_release_frees_it():
    ledger = make_ledger({"cpu": 10_000_000_000})
    ledger.reserve("vlm", "cpu", 8_000_000_000)
    ledger.commit("vlm")
    assert ledger.reserved_bytes("cpu") == 8_000_000_000
    ledger.release("vlm")
    assert ledger.reserved_bytes("cpu") == 0


def test_devices_are_independent():
    ledger = make_ledger({"cpu": 4_000_000_000, "cuda:0": 16_000_000_000})
    ledger.reserve("vlm", "cuda:0", 12_000_000_000)
    assert ledger.reserved_bytes("cpu") == 0


def test_duplicate_owner_reserve_raises():
    ledger = make_ledger({"cpu": 10_000_000_000})
    ledger.reserve("vlm", "cpu", 1_000_000_000)
    with pytest.raises(ValueError):
        ledger.reserve("vlm", "cpu", 1_000_000_000)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_residency.py -v`
Expected: FAIL (ImportError: no module `app.models.residency`)

- [ ] **Step 3: Implement** — create `app/models/residency.py`:

```python
from __future__ import annotations

import threading
from dataclasses import dataclass

import psutil
import torch

from app.models.model_manager import InsufficientResourcesError


@dataclass
class _Reservation:
    device: str
    nbytes: int
    committed: bool = False


class ResidencyLedger:
    """Per-device, atomic model-memory accounting.

    Protocol: reserve(owner, device, nbytes) before loading (atomic
    check-and-reserve against device_free - other_reservations - headroom),
    then commit(owner) on load success or rollback(owner) on failure.
    release(owner) frees a committed reservation (e.g. on model swap).

    Deployment target is CUDA where host RAM can be plentiful while VRAM is
    the bottleneck — hence per-device keying, not a single RAM number.
    """

    def __init__(self, headroom_gb: float = 2.0):
        self._lock = threading.Lock()
        self._reservations: dict[str, _Reservation] = {}
        self._headroom_bytes = int(headroom_gb * 1e9)

    def _device_free(self, device: str) -> int:
        if device.startswith("cuda"):
            free, _total = torch.cuda.mem_get_info()
            return free
        # cpu and mps (Apple unified memory) both draw from host RAM
        return psutil.virtual_memory().available

    def reserved_bytes(self, device: str) -> int:
        with self._lock:
            return sum(r.nbytes for r in self._reservations.values() if r.device == device)

    def reserve(self, owner: str, device: str, nbytes: int) -> None:
        with self._lock:
            if owner in self._reservations:
                raise ValueError(f"Owner '{owner}' already holds a reservation.")
            others = sum(r.nbytes for r in self._reservations.values() if r.device == device)
            available = self._device_free(device) - others - self._headroom_bytes
            if nbytes > available:
                raise InsufficientResourcesError(
                    f"Cannot reserve {nbytes / 1e9:.1f}GB on {device}: "
                    f"{available / 1e9:.1f}GB available after "
                    f"{others / 1e9:.1f}GB existing reservations and "
                    f"{self._headroom_bytes / 1e9:.1f}GB headroom."
                )
            self._reservations[owner] = _Reservation(device=device, nbytes=nbytes)

    def commit(self, owner: str) -> None:
        with self._lock:
            self._reservations[owner].committed = True

    def rollback(self, owner: str) -> None:
        with self._lock:
            self._reservations.pop(owner, None)

    def release(self, owner: str) -> None:
        with self._lock:
            self._reservations.pop(owner, None)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_residency.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/models/residency.py tests/test_residency.py
git commit -m "feat(gemma): per-device atomic residency ledger"
```

---

### Task 5: ModelManager consults the ledger

**Files:**
- Modify: `app/models/model_manager.py` (constructor `:120-129`, `_check_resources` `:211-236`, `load_model` `:166-201`)
- Test: `tests/test_model_manager.py`

**Why:** `_check_resources` currently credits back the active model's RAM assuming unload-before-load; with a resident Gemma that math approves loads that OOM. The manager must subtract ledger reservations and register its own.

- [ ] **Step 1: Write the failing test** — append to `tests/test_model_manager.py` (it already has helpers constructing `ModelManager(cache_dir=...)`; follow its mock style for psutil):

```python
def test_check_resources_subtracts_ledger_reservations(monkeypatch, tmp_path):
    from app.models.model_manager import ModelManager, ModelConfig, InsufficientResourcesError
    from app.models.residency import ResidencyLedger
    import app.models.model_manager as mm

    ledger = ResidencyLedger(headroom_gb=0.0)
    ledger._device_free = lambda device: 20_000_000_000
    ledger.reserve("vlm", "cpu", 12_000_000_000)
    ledger.commit("vlm")

    manager = ModelManager(cache_dir=str(tmp_path), ledger=ledger)

    class FakeMem:
        total = 20_000_000_000
        available = 16_000_000_000

    monkeypatch.setattr(mm.psutil, "virtual_memory", lambda: FakeMem())

    config = ModelConfig(
        model_id="big", display_name="Big", model_type="siglip2",
        hf_repo="x/big", params="2B", resolution=384, min_ram_gb=10,
    )
    # 10GB*1.2 = 12GB required; available 16GB - 12GB vlm reservation = 4GB → must fail
    with pytest.raises(InsufficientResourcesError):
        manager._check_resources(config)


def test_check_resources_passes_without_reservations(monkeypatch, tmp_path):
    from app.models.model_manager import ModelManager, ModelConfig
    from app.models.residency import ResidencyLedger
    import app.models.model_manager as mm

    manager = ModelManager(cache_dir=str(tmp_path), ledger=ResidencyLedger(headroom_gb=0.0))

    class FakeMem:
        total = 20_000_000_000
        available = 16_000_000_000

    monkeypatch.setattr(mm.psutil, "virtual_memory", lambda: FakeMem())

    config = ModelConfig(
        model_id="big", display_name="Big", model_type="siglip2",
        hf_repo="x/big", params="2B", resolution=384, min_ram_gb=10,
    )
    manager._check_resources(config)  # must not raise
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_model_manager.py -v -k check_resources`
Expected: FAIL (`TypeError: unexpected keyword argument 'ledger'`)

- [ ] **Step 3: Implement** — in `app/models/model_manager.py`:

Constructor (keep `ledger` optional so existing tests/callers stand):

```python
    def __init__(self, cache_dir: str, offline: bool = False, ledger=None):
        self.registry = dict(SIGLIP2_REGISTRY)
        self.active_model: BaseModel | None = None
        self.active_model_id: str | None = None
        self.cache_dir = cache_dir
        self._offline = offline
        self._ledger = ledger
        self._condition = asyncio.Condition()
        self._swapping = False
        self._active_leases = 0
        self._load_manifest()
```

In `_check_resources`, after `estimated_available` is computed (line ~229, after the active-model credit-back), subtract ledger reservations:

```python
        if self._ledger is not None:
            estimated_available -= self._ledger.reserved_bytes("cpu")
```

In `load_model`, register the new model's footprint after the successful swap (replace the final `async with self._condition:` block):

```python
        async with self._condition:
            self.active_model = new_model
            self.active_model_id = model_id
            self._swapping = False
            self._condition.notify_all()

        if self._ledger is not None and config.min_ram_gb:
            # Best-effort: re-register this manager's resident footprint so the
            # vlm slot's preflight sees it. Reserve-after-load (the bytes are
            # already resident, this is bookkeeping, not admission control).
            self._ledger.release("siglip2")
            try:
                self._ledger.reserve("siglip2", "cpu", int(config.min_ram_gb * 1e9))
                self._ledger.commit("siglip2")
            except Exception:
                pass  # already loaded; ledger refusal must not fail the swap
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_model_manager.py tests/test_residency.py -v`
Expected: all PASS (pre-existing manager tests unaffected — `ledger` defaults to `None`)

- [ ] **Step 5: Commit**

```bash
git add app/models/model_manager.py tests/test_model_manager.py
git commit -m "feat(gemma): ModelManager consults shared residency ledger"
```

---

### Task 6: VlmSlot state machine

**Files:**
- Create: `app/models/vlm_slot.py`
- Test: `tests/test_vlm_slot.py`

**Contract (from spec §1):** warm-only loading; `warm()` returns immediately (load runs as a background task in a worker thread); the lock guards only state transitions; concurrent warms coalesce; load failure → `failed` with error detail, retryable; ledger reserve→commit/rollback around the load.

- [ ] **Step 1: Write the failing test** — create `tests/test_vlm_slot.py`:

```python
import anyio
import pytest

from app.models.residency import ResidencyLedger
from app.models.model_manager import InsufficientResourcesError
from app.models.vlm_slot import VlmSlot, VlmState


def make_ledger(free=100_000_000_000):
    ledger = ResidencyLedger(headroom_gb=0.0)
    ledger._device_free = lambda device: free
    return ledger


def make_slot(loader, ledger=None, reserve_bytes=12_000_000_000, enabled=True):
    return VlmSlot(
        loader=loader,
        ledger=ledger or make_ledger(),
        device="cpu",
        reserve_bytes=reserve_bytes,
        enabled=enabled,
    )


@pytest.mark.anyio
async def test_initial_state_is_idle():
    slot = make_slot(loader=lambda: object())
    assert slot.state == VlmState.IDLE
    assert slot.model is None


@pytest.mark.anyio
async def test_warm_loads_and_transitions_to_loaded():
    sentinel = object()
    slot = make_slot(loader=lambda: sentinel)
    await slot.warm()
    await slot.wait_settled()
    assert slot.state == VlmState.LOADED
    assert slot.model is sentinel


@pytest.mark.anyio
async def test_loader_failure_transitions_to_failed_and_rolls_back():
    ledger = make_ledger()

    def boom():
        raise RuntimeError("download exploded")

    slot = make_slot(loader=boom, ledger=ledger)
    await slot.warm()
    await slot.wait_settled()
    assert slot.state == VlmState.FAILED
    assert "download exploded" in (slot.error or "")
    assert ledger.reserved_bytes("cpu") == 0  # rolled back


@pytest.mark.anyio
async def test_failed_is_retryable():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first time fails")
        return object()

    slot = make_slot(loader=flaky)
    await slot.warm()
    await slot.wait_settled()
    assert slot.state == VlmState.FAILED
    await slot.warm()
    await slot.wait_settled()
    assert slot.state == VlmState.LOADED


@pytest.mark.anyio
async def test_concurrent_warms_load_once():
    started = anyio.Event()
    release = anyio.Event()
    calls = {"n": 0}

    def slow_loader():
        calls["n"] += 1
        anyio.from_thread.run_sync(started.set)
        anyio.from_thread.run(release.wait)
        return object()

    slot = make_slot(loader=slow_loader)
    await slot.warm()
    await started.wait()
    assert slot.state == VlmState.LOADING
    await slot.warm()  # second warm while loading: must coalesce, not double-load
    release.set()
    await slot.wait_settled()
    assert slot.state == VlmState.LOADED
    assert calls["n"] == 1


@pytest.mark.anyio
async def test_insufficient_residency_refuses_without_loading():
    ledger = make_ledger(free=4_000_000_000)
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        return object()

    slot = make_slot(loader=loader, ledger=ledger, reserve_bytes=12_000_000_000)
    with pytest.raises(InsufficientResourcesError):
        await slot.warm()
    assert slot.state == VlmState.IDLE
    assert calls["n"] == 0


@pytest.mark.anyio
async def test_disabled_slot_refuses_warm():
    slot = make_slot(loader=lambda: object(), enabled=False)
    with pytest.raises(RuntimeError):
        await slot.warm()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_vlm_slot.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement** — create `app/models/vlm_slot.py`:

```python
from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any, Callable, Optional

import anyio

logger = logging.getLogger(__name__)


class VlmState(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"


class VlmSlot:
    """Load-once holder for the generative VLM.

    Deliberately NOT the lease/hot-swap ModelManager: this slot loads once via
    an explicit warm call and never swaps. warm() is the ONLY load trigger
    (spec §1) — video endpoints fail fast unless state is LOADED.

    The asyncio.Lock guards state transitions only; the blocking
    from_pretrained runs in a worker thread so status polling stays live.
    """

    def __init__(
        self,
        loader: Callable[[], Any],
        ledger,
        device: str,
        reserve_bytes: int,
        enabled: bool = True,
    ):
        self._loader = loader
        self._ledger = ledger
        self._device = device
        self._reserve_bytes = reserve_bytes
        self._enabled = enabled
        self._state = VlmState.IDLE
        self._error: Optional[str] = None
        self._model: Any = None
        self._lock = asyncio.Lock()
        self._load_task: Optional[asyncio.Task] = None

    @property
    def state(self) -> VlmState:
        return self._state

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def model(self) -> Any:
        return self._model

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def device(self) -> str:
        return self._device

    async def warm(self) -> VlmState:
        """Trigger the load (idempotent). Returns the current state immediately;
        the load itself runs as a background task. Raises
        InsufficientResourcesError if the residency reservation is refused,
        RuntimeError if the slot is disabled."""
        if not self._enabled:
            raise RuntimeError("Gemma slot is disabled (GEMMA_ENABLED=false).")
        async with self._lock:
            if self._state in (VlmState.LOADING, VlmState.LOADED):
                return self._state
            # Atomic reserve before any load work; raises InsufficientResourcesError
            self._ledger.reserve("vlm", self._device, self._reserve_bytes)
            self._state = VlmState.LOADING
            self._error = None
        self._load_task = asyncio.create_task(self._load())
        return self._state

    async def wait_settled(self) -> VlmState:
        """Await the in-flight load task, if any (used by tests and shutdown)."""
        task = self._load_task
        if task is not None:
            try:
                await task
            except Exception:
                pass
        return self._state

    async def _load(self) -> None:
        try:
            model = await anyio.to_thread.run_sync(self._loader)
        except BaseException as e:
            self._ledger.rollback("vlm")
            async with self._lock:
                self._state = VlmState.FAILED
                self._error = str(e)
            logger.error(f"VLM load failed: {e}")
            return
        self._ledger.commit("vlm")
        async with self._lock:
            self._model = model
            self._state = VlmState.LOADED
        logger.info("VLM loaded.")
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_vlm_slot.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/models/vlm_slot.py tests/test_vlm_slot.py
git commit -m "feat(gemma): VlmSlot warm-only load state machine with ledger"
```

---

### Task 7: GemmaVLM wrapper

**Files:**
- Create: `app/models/gemma_vlm.py`
- Test: `tests/test_vlm_slot.py` (append — the cancel criteria is pure logic)

**Note:** Real model loading is exercised only in the gated integration test (Task 14). Unit scope here: the stopping criteria, device/dtype pick, and message construction (pure functions).

- [ ] **Step 1: Write the failing test** — append to `tests/test_vlm_slot.py`:

```python
def test_cancel_stopping_criteria_reads_event():
    import threading
    from app.models.gemma_vlm import CancelStoppingCriteria

    ev = threading.Event()
    crit = CancelStoppingCriteria(ev)
    assert crit(None, None) is False
    ev.set()
    assert crit(None, None) is True


def test_pick_device_and_dtype_cpu(monkeypatch):
    import torch
    from app.models import gemma_vlm

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    device, dtype = gemma_vlm.pick_device_and_dtype()
    assert device == "cpu"
    assert dtype == torch.float32  # bf16 on x86 CPU hits slow/absent kernels


def test_build_messages_interleaves_images_then_text():
    from PIL import Image
    from app.models.gemma_vlm import build_messages

    imgs = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
    messages = build_messages(imgs, "describe")
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert [c["type"] for c in content] == ["image", "image", "text"]
    assert content[-1]["text"] == "describe"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_vlm_slot.py -v -k "cancel or pick or messages"`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement** — create `app/models/gemma_vlm.py`:

```python
from __future__ import annotations

import threading

import torch
from PIL import Image
from transformers import StoppingCriteria, StoppingCriteriaList


class CancelStoppingCriteria(StoppingCriteria):
    """Reads the SAME threading.Event that InferenceRunner sets on deadline.

    Consulted between decode steps only — blind during prefill. This makes the
    request timeout a soft deadline (spec §3); max_new_tokens is the real bound.
    """

    def __init__(self, cancel_event: threading.Event):
        super().__init__()
        self._cancel_event = cancel_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self._cancel_event.is_set()


def pick_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    # bf16 on x86 CPU falls back to slow kernels or silently upcasts; fp32 is
    # the honest CPU dtype (≈22GB resident — CPU is demo-only per spec).
    return "cpu", torch.float32


def build_messages(frames: list[Image.Image], prompt: str) -> list[dict]:
    content: list[dict] = [{"type": "image", "image": img} for img in frames]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


class GemmaVLM:
    """Thin wrapper around Gemma 4 E2B via transformers.

    NOTE for implementer: exact apply_chat_template / processor kwargs must be
    verified against the installed transformers version using the model card
    examples (https://huggingface.co/google/gemma-4-E2B-it) in the gated
    integration test. enable_thinking=False is REQUIRED — the E2B/E4B
    "ghost thought channel" token leak breaks JSON parsing otherwise (spec §5).
    """

    def __init__(self, hf_repo: str, cache_dir: str, image_token_budget: int = 280):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.device, self.dtype = pick_device_and_dtype()
        self.image_token_budget = image_token_budget
        self.processor = AutoProcessor.from_pretrained(hf_repo, cache_dir=cache_dir)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            hf_repo,
            cache_dir=cache_dir,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        frames: list[Image.Image],
        prompt: str,
        max_new_tokens: int,
        cancel_event: threading.Event,
    ) -> str:
        messages = build_messages(frames, prompt)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([CancelStoppingCriteria(cancel_event)]),
            )
        new_tokens = output_ids[:, inputs["input_ids"].shape[-1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_vlm_slot.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/models/gemma_vlm.py tests/test_vlm_slot.py
git commit -m "feat(gemma): GemmaVLM wrapper with cancel stopping criteria"
```

---

### Task 8: Gemma frame sampler

**Files:**
- Create: `app/services/gemma_sampler.py`
- Test: `tests/test_gemma_sampler.py`

**Contract (spec §6):** the existing `FrameExtractor` front-loads (fps filter + `-frames:v`) and reconstructs timestamps as `i/fps` — unusable for Gemma. This sampler plans explicit timestamps over an analysis window, seeks each one, and records exact requested timestamps. Upload duration limit still applies; the frames-vs-duration rejection does not.

- [ ] **Step 1: Write the failing test** — create `tests/test_gemma_sampler.py`:

```python
import threading
from pathlib import Path

import pytest

from app.config import Settings
from app.errors.handlers import DurationTooLongError, InvalidGemmaParamsError
from app.services.gemma_sampler import plan_timestamps, resolve_window, validate_gemma_video
from app.services.video import VideoInfo


def make_info(duration=120.0, width=1280, height=720, streams=1):
    return VideoInfo(duration=duration, width=width, height=height,
                     video_stream_count=streams, format_name="mp4")


def settings():
    return Settings(allow_unauthenticated=True)


# --- resolve_window ---

def test_window_clamps_to_video_end():
    start, end = resolve_window(duration=45.0, window_start=0.0, window_seconds=60.0)
    assert (start, end) == (0.0, 45.0)


def test_window_with_offset():
    start, end = resolve_window(duration=180.0, window_start=30.0, window_seconds=60.0)
    assert (start, end) == (30.0, 90.0)


def test_window_start_beyond_duration_raises():
    with pytest.raises(InvalidGemmaParamsError):
        resolve_window(duration=50.0, window_start=60.0, window_seconds=60.0)


def test_negative_window_start_raises():
    with pytest.raises(InvalidGemmaParamsError):
        resolve_window(duration=50.0, window_start=-1.0, window_seconds=60.0)


# --- plan_timestamps ---

def test_timestamps_are_uniform_midpoints():
    ts = plan_timestamps(span_start=0.0, span_end=80.0, n_frames=8)
    assert ts == [5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0]


def test_timestamps_respect_offset_window():
    ts = plan_timestamps(span_start=30.0, span_end=90.0, n_frames=4)
    assert ts == [37.5, 52.5, 67.5, 82.5]


def test_short_video_fewer_frames_than_requested():
    # 1.5s video at most 1 frame per 0.5s spacing floor → fewer frames, no dupes
    ts = plan_timestamps(span_start=0.0, span_end=1.5, n_frames=8)
    assert len(ts) == len(set(ts))
    assert all(0.0 <= t <= 1.5 for t in ts)
    assert len(ts) <= 8


# --- validate_gemma_video ---

def test_duration_limit_still_applies():
    with pytest.raises(DurationTooLongError):
        validate_gemma_video(make_info(duration=999.0), settings())


def test_long_video_within_limit_passes():
    # 290s exceeds 60s window but is under max_duration_seconds=300 — must NOT
    # raise (Gemma analyzes a window; the frames-vs-duration rule is SigLIP2's)
    validate_gemma_video(make_info(duration=290.0), settings())


def test_resolution_limit_applies():
    from app.errors.handlers import ResolutionTooHighError
    with pytest.raises(ResolutionTooHighError):
        validate_gemma_video(make_info(width=4000, height=2200), settings())


def test_multi_stream_rejected():
    from app.errors.handlers import MultipleVideoStreamsError
    with pytest.raises(MultipleVideoStreamsError):
        validate_gemma_video(make_info(streams=2), settings())
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_gemma_sampler.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement** — create `app/services/gemma_sampler.py`:

```python
from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.inference_runner import InferenceRunner

from app.config import Settings
from app.errors.handlers import (
    DurationTooLongError,
    InvalidGemmaParamsError,
    MultipleVideoStreamsError,
    ResolutionTooHighError,
)
from app.services.video import VideoInfo

# Spacing floor: two frames closer than this are visually redundant for a VLM
_MIN_SPACING_SECONDS = 0.5


@dataclass
class GemmaFrame:
    path: Path
    timestamp_seconds: float


def validate_gemma_video(info: VideoInfo, settings: Settings) -> None:
    """Gemma-specific constraints: duration/resolution/stream rules apply,
    but NOT the frames-vs-duration rule (Gemma samples a fixed frame count
    from a window regardless of total duration)."""
    if info.duration > settings.max_duration_seconds:
        raise DurationTooLongError(info.duration, settings.max_duration_seconds)
    if info.width > 3840 or info.height > 2160:
        raise ResolutionTooHighError(info.width, info.height)
    if info.width * info.height > 8_300_000:
        raise ResolutionTooHighError(info.width, info.height)
    if info.video_stream_count > 1:
        raise MultipleVideoStreamsError(info.video_stream_count)


def resolve_window(duration: float, window_start: float, window_seconds: float) -> tuple[float, float]:
    if window_start < 0:
        raise InvalidGemmaParamsError("window_start must be >= 0.")
    if window_start >= duration:
        raise InvalidGemmaParamsError(
            f"window_start {window_start:.1f}s is beyond the video duration {duration:.1f}s."
        )
    return window_start, min(duration, window_start + window_seconds)


def plan_timestamps(span_start: float, span_end: float, n_frames: int) -> list[float]:
    """Uniform midpoint sampling: frame i sits at the center of slice i.
    These exact values are recorded with the frames and enumerated to the
    model — the closed timestamp set for events mode (spec §4/§6)."""
    span = span_end - span_start
    n = min(n_frames, max(1, int(span / _MIN_SPACING_SECONDS)))
    slice_len = span / n
    return [round(span_start + (i + 0.5) * slice_len, 3) for i in range(n)]


def extract_frames(
    video_path: Path,
    timestamps: list[float],
    frame_dir: Path,
    cancel_event: threading.Event,
    ffmpeg_timeout: int = 120,
    runner: "Optional[InferenceRunner]" = None,
) -> list[GemmaFrame]:
    """One ffmpeg seek per timestamp. N is small (<=16) so per-seek process
    overhead is acceptable and -ss before -i makes each seek fast."""
    frames: list[GemmaFrame] = []
    for i, ts in enumerate(timestamps):
        if cancel_event.is_set():
            break
        out_path = frame_dir / f"gemma_{i:03d}.jpg"
        cmd = [
            "ffmpeg", "-nostdin", "-v", "error",
            "-ss", f"{ts:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-vf", "scale='min(896,iw)':'min(896,ih)':force_original_aspect_ratio=decrease",
            "-q:v", "2",
            str(out_path),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if runner is not None:
            runner.register_process(proc)
        try:
            _, stderr = proc.communicate(timeout=ffmpeg_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise RuntimeError(f"ffmpeg seek at {ts:.1f}s timed out after {ffmpeg_timeout}s")
        finally:
            if runner is not None:
                runner.unregister_process()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg seek at {ts:.1f}s failed: {stderr.decode(errors='replace').strip()}")
        if out_path.exists():
            frames.append(GemmaFrame(path=out_path, timestamp_seconds=ts))
    return frames
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_gemma_sampler.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/gemma_sampler.py tests/test_gemma_sampler.py
git commit -m "feat(gemma): timestamp-seek frame sampler with analysis window"
```

---

### Task 9: Schemas + prompt build/parse layer

**Files:**
- Create: `app/schemas/gemma.py`
- Create: `app/services/gemma_prompts.py`
- Test: `tests/test_gemma_prompts.py` (append)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_gemma_prompts.py`:

```python
import pytest

from app.services.gemma_prompts import (
    build_label_scores_prompt,
    build_qa_prompt,
    label_scores_token_budget,
    parse_label_scores,
)

LABELS = ["texting while driving", "sleeping", "eating"]


def test_prompt_numbers_labels_from_1():
    p = build_label_scores_prompt(LABELS, evidence_top_k=3)
    assert "1: texting while driving" in p
    assert "3: eating" in p
    assert '"id"' in p and '"score"' in p


def test_token_budget_scales_with_label_count():
    assert label_scores_token_budget(1) < label_scores_token_budget(16)
    assert label_scores_token_budget(16) <= 640


def test_parse_happy_path():
    text = '[{"id": 1, "score": 0.9, "evidence": "phone visible"}, {"id": 2, "score": 0.1}, {"id": 3, "score": 0.2}]'
    items = parse_label_scores(text, LABELS)
    assert items[0].label == "texting while driving"
    assert items[0].score == 0.9
    assert items[0].evidence == "phone visible"
    assert items[1].evidence is None


def test_parse_strips_code_fences():
    text = '```json\n[{"id": 1, "score": 0.5}]\n```'
    items = parse_label_scores(text, ["only label"])
    assert items[0].score == 0.5


def test_parse_unknown_id_rejected():
    with pytest.raises(ValueError):
        parse_label_scores('[{"id": 9, "score": 0.5}]', LABELS)


def test_parse_duplicate_id_rejected():
    with pytest.raises(ValueError):
        parse_label_scores('[{"id": 1, "score": 0.5}, {"id": 1, "score": 0.6}]', LABELS)


def test_parse_score_out_of_range_rejected():
    with pytest.raises(ValueError):
        parse_label_scores('[{"id": 1, "score": 1.5}]', LABELS)


def test_parse_score_as_string_rejected():
    with pytest.raises(ValueError):
        parse_label_scores('[{"id": 1, "score": "0.5"}]', LABELS)


def test_parse_score_as_bool_rejected():
    with pytest.raises(ValueError):
        parse_label_scores('[{"id": 1, "score": true}]', LABELS)


def test_parse_missing_ids_yield_null_scores():
    items = parse_label_scores('[{"id": 2, "score": 0.7}]', LABELS)
    by_label = {i.label: i.score for i in items}
    assert by_label["sleeping"] == 0.7
    assert by_label["texting while driving"] is None
    assert by_label["eating"] is None


def test_parse_non_list_rejected():
    with pytest.raises(ValueError):
        parse_label_scores('{"id": 1, "score": 0.5}', LABELS)


def test_qa_prompt_embeds_user_text():
    p = build_qa_prompt("what is the driver doing?")
    assert "what is the driver doing?" in p
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_gemma_prompts.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement** — create `app/schemas/gemma.py`:

```python
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

GEMMA_SCORE_SEMANTICS = "gemma4_verbalized_uncalibrated"

GEMMA_DISCLAIMER = (
    "Scores are verbalized self-reports from a generative model: uncalibrated, "
    "quantized to round numbers, and implicitly contrastive across labels. "
    "Treat as ordinal. NOT comparable in magnitude to SigLIP2 sigmoid scores. "
    "Not suitable for safety-critical decisions."
)


class GemmaScoreItem(BaseModel):
    label: str
    score: Optional[float]  # None = model omitted this label id
    evidence: Optional[str] = None


class GemmaLatency(BaseModel):
    extract_seconds: float
    generate_seconds: float
    parse_seconds: float


class GemmaMetadata(BaseModel):
    model: str
    device: str
    frames_analyzed: int
    window_start_seconds: float
    window_end_seconds: float
    video_duration_seconds: float
    score_semantics: str = GEMMA_SCORE_SEMANTICS
    disclaimer: str = GEMMA_DISCLAIMER
    latency: GemmaLatency
    parse_retries: int = 0


class GemmaLabelScoresResponse(BaseModel):
    scores: list[GemmaScoreItem]
    metadata: GemmaMetadata


class GemmaQAResponse(BaseModel):
    answer: str
    metadata: GemmaMetadata


class GemmaStatusResponse(BaseModel):
    enabled: bool
    state: str  # idle | loading | loaded | failed
    error: Optional[str] = None
    model_id: str
    device: str
```

Create `app/services/gemma_prompts.py`:

```python
from __future__ import annotations

import json
import re

from app.schemas.gemma import GemmaScoreItem

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)

# Token budget: fixed overhead + per-label allowance ({"id": N, "score": 0.XX},
# plus evidence strings for the top-k items). Bounded well under qa's budget.
_BUDGET_BASE = 64
_BUDGET_PER_LABEL = 24
_BUDGET_MAX = 640


def label_scores_token_budget(n_labels: int) -> int:
    return min(_BUDGET_BASE + _BUDGET_PER_LABEL * n_labels, _BUDGET_MAX)


def build_label_scores_prompt(labels: list[str], evidence_top_k: int) -> str:
    numbered = "\n".join(f"{i + 1}: {label}" for i, label in enumerate(labels))
    return (
        "You are analyzing frames sampled from a video, in chronological order.\n"
        "Score how strongly each numbered behavior below is visible anywhere in these frames.\n\n"
        f"Behaviors:\n{numbered}\n\n"
        "Respond with ONLY a JSON array, no other text. One object per behavior id:\n"
        '[{"id": <behavior number>, "score": <number from 0.0 to 1.0>}]\n'
        f"For the {evidence_top_k} highest-scoring behaviors only, add an "
        '"evidence" field: one short sentence describing what you saw.\n'
        "Scores must be JSON numbers, not strings. Include every id exactly once."
    )


def build_qa_prompt(user_prompt: str) -> str:
    return (
        "You are analyzing frames sampled from a video, in chronological order. "
        "Answer the following question about the video concisely.\n\n"
        f"Question: {user_prompt}"
    )


def parse_label_scores(text: str, labels: list[str]) -> list[GemmaScoreItem]:
    """Strict ID-keyed parse (spec §4/§5): reject unknown ids, duplicate ids,
    non-numeric or out-of-range scores. Missing ids become score=None.
    Raises ValueError on any violation (route does one bounded retry)."""
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e}") from e
    if not isinstance(data, list):
        raise ValueError("expected a JSON array")

    by_id: dict[int, dict] = {}
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("array items must be objects")
        item_id = item.get("id")
        if not isinstance(item_id, int) or isinstance(item_id, bool):
            raise ValueError(f"id must be an integer, got {item_id!r}")
        if item_id < 1 or item_id > len(labels):
            raise ValueError(f"unknown id {item_id} (valid: 1..{len(labels)})")
        if item_id in by_id:
            raise ValueError(f"duplicate id {item_id}")
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"score for id {item_id} must be a JSON number, got {score!r}")
        if score < 0.0 or score > 1.0:
            raise ValueError(f"score {score} for id {item_id} out of range [0, 1]")
        evidence = item.get("evidence")
        if evidence is not None and not isinstance(evidence, str):
            raise ValueError(f"evidence for id {item_id} must be a string")
        by_id[item_id] = {"score": float(score), "evidence": evidence}

    return [
        GemmaScoreItem(
            label=label,
            score=by_id.get(i + 1, {}).get("score"),
            evidence=by_id.get(i + 1, {}).get("evidence"),
        )
        for i, label in enumerate(labels)
    ]
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_gemma_prompts.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/schemas/gemma.py app/services/gemma_prompts.py tests/test_gemma_prompts.py
git commit -m "feat(gemma): response schemas, prompt builders, strict ID-keyed parse"
```

---

### Task 10: Middleware route policy

**Files:**
- Modify: `app/middleware.py`
- Test: `tests/test_middleware.py`

**Contract (spec §7):** explicit policy table. POST video routes: auth → **pre-body slot-state gate** (503 + `Retry-After` before draining multipart) → upload admission → body size. `status`/`warm`: auth only. `/gemma` page: pass-through.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_middleware.py` (add `from contextlib import asynccontextmanager` to its imports):

```python
class NoUploadGates(ResourceGates):
    """Test double: any path that consumes upload admission fails the test.
    (CapacityLimiter requires >= 1 token, so 'capacity 0' can't be expressed
    directly — assert non-consumption instead.)"""

    @asynccontextmanager
    async def upload_admission(self):
        raise AssertionError("upload_admission must not be consumed by this path")
        yield  # pragma: no cover


def make_gemma_test_app(api_key=None, max_body=1024, max_upload=2,
                        vlm_state=lambda: "loaded", gates_cls=ResourceGates):
    gates = gates_cls(max_upload_concurrency=max_upload, max_inference_concurrency=1)
    app = Starlette(routes=[
        Route("/api/v1/classify", echo_app, methods=["POST"]),
        Route("/api/v1/gemma/label_scores", echo_app, methods=["POST"]),
        Route("/api/v1/gemma/qa", echo_app, methods=["POST"]),
        Route("/api/v1/gemma/status", health_app, methods=["GET"]),
        Route("/api/v1/gemma/warm", health_app, methods=["POST"]),
        Route("/gemma", health_app, methods=["GET"]),
        Route("/live", health_app, methods=["GET"]),
    ])
    return RequestGateMiddleware(app, gates=gates, api_key=api_key,
                                 max_body_bytes=max_body, vlm_state=vlm_state)


@pytest.mark.anyio
async def test_gemma_upload_route_requires_auth():
    app = make_gemma_test_app(api_key="secret")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/label_scores", content=b"data")
        assert r.status_code == 401


@pytest.mark.anyio
async def test_gemma_upload_route_enforces_body_size():
    app = make_gemma_test_app(api_key=None, max_body=10)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/qa", content=b"x" * 100)
        assert r.status_code == 413


@pytest.mark.anyio
async def test_gemma_cold_post_503_before_body():
    app = make_gemma_test_app(api_key=None, vlm_state=lambda: "idle")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/label_scores", content=b"data")
        assert r.status_code == 503
        assert "Retry-After" in r.headers


@pytest.mark.anyio
async def test_gemma_loading_post_503():
    app = make_gemma_test_app(api_key=None, vlm_state=lambda: "loading")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/qa", content=b"data")
        assert r.status_code == 503


@pytest.mark.anyio
async def test_gemma_status_auth_only_no_upload_gate():
    # NoUploadGates raises if any route consumes upload admission —
    # status must NOT consume an upload slot.
    app = make_gemma_test_app(api_key="secret", gates_cls=NoUploadGates)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/v1/gemma/status", headers={"X-API-Key": "secret"})
        assert r.status_code == 200
        r = await c.get("/api/v1/gemma/status")
        assert r.status_code == 401


@pytest.mark.anyio
async def test_gemma_warm_auth_only():
    app = make_gemma_test_app(api_key="secret", gates_cls=NoUploadGates)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/warm", headers={"X-API-Key": "secret"})
        assert r.status_code == 200
        r = await c.post("/api/v1/gemma/warm")
        assert r.status_code == 401


@pytest.mark.anyio
async def test_gemma_page_passes_through():
    app = make_gemma_test_app(api_key="secret")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/gemma")  # no auth, like /
        assert r.status_code == 200


@pytest.mark.anyio
async def test_classify_unaffected_by_vlm_state():
    app = make_gemma_test_app(api_key=None, vlm_state=lambda: "idle")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/classify", content=b"data")
        assert r.status_code == 200
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_middleware.py -v -k gemma`
Expected: FAIL (`TypeError: unexpected keyword argument 'vlm_state'`)

- [ ] **Step 3: Implement** — in `app/middleware.py`:

Add module-level path sets after `ASGIApp = Callable`:

```python
# Route policy (spec §7): video-upload POST routes get the full gate stack;
# status/warm are auth-only; the /gemma page passes through like /.
GEMMA_UPLOAD_PATHS = frozenset({
    "/api/v1/gemma/label_scores",
    "/api/v1/gemma/qa",
    "/api/v1/gemma/events",
})
GEMMA_AUTH_ONLY_PATHS = frozenset({
    "/api/v1/gemma/status",
    "/api/v1/gemma/warm",
})
```

Extend `__init__`:

```python
    def __init__(
        self,
        app: ASGIApp,
        gates: ResourceGates,
        api_key: str | None,
        max_body_bytes: int,
        vlm_state: Callable[[], str] | None = None,
    ) -> None:
        self._app = app
        self._gates = gates
        self._api_key = api_key
        self._max_body_bytes = max_body_bytes
        self._vlm_state = vlm_state
```

In `__call__`, replace the line `if path == "/api/v1/classify":` with:

```python
        is_gemma_upload = path in GEMMA_UPLOAD_PATHS

        if path == "/api/v1/classify" or is_gemma_upload:
```

and immediately after the auth check inside that branch (after the 401 `return`), insert the pre-body slot gate:

```python
            if is_gemma_upload:
                state = self._vlm_state() if self._vlm_state is not None else "idle"
                if state != "loaded":
                    # Fail fast BEFORE draining the multipart body: a cold
                    # request must not stream 500MB while no model can serve it.
                    payload = json.dumps({
                        "detail": f"Gemma model is not loaded (state: {state}). "
                                  f"Trigger loading via POST /api/v1/gemma/warm and poll "
                                  f"GET /api/v1/gemma/status."
                    }).encode()
                    await send({
                        "type": "http.response.start",
                        "status": 503,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(payload)).encode()),
                            (b"retry-after", b"10"),
                        ],
                    })
                    await send({"type": "http.response.body", "body": payload, "more_body": False})
                    return
```

After the `/api/v1/models` branch, add the auth-only branch:

```python
        # Gemma status/warm: auth only — must NOT consume upload slots or body plumbing
        if path in GEMMA_AUTH_ONLY_PATHS:
            if not self._check_auth(scope):
                await _send_json_response(
                    send,
                    401,
                    {"detail": "Invalid or missing API key. Provide a valid key in the X-API-Key header."},
                )
                return
            await self._app(scope, receive, send)
            return
```

(`/gemma` page route needs no branch — it hits the final pass-through, same as `/`.)

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_middleware.py -v`
Expected: all PASS (pre-existing tests must still pass — `vlm_state` defaults to `None` and only affects gemma paths)

- [ ] **Step 5: Commit**

```bash
git add app/middleware.py tests/test_middleware.py
git commit -m "feat(gemma): middleware route policy with pre-body slot-state gate"
```

---### Task 11: Routes + app wiring

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_gemma_api.py`

**Wiring decisions:** `ResidencyLedger` and `VlmSlot` are created in `create_app` (cheap, no model work) so both the middleware (`vlm_state` callable) and routes share them via closure. The loader closure imports `GemmaVLM` lazily so unit tests never touch transformers' multimodal stack.

- [ ] **Step 1: Write the failing tests** — create `tests/test_gemma_api.py`:

```python
import asyncio
import io
import threading

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import create_app
from app.models.vlm_slot import VlmState


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
    # create_app returns the middleware; the slot is reachable via its app's state dict
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
    # test stubs probe+extract below.
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
        assert "reserve" in r.json()["detail"].lower() or "GB" in r.json()["detail"]


@pytest.mark.anyio
async def test_cold_label_scores_503_with_retry_after(app_and_slot):
    app, slot = app_and_slot
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/label_scores", files=tiny_upload(),
                         data={"labels": '["texting"]'})
        assert r.status_code == 503
        assert r.headers.get("retry-after") == "10"


@pytest.mark.anyio
async def test_label_scores_validates_label_count(app_and_slot, monkeypatch):
    app, slot = app_and_slot
    await force_loaded(slot)
    too_many = "[" + ",".join(f'"label {i}"' for i in range(17)) + "]"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/label_scores", files=tiny_upload(),
                         data={"labels": too_many})
        assert r.status_code == 422
        assert "16" in r.json()["detail"]


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
        md = body["metadata"]
        assert md["score_semantics"] == "gemma4_verbalized_uncalibrated"
        assert md["window_start_seconds"] == 0.0
        assert md["window_end_seconds"] == 30.0
        assert md["frames_analyzed"] == 8
        assert set(md["latency"]) == {"extract_seconds", "generate_seconds", "parse_seconds"}


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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_gemma_api.py -v`
Expected: FAIL (`AttributeError: ... 'vlm_slot_for_tests'` / 404s)

- [ ] **Step 3: Implement** — modify `app/main.py`:

**3a. Imports** — add to the existing import block:

```python
from app.errors.handlers import (
    # ... existing names ...
    GemmaOutputParseError,
    InvalidGemmaParamsError,
)
from app.models.residency import ResidencyLedger
from app.models.model_manager import InsufficientResourcesError  # already imported — keep
from app.models.vlm_slot import VlmSlot, VlmState
from app.schemas.gemma import (
    GemmaLabelScoresResponse,
    GemmaLatency,
    GemmaMetadata,
    GemmaQAResponse,
    GemmaStatusResponse,
)
from app.services.gemma_prompts import (
    build_label_scores_prompt,
    build_qa_prompt,
    label_scores_token_budget,
    parse_label_scores,
)
from app.services.gemma_sampler import (
    extract_frames as gemma_extract_frames,
    plan_timestamps,
    resolve_window,
    validate_gemma_video,
)
```

**3b. Slot + ledger construction** — in `create_app`, after `settings.validate_auth_config()` and before the `state` dict:

```python
    ledger = ResidencyLedger(headroom_gb=settings.residency_headroom_gb)

    def _gemma_loader():
        # Lazy import: unit tests and SigLIP2-only deployments never touch
        # the multimodal transformers stack.
        from app.models.gemma_vlm import GemmaVLM, pick_device_and_dtype  # noqa: F401
        return GemmaVLM(
            hf_repo=settings.gemma_model_id,
            cache_dir=settings.clip_cache_dir,
            image_token_budget=settings.gemma_image_token_budget,
        )

    def _gemma_device() -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"  # mps draws from host RAM; ledger treats both as host memory

    vlm_slot = VlmSlot(
        loader=_gemma_loader,
        ledger=ledger,
        device=_gemma_device(),
        reserve_bytes=int(settings.gemma_reserve_gb * 1e9),
        enabled=settings.gemma_enabled,
    )
```

**3c. Manager gets the ledger** — in `lifespan`, change the `ModelManager(...)` construction to:

```python
        manager = ModelManager(
            cache_dir=settings.clip_cache_dir,
            offline=settings.clipcc_offline,
            ledger=ledger,
        )
```

**3d. Routes** — add after the `serve_vendor` route:

```python
    @app.get("/gemma")
    async def serve_gemma_ui():
        return FileResponse(STATIC_DIR / "gemma.html")

    @app.get("/api/v1/gemma/status", response_model=GemmaStatusResponse)
    async def gemma_status():
        return GemmaStatusResponse(
            enabled=vlm_slot.enabled,
            state=vlm_slot.state.value,
            error=vlm_slot.error,
            model_id=settings.gemma_model_id,
            device=vlm_slot.device,
        )

    @app.post("/api/v1/gemma/warm", status_code=202)
    async def gemma_warm():
        try:
            state_now = await vlm_slot.warm()
        except InsufficientResourcesError as e:
            return JSONResponse(status_code=503, content={"detail": str(e)})
        except RuntimeError as e:
            return JSONResponse(status_code=409, content={"detail": str(e)})
        return JSONResponse(status_code=202, content={"state": state_now.value})

    def _check_gemma_upload(video: UploadFile) -> None:
        filename = video.filename or ""
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(ext if ext else filename)
        if vlm_slot.state != VlmState.LOADED:
            # Defensive double-check; the middleware pre-body gate is primary.
            raise InvalidGemmaParamsError("Gemma model is not loaded.")

    async def _run_gemma_pipeline(
        video: UploadFile,
        window_start: float,
        prompt_for_model: str,
        max_new_tokens: int,
        parse_fn,  # (text) -> parsed | raises ValueError; None = return raw text
    ):
        """Shared label_scores/qa pipeline. Returns (result_text_or_parsed,
        window, n_frames, video_info, latency, parse_retries)."""
        temp_store: TempStore = state["temp_store"]
        gates: ResourceGates = state["gates"]
        model = vlm_slot.model

        request_id = str(uuid.uuid4())
        try:
            stored = await anyio.to_thread.run_sync(
                temp_store.save_upload, request_id, video.file
            )
            video_info = await anyio.to_thread.run_sync(
                partial(probe_video, stored.path, timeout=settings.ffmpeg_timeout_seconds)
            )
            validate_gemma_video(video_info, settings)
            span_start, span_end = resolve_window(
                video_info.duration, window_start, settings.gemma_analysis_window_seconds
            )
            timestamps = plan_timestamps(span_start, span_end, settings.effective_gemma_max_frames)

            async with gates.vlm_admission():
                runner = InferenceRunner(timeout_seconds=settings.request_timeout_seconds)
                deadline = time.monotonic() + settings.request_timeout_seconds

                def pipeline(cancel_event, runner_ref):
                    t0 = time.monotonic()
                    frame_dir = temp_store.create_frame_dir(request_id)
                    frames = gemma_extract_frames(
                        stored.path, timestamps, frame_dir, cancel_event,
                        ffmpeg_timeout=settings.ffmpeg_timeout_seconds, runner=runner_ref,
                    )
                    if not frames:
                        raise NoFramesExtractedError()
                    images = [Image.open(f.path).convert("RGB") for f in frames]
                    t1 = time.monotonic()

                    text = ""
                    parsed = None
                    parse_err = None
                    retries = 0
                    attempts = 2 if parse_fn is not None else 1
                    for attempt in range(attempts):
                        if cancel_event.is_set():
                            break
                        text = model.generate(images, prompt_for_model, max_new_tokens, cancel_event)
                        if parse_fn is None:
                            break
                        try:
                            parsed = parse_fn(text)
                            parse_err = None
                            break
                        except ValueError as e:
                            parse_err = e
                            retries = attempt + 1 if attempt + 1 < attempts else retries
                    t2 = time.monotonic()
                    if parse_fn is not None and parsed is None:
                        raise GemmaOutputParseError(str(parse_err))
                    t3 = time.monotonic()
                    latency = GemmaLatency(
                        extract_seconds=round(t1 - t0, 3),
                        generate_seconds=round(t2 - t1, 3),
                        parse_seconds=round(t3 - t2, 3),
                    )
                    n_retries = attempts - 1 if (parse_fn is not None and retries) else 0
                    return (parsed if parse_fn is not None else text), len(frames), latency, n_retries

                result = await runner.run(pipeline)

                if result is None:
                    # Soft timeout: runner already waited for the worker to
                    # unwind; measure and log the overrun (spec §3 metric).
                    overrun = time.monotonic() - deadline
                    logger.warning(
                        f"Gemma request timed out; worker unwind overran the deadline by {overrun:.1f}s"
                    )
                    raise InferenceTimeoutError(settings.request_timeout_seconds)

            payload, n_frames, latency, parse_retries = result
            return payload, (span_start, span_end), n_frames, video_info, latency, parse_retries
        finally:
            temp_store.cleanup(request_id)

    def _gemma_metadata(window, n_frames, video_info, latency, parse_retries) -> GemmaMetadata:
        return GemmaMetadata(
            model=settings.gemma_model_id,
            device=vlm_slot.model.device if vlm_slot.model else vlm_slot.device,
            frames_analyzed=n_frames,
            window_start_seconds=window[0],
            window_end_seconds=window[1],
            video_duration_seconds=video_info.duration,
            latency=latency,
            parse_retries=parse_retries,
        )

    @app.post("/api/v1/gemma/label_scores", response_model=GemmaLabelScoresResponse)
    async def gemma_label_scores(
        video: UploadFile,
        labels: str = Form(),
        window_start: float = Form(default=0.0, ge=0.0),
    ):
        _check_gemma_upload(video)
        parsed_labels = _parse_label_array(labels, "labels")
        _validate_label_group(parsed_labels, "labels", max_count=settings.gemma_max_labels)

        prompt = build_label_scores_prompt(parsed_labels, settings.gemma_evidence_top_k)
        budget = label_scores_token_budget(len(parsed_labels))

        payload, window, n_frames, video_info, latency, retries = await _run_gemma_pipeline(
            video, window_start, prompt, budget,
            parse_fn=lambda text: parse_label_scores(text, parsed_labels),
        )
        return GemmaLabelScoresResponse(
            scores=payload,
            metadata=_gemma_metadata(window, n_frames, video_info, latency, retries),
        )

    @app.post("/api/v1/gemma/qa", response_model=GemmaQAResponse)
    async def gemma_qa(
        video: UploadFile,
        prompt: str = Form(),
        window_start: float = Form(default=0.0, ge=0.0),
    ):
        _check_gemma_upload(video)
        if not prompt.strip():
            raise InvalidGemmaParamsError("prompt must be a non-empty string.")
        if len(prompt) > 2000:
            raise InvalidGemmaParamsError("prompt must be 2000 characters or fewer.")

        payload, window, n_frames, video_info, latency, retries = await _run_gemma_pipeline(
            video, window_start, build_qa_prompt(prompt),
            settings.gemma_max_new_tokens_qa, parse_fn=None,
        )
        return GemmaQAResponse(
            answer=payload,
            metadata=_gemma_metadata(window, n_frames, video_info, latency, retries),
        )
```

**3e. Middleware wiring + test seam** — replace the final `return RequestGateMiddleware(...)` block:

```python
    middleware = RequestGateMiddleware(
        app=app,
        gates=ResourceGates(
            max_upload_concurrency=settings.effective_upload_concurrency,
            max_inference_concurrency=settings.max_concurrent_requests,
        ),
        api_key=settings.api_key,
        max_body_bytes=settings.max_file_size_bytes,
        vlm_state=lambda: vlm_slot.state.value,
    )
    # Test seam: lets tests inject a fake loader and inspect slot state.
    middleware.vlm_slot_for_tests = vlm_slot
    return middleware
```

Note: `GemmaOutputParseError` raised inside the worker thread propagates through `runner.run` (re-raised from `error_holder`) and, being an `HTTPException`, FastAPI converts it to the 502 response. `NoFramesExtractedError` likewise → 422.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_gemma_api.py tests/test_api.py -v`
Expected: all PASS (existing `/classify` tests unaffected)

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_gemma_api.py
git commit -m "feat(gemma): status/warm/label_scores/qa routes with shared pipeline"
```

---

### Task 12: Web UI — gemma.html + nav

**Files:**
- Create: `app/static/gemma.html`
- Modify: `app/static/index.html` (line ~352, after `<h1>clipCC</h1>`)

- [ ] **Step 1: Add nav to index.html** — directly under `<h1>clipCC</h1>` insert:

```html
<nav style="margin-bottom:16px;border-bottom:1px solid #ddd;padding-bottom:8px;">
  <a href="/" style="font-weight:bold;margin-right:16px;">SigLIP2</a>
  <a href="/gemma" style="margin-right:16px;">Gemma 4</a>
</nav>
```

- [ ] **Step 2: Create `app/static/gemma.html`** (complete file):

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>clipCC — Gemma 4</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #f5f6f8; color: #1c1e21; }
  .container { max-width: 860px; margin: 0 auto; padding: 24px; }
  nav { margin-bottom: 16px; border-bottom: 1px solid #ddd; padding-bottom: 8px; }
  nav a { margin-right: 16px; text-decoration: none; color: #1a73e8; }
  nav a.active { font-weight: bold; color: #1c1e21; }
  .card { background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .status-pill { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 13px; }
  .status-idle { background: #eee; } .status-loading { background: #fff3cd; }
  .status-loaded { background: #d4edda; } .status-failed { background: #f8d7da; }
  label { display: block; margin: 10px 0 4px; font-weight: 600; font-size: 14px; }
  input[type=text], textarea { width: 100%; box-sizing: border-box; padding: 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }
  button { background: #1a73e8; color: #fff; border: none; border-radius: 4px; padding: 10px 18px; font-size: 14px; cursor: pointer; margin-top: 12px; }
  button:disabled { background: #aaa; cursor: not-allowed; }
  .bar-row { display: flex; align-items: center; margin: 6px 0; gap: 8px; }
  .bar-label { width: 220px; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .bar-track { flex: 1; background: #eee; border-radius: 4px; height: 18px; position: relative; }
  .bar-fill { background: #7b1fa2; height: 100%; border-radius: 4px; }
  .bar-value { width: 48px; font-size: 13px; text-align: right; }
  .evidence { font-size: 12px; color: #555; margin: 0 0 8px 228px; white-space: pre-wrap; }
  #qaAnswer { white-space: pre-wrap; font-size: 14px; }
  .meta { font-size: 12px; color: #666; margin-top: 12px; white-space: pre-wrap; }
  .error { color: #b00020; white-space: pre-wrap; font-size: 13px; }
  .disclaimer { font-size: 12px; color: #8a6d3b; background: #fcf8e3; padding: 8px; border-radius: 4px; margin-top: 10px; }
</style>
</head>
<body>
<div class="container">
  <h1>clipCC</h1>
  <nav>
    <a href="/">SigLIP2</a>
    <a href="/gemma" class="active">Gemma 4</a>
  </nav>

  <div class="card">
    <strong>Model:</strong> <span id="modelId"></span>
    <span id="statusPill" class="status-pill status-idle">idle</span>
    <span id="statusError" class="error"></span>
    <div><button id="warmBtn">Warm model</button></div>
  </div>

  <div class="card">
    <label for="videoInput">Video (mp4/avi/mov/mkv; first 60&nbsp;s analyzed by default)</label>
    <input type="file" id="videoInput" accept=".mp4,.avi,.mov,.mkv">
    <label for="windowStart">Analysis window start (seconds)</label>
    <input type="text" id="windowStart" value="0">
    <label for="modeSelect">Mode</label>
    <select id="modeSelect">
      <option value="label_scores">Label scores</option>
      <option value="qa">Ask (free-form Q&amp;A)</option>
    </select>
    <div id="labelsField">
      <label for="labelsInput">Labels (JSON array, max 16)</label>
      <input type="text" id="labelsInput" value='["texting while driving", "sleeping while driving", "eating while driving"]'>
    </div>
    <div id="promptField" style="display:none;">
      <label for="promptInput">Question</label>
      <textarea id="promptInput" rows="2" placeholder="What is the driver doing?"></textarea>
    </div>
    <button id="runBtn" disabled>Analyze</button>
  </div>

  <div class="card" id="resultsCard" style="display:none;">
    <h3>Results</h3>
    <div id="scoresView"></div>
    <div id="qaAnswer"></div>
    <div id="disclaimer" class="disclaimer" style="display:none;"></div>
    <div id="metaView" class="meta"></div>
    <div id="errView" class="error"></div>
  </div>
</div>

<script>
const $ = (id) => document.getElementById(id);
let pollTimer = null;

async function fetchStatus() {
  try {
    const r = await fetch('/api/v1/gemma/status');
    const s = await r.json();
    $('modelId').textContent = s.model_id;
    const pill = $('statusPill');
    pill.textContent = s.state;
    pill.className = 'status-pill status-' + s.state;
    $('statusError').textContent = s.error || '';
    $('runBtn').disabled = s.state !== 'loaded';
    $('warmBtn').disabled = (s.state === 'loading' || s.state === 'loaded');
    if (s.state === 'loading' && !pollTimer) pollTimer = setInterval(fetchStatus, 3000);
    if (s.state !== 'loading' && pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  } catch (e) { $('statusError').textContent = String(e); }
}

$('warmBtn').addEventListener('click', async () => {
  const r = await fetch('/api/v1/gemma/warm', { method: 'POST' });
  if (!r.ok) { const b = await r.json().catch(() => ({})); $('statusError').textContent = b.detail || r.statusText; }
  fetchStatus();
});

$('modeSelect').addEventListener('change', () => {
  const qa = $('modeSelect').value === 'qa';
  $('labelsField').style.display = qa ? 'none' : '';
  $('promptField').style.display = qa ? '' : 'none';
});

function renderScores(scores) {
  const view = $('scoresView');
  view.replaceChildren();
  for (const item of scores) {
    const row = document.createElement('div'); row.className = 'bar-row';
    const lab = document.createElement('div'); lab.className = 'bar-label';
    lab.textContent = item.label;                       // textContent: model output is untrusted
    const track = document.createElement('div'); track.className = 'bar-track';
    const fill = document.createElement('div'); fill.className = 'bar-fill';
    fill.style.width = (item.score === null ? 0 : item.score * 100) + '%';
    track.appendChild(fill);
    const val = document.createElement('div'); val.className = 'bar-value';
    val.textContent = item.score === null ? 'n/a' : item.score.toFixed(2);
    row.append(lab, track, val);
    view.appendChild(row);
    if (item.evidence) {
      const ev = document.createElement('p'); ev.className = 'evidence';
      ev.textContent = item.evidence;                   // textContent, never innerHTML
      view.appendChild(ev);
    }
  }
}

$('runBtn').addEventListener('click', async () => {
  const file = $('videoInput').files[0];
  if (!file) { alert('Choose a video first.'); return; }
  const mode = $('modeSelect').value;
  const fd = new FormData();
  fd.append('video', file);
  fd.append('window_start', $('windowStart').value || '0');
  if (mode === 'label_scores') fd.append('labels', $('labelsInput').value);
  else fd.append('prompt', $('promptInput').value);

  $('resultsCard').style.display = '';
  $('scoresView').replaceChildren(); $('qaAnswer').textContent = '';
  $('metaView').textContent = ''; $('errView').textContent = '';
  $('disclaimer').style.display = 'none';
  $('runBtn').disabled = true;
  const t0 = performance.now();
  try {
    const r = await fetch('/api/v1/gemma/' + mode, { method: 'POST', body: fd });
    const body = await r.json();
    if (!r.ok) { $('errView').textContent = (body.detail || r.statusText); return; }
    if (mode === 'label_scores') {
      renderScores(body.scores);
      $('disclaimer').textContent = body.metadata.disclaimer;
      $('disclaimer').style.display = '';
    } else {
      $('qaAnswer').textContent = body.answer;          // textContent: XSS rule (spec §9)
    }
    const md = body.metadata;
    const wall = ((performance.now() - t0) / 1000).toFixed(1);
    $('metaView').textContent =
      `window: ${md.window_start_seconds}–${md.window_end_seconds}s of ${md.video_duration_seconds.toFixed(1)}s | ` +
      `frames: ${md.frames_analyzed} | device: ${md.device}\n` +
      `latency — extract: ${md.latency.extract_seconds}s, generate: ${md.latency.generate_seconds}s, ` +
      `parse: ${md.latency.parse_seconds}s, wall: ${wall}s | parse retries: ${md.parse_retries}`;
  } catch (e) {
    $('errView').textContent = String(e);
  } finally {
    $('runBtn').disabled = false;
    fetchStatus();
  }
});

fetchStatus();
</script>
</body>
</html>
```

- [ ] **Step 3: Verify** — the page-served test from Task 11 covers routing:

Run: `python -m pytest tests/test_gemma_api.py::test_gemma_page_served -v`
Expected: PASS

Manual check: `ALLOW_UNAUTHENTICATED=true SKIP_MODEL_AUTOLOAD=true uvicorn app.main:create_app --factory --port 8000` → open `http://127.0.0.1:8000/gemma` → status pill shows `idle`, Analyze disabled, nav links work both directions.

- [ ] **Step 4: Commit**

```bash
git add app/static/gemma.html app/static/index.html
git commit -m "feat(gemma): exploration UI page with status-gated submit + nav"
```

---

### Task 13: Dependencies + Docker

**Files:**
- Modify: `requirements.txt`, `Dockerfile`
- Regenerate: `requirements-prod.txt`

- [ ] **Step 1: Add to `requirements.txt`** (next to torch/transformers):

```
accelerate>=1.0.0
torchvision>=0.20.0
```

Do NOT add `librosa` — audio is out of scope (spec §10); its absence is a decision.

- [ ] **Step 2: Regenerate the lock** (pip-compile produced the existing `requirements-prod.txt` — match its invocation; check the file header comment for the exact command, typically):

Run: `pip-compile requirements.txt -o requirements-prod.txt`
Expected: lock now pins `accelerate`, `torchvision` with a version compatible with `torch==2.12.0`.

- [ ] **Step 3: Dockerfile** — change the torch install line (`Dockerfile:19`) so torchvision comes from the SAME variant index (mixed-index torch/torchvision pairs break ABI). Use the exact torchvision pin from the regenerated `requirements-prod.txt` (call it `<TV_PIN>` below — substitute the real value):

```dockerfile
RUN pip install --no-cache-dir torch==2.12.0 torchvision==<TV_PIN> --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    && pip install --no-cache-dir -r requirements-prod.txt
```

- [ ] **Step 4: Verify** — clean venv resolution check:

Run: `pip install --dry-run -r requirements-prod.txt 2>&1 | tail -5`
Expected: no resolution conflicts.

Run: `python -m pytest tests/ -v -x --ignore=tests/test_integration.py -k "not siglip2"`
Expected: full unit suite green.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt requirements-prod.txt Dockerfile
git commit -m "feat(gemma): add accelerate + torchvision (variant-index matched)"
```

---

### Task 14: Gated integration smoke test

**Files:**
- Create: `tests/test_gemma_integration.py`

**Purpose:** the only place real model behavior is verified: `AutoModelForMultimodalLM` + `apply_chat_template` kwargs against the installed transformers, `enable_thinking=False` JSON cleanliness, librosa-not-required, real latency numbers. Skipped unless `GEMMA_INTEGRATION=1` (model download is ~11 GB).

- [ ] **Step 1: Create `tests/test_gemma_integration.py`:**

```python
"""Real-model smoke test. Run explicitly:

    GEMMA_INTEGRATION=1 ALLOW_UNAUTHENTICATED=true python -m pytest tests/test_gemma_integration.py -v -s

Requires ~12GB free memory, ~11GB disk for weights, and ffmpeg.
Exploration deliverables printed: load time, per-stage latency, parse success.
"""
import json
import os
import subprocess
import threading
import time
from pathlib import Path

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


def test_librosa_not_required():
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
```

- [ ] **Step 2: Verify it's skipped in normal runs**

Run: `python -m pytest tests/test_gemma_integration.py -v`
Expected: all SKIPPED.

- [ ] **Step 3: Run for real (on the dev Mac, natively — not Docker; needs ~12GB free):**

Run: `GEMMA_INTEGRATION=1 ALLOW_UNAUTHENTICATED=true python -m pytest tests/test_gemma_integration.py -v -s`
Expected: PASS. If `apply_chat_template`/`AutoModelForMultimodalLM` kwargs fail against the installed transformers, fix `app/models/gemma_vlm.py` against the model card examples (https://huggingface.co/google/gemma-4-E2B-it) — that file is the single place runtime API drift lands.

- [ ] **Step 4: Record exploration numbers** — paste load time, generate latency, and parse success/retry observations into the PR description (these are deliverables, spec Goal section).

- [ ] **Step 5: Commit**

```bash
git add tests/test_gemma_integration.py
git commit -m "test(gemma): env-gated real-model integration smoke"
```

---

### Task 15: Full-suite gate + docs touch-up

- [ ] **Step 1: Full unit suite**

Run: `python -m pytest tests/ -v --ignore=tests/test_integration.py --ignore=tests/test_gemma_integration.py -k "not siglip2"`
Expected: all PASS, no regressions.

- [ ] **Step 2: Update `CLAUDE.md`** — add to the Architecture tree and Key Routes:

```
  models/
    residency.py        # Per-device atomic model-memory ledger
    vlm_slot.py         # Gemma load-once state machine (warm-only)
    gemma_vlm.py        # Gemma 4 E2B wrapper (AutoModelForMultimodalLM)
  services/
    gemma_sampler.py    # Timestamp-seek frame sampling over analysis window
    gemma_prompts.py    # Gemma prompt build + strict ID-keyed parse
```

Routes section:

```
- `POST /api/v1/gemma/{label_scores,qa}` — Gemma 4 E2B exploration (warm first)
- `GET /api/v1/gemma/status` | `POST /api/v1/gemma/warm` — VLM slot lifecycle
- `GET /gemma` — Gemma web UI
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(gemma): CLAUDE.md architecture + routes for Phase A"
```

---

## Out of Plan (per spec)

`/api/v1/gemma/events` (Phase B), compare view (Phase C), logprob scoring, hybrid triage, audio, streaming, quantized weights, hard-deadline subprocess isolation, markdown rendering, ledger unload policy.
