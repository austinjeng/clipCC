import asyncio

import anyio
import pytest

from app.models.residency import ResidencyLedger
from app.models.model_manager import InsufficientResourcesError
from app.models.vlm_slot import VlmSlot, VlmState


@pytest.fixture
def anyio_backend():
    return "asyncio"


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


@pytest.mark.anyio
async def test_cancel_during_load_propagates_and_rolls_back():
    ledger = make_ledger()
    loading = anyio.Event()
    hold = anyio.Event()

    def slow_loader():
        anyio.from_thread.run_sync(loading.set)
        anyio.from_thread.run(hold.wait)
        return object()

    slot = make_slot(loader=slow_loader, ledger=ledger)
    await slot.warm()
    await loading.wait()
    slot._load_task.cancel()
    hold.set()  # let the non-cancellable worker thread finish so cancel can deliver
    with pytest.raises(asyncio.CancelledError):
        await slot._load_task
    assert slot._load_task.cancelled()
    assert ledger.reserved_bytes("cpu") == 0  # rolled back


@pytest.mark.anyio
async def test_aclose_noop_when_idle():
    slot = make_slot(loader=lambda: object())
    await slot.aclose()  # never warmed: must be a clean no-op
    assert slot.state == VlmState.IDLE


@pytest.mark.anyio
async def test_aclose_drains_completed_load():
    slot = make_slot(loader=lambda: object())
    await slot.warm()
    await slot.wait_settled()
    assert slot.state == VlmState.LOADED
    await slot.aclose()  # already settled: must not raise
    assert slot._load_task.done()


@pytest.mark.anyio
async def test_aclose_cancels_and_drains_inflight_load():
    # Shutdown while a warm is in flight must drain the task (no pending-task
    # warning) and roll back the residency reservation.
    ledger = make_ledger()
    loading = anyio.Event()
    hold = anyio.Event()

    def slow_loader():
        anyio.from_thread.run_sync(loading.set)
        anyio.from_thread.run(hold.wait)
        return object()

    slot = make_slot(loader=slow_loader, ledger=ledger)
    await slot.warm()
    await loading.wait()
    aclose_task = asyncio.create_task(slot.aclose())
    await asyncio.sleep(0.05)  # let aclose issue the cancel
    hold.set()  # unblock the non-cancellable worker thread so cancel can deliver
    await aclose_task  # must return cleanly (CancelledError swallowed)
    assert slot._load_task.done()
    assert ledger.reserved_bytes("cpu") == 0  # rolled back on cancel


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
