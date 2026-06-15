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

    async def aclose(self) -> None:
        """Cancel and drain the in-flight load task so the event loop never tears
        down with a pending task; _load rolls back the residency reservation on
        cancellation. A no-op when no load is in flight. Note: a load already
        running inside the worker thread is non-cancellable, so this awaits it to
        settle before returning."""
        task = self._load_task
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"VLM load task error during shutdown: {e}")

    async def _load(self) -> None:
        try:
            model = await anyio.to_thread.run_sync(self._loader)
        except BaseException as e:
            self._ledger.rollback("vlm")
            if not isinstance(e, Exception):
                raise  # CancelledError etc. must propagate — never swallow cancellation
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
