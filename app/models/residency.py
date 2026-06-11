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
    committed: bool = False  # Set on success; reserved for future auditing/assertions.


class ResidencyLedger:
    """Per-device, atomic model-memory accounting.

    Protocol: reserve(owner, device, nbytes) before loading (atomic
    check-and-reserve against device_free - other_reservations - headroom),
    then commit(owner) on load success or rollback(owner) on failure.
    release(owner) frees a committed reservation (e.g. on model swap).
    commit() raises KeyError if the owner has no active reservation;
    rollback() and release() are idempotent.

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
