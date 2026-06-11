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


def test_commit_unknown_owner_raises_keyerror():
    ledger = make_ledger({"cpu": 10_000_000_000})
    with pytest.raises(KeyError):
        ledger.commit("nobody")
