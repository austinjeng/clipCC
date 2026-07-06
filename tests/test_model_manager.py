import asyncio
import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from app.models.model_manager import (
    ModelManager,
    ModelConfig,
    NoModelLoadedError,
    ModelNotCachedError,
    InsufficientResourcesError,
    SIGLIP2_REGISTRY,
)


@pytest.fixture
def manager(temp_dir):
    return ModelManager(cache_dir=str(temp_dir / "models"))


class TestModelConfig:
    def test_revision_field_defaults_none(self):
        config = ModelConfig(
            model_id="test",
            display_name="Test",
            model_type="siglip2",
            hf_repo="google/test",
            params="0.1B",
            resolution=256,
        )
        assert config.revision is None
        assert config.min_ram_gb is None
        assert config.min_vram_gb is None

    def test_revision_field_accepts_value(self):
        config = ModelConfig(
            model_id="test",
            display_name="Test",
            model_type="siglip2",
            hf_repo="google/test",
            params="0.1B",
            resolution=256,
            revision="abc123",
            min_ram_gb=4,
            min_vram_gb=2,
        )
        assert config.revision == "abc123"
        assert config.min_ram_gb == 4
        assert config.min_vram_gb == 2


class TestRegistryExpansion:
    def test_registry_has_8_models(self):
        assert len(SIGLIP2_REGISTRY) == 8

    def test_large_512_in_registry(self):
        config = SIGLIP2_REGISTRY["siglip2-large-patch16-512"]
        assert config.hf_repo == "google/siglip2-large-patch16-512"
        assert config.params == "0.9B"
        assert config.resolution == 512
        assert config.min_ram_gb == 4

    def test_giant_opt_in_registry(self):
        config = SIGLIP2_REGISTRY["siglip2-giant-opt-patch16-384"]
        assert config.hf_repo == "google/siglip2-giant-opt-patch16-384"
        assert config.params == "~2B"
        assert config.resolution == 384
        assert config.min_ram_gb == 10

    def test_existing_models_unchanged(self):
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        assert config.hf_repo == "google/siglip2-base-patch16-256"
        assert config.params == "0.4B"


class TestRegistry:
    def test_registry_has_models(self, manager):
        models = manager.list_models()
        assert len(models) >= 6
        assert any(m["model_id"] == "siglip2-base-patch16-256" for m in models)

    def test_list_models_includes_status(self, manager):
        models = manager.list_models()
        for m in models:
            assert "model_id" in m
            assert "display_name" in m
            assert "loaded" in m
            assert "cached" in m


class TestAcquire:
    @pytest.mark.asyncio
    async def test_acquire_raises_when_no_model(self, manager):
        with pytest.raises(NoModelLoadedError):
            async with manager.acquire(timeout=1.0):
                pass

    @pytest.mark.asyncio
    async def test_acquire_returns_model_after_load(self, manager):
        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_instance = MagicMock()
            mock_instance.model_type = "siglip2"
            MockModel.return_value = mock_instance
            await manager.load_model("siglip2-base-patch16-256")

        async with manager.acquire(timeout=5.0) as lease:
            assert lease.model is mock_instance

    @pytest.mark.asyncio
    async def test_concurrent_leases_allowed(self, manager):
        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            await manager.load_model("siglip2-base-patch16-256")

        async def hold_lease(duration):
            async with manager.acquire(timeout=5.0) as lease:
                await asyncio.sleep(duration)
                return lease.model

        results = await asyncio.gather(
            hold_lease(0.1), hold_lease(0.1), hold_lease(0.1)
        )
        assert all(r is mock_instance for r in results)


class TestLoadModel:
    @pytest.mark.asyncio
    async def test_load_sets_active(self, manager):
        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            await manager.load_model("siglip2-base-patch16-256")

        assert manager.active_model_id == "siglip2-base-patch16-256"
        assert manager.active_model is mock_instance

    @pytest.mark.asyncio
    async def test_load_same_model_noop(self, manager):
        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            await manager.load_model("siglip2-base-patch16-256")
            await manager.load_model("siglip2-base-patch16-256")

        assert MockModel.call_count == 1

    @pytest.mark.asyncio
    async def test_load_waits_for_leases_to_drain(self, manager):
        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_a = MagicMock()
            mock_b = MagicMock()
            MockModel.side_effect = [mock_a, mock_b]
            await manager.load_model("siglip2-base-patch16-256")

        order = []

        async def hold_then_release():
            async with manager.acquire(timeout=5.0) as lease:
                await asyncio.sleep(0.2)
                order.append("lease_released")

        async def swap_model():
            await asyncio.sleep(0.05)
            with patch("app.models.model_manager.SigLip2Model") as MockModel2:
                mock_new = MagicMock()
                MockModel2.return_value = mock_new
                await manager.load_model("siglip2-base-patch16-384")
                order.append("swap_complete")

        await asyncio.gather(hold_then_release(), swap_model())
        assert order == ["lease_released", "swap_complete"]

    @pytest.mark.asyncio
    async def test_load_failure_recovers(self, manager):
        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            await manager.load_model("siglip2-base-patch16-256")

        with patch("app.models.model_manager.SigLip2Model", side_effect=RuntimeError("download failed")):
            with pytest.raises(RuntimeError):
                await manager.load_model("siglip2-base-patch16-384")

        assert manager._swapping is False
        assert manager.active_model is None

    @pytest.mark.asyncio
    async def test_invalid_model_id_raises(self, manager):
        with pytest.raises(KeyError):
            await manager.load_model("nonexistent-model")


class TestErrorTypes:
    def test_model_not_cached_error_is_exception(self):
        err = ModelNotCachedError("siglip2-base-patch16-256")
        assert isinstance(err, Exception)
        assert "siglip2-base-patch16-256" in str(err)

    def test_insufficient_resources_error_is_exception(self):
        err = InsufficientResourcesError("Need 10GB RAM, only 4GB available")
        assert isinstance(err, Exception)
        assert "10GB" in str(err)


class TestCacheValidation:
    def _write_marker(self, cache_dir, hf_repo, marker_data):
        """Helper: write a .validated marker for a model."""
        model_dir = Path(cache_dir) / f"models--{hf_repo.replace('/', '--')}"
        model_dir.mkdir(parents=True, exist_ok=True)
        marker_path = model_dir / ".validated"
        marker_path.write_text(json.dumps(marker_data))

    def test_not_cached_when_no_directory(self, manager):
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        assert manager._is_cached(config) is False

    def test_not_cached_when_no_marker(self, manager):
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        model_dir = Path(manager.cache_dir) / "models--google--siglip2-base-patch16-256"
        model_dir.mkdir(parents=True)
        assert manager._is_cached(config) is False

    def test_cached_when_valid_marker(self, manager):
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        self._write_marker(manager.cache_dir, config.hf_repo, {
            "schema_version": 1,
            "model_id": "siglip2-base-patch16-256",
            "hf_repo": "google/siglip2-base-patch16-256",
            "revision": "abc123",
            "validated_at": "2026-05-19T10:00:00Z",
        })
        assert manager._is_cached(config) is True

    def test_not_cached_when_model_id_mismatch(self, manager):
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        self._write_marker(manager.cache_dir, config.hf_repo, {
            "schema_version": 1,
            "model_id": "wrong-model-id",
            "hf_repo": "google/siglip2-base-patch16-256",
            "revision": "abc123",
            "validated_at": "2026-05-19T10:00:00Z",
        })
        assert manager._is_cached(config) is False

    def test_not_cached_when_hf_repo_mismatch(self, manager):
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        self._write_marker(manager.cache_dir, config.hf_repo, {
            "schema_version": 1,
            "model_id": "siglip2-base-patch16-256",
            "hf_repo": "wrong/repo",
            "revision": "abc123",
            "validated_at": "2026-05-19T10:00:00Z",
        })
        assert manager._is_cached(config) is False

    def test_not_cached_when_revision_mismatch(self, manager):
        from dataclasses import replace
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        pinned = replace(config, revision="expected-sha")
        self._write_marker(manager.cache_dir, config.hf_repo, {
            "schema_version": 1,
            "model_id": "siglip2-base-patch16-256",
            "hf_repo": "google/siglip2-base-patch16-256",
            "revision": "different-sha",
            "validated_at": "2026-05-19T10:00:00Z",
        })
        assert manager._is_cached(pinned) is False

    def test_cached_when_no_pinned_revision(self, manager):
        """Config has revision=None, marker has a revision -- still valid."""
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        assert config.revision is None
        self._write_marker(manager.cache_dir, config.hf_repo, {
            "schema_version": 1,
            "model_id": "siglip2-base-patch16-256",
            "hf_repo": "google/siglip2-base-patch16-256",
            "revision": "abc123",
            "validated_at": "2026-05-19T10:00:00Z",
        })
        assert manager._is_cached(config) is True

    def test_not_cached_when_marker_is_corrupt_json(self, manager):
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        model_dir = Path(manager.cache_dir) / "models--google--siglip2-base-patch16-256"
        model_dir.mkdir(parents=True)
        (model_dir / ".validated").write_text("not json{{{")
        assert manager._is_cached(config) is False


class TestResourceGating:
    @pytest.mark.asyncio
    async def test_load_rejects_when_total_ram_insufficient(self, manager):
        """Tier 1: model needs more RAM than the system has."""
        config = SIGLIP2_REGISTRY["siglip2-giant-opt-patch16-384"]
        assert config.min_ram_gb == 10

        with patch("app.models.model_manager.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = MagicMock(
                total=4 * 1e9, available=4 * 1e9
            )
            with pytest.raises(InsufficientResourcesError):
                await manager.load_model("siglip2-giant-opt-patch16-384")

    @pytest.mark.asyncio
    async def test_load_estimates_post_unload_capacity(self, manager):
        """Tier 2: available RAM is low but unloading active model would free enough."""
        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            await manager.load_model("siglip2-base-patch16-256")

        with patch("app.models.model_manager.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = MagicMock(
                total=16 * 1e9,
                available=2 * 1e9,
            )
            with patch("app.models.model_manager.SigLip2Model") as MockModel2:
                mock_new = MagicMock()
                MockModel2.return_value = mock_new
                await manager.load_model("siglip2-base-patch16-384")
                assert manager.active_model_id == "siglip2-base-patch16-384"

    @pytest.mark.asyncio
    async def test_load_no_gate_when_min_ram_is_none(self, manager):
        """Models without min_ram_gb skip the resource check."""
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        assert config.min_ram_gb is None

        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            await manager.load_model("siglip2-base-patch16-256")
        assert manager.active_model_id == "siglip2-base-patch16-256"


class TestSafeModelSwap:
    @pytest.mark.asyncio
    async def test_preflight_failure_preserves_active_model(self, manager):
        """If preflight fails, the currently loaded model stays active."""
        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            await manager.load_model("siglip2-base-patch16-256")

        assert manager.active_model_id == "siglip2-base-patch16-256"
        assert manager.active_model is mock_instance

        with patch("app.models.model_manager.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = MagicMock(
                total=4 * 1e9, available=4 * 1e9
            )
            with pytest.raises(InsufficientResourcesError):
                await manager.load_model("siglip2-giant-opt-patch16-384")

        assert manager.active_model_id == "siglip2-base-patch16-256"
        assert manager.active_model is mock_instance


class TestOfflineVisibility:
    def test_online_list_models_returns_all(self, manager):
        models = manager.list_models()
        assert len(models) == 8

    def test_offline_list_models_excludes_uncached(self, temp_dir):
        mgr = ModelManager(cache_dir=str(temp_dir / "models"), offline=True)
        models = mgr.list_models()
        assert len(models) == 0

    def test_offline_list_models_includes_cached(self, temp_dir):
        cache_dir = str(temp_dir / "models")
        mgr = ModelManager(cache_dir=cache_dir, offline=True)
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        model_dir = Path(cache_dir) / "models--google--siglip2-base-patch16-256"
        model_dir.mkdir(parents=True)
        (model_dir / ".validated").write_text(json.dumps({
            "schema_version": 1,
            "model_id": "siglip2-base-patch16-256",
            "hf_repo": "google/siglip2-base-patch16-256",
            "revision": "abc123",
            "validated_at": "2026-05-19T10:00:00Z",
        }))
        models = mgr.list_models()
        assert len(models) == 1
        assert models[0]["model_id"] == "siglip2-base-patch16-256"


class TestOfflinePreflight:
    @pytest.mark.asyncio
    async def test_offline_load_uncached_raises(self, temp_dir):
        mgr = ModelManager(cache_dir=str(temp_dir / "models"), offline=True)
        with pytest.raises(ModelNotCachedError):
            await mgr.load_model("siglip2-base-patch16-256")

    @pytest.mark.asyncio
    async def test_offline_load_uncached_preserves_active(self, temp_dir):
        mgr = ModelManager(cache_dir=str(temp_dir / "models"), offline=True)
        mock_model = MagicMock()
        mgr.active_model = mock_model
        mgr.active_model_id = "siglip2-base-patch16-384"

        with pytest.raises(ModelNotCachedError):
            await mgr.load_model("siglip2-base-patch16-256")

        assert mgr.active_model is mock_model
        assert mgr.active_model_id == "siglip2-base-patch16-384"


class TestManifestLoading:
    def test_loads_revisions_from_manifest(self, temp_dir):
        cache_dir = str(temp_dir / "models")
        Path(cache_dir).mkdir(parents=True)
        manifest = {
            "siglip2-base-patch16-256": {
                "revision": "sha-abc123",
                "hf_repo": "google/siglip2-base-patch16-256",
                "validated_at": "2026-05-19T10:00:00Z",
            }
        }
        (Path(cache_dir) / "manifest.json").write_text(json.dumps(manifest))

        mgr = ModelManager(cache_dir=cache_dir)
        config = mgr.registry["siglip2-base-patch16-256"]
        assert config.revision == "sha-abc123"

    def test_no_manifest_leaves_revisions_none(self, temp_dir):
        mgr = ModelManager(cache_dir=str(temp_dir / "models"))
        config = mgr.registry["siglip2-base-patch16-256"]
        assert config.revision is None

    def test_manifest_only_updates_known_models(self, temp_dir):
        cache_dir = str(temp_dir / "models")
        Path(cache_dir).mkdir(parents=True)
        manifest = {
            "unknown-model": {
                "revision": "sha-xyz",
                "hf_repo": "google/unknown",
                "validated_at": "2026-05-19T10:00:00Z",
            }
        }
        (Path(cache_dir) / "manifest.json").write_text(json.dumps(manifest))

        mgr = ModelManager(cache_dir=cache_dir)
        assert len(mgr.registry) == 8

    def test_corrupt_manifest_ignored(self, temp_dir):
        cache_dir = str(temp_dir / "models")
        Path(cache_dir).mkdir(parents=True)
        (Path(cache_dir) / "manifest.json").write_text("not valid json{{{")

        mgr = ModelManager(cache_dir=cache_dir)
        config = mgr.registry["siglip2-base-patch16-256"]
        assert config.revision is None


class TestOfflineIntegration:
    @pytest.mark.asyncio
    async def test_full_offline_flow(self, temp_dir):
        """Simulates: download script cached a model, app runs offline, loads it."""
        cache_dir = str(temp_dir / "models")

        # Simulate what download script does: write marker + manifest
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        model_dir = Path(cache_dir) / f"models--{config.hf_repo.replace('/', '--')}"
        model_dir.mkdir(parents=True)
        marker = {
            "schema_version": 1,
            "model_id": config.model_id,
            "hf_repo": config.hf_repo,
            "revision": "test-sha-abc",
            "validated_at": "2026-05-19T10:00:00Z",
        }
        (model_dir / ".validated").write_text(json.dumps(marker))
        manifest = {
            config.model_id: {
                "revision": "test-sha-abc",
                "hf_repo": config.hf_repo,
                "validated_at": "2026-05-19T10:00:00Z",
            }
        }
        (Path(cache_dir) / "manifest.json").write_text(json.dumps(manifest))

        # Create offline manager
        mgr = ModelManager(cache_dir=cache_dir, offline=True)

        # Verify manifest loaded revision
        assert mgr.registry[config.model_id].revision == "test-sha-abc"

        # Verify list_models only shows cached model
        models = mgr.list_models()
        cached_ids = [m["model_id"] for m in models]
        assert config.model_id in cached_ids
        assert len(cached_ids) == 1

        # Verify uncached model load is rejected
        with pytest.raises(ModelNotCachedError):
            await mgr.load_model("siglip2-base-patch16-384")

        # Verify cached model passes preflight (mock actual model construction)
        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            await mgr.load_model(config.model_id)

        assert mgr.active_model_id == config.model_id
        assert mgr.active_model is mock_instance


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


@pytest.mark.asyncio
async def test_ledger_releases_siglip2_reservation_on_swap_to_no_min_ram(monkeypatch, tmp_path):
    """Swapping from a min_ram_gb model to one without must release the stale
    'siglip2' reservation, not leave phantom bytes the resource check subtracts."""
    from app.models.residency import ResidencyLedger
    import app.models.model_manager as mm

    ledger = ResidencyLedger(headroom_gb=0.0)
    ledger._device_free = lambda device: 50_000_000_000  # plenty for replace()
    manager = ModelManager(cache_dir=str(tmp_path), ledger=ledger)

    class FakeMem:
        total = 64_000_000_000
        available = 32_000_000_000

    monkeypatch.setattr(mm.psutil, "virtual_memory", lambda: FakeMem())

    with patch("app.models.model_manager.SigLip2Model") as MockModel:
        MockModel.return_value = MagicMock()
        # Load a model WITH min_ram_gb=4 → creates the 'siglip2' reservation.
        await manager.load_model("siglip2-large-patch16-512")
        assert ledger.reserved_bytes("cpu") == int(4 * 1e9)

        # Swap to a model WITHOUT min_ram_gb → reservation must be released.
        await manager.load_model("siglip2-base-patch16-256")
        assert ledger.reserved_bytes("cpu") == 0

        # And a subsequent large load must see the full budget (would spuriously
        # fail with InsufficientResourcesError if the 4GB phantom lingered).
        await manager.load_model("siglip2-large-patch16-512")
        assert manager.active_model_id == "siglip2-large-patch16-512"
        assert ledger.reserved_bytes("cpu") == int(4 * 1e9)


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


class TestLoadModelEventLoop:
    async def test_load_does_not_block_event_loop(self, manager):
        # The blocking constructor must run in a worker thread: the event loop
        # (and /live, /ready) has to stay responsive during a slow model load.
        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            def slow_constructor(**kwargs):
                time.sleep(0.5)
                return MagicMock()

            MockModel.side_effect = slow_constructor

            ticks = 0

            async def heartbeat():
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.02)
                    ticks += 1

            hb = asyncio.create_task(heartbeat())
            try:
                await manager.load_model("siglip2-base-patch16-256")
            finally:
                hb.cancel()
            assert ticks >= 5
