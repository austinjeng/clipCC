import asyncio
import json
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
