import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class TestPresets:
    def test_dev_preset_has_4_models(self):
        from download_models import DEV_MODELS
        assert len(DEV_MODELS) == 4
        assert "siglip2-base-patch16-256" in DEV_MODELS
        assert "siglip2-large-patch16-512" in DEV_MODELS
        assert "siglip2-so400m-patch14-384" in DEV_MODELS
        assert "siglip2-giant-opt-patch16-384" in DEV_MODELS

    def test_all_preset_matches_registry(self):
        from download_models import get_models_for_preset
        from app.models.model_manager import SIGLIP2_REGISTRY
        models = get_models_for_preset("all")
        assert set(models) == set(SIGLIP2_REGISTRY.keys())

    def test_dev_preset_returns_subset(self):
        from download_models import get_models_for_preset
        models = get_models_for_preset("dev")
        assert len(models) == 4


class TestResolveDefaultCacheDir:
    def test_resolves_to_repo_root(self):
        from download_models import resolve_default_cache_dir
        result = resolve_default_cache_dir()
        assert result.name == "models"
        assert result.parent == Path(__file__).resolve().parent.parent


class TestWriteMarker:
    def test_writes_valid_json_marker(self, tmp_path):
        from download_models import write_validated_marker
        model_dir = tmp_path / "models--google--test"
        model_dir.mkdir()
        write_validated_marker(
            model_dir=model_dir,
            model_id="test",
            hf_repo="google/test",
            revision="abc123",
        )
        marker = json.loads((model_dir / ".validated").read_text())
        assert marker["schema_version"] == 1
        assert marker["model_id"] == "test"
        assert marker["hf_repo"] == "google/test"
        assert marker["revision"] == "abc123"
        assert "validated_at" in marker

    def test_atomic_write(self, tmp_path):
        from download_models import write_validated_marker
        model_dir = tmp_path / "models--google--test"
        model_dir.mkdir()
        write_validated_marker(
            model_dir=model_dir,
            model_id="test",
            hf_repo="google/test",
            revision="abc123",
        )
        marker_path = model_dir / ".validated"
        assert marker_path.exists()
        json.loads(marker_path.read_text())


class TestWriteManifest:
    def test_writes_manifest_json(self, tmp_path):
        from download_models import write_manifest
        records = {
            "model-a": {"revision": "sha1", "hf_repo": "google/a", "validated_at": "2026-05-19T10:00:00Z"},
            "model-b": {"revision": "sha2", "hf_repo": "google/b", "validated_at": "2026-05-19T10:00:00Z"},
        }
        write_manifest(tmp_path, records)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["model-a"]["revision"] == "sha1"
        assert manifest["model-b"]["revision"] == "sha2"
