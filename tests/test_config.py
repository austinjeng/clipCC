from __future__ import annotations

import pytest

from app.config import Settings


def test_default_settings() -> None:
    s = Settings()
    assert s.max_file_size_mb == 500
    assert s.max_duration_seconds == 300
    assert s.max_frames == 300
    assert s.default_fps == 1.0
    assert s.batch_size == 32
    assert s.max_concurrent_requests == 2
    assert s.max_upload_concurrency is None
    assert s.api_key is None
    assert s.allow_unauthenticated is False
    assert s.ffmpeg_timeout_seconds == 120
    assert s.request_timeout_seconds == 300
    assert s.clip_cache_dir == "/app/models"
    assert s.temp_dir == "/tmp/clipcc"
    assert s.clipcc_offline is False


def test_custom_upload_concurrency() -> None:
    s = Settings(max_upload_concurrency=5)
    assert s.effective_upload_concurrency == 5

    s2 = Settings(max_concurrent_requests=3)
    assert s2.effective_upload_concurrency == 5  # 3 + 2


def test_max_file_size_bytes() -> None:
    s = Settings(max_file_size_mb=100)
    assert s.max_file_size_bytes == 100 * 1024 * 1024


def test_validate_auth_config_fails_without_key_or_flag() -> None:
    s = Settings(api_key=None, allow_unauthenticated=False)
    with pytest.raises(RuntimeError):
        s.validate_auth_config()


def test_validate_auth_config_passes_with_key() -> None:
    s = Settings(api_key="secret-key", allow_unauthenticated=False)
    s.validate_auth_config()  # should not raise


def test_validate_auth_config_passes_with_flag() -> None:
    s = Settings(api_key=None, allow_unauthenticated=True)
    s.validate_auth_config()  # should not raise


def test_default_model_id_default():
    s = Settings(allow_unauthenticated=True)
    assert s.default_model_id == "siglip2-base-patch16-256"


def test_default_model_id_from_env(monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL_ID", "siglip2-large-patch16-384")
    s = Settings(allow_unauthenticated=True)
    assert s.default_model_id == "siglip2-large-patch16-384"


def test_skip_model_autoload_default():
    s = Settings(allow_unauthenticated=True)
    assert s.skip_model_autoload is False


def test_skip_model_autoload_from_env(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL_AUTOLOAD", "true")
    s = Settings(allow_unauthenticated=True)
    assert s.skip_model_autoload is True


def test_default_labels_default():
    s = Settings(allow_unauthenticated=True)
    assert s.default_labels == [
        "texting while driving",
        "sleeping while driving",
        "eating while driving",
    ]


def test_default_labels_from_env(monkeypatch):
    monkeypatch.setenv("DEFAULT_LABELS", '["label1", "label2"]')
    s = Settings(allow_unauthenticated=True)
    assert s.default_labels == ["label1", "label2"]


class TestOfflineSetting:
    def test_offline_defaults_false(self):
        s = Settings(allow_unauthenticated=True)
        assert s.clipcc_offline is False

    def test_offline_from_env(self, monkeypatch):
        monkeypatch.setenv("CLIPCC_OFFLINE", "1")
        s = Settings(allow_unauthenticated=True)
        assert s.clipcc_offline is True


def test_gemma_defaults():
    s = Settings(allow_unauthenticated=True)
    assert s.gemma_model_id == "google/gemma-4-E2B-it"
    assert s.gemma_enabled is True
    assert s.gemma_max_frames == 8
    assert s.gemma_max_frames_cap == 16
    assert s.gemma_max_labels == 50
    assert s.gemma_analysis_window_seconds == 60.0
    assert s.gemma_max_new_tokens_qa == 400
    assert s.gemma_image_token_budget == 280
    assert s.gemma_evidence_top_k == 3
    assert s.gemma_reserve_gb == 12.0
    assert s.residency_headroom_gb == 2.0


def test_gemma_model_id_from_env(monkeypatch):
    monkeypatch.setenv("GEMMA_MODEL_ID", "google/gemma-4-E4B-it")
    s = Settings(allow_unauthenticated=True)
    assert s.gemma_model_id == "google/gemma-4-E4B-it"


def test_gemma_enabled_from_env(monkeypatch):
    monkeypatch.setenv("GEMMA_ENABLED", "false")
    s = Settings(allow_unauthenticated=True)
    assert s.gemma_enabled is False


def test_gemma_max_frames_clamped_to_cap():
    s = Settings(allow_unauthenticated=True, gemma_max_frames=99)
    assert s.effective_gemma_max_frames == 16
    assert Settings(allow_unauthenticated=True, gemma_max_frames=5).effective_gemma_max_frames == 5
