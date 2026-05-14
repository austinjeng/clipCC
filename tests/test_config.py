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
