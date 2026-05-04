from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    max_file_size_mb: int = 500
    max_duration_seconds: int = 300
    max_frames: int = 300
    default_fps: float = 1.0
    batch_size: int = 32
    max_concurrent_requests: int = 2
    max_upload_concurrency: int | None = None
    api_key: str | None = None
    allow_unauthenticated: bool = False
    ffmpeg_timeout_seconds: int = 120
    request_timeout_seconds: int = 300
    clip_cache_dir: str = "/app/models"
    temp_dir: str = "/tmp/clipcc"
    default_model_id: str = "siglip2-base-patch16-256"

    model_config = {"env_prefix": "", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def effective_upload_concurrency(self) -> int:
        if self.max_upload_concurrency is not None:
            return self.max_upload_concurrency
        return self.max_concurrent_requests + 2

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    def validate_auth_config(self) -> None:
        if self.api_key is None and not self.allow_unauthenticated:
            raise RuntimeError(
                "Authentication is not configured: set API_KEY or set ALLOW_UNAUTHENTICATED=true"
            )
