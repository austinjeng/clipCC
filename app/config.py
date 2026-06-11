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
    skip_model_autoload: bool = False
    clipcc_offline: bool = False
    default_labels: list[str] = [
        "texting while driving",
        "sleeping while driving",
        "eating while driving",
    ]
    # --- Gemma 4 E2B exploration (spec: docs/superpowers/specs/2026-06-12-gemma4-e2b-exploration-design.md) ---
    gemma_model_id: str = "google/gemma-4-E2B-it"
    gemma_enabled: bool = True
    gemma_max_frames: int = 8
    gemma_max_frames_cap: int = 16
    gemma_max_labels: int = 50
    gemma_analysis_window_seconds: float = 60.0
    gemma_max_new_tokens_qa: int = 400
    gemma_image_token_budget: int = 280
    gemma_evidence_top_k: int = 3
    # 11.4 GB bf16 weights + KV/activations margin; reserved in the residency ledger
    gemma_reserve_gb: float = 12.0
    residency_headroom_gb: float = 2.0

    model_config = {"env_prefix": "", "env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def effective_upload_concurrency(self) -> int:
        if self.max_upload_concurrency is not None:
            return self.max_upload_concurrency
        return self.max_concurrent_requests + 2

    @property
    def effective_gemma_max_frames(self) -> int:
        return min(self.gemma_max_frames, self.gemma_max_frames_cap)

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    def validate_auth_config(self) -> None:
        if self.api_key is None and not self.allow_unauthenticated:
            raise RuntimeError(
                "Authentication is not configured: set API_KEY or set ALLOW_UNAUTHENTICATED=true"
            )
