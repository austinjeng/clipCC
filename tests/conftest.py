from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.config import Settings


@pytest.fixture(autouse=True)
def _ignore_local_dotenv(monkeypatch):
    """Stop a developer's local .env from leaking into Settings() during tests.

    Settings.model_config pins env_file='.env'. CI has no .env so tests pass,
    but locally an ambient .env (e.g. CLIPCC_OFFLINE=1, ALLOW_UNAUTHENTICATED=true)
    overrides fixture defaults and breaks otherwise-green tests. Disable dotenv
    loading for every test so behavior matches CI.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def settings(temp_dir: Path) -> Settings:
    return Settings(
        max_file_size_mb=10,
        max_duration_seconds=30,
        max_frames=30,
        default_fps=1.0,
        batch_size=8,
        max_concurrent_requests=1,
        allow_unauthenticated=True,
        clip_cache_dir=str(temp_dir / "models"),
        temp_dir=str(temp_dir / "tmp"),
    )


@pytest.fixture
def small_video(temp_dir: Path) -> Path:
    output = temp_dir / "test_video.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", "color=c=blue:s=64x64:d=3",
            "-c:v", "libx264",
            "-t", "3",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    return output
