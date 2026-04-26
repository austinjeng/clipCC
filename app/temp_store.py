import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass
class StoredUpload:
    path: Path
    size: int


class TempStore:
    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(
        self, request_id: str, file: BinaryIO, chunk_size: int = 64 * 1024
    ) -> StoredUpload:
        request_dir = self.base_dir / request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        dest = request_dir / f"upload_{uuid.uuid4().hex[:8]}.tmp"
        total = 0
        with open(dest, "wb") as f:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
        return StoredUpload(path=dest, size=total)

    def create_frame_dir(self, request_id: str) -> Path:
        frame_dir = self.base_dir / request_id / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        return frame_dir

    def cleanup(self, request_id: str) -> None:
        request_dir = self.base_dir / request_id
        if request_dir.exists():
            shutil.rmtree(request_dir, ignore_errors=True)

    def run_janitor(self, max_age_seconds: int = 3600) -> None:
        if not self.base_dir.exists():
            return
        cutoff = time.time() - max_age_seconds
        for child in self.base_dir.iterdir():
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
