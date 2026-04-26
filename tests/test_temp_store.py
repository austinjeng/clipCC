import io
from pathlib import Path
import pytest
from app.temp_store import TempStore, StoredUpload

@pytest.fixture
def store(temp_dir):
    return TempStore(base_dir=temp_dir / "temp")

def test_save_upload_creates_file(store):
    content = b"fake video content" * 1000
    upload_file = io.BytesIO(content)
    result = store.save_upload("req-1", upload_file)
    assert isinstance(result, StoredUpload)
    assert result.path.exists()
    assert result.size == len(content)

def test_save_upload_streams_in_chunks(store):
    content = b"x" * (128 * 1024)
    upload_file = io.BytesIO(content)
    result = store.save_upload("req-1", upload_file, chunk_size=64 * 1024)
    assert result.path.read_bytes() == content

def test_create_frame_dir(store):
    frame_dir = store.create_frame_dir("req-1")
    assert frame_dir.is_dir()

def test_cleanup_removes_all_request_files(store):
    content = b"video"
    store.save_upload("req-1", io.BytesIO(content))
    frame_dir = store.create_frame_dir("req-1")
    (frame_dir / "frame_00001.jpg").write_bytes(b"jpeg")
    store.cleanup("req-1")
    assert not any(store.base_dir.glob("req-1*"))

def test_janitor_removes_old_files(store):
    import os, time
    old_dir = store.base_dir / "old-req"
    old_dir.mkdir(parents=True)
    (old_dir / "video.mp4").write_bytes(b"old")
    one_hour_ago = time.time() - 3601
    os.utime(old_dir, (one_hour_ago, one_hour_ago))
    store.run_janitor(max_age_seconds=3600)
    assert not old_dir.exists()
