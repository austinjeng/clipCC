import hashlib
from tools.android_assets.hashing import sha256_file


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"clipcc" * 100000)  # multi-chunk
    expected = hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha256_file(p) == expected
