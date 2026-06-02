import json
import numpy as np
from pathlib import Path
from tools.android_assets.gen_fixtures import gen_all

FRAMES = Path("tools/android_assets/tests/fixtures/lossless")
LABELS = ["Car", "texting while driving", "a dog"]
HF_REPO = "google/siglip2-base-patch16-256"  # reference = HF torch model (cached after first run)


def test_gen_all_produces_consistent_fixtures(tmp_path):
    out = gen_all(HF_REPO, FRAMES, LABELS, tmp_path)

    tok = json.loads((tmp_path / "tokenizer_golden.json").read_text())
    assert all(len(r["input_ids"]) == 64 for r in tok)
    assert tok[0]["text"] == "Car"

    npz = np.load(tmp_path / "preprocess_golden.npz")
    res = 256
    assert npz["pixel_values"].shape[1:] == (3, res, res)   # CHW
    assert float(npz["pixel_values"].min()) >= -1.0001
    assert float(npz["pixel_values"].max()) <= 1.0001

    scores = json.loads((tmp_path / "scores_golden.json").read_text())
    n_frames = len(list(FRAMES.glob("*.png")))
    assert len(scores["confidence"]) == n_frames
    assert len(scores["confidence"][0]) == len(LABELS)
    for row in scores["confidence"]:
        for v in row:
            assert 0.0 <= v <= 1.0
    # cosine is half the SigLIP2 numerical contract — gate it too
    assert len(scores["cosine"]) == n_frames
    assert len(scores["cosine"][0]) == len(LABELS)
    for row in scores["cosine"]:
        for v in row:
            assert -1.0001 <= v <= 1.0001

    # exact resample contract captured for the Android bicubic target (M16)
    rs = json.loads((tmp_path / "resample_contract.json").read_text())
    assert rs["resample"] in ("bicubic", 3, "bilinear", 2)  # slow processor returns 2 (BILINEAR)
    assert rs["size"]["height"] == res and rs["size"]["width"] == res
    assert rs["image_mean"] == [0.5, 0.5, 0.5] and rs["image_std"] == [0.5, 0.5, 0.5]
