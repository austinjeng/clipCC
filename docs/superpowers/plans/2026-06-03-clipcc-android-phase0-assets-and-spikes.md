# clipCC-Android Phase 0 — Model Assets & De-Risking Spikes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the four SigLIP2 ONNX model bundles + generated manifests + lossless golden fixtures, and run four de-risking spikes (tokenizer lowercasing, ORT node-coverage, external-data downloader, so400m memory) that must pass before the Android engine is planned.

**Architecture:** A host-side Python tool (`tools/android_assets/`) exports/downloads ONNX towers, converts precision, extracts `logit_scale`/`logit_bias`, emits a versioned `ModelBundleManifest` per model, and generates golden fixtures from a **lossless** reference pipeline. Four spikes answer the open unknowns from the spec — three on-device (Pixel 7a, already connected in Android Studio), one in Python.

**Tech Stack:** Python 3.12 (transformers, optimum-onnx, onnx, onnxruntime, torch, Pillow, numpy, tokenizers, huggingface_hub, pytest); Kotlin + ONNX Runtime Mobile (`onnxruntime-android`) + Media3 for the on-device spikes.

**Spec:** `docs/superpowers/specs/2026-06-02-clipcc-android-design.md` (§5.0 manifest, §5.1 pipeline, §5.2 tokenizer, §8 fixtures, §10 phases, §9 downloader).

---

## Shared contracts (locked before tasks — referenced by every task and by Plans 1–3)

### Benchmark profile `benchmark-v1`
| model_id | hf_repo | resolution | device precision |
|---|---|---|---|
| `siglip2-base-patch16-256` | `google/siglip2-base-patch16-256` | 256 | fp32 |
| `siglip2-base-patch16-384` | `google/siglip2-base-patch16-384` | 384 | fp32 |
| `siglip2-large-patch16-384` | `google/siglip2-large-patch16-384` | 384 | fp16 |
| `siglip2-so400m-patch14-384` | `google/siglip2-so400m-patch14-384` | 384 | fp16 |

### `ModelBundleManifest` JSON (schema_version 1) — the parity boundary
```json
{
  "schema_version": 1,
  "profile": "benchmark-v1",
  "model_id": "siglip2-base-patch16-256",
  "hf_repo": "google/siglip2-base-patch16-256",
  "hf_revision": "<40-char google-repo sha>",
  "onnx_source": "onnx-community",
  "onnx_source_repo": "onnx-community/siglip2-base-patch16-256-ONNX",
  "onnx_source_revision": "<40-char onnx-community sha>",
  "display_name": "SigLIP2 Base (256px)",
  "params": "0.4B",
  "resolution": 256,
  "precision": "fp32",
  "ram_budget_mb": 1600,
  "transformers_version": "<captured at export>",
  "score_semantics": "siglip2_pairwise_sigmoid",
  "logit_scale": 4.7654,
  "logit_bias": -16.53,
  "vision": {"file": "vision_model.onnx", "data_file": null, "bytes": 0, "sha256": "", "data_sha256": null},
  "text":   {"file": "text_model.onnx",   "data_file": null, "bytes": 0, "sha256": "", "data_sha256": null},
  "tokenizer": {
    "file": "tokenizer.json", "sha256": "",
    "max_length": 64, "pad_id": 0, "padding": "max_length", "padding_side": "right",
    "truncation": true, "lowercase_applied_by": "unknown"
  },
  "preprocess": {
    "resize": "stretch_square", "resample": "bilinear",
    "rescale": 0.00392156862745098,
    "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "layout": "CHW"
  },
  "frame_pipeline": {
    "fps": 1.0, "max_frames": 300,
    "prescale": "none", "intermediate_codec": "none", "resample": "bilinear"
  }
}
```
Notes:
- `logit_scale` is stored **raw**; runtime applies `exp(logit_scale)` (HF `SiglipModel` does
  `logit_scale.exp()`). `logit_bias` applied directly.
- **Acquisition is prebuilt-download, not conversion.** `onnx-community/siglip2-*-ONNX` ships
  both `vision_model.onnx`/`text_model.onnx` (fp32) **and** `vision_model_fp16.onnx`/
  `text_model_fp16.onnx`. We download the variant matching the target precision — no
  in-process fp16 conversion for the prebuilt path.
- **External data (`data_file`), corrected from live-repo inspection:** in the **fp32** repos
  it is the **TEXT** tower that is a tiny stub + large `text_model.onnx_data` (≈2.3–2.8 GB for
  large/so400m); vision towers are single-file. For our profile the only fp32 models are the
  small **base** models (single-file both towers), and large/so400m ship **fp16** (verified
  ≈0.6–1.4 GB per tower → **all single-file**). So in practice **none of the 4 profile bundles
  needs external data**; the `data_file` field stays nullable and the downloader still pulls a
  co-located `.onnx_data` sibling **if present** (defensive, e.g. optimum-fallback or a future
  fp32 large). When `data_file` is non-null it carries its own `data_sha256`.
- `onnx_source_revision` pins the **onnx-community** repo commit the shipped bytes came from
  (the `hf_revision` field pins the `google/` source repo — kept for `logit_scale` provenance).
- `lowercase_applied_by` is `"unknown"` until Spike 0a sets it to `"tokenizer_json"` or
  `"kotlin_wrapper"`.
- **Schema v1 scope (recorded decision):** v1 carries the fields above plus `ram_budget_mb`
  and tokenizer `padding_side`. It intentionally **omits** `FramePipelineSpec` rotation/
  color-range/SDR-HDR/color-space/timestamp policy and the full `ScoringPolicySpec` (rounding,
  temporal gap=2.0/min-dur=1.0, contrast defaults, response shape). Those are deferred to a
  **schema v2 bump in Plan 3** (consuming temporal/contrast defaults requires it). This keeps
  the parity boundary explicit, not silently incomplete.

### File map (created by this plan)
```
tools/android_assets/
  __init__.py
  manifest.py            # ModelBundleManifest dataclass + (de)serialize
  hashing.py             # sha256_file
  export_models.py       # acquire ONNX, convert precision, extract scalars, write manifest
  gen_fixtures.py        # tokenizer/preprocess/scores golden fixtures (lossless)
  spike_tokenizer.py     # Spike 0a (Python: AutoProcessor vs tokenizers Rust lib)
  requirements-export.txt
tools/android_assets/tests/
  __init__.py
  test_manifest.py
  test_hashing.py
  test_export_models.py
  test_gen_fixtures.py
  fixtures/lossless/      # checked-in tiny lossless PNG frames (≤4, ≤256px) for golden gen
docs/superpowers/plans/phase0-spike-results.md   # written by Task 13
```
Android spikes (0b/0c/0d) live in the existing Android Studio project under
`androidTest` — exact module path resolved in Task 9 and recorded in the spike report.

### Output layout (generated assets — git-ignored, pushed to device via adb)
```
build/android_assets/<model_id>/
  manifest.json
  vision_model.onnx[ + vision_model.onnx_data]
  text_model.onnx[ + text_model.onnx_data]
  tokenizer.json
build/android_assets/fixtures/
  tokenizer_golden.json
  preprocess_golden.npz
  scores_golden.json
```

---

## Task 1: Host tooling scaffold + pinned deps

**Files:**
- Create: `tools/android_assets/__init__.py`
- Create: `tools/android_assets/requirements-export.txt`
- Create: `tools/android_assets/tests/__init__.py`
- Create: `tools/android_assets/.gitignore` (ignore `build/` outputs if placed here)
- Modify: `.gitignore` (add `build/android_assets/`)

- [ ] **Step 1: Create the package + test dirs**

```bash
mkdir -p tools/android_assets/tests/fixtures/lossless
: > tools/android_assets/__init__.py
: > tools/android_assets/tests/__init__.py
```

- [ ] **Step 2: Write `requirements-export.txt` (versions pinned at install time)**

Create `tools/android_assets/requirements-export.txt`:
```
transformers
optimum-onnx
onnx
onnxruntime
torch
Pillow
numpy
huggingface_hub
tokenizers
pytest
```

- [ ] **Step 3: Create the venv and install; capture resolved versions**

Run:
```bash
python3 -m venv .venv-export
.venv-export/bin/pip install -r tools/android_assets/requirements-export.txt
.venv-export/bin/pip freeze | grep -iE '^(transformers|tokenizers|optimum|onnx|onnxruntime|torch|numpy|Pillow)==' \
  > tools/android_assets/requirements-export.lock.txt
```
Expected: `requirements-export.lock.txt` lists exact versions. The `transformers==` line is
the value baked into every manifest's `transformers_version`.

- [ ] **Step 4: Add output dir to `.gitignore`**

Add to `/Users/austin/MITAC/clipCC/.gitignore`:
```
build/android_assets/
.venv-export/
```

- [ ] **Step 5: Commit**

```bash
git add tools/android_assets/ .gitignore
git commit -m "chore(android-assets): scaffold export tooling + pinned deps"
```

---

## Task 2: Manifest dataclass (schema_version 1) + round-trip

**Files:**
- Create: `tools/android_assets/manifest.py`
- Test: `tools/android_assets/tests/test_manifest.py`

- [ ] **Step 1: Write the failing test**

Create `tools/android_assets/tests/test_manifest.py`:
```python
import json
from tools.android_assets.manifest import ModelBundleManifest, FileRef, SCHEMA_VERSION


def _sample() -> ModelBundleManifest:
    return ModelBundleManifest(
        model_id="siglip2-base-patch16-256",
        hf_repo="google/siglip2-base-patch16-256",
        hf_revision="a" * 40,
        onnx_source="onnx-community",
        onnx_source_repo="onnx-community/siglip2-base-patch16-256-ONNX",
        onnx_source_revision="b" * 40,
        display_name="SigLIP2 Base (256px)",
        params="0.4B",
        resolution=256,
        precision="fp32",
        ram_budget_mb=1600,
        transformers_version="5.0.0",
        logit_scale=4.7654,
        logit_bias=-16.53,
        vision=FileRef(file="vision_model.onnx", data_file=None, bytes=10, sha256="x", data_sha256=None),
        text=FileRef(file="text_model.onnx", data_file="text_model.onnx_data", bytes=10, sha256="y", data_sha256="yy"),
        tokenizer_sha256="z",
    )


def test_round_trip_preserves_fields_and_schema_version():
    m = _sample()
    blob = m.to_json()
    parsed = json.loads(blob)
    assert parsed["schema_version"] == SCHEMA_VERSION == 1
    assert parsed["profile"] == "benchmark-v1"
    assert parsed["score_semantics"] == "siglip2_pairwise_sigmoid"
    assert parsed["preprocess"]["resample"] == "bilinear"
    assert parsed["frame_pipeline"]["prescale"] == "none"
    assert parsed["tokenizer"]["lowercase_applied_by"] == "unknown"
    assert parsed["tokenizer"]["padding_side"] == "right"
    assert parsed["ram_budget_mb"] == 1600
    assert parsed["onnx_source_revision"] == "b" * 40
    assert parsed["text"]["data_sha256"] == "yy"
    back = ModelBundleManifest.from_json(blob)
    assert back == m


def test_from_json_rejects_wrong_schema_version():
    blob = _sample().to_json().replace('"schema_version": 1', '"schema_version": 99')
    try:
        ModelBundleManifest.from_json(blob)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "schema_version" in str(e)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-export/bin/python -m pytest tools/android_assets/tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: tools.android_assets.manifest`

- [ ] **Step 3: Implement `manifest.py`**

Create `tools/android_assets/manifest.py`:
```python
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Optional

SCHEMA_VERSION = 1
PROFILE = "benchmark-v1"
SCORE_SEMANTICS = "siglip2_pairwise_sigmoid"


@dataclass
class FileRef:
    file: str
    data_file: Optional[str]
    bytes: int
    sha256: str
    data_sha256: Optional[str] = None   # sha256 of the external .onnx_data, when present


@dataclass
class ModelBundleManifest:
    model_id: str
    hf_repo: str
    hf_revision: str                 # google/ source-repo sha (logit_scale provenance)
    onnx_source: str                 # "onnx-community" | "optimum"
    onnx_source_repo: str            # repo the shipped ONNX bytes came from
    onnx_source_revision: str        # sha of that repo ("" for optimum local export)
    display_name: str
    params: str
    resolution: int
    precision: str                   # "fp32" | "fp16"
    ram_budget_mb: int
    transformers_version: str
    logit_scale: float
    logit_bias: float
    vision: FileRef
    text: FileRef
    tokenizer_sha256: str
    tokenizer_padding_side: str = "right"
    tokenizer_lowercase_applied_by: str = "unknown"  # "tokenizer_json" | "kotlin_wrapper"

    def to_json(self) -> str:
        doc = {
            "schema_version": SCHEMA_VERSION,
            "profile": PROFILE,
            "model_id": self.model_id,
            "hf_repo": self.hf_repo,
            "hf_revision": self.hf_revision,
            "onnx_source": self.onnx_source,
            "onnx_source_repo": self.onnx_source_repo,
            "onnx_source_revision": self.onnx_source_revision,
            "display_name": self.display_name,
            "params": self.params,
            "resolution": self.resolution,
            "precision": self.precision,
            "ram_budget_mb": self.ram_budget_mb,
            "transformers_version": self.transformers_version,
            "score_semantics": SCORE_SEMANTICS,
            "logit_scale": self.logit_scale,
            "logit_bias": self.logit_bias,
            "vision": asdict(self.vision),
            "text": asdict(self.text),
            "tokenizer": {
                "file": "tokenizer.json",
                "sha256": self.tokenizer_sha256,
                "max_length": 64,
                "pad_id": 0,
                "padding": "max_length",
                "padding_side": self.tokenizer_padding_side,
                "truncation": True,
                "lowercase_applied_by": self.tokenizer_lowercase_applied_by,
            },
            "preprocess": {
                "resize": "stretch_square",
                "resample": "bilinear",
                "rescale": 0.00392156862745098,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
                "layout": "CHW",
            },
            "frame_pipeline": {
                "fps": 1.0,
                "max_frames": 300,
                "prescale": "none",
                "intermediate_codec": "none",
                "resample": "bilinear",
            },
        }
        return json.dumps(doc, indent=2)

    @classmethod
    def from_json(cls, blob: str) -> "ModelBundleManifest":
        d = json.loads(blob)
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {d.get('schema_version')!r}; expected {SCHEMA_VERSION}"
            )
        tok = d["tokenizer"]
        return cls(
            model_id=d["model_id"],
            hf_repo=d["hf_repo"],
            hf_revision=d["hf_revision"],
            onnx_source=d["onnx_source"],
            onnx_source_repo=d["onnx_source_repo"],
            onnx_source_revision=d["onnx_source_revision"],
            display_name=d["display_name"],
            params=d["params"],
            resolution=d["resolution"],
            precision=d["precision"],
            ram_budget_mb=d["ram_budget_mb"],
            transformers_version=d["transformers_version"],
            logit_scale=d["logit_scale"],
            logit_bias=d["logit_bias"],
            vision=FileRef(**d["vision"]),
            text=FileRef(**d["text"]),
            tokenizer_sha256=tok["sha256"],
            tokenizer_padding_side=tok["padding_side"],
            tokenizer_lowercase_applied_by=tok["lowercase_applied_by"],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-export/bin/python -m pytest tools/android_assets/tests/test_manifest.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/android_assets/manifest.py tools/android_assets/tests/test_manifest.py
git commit -m "feat(android-assets): ModelBundleManifest schema v1 + round-trip"
```

---

## Task 3: sha256 helper

**Files:**
- Create: `tools/android_assets/hashing.py`
- Test: `tools/android_assets/tests/test_hashing.py`

- [ ] **Step 1: Write the failing test**

Create `tools/android_assets/tests/test_hashing.py`:
```python
import hashlib
from tools.android_assets.hashing import sha256_file


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"clipcc" * 100000)  # multi-chunk
    expected = hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha256_file(p) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-export/bin/python -m pytest tools/android_assets/tests/test_hashing.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `hashing.py`**

Create `tools/android_assets/hashing.py`:
```python
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-export/bin/python -m pytest tools/android_assets/tests/test_hashing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/android_assets/hashing.py tools/android_assets/tests/test_hashing.py
git commit -m "feat(android-assets): sha256_file helper"
```

---

## Task 4: ONNX acquisition — repo-name resolution (unit) + acquire function

**Files:**
- Create: `tools/android_assets/export_models.py`
- Test: `tools/android_assets/tests/test_export_models.py`

- [ ] **Step 1: Write the failing test (pure logic only — no network)**

Create `tools/android_assets/tests/test_export_models.py`:
```python
from tools.android_assets.export_models import onnx_community_repo, PROFILE_MODELS


def test_profile_has_exactly_four_models():
    assert list(PROFILE_MODELS.keys()) == [
        "siglip2-base-patch16-256",
        "siglip2-base-patch16-384",
        "siglip2-large-patch16-384",
        "siglip2-so400m-patch14-384",
    ]


def test_onnx_community_repo_maps_google_repo():
    assert (
        onnx_community_repo("google/siglip2-base-patch16-256")
        == "onnx-community/siglip2-base-patch16-256-ONNX"
    )
    assert (
        onnx_community_repo("google/siglip2-so400m-patch14-384")
        == "onnx-community/siglip2-so400m-patch14-384-ONNX"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-export/bin/python -m pytest tools/android_assets/tests/test_export_models.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the resolution logic + acquisition (heavy parts behind functions)**

Create `tools/android_assets/export_models.py`:
```python
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    hf_repo: str
    display_name: str
    params: str
    resolution: int
    precision: str       # target device precision
    ram_budget_mb: int   # rough estimate; so400m refined by Spike 0d (Task 12)


PROFILE_MODELS: dict[str, ModelSpec] = {
    "siglip2-base-patch16-256": ModelSpec(
        "siglip2-base-patch16-256", "google/siglip2-base-patch16-256",
        "SigLIP2 Base (256px)", "0.4B", 256, "fp32", 1600),
    "siglip2-base-patch16-384": ModelSpec(
        "siglip2-base-patch16-384", "google/siglip2-base-patch16-384",
        "SigLIP2 Base (384px)", "0.4B", 384, "fp32", 1900),
    "siglip2-large-patch16-384": ModelSpec(
        "siglip2-large-patch16-384", "google/siglip2-large-patch16-384",
        "SigLIP2 Large (384px)", "0.9B", 384, "fp16", 2600),
    "siglip2-so400m-patch14-384": ModelSpec(
        "siglip2-so400m-patch14-384", "google/siglip2-so400m-patch14-384",
        "SigLIP2 SO400M (384px)", "1.0B", 384, "fp16", 3600),
}


def onnx_community_repo(hf_repo: str) -> str:
    """google/siglip2-X -> onnx-community/siglip2-X-ONNX."""
    name = hf_repo.split("/", 1)[1]
    return f"onnx-community/{name}-ONNX"


def _tower_basenames(precision: str) -> tuple[str, str]:
    """Prebuilt onnx-community filenames for the target precision."""
    if precision == "fp16":
        return "vision_model_fp16.onnx", "text_model_fp16.onnx"
    return "vision_model.onnx", "text_model.onnx"


def _place(src: Path, dst: Path) -> str | None:
    """Copy an .onnx to a canonical path; copy its co-located .onnx_data sibling if present.
    Returns the canonical data-file name when one was copied, else None."""
    import shutil
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    src_data = src.parent / (src.name + "_data")
    if src_data.exists():
        dst_data = dst.parent / (dst.name + "_data")
        shutil.copy(src_data, dst_data)
        return dst_data.name
    return None


def acquire_onnx(hf_repo: str, precision: str, out_dir: Path) -> dict:
    """Acquire prebuilt onnx-community towers for the target precision (else optimum fallback).

    Downloads the matching .onnx (+ any external .onnx_data sibling) + tokenizer.json, then
    normalizes to canonical bundle paths: out_dir/vision_model.onnx[+_data],
    out_dir/text_model.onnx[+_data], out_dir/tokenizer.json. No in-process precision conversion
    for the prebuilt path (onnx-community ships both fp32 and *_fp16 variants).

    Returns: {vision, text, tokenizer: Path, vision_data, text_data: str|None,
              source, source_repo, source_revision}. Network + disk heavy — integration only.
    """
    from huggingface_hub import snapshot_download, model_info
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    out_dir.mkdir(parents=True, exist_ok=True)
    repo = onnx_community_repo(hf_repo)
    v_name, t_name = _tower_basenames(precision)
    vision = out_dir / "vision_model.onnx"
    text = out_dir / "text_model.onnx"
    tok = out_dir / "tokenizer.json"
    try:
        snap = Path(snapshot_download(repo, allow_patterns=[
            f"onnx/{v_name}", f"onnx/{v_name}_data",
            f"onnx/{t_name}", f"onnx/{t_name}_data",
            "tokenizer.json",
        ]))
        v_data = _place(snap / "onnx" / v_name, vision)
        t_data = _place(snap / "onnx" / t_name, text)
        import shutil
        shutil.copy(snap / "tokenizer.json", tok)
        return {"vision": vision, "text": text, "tokenizer": tok,
                "vision_data": v_data, "text_data": t_data,
                "source": "onnx-community", "source_repo": repo,
                "source_revision": model_info(repo).sha}
    except (EntryNotFoundError, RepositoryNotFoundError):
        export_dir = out_dir / "optimum_export"
        subprocess.run(
            ["optimum-cli", "export", "onnx",
             "--model", hf_repo, "--task", "zero-shot-image-classification",
             str(export_dir)],
            check=True,
        )
        produced = {p.name for p in export_dir.glob("*.onnx")}
        assert "vision_model.onnx" in produced and "text_model.onnx" in produced, (
            f"optimum export missing towers; produced: {sorted(produced)}")
        v_data = _place(export_dir / "vision_model.onnx", vision)
        t_data = _place(export_dir / "text_model.onnx", text)
        import shutil
        shutil.copy(export_dir / "tokenizer.json", tok)
        return {"vision": vision, "text": text, "tokenizer": tok,
                "vision_data": v_data, "text_data": t_data,
                "source": "optimum", "source_repo": "", "source_revision": ""}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-export/bin/python -m pytest tools/android_assets/tests/test_export_models.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/android_assets/export_models.py tools/android_assets/tests/test_export_models.py
git commit -m "feat(android-assets): ONNX acquisition (onnx-community else optimum)"
```

---

## Task 5: precision-conversion + safe ONNX save (optimum-fallback path only)

> The normal prebuilt path (Task 4) downloads `*_fp16.onnx` directly and relocates by file
> copy, so it never needs these helpers. `to_fp16`/`save_onnx` exist only for the
> optimum-fallback path (a model with no prebuilt onnx-community ONNX). Kept + unit-tested so
> the fallback is real, not a placeholder.

**Files:**
- Modify: `tools/android_assets/export_models.py`
- Test: `tools/android_assets/tests/test_export_models.py`

- [ ] **Step 1: Write the failing test (synthetic ONNX, no big model)**

Append to `tools/android_assets/tests/test_export_models.py`:
```python
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
from tools.android_assets.export_models import to_fp16, save_onnx


def _identity_fp32_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    w = helper.make_tensor("w", TensorProto.FLOAT, [4], np.ones(4, np.float32))
    node = helper.make_node("Mul", ["x", "w"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y], initializer=[w])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_to_fp16_runs_and_keeps_shape(tmp_path):
    m32 = _identity_fp32_model()
    m16 = to_fp16(m32)
    p = tmp_path / "m16.onnx"
    save_onnx(m16, p)
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"x": np.array([[1, 2, 3, 4]], np.float32)})[0]
    assert out.shape == (1, 4)


def test_save_onnx_external_data_when_forced(tmp_path):
    m = _identity_fp32_model()
    p = tmp_path / "ext.onnx"
    save_onnx(m, p, force_external=True)
    assert p.exists()
    assert (tmp_path / "ext.onnx_data").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-export/bin/python -m pytest tools/android_assets/tests/test_export_models.py -k "fp16 or external" -v`
Expected: FAIL with `ImportError: cannot import name 'to_fp16'`

- [ ] **Step 3: Implement `to_fp16` + `save_onnx`**

Append to `tools/android_assets/export_models.py`:
```python
EXTERNAL_DATA_THRESHOLD = 2_000_000_000  # under the 2 GiB protobuf limit


def to_fp16(model):
    import onnx
    from onnxconverter_common import float16
    return float16.convert_float_to_float16(model, keep_io_types=True)


def _exceeds_protobuf_limit(model) -> bool:
    """True if the model would exceed the 2 GB protobuf serialization ceiling.

    `ModelProto.ByteSize()` itself RAISES `ValueError(... exceeds maximum protobuf size of
    2GB ...)` for >2 GB models — so catch that and treat it as 'too big' rather than
    serialize-then-compare (which can never return True for the models that need external data).
    """
    try:
        return model.ByteSize() >= EXTERNAL_DATA_THRESHOLD
    except ValueError:
        return True


def save_onnx(model, path, force_external: bool | None = None) -> None:
    """Save ONNX; externalize tensors when the model is large (or forced).

    NOTE: only the **optimum-fallback** path serializes a ModelProto. The normal prebuilt path
    relocates files by copy (see `_place`) and never round-trips through the 2 GB serializer.
    """
    import onnx

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    use_external = force_external if force_external is not None else _exceeds_protobuf_limit(model)
    if use_external:
        onnx.save_model(
            model, str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=path.name + "_data",
        )
    else:
        onnx.save_model(model, str(path))
```

If `onnxconverter_common` is not present, add it:
```bash
.venv-export/bin/pip install onnxconverter-common
echo "onnxconverter-common" >> tools/android_assets/requirements-export.txt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-export/bin/python -m pytest tools/android_assets/tests/test_export_models.py -k "fp16 or external" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/android_assets/export_models.py tools/android_assets/tests/test_export_models.py tools/android_assets/requirements-export.txt
git commit -m "feat(android-assets): fp16 conversion + external-data ONNX save"
```

---

## Task 6: Extract `logit_scale`/`logit_bias` + build one manifest (unit on extraction)

**Files:**
- Modify: `tools/android_assets/export_models.py`
- Test: `tools/android_assets/tests/test_export_models.py`

- [ ] **Step 1: Write the failing test (stub object, no model download)**

Append to `tools/android_assets/tests/test_export_models.py`:
```python
import torch
from tools.android_assets.export_models import extract_logit_params


class _StubModel:
    def __init__(self):
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.7654))
        self.logit_bias = torch.nn.Parameter(torch.tensor(-16.53))


def test_extract_logit_params_reads_raw_scalars():
    scale, bias = extract_logit_params(_StubModel())
    assert round(scale, 4) == 4.7654
    assert round(bias, 2) == -16.53
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-export/bin/python -m pytest tools/android_assets/tests/test_export_models.py -k extract -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement extraction + a `build_manifest` assembler**

Append to `tools/android_assets/export_models.py`:
```python
# Set by Spike 0a (Task 7). "unknown" until then; "tokenizer_json" if the Rust normalizer
# already lowercases, else "kotlin_wrapper" (Android must lowercase before encoding).
LOWERCASE_APPLIED_BY = "unknown"


def extract_logit_params(model) -> tuple[float, float]:
    """Read raw logit_scale and logit_bias (HF SiglipModel applies exp(scale) at runtime)."""
    return float(model.logit_scale.detach().item()), float(model.logit_bias.detach().item())


def build_manifest(spec: "ModelSpec", *, hf_revision: str, transformers_version: str,
                   logit_scale: float, logit_bias: float,
                   vision_path: Path, text_path: Path, tokenizer_path: Path,
                   onnx_source: str, onnx_source_repo: str, onnx_source_revision: str):
    from tools.android_assets.manifest import ModelBundleManifest, FileRef
    from tools.android_assets.hashing import sha256_file

    def _ref(p: Path) -> "FileRef":
        data = p.parent / (p.name + "_data")
        has_data = data.exists()
        return FileRef(
            file=p.name,
            data_file=(data.name if has_data else None),
            bytes=p.stat().st_size,
            sha256=sha256_file(p),
            data_sha256=(sha256_file(data) if has_data else None),
        )

    return ModelBundleManifest(
        model_id=spec.model_id, hf_repo=spec.hf_repo, hf_revision=hf_revision,
        onnx_source=onnx_source, onnx_source_repo=onnx_source_repo,
        onnx_source_revision=onnx_source_revision,
        display_name=spec.display_name, params=spec.params, resolution=spec.resolution,
        precision=spec.precision, ram_budget_mb=spec.ram_budget_mb,
        transformers_version=transformers_version,
        logit_scale=logit_scale, logit_bias=logit_bias,
        vision=_ref(vision_path), text=_ref(text_path),
        tokenizer_sha256=sha256_file(tokenizer_path),
        tokenizer_lowercase_applied_by=LOWERCASE_APPLIED_BY,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-export/bin/python -m pytest tools/android_assets/tests/test_export_models.py -k extract -v`
Expected: PASS

- [ ] **Step 5: Add the `main()` CLI that ties acquisition → fp16 → manifest**

Append to `tools/android_assets/export_models.py`:
```python
def export_one(model_id: str, out_root: Path) -> Path:
    from transformers import AutoModel
    import transformers as _t
    from huggingface_hub import model_info

    spec = PROFILE_MODELS[model_id]
    out_dir = out_root / model_id

    # Acquire prebuilt towers at the target precision (+ tokenizer.json), normalized to
    # canonical bundle paths. No precision conversion / serializer round-trip for prebuilt.
    acquired = acquire_onnx(spec.hf_repo, spec.precision, out_dir)

    # logit_scale / logit_bias come from the torch source model (downloaded to HF cache).
    model = AutoModel.from_pretrained(spec.hf_repo)
    scale, bias = extract_logit_params(model)

    manifest = build_manifest(
        spec,
        hf_revision=model_info(spec.hf_repo).sha,
        transformers_version=_t.__version__,
        logit_scale=scale, logit_bias=bias,
        vision_path=acquired["vision"], text_path=acquired["text"],
        tokenizer_path=acquired["tokenizer"],
        onnx_source=acquired["source"], onnx_source_repo=acquired["source_repo"],
        onnx_source_revision=acquired["source_revision"],
    )
    (out_dir / "manifest.json").write_text(manifest.to_json())
    return out_dir / "manifest.json"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("build/android_assets"))
    ap.add_argument("--models", default=",".join(PROFILE_MODELS))
    args = ap.parse_args()
    for mid in [m.strip() for m in args.models.split(",")]:
        print(f"[{mid}] exporting...")
        print("  manifest:", export_one(mid, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Integration run (network + disk; produces the real bundles)**

Run:
```bash
.venv-export/bin/python -m tools.android_assets.export_models --models siglip2-base-patch16-256
```
Expected: `build/android_assets/siglip2-base-patch16-256/manifest.json` exists; `vision_model.onnx`,
`text_model.onnx`, `tokenizer.json` present; manifest `logit_scale`/`logit_bias` populated;
`transformers_version` matches the lock file. Inspect:
```bash
.venv-export/bin/python -c "import json;d=json.load(open('build/android_assets/siglip2-base-patch16-256/manifest.json'));print(d['logit_scale'],d['logit_bias'],d['precision'],d['vision'])"
```

- [ ] **Step 7: Commit**

```bash
git add tools/android_assets/export_models.py tools/android_assets/tests/test_export_models.py
git commit -m "feat(android-assets): export_one + CLI (acquire->fp16->manifest)"
```

---

## Task 7: Spike 0a — tokenizer lowercasing (Python, resolves the UNRESOLVED §5.2 question)

**Files:**
- Create: `tools/android_assets/spike_tokenizer.py`
- Test: `tools/android_assets/tests/test_gen_fixtures.py` (parity assertions added in Task 8)

> This spike is exploratory: its deliverable is **evidence + a decision**, recorded in the
> manifest field `tokenizer_lowercase_applied_by` and the spike report (Task 13). The Python
> `tokenizers` package is the same Rust engine the Android JNI will use, so this resolves the
> lowercasing question without Android.

- [ ] **Step 1: Write the spike harness**

Create `tools/android_assets/spike_tokenizer.py`:
```python
from __future__ import annotations

import json
import sys
from pathlib import Path

CASES = ["Car", "car", "CAR", "Texting While Driving", "texting while driving",
         "EATING", "a dog.", "two words", "ALLCAPS PHRASE"]


def _pad_trunc(ids: list[int], max_length: int = 64, pad_id: int = 0) -> list[int]:
    """Reproduce the Kotlin-side wrapper: truncate to max_length, pad right with pad_id."""
    ids = list(ids[:max_length])
    return ids + [pad_id] * (max_length - len(ids))


def run(hf_repo: str, bundle_dir: Path) -> dict:
    from transformers import AutoProcessor
    from tokenizers import Tokenizer

    # Reference = HF AutoProcessor (torch side). Candidate = the shipped tokenizer.json read by
    # the SAME Rust engine the Android JNI uses.
    proc = AutoProcessor.from_pretrained(hf_repo)
    tok_json_path = bundle_dir / "tokenizer.json"
    rust = Tokenizer.from_file(str(tok_json_path))

    # (1) Structural ground truth: does the serialized normalizer contain a Lowercase step?
    norm = json.loads(tok_json_path.read_text()).get("normalizer")
    normalizer_lowercases = "Lowercase" in json.dumps(norm)

    # (2) Behavioral diagnostic: does the Rust engine lowercase on its own?
    rust_lowercases = all(
        rust.encode(t).ids == rust.encode(t.lower()).ids
        for t in CASES if t != t.lower()
    )

    # (3) The §10 0a GATE — decide from what AutoProcessor ACTUALLY emits, not from a
    #     structural guess. The candidate pipeline (rust.encode -> _pad_trunc) must equal the
    #     AutoProcessor max_length=64 output. Test BOTH raw and lowered; whichever matches for
    #     ALL cases is the decision:
    #       - raw matches   -> "tokenizer_json"  (Android must NOT lowercase; case-sensitive)
    #       - lower matches -> "kotlin_wrapper"  (Android must .lowercase() before encoding)
    def ref_ids(text: str) -> list[int]:
        return proc(text=[text], padding="max_length", max_length=64,
                    truncation=True, return_tensors="np")["input_ids"][0].tolist()

    def cand_ids(text: str) -> list[int]:
        return _pad_trunc(rust.encode(text).ids)

    matches_raw = all(cand_ids(t) == ref_ids(t) for t in CASES)
    matches_lower = all(cand_ids(t.lower()) == ref_ids(t) for t in CASES)
    if matches_raw:
        decision = "tokenizer_json"
    elif matches_lower:
        decision = "kotlin_wrapper"
    else:
        decision = "needs_investigation"
    parity_ok = matches_raw or matches_lower

    lower_first = decision == "kotlin_wrapper"
    rows = [(text, cand_ids(text.lower() if lower_first else text) == ref_ids(text))
            for text in CASES]

    return {
        "decision": decision,
        "rust_lowercases": rust_lowercases,
        "normalizer_lowercases": normalizer_lowercases,
        "matches_raw": matches_raw,
        "matches_lower": matches_lower,
        "parity_ok": parity_ok,
        "rows": rows,
    }


if __name__ == "__main__":
    # argv: <hf_repo> <bundle_dir>
    out = run(sys.argv[1], Path(sys.argv[2]))
    for text, ok in out["rows"]:
        print(f"{text!r:24} full_pipeline_parity:{ok}")
    print("rust_lowercases =", out["rust_lowercases"],
          " normalizer_lowercases =", out["normalizer_lowercases"])
    print("matches_raw =", out["matches_raw"], " matches_lower =", out["matches_lower"])
    print("DECISION lowercase_applied_by =", out["decision"])
    print("GATE 0a parity_ok =", out["parity_ok"])
    raise SystemExit(0 if out["parity_ok"] else 1)
```

> **RESOLVED 2026-06-03 (transformers 4.57.6):** decision = **`tokenizer_json`**,
> `parity_ok = True` (9/9 cases incl. mixed/upper). SigLIP2's fast tokenizer
> (`GemmaTokenizerFast`) is **case-sensitive** — it does NOT lowercase, and `tokenizer.json`
> has no `Lowercase` normalizer. `AutoProcessor("Car")`=`[3726,1]`, `("car")`=`[2269,1]`,
> `("CAR")`=`[15547,1]` (distinct), and `rust.encode(text)` byte-matches each. **The Android
> Kotlin wrapper must NOT lowercase labels** — doing so would break parity. Corroborated by
> `app/models/siglip2_model.py:82` ("the Gemma tokenizer is case-sensitive"). The decision is
> made by comparing both raw and lowered candidates to `AutoProcessor` and picking the match —
> not by inspecting the normalizer (an earlier rule made that mistake).

- [ ] **Step 2: Run the spike against the base-256 bundle from Task 6**

Run:
```bash
.venv-export/bin/python -m tools.android_assets.spike_tokenizer \
  google/siglip2-base-patch16-256 build/android_assets/siglip2-base-patch16-256
```
Expected: a per-case `full_pipeline_parity:True` table, the `rust_lowercases` /
`normalizer_lowercases` flags, `DECISION lowercase_applied_by = tokenizer_json` (expected) or
`= kotlin_wrapper` (red flag — investigate), and `GATE 0a parity_ok = True`. The process exits
non-zero if parity fails — that is the hard 0a gate. Record the full output in the spike report
(Task 13).

- [ ] **Step 3: Bake the decision into the exporter**

Set the module constant in `tools/android_assets/export_models.py` to the printed decision:
```python
LOWERCASE_APPLIED_BY = "tokenizer_json"   # or "kotlin_wrapper", per the spike output
```
Re-run the export for base-256 and confirm the manifest reflects it:
```bash
.venv-export/bin/python -m tools.android_assets.export_models --models siglip2-base-patch16-256
.venv-export/bin/python -c "import json;print(json.load(open('build/android_assets/siglip2-base-patch16-256/manifest.json'))['tokenizer']['lowercase_applied_by'])"
```
Expected: prints the decision (not `unknown`).

- [ ] **Step 4: Commit**

```bash
git add tools/android_assets/spike_tokenizer.py tools/android_assets/export_models.py
git commit -m "spike(0a): resolve SigLIP2 tokenizer lowercasing source"
```

---

## Task 8: Golden fixtures — tokenizer, preprocess, scores (lossless reference)

**Files:**
- Create: `tools/android_assets/gen_fixtures.py`
- Create: `tools/android_assets/tests/fixtures/lossless/frame_000.png` (+ up to 3 more)
- Test: `tools/android_assets/tests/test_gen_fixtures.py`

- [ ] **Step 1: Add tiny checked-in lossless frames**

Run (generates 2 deterministic PNGs — no external assets needed):
```bash
.venv-export/bin/python - <<'PY'
import numpy as np
from PIL import Image
from pathlib import Path
d = Path("tools/android_assets/tests/fixtures/lossless"); d.mkdir(parents=True, exist_ok=True)
for i in range(2):
    rng = np.random.default_rng(i)
    arr = (rng.integers(0, 256, (180, 320, 3))).astype("uint8")  # non-square, like real video
    Image.fromarray(arr).save(d / f"frame_{i:03d}.png")
print("wrote", list(d.glob("*.png")))
PY
```

- [ ] **Step 2: Write the failing test**

Create `tools/android_assets/tests/test_gen_fixtures.py`:
```python
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

    # exact resample contract captured for the Android bilinear target (M16)
    rs = json.loads((tmp_path / "resample_contract.json").read_text())
    assert rs["resample"] in (2, "bilinear")  # SigLIP2 = PIL BILINEAR (resample=2)
    assert rs["size"]["height"] == res and rs["size"]["width"] == res
    assert rs["image_mean"] == [0.5, 0.5, 0.5] and rs["image_std"] == [0.5, 0.5, 0.5]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv-export/bin/python -m pytest tools/android_assets/tests/test_gen_fixtures.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement `gen_fixtures.py` (lossless: PNG → bilinear square → normalize)**

Create `tools/android_assets/gen_fixtures.py`:
```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def gen_all(hf_repo: str, frames_dir: Path, labels: list[str], out_dir: Path) -> dict:
    """Generate golden fixtures from the HF torch reference (the source of truth).

    The reference is the fp32 PyTorch model + AutoProcessor loaded from `hf_repo` — NOT the
    exported ONNX bundle (which has no torch weights). On-device ONNX output is later compared
    against these fixtures within tolerance.
    """
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(hf_repo)
    model = AutoModel.from_pretrained(hf_repo).eval()

    # --- tokenizer golden ---
    tok_rows = []
    for text in labels:
        ids = proc(text=[text], padding="max_length", max_length=64,
                   truncation=True, return_tensors="np")["input_ids"][0].tolist()
        tok_rows.append({"text": text, "input_ids": ids})
    (out_dir / "tokenizer_golden.json").write_text(json.dumps(tok_rows, indent=2))

    # --- lossless frames: PNG -> AutoProcessor image path (bilinear square, normalize) ---
    frame_paths = sorted(frames_dir.glob("*.png"))
    images = [Image.open(p).convert("RGB") for p in frame_paths]
    pixel_values = proc(images=images, return_tensors="np")["pixel_values"]  # [F,3,R,R]
    np.savez(out_dir / "preprocess_golden.npz", pixel_values=pixel_values)

    # --- scores golden (fp32 reference) ---
    enc = proc(text=labels, images=images, padding="max_length", max_length=64,
               truncation=True, return_tensors="pt")
    with torch.inference_mode():
        o = model(**enc)
    confidence = torch.sigmoid(o.logits_per_image)            # [F,L]
    ie = o.image_embeds / o.image_embeds.norm(p=2, dim=-1, keepdim=True)
    te = o.text_embeds / o.text_embeds.norm(p=2, dim=-1, keepdim=True)
    cosine = ie @ te.T                                        # [F,L]
    (out_dir / "scores_golden.json").write_text(json.dumps({
        "labels": labels,
        "frames": [p.name for p in frame_paths],
        "confidence": confidence.tolist(),
        "cosine": cosine.tolist(),
    }, indent=2))

    # --- exact resample/normalize contract for the Android bilinear target (M16) ---
    ip = proc.image_processor
    contract = {
        "do_resize": getattr(ip, "do_resize", True),
        "size": dict(getattr(ip, "size", {})),
        "resample": getattr(ip, "resample", 2),  # PIL.Image.BILINEAR == 2 (SigLIP2 default)
        "antialias": getattr(ip, "antialias", None),
        "do_rescale": getattr(ip, "do_rescale", True),
        "rescale_factor": getattr(ip, "rescale_factor", 1 / 255),
        "do_normalize": getattr(ip, "do_normalize", True),
        "image_mean": list(getattr(ip, "image_mean", [0.5, 0.5, 0.5])),
        "image_std": list(getattr(ip, "image_std", [0.5, 0.5, 0.5])),
        "do_convert_rgb": getattr(ip, "do_convert_rgb", True),
    }
    (out_dir / "resample_contract.json").write_text(json.dumps(contract, indent=2))
    return {"out_dir": str(out_dir)}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv-export/bin/python -m pytest tools/android_assets/tests/test_gen_fixtures.py -v`
Expected: PASS (requires the base-256 bundle from Task 6 Step 6)

- [ ] **Step 6: Generate the shipped fixtures**

Run:
```bash
.venv-export/bin/python -c "from pathlib import Path; from tools.android_assets.gen_fixtures import gen_all; gen_all('google/siglip2-base-patch16-256', Path('tools/android_assets/tests/fixtures/lossless'), ['Car','texting while driving','a dog'], Path('build/android_assets/fixtures'))"
ls build/android_assets/fixtures
```
Expected: `tokenizer_golden.json`, `preprocess_golden.npz`, `scores_golden.json`.

- [ ] **Step 7: Commit**

```bash
git add tools/android_assets/gen_fixtures.py tools/android_assets/tests/test_gen_fixtures.py tools/android_assets/tests/fixtures/lossless/*.png
git commit -m "feat(android-assets): lossless golden fixtures (tokenizer/preprocess/scores)"
```

---

## Task 9: Export the remaining 3 models + verify external data on the big ones

**Files:** (no new code — exercises Tasks 4–6 at scale)

- [ ] **Step 1: Export base-384, large-384 (fp16), so400m-384 (fp16)**

Run:
```bash
.venv-export/bin/python -m tools.android_assets.export_models \
  --models siglip2-base-patch16-384,siglip2-large-patch16-384,siglip2-so400m-patch14-384
```
Expected: three more `build/android_assets/<id>/manifest.json` produced.

- [ ] **Step 2: Record actual tower sizes + verify integrity (do NOT assume external data)**

Our profile ships **base = fp32** (small, single-file) and **large/so400m = fp16** (verified
≈0.6–1.4 GB per tower → single-file). So we expect **no** external `.onnx_data` for any of the
four. The check records reality and verifies every declared file, rather than asserting
external data that should not exist:

```bash
.venv-export/bin/python - <<'PY'
import json
from pathlib import Path
from tools.android_assets.hashing import sha256_file
root = Path("build/android_assets")
for mid in ["siglip2-base-patch16-256","siglip2-base-patch16-384",
            "siglip2-large-patch16-384","siglip2-so400m-patch14-384"]:
    m = json.load(open(root/mid/"manifest.json"))
    print(f"\n[{mid}] precision={m['precision']} onnx_source={m['onnx_source']}")
    for tower in ("vision","text"):
        ref = m[tower]
        size_mb = ref["bytes"] / 1e6
        assert sha256_file(root/mid/ref["file"]) == ref["sha256"], f"{tower} sha mismatch"
        df = ref["data_file"]
        if df:
            assert (root/mid/df).exists(), f"{tower} declares data_file but it is missing"
            assert sha256_file(root/mid/df) == ref["data_sha256"], f"{tower} data sha mismatch"
        print(f"  {tower}: {size_mb:.0f} MB  data_file={df}")
PY
```
Expected: each tower's sha256 matches the manifest; for the profile bundles `data_file` is
`null` everywhere. Record the printed sizes (and any unexpected non-null `data_file`) in the
spike report. (External-data handling stays in the code for the optimum-fallback / future-fp32
case; it is just not exercised by this profile.)

- [ ] **Step 3: Commit (manifests are tiny; ONNX stays git-ignored)**

```bash
git add tools/android_assets
git commit -m "chore(android-assets): export full benchmark-v1 model set" --allow-empty
```

---

## Task 10: Spike 0b — ORT node-coverage on Android (instrumented)

**Files:** (in the existing Android Studio project)
- Create: `app/src/androidTest/java/.../spike/OrtCoverageSpikeTest.kt`
- Modify: app module `build.gradle(.kts)` — add `com.microsoft.onnxruntime:onnxruntime-android`

> Exploratory spike. Deliverable: a documented way to read **% of nodes assigned to the target
> EP vs CPU fallback** from the ORT Java API, or a fallback parse of the profiling JSON.

- [ ] **Step 1: Record the project/module path**

Run: `find . -name 'build.gradle*' -not -path '*/node_modules/*' | head` and note the app
module dir (e.g. `app/`). Record the package name from its `namespace`. Write both into the
spike report (Task 13).

- [ ] **Step 2: Add ORT dependency**

In the app module `build.gradle.kts` `dependencies {}`:
```kotlin
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.26.0")
androidTestImplementation("androidx.test.ext:junit:1.2.1")
androidTestImplementation("androidx.test:runner:1.6.2")
```

- [ ] **Step 3: Push a small real tower to the device**

Run:
```bash
adb shell mkdir -p /sdcard/Android/data/<app.package>/files/spike
adb push build/android_assets/siglip2-base-patch16-256/vision_model.onnx \
  /sdcard/Android/data/<app.package>/files/spike/
```

- [ ] **Step 4: Write the instrumented spike test**

Create `OrtCoverageSpikeTest.kt`:
```kotlin
package <app.package>.spike

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.json.JSONArray
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

@RunWith(AndroidJUnit4::class)
class OrtCoverageSpikeTest {
    private val dir = "/sdcard/Android/data/<app.package>/files/spike"
    private val model = "$dir/vision_model.onnx"
    private val res = 256  // base-256 vision input is [N,3,256,256]

    /** Per-NODE provider histogram from the ORT profile JSON (dedupe the
     *  _kernel_time/_fence_before/_fence_after events down to one record per node). */
    private fun coverage(profilePath: String, targetEp: String): String {
        val events = JSONArray(File(profilePath).readText())
        val nodeProvider = HashMap<String, String>()
        for (i in 0 until events.length()) {
            val e = events.optJSONObject(i) ?: continue
            if (e.optString("cat") != "Node") continue
            val provider = e.optJSONObject("args")?.optString("provider", "") ?: ""
            if (provider.isEmpty()) continue
            val node = e.optString("name")
                .removeSuffix("_kernel_time").removeSuffix("_fence_before").removeSuffix("_fence_after")
            nodeProvider[node] = provider
        }
        val hist = nodeProvider.values.groupingBy { it }.eachCount()
        val total = nodeProvider.size
        val onTarget = hist[targetEp] ?: 0
        val pct = if (total > 0) 100.0 * onTarget / total else 0.0
        // NOTE: NNAPI fuses its supported subgraph into ONE node, so node-count coverage is a
        // LOWER BOUND for NNAPI; for XNNPACK the per-op count is meaningful.
        return "nodes=$total hist=$hist target=$targetEp delegated=${"%.1f".format(pct)}%"
    }

    private fun pixelInput(s: OrtSession): String =
        s.inputNames.firstOrNull { it.contains("pixel_values") } ?: s.inputNames.first()

    private fun runWith(makeOpts: () -> OrtSession.SessionOptions, tag: String, targetEp: String) {
        val env = OrtEnvironment.getEnvironment()
        val session: OrtSession
        try {
            val opts = makeOpts().apply { enableProfiling("$dir/$tag") }
            session = env.createSession(model, opts)
        } catch (e: OrtException) {
            println("SPIKE0b[$tag] available=false reason=${e.message}")
            return
        }
        try {
            println("SPIKE0b[$tag] inputs=${session.inputNames}")
            val inputName = pixelInput(session)
            val buf = ByteBuffer.allocateDirect(1 * 3 * res * res * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer()
            // Load-bearing run: without it the profile has no per-node events.
            OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, res.toLong(), res.toLong())).use { t ->
                session.run(mapOf(inputName to t)).close()
            }
            val profilePath = session.endProfiling()
            println("SPIKE0b[$tag] available=true ${coverage(profilePath, targetEp)}")
        } finally {
            session.close()
        }
    }

    @Test fun xnnpack_then_nnapi() {
        runWith({ OrtSession.SessionOptions().apply { addXnnpack(mapOf("intra_op_num_threads" to "4")) } },
                "xnnpack", "XnnpackExecutionProvider")
        runWith({ OrtSession.SessionOptions().apply { addNnapi() } },
                "nnapi", "NnapiExecutionProvider")
    }
}
```

- [ ] **Step 5: Run the spike**

Run: `./gradlew :app:connectedDebugAndroidTest --tests "*OrtCoverageSpikeTest*"`
Expected: `SPIKE0b[xnnpack] available=true ... delegated=NN.N%` and a `SPIKE0b[nnapi]` line
(`available=true` with a delegated% — likely low/fused — or `available=false reason=...` if
the EP is not compiled into the AAR). Capture:
```bash
adb logcat -d | grep SPIKE0b
```

- [ ] **Step 6: Decide & record**

In the spike report (Task 13) record: whether each EP is available in the shipped
`onnxruntime-android` AAR; the per-node provider histogram + `delegated%`; and whether the
profiling JSON is the chosen `BackendCapabilityReport` source for Plan 2 (note the NNAPI
single-fused-node caveat — node-count is a lower bound there, so also capture VERBOSE
"Node placements" logs if a finer NNAPI number is needed).

- [ ] **Step 7: Commit (Android project)**

```bash
git add -A && git commit -m "spike(0b): ORT node-coverage probe (xnnpack vs nnapi)"
```

---

## Task 11: Spike 0c — external-data downloader on Android (instrumented)

**Files:**
- Create: `app/src/androidTest/java/.../spike/DownloaderSpikeTest.kt`

> Deliverable (re-scoped — the profile has no external-data model): prove ORT loads the
> **largest profile towers** (`so400m` fp16, single-file ~0.86–1.4 GB) with **sha256 integrity
> asserted against the manifest**, AND (optional but recommended) that ORT loads an
> **external-data** model using the so400m **fp32 text** tower (`text_model.onnx` stub +
> `text_model.onnx_data` ~2.8 GB) as the representative external-data case Plan 2's downloader
> must support. Resume / free-space-preflight / Xet range-GET are **explicitly deferred to
> Plan 2** and carried as a Plan-2 risk (not de-risked here).

- [ ] **Step 1: Stage the so400m fp16 towers (and, for the optional probe, the fp32 text tower)**

The network downloader is built in Plan 2; here we adb-push to prove ORT load + integrity:
```bash
APP=<app.package>
adb shell mkdir -p /sdcard/Android/data/$APP/files/spike/so400m
adb push build/android_assets/siglip2-so400m-patch14-384/vision_model.onnx \
  build/android_assets/siglip2-so400m-patch14-384/text_model.onnx \
  build/android_assets/siglip2-so400m-patch14-384/manifest.json \
  /sdcard/Android/data/$APP/files/spike/so400m/
# Optional external-data probe: fetch the fp32 text tower (stub + .onnx_data) and push both.
.venv-export/bin/python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
p = snapshot_download("onnx-community/siglip2-so400m-patch14-384-ONNX",
                      allow_patterns=["onnx/text_model.onnx","onnx/text_model.onnx_data"])
print(Path(p)/"onnx")
PY
# adb push the printed onnx/text_model.onnx + onnx/text_model.onnx_data into .../so400m/ext/
```

- [ ] **Step 2: Write the instrumented test**

Create `DownloaderSpikeTest.kt`:
```kotlin
package <app.package>.spike

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.security.MessageDigest

@RunWith(AndroidJUnit4::class)
class DownloaderSpikeTest {
    private val dir = "/sdcard/Android/data/<app.package>/files/spike/so400m"

    private fun sha256(f: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { s -> val b = ByteArray(1 shl 20)
            while (true) { val n = s.read(b); if (n < 0) break; md.update(b, 0, n) } }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    /** Largest profile towers load + integrity is GATED (assertEquals, not printed). */
    @Test fun loads_profile_towers_with_integrity() {
        val env = OrtEnvironment.getEnvironment()
        val manifest = JSONObject(File("$dir/manifest.json").readText())
        println("SPIKE0c free_bytes=${File(dir).freeSpace}")
        for (tower in listOf("vision", "text")) {
            val ref = manifest.getJSONObject(tower)
            val f = File("$dir/${ref.getString("file")}")
            assertTrue("$tower present", f.exists())
            assertEquals("$tower sha256", ref.getString("sha256"), sha256(f))
            env.createSession(f.absolutePath, OrtSession.SessionOptions()).use { s ->
                println("SPIKE0c $tower loaded_ok=true size_mb=${f.length() / 1_000_000} inputs=${s.inputNames}")
            }
        }
    }

    /** Optional: prove ORT loads an EXTERNAL-DATA model (fp32 text stub + .onnx_data). */
    @Test fun loads_external_data_model_optional() {
        val onnx = File("$dir/ext/text_model.onnx")
        assumeTrue("external-data probe staged",
            onnx.exists() && File("$dir/ext/text_model.onnx_data").exists())
        val env = OrtEnvironment.getEnvironment()
        env.createSession(onnx.absolutePath, OrtSession.SessionOptions()).use { s ->
            println("SPIKE0c external_data loaded_ok=true inputs=${s.inputNames}")
        }
    }
}
```

- [ ] **Step 3: Run + capture**

Run: `./gradlew :app:connectedDebugAndroidTest --tests "*DownloaderSpikeTest*"`
Then: `adb logcat -d | grep SPIKE0c`
Expected: `loads_profile_towers_with_integrity` PASSES (the sha256 `assertEquals` is the gate —
a mismatch fails the test, not just a printed line); logcat shows `vision loaded_ok=true` and
`text loaded_ok=true` with sizes. The optional external-data test passes if its files were
staged, else it is skipped (Assume). Record sizes + whether the external-data load succeeded.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "spike(0c): ORT loads >2GB external-data model on device"
```

---

## Task 12: Spike 0d — so400m memory on the Pixel 7a (instrumented)

**Files:**
- Create: `app/src/androidTest/java/.../spike/So400mMemorySpikeTest.kt`

> Deliverable: confirm `so400m` fp16 vision tower runs batch 32 over a 300-frame loop within
> the device memory budget using the text-first/release strategy, or that the auto-shrink path
> is needed. Drives the §5.5 memory plan in Plan 1/2.

- [ ] **Step 1: Write the instrumented test (exercises the §5.5 text-first/release sequence)**

Reuses the so400m bundle staged in Task 11 (`.../spike/so400m/` has both `vision_model.onnx`
and `text_model.onnx`). Dummy tensors — this measures memory + the load/release sequence, not
parity. Create `So400mMemorySpikeTest.kt`:
```kotlin
package <app.package>.spike

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import android.os.Debug
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

@RunWith(AndroidJUnit4::class)
class So400mMemorySpikeTest {
    private val dir = "/sdcard/Android/data/<app.package>/files/spike/so400m"
    private val res = 384
    private val totalFrames = 300
    private val ladder = listOf(32, 16, 8, 1)

    private fun marker(msg: String) {
        println("SPIKE0d $msg")
        File("$dir/spike0d_progress.log").appendText("$msg\n")  // survives an LMK SIGKILL
    }

    private fun pixelInput(s: OrtSession) =
        s.inputNames.firstOrNull { it.contains("pixel_values") } ?: s.inputNames.first()

    @Test fun text_first_release_then_vision_within_budget() {
        val env = OrtEnvironment.getEnvironment()
        val stop = AtomicBoolean(false)
        val peak = AtomicLong(0)
        // Background sampler: peak PSS catches intra-run spikes the end-of-batch sample misses.
        val poller = Thread {
            while (!stop.get()) {
                val mi = Debug.MemoryInfo().also { Debug.getMemoryInfo(it) }
                peak.updateAndGet { maxOf(it, mi.totalPss.toLong()) }
                try { Thread.sleep(200) } catch (_: InterruptedException) { break }
            }
        }.apply { start() }
        try {
            // 1) TEXT tower: load + run, then RELEASE (the §5.5 strategy).
            marker("text_load_begin")
            val textSession = env.createSession("$dir/text_model.onnx", OrtSession.SessionOptions())
            val idsName = textSession.inputNames.firstOrNull { it.contains("input_ids") }
                ?: textSession.inputNames.first()
            val ids = ByteBuffer.allocateDirect(1 * 64 * 8).order(ByteOrder.nativeOrder()).asLongBuffer()
            OnnxTensor.createTensor(env, ids, longArrayOf(1, 64)).use { t ->
                textSession.run(mapOf(idsName to t)).close()
            }
            marker("text_done peak_pss_kb=${peak.get()}")

            // 2) Worst case: create VISION session while TEXT is still resident.
            marker("vision_load_begin_overlap")
            val visionSession = env.createSession("$dir/vision_model.onnx",
                OrtSession.SessionOptions().apply { addXnnpack(mapOf("intra_op_num_threads" to "4")) })
            marker("overlap_peak_pss_kb=${peak.get()}")

            // 3) RELEASE text — the two-resident window is over.
            textSession.close()
            marker("text_released peak_pss_kb=${peak.get()}")

            // 4) Batched 300-frame vision pass with auto-shrink ladder. One direct buffer,
            //    reused; window it per batch via Buffer.limit (cast avoids covariant issues).
            val inputName = pixelInput(visionSession)
            val buf = ByteBuffer.allocateDirect(ladder.first() * 3 * res * res * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer()
            var chosenBatch = -1
            for (b in ladder) {
                marker("attempt_batch=$b")  // BEFORE the run → SIGKILL leaves the last attempt in the log
                try {
                    var processed = 0
                    while (processed < totalFrames) {
                        val n = minOf(b, totalFrames - processed)
                        val elems = n * 3 * res * res
                        (buf as Buffer).clear(); (buf as Buffer).limit(elems)
                        OnnxTensor.createTensor(env, buf,
                            longArrayOf(n.toLong(), 3, res.toLong(), res.toLong())).use { t ->
                            visionSession.run(mapOf(inputName to t)).use { r ->
                                assertTrue("real output", r[0].value != null)  // skipped run != success
                            }
                        }
                        processed += n
                    }
                    chosenBatch = b
                    marker("ok batch=$b processed=$processed peak_pss_kb=${peak.get()}")
                    break
                } catch (e: OrtException) {
                    marker("OOM_recoverable batch=$b reason=${e.message}")
                }
            }
            visionSession.close()
            marker("DONE chosen_batch=$chosenBatch peak_pss_kb=${peak.get()} completed=${chosenBatch > 0}")
        } finally {
            stop.set(true); poller.join(1000)
        }
    }
}
```

- [ ] **Step 2: Run + capture**

Run: `./gradlew :app:connectedDebugAndroidTest --tests "*So400mMemorySpikeTest*"`
Then capture both the logcat and the SIGKILL-surviving progress file:
```bash
adb logcat -d | grep SPIKE0d
adb shell cat /sdcard/Android/data/<app.package>/files/spike/so400m/spike0d_progress.log
```
Expected: markers through `text_done` → `overlap_peak_pss_kb` (the worst-case two-resident
moment) → `text_released` → `ok batch=N` → `DONE chosen_batch=N completed=true`. If the process
is SIGKILLed by the low-memory killer (uncatchable), the progress file's last `attempt_batch=N`
line shows where it died; re-run starts the ladder lower automatically next time.

- [ ] **Step 3: Decide & record**

State in the spike report: peak PSS at the text/vision **overlap** vs vision-only, the largest
workable batch (`chosen_batch`), and whether text-first/release alone suffices or auto-shrink
is mandatory. This sets the default batch + OOM policy and refines `ram_budget_mb` for so400m
in Plan 1/2.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "spike(0d): so400m fp16 memory + batch ceiling on Pixel 7a"
```

---

## Task 13: Spike results report + unblock Plan 1

**Files:**
- Create: `docs/superpowers/plans/phase0-spike-results.md`
- Modify: `docs/superpowers/specs/2026-06-02-clipcc-android-design.md` (fold in resolved decisions)

- [ ] **Step 1: Write the consolidated report**

Create `docs/superpowers/plans/phase0-spike-results.md` with one section per spike, each
containing: the question, the exact evidence captured (paste the `SPIKE0a/b/c/d` output lines
+ the 0a table), and the **decision**:
- 0a → `tokenizer_lowercase_applied_by` value + whether Kotlin must lowercase.
- 0b → how to read node coverage (API field or JSON parse) for `BackendCapabilityReport`.
- 0c → confirmation external-data models load; any path/placement gotchas.
- 0d → so400m peak PSS, default batch, OOM policy.
- Also record: app module path + package name, actual tower file sizes / which got external data.

- [ ] **Step 2: Fold resolved decisions back into the spec**

In `2026-06-02-clipcc-android-design.md`: update §5.2 (lowercasing resolved), §3/§5.0
(node-coverage mechanism), §5.5 (default batch + OOM policy) to reference the spike results
and drop the "UNRESOLVED"/spike-pending language.

- [ ] **Step 3: Verify all Phase 0 gates are green**

Run the full host test suite:
```bash
.venv-export/bin/python -m pytest tools/android_assets/tests/ -v
```
Expected: all pass. Confirm checklist: 4 manifests exist; fixtures exist; spikes 0a–0d each
have a recorded decision.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/phase0-spike-results.md docs/superpowers/specs/2026-06-02-clipcc-android-design.md
git commit -m "docs(phase0): spike results + fold decisions into spec"
```

---

## Phase 0 Definition of Done (gate into Plan 1)

- [ ] 4 model bundles (`benchmark-v1`) exported with valid `manifest.json` (schema_version 1),
      `logit_scale`/`logit_bias` + provenance (`onnx_source_revision`) populated, every file's
      `sha256` verified (towers expected single-file for this profile; `data_file` null).
- [ ] Golden fixtures (`tokenizer_golden.json`, `preprocess_golden.npz`, `scores_golden.json`)
      generated from the lossless reference pipeline.
- [ ] All host unit tests pass.
- [ ] Spike 0a decision recorded → `tokenizer_lowercase_applied_by` set in manifests.
- [ ] Spike 0b decision recorded → node-coverage mechanism known.
- [ ] Spike 0c proven → external-data model loads in ORT on device.
- [ ] Spike 0d recorded → so400m peak memory + default batch + OOM policy.
- [ ] `phase0-spike-results.md` written; spec updated.
