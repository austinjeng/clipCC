# Local SigLip2 Model Cache

Pre-download SigLip2 model variants into a local `./models/` directory so development works fully offline. One model per size tier for full coverage.

## Target Models

| Model ID | HF Repo | Params | Resolution | Min RAM (GB) | Status |
|---|---|---|---|---|---|
| `siglip2-base-patch16-256` | `google/siglip2-base-patch16-256` | 0.4B | 256px | 2 | Existing |
| `siglip2-large-patch16-512` | `google/siglip2-large-patch16-512` | 0.9B | 512px | 4 | New registry entry |
| `siglip2-so400m-patch14-384` | `google/siglip2-so400m-patch14-384` | 1B | 384px | 5 | Existing |
| `siglip2-giant-opt-patch16-384` | `google/siglip2-giant-opt-patch16-384` | ~2B | 384px | 10 | New registry entry |

## Approach

Pre-populate the HuggingFace cache by calling `AutoModel.from_pretrained()` and `AutoProcessor.from_pretrained()` with `cache_dir=./models`. Add `local_files_only=True` to the app's model loading path when offline mode is enabled. Enforce revision pinning end-to-end so the same SHA flows from download through manifest to runtime loading. Validate cache completeness with round-trip loading and structured markers. Gate model loading on host resource capacity. Surface all new failure modes as typed exceptions with appropriate HTTP status codes.

## Changes

### 1. Registry Expansion (`app/models/model_manager.py`)

Expand `ModelConfig` with resource and revision metadata:

```python
@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    display_name: str
    model_type: str
    hf_repo: str
    params: str
    resolution: int | str
    revision: str | None = None    # pinned HF commit SHA (populated from manifest)
    min_ram_gb: int | None = None   # minimum RAM for float32 CPU loading
    min_vram_gb: int | None = None  # minimum VRAM for float16 GPU loading
```

Add two new entries to `SIGLIP2_REGISTRY`:

- `siglip2-large-patch16-512`: large variant at 512px resolution, 0.9B params, `min_ram_gb=4`
- `siglip2-giant-opt-patch16-384`: giant-opt variant at 384px resolution, ~2B params, `min_ram_gb=10`

Both follow the existing pattern with `model_type="siglip2"`.

### 2. Offline Mode (`app/models/siglip2_model.py`)

When `CLIPCC_OFFLINE=1` (or `TRANSFORMERS_OFFLINE=1`), pass `local_files_only=True` to both `AutoProcessor.from_pretrained()` and `AutoModel.from_pretrained()`. Also pass `revision=` when the config provides a pinned SHA, ensuring the loaded snapshot matches exactly what was downloaded.

Default behavior (no env var): unchanged — models download on first use as before. Revision is still passed when available for reproducibility.

### 3. Download Script (`scripts/download_models.py`)

Imports `SIGLIP2_REGISTRY` from `app.models.model_manager` — single source of truth, no drift.

Behavior:
- For each model: downloads processor and model via `from_pretrained(hf_repo, cache_dir=target_dir)`
- Records the resolved HF commit SHA per model
- **Validation**: after download, performs a `local_files_only=True` round-trip load of both processor and model
- **Marker write**: on validation success, writes a structured `.validated` JSON marker atomically (write to temp file, then `os.replace` to final path). Marker contents:

```json
{
  "schema_version": 1,
  "model_id": "siglip2-base-patch16-256",
  "hf_repo": "google/siglip2-base-patch16-256",
  "revision": "abc123def456...",
  "validated_at": "2026-05-19T10:30:00Z"
}
```

- Writes `<cache_dir>/manifest.json` aggregating all validated models and their SHAs
- Prints progress (model name, download status, SHA)
- CLI args:
  - `--cache-dir` (default: `./models` resolved relative to repo root via `Path(__file__).resolve().parent.parent / "models"`)
  - `--preset` — named model sets: `dev` (the 4 target models), `all` (entire registry)
  - `--models` (optional) — comma-separated model IDs from the registry, overrides preset
- Exit code 0 on success, 1 on any download or validation failure

### 4. Cache Validation (`app/models/model_manager.py`)

Replace `_is_cached` directory-existence check with structured marker validation:

```python
def _is_cached(self, config: ModelConfig) -> bool:
    cache_path = Path(self.cache_dir) / f"models--{config.hf_repo.replace('/', '--')}"
    marker_path = cache_path / ".validated"
    if not marker_path.exists():
        return False
    try:
        marker = json.loads(marker_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if marker.get("model_id") != config.model_id:
        return False
    if marker.get("hf_repo") != config.hf_repo:
        return False
    if config.revision and marker.get("revision") != config.revision:
        return False
    return True
```

Validates model identity and revision match, not just file presence. A stale or wrong-revision marker returns `False`.

Method signature changes from `_is_cached(self, hf_repo: str)` to `_is_cached(self, config: ModelConfig)` — all existing call sites already have the config available.

### 5. Offline Model Visibility (`app/models/model_manager.py`)

In offline mode (`CLIPCC_OFFLINE=1`), `list_models` **filters out** uncached models entirely from the response. The API returns only models that are cached and loadable — the UI never sees models it cannot use.

Online mode: unchanged — all registry models returned, `cached` field indicates download status.

### 6. Dependencies (`requirements.txt`, `requirements-prod.txt`)

Add `psutil>=5.9.0` to both files. Used for RAM capacity checks in resource gating.

### 7. Resource Gating (`app/models/model_manager.py`)

Two-tier resource check:

**Tier 1 — Hard ceiling (total system capacity):** Compare `min_ram_gb` against `psutil.virtual_memory().total` (and `min_vram_gb` against `torch.cuda.get_device_properties().total_memory` when CUDA is available). If the model exceeds total system capacity, reject immediately — no amount of unloading helps. Apply a 1.2x headroom multiplier to account for transient loading overhead (model weights must be deserialized before the old model's memory is fully reclaimed).

**Tier 2 — Estimated post-unload capacity:** Compare `min_ram_gb` against `available + estimated_active_model_memory`. Estimate the active model's footprint from its `min_ram_gb` registry metadata (or 0 if no model is loaded). This avoids rejecting a larger replacement model that would fit after unload frees memory.

```
estimated_available = psutil.virtual_memory().available
if active_model_config:
    estimated_available += active_model_config.min_ram_gb * 1e9
passes = (target_config.min_ram_gb * 1.2 * 1e9) <= estimated_available
```

Same logic for VRAM using `torch.cuda.mem_get_info()` free memory + active model's `min_vram_gb`.

If the host cannot support the model after both tiers, `load_model` raises `InsufficientResourcesError` *before* clearing the active model. The service continues running with whatever model was previously loaded.

### 8. Safe Model Swap (`app/models/model_manager.py`)

Restructure `load_model` to preflight before unloading:

1. **Preflight**: check cache completeness (`_is_cached`) and resource capacity (tier 1 + tier 2 from Section 7). Reject with typed error if either fails. Active model untouched.
2. **Unload**: clear active model, free memory.
3. **Load**: construct new model. On failure, leave service in no-model state (existing behavior for construction errors, but now only reachable after preflight passed).

This eliminates the case where a missing cache or insufficient resources drops the active model with no replacement.

### 9. Error Types and HTTP Status Codes

New exception hierarchy in `app/models/model_manager.py`:

```python
class ModelNotCachedError(Exception):
    """Raised when loading an uncached model in offline mode."""
    pass

class InsufficientResourcesError(Exception):
    """Raised when host RAM/VRAM is below model's minimum requirements."""
    pass
```

HTTP status mapping in `app/main.py` at the `/api/v1/models/load` endpoint:

| Exception | HTTP Status | When |
|---|---|---|
| `KeyError` (unknown model_id) | `400 Bad Request` | Model ID not in registry |
| `ModelNotCachedError` | `409 Conflict` | Offline mode, model not cached or marker invalid |
| `InsufficientResourcesError` | `422 Unprocessable Entity` | Host RAM/VRAM below `min_ram_gb`/`min_vram_gb` |
| Other `Exception` | `500 Internal Server Error` | Unexpected construction failure |

Response body for all error cases: `{"detail": "<descriptive message>", "error_type": "<exception class name>"}`.

### 10. Revision Pinning End-to-End

Single flow for revision consistency:

1. **Download script** resolves the HF commit SHA at download time, writes it to `.validated` marker (structured JSON) and `manifest.json`.
2. **App startup** (when `CLIPCC_OFFLINE=1`): reads `manifest.json` from the cache dir, populates `ModelConfig.revision` for each model found in the manifest.
3. **`from_pretrained`** calls pass `revision=<sha>` when available, both for processor and model. This ensures the loaded snapshot is exactly what was downloaded, even if HF's `main` branch has moved.
4. **`_is_cached`** cross-checks the marker's revision against the config's revision. Mismatches return not-cached.

Online mode without manifest: `revision=None`, behaves as before (latest from HF).

### 11. `.gitignore` (new file)

```
models/
__pycache__/
*.pyc
.env
.env.*
tmp/
videos/
.pytest_cache/
```

### 12. `.dockerignore` Update

Add `models/` and `scripts/` to prevent the local model cache and dev scripts from bloating Docker build context.

### 13. `.env.example` Update

Add commented lines:
```
# Local development: point to repo-local model cache (run scripts/download_models.py first)
# CLIP_CACHE_DIR=./models
# CLIPCC_OFFLINE=1
```

## Dev Workflow

```bash
# One-time setup (resolves to repo-root/models/ regardless of cwd)
python scripts/download_models.py --preset dev

# For fully offline registry (all models):
# python scripts/download_models.py --preset all

# Set env for local dev
export CLIP_CACHE_DIR=./models
export CLIPCC_OFFLINE=1

# Run the app — only cached models are visible, fail-fast on issues
uvicorn app.main:app
```

Note: `--preset dev` downloads the 4 target models. Uncached models are excluded from API responses when `CLIPCC_OFFLINE=1`. Use `--preset all` to cache the entire registry for a fully offline experience across all variants.

## Disk Estimate

~15-25GB total for `--preset dev` (4 models, safetensors format). ~20-30GB for `--preset all` (8 models). The HF cache layout uses `models--google--siglip2-*` subdirectories with blob storage and symlinks.

## Out of Scope

- Docker image pre-baking (models stay as runtime downloads in production)
- Automatic revision refresh (manually re-run the download script to update)
