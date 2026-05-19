# Local SigLip2 Model Cache — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pre-download SigLip2 models into a local `./models/` directory with offline mode, structured cache validation, two-tier resource gating, revision pinning, and typed error handling.

**Architecture:** Extend ModelManager with preflight checks (cache validity, resource capacity) before model swap. Add `CLIPCC_OFFLINE` env var to control `local_files_only` in HuggingFace loading. Standalone download script imports the registry as source of truth and writes structured `.validated` markers after round-trip validation.

**Tech Stack:** Python 3.11+, transformers, torch, psutil, FastAPI, pydantic-settings

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `app/models/model_manager.py` | ModelConfig expansion, 2 new registry entries, error types, _is_cached rewrite, resource gating, safe model swap, offline visibility, manifest loading |
| Modify | `app/models/siglip2_model.py` | `local_files_only` + `revision` in from_pretrained calls |
| Modify | `app/config.py` | `CLIPCC_OFFLINE` setting |
| Modify | `app/main.py` | Typed error handling at `/api/v1/models/load` endpoint |
| Create | `scripts/download_models.py` | Pre-download script with validation and manifest writing |
| Modify | `requirements.txt` | Add `psutil>=5.9.0` |
| Modify | `requirements-prod.in` | Add `psutil>=5.9.0` |
| Create | `.gitignore` | Ignore models/, __pycache__, .env, etc. |
| Modify | `.dockerignore` | Add models/ and scripts/ |
| Modify | `.env.example` | Add CLIPCC_OFFLINE and CLIP_CACHE_DIR comments |
| Modify | `tests/test_model_manager.py` | Tests for all new model_manager behavior |
| Create | `tests/test_download_script.py` | Tests for download script logic |
| Modify | `tests/conftest.py` | Add offline settings fixture |

---

### Task 1: Error Types

**Files:**
- Modify: `app/models/model_manager.py:15-17`
- Test: `tests/test_model_manager.py`

- [ ] **Step 1: Write failing tests for new error types**

Add to `tests/test_model_manager.py`:

```python
from app.models.model_manager import (
    ModelManager,
    NoModelLoadedError,
    ModelNotCachedError,
    InsufficientResourcesError,
    SIGLIP2_REGISTRY,
)


class TestErrorTypes:
    def test_model_not_cached_error_is_exception(self):
        err = ModelNotCachedError("siglip2-base-patch16-256")
        assert isinstance(err, Exception)
        assert "siglip2-base-patch16-256" in str(err)

    def test_insufficient_resources_error_is_exception(self):
        err = InsufficientResourcesError("Need 10GB RAM, only 4GB available")
        assert isinstance(err, Exception)
        assert "10GB" in str(err)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_model_manager.py::TestErrorTypes -v`
Expected: FAIL with `ImportError` — `ModelNotCachedError` and `InsufficientResourcesError` not defined.

- [ ] **Step 3: Implement error types**

Add after `NoModelLoadedError` in `app/models/model_manager.py`:

```python
class ModelNotCachedError(Exception):
    """Raised when loading an uncached model in offline mode."""
    pass


class InsufficientResourcesError(Exception):
    """Raised when host RAM/VRAM is below model's minimum requirements."""
    pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_model_manager.py::TestErrorTypes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/models/model_manager.py tests/test_model_manager.py
git commit -m "feat: add ModelNotCachedError and InsufficientResourcesError types"
```

---

### Task 2: ModelConfig Expansion + New Registry Entries

**Files:**
- Modify: `app/models/model_manager.py:19-83`
- Test: `tests/test_model_manager.py`

- [ ] **Step 1: Write failing tests for new ModelConfig fields and registry entries**

Add to `tests/test_model_manager.py`:

```python
from app.models.model_manager import ModelConfig


class TestModelConfig:
    def test_revision_field_defaults_none(self):
        config = ModelConfig(
            model_id="test",
            display_name="Test",
            model_type="siglip2",
            hf_repo="google/test",
            params="0.1B",
            resolution=256,
        )
        assert config.revision is None
        assert config.min_ram_gb is None
        assert config.min_vram_gb is None

    def test_revision_field_accepts_value(self):
        config = ModelConfig(
            model_id="test",
            display_name="Test",
            model_type="siglip2",
            hf_repo="google/test",
            params="0.1B",
            resolution=256,
            revision="abc123",
            min_ram_gb=4,
            min_vram_gb=2,
        )
        assert config.revision == "abc123"
        assert config.min_ram_gb == 4
        assert config.min_vram_gb == 2


class TestRegistryExpansion:
    def test_registry_has_8_models(self):
        assert len(SIGLIP2_REGISTRY) == 8

    def test_large_512_in_registry(self):
        config = SIGLIP2_REGISTRY["siglip2-large-patch16-512"]
        assert config.hf_repo == "google/siglip2-large-patch16-512"
        assert config.params == "0.9B"
        assert config.resolution == 512
        assert config.min_ram_gb == 4

    def test_giant_opt_in_registry(self):
        config = SIGLIP2_REGISTRY["siglip2-giant-opt-patch16-384"]
        assert config.hf_repo == "google/siglip2-giant-opt-patch16-384"
        assert config.params == "~2B"
        assert config.resolution == 384
        assert config.min_ram_gb == 10

    def test_existing_models_unchanged(self):
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        assert config.hf_repo == "google/siglip2-base-patch16-256"
        assert config.params == "0.4B"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_model_manager.py::TestModelConfig tests/test_model_manager.py::TestRegistryExpansion -v`
Expected: FAIL — `ModelConfig` doesn't accept `revision`/`min_ram_gb`/`min_vram_gb`; registry has 6 models, not 8.

- [ ] **Step 3: Expand ModelConfig and add registry entries**

In `app/models/model_manager.py`, replace the `ModelConfig` dataclass:

```python
@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    display_name: str
    model_type: str
    hf_repo: str
    params: str
    resolution: int | str
    revision: str | None = None
    min_ram_gb: int | None = None
    min_vram_gb: int | None = None
```

Add two new entries to `SIGLIP2_REGISTRY` after the `siglip2-so400m-patch16-512` entry:

```python
    "siglip2-large-patch16-512": ModelConfig(
        model_id="siglip2-large-patch16-512",
        display_name="SigLIP2 Large (512px)",
        model_type="siglip2",
        hf_repo="google/siglip2-large-patch16-512",
        params="0.9B",
        resolution=512,
        min_ram_gb=4,
    ),
    "siglip2-giant-opt-patch16-384": ModelConfig(
        model_id="siglip2-giant-opt-patch16-384",
        display_name="SigLIP2 Giant-Opt (384px)",
        model_type="siglip2",
        hf_repo="google/siglip2-giant-opt-patch16-384",
        params="~2B",
        resolution=384,
        min_ram_gb=10,
    ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_model_manager.py::TestModelConfig tests/test_model_manager.py::TestRegistryExpansion -v`
Expected: PASS

- [ ] **Step 5: Run existing tests to check nothing broke**

Run: `python -m pytest tests/test_model_manager.py -v`
Expected: PASS — existing `TestRegistry.test_registry_has_models` uses `>= 6`, so 8 passes.

- [ ] **Step 6: Commit**

```bash
git add app/models/model_manager.py tests/test_model_manager.py
git commit -m "feat: expand ModelConfig with resource metadata, add large-512 and giant-opt models"
```

---

### Task 3: Settings (CLIPCC_OFFLINE)

**Files:**
- Modify: `app/config.py:22`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_config.py` (create if needed — check if it exists first):

```python
from app.config import Settings


class TestOfflineSetting:
    def test_offline_defaults_false(self):
        s = Settings(allow_unauthenticated=True)
        assert s.clipcc_offline is False

    def test_offline_from_env(self, monkeypatch):
        monkeypatch.setenv("CLIPCC_OFFLINE", "1")
        s = Settings(allow_unauthenticated=True)
        assert s.clipcc_offline is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py::TestOfflineSetting -v`
Expected: FAIL — `clipcc_offline` not defined on Settings.

- [ ] **Step 3: Add setting**

In `app/config.py`, add after `skip_model_autoload`:

```python
    clipcc_offline: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py::TestOfflineSetting -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/test_config.py
git commit -m "feat: add CLIPCC_OFFLINE setting for offline model loading"
```

---

### Task 4: Cache Validation (_is_cached Rewrite)

**Files:**
- Modify: `app/models/model_manager.py:162-164`
- Test: `tests/test_model_manager.py`

- [ ] **Step 1: Write failing tests for structured marker validation**

Add to `tests/test_model_manager.py`:

```python
import json


class TestCacheValidation:
    def _write_marker(self, cache_dir, hf_repo, marker_data):
        """Helper: write a .validated marker for a model."""
        model_dir = Path(cache_dir) / f"models--{hf_repo.replace('/', '--')}"
        model_dir.mkdir(parents=True, exist_ok=True)
        marker_path = model_dir / ".validated"
        marker_path.write_text(json.dumps(marker_data))

    def test_not_cached_when_no_directory(self, manager):
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        assert manager._is_cached(config) is False

    def test_not_cached_when_no_marker(self, manager):
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        model_dir = Path(manager.cache_dir) / "models--google--siglip2-base-patch16-256"
        model_dir.mkdir(parents=True)
        assert manager._is_cached(config) is False

    def test_cached_when_valid_marker(self, manager):
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        self._write_marker(manager.cache_dir, config.hf_repo, {
            "schema_version": 1,
            "model_id": "siglip2-base-patch16-256",
            "hf_repo": "google/siglip2-base-patch16-256",
            "revision": "abc123",
            "validated_at": "2026-05-19T10:00:00Z",
        })
        assert manager._is_cached(config) is True

    def test_not_cached_when_model_id_mismatch(self, manager):
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        self._write_marker(manager.cache_dir, config.hf_repo, {
            "schema_version": 1,
            "model_id": "wrong-model-id",
            "hf_repo": "google/siglip2-base-patch16-256",
            "revision": "abc123",
            "validated_at": "2026-05-19T10:00:00Z",
        })
        assert manager._is_cached(config) is False

    def test_not_cached_when_hf_repo_mismatch(self, manager):
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        self._write_marker(manager.cache_dir, config.hf_repo, {
            "schema_version": 1,
            "model_id": "siglip2-base-patch16-256",
            "hf_repo": "wrong/repo",
            "revision": "abc123",
            "validated_at": "2026-05-19T10:00:00Z",
        })
        assert manager._is_cached(config) is False

    def test_not_cached_when_revision_mismatch(self, manager):
        from dataclasses import replace
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        pinned = replace(config, revision="expected-sha")
        self._write_marker(manager.cache_dir, config.hf_repo, {
            "schema_version": 1,
            "model_id": "siglip2-base-patch16-256",
            "hf_repo": "google/siglip2-base-patch16-256",
            "revision": "different-sha",
            "validated_at": "2026-05-19T10:00:00Z",
        })
        assert manager._is_cached(pinned) is False

    def test_cached_when_no_pinned_revision(self, manager):
        """Config has revision=None, marker has a revision -- still valid."""
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        assert config.revision is None
        self._write_marker(manager.cache_dir, config.hf_repo, {
            "schema_version": 1,
            "model_id": "siglip2-base-patch16-256",
            "hf_repo": "google/siglip2-base-patch16-256",
            "revision": "abc123",
            "validated_at": "2026-05-19T10:00:00Z",
        })
        assert manager._is_cached(config) is True

    def test_not_cached_when_marker_is_corrupt_json(self, manager):
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        model_dir = Path(manager.cache_dir) / "models--google--siglip2-base-patch16-256"
        model_dir.mkdir(parents=True)
        (model_dir / ".validated").write_text("not json{{{")
        assert manager._is_cached(config) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_model_manager.py::TestCacheValidation -v`
Expected: FAIL — `_is_cached` takes `hf_repo: str` not `config: ModelConfig`.

- [ ] **Step 3: Rewrite _is_cached**

Add `import json` to the top of `app/models/model_manager.py` if not already present.

Replace `_is_cached` in `app/models/model_manager.py`:

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

Update `list_models` to pass config instead of hf_repo:

```python
    def list_models(self) -> list[dict]:
        result = []
        for config in self.registry.values():
            cached = self._is_cached(config)
            result.append({
                "model_id": config.model_id,
                "display_name": config.display_name,
                "model_type": config.model_type,
                "params": config.params,
                "resolution": config.resolution,
                "loaded": self.active_model_id == config.model_id,
                "cached": cached,
            })
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_model_manager.py::TestCacheValidation -v`
Expected: PASS

- [ ] **Step 5: Run all model_manager tests**

Run: `python -m pytest tests/test_model_manager.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/models/model_manager.py tests/test_model_manager.py
git commit -m "feat: rewrite _is_cached with structured .validated marker validation"
```

---

### Task 5: Resource Gating + Safe Model Swap

**Files:**
- Modify: `app/models/model_manager.py:114-145`
- Modify: `requirements.txt`
- Modify: `requirements-prod.in`
- Test: `tests/test_model_manager.py`

- [ ] **Step 1: Add psutil dependency**

Add `psutil>=5.9.0` to `requirements.txt` (after the last line) and to `requirements-prod.in` (after the last line).

Run: `pip install psutil>=5.9.0`

- [ ] **Step 2: Write failing tests for resource gating**

Add to `tests/test_model_manager.py`:

```python
from dataclasses import replace


class TestResourceGating:
    @pytest.mark.asyncio
    async def test_load_rejects_when_total_ram_insufficient(self, manager):
        """Tier 1: model needs more RAM than the system has."""
        config = SIGLIP2_REGISTRY["siglip2-giant-opt-patch16-384"]
        assert config.min_ram_gb == 10

        with patch("app.models.model_manager.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = MagicMock(
                total=4 * 1e9, available=4 * 1e9
            )
            with pytest.raises(InsufficientResourcesError):
                await manager.load_model("siglip2-giant-opt-patch16-384")

    @pytest.mark.asyncio
    async def test_load_estimates_post_unload_capacity(self, manager):
        """Tier 2: available RAM is low but unloading active model would free enough."""
        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            await manager.load_model("siglip2-base-patch16-256")

        with patch("app.models.model_manager.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = MagicMock(
                total=16 * 1e9,
                available=2 * 1e9,
            )
            with patch("app.models.model_manager.SigLip2Model") as MockModel2:
                mock_new = MagicMock()
                MockModel2.return_value = mock_new
                # base model has min_ram_gb=None, large-384 has min_ram_gb=None
                # so load should pass (no min_ram_gb means no gate)
                await manager.load_model("siglip2-base-patch16-384")
                assert manager.active_model_id == "siglip2-base-patch16-384"

    @pytest.mark.asyncio
    async def test_load_no_gate_when_min_ram_is_none(self, manager):
        """Models without min_ram_gb skip the resource check."""
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        assert config.min_ram_gb is None

        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            await manager.load_model("siglip2-base-patch16-256")
        assert manager.active_model_id == "siglip2-base-patch16-256"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_model_manager.py::TestResourceGating -v`
Expected: FAIL — `load_model` doesn't import or check psutil.

- [ ] **Step 4: Write failing tests for safe model swap preflight**

Add to `tests/test_model_manager.py`:

```python
class TestSafeModelSwap:
    @pytest.mark.asyncio
    async def test_preflight_failure_preserves_active_model(self, manager):
        """If preflight fails, the currently loaded model stays active."""
        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            await manager.load_model("siglip2-base-patch16-256")

        assert manager.active_model_id == "siglip2-base-patch16-256"
        assert manager.active_model is mock_instance

        with patch("app.models.model_manager.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = MagicMock(
                total=4 * 1e9, available=4 * 1e9
            )
            with pytest.raises(InsufficientResourcesError):
                await manager.load_model("siglip2-giant-opt-patch16-384")

        # Active model preserved
        assert manager.active_model_id == "siglip2-base-patch16-256"
        assert manager.active_model is mock_instance
```

- [ ] **Step 5: Run test to verify it fails**

Run: `python -m pytest tests/test_model_manager.py::TestSafeModelSwap -v`
Expected: FAIL — no preflight in `load_model`.

- [ ] **Step 6: Implement resource gating and safe model swap**

Add `import psutil` to the top of `app/models/model_manager.py`.

Update `ModelManager.__init__` to accept `offline`:

```python
    def __init__(self, cache_dir: str, offline: bool = False):
        self.registry = dict(SIGLIP2_REGISTRY)
        self.active_model: BaseModel | None = None
        self.active_model_id: str | None = None
        self.cache_dir = cache_dir
        self._offline = offline
        self._condition = asyncio.Condition()
        self._swapping = False
        self._active_leases = 0
```

Replace the `load_model` method:

```python
    async def load_model(self, model_id: str) -> None:
        config = self.registry[model_id]

        self._preflight_check(config)

        async with self._condition:
            await self._condition.wait_for(lambda: not self._swapping)
            if self.active_model_id == model_id:
                return
            self._swapping = True
            self._condition.notify_all()
            await self._condition.wait_for(lambda: self._active_leases == 0)
            old_model = self.active_model
            self.active_model = None
            self.active_model_id = None

        try:
            del old_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            new_model = SigLip2Model(
                hf_repo=config.hf_repo, cache_dir=self.cache_dir
            )
        except Exception:
            async with self._condition:
                self._swapping = False
                self._condition.notify_all()
            raise

        async with self._condition:
            self.active_model = new_model
            self.active_model_id = model_id
            self._swapping = False
            self._condition.notify_all()

    def _preflight_check(self, config: ModelConfig) -> None:
        self._check_resources(config)

    def _check_resources(self, config: ModelConfig) -> None:
        if config.min_ram_gb is None:
            return

        mem = psutil.virtual_memory()
        required_bytes = config.min_ram_gb * 1.2 * 1e9

        # Tier 1: hard ceiling -- model exceeds total system RAM
        if required_bytes > mem.total:
            raise InsufficientResourcesError(
                f"Model {config.model_id} requires ~{config.min_ram_gb}GB RAM "
                f"(with 1.2x headroom), but system total is "
                f"{mem.total / 1e9:.1f}GB"
            )

        # Tier 2: estimated post-unload capacity
        estimated_available = mem.available
        if self.active_model_id:
            active_config = self.registry.get(self.active_model_id)
            if active_config and active_config.min_ram_gb:
                estimated_available += active_config.min_ram_gb * 1e9

        if required_bytes > estimated_available:
            raise InsufficientResourcesError(
                f"Model {config.model_id} requires ~{config.min_ram_gb}GB RAM "
                f"(with 1.2x headroom), but estimated post-unload available is "
                f"{estimated_available / 1e9:.1f}GB"
            )

        # VRAM check when CUDA available
        if config.min_vram_gb and torch.cuda.is_available():
            free_vram, total_vram = torch.cuda.mem_get_info()
            vram_required = config.min_vram_gb * 1.2 * 1e9
            if vram_required > total_vram:
                raise InsufficientResourcesError(
                    f"Model {config.model_id} requires ~{config.min_vram_gb}GB VRAM, "
                    f"but GPU total is {total_vram / 1e9:.1f}GB"
                )
```

Note: `SigLip2Model` call stays unchanged for now (no `revision`/`offline` args). Those will be added in Task 7 when the constructor is updated.

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_model_manager.py::TestResourceGating tests/test_model_manager.py::TestSafeModelSwap -v`
Expected: PASS

- [ ] **Step 8: Run all model_manager tests**

Run: `python -m pytest tests/test_model_manager.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add app/models/model_manager.py tests/test_model_manager.py requirements.txt requirements-prod.in
git commit -m "feat: add two-tier resource gating and safe model swap with preflight"
```

---

### Task 6: Offline Model Visibility + Cache Preflight

**Files:**
- Modify: `app/models/model_manager.py` (list_models, _preflight_check)
- Test: `tests/test_model_manager.py`

- [ ] **Step 1: Write failing tests for offline filtering**

Add to `tests/test_model_manager.py`:

```python
class TestOfflineVisibility:
    def test_online_list_models_returns_all(self, manager):
        models = manager.list_models()
        assert len(models) == 8

    def test_offline_list_models_excludes_uncached(self, temp_dir):
        mgr = ModelManager(cache_dir=str(temp_dir / "models"), offline=True)
        models = mgr.list_models()
        assert len(models) == 0

    def test_offline_list_models_includes_cached(self, temp_dir):
        cache_dir = str(temp_dir / "models")
        mgr = ModelManager(cache_dir=cache_dir, offline=True)
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        # Write valid marker
        model_dir = Path(cache_dir) / "models--google--siglip2-base-patch16-256"
        model_dir.mkdir(parents=True)
        (model_dir / ".validated").write_text(json.dumps({
            "schema_version": 1,
            "model_id": "siglip2-base-patch16-256",
            "hf_repo": "google/siglip2-base-patch16-256",
            "revision": "abc123",
            "validated_at": "2026-05-19T10:00:00Z",
        }))
        models = mgr.list_models()
        assert len(models) == 1
        assert models[0]["model_id"] == "siglip2-base-patch16-256"


class TestOfflinePreflight:
    @pytest.mark.asyncio
    async def test_offline_load_uncached_raises(self, temp_dir):
        mgr = ModelManager(cache_dir=str(temp_dir / "models"), offline=True)
        with pytest.raises(ModelNotCachedError):
            await mgr.load_model("siglip2-base-patch16-256")

    @pytest.mark.asyncio
    async def test_offline_load_uncached_preserves_active(self, temp_dir):
        mgr = ModelManager(cache_dir=str(temp_dir / "models"), offline=True)
        # Manually set an active model to verify preservation
        mock_model = MagicMock()
        mgr.active_model = mock_model
        mgr.active_model_id = "siglip2-base-patch16-384"

        with pytest.raises(ModelNotCachedError):
            await mgr.load_model("siglip2-base-patch16-256")

        assert mgr.active_model is mock_model
        assert mgr.active_model_id == "siglip2-base-patch16-384"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_model_manager.py::TestOfflineVisibility tests/test_model_manager.py::TestOfflinePreflight -v`
Expected: FAIL — `list_models` doesn't filter; `_preflight_check` doesn't check cache.

- [ ] **Step 3: Implement offline filtering and cache preflight**

Update `list_models` in `app/models/model_manager.py`:

```python
    def list_models(self) -> list[dict]:
        result = []
        for config in self.registry.values():
            cached = self._is_cached(config)
            if self._offline and not cached:
                continue
            result.append({
                "model_id": config.model_id,
                "display_name": config.display_name,
                "model_type": config.model_type,
                "params": config.params,
                "resolution": config.resolution,
                "loaded": self.active_model_id == config.model_id,
                "cached": cached,
            })
        return result
```

Update `_preflight_check`:

```python
    def _preflight_check(self, config: ModelConfig) -> None:
        if self._offline and not self._is_cached(config):
            raise ModelNotCachedError(
                f"Model {config.model_id} is not cached and CLIPCC_OFFLINE is enabled. "
                f"Run: python scripts/download_models.py --models {config.model_id}"
            )
        self._check_resources(config)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_model_manager.py::TestOfflineVisibility tests/test_model_manager.py::TestOfflinePreflight -v`
Expected: PASS

- [ ] **Step 5: Run all model_manager tests**

Run: `python -m pytest tests/test_model_manager.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/models/model_manager.py tests/test_model_manager.py
git commit -m "feat: filter uncached models in offline mode, add cache preflight to load_model"
```

---

### Task 7: SigLip2Model Offline Mode + Revision

**Files:**
- Modify: `app/models/siglip2_model.py:15-24`
- Modify: `app/models/model_manager.py` (SigLip2Model call in load_model)
- Test: `tests/test_siglip2_model.py`

- [ ] **Step 1: Write failing tests for new constructor params**

Add to `tests/test_siglip2_model.py`:

```python
from unittest.mock import patch, MagicMock


class TestSigLip2ModelOffline:
    def test_constructor_accepts_revision_and_offline(self):
        """Verify the constructor passes revision and local_files_only through."""
        with patch("app.models.siglip2_model.AutoProcessor") as MockProc, \
             patch("app.models.siglip2_model.AutoModel") as MockModel:
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            MockModel.from_pretrained.return_value = mock_model
            MockProc.from_pretrained.return_value = MagicMock()

            SigLip2Model(
                hf_repo="google/siglip2-base-patch16-256",
                cache_dir="/tmp/test",
                revision="abc123",
                offline=True,
            )

            MockProc.from_pretrained.assert_called_once_with(
                "google/siglip2-base-patch16-256",
                cache_dir="/tmp/test",
                revision="abc123",
                local_files_only=True,
            )
            call_kwargs = MockModel.from_pretrained.call_args[1]
            assert call_kwargs["revision"] == "abc123"
            assert call_kwargs["local_files_only"] is True

    def test_constructor_defaults_online_no_revision(self):
        """Default behavior: no revision, no local_files_only."""
        with patch("app.models.siglip2_model.AutoProcessor") as MockProc, \
             patch("app.models.siglip2_model.AutoModel") as MockModel:
            mock_model = MagicMock()
            mock_model.to.return_value = mock_model
            MockModel.from_pretrained.return_value = mock_model
            MockProc.from_pretrained.return_value = MagicMock()

            SigLip2Model(
                hf_repo="google/siglip2-base-patch16-256",
                cache_dir="/tmp/test",
            )

            MockProc.from_pretrained.assert_called_once_with(
                "google/siglip2-base-patch16-256",
                cache_dir="/tmp/test",
                revision=None,
                local_files_only=False,
            )
            call_kwargs = MockModel.from_pretrained.call_args[1]
            assert call_kwargs["revision"] is None
            assert call_kwargs["local_files_only"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_siglip2_model.py::TestSigLip2ModelOffline -v`
Expected: FAIL — `SigLip2Model.__init__` doesn't accept `revision` or `offline`.

- [ ] **Step 3: Update SigLip2Model constructor**

Replace `__init__` in `app/models/siglip2_model.py`:

```python
    def __init__(
        self,
        hf_repo: str,
        cache_dir: str,
        revision: str | None = None,
        offline: bool = False,
    ):
        self.hf_repo = hf_repo
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(
            hf_repo,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=offline,
        )
        self.model = AutoModel.from_pretrained(
            hf_repo,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=offline,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_siglip2_model.py::TestSigLip2ModelOffline -v`
Expected: PASS

- [ ] **Step 5: Update ModelManager.load_model to pass new args**

In `app/models/model_manager.py`, update the `SigLip2Model` construction in `load_model`:

```python
            new_model = SigLip2Model(
                hf_repo=config.hf_repo,
                cache_dir=self.cache_dir,
                revision=config.revision,
                offline=self._offline,
            )
```

- [ ] **Step 6: Run all tests**

Run: `python -m pytest tests/test_model_manager.py tests/test_siglip2_model.py -v`
Expected: PASS — existing tests that mock `SigLip2Model` are unaffected; integration tests that construct real models pass because defaults are backward-compatible.

- [ ] **Step 7: Commit**

```bash
git add app/models/siglip2_model.py app/models/model_manager.py tests/test_siglip2_model.py
git commit -m "feat: add offline mode and revision pinning to SigLip2Model loading"
```

---

### Task 8: API Error Handling + Manifest Loading

**Files:**
- Modify: `app/main.py:33,78-80,127-142`
- Modify: `app/models/model_manager.py` (ModelManager.__init__)
- Test: `tests/test_model_manager.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write failing tests for manifest loading**

Add to `tests/test_model_manager.py`:

```python
class TestManifestLoading:
    def test_loads_revisions_from_manifest(self, temp_dir):
        cache_dir = str(temp_dir / "models")
        Path(cache_dir).mkdir(parents=True)
        manifest = {
            "siglip2-base-patch16-256": {
                "revision": "sha-abc123",
                "hf_repo": "google/siglip2-base-patch16-256",
                "validated_at": "2026-05-19T10:00:00Z",
            }
        }
        (Path(cache_dir) / "manifest.json").write_text(json.dumps(manifest))

        mgr = ModelManager(cache_dir=cache_dir)
        config = mgr.registry["siglip2-base-patch16-256"]
        assert config.revision == "sha-abc123"

    def test_no_manifest_leaves_revisions_none(self, temp_dir):
        mgr = ModelManager(cache_dir=str(temp_dir / "models"))
        config = mgr.registry["siglip2-base-patch16-256"]
        assert config.revision is None

    def test_manifest_only_updates_known_models(self, temp_dir):
        cache_dir = str(temp_dir / "models")
        Path(cache_dir).mkdir(parents=True)
        manifest = {
            "unknown-model": {
                "revision": "sha-xyz",
                "hf_repo": "google/unknown",
                "validated_at": "2026-05-19T10:00:00Z",
            }
        }
        (Path(cache_dir) / "manifest.json").write_text(json.dumps(manifest))

        mgr = ModelManager(cache_dir=cache_dir)
        assert len(mgr.registry) == 8

    def test_corrupt_manifest_ignored(self, temp_dir):
        cache_dir = str(temp_dir / "models")
        Path(cache_dir).mkdir(parents=True)
        (Path(cache_dir) / "manifest.json").write_text("not valid json{{{")

        mgr = ModelManager(cache_dir=cache_dir)
        config = mgr.registry["siglip2-base-patch16-256"]
        assert config.revision is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_model_manager.py::TestManifestLoading -v`
Expected: FAIL — `ModelManager.__init__` doesn't read manifest.

- [ ] **Step 3: Implement manifest loading**

Add to `ModelManager.__init__` in `app/models/model_manager.py`, after `self._active_leases = 0`:

```python
        self._load_manifest()
```

Add the method:

```python
    def _load_manifest(self) -> None:
        manifest_path = Path(self.cache_dir) / "manifest.json"
        if not manifest_path.exists():
            return
        try:
            manifest = json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        from dataclasses import replace
        for model_id, entry in manifest.items():
            if model_id in self.registry:
                revision = entry.get("revision")
                if revision:
                    self.registry[model_id] = replace(
                        self.registry[model_id], revision=revision
                    )
```

- [ ] **Step 4: Run manifest tests**

Run: `python -m pytest tests/test_model_manager.py::TestManifestLoading -v`
Expected: PASS

- [ ] **Step 5: Write failing tests for typed error responses**

Add to `tests/test_api.py` (check if it exists first, create the test class appropriately):

```python
import pytest
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport
from app.main import create_app
from app.config import Settings


@pytest.fixture
def test_settings(temp_dir):
    return Settings(
        allow_unauthenticated=True,
        clip_cache_dir=str(temp_dir / "models"),
        temp_dir=str(temp_dir / "tmp"),
        skip_model_autoload=True,
    )


class TestLoadModelErrorHandling:
    @pytest.mark.asyncio
    async def test_unknown_model_returns_400(self, test_settings):
        outer_app = create_app(test_settings)
        transport = ASGITransport(app=outer_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/models/load", json={"model_id": "nonexistent"})
            assert resp.status_code == 400
            assert "error_type" in resp.json()

    @pytest.mark.asyncio
    async def test_uncached_offline_returns_409(self, test_settings):
        test_settings.clipcc_offline = True
        outer_app = create_app(test_settings)
        transport = ASGITransport(app=outer_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/models/load",
                json={"model_id": "siglip2-base-patch16-256"},
            )
            assert resp.status_code == 409
            body = resp.json()
            assert body["error_type"] == "ModelNotCachedError"

    @pytest.mark.asyncio
    async def test_insufficient_resources_returns_422(self, test_settings):
        outer_app = create_app(test_settings)
        transport = ASGITransport(app=outer_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch("app.models.model_manager.psutil") as mock_psutil:
                mock_psutil.virtual_memory.return_value = MagicMock(
                    total=4 * 1e9, available=4 * 1e9
                )
                resp = await client.post(
                    "/api/v1/models/load",
                    json={"model_id": "siglip2-giant-opt-patch16-384"},
                )
            assert resp.status_code == 422
            body = resp.json()
            assert body["error_type"] == "InsufficientResourcesError"
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `python -m pytest tests/test_api.py::TestLoadModelErrorHandling -v`
Expected: FAIL — endpoint returns 500 for all load errors; no `error_type` field.

- [ ] **Step 7: Update main.py**

In `app/main.py`, update the import:

```python
from app.models.model_manager import (
    ModelManager,
    NoModelLoadedError,
    ModelNotCachedError,
    InsufficientResourcesError,
)
```

Update `lifespan` to pass `offline`:

```python
        manager = ModelManager(
            cache_dir=settings.clip_cache_dir,
            offline=settings.clipcc_offline,
        )
```

Replace the `load_model_endpoint`:

```python
    @app.post("/api/v1/models/load")
    async def load_model_endpoint(request: LoadModelRequest):
        manager: ModelManager = state["manager"]
        if request.model_id not in manager.registry:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": f"Unknown model_id: {request.model_id}",
                    "error_type": "ValueError",
                },
            )
        try:
            await manager.load_model(request.model_id)
        except ModelNotCachedError as e:
            return JSONResponse(
                status_code=409,
                content={
                    "detail": str(e),
                    "error_type": "ModelNotCachedError",
                },
            )
        except InsufficientResourcesError as e:
            return JSONResponse(
                status_code=422,
                content={
                    "detail": str(e),
                    "error_type": "InsufficientResourcesError",
                },
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "detail": f"Failed to load model: {str(e)}",
                    "error_type": type(e).__name__,
                },
            )
        return {"status": "loaded", "model_id": request.model_id}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_api.py::TestLoadModelErrorHandling tests/test_model_manager.py::TestManifestLoading -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add app/main.py app/models/model_manager.py tests/test_model_manager.py tests/test_api.py
git commit -m "feat: add manifest loading at startup and typed HTTP error responses"
```

---

### Task 9: Download Script

**Files:**
- Create: `scripts/download_models.py`
- Create: `tests/test_download_script.py`

- [ ] **Step 1: Write tests for download script logic**

Create `tests/test_download_script.py`:

```python
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


class TestPresets:
    def test_dev_preset_has_4_models(self):
        from download_models import DEV_MODELS
        assert len(DEV_MODELS) == 4
        assert "siglip2-base-patch16-256" in DEV_MODELS
        assert "siglip2-large-patch16-512" in DEV_MODELS
        assert "siglip2-so400m-patch14-384" in DEV_MODELS
        assert "siglip2-giant-opt-patch16-384" in DEV_MODELS

    def test_all_preset_matches_registry(self):
        from download_models import get_models_for_preset
        from app.models.model_manager import SIGLIP2_REGISTRY
        models = get_models_for_preset("all")
        assert set(models) == set(SIGLIP2_REGISTRY.keys())

    def test_dev_preset_returns_subset(self):
        from download_models import get_models_for_preset
        models = get_models_for_preset("dev")
        assert len(models) == 4


class TestResolveDefaultCacheDir:
    def test_resolves_to_repo_root(self):
        from download_models import resolve_default_cache_dir
        result = resolve_default_cache_dir()
        assert result.name == "models"
        assert result.parent == Path(__file__).resolve().parent.parent


class TestWriteMarker:
    def test_writes_valid_json_marker(self, tmp_path):
        from download_models import write_validated_marker
        model_dir = tmp_path / "models--google--test"
        model_dir.mkdir()
        write_validated_marker(
            model_dir=model_dir,
            model_id="test",
            hf_repo="google/test",
            revision="abc123",
        )
        marker = json.loads((model_dir / ".validated").read_text())
        assert marker["schema_version"] == 1
        assert marker["model_id"] == "test"
        assert marker["hf_repo"] == "google/test"
        assert marker["revision"] == "abc123"
        assert "validated_at" in marker

    def test_atomic_write(self, tmp_path):
        """Marker should not exist as partial write."""
        from download_models import write_validated_marker
        model_dir = tmp_path / "models--google--test"
        model_dir.mkdir()
        write_validated_marker(
            model_dir=model_dir,
            model_id="test",
            hf_repo="google/test",
            revision="abc123",
        )
        marker_path = model_dir / ".validated"
        assert marker_path.exists()
        json.loads(marker_path.read_text())


class TestWriteManifest:
    def test_writes_manifest_json(self, tmp_path):
        from download_models import write_manifest
        records = {
            "model-a": {"revision": "sha1", "hf_repo": "google/a", "validated_at": "2026-05-19T10:00:00Z"},
            "model-b": {"revision": "sha2", "hf_repo": "google/b", "validated_at": "2026-05-19T10:00:00Z"},
        }
        write_manifest(tmp_path, records)
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["model-a"]["revision"] == "sha1"
        assert manifest["model-b"]["revision"] == "sha2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_download_script.py -v`
Expected: FAIL — `scripts/download_models.py` does not exist.

- [ ] **Step 3: Create scripts directory**

Run: `mkdir -p scripts`

- [ ] **Step 4: Create download_models.py**

Create `scripts/download_models.py`:

```python
#!/usr/bin/env python3
"""Pre-download SigLip2 models into a local cache directory.

Usage:
    python scripts/download_models.py --preset dev
    python scripts/download_models.py --preset all
    python scripts/download_models.py --models siglip2-base-patch16-256,siglip2-giant-opt-patch16-384
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.model_manager import SIGLIP2_REGISTRY

DEV_MODELS = [
    "siglip2-base-patch16-256",
    "siglip2-large-patch16-512",
    "siglip2-so400m-patch14-384",
    "siglip2-giant-opt-patch16-384",
]


def resolve_default_cache_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models"


def get_models_for_preset(preset: str) -> list[str]:
    if preset == "dev":
        return list(DEV_MODELS)
    if preset == "all":
        return list(SIGLIP2_REGISTRY.keys())
    raise ValueError(f"Unknown preset: {preset}")


def write_validated_marker(
    model_dir: Path,
    model_id: str,
    hf_repo: str,
    revision: str,
) -> None:
    marker_data = {
        "schema_version": 1,
        "model_id": model_id,
        "hf_repo": hf_repo,
        "revision": revision,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    marker_path = model_dir / ".validated"
    fd, tmp_path = tempfile.mkstemp(dir=str(model_dir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(marker_data, f, indent=2)
        os.replace(tmp_path, str(marker_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_manifest(cache_dir: Path, records: dict) -> None:
    manifest_path = cache_dir / "manifest.json"
    fd, tmp_path = tempfile.mkstemp(dir=str(cache_dir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(records, f, indent=2)
        os.replace(tmp_path, str(manifest_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def resolve_revision(hf_repo: str) -> str:
    from huggingface_hub import model_info
    info = model_info(hf_repo)
    return info.sha


def download_model(
    model_id: str,
    hf_repo: str,
    cache_dir: Path,
) -> str:
    from transformers import AutoModel, AutoProcessor

    print(f"  Downloading processor for {model_id}...")
    AutoProcessor.from_pretrained(hf_repo, cache_dir=str(cache_dir))

    print(f"  Downloading model for {model_id}...")
    AutoModel.from_pretrained(hf_repo, cache_dir=str(cache_dir))

    revision = resolve_revision(hf_repo)

    print(f"  Validating with local_files_only=True...")
    AutoProcessor.from_pretrained(
        hf_repo, cache_dir=str(cache_dir), local_files_only=True, revision=revision,
    )
    AutoModel.from_pretrained(
        hf_repo, cache_dir=str(cache_dir), local_files_only=True, revision=revision,
    )

    model_dir = cache_dir / f"models--{hf_repo.replace('/', '--')}"
    write_validated_marker(model_dir, model_id, hf_repo, revision)

    return revision


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download SigLip2 models for local development"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory (default: <repo-root>/models)",
    )
    parser.add_argument(
        "--preset",
        choices=["dev", "all"],
        default="dev",
        help="Model set to download (default: dev)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model IDs (overrides --preset)",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or resolve_default_cache_dir()
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.models:
        model_ids = [m.strip() for m in args.models.split(",")]
        for mid in model_ids:
            if mid not in SIGLIP2_REGISTRY:
                print(f"ERROR: Unknown model_id: {mid}")
                print(f"Available: {', '.join(SIGLIP2_REGISTRY.keys())}")
                return 1
    else:
        model_ids = get_models_for_preset(args.preset)

    print(f"Cache directory: {cache_dir}")
    print(f"Models to download: {len(model_ids)}")
    print()

    manifest_records: dict = {}
    failed: list[str] = []

    for model_id in model_ids:
        config = SIGLIP2_REGISTRY[model_id]
        print(f"[{model_id}] ({config.params}, {config.resolution}px)")
        try:
            revision = download_model(model_id, config.hf_repo, cache_dir)
            manifest_records[model_id] = {
                "revision": revision,
                "hf_repo": config.hf_repo,
                "validated_at": datetime.now(timezone.utc).isoformat(),
            }
            print(f"  OK (revision: {revision[:12]}...)")
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(model_id)
        print()

    if manifest_records:
        write_manifest(cache_dir, manifest_records)
        print(f"Manifest written to {cache_dir / 'manifest.json'}")

    print()
    print(f"Results: {len(manifest_records)} succeeded, {len(failed)} failed")
    if failed:
        print(f"Failed models: {', '.join(failed)}")
        return 1

    print()
    print("To use in development:")
    print(f"  export CLIP_CACHE_DIR={cache_dir}")
    print("  export CLIPCC_OFFLINE=1")

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_download_script.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/download_models.py tests/test_download_script.py
git commit -m "feat: add download script with presets, validation markers, and manifest"
```

---

### Task 10: Housekeeping (gitignore, dockerignore, env, deps)

**Files:**
- Create: `.gitignore`
- Modify: `.dockerignore`
- Modify: `.env.example`

- [ ] **Step 1: Create .gitignore**

Create `.gitignore`:

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

- [ ] **Step 2: Update .dockerignore**

Add these lines to the end of `.dockerignore`:

```
models/
scripts/
```

- [ ] **Step 3: Update .env.example**

Add these commented lines at the end of `.env.example`:

```

# Local development: point to repo-local model cache (run scripts/download_models.py first)
# CLIP_CACHE_DIR=./models
# CLIPCC_OFFLINE=1
```

- [ ] **Step 4: Regenerate requirements-prod.txt**

Run: `pip-compile requirements-prod.in -o requirements-prod.txt`

If `pip-compile` is not available, note this as a manual step: "Run `pip-compile requirements-prod.in -o requirements-prod.txt` when pip-tools is available."

- [ ] **Step 5: Commit**

```bash
git add .gitignore .dockerignore .env.example requirements.txt requirements-prod.in
git commit -m "chore: add gitignore, update dockerignore/env for local model cache"
```

---

### Task 11: Integration Smoke Test

**Files:**
- Test: `tests/test_model_manager.py`

- [ ] **Step 1: Write end-to-end offline flow test**

Add to `tests/test_model_manager.py`:

```python
class TestOfflineIntegration:
    @pytest.mark.asyncio
    async def test_full_offline_flow(self, temp_dir):
        """Simulates: download script cached a model, app runs offline, loads it."""
        cache_dir = str(temp_dir / "models")

        # Simulate what download script does: write marker + manifest
        config = SIGLIP2_REGISTRY["siglip2-base-patch16-256"]
        model_dir = Path(cache_dir) / f"models--{config.hf_repo.replace('/', '--')}"
        model_dir.mkdir(parents=True)
        marker = {
            "schema_version": 1,
            "model_id": config.model_id,
            "hf_repo": config.hf_repo,
            "revision": "test-sha-abc",
            "validated_at": "2026-05-19T10:00:00Z",
        }
        (model_dir / ".validated").write_text(json.dumps(marker))
        manifest = {
            config.model_id: {
                "revision": "test-sha-abc",
                "hf_repo": config.hf_repo,
                "validated_at": "2026-05-19T10:00:00Z",
            }
        }
        (Path(cache_dir) / "manifest.json").write_text(json.dumps(manifest))

        # Create offline manager
        mgr = ModelManager(cache_dir=cache_dir, offline=True)

        # Verify manifest loaded revision
        assert mgr.registry[config.model_id].revision == "test-sha-abc"

        # Verify list_models only shows cached model
        models = mgr.list_models()
        cached_ids = [m["model_id"] for m in models]
        assert config.model_id in cached_ids
        assert len(cached_ids) == 1

        # Verify uncached model load is rejected
        with pytest.raises(ModelNotCachedError):
            await mgr.load_model("siglip2-base-patch16-384")

        # Verify cached model passes preflight (mock actual model construction)
        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            await mgr.load_model(config.model_id)

        assert mgr.active_model_id == config.model_id
        assert mgr.active_model is mock_instance
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest tests/test_model_manager.py::TestOfflineIntegration -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v --ignore=tests/test_siglip2_model.py --ignore=tests/test_integration.py`

(Ignoring tests that require actual model downloads. If those tests are already gated behind fixtures, run the full suite instead: `python -m pytest tests/ -v`)

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_model_manager.py
git commit -m "test: add end-to-end offline flow integration test"
```
