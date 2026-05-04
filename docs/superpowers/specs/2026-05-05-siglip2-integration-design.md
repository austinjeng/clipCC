# SigLIP2 Integration Design

## Overview

Add SigLIP2 model support to clipCC with hot-swappable model loading and a minimal web UI. SigLIP2 models use HuggingFace transformers (not open_clip), sigmoid similarity (not softmax), a Gemma-based tokenizer (max 64 tokens vs CLIP's 77), and require lowercased text input.

**Branch:** `SigLip2`

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Backend strategy | Dual-backend (Adapter Pattern) | Keep CLIP as fallback, clean abstraction |
| UI-exposed models | SigLIP2 only | CLIP stays as non-UI fallback |
| Model switching | Hot-swap (no restart) | Better UX, ~30-60s load time |
| Frontend | Plain HTML + vanilla JS | No build tools, fits Python-focused project |
| Default model | `siglip2-base-patch16-256` (auto-load) | Smallest (0.4B), fast startup |
| Model storage | Docker volume, download-on-demand | Small image, models persist across restarts |

## Architecture

### Model Abstraction Layer

```
BaseModel (ABC)
├── ClipModel (open_clip) — existing, refactored to extend BaseModel
└── SigLip2Model (transformers) — new, all UI-selectable models
```

#### `BaseModel` (app/models/base_model.py)

Abstract base class defining the interface:

```python
@dataclass
class ScoreBatch:
    confidence: torch.Tensor      # (num_images, num_texts) — model-appropriate activation applied
    raw_similarity: torch.Tensor  # (num_images, num_texts) — normalized cosine for diagnostics
    logits: torch.Tensor          # (num_images, num_texts) — pre-activation scores
    semantics: str                # "clip_relative_softmax" | "siglip2_pairwise_sigmoid"

class BaseModel(ABC):
    model_type: str              # "clip" or "siglip2"
    device: str                  # "cuda" or "cpu"
    max_token_length: int        # 77 for CLIP, 64 for SigLIP2

    @abstractmethod
    def encode_images(self, images: list[Image]) -> torch.Tensor: ...

    @abstractmethod
    def score_batch(self, images: list[Image], texts: list[str]) -> ScoreBatch: ...
    # Each model owns its scoring pipeline end-to-end:
    # CLIP: encode separately → scale cosine by logit_scale → softmax → ScoreBatch
    # SigLIP2: joint forward → logits_per_image → sigmoid → ScoreBatch
    # raw_similarity always = L2-normalized image_embeds @ text_embeds.T (diagnostic only)

    @abstractmethod
    def validate_prompts(self, prompts: list[str]) -> list[int]: ...
    # Returns token counts using UNTRUNCATED tokenization.
    # Raises or returns counts > max_token_length so caller can reject.

    @abstractmethod
    def tokenize_for_inference(self, prompts: list[str]) -> Any: ...
    # Returns model-ready tokenized inputs (truncated/padded per model requirements).

    @abstractmethod
    def tokenize_raw(self, prompts: list[str]) -> list[torch.Tensor]: ...
    # For duplicate-token-sequence detection. Returns normalized, model-visible token IDs.
    # SigLIP2: lowercased text before tokenization (matching inference behavior).
```

#### `ClipModel` (app/models/clip_model.py)

Refactored to extend `BaseModel`. Sets `model_type = "clip"`, `max_token_length = 77`.

- **`score_batch`:** Encodes text/images separately via open_clip, computes `cosine = image_feats @ text_feats.T`, scales by `logit_scale`, applies softmax. Also stores unscaled cosine as `raw_similarity`. Returns `ScoreBatch(semantics="clip_relative_softmax")`.
- **`validate_prompts`:** Same as current `tokenize_and_check` — uses `tokenizer.encode()` to get raw untruncated count.
- **`tokenize_raw`:** Unchanged — returns full token tensors for duplicate detection.

#### `SigLip2Model` (app/models/siglip2_model.py)

New implementation using HuggingFace transformers:

- **Loading:** `AutoModel.from_pretrained(hf_repo)` + `AutoProcessor.from_pretrained(hf_repo)` with `cache_dir` pointing to Docker volume.
- **`score_batch`:** Uses `model.forward(**processor(text=texts, images=images, padding="max_length", max_length=64, truncation=True, return_tensors="pt"))`. Takes `outputs.logits_per_image` as logits (these include learned scale/bias). Applies `torch.sigmoid(logits)` for confidence. Separately computes `raw_similarity` from L2-normalized `outputs.image_embeds @ outputs.text_embeds.T` (diagnostic cosine only). Returns `ScoreBatch(semantics="siglip2_pairwise_sigmoid")`.
- **`encode_images`:** Processor handles resize/normalize, then `model.get_image_features()`, L2-normalized. Used for batched frame-level scoring where text features are precomputed.
- **`validate_prompts`:** Tokenizes with `truncation=False`, `padding=False` to get true token count. Compares against `max_token_length=64`. Does NOT use processor's default truncating mode — overflow is detected, not hidden.
- **`tokenize_for_inference`:** Tokenizes with `padding="max_length"`, `max_length=64`, `truncation=True` — the required SigLIP2 inference format.
- **`tokenize_raw`:** Lowercases text first (matching SigLIP2 normalization), then tokenizes without truncation. Returns model-visible token IDs for duplicate detection.
- Sets `model_type = "siglip2"`, `max_token_length = 64`.
- Uses `torch.inference_mode()` and `torch.autocast("cuda")` on GPU.

**Batched frame scoring path:** HuggingFace `Siglip2Model.forward()` takes raw token/pixel tensors, not precomputed embeddings. Two options for video frame batching:

- **Simple path (recommended for v1):** Call `score_batch(images, texts)` per frame batch. Text is re-tokenized/re-encoded each batch. Acceptable overhead because text encoding is cheap relative to image encoding for typical label counts (3-10 labels).

- **Optimized path (future):** Expose `prepare_text(texts) → PreparedText` that caches tokenized text inputs. Then `score_images_against_prepared_text(images, prepared) → ScoreBatch` computes image embeddings, then manually applies the learned parameters: `logits = image_embeds @ text_embeds.T * exp(model.logit_scale) + model.logit_bias`. This requires accessing `model.logit_scale` and `model.logit_bias` attributes directly.

For v1, use the simple path. The optimized path can be added later if profiling shows text re-encoding is a bottleneck (unlikely for ≤10 labels).

### Model Manager (app/models/model_manager.py)

```python
@dataclass
class ModelConfig:
    model_id: str               # e.g. "siglip2-base-patch16-256"
    display_name: str           # e.g. "SigLIP2 Base (256px)"
    model_type: str             # "siglip2"
    hf_repo: str                # "google/siglip2-base-patch16-256"
    params: str                 # "0.4B"
    resolution: int | str       # 256, 384, or "adaptive" for NaFlex

class ModelLease:
    """Short-lived reference to the active model. Prevents unload while in use."""
    model: BaseModel
    _manager: ModelManager
    # Used as async context manager:
    # async with manager.acquire() as lease:
    #     lease.model.score_batch(...)

class ModelManager:
    registry: dict[str, ModelConfig]
    active_model: BaseModel | None
    active_model_id: str | None
    cache_dir: str
    _swap_lock: asyncio.Lock          # Exclusive during load/unload
    _active_leases: int               # Count of in-flight requests holding a model reference
    _leases_drained: asyncio.Event    # Signaled when _active_leases drops to 0
```

**Registry — hardcoded SigLIP2 models:**

| model_id | HF repo | Params | Resolution |
|---|---|---|---|
| `siglip2-base-patch16-256` | `google/siglip2-base-patch16-256` | 0.4B | 256 |
| `siglip2-base-patch16-384` | `google/siglip2-base-patch16-384` | 0.4B | 384 |
| `siglip2-large-patch16-256` | `google/siglip2-large-patch16-256` | 0.9B | 256 |
| `siglip2-large-patch16-384` | `google/siglip2-large-patch16-384` | 0.9B | 384 |
| `siglip2-so400m-patch14-384` | `google/siglip2-so400m-patch14-384` | 1B | 384 |
| `siglip2-so400m-patch16-512` | `google/siglip2-so400m-patch16-512` | 1B | 512 |

**`acquire()` → ModelLease (async context manager):**

Lease acquisition must be atomic with respect to `_swap_lock` to prevent a race where `load_model` acquires the lock between the "is lock free?" check and the lease increment:

```python
async def acquire(self, timeout: float) -> AsyncContextManager[ModelLease]:
    async with asyncio.wait_for(self._swap_lock.acquire(), timeout):
        # Briefly hold the lock to atomically check model + increment leases
        if self.active_model is None:
            self._swap_lock.release()
            raise NoModelLoadedError()
        self._active_leases += 1
        model_ref = self.active_model
        self._swap_lock.release()

    # Yield outside the lock — concurrent reads proceed freely
    try:
        yield ModelLease(model=model_ref)
    finally:
        self._active_leases -= 1
        if self._active_leases == 0:
            self._leases_drained.set()
```

Key properties:
- Lock is held only for the brief check+increment (nanoseconds, not for inference duration)
- Multiple concurrent requests can all hold leases simultaneously (no exclusion during inference)
- `load_model` cannot sneak between check and increment because both are under the same lock acquisition
- Wrapped in `asyncio.wait_for(timeout)` so lease acquisition respects `request_timeout_seconds`

**`load_model(model_id)` flow:**

1. Validate `model_id` exists in registry
2. If already loaded, no-op and return
3. Acquire `_swap_lock` and **hold it** (new `acquire()` calls will queue here)
4. Wait for `_leases_drained` event (existing in-flight requests finish, decrement leases to 0)
5. Unload current model (`del` + `torch.cuda.empty_cache()`)
6. Instantiate `SigLip2Model` with config (downloads to `cache_dir` if not cached)
7. Set `active_model` and `active_model_id`
8. Clear `_leases_drained` event (reset for next swap)
9. Release `_swap_lock` (queued requests proceed with new model)

The lock is held for the entire swap duration (steps 3-9). This means new requests queue during a swap, but existing requests with leases complete unblocked.

**`list_models()`** returns registry entries with `loaded: bool` and `cached: bool` (weights exist on disk).

**Startup behavior:**

- FastAPI lifespan completes immediately — no blocking download
- `/live` returns 200 always (process is healthy)
- `/ready` returns 503 until a model is loaded
- Background task (`asyncio.create_task`) attempts to load `siglip2-base-patch16-256` after startup
- If HuggingFace is unreachable, background task logs error and retries with backoff; app remains live
- UI shows "Loading default model..." state on first page load

### Scoring Adaptation (app/services/scoring.py)

Scoring is now model-owned. Each model's `score_batch()` returns a `ScoreBatch` with confidence already computed using the correct activation function. The scoring service receives `ScoreBatch` objects and only handles aggregation (mean/max across frames):

```python
def aggregate_frame_scores(
    batches: list[ScoreBatch],
    labels: list[str],
    frames: list[FrameSample],
    aggregation: str,
) -> tuple[list[ScoreItem], BestMatch]:
    # Stack confidence tensors from all batches
    all_confidence = torch.cat([b.confidence for b in batches], dim=0)
    all_raw_sim = torch.cat([b.raw_similarity for b in batches], dim=0)
    # Delegate to aggregate_mean or aggregate_max (unchanged logic)
    ...
```

The scoring service no longer decides softmax vs sigmoid — that decision lives in the model implementation where it belongs. `ScoreBatch.semantics` is passed through to the response metadata so API consumers know the confidence interpretation.

### API Changes

#### New Endpoints

```
GET  /api/v1/models        → list available models with active/cached status
POST /api/v1/models/load   → {"model_id": "siglip2-base-patch16-256"} → loads model
GET  /api/v1/models/active  → current model info or 404 if none loaded
```

#### Authentication for Model Endpoints

`RequestGateMiddleware` currently only authenticates `/ready` and `/api/v1/classify` — all other paths pass through (line 155-156). Model management endpoints must be protected:

- Extend `RequestGateMiddleware._check_auth()` to cover all `/api/v1/models*` paths
- Same `X-API-Key` header check as other authenticated endpoints
- Without this, anyone with network access could trigger large downloads and GPU memory churn

Implementation: add a path-prefix check in the middleware's `__call__`:

```python
# /api/v1/models*: auth required, no body size or concurrency gates
if path.startswith("/api/v1/models"):
    if not self._check_auth(scope):
        await _send_json_response(send, 401, {"detail": "..."})
        return
    await self._app(scope, receive, send)
    return
```

#### Modified Endpoints

- **`GET /live`** — always 200 (process healthy, no model dependency)
- **`GET /ready`** — 200 with active model info if loaded, 503 if no model loaded yet
- **`POST /api/v1/classify`** — acquires a `ModelLease` via `manager.acquire()`. Uses `lease.model.max_token_length` for validation, `lease.model.validate_prompts()` for token overflow detection, `lease.model.score_batch()` for scoring. Lease released when request completes.

#### Behavior During Model Swap

1. New requests calling `manager.acquire()` queue on `_swap_lock`
2. In-flight requests (holding leases) complete naturally
3. Once all leases drain, old model is freed and new model loads
4. Queued requests resume with the new model

If a request waits longer than `request_timeout_seconds` for a lease (because a swap is in progress), `asyncio.wait_for` raises `TimeoutError` → mapped to existing `InferenceTimeoutError`.

#### Response Schema

`ClassifyResponse` unchanged. Additions to `ClassifyMetadata`:

```python
model_type: str       # "clip" or "siglip2"
score_semantics: str  # "clip_relative_softmax" or "siglip2_pairwise_sigmoid"
                      # Tells consumers whether confidences sum to 1 or are independent
```

### UI (app/static/index.html)

Single static HTML file served at `GET /`. Plain HTML + vanilla JS, no dependencies.

```
┌─────────────────────────────────────┐
│  clipCC                             │
├─────────────────────────────────────┤
│  Model: [siglip2-base-patch16-256▼] │
│         [Load Model]                │
│         Status: ● Loaded (0.4B)     │
│                                     │
│  Video: [Choose file]               │
│  Labels: [________________]         │
│  Prompt: [This is a photo of {}. ]  │
│  FPS:    [1.0]  Aggregation: [mean] │
│                                     │
│         [Classify]                   │
├─────────────────────────────────────┤
│  Results:                           │
│  ● driving    0.847  ██████████░░   │
│  ○ parking    0.312  ████░░░░░░░░   │
│  ○ reversing  0.091  █░░░░░░░░░░░   │
└─────────────────────────────────────┘
```

**Behaviors:**

- Dropdown populated from `GET /api/v1/models` on page load
- "Load Model" → `POST /api/v1/models/load`, spinner during download/load
- Prompt template auto-fills `"This is a photo of {}."` for SigLIP2 models
- Labels input accepts comma-separated (split on comma) or JSON array
- Results rendered as horizontal bars with confidence values
- Classify button disabled until a model is loaded
- Error states shown inline (timeout, validation errors, 429)

### Docker Changes

#### Dockerfile

- Remove the `RUN python -c "import open_clip..."` model bake step
- Add `transformers>=4.50.0` and `sentencepiece` to requirements
- Keep `open-clip-torch` in requirements (ClipModel fallback)
- Default `CMD` unchanged

#### docker-compose.yml (modify existing, not replace)

Extend existing CPU/GPU profiles with model-cache volume and new env var:

```yaml
services:
  clipcc-cpu:
    build:
      context: .
      args:
        TORCH_VARIANT: cpu
    ports:
      - "8000:8000"
    volumes:
      - clipcc-models:/app/models
    environment:
      - MAX_FILE_SIZE_MB=500
      - MAX_DURATION_SECONDS=300
      - MAX_FRAMES=300
      - DEFAULT_FPS=1.0
      - MAX_CONCURRENT_REQUESTS=2
      - ALLOW_UNAUTHENTICATED=true
      - DEFAULT_MODEL_ID=siglip2-base-patch16-256
    profiles: ["cpu"]

  clipcc-gpu:
    build:
      context: .
      args:
        TORCH_VARIANT: cu121
    ports:
      - "8000:8000"
    volumes:
      - clipcc-models:/app/models
    environment:
      - MAX_FILE_SIZE_MB=500
      - MAX_DURATION_SECONDS=300
      - MAX_FRAMES=300
      - DEFAULT_FPS=1.0
      - MAX_CONCURRENT_REQUESTS=1
      - ALLOW_UNAUTHENTICATED=true
      - DEFAULT_MODEL_ID=siglip2-so400m-patch14-384
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    profiles: ["gpu"]

volumes:
  clipcc-models:
```

Changes from existing compose:
- Added `clipcc-models` named volume to both services
- Added `DEFAULT_MODEL_ID` env var (larger default for GPU, smaller for CPU)
- Preserved all existing env vars, build args, GPU reservations, and profiles

Models download to `/app/models` on first load. Volume persists across container restarts.

### Dependencies Added

```
transformers>=4.50.0
sentencepiece
```

### Files Changed/Created

| File | Action |
|---|---|
| `app/models/base_model.py` | **Create** — ABC |
| `app/models/siglip2_model.py` | **Create** — SigLIP2 implementation |
| `app/models/model_manager.py` | **Create** — registry + hot-swap |
| `app/models/clip_model.py` | **Modify** — extend BaseModel |
| `app/services/scoring.py` | **Modify** — sigmoid branch |
| `app/main.py` | **Modify** — ModelManager, new endpoints, static file serving |
| `app/schemas/response.py` | **Modify** — add model_type to metadata |
| `app/config.py` | **Modify** — add `default_model_id` setting (env: `DEFAULT_MODEL_ID`) |
| `app/static/index.html` | **Create** — UI |
| `requirements.txt` | **Modify** — add transformers, sentencepiece |
| `Dockerfile` | **Modify** — remove bake step |
| `docker-compose.yml` | **Create** — volume mount |
| `tests/test_siglip2_model.py` | **Create** — SigLIP2 model tests |
| `tests/test_model_manager.py` | **Create** — manager tests |
| `tests/test_clip_model.py` | **Modify** — verify BaseModel conformance |

### Testing Strategy

- **Unit tests:** SigLip2Model encode/tokenize methods, ModelManager load/swap/list
- **Integration tests:** `/api/v1/models` endpoints, `/classify` with SigLIP2 active
- **Existing tests:** ClipModel tests updated to verify BaseModel interface compliance
- **Manual:** UI workflow — pick model → load → upload video → see results
