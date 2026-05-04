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
class BaseModel(ABC):
    model_type: str              # "clip" or "siglip2"
    device: str                  # "cuda" or "cpu"
    max_token_length: int        # 77 for CLIP, 64 for SigLIP2

    @abstractmethod
    def encode_text(self, texts: list[str]) -> torch.Tensor: ...

    @abstractmethod
    def encode_images(self, images: list[Image]) -> torch.Tensor: ...

    @abstractmethod
    def compute_similarities(self, images, texts) -> tuple[torch.Tensor, float | None]: ...
    # Returns (raw_logits, logit_scale_or_None)
    # CLIP: logit_scale is meaningful, used for softmax scaling
    # SigLIP2: logit_scale is None, logits go through sigmoid directly

    @abstractmethod
    def tokenize_and_check(self, prompts: list[str], max_tokens: int) -> list[int]: ...

    @abstractmethod
    def tokenize_raw(self, prompts: list[str]) -> list[torch.Tensor]: ...
```

#### `ClipModel` (app/models/clip_model.py)

Refactored to extend `BaseModel`. Sets `model_type = "clip"`, `max_token_length = 77`. Internal logic unchanged.

#### `SigLip2Model` (app/models/siglip2_model.py)

New implementation using HuggingFace transformers:

- **Loading:** `AutoModel.from_pretrained(hf_repo)` + `AutoProcessor.from_pretrained(hf_repo)` with `cache_dir` pointing to Docker volume
- **`encode_text`:** Processor tokenizes with `padding="max_length"`, `max_length=64`, `truncation=True`. Then `model.get_text_features()`, L2-normalized.
- **`encode_images`:** Processor handles resize/normalize. Then `model.get_image_features()`, L2-normalized.
- **`compute_similarities`:** Encodes text and images separately via `get_text_features()` / `get_image_features()`, L2-normalizes both, computes `image_features @ text_features.T` for raw cosine similarity. Returns `(cosine_sim, None)` — no logit_scale, caller uses sigmoid.
- **`tokenize_and_check`:** Uses the processor's tokenizer to encode, checks against 64-token limit.
- **`tokenize_raw`:** Tokenizes each prompt individually, returns tensor list.
- Sets `model_type = "siglip2"`, `max_token_length = 64`.
- Uses `torch.inference_mode()` and `torch.autocast("cuda")` on GPU, matching ClipModel patterns.

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

class ModelManager:
    registry: dict[str, ModelConfig]
    active_model: BaseModel | None
    active_model_id: str | None
    cache_dir: str
    _lock: asyncio.Lock
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

**`load_model(model_id)` flow:**

1. Validate `model_id` exists in registry
2. If already loaded, no-op and return
3. Acquire exclusive `_lock`
4. Unload current model (`del` + `torch.cuda.empty_cache()`)
5. Instantiate `SigLip2Model` with config (downloads to `cache_dir` if not cached)
6. Set `active_model` and `active_model_id`
7. Release lock

**`list_models()`** returns registry entries with a `loaded: bool` flag and `cached: bool` (weights exist on disk).

**Startup:** Auto-loads `siglip2-base-patch16-256` during FastAPI lifespan.

### Scoring Adaptation (app/services/scoring.py)

```python
def compute_frame_scores(
    cosine_sim: torch.Tensor,
    logit_scale: float | None,    # None for SigLIP2
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_similarity = cosine_sim.clone()

    if logit_scale is not None:
        # CLIP: scale then softmax across labels
        scaled_logits = cosine_sim * logit_scale
        confidence = torch.softmax(scaled_logits, dim=-1)
    else:
        # SigLIP2: sigmoid (independent per-pair)
        confidence = torch.sigmoid(cosine_sim)

    return confidence, raw_similarity
```

The `logit_scale` parameter flows from `BaseModel.compute_similarities()` through the pipeline. CLIP returns a float, SigLIP2 returns `None`.

### API Changes

#### New Endpoints

```
GET  /api/v1/models        → list available models with active/cached status
POST /api/v1/models/load   → {"model_id": "siglip2-base-patch16-256"} → loads model
GET  /api/v1/models/active  → current model info or 404 if none loaded
```

#### Modified Endpoints

- **`GET /ready`** — returns active model info from ModelManager
- **`POST /api/v1/classify`** — uses `model.max_token_length` for token validation (not hardcoded 77). Passes `logit_scale` (or `None`) to scoring service.

#### Behavior During Model Swap

Requests arriving during a model load queue on the `asyncio.Lock`. They wait rather than getting 503. If the wait exceeds `request_timeout_seconds`, the existing timeout handling applies.

#### Response Schema

`ClassifyResponse` unchanged. One addition to `ClassifyMetadata`:

```python
model_type: str  # "clip" or "siglip2" — so consumers know confidence semantics
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

#### docker-compose.yml (new)

```yaml
services:
  clipcc:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - clipcc-models:/app/models
    environment:
      - ALLOW_UNAUTHENTICATED=true

volumes:
  clipcc-models:
```

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
| `app/config.py` | **Modify** — add default_model_id setting |
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
