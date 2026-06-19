# clipCC

Video classification API that scores frames against text labels using the SigLIP2 vision-language model. FastAPI backend, single-file web UI, Docker-ready.

## Architecture

```
app/
  main.py              # FastAPI app factory, all routes, lifespan
  config.py            # Pydantic settings from env vars
  middleware.py         # Auth, upload concurrency gate, body size
  inference_runner.py   # Threaded pipeline with cooperative timeout
  resource_gates.py     # anyio CapacityLimiters
  temp_store.py         # Temp file lifecycle + janitor task
  models/
    base_model.py       # BaseModel ABC, ScoreBatch dataclass
    siglip2_model.py    # SigLIP2 (sigmoid scoring, independent per-label)
    model_manager.py    # Registry, hot-swap, lease-based concurrency
    residency.py        # Per-device atomic model-memory ledger
    vlm_slot.py         # Gemma load-once state machine (warm-only)
    gemma_vlm.py        # Gemma 4 E2B wrapper (AutoModelForMultimodalLM)
  services/
    scoring.py          # aggregate_mean / aggregate_max / aggregate_temporal
    video.py            # ffprobe + ffmpeg frame extraction
    frame_timeline.py   # Frame interval math, gap/duration calc
    temporal_policy.py  # Threshold behavior per model type
    gemma_sampler.py    # Timestamp-seek frame sampling over analysis window
    gemma_prompts.py    # Gemma prompt build + strict ID-keyed parse
  schemas/
    response.py         # ClassifyResponse, ScoreItem, temporal models
    gemma.py            # Gemma response models (uncalibrated score semantics)
  errors/
    handlers.py         # Typed HTTP exceptions (401, 413, 415, 422, 429, 503, 504)
  static/
    index.html          # Web UI (model selector, upload, Chart.js viz)
    gemma.html          # Gemma 4 exploration UI (separate page, top nav)
tests/                  # ~130 tests, pytest + pytest-asyncio
```

## Key Routes

- `POST /api/v1/classify` — Main endpoint (multipart: video + labels + options)
- `GET/POST /api/v1/models` — List models / load by ID (hot-swap)
- `GET /api/v1/models/active` — Active model metadata + temporal defaults
- `POST /api/v1/gemma/{label_scores,qa}` — Gemma 4 E2B exploration (warm first)
- `GET /api/v1/gemma/status` | `POST /api/v1/gemma/warm` — VLM slot lifecycle
- `GET /gemma` — Gemma web UI
- `GET /live` | `GET /ready` — Health probes

## Dev Workflow

```bash
# Run locally
ALLOW_UNAUTHENTICATED=true uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --reload

# Tests (no model needed for unit tests)
python -m pytest tests/test_scoring.py tests/test_config.py -v

# Full suite (downloads ~800MB model on first run, needs ffmpeg)
python -m pytest tests/ -v

# Docker
docker compose --profile cpu up
```

## Key Env Vars

- `API_KEY` / `ALLOW_UNAUTHENTICATED` — Auth (fail-closed: must set one)
- `DEFAULT_MODEL_ID` — Model to load at startup (default: siglip2-base-patch16-256)
- `CLIP_CACHE_DIR` — Model download cache (default: /app/models)
- `SKIP_MODEL_AUTOLOAD=true` — Skip model load for CI/fast startup
- `DEFAULT_LABELS` — JSON array of preset labels for the UI (default: 3 driving behaviors)
- `GEMMA_ENABLED` / `GEMMA_MODEL_ID` — Gemma 4 E2B slot (warm-only load; 11.4GB bf16, needs ~14GB free)

## Scoring Semantics

- **SigLIP2**: `sigmoid` per label independently. Scores don't sum to 1. Threshold: 0.5.
- Response includes `score_semantics` field (`siglip2_pairwise_sigmoid`).

## Design Patterns

- Blocking inference runs in worker thread (`anyio.to_thread.run_sync`), event loop stays responsive
- Two-level concurrency: separate limiters for uploads vs inference
- Lease-based model hot-swap: in-flight requests finish before new model loads
- Cooperative timeout via `threading.Event` checked between batches

---

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---
