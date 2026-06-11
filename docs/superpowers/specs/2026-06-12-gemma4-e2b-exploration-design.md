# Gemma 4 E2B Exploration — Design

**Date:** 2026-06-12
**Status:** Approved (adversarially reviewed by 3 independent agents; all blockers folded in)

## Goal

Explore whether a small generative multimodal model (Gemma 4 E2B) is useful for
video understanding tasks that SigLIP2 structurally cannot do (temporal
reasoning, free-text description), and how its label scoring compares to
SigLIP2's sigmoid scores. Exploration-stage: web UI workbench first, hardened
API paths later. Deliverables include *measurements* (latency, RAM, JSON parse
failure rate), not just features.

## Decisions (settled with user)

| Decision | Choice |
|---|---|
| Runtime | HF transformers in-process (`AutoModelForMultimodalLM`), pinned transformers version |
| Memory plan | **Co-resident, big-RAM only.** Gemma 4 E2B is 11.4 GB bf16 (5.1B raw params; PLE streaming is a LiteRT trick, unavailable in transformers). Floor: ~14 GB with SigLIP2 (+1.5 GB) resident. RAM preflight refuses Gemma load below floor. |
| Platforms | Mac dev runs **natively** (MPS bf16; Docker on macOS has no GPU passthrough). Docker CPU = demo-only (fp32 fallback ≈ 22 GB, minutes-scale latency — documented, not optimized). Deployment target: CUDA GPU ≥ 16 GB. |
| UI | Separate page `gemma.html`, top nav on both pages (SigLIP2 \| Gemma 4) |
| Scope | All four interactions, phased: label scoring → Q&A → temporal events → side-by-side compare |

## Architecture

### 1. VLM slot — purpose-built, not ModelManager reuse

`ModelManager` is deliberately single-slot and hardcodes `SigLip2Model`
construction; its lease/hot-swap machinery solves a problem the VLM slot
doesn't have (load-once, no swap). Per Simplicity First, the VLM gets its own
small holder, not a generalized manager:

- `app/models/gemma_vlm.py` — `GemmaVLM` with narrow interface
  `generate(frames, prompt, *, max_new_tokens, cancel_event) -> str`. Does NOT
  implement `BaseModel` (that ABC is embedding/score-shaped).
- `app/models/vlm_slot.py` — `VlmSlot`: lazy single-flight load behind an
  `asyncio.Lock` (concurrent first requests: one loads, others await; no
  double-load). RAM preflight before load (floor configurable, default 14 GB
  co-resident budget); refuse with 503 + clear detail below floor.
- Config: `GEMMA_MODEL_ID` (default `google/gemma-4-E2B-it`),
  `GEMMA_ENABLED` (default true; replaces the incoherent `SKIP_VLM_AUTOLOAD`
  — there is no autoload to skip), `GEMMA_MAX_NEW_TOKENS_*` per mode,
  `GEMMA_MAX_FRAMES` (default 8, hard cap 16).

### 2. Concurrency — dedicated limiter

Gemma generations run tens of seconds (GPU/MPS) to minutes (CPU). The existing
inference `CapacityLimiter` (capacity 2 cpu / 1 gpu) hard-rejects with 429 —
sharing it means one Gemma call starves all SigLIP2 traffic. Therefore:

- New `vlm_limiter` (capacity 1) + `vlm_admission()` in `resource_gates.py`.
- SigLIP2 path untouched.

### 3. Timeout & cancellation — honest bounds

The existing cancellation path kills the ffmpeg *subprocess*; `model.generate`
has no subprocess. Real mechanism:

- Per-mode `max_new_tokens` is the primary bound (label_scores ≈ 150,
  qa ≈ 400, events ≈ 300).
- A `StoppingCriteria` reads the runner's own `cancel_event` (the same one
  `InferenceRunner` sets on deadline) — checked between decode steps only.
- Documented limits: timeout is blind during prefill (vision encode of N
  frames + prompt forward, the largest single chunk); worst-case overrun is
  one decode step. No hard kill exists. Frame cap and token budget are the
  levers, and they are capped accordingly.
- Generation runs in worker thread via `anyio.to_thread.run_sync` as today.

### 4. Routes — three typed endpoints, not a mode discriminator

One endpoint with three response shapes would force `response_model=None`.
Instead, three routes, each multipart like `/classify`, each with its own
response model and FastAPI-native required-field validation:

- `POST /api/v1/gemma/label_scores` — video + labels (parsed/validated with
  the existing `_parse_label_array` discipline). One generation pass; model
  prompted for JSON array. Response: `GemmaScoreItem {label, score, evidence}`
  + metadata with `score_semantics: "gemma4_verbalized_uncalibrated"`.
  Explicitly NOT `ScoreItem` (no `raw_similarity` to fabricate; existing
  `get_policy()` must not receive this semantics value).
- `POST /api/v1/gemma/qa` — video + `prompt` (length-capped). Response:
  markdown text + metadata.
- `POST /api/v1/gemma/events` — video, no labels/prompt. Frames captioned
  with their timestamps; model must **choose from the enumerated frame
  timestamps** (closed set — continuous intervals from 8 snapshots ~7.5 s
  apart are invented precision). Server-side validation: snap to actual frame
  times, clamp to duration, drop inverted intervals. Response reuses the
  existing `Segment` vocabulary (`start_time`/`end_time`/`label`) for future
  overlay compatibility.

All three: latency breakdown in response metadata (extract / prefill+generate /
parse) — a primary exploration deliverable.

### 5. JSON reliability — mechanism, not hope

Plain transformers has no constrained decoding. Plan:

- Pin a transformers version with the patched Gemma 4 chat template;
  `enable_thinking=False` (known E2B/E4B "ghost thought channel" token-leak
  bug breaks JSON parsing otherwise).
- Parse layer: strip code fences → `json.loads` → Pydantic validate → key by
  label, null missing labels → one bounded retry on failure → typed
  `GemmaOutputParseError` (HTTP 502: upstream model produced unusable output)
  if retry fails.
- Parse failure rate is logged and surfaced in metadata — an exploration
  metric. lm-format-enforcer as logits processor is the named escalation if
  the rate is bad.

### 6. Frame pipeline

Reuse `services/video.py` ffprobe/extraction, then a new uniform-subsample
step down to `GEMMA_MAX_FRAMES` (extractor's `max_frames` setting and 512
scale are SigLIP2-shaped; duration validation must not reject long videos
Gemma only needs 8 frames from). Image cost is controlled via the Gemma
processor's **per-image token budget** (default 280), not ad-hoc downscaling.
Route owns the same `try/finally: temp_store.cleanup(request_id)` discipline
as `/classify`.

### 7. Middleware — explicit gating (security)

`RequestGateMiddleware` exact-matches `path == "/api/v1/classify"`; the new
routes would otherwise ship **unauthenticated, with no 413 body limit and no
upload gate**. Change: generalize the upload branch to a set/prefix of upload
paths covering `/api/v1/gemma/*`, applying auth + body-size + upload
admission identically. Middleware tests extended to the new paths.

### 8. Observability

- `GET /api/v1/gemma/status` — `{enabled, loaded, loading, model_id, ram_floor_ok}`.
  The Gemma page polls it to gate its submit button (mirrors index.html's
  model-status gating) and to show a "warming" state during first-load.
- `GET /ready` unchanged (SigLIP2 remains the readiness criterion); Gemma
  cold-start latency on first request is documented behavior, surfaced via
  the status endpoint instead.
- `/api/v1/models` untouched — it is the SigLIP2 registry surface; Gemma is
  not in that registry by design at this stage.

### 9. Web UI

- `app/static/gemma.html`, single-file pattern; new route to serve it (only
  `/` and `/static/vendor/` are routed today).
- Top nav added to both pages: SigLIP2 | Gemma 4.
- Gemma page: upload, mode selector (Label scores / Ask / Events),
  labels-or-prompt input, results pane (bars for scores, text for qa,
  timestamped list for events), latency breakdown display, status-gated
  submit.
- **Compare view (Phase C):** frontend fires `/api/v1/classify`
  (aggregation forced to `mean`/`max` — temporal mode excluded, shape
  mismatch) and `/gemma/label_scores` in parallel. Rendered as a NEW grouped
  bar chart with an **absolute 0–100 % axis** (existing bars are normalized
  to top label — overlaying would lie twice). Visible per-model semantics
  disclaimer: sigmoid similarity vs verbalized uncalibrated self-report —
  positions comparable, magnitudes not. Double upload accepted for
  exploration (capped file size); server-side single-upload compare noted as
  the API-path follow-up.

## Phasing

- **Phase A:** VlmSlot + GemmaVLM + middleware gating + vlm limiter +
  `/gemma/label_scores` + `/gemma/qa` + `/gemma/status` + gemma.html + nav.
  Tests: prompt-build/parse/subsample/validation units (no model needed),
  middleware tests for new paths, one gated integration test (real model,
  skipped in CI).
- **Phase B:** `/gemma/events` (closed-set timestamp prompting + server-side
  snap/clamp validation).
- **Phase C:** compare view (frontend grouped chart + forced aggregation).

## Out of scope (exploration stage)

Logprob-calibrated per-label scoring, hybrid SigLIP2-triage pipeline, audio
track input, streaming generation, CPU latency optimization (measured only),
quantized weights, server-side single-upload compare, Gemma in the
`/api/v1/models` registry.

## Known limitations (stated, accepted)

- Verbalized scores are uncalibrated and implicitly contrastive across
  labels; treated as ordinal, not comparable in magnitude to SigLIP2.
- Timeout cannot interrupt prefill; bounded by frame/token caps.
- CPU Docker is demo-only: fp32 fallback ≈ 22 GB RAM, minutes-scale latency.
- First Gemma request pays model load (download + 11.4 GB load) unless warmed.
