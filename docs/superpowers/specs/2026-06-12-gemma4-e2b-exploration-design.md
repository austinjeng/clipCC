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
- `app/models/vlm_slot.py` — `VlmSlot`: lazy load as a non-blocking state
  machine `idle → loading → loaded | failed`. The blocking `from_pretrained`
  (download + 11.4 GB load) runs in a worker thread via
  `anyio.to_thread.run_sync` — never on the event loop and never while
  holding the `asyncio.Lock` for the whole load (the lock only guards the
  state transition, so status polling and unrelated requests stay live).
  **`POST /api/v1/gemma/warm` is the only load trigger.** Video endpoints
  never load: a cold multipart request would otherwise consume the whole
  upload and hold an upload-admission slot while an 11.4 GB download/load
  runs. Instead they fail fast with 503 + Retry-After unless state is
  `loaded`, checked in middleware **before body consumption** (path + slot
  state is a cheap pre-body gate). `failed` state is exposed with the error
  detail and is retryable via another warm call. Residency preflight before
  load (see ledger below); refuse with 503 below floor.
- **Shared residency ledger** (`app/models/residency.py`): both the SigLIP2
  `ModelManager` preflight and `VlmSlot` go through one **per-device,
  atomic** ledger. Accounting is keyed by device (`cpu`, `cuda:N`, `mps`) —
  the deployment target is CUDA, where host RAM can be plentiful while VRAM
  is the real bottleneck, so a RAM-only check would approve loads that OOM
  the GPU. Protocol: **reserve** pending bytes on the target device before
  load (atomic check-and-reserve against
  `device_free - sum(reservations) - headroom`, using
  `torch.cuda.mem_get_info()` for CUDA, psutil for cpu/mps unified memory),
  **commit** on load success, **rollback** on failure. This replaces the
  current swap-credit assumption in `_check_resources`, which is invalidated
  once a second resident model exists (SigLIP2 hot-swap could otherwise OOM
  a loaded Gemma, and vice versa). Unload policy deferred.
- Config: `GEMMA_MODEL_ID` (default `google/gemma-4-E2B-it`),
  `GEMMA_ENABLED` (default true; replaces the incoherent `SKIP_VLM_AUTOLOAD`
  — there is no autoload to skip), `GEMMA_MAX_NEW_TOKENS_*` per mode,
  `GEMMA_MAX_FRAMES` (default 8, hard cap 16), `GEMMA_MAX_LABELS`
  (default 16 — NOT the classify 50-label cap; see §4),
  `GEMMA_ANALYSIS_WINDOW_SECONDS` (default 60; see §6).

### 2. Concurrency — dedicated limiter

Gemma generations run tens of seconds (GPU/MPS) to minutes (CPU). The existing
inference `CapacityLimiter` (capacity 2 cpu / 1 gpu) hard-rejects with 429 —
sharing it means one Gemma call starves all SigLIP2 traffic. Therefore:

- New `vlm_limiter` (capacity 1) + `vlm_admission()` in `resource_gates.py`.
- SigLIP2 path untouched.

### 3. Timeout & cancellation — soft deadline, stated as such

The existing cancellation path kills the ffmpeg *subprocess*; `model.generate`
has no subprocess. Furthermore `InferenceRunner.run` *waits for the worker to
unwind* after a timeout (`await done.wait()`, inference_runner.py:65) before
returning — correct for lease safety, but it means the HTTP request stays
open until the in-flight compute finishes. For Gemma that in-flight chunk is
the entire prefill or the current decode step, not one SigLIP2 batch. This is
a **soft timeout** and the design names it that:

- Per-mode `max_new_tokens` is the primary real bound, scaled where input
  size varies (see §4).
- A `StoppingCriteria` reads the runner's own `cancel_event` (the same one
  `InferenceRunner` sets on deadline) — checked between decode steps only;
  blind during prefill.
- **Timeout overrun is measured**: time from deadline to actual worker
  unwind is logged per request — an exploration metric that decides whether
  the API path later needs a killable subprocess/service for a hard
  deadline (named follow-up, out of scope now).
- Generation runs in worker thread via `anyio.to_thread.run_sync` as today.

### 4. Routes — three typed endpoints, not a mode discriminator

One endpoint with three response shapes would force `response_model=None`.
Instead, three routes, each multipart like `/classify`, each with its own
response model and FastAPI-native required-field validation:

- `POST /api/v1/gemma/label_scores` — video + labels (parsed/validated with
  the existing `_parse_label_array` discipline, but capped at
  `GEMMA_MAX_LABELS` = 16, not classify's 50 — a fixed token budget cannot
  carry 50 × `{label, score, evidence}`). One generation pass. Prompt uses
  **numeric label IDs** (`1: <label text>`); the model returns IDs, not label
  text — robust against paraphrased labels breaking the join. `max_new_tokens`
  **scales with label count** (base + per-label allowance) instead of a flat
  150. `evidence` is requested only for the top-k (default 3) labels. Parse
  layer validates `score` as strict numeric in [0, 1] (reject strings/out of
  range). Response: `GemmaScoreItem {label, score, evidence?}` + metadata
  with `score_semantics: "gemma4_verbalized_uncalibrated"`. Explicitly NOT
  `ScoreItem` (no `raw_similarity` to fabricate; existing `get_policy()`
  must not receive this semantics value).
- `POST /api/v1/gemma/qa` — video + `prompt` (length-capped). Response:
  markdown text + metadata.
- `POST /api/v1/gemma/events` — video, no labels/prompt. Frames captioned
  with their timestamps; model must **choose from the enumerated frame
  timestamps** (closed set — continuous intervals from 8 snapshots ~7.5 s
  apart are invented precision). Server-side validation: snap to actual frame
  times, clamp to duration, drop inverted intervals. Response: a dedicated
  `GemmaEvent {label, start_time, end_time}` schema — NOT the existing
  `Segment`, which requires `duration`, `stats`, `peak_confidence`, and
  `peak_timestamp` (response.py:100-107) that a VLM cannot honestly supply.
  Field *names* follow the `Segment` vocabulary so a thin adapter can feed
  the temporal overlay later if wanted.

All three: latency breakdown in response metadata (extract / prefill+generate /
parse) — a primary exploration deliverable.

### 5. JSON reliability — mechanism, not hope

Plain transformers has no constrained decoding. Plan:

- Pin a transformers version with the patched Gemma 4 chat template;
  `enable_thinking=False` (known E2B/E4B "ghost thought channel" token-leak
  bug breaks JSON parsing otherwise).
- Parse layer: strip code fences → `json.loads` → Pydantic validate →
  **key strictly by numeric label ID** (the prompt contract from §4): reject
  unknown IDs, reject duplicate IDs, map IDs back to the original label
  strings server-side, null missing IDs → one bounded retry on failure →
  typed `GemmaOutputParseError` (HTTP 502: upstream model produced unusable
  output) if retry fails.
- Parse failure rate is logged and surfaced in metadata — an exploration
  metric. lm-format-enforcer as logits processor is the named escalation if
  the rate is bad.

### 6. Frame pipeline — dedicated Gemma sampler

The existing `FrameExtractor` cannot serve Gemma: it samples with an
`fps=` filter from the start of the video and truncates via `-frames:v`
(video.py:110-119), so "subsample afterwards" would draw all frames from the
opening seconds of a long video, and frame timestamps are reconstructed as
`i / fps` rather than recorded. Instead, a dedicated
`services/gemma_sampler.py`:

- Reuse `ffprobe` metadata (`get_video_info`) and a Gemma-specific
  constraint check (upload duration limit still applies; the
  `max_frames`-vs-duration rejection does not).
- Declare an **analysis window**: `GEMMA_ANALYSIS_WINDOW_SECONDS`
  (default 60, matching the model card's ~60 s @ ~1 fps video support).
  Videos longer than the window analyze the first window by default; window
  `start` is a request parameter so the UI can probe other spans. The
  analyzed span is reported in response metadata.
- Choose `GEMMA_MAX_FRAMES` explicit timestamps uniformly across the window,
  extract each with an ffmpeg seek (`-ss <t> -frames:v 1`), and record the
  **exact requested timestamps** with the frames — these are the same
  timestamps enumerated to the model in events mode, so the closed-set
  prompt and the sampler agree by construction.
- Image cost is controlled via the Gemma processor's **per-image token
  budget** (default 280), not ad-hoc downscaling.
- Route owns the same `try/finally: temp_store.cleanup(request_id)`
  discipline as `/classify`.

### 7. Middleware — explicit gating (security)

`RequestGateMiddleware` exact-matches `path == "/api/v1/classify"`; the new
routes would otherwise ship **unauthenticated, with no 413 body limit and no
upload gate**. Change: an explicit **route policy table**, not a blanket
`/api/v1/gemma/*` prefix (a prefix would push `GET /gemma/status` through
upload admission and body-size plumbing it must not consume):

| Path | Policy |
|---|---|
| `POST /api/v1/gemma/{label_scores,qa,events}` | auth + **pre-body slot-state gate** (503 + Retry-After unless `loaded` — before draining the multipart body, §1) + body-size + upload admission (same as `/classify`) |
| `GET /api/v1/gemma/status`, `POST /api/v1/gemma/warm` | auth only (no upload gate, no body plumbing) |
| `GET /gemma` (UI page) | pass-through (same as `/`) |

Middleware tests extended to all new paths, including the negative cases
(status not consuming upload slots).

### 8. Observability

- `GET /api/v1/gemma/status` — `{enabled, state: idle|loading|loaded|failed,
  error, model_id, ram_floor_ok}`. The Gemma page polls it to gate its
  submit button (mirrors index.html's model-status gating) and to show a
  "warming" state during first-load. `POST /api/v1/gemma/warm` is the
  **only** load trigger (returns 202 + current state); the Gemma page calls
  it on load (or via a "Warm model" button when state is `idle`/`failed`)
  and polls status until `loaded`.
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
- **XSS:** qa output is model-generated text and is rendered via
  `textContent` with `white-space: pre-wrap` — never `innerHTML`, no
  client-side markdown-to-HTML conversion. (Markdown rendering with a
  sanitizer allowlist is a UI follow-up, not exploration scope.) The same
  rule applies to `evidence` strings and event labels.
- **Compare view (Phase C):** frontend fires `/api/v1/classify`
  (aggregation forced to `mean`/`max` — temporal mode excluded, shape
  mismatch) and `/gemma/label_scores` in parallel. Rendered as a NEW grouped
  bar chart with an **absolute 0–100 % axis** (existing bars are normalized
  to top label — overlaying would lie twice). Visible per-model semantics
  disclaimer: sigmoid similarity vs verbalized uncalibrated self-report —
  positions comparable, magnitudes not. **Span parity:** SigLIP2 classifies
  the full video; Gemma analyzes its window (§6). When
  `duration ≤ GEMMA_ANALYSIS_WINDOW_SECONDS` the spans match and the chart
  says so; when longer, each result is annotated with its analyzed span
  ("SigLIP2: full video 0–180 s / Gemma: 0–60 s") and the UI shows a
  span-mismatch notice — no silent cross-span comparison. (Windowed classify
  is not added; honest labeling over new backend scope.) Double upload
  accepted for exploration (capped file size); server-side single-upload
  compare noted as the API-path follow-up.

### 10. Dependencies & Docker

Current `requirements.txt` carries only `torch` + `transformers`;
`requirements-prod.txt` is locked accordingly and the Dockerfile installs
just `torch==2.12.0` from the PyTorch variant index. Gemma 4 multimodal
loading per the model card needs more:

- `accelerate` (pinned) — required for `device_map="auto"`.
- `torchvision` (pinned) — image/video processor path; installed **from the
  same PyTorch variant index as `torch`** in the Dockerfile (mixed-index
  torch/torchvision pairs break ABI), pin matched to the torch pin.
- `transformers` pin raised to a version with `AutoModelForMultimodalLM` and
  the patched Gemma 4 chat template (§5).
- `librosa` NOT added — it serves the audio path, and audio input is out of
  scope; noted here so its absence is a decision, not an oversight. Verify
  at Phase A that the processor does not import it for image/video-only use.

Lock refresh of `requirements-prod.txt` and the Dockerfile torch/torchvision
install line are part of Phase A.

## Phasing

- **Phase A:** VlmSlot (state machine + threaded load) + residency ledger +
  GemmaVLM + gemma_sampler + middleware route policy + vlm limiter +
  `/gemma/label_scores` + `/gemma/qa` + `/gemma/status` + `/gemma/warm` +
  gemma.html + nav + dependency bundle/lock refresh (§10). Tests:
  prompt-build/parse/sampler-timestamp/validation/ledger units (no model
  needed), middleware tests for new paths incl. negative cases (status not
  consuming upload slots; cold POST 503s pre-body), one gated integration
  test (real model, skipped in CI).
- **Phase B:** `/gemma/events` (closed-set timestamp prompting + server-side
  snap/clamp validation).
- **Phase C:** compare view (frontend grouped chart + forced aggregation).

## Out of scope (exploration stage)

Logprob-calibrated per-label scoring, hybrid SigLIP2-triage pipeline, audio
track input, streaming generation, CPU latency optimization (measured only),
quantized weights, server-side single-upload compare, Gemma in the
`/api/v1/models` registry, hard-deadline generation isolation (killable
subprocess — decided later from measured timeout overruns), model unload
policy in the residency ledger, sanitized markdown rendering for qa output.

## Known limitations (stated, accepted)

- Verbalized scores are uncalibrated and implicitly contrastive across
  labels; treated as ordinal, not comparable in magnitude to SigLIP2.
- Timeout is soft: it cannot interrupt prefill or an in-flight decode step,
  and the request stays open until the worker unwinds; overruns are
  measured, frame/token caps are the real bounds.
- CPU Docker is demo-only: fp32 fallback ≈ 22 GB RAM, minutes-scale latency.
- Gemma is unavailable (503 + Retry-After) until explicitly warmed; warm
  pays download + 11.4 GB load.
