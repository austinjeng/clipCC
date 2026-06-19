# Hybrid Mode — Design Spec

**Date:** 2026-06-17
**Status:** Approved (design); ready for implementation planning
**Topic:** SigLIP2 → top-k frames per label → Gemma verdict + explanation

## Summary

A new **Hybrid** mode chains the two existing models. SigLIP2 (user-chosen
variant) scores every sampled frame against the text labels. For each label
whose peak score clears a threshold, Hybrid re-extracts that label's top-k
strongest frames at Gemma's native resolution and sends *only those frames* to
Gemma 4 E2B for an independent verdict (`present` / `not_present` /
`uncertain`) plus a one-sentence explanation. The response shows SigLIP2's
score and Gemma's verdict side by side — a fast statistical pass narrowed and
audited by a slower generative second opinion.

## Decisions (locked during brainstorming + review)

| Decision | Choice | Why |
|---|---|---|
| Gemma call topology | **One call per label** | Faithful to "top-k frames per label"; clean per-label evidence. Cost bounded by gate + cap below. |
| Gemma output | **Verdict + explanation** (`present`/`not_present`/`uncertain` + reason) | Gemma scores are uncalibrated and *not* magnitude-comparable to SigLIP2 sigmoid; a verdict avoids the false-comparison trap. |
| Which labels verified | **Score ≥ threshold, then top-N by score capped at `max_verified_labels`** | Verify the "hits"; bound worst-case Gemma calls (H2). Known tradeoff: does not catch SigLIP2 false-negatives. |
| Gate aggregation default | **`max` (peak frame)** | Hybrid asks "did this behavior occur at all"; a brief event must clear the gate. `mean` would dilute short events. User can switch to `mean`. |
| UI placement | **New standalone `/hybrid` page** | Clean conceptual separation. Accepts duplicated model/label/warm UI (lifted from the other two pages). |
| Frame selection | **Spread (top-k by per-frame score with ≥0.5s min spacing)**, default k=3 | Avoids handing Gemma 3 near-identical clustered frames; maximizes evidence variety per call. |
| Gemma frame source | **Re-extract selected timestamps at 896px** via `gemma_sampler.extract_frames` | SigLIP2 frames are 512px and deleted mid-scoring (see Architecture); re-extracting a handful of timestamps is cheap, full-res, and yields fresh frames to thumbnail. |
| Thumbnails | **Inline base64 in the response** | No static route, no URL signing/expiry, no cleanup race. Temp files deleted right after encoding. |

## Architecture

### Data flow

```
video ──▶ [SigLIP2 phase]                         ──▶ [Gemma phase]                ──▶ HybridResponse
          extract all frames @ fps @ 512px            for each gated label:
          score every frame → ScoringContext            re-extract its top-k timestamps @ 896px
          (scoring loop deletes frames as it goes)       sort frames chronologically
          aggregate (max) → per-label gate score         downscale → base64 thumbnails
          gate: score ≥ threshold, top-N ≤ cap           1 Gemma call → verdict + why
          select top-k spread frame timestamps/label     (per-label parse isolation)
```

### Why re-extract instead of reusing SigLIP2's frames (review M1 + M2)

The existing classify scoring loop **unlinks each frame file immediately after
scoring its batch** (`app/main.py:742-746`), so no frame files survive the
SigLIP2 phase. SigLIP2 frames are also scaled to **512px** (`app/services/video.py:112`)
while Gemma's pipeline uses **896px** (`app/services/gemma_sampler.py:89`).

So Hybrid does **not** reuse frame files. It keeps only what survives —
per-label scores and per-frame timestamps from `ScoringContext`
(`confidence` tensor `(num_frames, num_labels)` + `FrameSample.approx_timestamp_seconds`,
`app/services/scoring.py:14`) — selects top-k timestamps, then re-extracts just
those few via `gemma_sampler.extract_frames(video, selected_timestamps, …)` at
native 896px. The SigLIP2 scoring loop is reused **as-is** (delete-as-you-go is
fine; we never need the files again). Re-seeking a handful of timestamps is
negligible next to a single Gemma generation.

## Backend

### Endpoint

`POST /api/v1/hybrid` (multipart form):

| Field | Type | Default | Notes |
|---|---|---|---|
| `video` | file | — | required |
| `labels` | JSON string array | — | required (existing max 50) |
| `fps` | float | 1.0 | SigLIP2 frame sampling rate (existing range 0.1–5.0) |
| `aggregation` | string | `max` | `max` or `mean`; drives the displayed SigLIP2 score and the gate |
| `threshold` | float | 0.5 | label gated iff aggregated score ≥ threshold |
| `top_k` | int | 3 | frames per Gemma call; validated to `1..gemma_max_frames_cap` (16) |
| `max_verified_labels` | int | 6 | hard cap on Gemma calls; after threshold gating, keep highest-scoring N (review H2) |
| `instruction` | string | — | optional override of the default Gemma verdict instruction (max 2000 chars) |

### New code (small; everything else reused)

1. **`app/services/hybrid_select.py`** — `select_topk_spread(ctx, label_idx, k, min_gap_seconds=0.5) -> list[FrameRef]`.
   Pure function: rank frames by `ctx.confidence[:, label_idx]` descending,
   greedily accept a frame only if it is ≥ `min_gap_seconds` from every
   already-accepted frame (by `ctx.frames[idx].approx_timestamp_seconds`),
   stop at k. Returns `(frame_index, timestamp_seconds, score)` tuples. Fully
   unit-testable, no model needed. **Caller sorts the result chronologically
   before sending to Gemma** (review L3 — the prompt assumes chronological frames).

2. **`app/services/gemma_prompts.py`** (add to existing file):
   - `build_verdict_prompt(label, instruction=None) -> str` — instructs Gemma
     to judge whether `label` is visible in the supplied chronological frames;
     demands a JSON object
     `{"verdict": "present"|"not_present"|"uncertain", "explanation": "<one sentence>"}`.
     Reuses the existing `enable_thinking=False` discipline.
   - `parse_verdict(text) -> dict` — strict parse tolerating prose
     preamble/postamble; validates `verdict` is one of the three literals and
     `explanation` is a string; raises `ValueError` otherwise.

Reused as-is: SigLIP2 scoring loop + `FrameExtractor.extract`
(`app/services/video.py`), `SigLip2Model.score_batch`
(`app/models/siglip2_model.py:50`), `ScoringContext` + `aggregate_frame_scores`
(`app/services/scoring.py`), `gemma_sampler.extract_frames`
(`app/services/gemma_sampler.py`), `vlm_slot` state machine
(`app/models/vlm_slot.py`), `GemmaVLM.generate(frames, prompt, max_new_tokens, cancel_event)`
(`app/models/gemma_vlm.py:70`), `InferenceRunner` cooperative-timeout
mechanism (`app/inference_runner.py`).

### Per-label parse isolation (review M4)

Hybrid does **not** reuse the route-level Gemma parse path, which raises
`GemmaOutputParseError` after one retry (`app/main.py:482`) and would fail the
whole request. Instead Hybrid wraps each label independently:

```
try parse_verdict(out)                       # attempt 1
except ValueError: retry generate + parse    # attempt 2
except ValueError: verdict = "uncertain";    # degrade — one bad label never
                   keep raw output in result   #   kills the other labels
```

Per-label raw output / parse-failure flag is surfaced in that result's entry.

### Concurrency, gates, deadline

**Middleware route policy (review H1) — required change.** Current middleware
(`app/middleware.py:123`) gates only `/api/v1/classify` plus the Gemma upload
paths; all other paths pass through with no auth/body-size/gate
(`app/middleware.py:235`). Add `/api/v1/hybrid` to the upload-gated routes
**and** the VLM-state precheck (it needs Gemma loaded), i.e. treat it like a
Gemma upload path: **auth → VLM-state fail-fast → upload concurrency → body size.**
This must ship with the endpoint, not after.

**Sequential phases — never both gates at once.** The handler runs two
non-overlapping phases against a **single shared deadline** (review M3):

```python
deadline = monotonic_now() + settings.request_timeout_seconds
async with gates.inference_admission(), manager.acquire() as lease:
    runner_s = InferenceRunner(timeout_seconds=remaining(deadline))
    ctx = <SigLIP2 scores all frames>      # inference (SigLIP2) gate held
# inference gate + SigLIP2 lease released here
async with gates.vlm_admission():
    runner_g = InferenceRunner(timeout_seconds=remaining(deadline))
    <per-label Gemma verdicts>             # VLM gate held
```

`remaining(deadline) = max(0, deadline − monotonic_now())` — the Gemma phase
inherits only the time left after SigLIP2, so total wall-clock cannot exceed
~`request_timeout_seconds` (not ~2×). SigLIP2 lease is dropped before Gemma
starts; a hybrid request never holds both pools simultaneously.

Handler-level preconditions (the middleware VLM-state check covers cold Gemma
before the body drains; the handler still verifies):
- `vlm_slot.state == VlmState.LOADED`, else `503`.
- An active SigLIP2 model in `manager`, else `503`.

## Response schema

New `app/schemas/hybrid.py`:

```jsonc
{
  "results": [
    {
      "label": "texting",
      "siglip2_score": 0.93,                 // peak (max) score
      "gemma_evaluated": true,               // renamed from "verified" (review M5)
      "verdict": "present",                  // null when gemma_evaluated == false
      "explanation": "hand holding phone, screen lit",  // null when not evaluated
      "parse_failed": false,                 // true if degraded to uncertain
      "frames_shown": [
        { "frame_index": 4,  "timestamp_seconds": 4.0,  "score": 0.93,
          "thumbnail": "data:image/jpeg;base64,/9j/4AAQ..." },
        { "frame_index": 22, "timestamp_seconds": 22.0, "score": 0.88,
          "thumbnail": "data:image/jpeg;base64,/9j/4AAQ..." }
      ]
    },
    {
      "label": "sleeping",
      "siglip2_score": 0.40,
      "gemma_evaluated": false,              // below threshold (or truncated by cap)
      "verdict": null,
      "explanation": null,
      "parse_failed": false,
      "frames_shown": []
    }
  ],
  "metadata": {
    "siglip2_model": "siglip2-base-patch16-256",
    "gemma_model": "google/gemma-4-E2B-it",      // real config value (review L1)
    "device": "cpu",
    "frames_analyzed": 60,
    "video_duration_seconds": 60.0,
    "aggregation": "max",
    "threshold": 0.5,
    "top_k": 3,
    "max_verified_labels": 6,
    "labels_above_threshold": 2,
    "labels_truncated": 0,                        // above threshold but dropped by cap (review H2)
    "gemma_calls": 2,
    "siglip2_score_semantics": "siglip2_pairwise_sigmoid",
    "gemma_score_semantics": "gemma4_verbalized_uncalibrated",
    "disclaimer": "<Gemma uncalibrated-score disclaimer>",
    "latency": { "siglip2_seconds": 0.0, "gemma_seconds": 0.0 },
    "parse_retries": 0
  }
}
```

`verdict` is an enum: `present | not_present | uncertain`. `gemma_evaluated`
means "Gemma was run on this label" — **not** "confirmed present"; confirmation
is `verdict == "present"`. `frames_shown[].thumbnail` is a small downscaled
base64 JPEG; temp frame files are deleted after encoding.

## UI — new `/hybrid.html`

Standalone page served at `GET /hybrid`. Top nav across all three pages becomes
`SigLIP2 | Gemma 4 | Hybrid` (add the link to `index.html`, `gemma.html`, and
the new page).

Components (patterns lifted from the existing pages):
- **SigLIP2 model selector + load** — from `index.html` (`GET /api/v1/models`, `POST /api/v1/models/load`).
- **Gemma warm pill + Warm button** — from `gemma.html` (`GET /api/v1/gemma/status`, `POST /api/v1/gemma/warm`).
- **Form** — video upload, labels, fps, aggregation (max/mean), threshold slider, top-k slider, max-verified-labels input, optional Gemma instruction textarea. Submits multipart to `POST /api/v1/hybrid`.
- **Results** — one row per label:
  - SigLIP2 peak-score bar,
  - verdict chip — green `present` / red `not_present` / grey `uncertain`,
  - explanation text,
  - inline thumbnails of `frames_shown` (`<img src={thumbnail}>`).
  - Below-threshold / truncated labels render greyed with "not verified."
- Both score-semantics disclaimers shown (SigLIP2 sigmoid vs Gemma verbalized/uncalibrated) so the two numbers are never read as the same scale.

## Error handling / edge cases

| Case | Behavior |
|---|---|
| No label ≥ threshold | `200`, every row `gemma_evaluated:false`, `gemma_calls:0`. Valid result, not an error. |
| More above-threshold labels than `max_verified_labels` | Verify the top-N by score; the rest render `gemma_evaluated:false`; `labels_truncated` reports the count (review H2). |
| Gemma parse failure (after the 1 retry) | That label degrades to `verdict:"uncertain"`, `parse_failed:true`, raw output retained; per-label isolation — never fails the whole response (review M4). |
| Timeout | Single shared deadline across both phases; cooperative `cancel_event` checked between batches/labels. Returns `408/504` per existing handlers (review M3). |
| `top_k` out of range | Validated to `1..gemma_max_frames_cap`; out-of-range → `422` (review L2). |
| Gemma not warm / no SigLIP2 model | `503` fail-fast — middleware VLM-state precheck (before body drain) + handler check (review H1). |

## Testing

**Unit (no model required):**
- `select_topk_spread` — ranking correctness, ≥0.5s spacing enforced, k truncation, fewer-than-k frames available, chronological re-sort.
- `parse_verdict` — valid JSON, prose-wrapped JSON, invalid verdict literal → raises, malformed → raises.
- Gate selection — given per-label scores, threshold, and `max_verified_labels`: correct labels chosen, correct truncation count, `max` vs `mean` aggregation.

**Middleware (review H1):**
- Unauthenticated `/api/v1/hybrid` → `401`.
- Oversized body → `413`.
- Cold Gemma → `503` before body drain.
- Upload-concurrency exhausted → `429`.

**Integration (gated; needs both models; follows the existing Gemma gated-test pattern):**
- End-to-end `POST /api/v1/hybrid` on a short clip → asserts response shape, that skipped labels carry `verdict:null` / `frames_shown:[]`, that `gemma_calls` equals the number of evaluated labels, and that frames are re-extracted (Gemma receives non-empty images).
- Bounded-calls: many above-threshold labels → `gemma_calls ≤ max_verified_labels`, `labels_truncated` correct (review H2).
- Per-label parse isolation: one label returns invalid JSON while adjacent labels still succeed (review M4).
- Shared deadline: a slow SigLIP2 phase leaves a reduced Gemma budget (review M3).

## Explicitly out of scope (YAGNI)

- Cross-label frame dedup / shared frame budget — unnecessary; each label calls Gemma alone with its own frames.
- A Gemma re-scored 0–1 number — rejected in favor of verdicts (uncalibrated-comparison trap).
- Verifying below-threshold labels (false-negative catch) — add later by switching the gate to "all, capped" if needed.
- Static-URL / lazy-loaded full-size frame serving — thumbnails are inline base64 for v1; revisit only if response size becomes a problem.
