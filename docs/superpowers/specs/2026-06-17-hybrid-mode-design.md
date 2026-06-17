# Hybrid Mode — Design Spec

**Date:** 2026-06-17
**Status:** Approved (design); ready for implementation planning
**Topic:** SigLIP2 → top-k frames per label → Gemma verdict + explanation

## Summary

A new **Hybrid** mode chains the two existing models. SigLIP2 (user-chosen
variant) scores every sampled frame against the text labels. For each label
whose aggregated score clears a threshold, Hybrid selects that label's top-k
strongest frames and sends *only those frames* to Gemma 4 E2B for an
independent verdict (`present` / `not_present` / `uncertain`) plus a
one-sentence explanation. The response shows SigLIP2's score and Gemma's
verdict side by side — a fast statistical pass narrowed and audited by a
slower generative second opinion.

## Decisions (locked during brainstorming)

| Decision | Choice | Why |
|---|---|---|
| Gemma call topology | **One call per label** | Faithful to "top-k frames per label"; clean per-label evidence. Cost knob = threshold gating below. |
| Gemma output | **Verdict + explanation** (`present`/`not_present`/`uncertain` + reason) | Gemma scores are uncalibrated and *not* magnitude-comparable to SigLIP2 sigmoid; a verdict avoids the false-comparison trap. |
| Which labels verified | **Above SigLIP2 threshold only** | Verify the "hits"; skip the rest. Cheapest meaningful pass (fewer Gemma calls). Known tradeoff: does not catch SigLIP2 false-negatives. |
| UI placement | **New standalone `/hybrid` page** | Clean conceptual separation. Accepts duplicated model/label/warm UI (lifted from the other two pages). |
| Frame selection | **Spread (top-k by score with ≥0.5s min spacing)**, default k=3 | Avoids handing Gemma 3 near-identical clustered frames; maximizes evidence variety per call. |

## Architecture

### Data flow

```
video ──▶ [SigLIP2 phase]                       ──▶ [Gemma phase]                ──▶ HybridResponse
          extract frames @ fps (to disk)            for each gated label:
          score every frame → ScoringContext          select top-k spread frames
          aggregate → per-label score                 re-open those .jpgs from disk
          gate: score ≥ threshold                      1 Gemma call → verdict + why
          select top-k spread frame indices/label
```

### Key reuse — no second ffmpeg pass, no Gemma sampler

SigLIP2 already extracts every sampled frame to disk (`FrameSample.path`,
`app/services/video.py:31`) and retains the full per-frame-per-label score
tensor (`ScoringContext.confidence`, shape `(num_frames, num_labels)`,
`app/services/scoring.py:14`). Hybrid therefore:

- picks top-k **frame indices** from the tensor (no re-extraction);
- re-opens those exact `.jpg` files for Gemma (`Image.open(ctx.frames[idx].path).convert("RGB")`);
- **bypasses** `gemma_sampler.plan_timestamps` and `gemma_sampler.extract_frames` entirely.

Gemma verifies the literal frames SigLIP2 ranked highest — the correct
semantics for "verification."

## Backend

### Endpoint

`POST /api/v1/hybrid` (multipart form):

| Field | Type | Default | Notes |
|---|---|---|---|
| `video` | file | — | required |
| `labels` | JSON string array | — | required |
| `fps` | float | 1.0 | SigLIP2 frame sampling rate (existing range 0.1–5.0) |
| `aggregation` | string | `mean` | `mean` or `max`; drives the displayed SigLIP2 score and the gate |
| `threshold` | float | 0.5 | label verified iff aggregated score ≥ threshold |
| `top_k` | int | 3 | frames per Gemma call; clamped to Gemma's 16-frame hard cap |
| `instruction` | string | — | optional override of the default Gemma verdict instruction (max 2000 chars) |

### New code (small; everything else reused)

1. **`app/services/hybrid_select.py`** — `select_topk_spread(ctx, label_idx, k, min_gap_seconds=0.5) -> list[int]`.
   Pure function: rank frames by `ctx.confidence[:, label_idx]` descending,
   greedily accept a frame only if it is ≥ `min_gap_seconds` from every
   already-accepted frame (using `ctx.frames[idx].approx_timestamp_seconds`),
   stop at k. Fully unit-testable, no model needed.

2. **`app/services/gemma_prompts.py`** (add to existing file):
   - `build_verdict_prompt(label, instruction=None) -> str` — instructs Gemma
     to judge whether `label` is visible in the supplied frames; demands a
     JSON object `{"verdict": "present"|"not_present"|"uncertain", "explanation": "<one sentence>"}`.
     Reuses the existing `enable_thinking=False` discipline.
   - `parse_verdict(text) -> dict` — strict parse tolerating prose
     preamble/postamble; validates `verdict` is one of the three literals and
     `explanation` is a string; raises `ValueError` otherwise (caller retries
     once, then degrades — see Error handling).

Reused as-is: `FrameExtractor.extract` (`app/services/video.py`),
`SigLip2Model.score_batch` (`app/models/siglip2_model.py:50`),
`ScoringContext` + `aggregate_frame_scores` (`app/services/scoring.py`),
`vlm_slot` state machine (`app/models/vlm_slot.py`),
`GemmaVLM.generate(frames, prompt, max_new_tokens, cancel_event)` (`app/models/gemma_vlm.py:70`),
`InferenceRunner` cooperative-timeout mechanism (`app/inference_runner.py`).

### Concurrency — sequential phases, never both gates at once

Middleware applies the **upload gate** to `/api/v1/hybrid`. The handler runs
two non-overlapping phases so a hybrid request never holds both pools at once:

```python
async with gates.inference_admission(), manager.acquire() as lease:
    ctx = <SigLIP2 scores all frames>      # inference (SigLIP2) gate held
# inference gate + SigLIP2 lease released here
async with gates.vlm_admission():
    <per-label Gemma verdicts>             # VLM gate held
```

Preconditions (fail fast, before any work):
- `vlm_slot.state == VlmState.LOADED`, else `503` "Gemma model is not loaded; warm it first."
- An active SigLIP2 model in `manager`, else `503` "No model loaded."

## Response schema

New `app/schemas/hybrid.py`:

```jsonc
{
  "results": [
    {
      "label": "texting",
      "siglip2_score": 0.91,
      "verified": true,
      "verdict": "present",                 // null when verified == false
      "explanation": "hand holding phone, screen lit",  // null when not verified
      "frames_shown": [
        { "frame_index": 4, "timestamp_seconds": 4.0, "score": 0.93 },
        { "frame_index": 22, "timestamp_seconds": 22.0, "score": 0.88 }
      ]
    },
    {
      "label": "sleeping",
      "siglip2_score": 0.40,
      "verified": false,                     // below threshold, Gemma skipped
      "verdict": null,
      "explanation": null,
      "frames_shown": []
    }
  ],
  "metadata": {
    "siglip2_model": "siglip2-base-patch16-256",
    "gemma_model": "gemma-3n-e2b",
    "device": "cpu",
    "frames_analyzed": 60,
    "video_duration_seconds": 60.0,
    "aggregation": "mean",
    "threshold": 0.5,
    "top_k": 3,
    "gemma_calls": 2,
    "siglip2_score_semantics": "siglip2_pairwise_sigmoid",
    "gemma_score_semantics": "gemma4_verbalized_uncalibrated",
    "disclaimer": "<Gemma uncalibrated-score disclaimer>",
    "latency": { "siglip2_seconds": 0.0, "gemma_seconds": 0.0 },
    "parse_retries": 0
  }
}
```

`verdict` is an enum: `present | not_present | uncertain`. `frames_shown`
lists the frames Gemma actually saw (so the UI can thumbnail them).

## UI — new `/hybrid.html`

Standalone page served at `GET /hybrid`. Top nav across all three pages
becomes `SigLIP2 | Gemma 4 | Hybrid` (add the link to `index.html`,
`gemma.html`, and the new page).

Components (patterns lifted from the existing pages):
- **SigLIP2 model selector + load** — from `index.html` (`GET /api/v1/models`, `POST /api/v1/models/load`).
- **Gemma warm pill + Warm button** — from `gemma.html` (`GET /api/v1/gemma/status`, `POST /api/v1/gemma/warm`).
- **Form** — video upload, labels, fps, aggregation (mean/max), threshold slider, top-k slider, optional Gemma instruction textarea. Submits multipart to `POST /api/v1/hybrid`.
- **Results** — one row per label:
  - SigLIP2 score bar,
  - verdict chip — green `present` / red `not_present` / grey `uncertain`,
  - explanation text,
  - thumbnails of `frames_shown`.
  - Below-threshold labels render greyed with "not verified."
- Both score-semantics disclaimers shown (SigLIP2 sigmoid vs Gemma verbalized/uncalibrated) so the two numbers are never read as the same scale.

## Error handling / edge cases

| Case | Behavior |
|---|---|
| No label ≥ threshold | `200`, every row `verified:false`, `gemma_calls:0`. Valid result, not an error. |
| Gemma parse failure (after the 1 retry) | That label degrades to `verdict:"uncertain"`, explanation notes the parse failure; raw output retained in metadata; `parse_retries` incremented. One bad label never fails the whole response. |
| Timeout | Shared `request_timeout_seconds` budget across both phases; cooperative `cancel_event` checked between batches/labels (existing `InferenceRunner` mechanism). Returns `408/504` per existing handlers. |
| `top_k > 16` | Clamped to Gemma's 16-frame hard cap. |
| Gemma not warm / no SigLIP2 model | `503` fail-fast before any work (see Preconditions). |

## Testing

**Unit (no model required):**
- `select_topk_spread` — ranking correctness, ≥0.5s spacing enforced, k truncation, fewer-than-k frames available.
- `parse_verdict` — valid JSON, prose-wrapped JSON, invalid verdict literal → raises, malformed → raises.
- Threshold gating — given per-label scores + threshold, the correct labels are selected for verification.

**Integration (gated; needs both models loaded; follows the existing Gemma gated-test pattern):**
- End-to-end `POST /api/v1/hybrid` on a short clip → asserts response shape, that skipped (below-threshold) labels carry `verdict:null` / `frames_shown:[]`, and that `gemma_calls` equals the number of verified labels.

## Explicitly out of scope (YAGNI)

- Cross-label frame dedup / shared frame budget — unnecessary; each label calls Gemma alone with its own frames.
- A Gemma re-scored 0–1 number — rejected during brainstorming in favor of verdicts (uncalibrated-comparison trap).
- Verifying below-threshold labels (false-negative catch) — can be added later by switching the gate to "all, capped" if needed.
