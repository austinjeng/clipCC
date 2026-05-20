# Contrast Scoring Mode

**Date:** 2026-05-20
**Status:** Approved (rev 4 — final)

## Overview

New aggregation mode (`contrast`) that scores a video against two label groups — positive and negative — and returns a three-way verdict (positive / negative / uncertain) based on the score difference exceeding a threshold.

## Motivation

Users need binary classification beyond individual label scoring. Example: "Does this video show safe driving (positive labels) or dangerous driving (negative labels)?" Rather than interpreting 30+ individual label scores, the user gets a single verdict with supporting detail.

## Design

### API

**Aggregation value:** `"contrast"` — 4th option alongside `mean`, `max`, `temporal`.

**Form fields when `aggregation=contrast`:**

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| `positive_labels` | JSON array string | Yes | 1-50 labels |
| `negative_labels` | JSON array string | Yes | 1-50 labels |
| `threshold` | float | No | 0.0-1.0, policy default |
| `contrast_reduce` | str | No | `"mean"` (default), `"top_k_mean"`, `"max"`, `"quantile"` |

**Parsed request layer:** Raw FastAPI form fields are all optional at the boundary. A `ParsedLabelRequest` discriminated union enforces mutual exclusivity after parsing:

- `StandardLabels(labels)` — used by mean/max/temporal
- `ContrastLabels(positive_labels, negative_labels, threshold, contrast_reduce)` — used by contrast

Sending `labels` with `aggregation=contrast` or `positive_labels`/`negative_labels` with other modes returns 422 with a clear error message.

Per-label validation is identical to existing `labels` — non-empty, max 200 chars, unique, token limit check. Uniqueness is enforced across both groups combined (a label cannot appear in both positive and negative).

### Scoring Pipeline

**New function: `aggregate_contrast()` in `scoring.py`**

#### Step 1: Per-frame group scoring

Group scoring strategy differs by model to handle label-count imbalance correctly:

**SigLIP2 (sigmoid):** Labels are scored independently (0-1 each). Group score = mean of label scores in the group. Group size doesn't bias the result because each score is independent.

```
frame_pos_score[t] = mean(confidence[t, 0:P])
frame_neg_score[t] = mean(confidence[t, P:P+N])
```

**CLIP (softmax):** Scores are relative probabilities summing to 1 across ALL labels. Naive sum or mean is biased by group size (50 labels vs 1 label gives ~50/51 vs ~1/51 before any evidence). Instead, compute normalized group evidence in logit space:

```
pos_evidence[t] = logsumexp(logits[t, 0:P]) - log(P)
neg_evidence[t] = logsumexp(logits[t, P:P+N]) - log(N)
[frame_pos_score[t], frame_neg_score[t]] = softmax([pos_evidence[t], neg_evidence[t]])
```

This treats each group as a single "super-label" normalized by group size. Many labels describing a concept add evidence, but more labels alone don't create bias.

#### Step 2: Temporal reduction

Per-frame margins are reduced to a single video margin:

```
frame_margin[t] = frame_pos_score[t] - frame_neg_score[t]
video_margin = temporal_reduce(frame_margin, mode)
```

Reduction modes (all sign-symmetric — detect both positive and negative sparse events equally):

- `"mean"` (default) — average margin across all frames. Best for whole-video classification (mood, topic, setting).
- `"top_k_mean"` — select `k = max(1, ceil(num_frames * 0.10))` frames ranked by `abs(frame_margin)`, return mean of their original signed values. Better for sparse-event detection (brief dangerous moment in long video).
- `"max"` — frame with largest `abs(frame_margin)`, returning its signed value: `idx = argmax(abs(frame_margin)); video_margin = frame_margin[idx]`. Most sensitive to outliers.
- `"quantile"` — computes both 90th percentile (positive tail) and 10th percentile (negative tail), returns whichever has larger absolute value: `pos = quantile(0.90); neg = quantile(0.10); video_margin = pos if abs(pos) >= abs(neg) else neg`. Robust to outliers while catching sparse events in either direction.

#### Step 3: Verdict

```
difference = video_margin  # already reduced to scalar
if difference > threshold:  verdict = "positive"
elif difference < -threshold: verdict = "negative"
else: verdict = "uncertain"
```

#### Default threshold

With logit-space normalization for CLIP and mean for SigLIP2, both models produce margins on a comparable scale. Default threshold is policy-based per model:
- SigLIP2: 0.15
- CLIP: 0.10

Threshold metadata in response includes `threshold_source: "model_policy" | "user" | "calibrated_dataset"` and `calibration_status: "uncalibrated" | "calibrated"`. Model policy defaults are heuristic, not calibrated — the distinction matters for consumers making safety-critical decisions.

**Contrast policy:** Extends existing policy pattern in `temporal_policy.py`. Explicit method names avoid collision with temporal policy:
- `contrast_label_pooling()` — how to compute per-frame group scores
- `contrast_default_reduction()` — default reduction mode
- `contrast_default_threshold()` — starting threshold value (distinct from existing `default_threshold()` used by temporal)
- `score_semantics` — reuses existing canonical constants (`siglip2_pairwise_sigmoid`, `clip_relative_softmax`)

### Response Schema

```
ContrastLabelScore:
  label: str
  score: float              # individual label's mean confidence across frames

ContrastGroupResult:
  group: str                # "positive" or "negative"
  mean_group_score: float   # mean of per-frame group scores across ALL frames (always mean, regardless of reduction mode)
  labels: list[ContrastLabelScore]

ContrastResult:
  verdict: str              # "positive" | "negative" | "uncertain"
  difference: float         # video_margin from temporal reduction (NOT necessarily mean_group_score subtraction)
  threshold: float          # effective threshold used
  threshold_was_defaulted: bool
  threshold_source: str     # "model_policy" | "user" | "calibrated_dataset"
  calibration_status: str   # "uncalibrated" | "calibrated"
  contrast_reduce: str      # reduction mode used
  positive: ContrastGroupResult
  negative: ContrastGroupResult
  score_semantics: str      # reuses existing constants
  label_pooling: str        # "mean" (SigLIP2) or "logsumexp_normalized" (CLIP)
  dominant_label: str | None  # strongest label inside winning group (None if uncertain)
```

**Top-level `ClassifyResponse` compatibility:** The existing required fields `best_match` and `scores` are populated for backward compatibility:
- `best_match` — strongest individual label across all labels (both groups)
- `scores` — flattened list of all individual label scores
- `contrast` — optional field with the authoritative verdict and full breakdown
- `metadata.aggregation` = `"contrast"` signals clients to read `contrast.verdict` as the classification result

### UI

**Label input** — when "contrast" is selected from aggregation dropdown:

- Existing single label section hides
- Two side-by-side panels appear: "Positive Labels" (green accent) and "Negative Labels" (red accent)
- Each panel has:
  - Text input (comma-separated)
  - CSV upload button
  - Label chips with x to remove
  - Counter: `N / 50`
- Preset labels hidden in contrast mode

**Parameter sliders:**

- "Contrast Threshold" slider, range 0.00-1.00, step 0.01
- Default tag "(model default)" — hidden once touched (same as temporal)
- "Reduction" dropdown: mean (default), top_k_mean, max, quantile
- Gap tolerance and min duration sliders hidden

**Results display:**

- Verdict banner: large colored badge — green POSITIVE, red NEGATIVE, yellow UNCERTAIN
- Score summary: mean group scores, video margin (difference), threshold, and reduction mode
- Dominant label callout below verdict (strongest label in winning group)
- Grouped horizontal bar chart (Chart.js): positive labels as green bars, negative as red bars, sorted by score within each group

### Files Changed

| File | Change |
|------|--------|
| `app/main.py` | Parsed request layer, contrast routing, label validation, aggregation dispatch |
| `app/services/scoring.py` | New `aggregate_contrast()` with logit-space pooling and temporal reduction |
| `app/services/temporal_policy.py` | Contrast policy per model (pooling, reduction, threshold, semantics) |
| `app/schemas/response.py` | `ContrastLabelScore`, `ContrastGroupResult`, `ContrastResult`; `ClassifyResponse` compatibility |
| `app/static/index.html` | Dual label panels, threshold slider, reduction dropdown, verdict display, bar chart |
| `tests/` | Unit tests for scoring (including group-size imbalance), validation, response schema; integration tests |

### Constraints

- Max 50 labels per group (100 total labels possible)
- Max 300 frames extracted (existing limit)
- Labels must be unique across both groups
- Logit tensor required for CLIP contrast (already available in `ScoringContext`)
