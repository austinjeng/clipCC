# Contrast Scoring Mode

**Date:** 2026-05-20
**Status:** Approved (rev 2 — addresses review findings)

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

Reduction modes:
- `"mean"` (default) — average margin across all frames. Best for whole-video classification (mood, topic, setting).
- `"top_k_mean"` — mean of top-K% frames by absolute margin (K=10). Better for sparse-event detection (brief dangerous moment in long video).
- `"max"` — single strongest frame margin. Most sensitive to outliers.
- `"quantile"` — 90th percentile of frame margins. Robust to outliers while still catching sparse events.

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

Threshold metadata in response includes `threshold_source: "model_policy" | "user"` so consumers know whether calibration was applied.

**Contrast policy:** Extends existing policy pattern in `temporal_policy.py`. Each model's policy provides:
- `label_pooling` — how to compute per-frame group scores
- `temporal_reduction` — default reduction mode
- `default_threshold` — starting threshold value
- `score_semantics` — reuses existing canonical constants (`siglip2_pairwise_sigmoid`, `clip_relative_softmax`)

### Response Schema

```
ContrastLabelScore:
  label: str
  score: float              # individual label's mean confidence across frames

ContrastGroupResult:
  group: str                # "positive" or "negative"
  group_score: float        # normalized group score (see scoring pipeline)
  labels: list[ContrastLabelScore]

ContrastResult:
  verdict: str              # "positive" | "negative" | "uncertain"
  difference: float         # video_margin (pos - neg after reduction)
  threshold: float          # effective threshold used
  threshold_was_defaulted: bool
  threshold_source: str     # "model_policy" | "user"
  contrast_reduce: str      # reduction mode used
  positive: ContrastGroupResult
  negative: ContrastGroupResult
  score_semantics: str      # reuses existing constants
  label_pooling: str        # "mean" (SigLIP2) or "logsumexp_normalized" (CLIP)
  dominant_label: str       # strongest label inside winning group (null if uncertain)
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
- Score summary line: `pos_score - neg_score = difference` with threshold and reduction mode shown
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
