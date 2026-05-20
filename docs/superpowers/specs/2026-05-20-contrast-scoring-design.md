# Contrast Scoring Mode

**Date:** 2026-05-20
**Status:** Approved

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
| `threshold` | float | No | 0.0-1.0, model-specific default |

**Mutual exclusivity:** `labels` is used for mean/max/temporal. `positive_labels` + `negative_labels` used for contrast. Sending the wrong combination returns 422.

Per-label validation is identical to existing `labels` — non-empty, max 200 chars, unique, token limit check. Uniqueness is enforced across both groups combined (a label cannot appear in both positive and negative).

### Scoring Pipeline

**New function: `aggregate_contrast()` in `scoring.py`**

1. Receive combined confidence tensor `[num_frames, num_labels]` where labels `0..P-1` are positive and `P..P+N-1` are negative
2. Slice tensor into two groups by index
3. Per frame, compute group scores:
   - **SigLIP2 (sigmoid):** `mean(frame_scores[group_indices])` — each label is independent 0-1, mean is directly meaningful
   - **CLIP (softmax):** `sum(frame_scores[group_indices])` — softmax distributes probability across all labels, summing gives total probability mass per group (individual means would be ~1/N and differences would be noise)
4. Average each group's per-frame scores across all frames -> `pos_score`, `neg_score` (for SigLIP2: mean of per-frame means; for CLIP: mean of per-frame sums)
5. `difference = pos_score - neg_score`
6. Apply verdict:
   - `difference > threshold` -> `"positive"`
   - `difference < -threshold` -> `"negative"`
   - otherwise -> `"uncertain"`

**Default thresholds:**
- SigLIP2: 0.15
- CLIP: 0.10

**Contrast policy:** New `ContrastPolicy` class (or method on existing policy classes) in `temporal_policy.py` providing default threshold and group scoring method per model semantics.

### Response Schema

```
ContrastLabelScore:
  label: str
  score: float              # individual label's mean score across frames

ContrastGroupResult:
  group: str                # "positive" or "negative"
  group_score: float        # mean (SigLIP2) or sum (CLIP) across labels
  labels: list[ContrastLabelScore]

ContrastResult:
  verdict: str              # "positive" | "negative" | "uncertain"
  difference: float         # pos_score - neg_score
  threshold: float          # effective threshold used
  threshold_was_defaulted: bool
  positive: ContrastGroupResult
  negative: ContrastGroupResult
  score_semantics: str      # "sigmoid_independent" or "softmax_relative"
  group_scoring: str        # "mean" or "sum"
```

`ClassifyResponse` gains an optional `contrast: ContrastResult` field, populated only when `aggregation=contrast`. Follows the same pattern as the existing `temporal` field.

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

**Threshold slider:**

- Single "Contrast Threshold" slider, range 0.00-1.00, step 0.01
- Default tag "(model default)" — hidden once touched (same as temporal)
- Gap tolerance and min duration sliders hidden

**Results display:**

- Verdict banner: large colored badge — green POSITIVE, red NEGATIVE, yellow UNCERTAIN
- Score summary line: `pos_score - neg_score = difference` with threshold shown
- Grouped horizontal bar chart (Chart.js): positive labels as green bars, negative as red bars, sorted by score within each group
- Time-series line chart hidden (no per-frame data in contrast mode)

### Files Changed

| File | Change |
|------|--------|
| `app/main.py` | Contrast routing, form field parsing, label validation, aggregation dispatch |
| `app/services/scoring.py` | New `aggregate_contrast()` function |
| `app/services/temporal_policy.py` | Contrast policy (default threshold, group scoring method) |
| `app/schemas/response.py` | `ContrastLabelScore`, `ContrastGroupResult`, `ContrastResult` models; `ClassifyResponse` updated |
| `app/static/index.html` | Dual label panels, threshold slider, verdict display, bar chart |
| `tests/` | Unit tests for scoring, validation, response schema; integration tests for API |

### Constraints

- Max 50 labels per group (100 total labels possible)
- Max 300 frames extracted (existing limit)
- Labels must be unique across both groups
- No temporal segments in contrast mode — single verdict per video
