# Temporal Burst Detection for Video Classification

**Date:** 2026-05-07
**Branch:** SigLip2
**Status:** Approved

## Problem

Mean aggregation dilutes short burst events in long videos. A 5-second drowsiness episode in a 60-second video gets averaged down to near-baseline confidence. Max aggregation identifies only a single peak frame without capturing event duration or temporal structure.

## Solution

A new `"temporal"` aggregation mode that produces:
1. Per-frame confidence timeline for all labels
2. Detected segments (contiguous windows above a configurable threshold)
3. Built-in visualization (line chart + segment summary table)

## API Changes

### Request Parameters

New parameters accepted when `aggregation="temporal"`:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `aggregation` | string | `"mean"` | `"mean"`, `"max"`, `"temporal"` | Third aggregation mode |
| `threshold` | float | `0.5` | `0.0 - 1.0` | Confidence cutoff for segment detection |
| `gap_tolerance` | float | `2.0` | `0.0 - 10.0` | Max gap in seconds to bridge between segments |
| `min_duration` | float | `1.0` | `0.0 - 10.0` | Minimum segment length in seconds to report |

`threshold`, `gap_tolerance`, and `min_duration` are silently ignored when aggregation is not `"temporal"`.

### Response Schema

```python
class FrameScore(BaseModel):
    timestamp: float          # seconds into video
    frame_index: int
    scores: dict[str, float]  # {label: confidence}

class Segment(BaseModel):
    label: str
    start_time: float         # seconds
    end_time: float           # seconds
    duration: float           # seconds
    avg_confidence: float
    peak_confidence: float
    peak_timestamp: float     # timestamp of highest confidence frame

class TemporalResult(BaseModel):
    timeline: list[FrameScore]
    segments: list[Segment]

class ClassifyResponse(BaseModel):
    best_match: BestMatch
    scores: list[ScoreItem]
    metadata: ClassifyMetadata
    temporal: TemporalResult | None = None  # populated only in temporal mode
```

In temporal mode, `scores` uses mean aggregation so `best_match` remains meaningful.

## Temporal Analysis Engine

Located in `app/services/scoring.py`.

### Algorithm: `aggregate_temporal()`

```
Input:
  confidence: Tensor (n_frames, n_labels)
  frames: list[FrameSample]
  labels: list[str]
  threshold: float
  gap_tolerance: float (seconds)
  min_duration: float (seconds)

Step 1 — Build timeline:
  For each frame i:
    FrameScore(timestamp=frames[i].approx_timestamp_seconds,
               frame_index=i,
               scores={label: confidence[i][j] for j, label in labels})

Step 2 — Detect segments (per label):
  For each label j:
    a) Binary mask: confidence[:, j] >= threshold
    b) Find contiguous runs of True → raw segments (start_idx, end_idx)
    c) Bridge gaps: if gap between consecutive segments <= gap_tolerance seconds → merge
    d) Filter: discard segments with duration < min_duration
    e) For each surviving segment, compute:
       - start_time, end_time from frame timestamps
       - duration = end_time - start_time + (1/fps)
         (accounts for the temporal width of the last frame)
       - avg_confidence = mean of confidence[start:end+1, j]
       - peak_confidence = max of confidence[start:end+1, j]
       - peak_timestamp = timestamp of the max frame

Step 3 — Return:
  scores: aggregate_mean(confidence, raw_sim, labels, frames)
  temporal: TemporalResult(timeline, segments)
```

### Edge Cases

- Single frame video: timeline has one entry; segment possible only if min_duration is 0
- All frames below threshold: empty segments list, timeline still populated
- All frames above threshold: single segment spanning full video
- Gap bridging: recalculates avg/peak over merged range including bridged frames

### Performance

Pure tensor operations on already-computed per-frame scores. No additional model inference. Negligible added latency.

## UI Design

### Mode Selection

Radio button group: `Mean | Max | Temporal`. Selecting "Temporal" reveals the parameter panel with animation.

### Temporal Parameter Panel

Three sliders, each with a `(i)` tooltip icon:

| Slider | Range | Default | Step | Tooltip |
|--------|-------|---------|------|---------|
| Confidence Threshold | 0.0 - 1.0 | 0.5 | 0.01 | "Minimum confidence score for a frame to be considered part of an event. Frames scoring above this value are grouped into detected segments. Lower = more sensitive, higher = stricter." |
| Gap Tolerance | 0.0 - 10.0s | 2.0s | 0.1 | "Maximum gap (in seconds) between high-confidence frames that will still be merged into a single segment. Prevents brief dips from splitting one continuous event into many fragments." |
| Min Duration | 0.0 - 10.0s | 1.0s | 0.1 | "Shortest segment duration to report. Segments shorter than this are discarded as noise. Set to 0 to see all detected segments regardless of length." |

Tooltip interaction: hover/tap the icon → floating card appears. Dismissed on mouse-out or tap elsewhere.

### Results View (Temporal Mode)

**1. Line Chart**
- X-axis: time (seconds)
- Y-axis: confidence (0.0 - 1.0)
- One colored line per label with legend
- Horizontal dashed line at threshold value
- Detected segments as semi-transparent shaded rectangles (same color as line, ~20% opacity)
- Library: Chart.js via CDN

**2. Segment Summary Table**

| Label | Start | End | Duration | Avg Confidence | Peak Confidence | Peak Time |
|-------|-------|-----|----------|----------------|-----------------|-----------|
| sleepy | 25.0s | 30.0s | 5.0s | 0.82 | 0.91 | 27.0s |

- Sorted by start time
- Row colors match chart line colors
- Empty state: "No segments detected above threshold"

**3. Score Summary**
- Same as current mean mode output, shown below the table
- `best_match` badge still prominent

### Results View (Mean/Max Mode)

Unchanged from current behavior.

## File Changes

### Modified

| File | Change |
|------|--------|
| `app/main.py` | Accept temporal params, validate, route to `aggregate_temporal()` |
| `app/services/scoring.py` | Add `aggregate_temporal()`, update `build_response_scores()` routing |
| `app/schemas/response.py` | Add `FrameScore`, `Segment`, `TemporalResult` models, add `temporal` field |
| `app/config.py` | Default values and bounds for temporal params |
| `static/index.html` | Radio buttons, sliders with tooltips, Chart.js chart, segment table |

### New Dependencies

- Chart.js (CDN, no Python deps)

### Unchanged

- `app/models/siglip2_model.py`
- `app/services/video.py`
- `app/models/model_manager.py`
- `app/inference_runner.py`

### Tests

- `tests/test_scoring.py` — unit tests for segment detection, gap bridging, min duration, edge cases
- `tests/test_api.py` — endpoint test with `aggregation="temporal"`, response shape validation
- `tests/test_integration.py` — end-to-end temporal flow
