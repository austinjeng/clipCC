# Temporal Burst Detection for Video Classification

**Date:** 2026-05-07
**Branch:** SigLip2
**Status:** Approved (revised after review)

## Problem

Mean aggregation dilutes short burst events in long videos. A 5-second drowsiness episode in a 60-second video gets averaged down to near-baseline confidence. Max aggregation identifies only a single peak frame without capturing event duration or temporal structure.

## Solution

A new `"temporal"` aggregation mode that produces:
1. Per-frame confidence timeline for all labels
2. Detected segments (contiguous windows above a configurable threshold)
3. Built-in visualization (line chart + segment summary table)

Built on three core abstractions that prevent subtle bugs:
- **`TemporalScoringPolicy`** — model-aware threshold semantics
- **`FrameTimeline`** — centralized timing math
- **`AggregationResult`** — clean return type replacing brittle tuples

## Core Abstractions

### 1. `TemporalScoringPolicy`

**Problem (Finding #1):** `threshold=0.5` assumes confidence means the same thing across models. SigLip2 uses independent sigmoid scores (0.5 is a meaningful absolute boundary). CLIP uses relative softmax scores (0.5 means nothing absolute — it depends on the label set).

**Location:** `app/services/temporal_policy.py`

```python
class TemporalScoringPolicy(ABC):
    @abstractmethod
    def detection_scores(self, batch: ScoreBatch) -> torch.Tensor:
        """Return the tensor to threshold against. Shape: (n_frames, n_labels)."""
        ...

    @abstractmethod
    def default_threshold(self) -> float:
        """Model-appropriate default threshold."""
        ...

    @abstractmethod
    def threshold_mode(self) -> str:
        """'absolute' or 'relative'. Informs UI tooltip text."""
        ...

class SigLip2Policy(TemporalScoringPolicy):
    def detection_scores(self, batch):
        return batch.confidence  # sigmoid scores, directly thresholdable
    def default_threshold(self):
        return 0.5
    def threshold_mode(self):
        return "absolute"

class SoftmaxPolicy(TemporalScoringPolicy):
    """For CLIP or other softmax-based models."""
    def detection_scores(self, batch):
        return batch.confidence  # softmax scores
    def default_threshold(self):
        return 0.3  # lower default; softmax scores are label-set-relative
    def threshold_mode(self):
        return "relative"
```

**Registry:** keyed by `ScoreBatch.semantics`:
- `"siglip2_pairwise_sigmoid"` → `SigLip2Policy`
- `"clip_softmax"` → `SoftmaxPolicy` (future)

When `threshold` is not explicitly provided by the user, the policy's `default_threshold()` is used. When explicitly provided, the user's value always wins. The `threshold_mode` is returned in metadata and drives the tooltip wording in the UI.

### 2. `FrameTimeline`

**Problem (Finding #2):** `duration = end_time - start_time + (1/fps)` was underspecified — `aggregate_temporal()` didn't receive `fps`, and the formula conflicted with single-frame edge cases. Timestamps are approximate `i / fps`, not true PTS.

**Location:** `app/services/frame_timeline.py`

```python
@dataclass
class FrameInterval:
    index: int
    start: float    # seconds, inclusive
    end: float      # seconds, exclusive

class FrameTimeline:
    def __init__(self, frames: list[FrameSample], fps: float, video_duration: float):
        self.frames = frames
        self.fps = fps
        self.video_duration = video_duration
        self.frame_interval = 1.0 / fps
        self.intervals: list[FrameInterval] = self._build_intervals()

    def _build_intervals(self) -> list[FrameInterval]:
        intervals = []
        for i, frame in enumerate(self.frames):
            start = frame.approx_timestamp_seconds
            end = min(start + self.frame_interval, self.video_duration)
            intervals.append(FrameInterval(index=i, start=start, end=end))
        return intervals

    def gap_seconds(self, seg_a_end_idx: int, seg_b_start_idx: int) -> float:
        """Time gap between end of segment A and start of segment B."""
        return self.intervals[seg_b_start_idx].start - self.intervals[seg_a_end_idx].end

    def segment_duration(self, start_idx: int, end_idx: int) -> float:
        """Duration from first frame's start to last frame's end."""
        return self.intervals[end_idx].end - self.intervals[start_idx].start

    def timestamp(self, idx: int) -> float:
        return self.intervals[idx].start
```

All duration/gap math goes through `FrameTimeline`. No raw `1/fps` arithmetic scattered elsewhere.

### 3. `AggregationResult`

**Problem (Finding #5):** `build_response_scores()` returns `tuple[list[ScoreItem], BestMatch]`. Bolting `temporal` onto this makes the plumbing brittle.

**Location:** `app/services/scoring.py`

```python
@dataclass
class AggregationResult:
    scores: list[ScoreItem]
    best_match: BestMatch
    temporal: TemporalResult | None = None
```

**Aggregator routing** replaces the if/else chain with a registry:

```python
AGGREGATORS = {
    "mean": aggregate_mean,
    "max": aggregate_max,
    "temporal": aggregate_temporal,
}

def aggregate_frame_scores(
    batches: list[ScoreBatch],
    labels: list[str],
    frames: list[FrameSample],
    aggregation: str,
    temporal_options: TemporalOptions | None = None,
    timeline: FrameTimeline | None = None,
    policy: TemporalScoringPolicy | None = None,
) -> AggregationResult:
    all_confidence = torch.cat([b.confidence for b in batches], dim=0)
    all_raw_sim = torch.cat([b.raw_similarity for b in batches], dim=0)

    aggregator = AGGREGATORS[aggregation]
    return aggregator(all_confidence, all_raw_sim, labels, frames,
                      temporal_options=temporal_options,
                      timeline=timeline, policy=policy)
```

Mean and max aggregators ignore the temporal-only params and return `AggregationResult(scores, best_match, temporal=None)`.

## API Changes

### Request Parameters

New parameters accepted when `aggregation="temporal"`:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `aggregation` | string | `"mean"` | `"mean"`, `"max"`, `"temporal"` | Third aggregation mode |
| `threshold` | float | *(policy default)* | `0.0 - 1.0` | Confidence cutoff for segment detection |
| `gap_tolerance` | float | `2.0` | `0.0 - 10.0` | Max gap in seconds to bridge between segments |
| `min_duration` | float | `1.0` | `0.0 - 10.0` | Minimum segment length in seconds to report |

**Parameter validation (Finding #6):** All temporal parameters are parsed through a `TemporalOptions` value object:

```python
class TemporalOptions(BaseModel):
    threshold: float = Field(ge=0.0, le=1.0)
    gap_tolerance: float = Field(ge=0.0, le=10.0)
    min_duration: float = Field(ge=0.0, le=10.0)
```

- When `aggregation="temporal"`: `TemporalOptions` is constructed (using policy default for threshold if not provided). Ranges are always validated.
- When `aggregation` is `"mean"` or `"max"`: if any temporal params are supplied, the API returns `422 Unprocessable Entity` with message `"Parameters 'threshold', 'gap_tolerance', 'min_duration' are only valid with aggregation='temporal'"`. This catches client bugs early rather than silently hiding them.

### Response Schema

```python
class FrameScore(BaseModel):
    timestamp: float          # seconds into video
    frame_index: int
    scores: dict[str, float]  # {label: confidence}

class SegmentStats(BaseModel):
    active_avg: float         # mean of above-threshold frames only
    interval_avg: float       # mean of all frames in [start, end] including bridged gaps
    coverage_ratio: float     # fraction of segment frames that are above threshold

class Segment(BaseModel):
    label: str
    start_time: float         # seconds
    end_time: float           # seconds
    duration: float           # seconds
    stats: SegmentStats       # (Finding #3)
    peak_confidence: float
    peak_timestamp: float

class LabelSummary(BaseModel):
    label: str
    segment_count: int
    total_active_duration: float    # sum of all segment durations
    peak_confidence: float          # best peak across all segments
    duration_weighted_confidence: float  # weighted by segment duration

class TemporalResult(BaseModel):
    timeline: list[FrameScore]
    segments: list[Segment]
    label_summaries: list[LabelSummary]       # (Finding #4) temporal ranking
    best_segment: Segment | None              # (Finding #4) highest peak_confidence segment
    threshold_mode: str                       # (Finding #1) "absolute" or "relative"

class ClassifyResponse(BaseModel):
    best_match: BestMatch
    scores: list[ScoreItem]
    metadata: ClassifyMetadata
    temporal: TemporalResult | None = None
```

**Finding #3 — SegmentStats:** Merged segments distinguish `active_avg` (only above-threshold frames — event strength) from `interval_avg` (all frames including bridged gaps). `coverage_ratio` shows how much of the segment was genuinely active vs. bridged. A segment with `active_avg=0.82, interval_avg=0.68, coverage_ratio=0.75` clearly communicates "strong event with some gaps bridged."

**Finding #4 — Temporal ranking:** `best_match` still uses mean aggregation for backward compatibility. `temporal.best_segment` surfaces the strongest burst (what this feature exists to find). `temporal.label_summaries` provides per-label temporal metrics for richer analysis: how many segments, total active time, and duration-weighted confidence.

## Temporal Analysis Engine

Located in `app/services/scoring.py`.

### Algorithm: `aggregate_temporal()`

```
Input:
  confidence: Tensor (n_frames, n_labels)
  raw_sim: Tensor (n_frames, n_labels)
  frames: list[FrameSample]
  labels: list[str]
  temporal_options: TemporalOptions
  timeline: FrameTimeline          # (Finding #2) centralized timing
  policy: TemporalScoringPolicy    # (Finding #1) model-aware scoring

Step 1 — Get detection scores:
  detection = policy.detection_scores(batches)
  (For SigLip2, this is the sigmoid confidence tensor directly)

Step 2 — Build timeline:
  For each frame i:
    FrameScore(timestamp=timeline.timestamp(i),
               frame_index=i,
               scores={label: detection[i][j] for j, label in labels})

Step 3 — Detect segments (per label):
  For each label j:
    a) Binary mask: detection[:, j] >= temporal_options.threshold
    b) Find contiguous runs of True → raw segments (start_idx, end_idx)
    c) Bridge gaps: if timeline.gap_seconds(seg_a_end, seg_b_start) <= gap_tolerance → merge
    d) Filter: discard where timeline.segment_duration(start, end) < min_duration
    e) For each surviving segment, compute:
       - start_time = timeline.timestamp(start_idx)
       - end_time = timeline.intervals[end_idx].end
       - duration = timeline.segment_duration(start_idx, end_idx)
       - active frames = frames where detection >= threshold within [start, end]
       - stats.active_avg = mean of active frames' scores
       - stats.interval_avg = mean of ALL frames' scores in [start, end]
       - stats.coverage_ratio = len(active frames) / len(all frames in range)
       - peak_confidence = max of detection[start:end+1, j]
       - peak_timestamp = timeline.timestamp(argmax frame)

Step 4 — Build label summaries:
  For each label, aggregate across its segments:
    - segment_count
    - total_active_duration = sum of segment durations
    - peak_confidence = max peak across segments
    - duration_weighted_confidence = sum(seg.duration * seg.stats.active_avg) / total_active_duration

Step 5 — Identify best_segment:
  Segment with highest peak_confidence across all labels. None if no segments detected.

Step 6 — Return:
  scores: aggregate_mean(confidence, raw_sim, labels, frames)  # reuse existing
  best_match: from mean scores (backward compat)
  temporal: TemporalResult(timeline, segments, label_summaries, best_segment,
                           threshold_mode=policy.threshold_mode())
```

### Edge Cases

- Single frame video: timeline has one entry with duration = `1/fps` (clamped to video_duration). Segment possible if that duration >= min_duration.
- All frames below threshold: empty segments, empty label_summaries, best_segment=None, timeline still populated.
- All frames above threshold: single segment spanning full video, coverage_ratio=1.0, active_avg == interval_avg.
- Gap bridging: active_avg excludes bridged gap frames; interval_avg includes them. coverage_ratio reflects the difference.
- Final frame interval clamped to video_duration by FrameTimeline, preventing overshoot.

### Performance

Pure tensor operations on already-computed per-frame scores. No additional model inference. Negligible added latency.

## UI Design

### Mode Selection

Radio button group: `Mean | Max | Temporal`. Selecting "Temporal" reveals the parameter panel with animation.

### Temporal Parameter Panel

Three sliders, each with a `(i)` tooltip icon:

| Slider | Range | Default | Step | Tooltip |
|--------|-------|---------|------|---------|
| Confidence Threshold | 0.0 - 1.0 | *(from policy)* | 0.01 | Dynamic based on `threshold_mode`: **absolute** → "Minimum confidence for event detection. SigLip2 sigmoid scores are directly interpretable — 0.5 means the model is neutral. Higher = stricter." **relative** → "Minimum confidence for event detection. Softmax scores are relative to the label set — optimal threshold depends on how many labels you provide." |
| Gap Tolerance | 0.0 - 10.0s | 2.0s | 0.1 | "Maximum gap (in seconds) between high-confidence frames that will still be merged into a single segment. Prevents brief dips from splitting one continuous event into many fragments." |
| Min Duration | 0.0 - 10.0s | 1.0s | 0.1 | "Shortest segment duration to report. Segments shorter than this are discarded as noise. Set to 0 to see all detected segments regardless of length." |

The threshold slider's default value is set from the API response's `threshold_mode` / policy default, not hardcoded.

Tooltip interaction: hover/tap the icon → floating card appears. Dismissed on mouse-out or tap elsewhere.

### Results View (Temporal Mode)

**1. Line Chart**
- X-axis: time (seconds)
- Y-axis: confidence (0.0 - 1.0)
- One colored line per label with legend
- Horizontal dashed line at threshold value
- Detected segments as semi-transparent shaded rectangles (same color as line, ~20% opacity)
- Library: Chart.js vendored under `static/vendor/chart.min.js` (Finding #7)
- Fallback: if Chart.js fails to load, render a table-only view with per-frame scores

**2. Segment Summary Table**

| Label | Start | End | Duration | Active Avg | Coverage | Peak | Peak Time |
|-------|-------|-----|----------|------------|----------|------|-----------|
| sleepy | 25.0s | 30.0s | 5.0s | 0.82 | 92% | 0.91 | 27.0s |

- Sorted by start time
- Row colors match chart line colors
- "Active Avg" and "Coverage" columns from SegmentStats (Finding #3)
- Empty state: "No segments detected above threshold"

**3. Best Segment Badge**
- Prominent display of `temporal.best_segment` (Finding #4): "Strongest burst: sleepy at 25-30s (peak 0.91)"
- Falls back to standard `best_match` display if no segments detected

**4. Score Summary**
- Same as current mean mode output, shown below
- `best_match` label still shown for API consistency

### Results View (Mean/Max Mode)

Unchanged from current behavior.

## File Changes

### New Files

| File | Purpose |
|------|---------|
| `app/services/temporal_policy.py` | `TemporalScoringPolicy` ABC, `SigLip2Policy`, policy registry |
| `app/services/frame_timeline.py` | `FrameTimeline`, `FrameInterval`, centralized timing math |
| `static/vendor/chart.min.js` | Vendored Chart.js (Finding #7) |

### Modified

| File | Change |
|------|--------|
| `app/main.py` | Accept temporal params via `TemporalOptions`, validate (reject if non-temporal), build `FrameTimeline`, resolve policy from semantics, pass to aggregator |
| `app/services/scoring.py` | Add `AggregationResult` dataclass, `aggregate_temporal()`, aggregator registry. Existing `aggregate_mean`/`aggregate_max` return `AggregationResult` instead of tuple. Update all callers. |
| `app/schemas/response.py` | Add `FrameScore`, `SegmentStats`, `Segment`, `LabelSummary`, `TemporalResult`, `TemporalOptions` models. Add `temporal` field to `ClassifyResponse`. |
| `app/config.py` | Default values and bounds for temporal params |
| `static/index.html` | Radio buttons, sliders with dynamic tooltips, Chart.js chart with fallback, segment table with SegmentStats columns, best segment badge |

### Unchanged

- `app/models/siglip2_model.py`
- `app/models/base_model.py`
- `app/services/video.py`
- `app/models/model_manager.py`
- `app/inference_runner.py`

### Tests

- `tests/test_temporal_policy.py` — policy selection from semantics, default thresholds, detection score extraction
- `tests/test_frame_timeline.py` — interval construction, gap calculation, duration math, video_duration clamping, single-frame edge case
- `tests/test_scoring.py` — `AggregationResult` return type for all modes, `aggregate_temporal()` segment detection, gap bridging, min duration, SegmentStats (active_avg vs interval_avg), label summaries, best_segment selection
- `tests/test_api.py` — temporal endpoint response shape, 422 on temporal params with non-temporal mode, threshold default from policy
- `tests/test_integration.py` — end-to-end temporal flow
