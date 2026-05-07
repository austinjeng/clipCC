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

### 1. `ScoringContext`

**Problem (Finding #8 — Pass 2):** `aggregate_frame_scores()` concatenates only confidence/raw tensors, discarding `semantics` and `logits`. By the time `aggregate_temporal()` runs, the policy can't access a `ScoreBatch` — the batch structure is gone. Passing `confidence`, `raw_sim`, `logits`, `semantics`, `policy`, `timeline`, and `temporal_options` as parallel arguments is fragile.

**Location:** `app/services/scoring.py`

```python
@dataclass
class ScoringContext:
    """Base scoring context for mean/max aggregation."""
    confidence: torch.Tensor      # (n_frames, n_labels) — concatenated from all batches
    raw_similarity: torch.Tensor  # (n_frames, n_labels)
    logits: torch.Tensor          # (n_frames, n_labels)
    semantics: str                # from ScoreBatch.semantics (validated same across batches)
    labels: list[str]
    frames: list[FrameSample]

    @classmethod
    def from_batches(cls, batches: list[ScoreBatch], labels: list[str],
                     frames: list[FrameSample]) -> "ScoringContext":
        if not batches:
            raise ValueError("Cannot build ScoringContext from empty batch list")
        semantics_set = {b.semantics for b in batches}
        if len(semantics_set) != 1:
            raise ValueError(
                f"Mixed model semantics in single request: {semantics_set}. "
                "All batches must come from the same model."
            )
        return cls(
            confidence=torch.cat([b.confidence for b in batches], dim=0),
            raw_similarity=torch.cat([b.raw_similarity for b in batches], dim=0),
            logits=torch.cat([b.logits for b in batches], dim=0),
            semantics=semantics_set.pop(),
            labels=labels,
            frames=frames,
        )

@dataclass
class TemporalScoringContext(ScoringContext):
    """Extended context for temporal aggregation — guarantees timeline is present."""
    timeline: FrameTimeline  # non-optional

    @classmethod
    def from_base(cls, ctx: ScoringContext, timeline: FrameTimeline) -> "TemporalScoringContext":
        return cls(
            confidence=ctx.confidence,
            raw_similarity=ctx.raw_similarity,
            logits=ctx.logits,
            semantics=ctx.semantics,
            labels=ctx.labels,
            frames=ctx.frames,
            timeline=timeline,
        )
```

`aggregate_mean` and `aggregate_max` receive `ScoringContext`. `aggregate_temporal` receives `TemporalScoringContext` — type-level guarantee that timeline exists, fails fast at construction if absent. The policy reads `ctx.semantics` and operates on `ctx.confidence` or `ctx.logits` as appropriate.

### 2. `TemporalScoringPolicy`

**Problem (Finding #1):** `threshold=0.5` assumes confidence means the same thing across models. SigLip2 uses independent sigmoid scores (0.5 is a meaningful absolute boundary). CLIP uses relative softmax scores (0.5 means nothing absolute — it depends on the label set).

**Location:** `app/services/temporal_policy.py`

```python
class TemporalScoringPolicy(ABC):
    @abstractmethod
    def detection_scores(self, ctx: ScoringContext) -> torch.Tensor:
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
    def detection_scores(self, ctx):
        return ctx.confidence  # sigmoid scores, directly thresholdable
    def default_threshold(self):
        return 0.5
    def threshold_mode(self):
        return "absolute"

class SoftmaxPolicy(TemporalScoringPolicy):
    """For CLIP or other softmax-based models."""
    def detection_scores(self, ctx):
        return ctx.confidence  # softmax scores
    def default_threshold(self):
        return 0.3  # lower default; softmax scores are label-set-relative
    def threshold_mode(self):
        return "relative"
```

**Semantics constants** (Finding #10 — avoids registry mismatch):

```python
class ScoreSemantics:
    SIGLIP2_SIGMOID = "siglip2_pairwise_sigmoid"
    CLIP_RELATIVE_SOFTMAX = "clip_relative_softmax"
```

**Registry:** keyed by semantics constants:
- `ScoreSemantics.SIGLIP2_SIGMOID` → `SigLip2Policy`
- `ScoreSemantics.CLIP_RELATIVE_SOFTMAX` → `SoftmaxPolicy`

Lookup raises `ValueError` with a clear message if an unregistered semantics string is encountered.

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

### 4. `AggregationResult`

**Problem (Finding #5):** `build_response_scores()` returns `tuple[list[ScoreItem], BestMatch]`. Bolting `temporal` onto this makes the plumbing brittle.

**Location:** `app/services/scoring.py`

```python
@dataclass
class AggregationResult:
    scores: list[ScoreItem]
    best_match: BestMatch
    temporal: TemporalResult | None = None
```

**Aggregator routing** replaces the if/else chain with a registry. All aggregators receive `ScoringContext` + optional `ResolvedTemporalOptions`:

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
    temporal_options: ResolvedTemporalOptions | None = None,
    timeline: FrameTimeline | None = None,
    policy: TemporalScoringPolicy | None = None,
) -> AggregationResult:
    ctx = ScoringContext.from_batches(batches, labels, frames, timeline)

    aggregator = AGGREGATORS[aggregation]
    return aggregator(ctx, temporal_options=temporal_options, policy=policy)
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

**Parameter validation (Finding #6, #9):** Split raw request parsing from resolved options:

```python
class RawTemporalParams(BaseModel):
    """What the client sent — threshold may be omitted to use policy default."""
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    gap_tolerance: float | None = Field(None, ge=0.0, le=10.0)
    min_duration: float | None = Field(None, ge=0.0, le=10.0)

class ResolvedTemporalOptions(BaseModel):
    """Fully resolved — all fields have concrete values."""
    threshold: float
    gap_tolerance: float
    min_duration: float
    threshold_was_defaulted: bool   # True if user didn't supply threshold

    @classmethod
    def resolve(cls, raw: RawTemporalParams, policy: TemporalScoringPolicy) -> "ResolvedTemporalOptions":
        return cls(
            threshold=raw.threshold if raw.threshold is not None else policy.default_threshold(),
            gap_tolerance=raw.gap_tolerance if raw.gap_tolerance is not None else 2.0,
            min_duration=raw.min_duration if raw.min_duration is not None else 1.0,
            threshold_was_defaulted=(raw.threshold is None),
        )
```

- When `aggregation="temporal"`: `RawTemporalParams` is parsed (ranges validated on any supplied value), then resolved against the policy into `ResolvedTemporalOptions`.
- When `aggregation` is `"mean"` or `"max"`: if any temporal params are supplied, the API returns `422 Unprocessable Entity` with message `"Parameters 'threshold', 'gap_tolerance', 'min_duration' are only valid with aggregation='temporal'"`. This catches client bugs early rather than silently hiding them.

### Response Schema

```python
class FrameScore(BaseModel):
    timestamp: float          # seconds into video
    frame_index: int
    scores: dict[str, float]  # {label: confidence}

class SegmentStats(BaseModel):
    active_avg: float         # mean of above-threshold frames only (event strength)
    interval_avg: float       # mean of all frames in [start, end] including bridged gaps
    coverage_ratio: float     # fraction of segment frames that are above threshold
    active_duration: float    # (Finding #12) sum of active frame intervals only (excludes gaps)

class Segment(BaseModel):
    label: str
    start_time: float         # seconds
    end_time: float           # seconds
    duration: float           # seconds (full segment span including bridged gaps)
    stats: SegmentStats       # (Finding #3)
    peak_confidence: float
    peak_timestamp: float

class LabelSummary(BaseModel):
    label: str
    segment_count: int
    total_active_duration: float    # (Finding #12) sum of active frame intervals across segments
    total_segment_duration: float   # sum of full segment durations (for display)
    peak_confidence: float          # best peak across all segments
    duration_weighted_confidence: float  # weighted by active_duration per segment

class TemporalResult(BaseModel):
    timeline: list[FrameScore]
    segments: list[Segment]
    label_summaries: list[LabelSummary]       # (Finding #4) temporal ranking
    best_segment: Segment | None              # (Finding #4) highest peak_confidence segment
    threshold_mode: str                       # (Finding #1) "absolute" or "relative"
    effective_threshold: float                # (Finding #11) the resolved threshold value used
    threshold_was_defaulted: bool             # (Finding #11) whether user supplied it or policy did

class ClassifyResponse(BaseModel):
    best_match: BestMatch
    scores: list[ScoreItem]
    metadata: ClassifyMetadata
    temporal: TemporalResult | None = None
```

**Finding #3 — SegmentStats:** Merged segments distinguish `active_avg` (only above-threshold frames — event strength) from `interval_avg` (all frames including bridged gaps). `coverage_ratio` shows frame-count fraction. `active_duration` gives the sum of active frame intervals in seconds (excludes bridged gap time). A segment with `active_avg=0.82, interval_avg=0.68, coverage_ratio=0.75, active_duration=3.75s` on a 5s segment clearly communicates "strong event with 1.25s of tolerated gaps."

**Finding #4 — Temporal ranking:** `best_match` still uses mean aggregation for backward compatibility. `temporal.best_segment` surfaces the strongest burst (what this feature exists to find). `temporal.label_summaries` provides per-label temporal metrics: `total_active_duration` sums only active frame intervals (not full segment spans including gaps), while `total_segment_duration` gives the full display-friendly span. `duration_weighted_confidence` weights by active time, not total time — so bridged gaps don't dilute the event strength metric.

**Finding #11 — Effective threshold in response:** `effective_threshold` tells the UI exactly what threshold line to draw on the chart, and `threshold_was_defaulted` lets the UI indicate "using model default" vs. "user-specified" — important for transparency when the policy picks a non-obvious value like 0.3 for softmax models.

## Temporal Analysis Engine

Located in `app/services/scoring.py`.

### Algorithm: `aggregate_temporal()`

```
Input:
  ctx: ScoringContext              # (Finding #8) single data object with all tensors + metadata
  temporal_options: ResolvedTemporalOptions  # (Finding #9) fully resolved, no None fields
  policy: TemporalScoringPolicy    # (Finding #1) resolved from ctx.semantics

Step 1 — Get detection scores:
  detection = policy.detection_scores(ctx)
  (For SigLip2, reads ctx.confidence — sigmoid scores, directly thresholdable)
  (Policy has full access to ctx.logits if a future policy needs raw logits)

Step 2 — Build timeline:
  For each frame i:
    FrameScore(timestamp=ctx.timeline.timestamp(i),
               frame_index=i,
               scores={label: detection[i][j] for j, label in ctx.labels})

Step 3 — Detect segments (per label):
  For each label j:
    a) Binary mask: detection[:, j] >= temporal_options.threshold
    b) Find contiguous runs of True → raw segments (start_idx, end_idx)
    c) Bridge gaps: if ctx.timeline.gap_seconds(seg_a_end, seg_b_start) <= gap_tolerance → merge
    d) Filter: discard where ctx.timeline.segment_duration(start, end) < min_duration
    e) For each surviving segment, compute:
       - start_time = ctx.timeline.timestamp(start_idx)
       - end_time = ctx.timeline.intervals[end_idx].end
       - duration = ctx.timeline.segment_duration(start_idx, end_idx)
       - active_indices = indices where detection >= threshold within [start, end]
       - stats.active_avg = mean of detection[active_indices, j]
       - stats.interval_avg = mean of detection[start:end+1, j]
       - stats.coverage_ratio = len(active_indices) / (end_idx - start_idx + 1)
       - stats.active_duration = sum of ctx.timeline.intervals[i].end - .start for active_indices
       - peak_confidence = max of detection[start:end+1, j]
       - peak_timestamp = ctx.timeline.timestamp(argmax frame)

Step 4 — Build label summaries:
  For each label, aggregate across its segments:
    - segment_count
    - total_active_duration = sum of seg.stats.active_duration across segments
    - total_segment_duration = sum of seg.duration across segments
    - peak_confidence = max peak across segments
    - duration_weighted_confidence = sum(seg.stats.active_duration * seg.stats.active_avg) / total_active_duration

Step 5 — Identify best_segment:
  Segment with highest peak_confidence across all labels. None if no segments detected.

Step 6 — Return:
  scores: aggregate_mean(ctx)  # reuse existing, operates on ctx.confidence/raw_similarity
  best_match: from mean scores (backward compat)
  temporal: TemporalResult(
      timeline, segments, label_summaries, best_segment,
      threshold_mode=policy.threshold_mode(),
      effective_threshold=temporal_options.threshold,
      threshold_was_defaulted=temporal_options.threshold_was_defaulted,
  )
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

### Temporal Defaults from Model Metadata

**Problem (Finding #13):** The threshold slider default should come from the model's policy, but the UI needs this *before* submitting a classify request. If the UI always sends its slider value, the server can never distinguish "user chose 0.5" from "UI sent the default 0.5."

**Solution:** Expose `temporal_defaults` on the existing `/api/v1/models/active` endpoint:

```json
{
  "model_id": "siglip2-base-patch16-256",
  "display_name": "SigLip2 Base 256",
  "model_type": "siglip2",
  ...
  "temporal_defaults": {
    "threshold": 0.5,
    "threshold_mode": "absolute",
    "gap_tolerance": 2.0,
    "min_duration": 1.0
  }
}
```

**UI behavior:**
1. On model load/change, UI fetches `/api/v1/models/active` and reads `temporal_defaults`.
2. Sliders initialize to these values. A `dirty` flag per slider tracks whether the user has touched it.
3. On submit: only include `threshold` in the request if the user explicitly moved the slider. If untouched, omit it — the server uses the policy default and `threshold_was_defaulted=true`.
4. After response: the chart draws the threshold line from `temporal.effective_threshold` (authoritative source).

### Temporal Parameter Panel

Three sliders, each with a `(i)` tooltip icon:

| Slider | Range | Default | Step | Tooltip |
|--------|-------|---------|------|---------|
| Confidence Threshold | 0.0 - 1.0 | *(from `temporal_defaults.threshold`)* | 0.01 | Dynamic based on `threshold_mode`: **absolute** → "Minimum confidence for event detection. SigLip2 sigmoid scores are directly interpretable — 0.5 means the model is neutral. Higher = stricter." **relative** → "Minimum confidence for event detection. Softmax scores are relative to the label set — optimal threshold depends on how many labels you provide." |
| Gap Tolerance | 0.0 - 10.0s | 2.0s | 0.1 | "Maximum gap (in seconds) between high-confidence frames that will still be merged into a single segment. Prevents brief dips from splitting one continuous event into many fragments." |
| Min Duration | 0.0 - 10.0s | 1.0s | 0.1 | "Shortest segment duration to report. Segments shorter than this are discarded as noise. Set to 0 to see all detected segments regardless of length." |

Threshold slider shows a subtle "(model default)" label when untouched; label disappears once the user moves it.

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
| `app/services/temporal_policy.py` | `TemporalScoringPolicy` ABC, `SigLip2Policy`, `SoftmaxPolicy`, `ScoreSemantics` constants, policy registry |
| `app/services/frame_timeline.py` | `FrameTimeline`, `FrameInterval`, centralized timing math |
| `static/vendor/chart.min.js` | Vendored Chart.js (Finding #7) |

### Modified

| File | Change |
|------|--------|
| `app/main.py` | Parse `RawTemporalParams`, validate (reject if non-temporal), build `FrameTimeline`, resolve policy from semantics, resolve `ResolvedTemporalOptions`, construct `TemporalScoringContext`, pass to aggregator. Extend `/api/v1/models/active` to include `temporal_defaults`. |
| `app/services/scoring.py` | Add `ScoringContext`, `TemporalScoringContext`, `AggregationResult`, `aggregate_temporal()`, aggregator registry. Existing `aggregate_mean`/`aggregate_max` accept `ScoringContext`, return `AggregationResult`. Update all callers. |
| `app/schemas/response.py` | Add `FrameScore`, `SegmentStats`, `Segment`, `LabelSummary`, `TemporalResult`, `RawTemporalParams`, `ResolvedTemporalOptions` models. Add `temporal` field to `ClassifyResponse`. |
| `app/config.py` | Default values and bounds for temporal params |
| `app/models/siglip2_model.py` | Import and use `ScoreSemantics.SIGLIP2_SIGMOID` constant instead of string literal |
| `app/models/clip_model.py` | Import and use `ScoreSemantics.CLIP_RELATIVE_SOFTMAX` constant instead of string literal |
| `app/models/model_manager.py` | Expose `temporal_defaults` via policy lookup when returning active model metadata |
| `static/index.html` | Radio buttons, sliders with dirty-tracking (omit threshold if untouched), fetch `temporal_defaults` on model change, Chart.js chart with table-only fallback, segment table with SegmentStats columns, best segment badge |

### Unchanged

- `app/models/base_model.py`
- `app/services/video.py`
- `app/inference_runner.py`

### Tests

- `tests/test_temporal_policy.py` — policy selection from semantics constants, default thresholds, detection score extraction from `ScoringContext`, unknown semantics raises `ValueError`
- `tests/test_frame_timeline.py` — interval construction, gap calculation, duration math, video_duration clamping, single-frame edge case, active_duration computation
- `tests/test_scoring.py` — `ScoringContext.from_batches()` (empty batches raises `ValueError`, mixed semantics raises `ValueError`, single batch works, multi-batch concatenation correct). `TemporalScoringContext.from_base()` construction. `AggregationResult` return type for all modes. `aggregate_temporal()` segment detection, gap bridging, min duration, SegmentStats (active_avg vs interval_avg vs active_duration), label summaries (total_active_duration uses active frame intervals not segment spans), best_segment selection, `duration_weighted_confidence` weights by active_duration.
- `tests/test_api.py` — temporal endpoint response shape, 422 on temporal params with non-temporal mode, `RawTemporalParams` → `ResolvedTemporalOptions` resolution, threshold default from policy when omitted, `effective_threshold` and `threshold_was_defaulted` in response, `/api/v1/models/active` returns `temporal_defaults`
- `tests/test_integration.py` — end-to-end temporal flow with threshold omitted (uses default) and with threshold specified (user override)
