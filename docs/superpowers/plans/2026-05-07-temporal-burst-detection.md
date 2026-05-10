# Temporal Burst Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `"temporal"` aggregation mode that detects short burst events in long videos via per-frame thresholding, gap bridging, and minimum duration filtering, with built-in chart visualization.

**Architecture:** Five core abstractions (`ScoringContext`/`TemporalScoringContext`, `TemporalScoringPolicy`, `FrameTimeline`, `AggregationResult`) are built bottom-up with tests before the temporal engine itself. Existing `aggregate_mean`/`aggregate_max` are refactored to the new `ScoringContext -> AggregationResult` pattern first, so temporal mode slots in cleanly. The UI adds a Chart.js line chart with segment overlays.

**Tech Stack:** Python 3.12, FastAPI, PyTorch, Pydantic v2, Chart.js 4.x (vendored)

**Spec:** `docs/superpowers/specs/2026-05-07-temporal-burst-detection-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `app/services/temporal_policy.py` | Create | `ScoreSemantics` constants, `TemporalScoringPolicy` ABC, `SigLip2Policy`, `SoftmaxPolicy`, policy registry |
| `app/services/frame_timeline.py` | Create | `FrameInterval`, `FrameTimeline` -- centralized timing math |
| `app/schemas/response.py` | Modify | Add `FrameScore`, `SegmentStats`, `Segment`, `LabelSummary`, `TemporalResult`, `RawTemporalParams`, `ResolvedTemporalOptions`. Add `temporal` field to `ClassifyResponse`. |
| `app/services/scoring.py` | Modify | Add `ScoringContext`, `TemporalScoringContext`, `AggregationResult`. Refactor `aggregate_mean`/`aggregate_max` to accept `ScoringContext` and return `AggregationResult`. Add `aggregate_temporal()`. Update `aggregate_frame_scores()` routing. |
| `app/models/siglip2_model.py` | Modify (line 65) | Use `ScoreSemantics.SIGLIP2_SIGMOID` constant |
| `app/models/clip_model.py` | Modify (line 49) | Use `ScoreSemantics.CLIP_RELATIVE_SOFTMAX` constant |
| `app/errors/handlers.py` | Modify | Add `InvalidTemporalParamsError` |
| `app/main.py` | Modify | Temporal params, validation, `FrameTimeline` construction, policy resolution, `TemporalScoringContext` routing, `temporal_defaults` on `/api/v1/models/active` |
| `app/static/vendor/chart.min.js` | Create | Vendored Chart.js 4.x |
| `app/static/index.html` | Modify | Radio buttons, temporal sliders with tooltips, Chart.js chart, segment table, best segment badge |
| `tests/test_temporal_policy.py` | Create | Policy registry, defaults, detection scores |
| `tests/test_frame_timeline.py` | Create | Intervals, gaps, durations, edge cases |
| `tests/test_scoring.py` | Modify | ScoringContext, AggregationResult, aggregate_temporal |
| `tests/test_api.py` | Modify | Temporal endpoint, 422 validation, temporal_defaults |

---

### Task 1: ScoreSemantics + TemporalScoringPolicy

**Files:**
- Create: `app/services/temporal_policy.py`
- Create: `tests/test_temporal_policy.py`

- [ ] **Step 1: Write tests for policy registry and behavior**

```python
# tests/test_temporal_policy.py
import torch
import pytest
from dataclasses import dataclass
from app.services.temporal_policy import (
    ScoreSemantics,
    TemporalScoringPolicy,
    SigLip2Policy,
    SoftmaxPolicy,
    get_policy,
)


def test_score_semantics_constants():
    assert ScoreSemantics.SIGLIP2_SIGMOID == "siglip2_pairwise_sigmoid"
    assert ScoreSemantics.CLIP_RELATIVE_SOFTMAX == "clip_relative_softmax"


def test_siglip2_policy_defaults():
    policy = SigLip2Policy()
    assert policy.default_threshold() == 0.5
    assert policy.threshold_mode() == "absolute"


def test_softmax_policy_defaults():
    policy = SoftmaxPolicy()
    assert policy.default_threshold() == 0.3
    assert policy.threshold_mode() == "relative"


def test_siglip2_detection_scores_returns_confidence():
    policy = SigLip2Policy()

    @dataclass
    class FakeCtx:
        confidence: torch.Tensor
        logits: torch.Tensor

    ctx = FakeCtx(
        confidence=torch.tensor([[0.8, 0.2], [0.3, 0.9]]),
        logits=torch.tensor([[1.5, -1.0], [-0.5, 2.0]]),
    )
    result = policy.detection_scores(ctx)
    assert torch.equal(result, ctx.confidence)


def test_softmax_detection_scores_returns_confidence():
    policy = SoftmaxPolicy()

    @dataclass
    class FakeCtx:
        confidence: torch.Tensor
        logits: torch.Tensor

    ctx = FakeCtx(
        confidence=torch.tensor([[0.7, 0.3]]),
        logits=torch.tensor([[1.0, -1.0]]),
    )
    result = policy.detection_scores(ctx)
    assert torch.equal(result, ctx.confidence)


def test_get_policy_siglip2():
    policy = get_policy(ScoreSemantics.SIGLIP2_SIGMOID)
    assert isinstance(policy, SigLip2Policy)


def test_get_policy_clip():
    policy = get_policy(ScoreSemantics.CLIP_RELATIVE_SOFTMAX)
    assert isinstance(policy, SoftmaxPolicy)


def test_get_policy_unknown_raises():
    with pytest.raises(ValueError, match="No temporal scoring policy"):
        get_policy("unknown_semantics")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_temporal_policy.py -v`
Expected: `ModuleNotFoundError: No module named 'app.services.temporal_policy'`

- [ ] **Step 3: Implement temporal_policy.py**

```python
# app/services/temporal_policy.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class ScoreSemantics:
    SIGLIP2_SIGMOID = "siglip2_pairwise_sigmoid"
    CLIP_RELATIVE_SOFTMAX = "clip_relative_softmax"


class TemporalScoringPolicy(ABC):
    @abstractmethod
    def detection_scores(self, ctx: Any) -> torch.Tensor:
        ...

    @abstractmethod
    def default_threshold(self) -> float:
        ...

    @abstractmethod
    def threshold_mode(self) -> str:
        ...


class SigLip2Policy(TemporalScoringPolicy):
    def detection_scores(self, ctx: Any) -> torch.Tensor:
        return ctx.confidence

    def default_threshold(self) -> float:
        return 0.5

    def threshold_mode(self) -> str:
        return "absolute"


class SoftmaxPolicy(TemporalScoringPolicy):
    def detection_scores(self, ctx: Any) -> torch.Tensor:
        return ctx.confidence

    def default_threshold(self) -> float:
        return 0.3

    def threshold_mode(self) -> str:
        return "relative"


_POLICY_REGISTRY: dict[str, type[TemporalScoringPolicy]] = {
    ScoreSemantics.SIGLIP2_SIGMOID: SigLip2Policy,
    ScoreSemantics.CLIP_RELATIVE_SOFTMAX: SoftmaxPolicy,
}


def get_policy(semantics: str) -> TemporalScoringPolicy:
    cls = _POLICY_REGISTRY.get(semantics)
    if cls is None:
        raise ValueError(
            f"No temporal scoring policy registered for semantics '{semantics}'. "
            f"Known: {list(_POLICY_REGISTRY.keys())}"
        )
    return cls()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_temporal_policy.py -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/temporal_policy.py tests/test_temporal_policy.py
git commit -m "feat: add ScoreSemantics constants and TemporalScoringPolicy"
```

---

### Task 2: FrameTimeline

**Files:**
- Create: `app/services/frame_timeline.py`
- Create: `tests/test_frame_timeline.py`

- [ ] **Step 1: Write tests for FrameTimeline**

```python
# tests/test_frame_timeline.py
import pytest
from pathlib import Path
from app.services.frame_timeline import FrameTimeline, FrameInterval
from app.services.video import FrameSample


def make_frames(count: int, fps: float = 1.0) -> list[FrameSample]:
    return [
        FrameSample(
            path=Path(f"/tmp/f{i}.jpg"),
            sample_index=i,
            approx_timestamp_seconds=i / fps,
        )
        for i in range(count)
    ]


def test_intervals_basic():
    frames = make_frames(3, fps=1.0)
    tl = FrameTimeline(frames, fps=1.0, video_duration=3.0)
    assert len(tl.intervals) == 3
    assert tl.intervals[0] == FrameInterval(index=0, start=0.0, end=1.0)
    assert tl.intervals[1] == FrameInterval(index=1, start=1.0, end=2.0)
    assert tl.intervals[2] == FrameInterval(index=2, start=2.0, end=3.0)


def test_final_frame_clamped_to_duration():
    frames = make_frames(3, fps=1.0)
    tl = FrameTimeline(frames, fps=1.0, video_duration=2.5)
    assert tl.intervals[2].end == 2.5


def test_gap_seconds_adjacent():
    frames = make_frames(5, fps=1.0)
    tl = FrameTimeline(frames, fps=1.0, video_duration=5.0)
    assert tl.gap_seconds(0, 1) == 0.0


def test_gap_seconds_with_skip():
    frames = make_frames(5, fps=1.0)
    tl = FrameTimeline(frames, fps=1.0, video_duration=5.0)
    assert tl.gap_seconds(1, 3) == 1.0


def test_segment_duration():
    frames = make_frames(5, fps=1.0)
    tl = FrameTimeline(frames, fps=1.0, video_duration=5.0)
    assert tl.segment_duration(1, 3) == 3.0


def test_segment_duration_single_frame():
    frames = make_frames(5, fps=1.0)
    tl = FrameTimeline(frames, fps=1.0, video_duration=5.0)
    assert tl.segment_duration(2, 2) == 1.0


def test_timestamp():
    frames = make_frames(3, fps=2.0)
    tl = FrameTimeline(frames, fps=2.0, video_duration=1.5)
    assert tl.timestamp(0) == 0.0
    assert tl.timestamp(1) == 0.5
    assert tl.timestamp(2) == 1.0


def test_high_fps():
    frames = make_frames(10, fps=5.0)
    tl = FrameTimeline(frames, fps=5.0, video_duration=2.0)
    assert tl.frame_interval == 0.2
    assert abs(tl.intervals[0].end - 0.2) < 1e-9
    assert abs(tl.segment_duration(0, 4) - 1.0) < 1e-9


def test_single_frame():
    frames = make_frames(1, fps=1.0)
    tl = FrameTimeline(frames, fps=1.0, video_duration=0.5)
    assert len(tl.intervals) == 1
    assert tl.intervals[0].start == 0.0
    assert tl.intervals[0].end == 0.5
    assert tl.segment_duration(0, 0) == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_frame_timeline.py -v`
Expected: `ModuleNotFoundError: No module named 'app.services.frame_timeline'`

- [ ] **Step 3: Implement frame_timeline.py**

```python
# app/services/frame_timeline.py
from __future__ import annotations

from dataclasses import dataclass

from app.services.video import FrameSample


@dataclass(frozen=True, eq=True)
class FrameInterval:
    index: int
    start: float
    end: float


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
        return self.intervals[seg_b_start_idx].start - self.intervals[seg_a_end_idx].end

    def segment_duration(self, start_idx: int, end_idx: int) -> float:
        return self.intervals[end_idx].end - self.intervals[start_idx].start

    def timestamp(self, idx: int) -> float:
        return self.intervals[idx].start
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_frame_timeline.py -v`
Expected: all 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/frame_timeline.py tests/test_frame_timeline.py
git commit -m "feat: add FrameTimeline for centralized temporal math"
```

---

### Task 3: Response Schemas

**Files:**
- Modify: `app/schemas/response.py`

- [ ] **Step 1: Add all new Pydantic models to response.py**

Add this import at the top of `app/schemas/response.py` (after existing imports):
```python
from pydantic import Field
```

Add the following classes **after** the existing `ClassifyMetadata` class and **before** the existing `ClassifyResponse` class:

```python
class RawTemporalParams(BaseModel):
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    gap_tolerance: float | None = Field(None, ge=0.0, le=10.0)
    min_duration: float | None = Field(None, ge=0.0, le=10.0)

    def has_any(self) -> bool:
        return any(v is not None for v in (self.threshold, self.gap_tolerance, self.min_duration))


class ResolvedTemporalOptions(BaseModel):
    threshold: float
    gap_tolerance: float
    min_duration: float
    threshold_was_defaulted: bool

    @classmethod
    def resolve(cls, raw: RawTemporalParams, default_threshold: float) -> "ResolvedTemporalOptions":
        return cls(
            threshold=raw.threshold if raw.threshold is not None else default_threshold,
            gap_tolerance=raw.gap_tolerance if raw.gap_tolerance is not None else 2.0,
            min_duration=raw.min_duration if raw.min_duration is not None else 1.0,
            threshold_was_defaulted=(raw.threshold is None),
        )


class FrameScore(BaseModel):
    timestamp: float
    frame_index: int
    scores: dict[str, float]


class SegmentStats(BaseModel):
    active_avg: float
    interval_avg: float
    coverage_ratio: float
    active_duration: float


class Segment(BaseModel):
    label: str
    start_time: float
    end_time: float
    duration: float
    stats: SegmentStats
    peak_confidence: float
    peak_timestamp: float


class LabelSummary(BaseModel):
    label: str
    segment_count: int
    total_active_duration: float
    total_segment_duration: float
    peak_confidence: float
    duration_weighted_confidence: float


class TemporalResult(BaseModel):
    timeline: list[FrameScore]
    segments: list[Segment]
    label_summaries: list[LabelSummary]
    best_segment: Segment | None
    threshold_mode: str
    effective_threshold: float
    threshold_was_defaulted: bool
```

Then modify the existing `ClassifyResponse` to add the `temporal` field:

```python
class ClassifyResponse(BaseModel):
    best_match: BestMatch
    scores: list[ScoreItem]
    metadata: ClassifyMetadata
    temporal: TemporalResult | None = None
```

- [ ] **Step 2: Run existing tests to verify nothing broke**

Run: `python -m pytest tests/test_scoring.py tests/test_api.py -v`
Expected: all existing tests PASS

- [ ] **Step 3: Commit**

```bash
git add app/schemas/response.py
git commit -m "feat: add temporal response schemas"
```

---

### Task 4: ScoringContext + AggregationResult + Refactor Existing Aggregators

**Files:**
- Modify: `app/services/scoring.py`
- Modify: `tests/test_scoring.py`
- Modify: `app/main.py`

This task refactors the existing scoring module. The key change: `aggregate_mean`/`aggregate_max` now accept `ScoringContext` and return `AggregationResult`. `aggregate_frame_scores()` builds the context and routes.

- [ ] **Step 1: Write tests for ScoringContext and AggregationResult**

Add these imports and tests to `tests/test_scoring.py`. Add the new imports at the top alongside existing ones:

```python
from app.services.scoring import ScoringContext, AggregationResult
from app.schemas.response import BestMatch
```

Add these new test functions after the existing `make_frame` helper:

```python
def test_scoring_context_from_batches_single():
    batch = ScoreBatch(
        confidence=torch.tensor([[0.8, 0.2]]),
        raw_similarity=torch.tensor([[0.5, 0.3]]),
        logits=torch.tensor([[1.5, -1.0]]),
        semantics="siglip2_pairwise_sigmoid",
    )
    frames = [make_frame(0)]
    ctx = ScoringContext.from_batches([batch], ["a", "b"], frames)
    assert ctx.confidence.shape == (1, 2)
    assert ctx.semantics == "siglip2_pairwise_sigmoid"
    assert ctx.labels == ["a", "b"]


def test_scoring_context_from_batches_multi():
    batch1 = ScoreBatch(
        confidence=torch.tensor([[0.8, 0.2]]),
        raw_similarity=torch.tensor([[0.5, 0.3]]),
        logits=torch.tensor([[1.5, -1.0]]),
        semantics="siglip2_pairwise_sigmoid",
    )
    batch2 = ScoreBatch(
        confidence=torch.tensor([[0.3, 0.9]]),
        raw_similarity=torch.tensor([[0.2, 0.7]]),
        logits=torch.tensor([[-0.5, 2.0]]),
        semantics="siglip2_pairwise_sigmoid",
    )
    frames = [make_frame(0), make_frame(1)]
    ctx = ScoringContext.from_batches([batch1, batch2], ["a", "b"], frames)
    assert ctx.confidence.shape == (2, 2)
    assert ctx.logits.shape == (2, 2)


def test_scoring_context_empty_batches_raises():
    with pytest.raises(ValueError, match="empty batch list"):
        ScoringContext.from_batches([], ["a"], [])


def test_scoring_context_mixed_semantics_raises():
    batch1 = ScoreBatch(
        confidence=torch.tensor([[0.8]]),
        raw_similarity=torch.tensor([[0.5]]),
        logits=torch.tensor([[1.5]]),
        semantics="siglip2_pairwise_sigmoid",
    )
    batch2 = ScoreBatch(
        confidence=torch.tensor([[0.3]]),
        raw_similarity=torch.tensor([[0.2]]),
        logits=torch.tensor([[-0.5]]),
        semantics="clip_relative_softmax",
    )
    with pytest.raises(ValueError, match="Mixed model semantics"):
        ScoringContext.from_batches([batch1, batch2], ["a"], [make_frame(0), make_frame(1)])
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `python -m pytest tests/test_scoring.py::test_scoring_context_from_batches_single -v`
Expected: `ImportError` -- `ScoringContext` doesn't exist yet

- [ ] **Step 3: Rewrite scoring.py with ScoringContext, AggregationResult, and refactored aggregators**

Replace the entire contents of `app/services/scoring.py` with:

```python
# app/services/scoring.py
from __future__ import annotations

from dataclasses import dataclass

import torch

from app.models.base_model import ScoreBatch
from app.schemas.response import BestMatch, ScoreItem
from app.services.video import FrameSample


@dataclass
class ScoringContext:
    confidence: torch.Tensor
    raw_similarity: torch.Tensor
    logits: torch.Tensor
    semantics: str
    labels: list[str]
    frames: list[FrameSample]

    @classmethod
    def from_batches(
        cls,
        batches: list[ScoreBatch],
        labels: list[str],
        frames: list[FrameSample],
    ) -> ScoringContext:
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
    timeline: "FrameTimeline"  # type: ignore[name-defined]

    @classmethod
    def from_base(
        cls, ctx: ScoringContext, timeline: "FrameTimeline",  # type: ignore[name-defined]
    ) -> TemporalScoringContext:
        return cls(
            confidence=ctx.confidence,
            raw_similarity=ctx.raw_similarity,
            logits=ctx.logits,
            semantics=ctx.semantics,
            labels=ctx.labels,
            frames=ctx.frames,
            timeline=timeline,
        )


@dataclass
class AggregationResult:
    scores: list[ScoreItem]
    best_match: BestMatch
    temporal: "TemporalResult | None" = None  # type: ignore[name-defined]


def compute_frame_scores(
    cosine_sim: torch.Tensor, logit_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_similarity = cosine_sim.clone()
    scaled_logits = cosine_sim * logit_scale
    confidence = torch.softmax(scaled_logits, dim=-1)
    return confidence, raw_similarity


def aggregate_mean(ctx: ScoringContext) -> AggregationResult:
    mean_conf = ctx.confidence.mean(dim=0)
    mean_raw = ctx.raw_similarity.mean(dim=0)
    scores = [
        ScoreItem(
            label=ctx.labels[i],
            confidence=round(mean_conf[i].item(), 6),
            raw_similarity=round(mean_raw[i].item(), 6),
        )
        for i in range(len(ctx.labels))
    ]
    best = max(scores, key=lambda s: s.confidence)
    return AggregationResult(
        scores=scores,
        best_match=BestMatch(label=best.label, confidence=best.confidence),
    )


def aggregate_max(ctx: ScoringContext) -> AggregationResult:
    max_conf, max_indices = ctx.confidence.max(dim=0)
    scores = [
        ScoreItem(
            label=ctx.labels[i],
            confidence=round(max_conf[i].item(), 6),
            raw_similarity=round(
                ctx.raw_similarity[max_indices[i].item(), i].item(), 6
            ),
            peak_frame_index=max_indices[i].item(),
            approx_timestamp_seconds=ctx.frames[
                max_indices[i].item()
            ].approx_timestamp_seconds,
        )
        for i in range(len(ctx.labels))
    ]
    best = max(scores, key=lambda s: s.confidence)
    return AggregationResult(
        scores=scores,
        best_match=BestMatch(label=best.label, confidence=best.confidence),
    )


def aggregate_temporal(
    ctx: TemporalScoringContext,
    temporal_options: "ResolvedTemporalOptions",  # type: ignore[name-defined]
    policy: "TemporalScoringPolicy",  # type: ignore[name-defined]
) -> AggregationResult:
    raise NotImplementedError("aggregate_temporal will be implemented in Task 6")


def aggregate_frame_scores(
    batches: list[ScoreBatch],
    labels: list[str],
    frames: list[FrameSample],
    aggregation: str,
    temporal_options: "ResolvedTemporalOptions | None" = None,  # type: ignore[name-defined]
    timeline: "FrameTimeline | None" = None,  # type: ignore[name-defined]
    policy: "TemporalScoringPolicy | None" = None,  # type: ignore[name-defined]
) -> AggregationResult:
    ctx = ScoringContext.from_batches(batches, labels, frames)

    if aggregation == "temporal":
        if timeline is None or policy is None or temporal_options is None:
            raise ValueError(
                "Temporal aggregation requires timeline, policy, and temporal_options"
            )
        temporal_ctx = TemporalScoringContext.from_base(ctx, timeline)
        return aggregate_temporal(temporal_ctx, temporal_options, policy)
    elif aggregation == "max":
        return aggregate_max(ctx)
    else:
        return aggregate_mean(ctx)
```

- [ ] **Step 4: Update existing tests to use new signatures**

Replace the existing test functions in `tests/test_scoring.py` (keep `make_frame`, `compute_frame_scores` tests, and the new `ScoringContext` tests). Update the import line at top from:
```python
from app.services.scoring import compute_frame_scores, aggregate_mean, aggregate_max, build_response_scores, aggregate_frame_scores
```
to:
```python
from app.services.scoring import compute_frame_scores, aggregate_mean, aggregate_max, aggregate_frame_scores, ScoringContext, AggregationResult
```

Replace `test_aggregate_mean`:
```python
def test_aggregate_mean():
    conf = torch.tensor([[0.3, 0.7], [0.4, 0.6]])
    raw = torch.tensor([[0.25, 0.30], [0.27, 0.28]])
    labels = ["driving", "parking"]
    frames = [make_frame(0), make_frame(1)]
    ctx = ScoringContext(
        confidence=conf, raw_similarity=raw,
        logits=torch.zeros_like(conf), semantics="test",
        labels=labels, frames=frames,
    )
    result = aggregate_mean(ctx)
    assert len(result.scores) == 2
    assert abs(result.scores[0].confidence - 0.35) < 1e-5
    assert result.scores[0].peak_frame_index is None
```

Replace `test_aggregate_max`:
```python
def test_aggregate_max():
    conf = torch.tensor([[0.3, 0.7], [0.8, 0.2]])
    raw = torch.tensor([[0.25, 0.30], [0.31, 0.22]])
    labels = ["driving", "parking"]
    frames = [make_frame(0), make_frame(1)]
    ctx = ScoringContext(
        confidence=conf, raw_similarity=raw,
        logits=torch.zeros_like(conf), semantics="test",
        labels=labels, frames=frames,
    )
    result = aggregate_max(ctx)
    assert len(result.scores) == 2
    assert abs(result.scores[0].confidence - 0.8) < 1e-5
    assert result.scores[0].peak_frame_index == 1
    assert abs(result.scores[1].confidence - 0.7) < 1e-5
    assert result.scores[1].peak_frame_index == 0
```

**Delete** `test_build_response_mean` and `test_build_response_max` (logic now lives inside `aggregate_mean`/`aggregate_max`).

Replace `test_aggregate_frame_scores_mean`:
```python
def test_aggregate_frame_scores_mean():
    batch1 = ScoreBatch(
        confidence=torch.tensor([[0.8, 0.1, 0.1], [0.6, 0.2, 0.2]]),
        raw_similarity=torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.3, 0.3]]),
        logits=torch.tensor([[2.0, -1.0, -1.0], [1.0, -0.5, -0.5]]),
        semantics="clip_relative_softmax",
    )
    frames = [make_frame(0), make_frame(1)]
    labels = ["cat", "dog", "bird"]

    result = aggregate_frame_scores([batch1], labels, frames, "mean")
    assert len(result.scores) == 3
    assert result.best_match.label == "cat"
    assert result.best_match.confidence > 0
```

Replace `test_aggregate_frame_scores_max`:
```python
def test_aggregate_frame_scores_max():
    batch1 = ScoreBatch(
        confidence=torch.tensor([[0.3, 0.9, 0.1]]),
        raw_similarity=torch.tensor([[0.2, 0.7, 0.1]]),
        logits=torch.tensor([[0.0, 2.0, -1.0]]),
        semantics="siglip2_pairwise_sigmoid",
    )
    batch2 = ScoreBatch(
        confidence=torch.tensor([[0.8, 0.2, 0.1]]),
        raw_similarity=torch.tensor([[0.6, 0.1, 0.05]]),
        logits=torch.tensor([[1.5, -0.5, -1.0]]),
        semantics="siglip2_pairwise_sigmoid",
    )
    frames = [make_frame(0), make_frame(1)]
    labels = ["cat", "dog", "bird"]

    result = aggregate_frame_scores([batch1, batch2], labels, frames, "max")
    assert len(result.scores) == 3
    assert result.best_match.label == "dog"
```

- [ ] **Step 5: Update main.py to use AggregationResult**

In `app/main.py`, change lines 287-289 from:
```python
                    scores, best_match = aggregate_frame_scores(
                        all_batches, parsed_labels, all_frames, aggregation
                    )
```
to:
```python
                    agg_result = aggregate_frame_scores(
                        all_batches, parsed_labels, all_frames, aggregation
                    )
```

Update the response construction at lines 295-309 to use `agg_result`:
```python
                    return ClassifyResponse(
                        best_match=agg_result.best_match,
                        scores=agg_result.scores,
                        metadata=ClassifyMetadata(
                            frames_analyzed=len(all_frames),
                            video_duration_seconds=video_info.duration,
                            model=manager.active_model_id or "",
                            device=model.device,
                            aggregation=aggregation,
                            processing_time_seconds=round(processing_time, 3),
                            disclaimer=disclaimer,
                            model_type=model.model_type,
                            score_semantics=semantics,
                        ),
                        temporal=agg_result.temporal,
                    )
```

- [ ] **Step 6: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add app/services/scoring.py app/main.py tests/test_scoring.py
git commit -m "refactor: scoring pipeline to ScoringContext and AggregationResult"
```

---

### Task 5: Model Files Use ScoreSemantics Constants

**Files:**
- Modify: `app/models/siglip2_model.py` (line 65)
- Modify: `app/models/clip_model.py` (line 49)

- [ ] **Step 1: Update siglip2_model.py**

Add import at top (after line 7):
```python
from app.services.temporal_policy import ScoreSemantics
```

Change line 65 from `semantics="siglip2_pairwise_sigmoid",` to:
```python
            semantics=ScoreSemantics.SIGLIP2_SIGMOID,
```

- [ ] **Step 2: Update clip_model.py**

Add import at top (after line 4):
```python
from app.services.temporal_policy import ScoreSemantics
```

Change line 49 from `semantics="clip_relative_softmax",` to:
```python
            semantics=ScoreSemantics.CLIP_RELATIVE_SOFTMAX,
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_scoring.py tests/test_temporal_policy.py -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add app/models/siglip2_model.py app/models/clip_model.py
git commit -m "refactor: use ScoreSemantics constants in model files"
```

---

### Task 6: aggregate_temporal() Implementation

**Files:**
- Modify: `app/services/scoring.py`
- Modify: `tests/test_scoring.py`

- [ ] **Step 1: Write tests for aggregate_temporal**

Add to `tests/test_scoring.py` (new imports and test functions):

```python
from app.services.scoring import aggregate_temporal, TemporalScoringContext
from app.services.frame_timeline import FrameTimeline
from app.services.temporal_policy import SigLip2Policy
from app.schemas.response import ResolvedTemporalOptions


def make_temporal_ctx(
    confidence: torch.Tensor,
    labels: list[str],
    fps: float = 1.0,
    video_duration: float | None = None,
) -> TemporalScoringContext:
    n_frames = confidence.shape[0]
    if video_duration is None:
        video_duration = float(n_frames) / fps
    frames = [make_frame(i, fps) for i in range(n_frames)]
    timeline = FrameTimeline(frames, fps, video_duration)
    ctx = ScoringContext(
        confidence=confidence,
        raw_similarity=confidence.clone(),
        logits=torch.zeros_like(confidence),
        semantics="siglip2_pairwise_sigmoid",
        labels=labels,
        frames=frames,
    )
    return TemporalScoringContext.from_base(ctx, timeline)


def default_options(threshold: float = 0.5) -> ResolvedTemporalOptions:
    return ResolvedTemporalOptions(
        threshold=threshold,
        gap_tolerance=2.0,
        min_duration=1.0,
        threshold_was_defaulted=True,
    )


def test_temporal_basic_segment_detection():
    scores = torch.tensor([
        [0.1], [0.2], [0.1],
        [0.7], [0.8], [0.9], [0.6], [0.7],
        [0.1], [0.2],
    ])
    ctx = make_temporal_ctx(scores, ["sleepy"])
    opts = default_options(threshold=0.5)
    policy = SigLip2Policy()

    result = aggregate_temporal(ctx, opts, policy)
    assert result.temporal is not None
    assert len(result.temporal.timeline) == 10
    assert len(result.temporal.segments) == 1

    seg = result.temporal.segments[0]
    assert seg.label == "sleepy"
    assert seg.start_time == 3.0
    assert seg.end_time == 8.0
    assert abs(seg.duration - 5.0) < 1e-6
    assert seg.peak_confidence == pytest.approx(0.9, abs=1e-6)
    assert seg.peak_timestamp == 5.0


def test_temporal_gap_bridging():
    scores = torch.tensor([[0.8], [0.7], [0.3], [0.8], [0.9]])
    ctx = make_temporal_ctx(scores, ["sleepy"])
    opts = ResolvedTemporalOptions(
        threshold=0.5, gap_tolerance=2.0, min_duration=0.0,
        threshold_was_defaulted=True,
    )
    result = aggregate_temporal(ctx, opts, SigLip2Policy())
    assert len(result.temporal.segments) == 1
    seg = result.temporal.segments[0]
    assert seg.start_time == 0.0
    assert seg.end_time == 5.0


def test_temporal_gap_not_bridged():
    scores = torch.tensor([[0.8], [0.7], [0.3], [0.8], [0.9]])
    ctx = make_temporal_ctx(scores, ["sleepy"])
    opts = ResolvedTemporalOptions(
        threshold=0.5, gap_tolerance=0.0, min_duration=0.0,
        threshold_was_defaulted=True,
    )
    result = aggregate_temporal(ctx, opts, SigLip2Policy())
    assert len(result.temporal.segments) == 2


def test_temporal_min_duration_filter():
    scores = torch.tensor([[0.1], [0.8], [0.1], [0.1], [0.1]])
    ctx = make_temporal_ctx(scores, ["sleepy"])
    opts = ResolvedTemporalOptions(
        threshold=0.5, gap_tolerance=0.0, min_duration=2.0,
        threshold_was_defaulted=True,
    )
    result = aggregate_temporal(ctx, opts, SigLip2Policy())
    assert len(result.temporal.segments) == 0


def test_temporal_all_below_threshold():
    scores = torch.tensor([[0.1], [0.2], [0.3]])
    ctx = make_temporal_ctx(scores, ["sleepy"])
    result = aggregate_temporal(ctx, default_options(), SigLip2Policy())
    assert len(result.temporal.segments) == 0
    assert result.temporal.best_segment is None
    assert len(result.temporal.timeline) == 3
    assert len(result.temporal.label_summaries) == 1
    assert result.temporal.label_summaries[0].segment_count == 0


def test_temporal_all_above_threshold():
    scores = torch.tensor([[0.8], [0.9], [0.7]])
    ctx = make_temporal_ctx(scores, ["sleepy"])
    opts = ResolvedTemporalOptions(
        threshold=0.5, gap_tolerance=0.0, min_duration=0.0,
        threshold_was_defaulted=True,
    )
    result = aggregate_temporal(ctx, opts, SigLip2Policy())
    assert len(result.temporal.segments) == 1
    seg = result.temporal.segments[0]
    assert seg.stats.coverage_ratio == 1.0
    assert seg.stats.active_avg == seg.stats.interval_avg


def test_temporal_segment_stats_with_gap():
    scores = torch.tensor([[0.8], [0.7], [0.3], [0.9]])
    ctx = make_temporal_ctx(scores, ["sleepy"])
    opts = ResolvedTemporalOptions(
        threshold=0.5, gap_tolerance=2.0, min_duration=0.0,
        threshold_was_defaulted=True,
    )
    result = aggregate_temporal(ctx, opts, SigLip2Policy())
    seg = result.temporal.segments[0]
    assert seg.stats.active_avg == pytest.approx(0.8, abs=1e-6)
    assert seg.stats.interval_avg == pytest.approx(0.675, abs=1e-6)
    assert seg.stats.coverage_ratio == pytest.approx(0.75, abs=1e-6)
    assert seg.stats.active_duration == pytest.approx(3.0, abs=1e-6)


def test_temporal_label_summaries():
    scores = torch.tensor([
        [0.8, 0.1],
        [0.9, 0.2],
        [0.7, 0.1],
    ])
    ctx = make_temporal_ctx(scores, ["sleepy", "awake"])
    opts = ResolvedTemporalOptions(
        threshold=0.5, gap_tolerance=0.0, min_duration=0.0,
        threshold_was_defaulted=True,
    )
    result = aggregate_temporal(ctx, opts, SigLip2Policy())
    summaries = {s.label: s for s in result.temporal.label_summaries}
    assert summaries["sleepy"].segment_count == 1
    assert summaries["sleepy"].total_active_duration == pytest.approx(3.0, abs=1e-6)
    assert summaries["sleepy"].peak_confidence == pytest.approx(0.9, abs=1e-6)
    assert summaries["awake"].segment_count == 0
    assert summaries["awake"].total_active_duration == 0.0


def test_temporal_best_segment():
    scores = torch.tensor([
        [0.7, 0.95],
        [0.6, 0.1],
    ])
    ctx = make_temporal_ctx(scores, ["a", "b"])
    opts = ResolvedTemporalOptions(
        threshold=0.5, gap_tolerance=0.0, min_duration=0.0,
        threshold_was_defaulted=True,
    )
    result = aggregate_temporal(ctx, opts, SigLip2Policy())
    assert result.temporal.best_segment is not None
    assert result.temporal.best_segment.label == "b"
    assert result.temporal.best_segment.peak_confidence == pytest.approx(0.95, abs=1e-6)


def test_temporal_effective_threshold_in_result():
    scores = torch.tensor([[0.8], [0.2]])
    ctx = make_temporal_ctx(scores, ["x"])
    opts = ResolvedTemporalOptions(
        threshold=0.6, gap_tolerance=2.0, min_duration=1.0,
        threshold_was_defaulted=False,
    )
    result = aggregate_temporal(ctx, opts, SigLip2Policy())
    assert result.temporal.effective_threshold == 0.6
    assert result.temporal.threshold_was_defaulted is False
    assert result.temporal.threshold_mode == "absolute"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scoring.py::test_temporal_basic_segment_detection -v`
Expected: `NotImplementedError`

- [ ] **Step 3: Implement aggregate_temporal**

Replace the `aggregate_temporal` stub in `app/services/scoring.py` with:

```python
def aggregate_temporal(
    ctx: TemporalScoringContext,
    temporal_options: "ResolvedTemporalOptions",
    policy: "TemporalScoringPolicy",
) -> AggregationResult:
    from app.schemas.response import (
        FrameScore,
        LabelSummary,
        Segment,
        SegmentStats,
        TemporalResult,
    )

    detection = policy.detection_scores(ctx)
    threshold = temporal_options.threshold
    gap_tol = temporal_options.gap_tolerance
    min_dur = temporal_options.min_duration
    tl = ctx.timeline

    timeline_entries = []
    for i in range(detection.shape[0]):
        scores_dict = {
            ctx.labels[j]: round(detection[i, j].item(), 6)
            for j in range(len(ctx.labels))
        }
        timeline_entries.append(
            FrameScore(timestamp=tl.timestamp(i), frame_index=i, scores=scores_dict)
        )

    all_segments: list[Segment] = []
    for j, label in enumerate(ctx.labels):
        label_scores = detection[:, j]
        mask = label_scores >= threshold

        raw_segments: list[tuple[int, int]] = []
        in_segment = False
        start_idx = 0
        for i in range(len(mask)):
            if mask[i] and not in_segment:
                start_idx = i
                in_segment = True
            elif not mask[i] and in_segment:
                raw_segments.append((start_idx, i - 1))
                in_segment = False
        if in_segment:
            raw_segments.append((start_idx, len(mask) - 1))

        merged: list[tuple[int, int]] = []
        for seg in raw_segments:
            if merged and tl.gap_seconds(merged[-1][1], seg[0]) <= gap_tol:
                merged[-1] = (merged[-1][0], seg[1])
            else:
                merged.append(seg)

        for start, end in merged:
            duration = tl.segment_duration(start, end)
            if duration < min_dur:
                continue

            interval_scores = label_scores[start : end + 1]
            active_mask = interval_scores >= threshold
            active_scores = interval_scores[active_mask]

            active_dur = sum(
                tl.intervals[start + k].end - tl.intervals[start + k].start
                for k in range(end - start + 1)
                if active_mask[k]
            )

            peak_val, peak_rel = interval_scores.max(dim=0)
            peak_idx = start + peak_rel.item()

            all_segments.append(
                Segment(
                    label=label,
                    start_time=tl.timestamp(start),
                    end_time=tl.intervals[end].end,
                    duration=round(duration, 6),
                    stats=SegmentStats(
                        active_avg=round(active_scores.mean().item(), 6),
                        interval_avg=round(interval_scores.mean().item(), 6),
                        coverage_ratio=round(
                            active_mask.sum().item() / len(interval_scores), 6
                        ),
                        active_duration=round(active_dur, 6),
                    ),
                    peak_confidence=round(peak_val.item(), 6),
                    peak_timestamp=tl.timestamp(peak_idx),
                )
            )

    label_summaries: list[LabelSummary] = []
    for label in ctx.labels:
        label_segs = [s for s in all_segments if s.label == label]
        total_active = sum(s.stats.active_duration for s in label_segs)
        total_segment = sum(s.duration for s in label_segs)
        peak = max((s.peak_confidence for s in label_segs), default=0.0)

        if total_active > 0:
            dwc = sum(
                s.stats.active_duration * s.stats.active_avg for s in label_segs
            ) / total_active
        else:
            dwc = 0.0

        label_summaries.append(
            LabelSummary(
                label=label,
                segment_count=len(label_segs),
                total_active_duration=round(total_active, 6),
                total_segment_duration=round(total_segment, 6),
                peak_confidence=round(peak, 6),
                duration_weighted_confidence=round(dwc, 6),
            )
        )

    best_segment = (
        max(all_segments, key=lambda s: s.peak_confidence) if all_segments else None
    )

    mean_result = aggregate_mean(ctx)

    return AggregationResult(
        scores=mean_result.scores,
        best_match=mean_result.best_match,
        temporal=TemporalResult(
            timeline=timeline_entries,
            segments=all_segments,
            label_summaries=label_summaries,
            best_segment=best_segment,
            threshold_mode=policy.threshold_mode(),
            effective_threshold=temporal_options.threshold,
            threshold_was_defaulted=temporal_options.threshold_was_defaulted,
        ),
    )
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/test_scoring.py tests/test_temporal_policy.py tests/test_frame_timeline.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/scoring.py tests/test_scoring.py
git commit -m "feat: implement aggregate_temporal with segment detection and stats"
```

---

### Task 7: API Endpoint -- Temporal Params + Validation

**Files:**
- Modify: `app/errors/handlers.py`
- Modify: `app/main.py`

- [ ] **Step 1: Add InvalidTemporalParamsError**

Add to the end of `app/errors/handlers.py`:

```python
class InvalidTemporalParamsError(HTTPException):
    def __init__(self) -> None:
        super().__init__(
            status_code=422,
            detail="Parameters 'threshold', 'gap_tolerance', 'min_duration' are only valid with aggregation='temporal'.",
        )
```

- [ ] **Step 2: Update main.py classify endpoint**

Add imports near the top of `app/main.py` (after line 40):
```python
from app.schemas.response import RawTemporalParams, ResolvedTemporalOptions
from app.services.temporal_policy import get_policy
from app.services.frame_timeline import FrameTimeline
```

Add to the error imports block:
```python
from app.errors.handlers import (
    ...existing imports...,
    InvalidTemporalParamsError,
)
```

Update the `classify` function signature to accept temporal params:
```python
    @app.post("/api/v1/classify", response_model=ClassifyResponse)
    async def classify(
        video: UploadFile,
        labels: str = Form(...),
        prompt_template: str = Form(default="This is a photo of {}."),
        fps: float = Form(default=1.0),
        aggregation: str = Form(default="mean"),
        threshold: float | None = Form(default=None, ge=0.0, le=1.0),
        gap_tolerance: float | None = Form(default=None, ge=0.0, le=10.0),
        min_duration: float | None = Form(default=None, ge=0.0, le=10.0),
    ):
```

Update the aggregation validation from `if aggregation not in ("mean", "max"):` to:
```python
        if aggregation not in ("mean", "max", "temporal"):
            raise InvalidAggregationError(aggregation)

        raw_temporal = RawTemporalParams(
            threshold=threshold,
            gap_tolerance=gap_tolerance,
            min_duration=min_duration,
        )
        if aggregation != "temporal" and raw_temporal.has_any():
            raise InvalidTemporalParamsError()
```

After `all_batches, all_frames = result` (around line 285), before the aggregation call, add temporal setup:
```python
                        temporal_opts = None
                        timeline = None
                        policy = None
                        if aggregation == "temporal":
                            batch_semantics = all_batches[0].semantics if all_batches else ""
                            policy = get_policy(batch_semantics)
                            temporal_opts = ResolvedTemporalOptions.resolve(
                                raw_temporal, policy.default_threshold()
                            )
                            timeline = FrameTimeline(all_frames, fps, video_info.duration)
```

Update the `aggregate_frame_scores` call to pass the new arguments:
```python
                        agg_result = aggregate_frame_scores(
                            all_batches, parsed_labels, all_frames, aggregation,
                            temporal_options=temporal_opts,
                            timeline=timeline,
                            policy=policy,
                        )
```

- [ ] **Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add app/errors/handlers.py app/main.py
git commit -m "feat: accept temporal params in classify endpoint with validation"
```

---

### Task 8: Model Metadata -- temporal_defaults

**Files:**
- Modify: `app/main.py` (active_model endpoint)

- [ ] **Step 1: Update `/api/v1/models/active` to include temporal_defaults**

Replace the `active_model` endpoint body (lines 139-151) with:

```python
    @app.get("/api/v1/models/active")
    async def active_model():
        manager: ModelManager = state["manager"]
        if manager.active_model is None:
            return JSONResponse(status_code=404, content={"detail": "No model loaded"})
        config = manager.registry[manager.active_model_id]

        from app.services.temporal_policy import ScoreSemantics

        model = manager.active_model
        semantics_str = ""
        if model.model_type == "siglip2":
            semantics_str = ScoreSemantics.SIGLIP2_SIGMOID
        elif model.model_type == "clip":
            semantics_str = ScoreSemantics.CLIP_RELATIVE_SOFTMAX

        temporal_defaults = None
        if semantics_str:
            try:
                policy = get_policy(semantics_str)
                temporal_defaults = {
                    "threshold": policy.default_threshold(),
                    "threshold_mode": policy.threshold_mode(),
                    "gap_tolerance": 2.0,
                    "min_duration": 1.0,
                }
            except ValueError:
                pass

        return {
            "model_id": config.model_id,
            "display_name": config.display_name,
            "model_type": config.model_type,
            "params": config.params,
            "resolution": config.resolution,
            "device": manager.active_model.device,
            "temporal_defaults": temporal_defaults,
        }
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add app/main.py
git commit -m "feat: expose temporal_defaults on /api/v1/models/active"
```

---

### Task 9: Vendor Chart.js

**Files:**
- Create: `app/static/vendor/chart.min.js`

- [ ] **Step 1: Download Chart.js 4.x**

```bash
mkdir -p app/static/vendor
curl -L -o app/static/vendor/chart.min.js https://cdn.jsdelivr.net/npm/chart.js@4.4.9/dist/chart.umd.min.js
```

- [ ] **Step 2: Verify the file exists and is non-trivial**

```bash
ls -lh app/static/vendor/chart.min.js
head -c 100 app/static/vendor/chart.min.js
```
Expected: file is ~200KB+, starts with JS code

- [ ] **Step 3: Add a static vendor route to main.py**

In `app/main.py`, after the `serve_ui` endpoint (around line 155), add:
```python
    @app.get("/static/vendor/{filename}")
    async def serve_vendor(filename: str):
        path = STATIC_DIR / "vendor" / filename
        if not path.exists():
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        return FileResponse(path)
```

- [ ] **Step 4: Commit**

```bash
git add app/static/vendor/chart.min.js app/main.py
git commit -m "vendor: add Chart.js 4.4.9 for temporal visualization"
```

---

### Task 10: UI -- Temporal Mode

**Files:**
- Modify: `app/static/index.html`

This is the largest single task. The UI adds: temporal option in aggregation dropdown, 3 sliders with tooltips that show/hide on temporal selection, dirty-tracking for threshold, Chart.js line chart with threshold line, segment summary table, and best segment badge.

- [ ] **Step 1: Replace index.html with temporal mode support**

Replace the entire contents of `app/static/index.html`. Key changes from existing:
1. `<script src="/static/vendor/chart.min.js"></script>` in `<head>`
2. `"temporal"` option in aggregation select
3. Temporal parameter panel (3 sliders + tooltip icons) that shows/hides
4. Temporal results section (chart canvas + segment table + best segment badge)
5. Dirty-tracking for threshold slider
6. Fetch `temporal_defaults` from `/api/v1/models/active` on model load
7. Only send `threshold` in request if user explicitly changed the slider

See the full file in the spec. The badge rendering must use safe DOM methods (no innerHTML):

```javascript
// Best segment badge -- safe DOM construction
if (temporal.best_segment) {
    var bs = temporal.best_segment;
    while (bestSegBadge.firstChild) bestSegBadge.removeChild(bestSegBadge.firstChild);
    var strong = document.createElement('strong');
    strong.textContent = 'Strongest burst: ';
    bestSegBadge.appendChild(strong);
    bestSegBadge.appendChild(document.createTextNode(
        bs.label + ' at ' + bs.start_time.toFixed(1) + 's–' +
        bs.end_time.toFixed(1) + 's (peak ' +
        (bs.peak_confidence * 100).toFixed(1) + '%)'
    ));
    bestSegBadge.style.display = '';
} else {
    bestSegBadge.style.display = 'none';
}
```

- [ ] **Step 2: Start dev server and test in browser**

Run: `python -m uvicorn app.main:create_app --factory --reload --port 8000`

Test at `http://localhost:8000`:
1. Mean/Max mode: unchanged behavior, temporal panel hidden
2. Select "temporal" from dropdown: panel appears with 3 sliders
3. Hover tooltip icons: tooltip cards appear
4. Threshold slider shows "(model default)" label; label disappears when slider moved
5. Submit a classify request in temporal mode: chart renders with lines + threshold dashed line
6. Segment table populates with correct data
7. Best segment badge shows strongest burst
8. Switch back to "mean" mode: temporal UI hidden, standard results shown

- [ ] **Step 3: Commit**

```bash
git add app/static/index.html
git commit -m "feat: temporal mode UI with chart, sliders, and segment table"
```

---

### Task 11: End-to-End Validation

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests PASS

- [ ] **Step 2: Verify API temporal mode with curl**

```bash
# Should return 422 when temporal params used with mean mode
curl -s -X POST http://localhost:8000/api/v1/classify \
  -F "video=@test_video.mp4" \
  -F 'labels=["a","b","c"]' \
  -F "aggregation=mean" \
  -F "threshold=0.5" | python -m json.tool

# Should succeed with temporal mode
curl -s -X POST http://localhost:8000/api/v1/classify \
  -F "video=@test_video.mp4" \
  -F 'labels=["a","b","c"]' \
  -F "aggregation=temporal" \
  -F "gap_tolerance=2.0" \
  -F "min_duration=1.0" | python -m json.tool
```

- [ ] **Step 3: Verify temporal_defaults in model metadata**

```bash
curl -s http://localhost:8000/api/v1/models/active | python -m json.tool
```

Expected: response includes `"temporal_defaults": {"threshold": 0.5, "threshold_mode": "absolute", ...}`

- [ ] **Step 4: Final commit if any fixups needed**

```bash
git add -A
git commit -m "fix: address any issues found during e2e validation"
```
