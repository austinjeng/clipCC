# Contrast Scoring Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `contrast` aggregation mode that scores video against positive and negative label groups and returns a three-way verdict (positive/negative/uncertain).

**Architecture:** Single inference pass with combined label list, then split results by group index for scoring. SigLIP2 uses mean pooling; CLIP uses logsumexp-normalized logit-space pooling to prevent group-size bias. Per-frame margins are reduced via configurable temporal reducer (mean/top_k_mean/max/quantile), all sign-symmetric.

**Tech Stack:** Python 3.12, FastAPI, PyTorch, Pydantic v2, Chart.js, pytest + pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-05-20-contrast-scoring-design.md` (rev 4)

---

### Task 1: Response Schema

**Files:**
- Modify: `app/schemas/response.py`
- Test: `tests/test_scoring.py` (schema import validation)

- [ ] **Step 1: Write test for ContrastResult serialization**

Add to `tests/test_scoring.py`:

```python
def test_contrast_result_serialization():
    from app.schemas.response import ContrastLabelScore, ContrastGroupResult, ContrastResult
    result = ContrastResult(
        verdict="positive",
        difference=0.27,
        threshold=0.15,
        threshold_was_defaulted=True,
        threshold_source="model_policy",
        calibration_status="uncalibrated",
        contrast_reduce="mean",
        positive=ContrastGroupResult(
            group="positive",
            mean_group_score=0.72,
            labels=[ContrastLabelScore(label="safe driving", score=0.72)],
        ),
        negative=ContrastGroupResult(
            group="negative",
            mean_group_score=0.45,
            labels=[ContrastLabelScore(label="dangerous driving", score=0.45)],
        ),
        score_semantics="siglip2_pairwise_sigmoid",
        label_pooling="mean",
        dominant_label="safe driving",
    )
    d = result.model_dump()
    assert d["verdict"] == "positive"
    assert d["dominant_label"] == "safe driving"
    assert d["positive"]["mean_group_score"] == 0.72
    assert len(d["positive"]["labels"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scoring.py::test_contrast_result_serialization -v`
Expected: FAIL — `ContrastLabelScore` not importable

- [ ] **Step 3: Add contrast models to response.py**

Add to `app/schemas/response.py` before `ClassifyResponse`:

```python
class ContrastLabelScore(BaseModel):
    label: str
    score: float


class ContrastGroupResult(BaseModel):
    group: str
    mean_group_score: float
    labels: list[ContrastLabelScore]


class ContrastResult(BaseModel):
    verdict: str
    difference: float
    threshold: float
    threshold_was_defaulted: bool
    threshold_source: str
    calibration_status: str
    contrast_reduce: str
    positive: ContrastGroupResult
    negative: ContrastGroupResult
    score_semantics: str
    label_pooling: str
    dominant_label: str | None
```

- [ ] **Step 4: Add contrast field to ClassifyResponse**

In `ClassifyResponse`, add:

```python
class ClassifyResponse(BaseModel):
    best_match: BestMatch
    scores: list[ScoreItem]
    metadata: ClassifyMetadata
    temporal: TemporalResult | None = None
    contrast: ContrastResult | None = None
```

- [ ] **Step 5: Add ResolvedContrastOptions**

Add to `app/schemas/response.py` after `ResolvedTemporalOptions`:

```python
class RawContrastParams(BaseModel):
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    contrast_reduce: str | None = None

    def has_any(self) -> bool:
        return self.threshold is not None or self.contrast_reduce is not None


class ResolvedContrastOptions(BaseModel):
    threshold: float
    threshold_was_defaulted: bool
    threshold_source: str
    calibration_status: str
    contrast_reduce: str

    @classmethod
    def resolve(
        cls,
        raw: RawContrastParams,
        policy_threshold: float,
        policy_reduce: str,
    ) -> "ResolvedContrastOptions":
        return cls(
            threshold=raw.threshold if raw.threshold is not None else policy_threshold,
            threshold_was_defaulted=(raw.threshold is None),
            threshold_source="user" if raw.threshold is not None else "model_policy",
            calibration_status="uncalibrated",
            contrast_reduce=raw.contrast_reduce if raw.contrast_reduce is not None else policy_reduce,
        )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_scoring.py::test_contrast_result_serialization -v`
Expected: PASS

- [ ] **Step 7: Test ResolvedContrastOptions.resolve**

Add to `tests/test_scoring.py`:

```python
def test_resolved_contrast_options_defaults():
    from app.schemas.response import RawContrastParams, ResolvedContrastOptions
    raw = RawContrastParams()
    opts = ResolvedContrastOptions.resolve(raw, policy_threshold=0.15, policy_reduce="mean")
    assert opts.threshold == 0.15
    assert opts.threshold_was_defaulted is True
    assert opts.threshold_source == "model_policy"
    assert opts.contrast_reduce == "mean"


def test_resolved_contrast_options_user_override():
    from app.schemas.response import RawContrastParams, ResolvedContrastOptions
    raw = RawContrastParams(threshold=0.25, contrast_reduce="top_k_mean")
    opts = ResolvedContrastOptions.resolve(raw, policy_threshold=0.15, policy_reduce="mean")
    assert opts.threshold == 0.25
    assert opts.threshold_was_defaulted is False
    assert opts.threshold_source == "user"
    assert opts.contrast_reduce == "top_k_mean"
```

Run: `python -m pytest tests/test_scoring.py::test_resolved_contrast_options_defaults tests/test_scoring.py::test_resolved_contrast_options_user_override -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add app/schemas/response.py tests/test_scoring.py
git commit -m "feat: add contrast response schema and ResolvedContrastOptions"
```

---

### Task 2: Contrast Policy

**Files:**
- Modify: `app/services/temporal_policy.py`
- Modify: `tests/test_temporal_policy.py`

- [ ] **Step 1: Write tests for contrast policy methods**

Add to `tests/test_temporal_policy.py`:

```python
def test_siglip2_contrast_defaults():
    policy = SigLip2Policy()
    assert policy.contrast_label_pooling() == "mean"
    assert policy.contrast_default_threshold() == 0.15
    assert policy.contrast_default_reduction() == "mean"


def test_softmax_contrast_defaults():
    policy = SoftmaxPolicy()
    assert policy.contrast_label_pooling() == "logsumexp_normalized"
    assert policy.contrast_default_threshold() == 0.10
    assert policy.contrast_default_reduction() == "mean"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_temporal_policy.py::test_siglip2_contrast_defaults tests/test_temporal_policy.py::test_softmax_contrast_defaults -v`
Expected: FAIL — `contrast_label_pooling` not found

- [ ] **Step 3: Add abstract methods to TemporalScoringPolicy**

In `app/services/temporal_policy.py`, add to `TemporalScoringPolicy`:

```python
class TemporalScoringPolicy(ABC):
    @abstractmethod
    def detection_scores(self, ctx: Any) -> torch.Tensor: ...
    @abstractmethod
    def default_threshold(self) -> float: ...
    @abstractmethod
    def threshold_mode(self) -> str: ...
    @abstractmethod
    def contrast_label_pooling(self) -> str: ...
    @abstractmethod
    def contrast_default_threshold(self) -> float: ...
    @abstractmethod
    def contrast_default_reduction(self) -> str: ...
```

- [ ] **Step 4: Implement on SigLip2Policy**

Add to `SigLip2Policy`:

```python
    def contrast_label_pooling(self) -> str:
        return "mean"
    def contrast_default_threshold(self) -> float:
        return 0.15
    def contrast_default_reduction(self) -> str:
        return "mean"
```

- [ ] **Step 5: Implement on SoftmaxPolicy**

Add to `SoftmaxPolicy`:

```python
    def contrast_label_pooling(self) -> str:
        return "logsumexp_normalized"
    def contrast_default_threshold(self) -> float:
        return 0.10
    def contrast_default_reduction(self) -> str:
        return "mean"
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_temporal_policy.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add app/services/temporal_policy.py tests/test_temporal_policy.py
git commit -m "feat: add contrast policy methods to scoring policies"
```

---

### Task 3: Temporal Reduction Functions

**Files:**
- Modify: `app/services/scoring.py`
- Modify: `tests/test_scoring.py`

- [ ] **Step 1: Write tests for all four reducers**

Add to `tests/test_scoring.py`:

```python
def test_reduce_mean():
    from app.services.scoring import contrast_reduce
    margins = torch.tensor([0.1, 0.2, 0.3, -0.1, -0.2])
    result = contrast_reduce(margins, "mean")
    assert abs(result - 0.06) < 1e-5


def test_reduce_top_k_mean_positive_event():
    from app.services.scoring import contrast_reduce
    margins = torch.tensor([0.01, 0.02, -0.01, 0.8, 0.9, 0.01, -0.02, 0.01, 0.03, 0.02])
    result = contrast_reduce(margins, "top_k_mean")
    # k = max(1, ceil(10*0.10)) = 1, picks largest abs = 0.9
    assert abs(result - 0.9) < 1e-5


def test_reduce_top_k_mean_negative_event():
    from app.services.scoring import contrast_reduce
    margins = torch.tensor([0.01, 0.02, -0.01, -0.8, -0.9, 0.01, -0.02, 0.01, 0.03, 0.02])
    result = contrast_reduce(margins, "top_k_mean")
    assert abs(result - (-0.9)) < 1e-5


def test_reduce_top_k_mean_single_frame():
    from app.services.scoring import contrast_reduce
    margins = torch.tensor([0.5])
    result = contrast_reduce(margins, "top_k_mean")
    assert abs(result - 0.5) < 1e-5


def test_reduce_max_positive():
    from app.services.scoring import contrast_reduce
    margins = torch.tensor([0.1, -0.3, 0.5, -0.2])
    result = contrast_reduce(margins, "max")
    assert abs(result - 0.5) < 1e-5


def test_reduce_max_negative_stronger():
    from app.services.scoring import contrast_reduce
    margins = torch.tensor([0.1, -0.8, 0.3, -0.2])
    result = contrast_reduce(margins, "max")
    assert abs(result - (-0.8)) < 1e-5


def test_reduce_quantile_positive_tail():
    from app.services.scoring import contrast_reduce
    margins = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7])
    result = contrast_reduce(margins, "quantile")
    # 90th percentile is positive and larger abs than 10th percentile
    assert result > 0


def test_reduce_quantile_negative_tail():
    from app.services.scoring import contrast_reduce
    margins = torch.tensor([-0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = contrast_reduce(margins, "quantile")
    # 10th percentile is negative and larger abs than 90th percentile
    assert result < 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scoring.py -k "reduce_" -v`
Expected: FAIL — `contrast_reduce` not importable

- [ ] **Step 3: Implement contrast_reduce**

Add to `app/services/scoring.py`, after the imports add `import math`, then add the function before `aggregate_mean`:

```python
VALID_CONTRAST_REDUCTIONS = {"mean", "top_k_mean", "max", "quantile"}


def contrast_reduce(margins: torch.Tensor, mode: str) -> float:
    if mode == "mean":
        return margins.mean().item()
    elif mode == "top_k_mean":
        k = max(1, math.ceil(len(margins) * 0.10))
        _, indices = margins.abs().topk(k)
        return margins[indices].mean().item()
    elif mode == "max":
        idx = margins.abs().argmax()
        return margins[idx].item()
    elif mode == "quantile":
        pos = torch.quantile(margins.float(), 0.90).item()
        neg = torch.quantile(margins.float(), 0.10).item()
        return pos if abs(pos) >= abs(neg) else neg
    else:
        raise ValueError(f"Unknown contrast reduction mode: {mode}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scoring.py -k "reduce_" -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/scoring.py tests/test_scoring.py
git commit -m "feat: add sign-symmetric temporal reduction functions for contrast mode"
```

---

### Task 4: aggregate_contrast() Scoring Function

**Files:**
- Modify: `app/services/scoring.py`
- Modify: `tests/test_scoring.py`

- [ ] **Step 1: Write test for SigLIP2 contrast scoring**

Add to `tests/test_scoring.py`:

```python
from app.schemas.response import ResolvedContrastOptions


def test_aggregate_contrast_siglip2_positive_verdict():
    from app.services.scoring import aggregate_contrast
    # 5 frames, 2 positive labels (high scores), 1 negative label (low scores)
    confidence = torch.tensor([
        [0.8, 0.7, 0.2],
        [0.9, 0.8, 0.1],
        [0.7, 0.6, 0.3],
        [0.8, 0.7, 0.2],
        [0.9, 0.8, 0.1],
    ])
    frames = [make_frame(i) for i in range(5)]
    ctx = ScoringContext(
        confidence=confidence,
        raw_similarity=confidence.clone(),
        logits=torch.zeros_like(confidence),
        semantics="siglip2_pairwise_sigmoid",
        labels=["safe", "calm", "dangerous"],
        frames=frames,
    )
    opts = ResolvedContrastOptions(
        threshold=0.15,
        threshold_was_defaulted=True,
        threshold_source="model_policy",
        calibration_status="uncalibrated",
        contrast_reduce="mean",
    )
    policy = SigLip2Policy()
    result = aggregate_contrast(ctx, pos_count=2, options=opts, policy=policy)

    assert result.contrast is not None
    assert result.contrast.verdict == "positive"
    assert result.contrast.difference > 0.15
    assert result.contrast.positive.group == "positive"
    assert result.contrast.negative.group == "negative"
    assert len(result.contrast.positive.labels) == 2
    assert len(result.contrast.negative.labels) == 1
    assert result.contrast.label_pooling == "mean"
    assert result.contrast.dominant_label is not None
    # Backward compat: scores and best_match populated
    assert len(result.scores) == 3
    assert result.best_match.label is not None
```

- [ ] **Step 2: Write test for negative verdict**

```python
def test_aggregate_contrast_siglip2_negative_verdict():
    from app.services.scoring import aggregate_contrast
    confidence = torch.tensor([
        [0.1, 0.2, 0.8, 0.9],
        [0.2, 0.1, 0.9, 0.8],
    ])
    frames = [make_frame(i) for i in range(2)]
    ctx = ScoringContext(
        confidence=confidence,
        raw_similarity=confidence.clone(),
        logits=torch.zeros_like(confidence),
        semantics="siglip2_pairwise_sigmoid",
        labels=["safe", "calm", "reckless", "texting"],
        frames=frames,
    )
    opts = ResolvedContrastOptions(
        threshold=0.15,
        threshold_was_defaulted=True,
        threshold_source="model_policy",
        calibration_status="uncalibrated",
        contrast_reduce="mean",
    )
    result = aggregate_contrast(ctx, pos_count=2, options=opts, policy=SigLip2Policy())
    assert result.contrast.verdict == "negative"
    assert result.contrast.difference < -0.15
```

- [ ] **Step 3: Write test for uncertain verdict**

```python
def test_aggregate_contrast_uncertain_verdict():
    from app.services.scoring import aggregate_contrast
    confidence = torch.tensor([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ])
    frames = [make_frame(i) for i in range(2)]
    ctx = ScoringContext(
        confidence=confidence,
        raw_similarity=confidence.clone(),
        logits=torch.zeros_like(confidence),
        semantics="siglip2_pairwise_sigmoid",
        labels=["a", "b", "c"],
        frames=frames,
    )
    opts = ResolvedContrastOptions(
        threshold=0.15,
        threshold_was_defaulted=True,
        threshold_source="model_policy",
        calibration_status="uncalibrated",
        contrast_reduce="mean",
    )
    result = aggregate_contrast(ctx, pos_count=2, options=opts, policy=SigLip2Policy())
    assert result.contrast.verdict == "uncertain"
    assert abs(result.contrast.difference) <= 0.15
    assert result.contrast.dominant_label is None
```

- [ ] **Step 4: Write test for CLIP logsumexp normalization with group-size imbalance**

```python
def test_aggregate_contrast_clip_group_size_imbalance():
    from app.services.scoring import aggregate_contrast
    from app.services.temporal_policy import SoftmaxPolicy
    # 50 positive labels, 1 negative label
    # With naive sum, uniform logits would give ~50/51 vs 1/51 → always positive
    # With logsumexp normalization, uniform logits should give ~uncertain
    n_pos, n_neg = 50, 1
    n_labels = n_pos + n_neg
    logits = torch.zeros(3, n_labels)  # uniform logits → no evidence either way
    confidence = torch.softmax(logits, dim=-1)
    frames = [make_frame(i) for i in range(3)]
    ctx = ScoringContext(
        confidence=confidence,
        raw_similarity=confidence.clone(),
        logits=logits,
        semantics="clip_relative_softmax",
        labels=[f"pos_{i}" for i in range(n_pos)] + [f"neg_{i}" for i in range(n_neg)],
        frames=frames,
    )
    opts = ResolvedContrastOptions(
        threshold=0.10,
        threshold_was_defaulted=True,
        threshold_source="model_policy",
        calibration_status="uncalibrated",
        contrast_reduce="mean",
    )
    result = aggregate_contrast(ctx, pos_count=n_pos, options=opts, policy=SoftmaxPolicy())
    # With uniform logits and logsumexp normalization, difference should be ~0
    assert result.contrast.verdict == "uncertain"
    assert abs(result.contrast.difference) < 0.10
    assert result.contrast.label_pooling == "logsumexp_normalized"
```

- [ ] **Step 5: Write test for top_k_mean reduction with sparse event**

```python
def test_aggregate_contrast_top_k_mean_sparse_negative():
    from app.services.scoring import aggregate_contrast
    # 10 frames: 9 neutral, 1 strong negative event
    confidence = torch.full((10, 2), 0.5)
    confidence[7, 0] = 0.1  # positive label drops
    confidence[7, 1] = 0.9  # negative label spikes
    frames = [make_frame(i) for i in range(10)]
    ctx = ScoringContext(
        confidence=confidence,
        raw_similarity=confidence.clone(),
        logits=torch.zeros_like(confidence),
        semantics="siglip2_pairwise_sigmoid",
        labels=["safe", "dangerous"],
        frames=frames,
    )
    # With mean reduction, the event gets washed out
    opts_mean = ResolvedContrastOptions(
        threshold=0.15, threshold_was_defaulted=True,
        threshold_source="model_policy", calibration_status="uncalibrated",
        contrast_reduce="mean",
    )
    result_mean = aggregate_contrast(ctx, pos_count=1, options=opts_mean, policy=SigLip2Policy())
    # With top_k_mean, the sparse event is detected
    opts_topk = ResolvedContrastOptions(
        threshold=0.15, threshold_was_defaulted=True,
        threshold_source="model_policy", calibration_status="uncalibrated",
        contrast_reduce="top_k_mean",
    )
    result_topk = aggregate_contrast(ctx, pos_count=1, options=opts_topk, policy=SigLip2Policy())
    # top_k_mean should find the negative event
    assert result_topk.contrast.verdict == "negative"
    assert result_topk.contrast.difference < result_mean.contrast.difference
```

- [ ] **Step 6: Run all contrast tests to verify they fail**

Run: `python -m pytest tests/test_scoring.py -k "contrast" -v`
Expected: FAIL — `aggregate_contrast` not importable

- [ ] **Step 7: Implement aggregate_contrast()**

Add to `app/services/scoring.py` after `aggregate_temporal`:

```python
def aggregate_contrast(
    ctx: ScoringContext,
    pos_count: int,
    options: "ResolvedContrastOptions",  # type: ignore[name-defined]
    policy: "TemporalScoringPolicy",  # type: ignore[name-defined]
) -> AggregationResult:
    from app.schemas.response import (
        ContrastGroupResult,
        ContrastLabelScore,
        ContrastResult,
    )

    neg_count = len(ctx.labels) - pos_count
    pooling = policy.contrast_label_pooling()

    if pooling == "logsumexp_normalized":
        pos_evidence = torch.logsumexp(ctx.logits[:, :pos_count], dim=1) - math.log(pos_count)
        neg_evidence = torch.logsumexp(ctx.logits[:, pos_count:], dim=1) - math.log(neg_count)
        stacked = torch.stack([pos_evidence, neg_evidence], dim=1)
        probs = torch.softmax(stacked, dim=1)
        frame_pos = probs[:, 0]
        frame_neg = probs[:, 1]
    else:
        frame_pos = ctx.confidence[:, :pos_count].mean(dim=1)
        frame_neg = ctx.confidence[:, pos_count:].mean(dim=1)

    frame_margins = frame_pos - frame_neg
    video_margin = contrast_reduce(frame_margins, options.contrast_reduce)

    if video_margin > options.threshold:
        verdict = "positive"
    elif video_margin < -options.threshold:
        verdict = "negative"
    else:
        verdict = "uncertain"

    mean_pos = frame_pos.mean().item()
    mean_neg = frame_neg.mean().item()

    mean_conf = ctx.confidence.mean(dim=0)
    pos_label_scores = [
        ContrastLabelScore(
            label=ctx.labels[i],
            score=round(mean_conf[i].item(), 6),
        )
        for i in range(pos_count)
    ]
    neg_label_scores = [
        ContrastLabelScore(
            label=ctx.labels[i],
            score=round(mean_conf[i].item(), 6),
        )
        for i in range(pos_count, len(ctx.labels))
    ]

    if verdict == "uncertain":
        dominant_label = None
    elif verdict == "positive":
        dominant_label = max(pos_label_scores, key=lambda s: s.score).label
    else:
        dominant_label = max(neg_label_scores, key=lambda s: s.score).label

    contrast_result = ContrastResult(
        verdict=verdict,
        difference=round(video_margin, 6),
        threshold=options.threshold,
        threshold_was_defaulted=options.threshold_was_defaulted,
        threshold_source=options.threshold_source,
        calibration_status=options.calibration_status,
        contrast_reduce=options.contrast_reduce,
        positive=ContrastGroupResult(
            group="positive",
            mean_group_score=round(mean_pos, 6),
            labels=pos_label_scores,
        ),
        negative=ContrastGroupResult(
            group="negative",
            mean_group_score=round(mean_neg, 6),
            labels=neg_label_scores,
        ),
        score_semantics=ctx.semantics,
        label_pooling=pooling,
        dominant_label=dominant_label,
    )

    base = aggregate_mean(ctx)
    return AggregationResult(
        scores=base.scores,
        best_match=base.best_match,
        contrast=contrast_result,
    )
```

- [ ] **Step 8: Update AggregationResult dataclass**

In `app/services/scoring.py`, update:

```python
@dataclass
class AggregationResult:
    scores: list[ScoreItem]
    best_match: BestMatch
    temporal: "TemporalResult | None" = None
    contrast: "ContrastResult | None" = None
```

- [ ] **Step 9: Wire into aggregate_frame_scores dispatch**

Update the `aggregate_frame_scores` function to handle contrast:

```python
def aggregate_frame_scores(
    batches: list[ScoreBatch],
    labels: list[str],
    frames: list[FrameSample],
    aggregation: str,
    temporal_options: "ResolvedTemporalOptions | None" = None,
    timeline: "FrameTimeline | None" = None,
    policy: "TemporalScoringPolicy | None" = None,
    contrast_options: "ResolvedContrastOptions | None" = None,
    pos_count: int = 0,
) -> AggregationResult:
    ctx = ScoringContext.from_batches(batches, labels, frames)

    if aggregation == "contrast":
        if policy is None or contrast_options is None:
            raise ValueError(
                "Contrast aggregation requires policy and contrast_options"
            )
        return aggregate_contrast(ctx, pos_count, contrast_options, policy)
    elif aggregation == "temporal":
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

- [ ] **Step 10: Run all contrast tests**

Run: `python -m pytest tests/test_scoring.py -k "contrast" -v`
Expected: ALL PASS

- [ ] **Step 11: Run full scoring test suite to check for regressions**

Run: `python -m pytest tests/test_scoring.py -v`
Expected: ALL PASS

- [ ] **Step 12: Commit**

```bash
git add app/services/scoring.py tests/test_scoring.py
git commit -m "feat: add aggregate_contrast with logsumexp pooling and temporal reduction"
```

---

### Task 5: Error Types + API Validation

**Files:**
- Modify: `app/errors/handlers.py`
- Modify: `app/main.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write validation tests**

Add to `tests/test_api.py`:

```python
@pytest.mark.anyio
async def test_contrast_requires_positive_and_negative_labels(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={
            "aggregation": "contrast",
            "positive_labels": json.dumps(["safe driving"]),
        },
    )
    assert r.status_code == 422
    assert "negative_labels" in r.json()["detail"]


@pytest.mark.anyio
async def test_contrast_rejects_labels_field(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={
            "aggregation": "contrast",
            "labels": json.dumps(["a", "b"]),
            "positive_labels": json.dumps(["safe"]),
            "negative_labels": json.dumps(["dangerous"]),
        },
    )
    assert r.status_code == 422


@pytest.mark.anyio
async def test_non_contrast_rejects_positive_labels(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={
            "aggregation": "mean",
            "labels": json.dumps(["a", "b"]),
            "positive_labels": json.dumps(["safe"]),
        },
    )
    assert r.status_code == 422


@pytest.mark.anyio
async def test_contrast_invalid_reduce_mode(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={
            "aggregation": "contrast",
            "positive_labels": json.dumps(["safe"]),
            "negative_labels": json.dumps(["dangerous"]),
            "contrast_reduce": "invalid_mode",
        },
    )
    assert r.status_code == 422


@pytest.mark.anyio
async def test_contrast_cross_group_duplicate_rejected(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={
            "aggregation": "contrast",
            "positive_labels": json.dumps(["driving"]),
            "negative_labels": json.dumps(["driving"]),
        },
    )
    assert r.status_code == 422
    assert "Duplicate" in r.json()["detail"] or "duplicate" in r.json()["detail"]


@pytest.mark.anyio
async def test_contrast_label_count_max_50_per_group(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={
            "aggregation": "contrast",
            "positive_labels": json.dumps([f"pos_{i}" for i in range(51)]),
            "negative_labels": json.dumps(["neg"]),
        },
    )
    assert r.status_code == 422
    assert "50" in r.json()["detail"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api.py -k "contrast" -v`
Expected: FAIL — fields not accepted

- [ ] **Step 3: Add InvalidContrastParamsError**

Add to `app/errors/handlers.py`:

```python
class InvalidContrastParamsError(HTTPException):
    def __init__(self, detail: str) -> None:
        super().__init__(status_code=422, detail=detail)
```

- [ ] **Step 4: Update classify endpoint signature**

In `app/main.py`, change the `classify` function signature to make `labels` optional and add contrast fields:

```python
    @app.post("/api/v1/classify", response_model=ClassifyResponse)
    async def classify(
        video: UploadFile,
        labels: str | None = Form(default=None),
        positive_labels: str | None = Form(default=None),
        negative_labels: str | None = Form(default=None),
        prompt_template: str = Form(default="This is a photo of {}."),
        fps: float = Form(default=1.0),
        aggregation: str = Form(default="mean"),
        threshold: float | None = Form(default=None, ge=0.0, le=1.0),
        gap_tolerance: float | None = Form(default=None, ge=0.0, le=10.0),
        min_duration: float | None = Form(default=None, ge=0.0, le=10.0),
        contrast_reduce: str | None = Form(default=None),
    ):
```

- [ ] **Step 5: Add validation logic after format/fps/aggregation checks**

Update the aggregation check to include `"contrast"`, then add contrast validation. Replace the existing validation block (from the `aggregation not in (...)` check through the label parsing) with:

```python
        if aggregation not in ("mean", "max", "temporal", "contrast"):
            raise InvalidAggregationError(aggregation)

        # Mutual exclusivity: contrast vs standard labels
        if aggregation == "contrast":
            if labels is not None:
                raise InvalidContrastParamsError(
                    "Use 'positive_labels' and 'negative_labels' with aggregation='contrast', not 'labels'."
                )
            if positive_labels is None or negative_labels is None:
                raise InvalidContrastParamsError(
                    "Both 'positive_labels' and 'negative_labels' are required with aggregation='contrast'."
                )
            if contrast_reduce is not None and contrast_reduce not in VALID_CONTRAST_REDUCTIONS:
                raise InvalidContrastParamsError(
                    f"Invalid contrast_reduce '{contrast_reduce}'. "
                    f"Valid: {', '.join(sorted(VALID_CONTRAST_REDUCTIONS))}."
                )
        else:
            if positive_labels is not None or negative_labels is not None:
                raise InvalidContrastParamsError(
                    "'positive_labels' and 'negative_labels' are only valid with aggregation='contrast'."
                )
            if contrast_reduce is not None:
                raise InvalidContrastParamsError(
                    "'contrast_reduce' is only valid with aggregation='contrast'."
                )
            if labels is None:
                raise InvalidLabelsError("labels must be a valid JSON array of strings.")
```

Add the import at the top of `app/main.py`:

```python
from app.errors.handlers import (
    ...
    InvalidContrastParamsError,
)
from app.services.scoring import aggregate_frame_scores, VALID_CONTRAST_REDUCTIONS
```

- [ ] **Step 6: Add label parsing for contrast mode**

After the mutual exclusivity checks, add the contrast label parsing. Restructure the label parsing block:

```python
        # Parse labels based on mode
        pos_count = 0
        if aggregation == "contrast":
            parsed_pos = _parse_label_array(positive_labels, "positive_labels")
            parsed_neg = _parse_label_array(negative_labels, "negative_labels")
            _validate_label_group(parsed_pos, "positive_labels", max_count=50)
            _validate_label_group(parsed_neg, "negative_labels", max_count=50)
            # Cross-group uniqueness
            all_labels_set: set[str] = set()
            for lb in parsed_pos + parsed_neg:
                if lb in all_labels_set:
                    raise InvalidLabelsError(f"Duplicate label across groups: '{lb}'.")
                all_labels_set.add(lb)
            parsed_labels = parsed_pos + parsed_neg
            pos_count = len(parsed_pos)
        else:
            parsed_labels = _parse_label_array(labels, "labels")
            _validate_label_group(parsed_labels, "labels", max_count=50)
```

Add these helper functions inside `create_app` (before the `classify` endpoint) or as module-level helpers:

```python
    def _parse_label_array(raw: str, field_name: str) -> list[str]:
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            raise InvalidLabelsError(f"{field_name} must be a valid JSON array of strings.")
        if not isinstance(parsed, list) or not all(isinstance(lb, str) for lb in parsed):
            raise InvalidLabelsError(f"{field_name} must be a valid JSON array of strings.")
        return parsed

    def _validate_label_group(label_list: list[str], field_name: str, max_count: int = 50) -> None:
        if len(label_list) < 1 or len(label_list) > max_count:
            raise InvalidLabelsError(
                f"Number of {field_name} must be between 1 and {max_count} (inclusive)."
            )
        seen: set[str] = set()
        for lb in label_list:
            if not lb.strip():
                raise InvalidLabelsError("Each label must be a non-empty string.")
            if len(lb) > 200:
                raise InvalidLabelsError(
                    f"Label '{lb[:50]}...' exceeds the maximum length of 200 characters."
                )
            if lb in seen:
                raise InvalidLabelsError(f"Duplicate label: '{lb}'.")
            seen.add(lb)
```

Remove the old inline label validation code (the block from `try: parsed_labels = json.loads(labels)` through the duplicate check). Replace it with the new `_parse_label_array` / `_validate_label_group` calls shown above.

- [ ] **Step 7: Update temporal params validation**

The existing temporal params check must also exclude contrast:

```python
        raw_temporal = RawTemporalParams(
            threshold=threshold if aggregation == "temporal" else None,
            gap_tolerance=gap_tolerance,
            min_duration=min_duration,
        )
        if aggregation not in ("temporal", "contrast") and raw_temporal.has_any():
            raise InvalidTemporalParamsError()
```

Note: `threshold` is shared between temporal and contrast modes, so only pass it to `RawTemporalParams` for temporal. For contrast, it goes to `RawContrastParams` (handled in Task 6).

- [ ] **Step 8: Run validation tests**

Run: `python -m pytest tests/test_api.py -k "contrast" -v`
Expected: ALL PASS

- [ ] **Step 9: Run full API test suite for regressions**

Run: `python -m pytest tests/test_api.py -v`
Expected: ALL PASS (existing tests still work with `labels` being optional but still sent by existing tests)

- [ ] **Step 10: Commit**

```bash
git add app/errors/handlers.py app/main.py tests/test_api.py
git commit -m "feat: add contrast validation — mutual exclusivity, per-group label checks"
```

---

### Task 6: Wire Contrast Into Classify Pipeline

**Files:**
- Modify: `app/main.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write integration test for contrast classify**

Add to `tests/test_api.py`:

```python
@pytest.mark.anyio
async def test_classify_contrast(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={
            "aggregation": "contrast",
            "positive_labels": json.dumps(["outdoor scene", "nature"]),
            "negative_labels": json.dumps(["indoor scene"]),
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["metadata"]["aggregation"] == "contrast"
    assert "contrast" in data
    c = data["contrast"]
    assert c["verdict"] in ("positive", "negative", "uncertain")
    assert "difference" in c
    assert "threshold" in c
    assert c["threshold_was_defaulted"] is True
    assert c["threshold_source"] == "model_policy"
    assert c["calibration_status"] == "uncalibrated"
    assert c["contrast_reduce"] == "mean"
    assert c["positive"]["group"] == "positive"
    assert c["negative"]["group"] == "negative"
    assert len(c["positive"]["labels"]) == 2
    assert len(c["negative"]["labels"]) == 1
    assert c["score_semantics"] == "siglip2_pairwise_sigmoid"
    # Backward compat
    assert "best_match" in data
    assert "scores" in data
    assert len(data["scores"]) == 3


@pytest.mark.anyio
async def test_classify_contrast_with_user_threshold(client, small_video):
    r = await client.post(
        "/api/v1/classify",
        files={"video": ("test.mp4", small_video.read_bytes(), "video/mp4")},
        data={
            "aggregation": "contrast",
            "positive_labels": json.dumps(["outdoor"]),
            "negative_labels": json.dumps(["indoor"]),
            "threshold": "0.30",
            "contrast_reduce": "top_k_mean",
        },
    )
    assert r.status_code == 200
    c = r.json()["contrast"]
    assert c["threshold"] == 0.30
    assert c["threshold_was_defaulted"] is False
    assert c["threshold_source"] == "user"
    assert c["contrast_reduce"] == "top_k_mean"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api.py::test_classify_contrast tests/test_api.py::test_classify_contrast_with_user_threshold -v`
Expected: FAIL — contrast routing not wired yet

- [ ] **Step 3: Wire contrast into the classify pipeline**

In `app/main.py`, add the contrast imports:

```python
from app.schemas.response import (
    ClassifyMetadata,
    ClassifyResponse,
    HealthResponse,
    RawContrastParams,
    RawTemporalParams,
    ReadyResponse,
    ResolvedContrastOptions,
    ResolvedTemporalOptions,
)
```

In the classify endpoint, after inference completes and before `agg_result = aggregate_frame_scores(...)`, add the contrast options resolution:

```python
                        contrast_opts = None
                        if aggregation == "contrast":
                            batch_semantics = all_batches[0].semantics if all_batches else ""
                            policy = get_policy(batch_semantics)
                            raw_contrast = RawContrastParams(
                                threshold=threshold,
                                contrast_reduce=contrast_reduce,
                            )
                            contrast_opts = ResolvedContrastOptions.resolve(
                                raw_contrast,
                                policy.contrast_default_threshold(),
                                policy.contrast_default_reduction(),
                            )
                        elif aggregation == "temporal":
                            batch_semantics = all_batches[0].semantics if all_batches else ""
                            policy = get_policy(batch_semantics)
                            temporal_opts = ResolvedTemporalOptions.resolve(
                                raw_temporal, policy.default_threshold()
                            )
                            timeline = FrameTimeline(all_frames, fps, video_info.duration)
```

Note: This replaces the existing `if aggregation == "temporal"` block.

Update the `aggregate_frame_scores` call:

```python
                    agg_result = aggregate_frame_scores(
                        all_batches, parsed_labels, all_frames, aggregation,
                        temporal_options=temporal_opts,
                        timeline=timeline,
                        policy=policy,
                        contrast_options=contrast_opts,
                        pos_count=pos_count,
                    )
```

Add a contrast disclaimer constant:

```python
DISCLAIMER_CONTRAST = (
    "Contrast verdict is based on group score difference vs threshold. "
    "Model policy defaults are heuristic, not calibrated. "
    "Not suitable for safety-critical decisions."
)
```

Update the disclaimer selection:

```python
                    if aggregation == "contrast":
                        disclaimer = DISCLAIMER_CONTRAST
                    elif aggregation == "max":
                        disclaimer = DISCLAIMER_MAX
                    else:
                        disclaimer = DISCLAIMER_MEAN
```

Update the response construction:

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
                        contrast=agg_result.contrast,
                    )
```

- [ ] **Step 4: Run integration tests**

Run: `python -m pytest tests/test_api.py -k "contrast" -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/test_api.py tests/test_scoring.py tests/test_temporal_policy.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: wire contrast scoring into classify pipeline"
```

---

### Task 7: Active Model Endpoint — Contrast Defaults

**Files:**
- Modify: `app/main.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write test**

Add to `tests/test_api.py`:

```python
@pytest.mark.anyio
async def test_active_model_includes_contrast_defaults(client):
    r = await client.get("/api/v1/models/active")
    assert r.status_code == 200
    data = r.json()
    assert "contrast_defaults" in data
    cd = data["contrast_defaults"]
    assert "threshold" in cd
    assert "contrast_reduce" in cd
    assert "label_pooling" in cd
    assert cd["threshold"] == 0.15  # SigLIP2 default
    assert cd["contrast_reduce"] == "mean"
    assert cd["label_pooling"] == "mean"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api.py::test_active_model_includes_contrast_defaults -v`
Expected: FAIL — `contrast_defaults` not in response

- [ ] **Step 3: Add contrast defaults to active model endpoint**

In `app/main.py`, in the `active_model` endpoint, after the `temporal_defaults` block, add:

```python
        contrast_defaults = None
        if semantics_str:
            try:
                policy = get_policy(semantics_str)
                contrast_defaults = {
                    "threshold": policy.contrast_default_threshold(),
                    "contrast_reduce": policy.contrast_default_reduction(),
                    "label_pooling": policy.contrast_label_pooling(),
                }
            except ValueError:
                pass
```

Note: `policy` may already be assigned from the temporal_defaults block — reuse it or reassign. Add `"contrast_defaults": contrast_defaults` to the return dict.

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_api.py::test_active_model_includes_contrast_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: add contrast defaults to active model endpoint"
```

---

### Task 8: UI — Contrast Mode

**Files:**
- Modify: `app/static/index.html`

This task modifies the single-file UI to support contrast mode: dual label panels, contrast parameter panel, verdict display, and grouped bar chart.

- [ ] **Step 1: Add CSS for contrast UI**

Add before the closing `</style>` tag:

```css
/* Contrast mode */
#contrast-label-section { display: none; }
.contrast-panels { display: flex; gap: 1rem; }
.contrast-panel {
  flex: 1;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  padding: 0.75rem;
}
.contrast-panel.positive { border-left: 3px solid #22c55e; }
.contrast-panel.negative { border-left: 3px solid #ef4444; }
.contrast-panel h4 {
  font-size: 0.85rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
}
.contrast-panel.positive h4 { color: #16a34a; }
.contrast-panel.negative h4 { color: #dc2626; }
.contrast-panel input[type="text"] { margin-bottom: 0.4rem; }
.contrast-chips { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-bottom: 0.4rem; min-height: 1.5rem; }
.contrast-chip {
  padding: 0.2rem 0.5rem;
  border-radius: 999px;
  font-size: 0.78rem;
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
}
.contrast-chip.pos { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
.contrast-chip.neg { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
.contrast-chip .remove-chip {
  cursor: pointer;
  font-weight: 700;
  font-size: 0.7rem;
  opacity: 0.6;
}
.contrast-chip .remove-chip:hover { opacity: 1; }
.contrast-counter { font-size: 0.75rem; color: #9ca3af; text-align: right; }
.contrast-csv-row { display: flex; gap: 0.4rem; align-items: center; }

#contrast-panel {
  display: none;
  background: #f9fafb;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  padding: 0.85rem 1rem;
  margin-bottom: 0.75rem;
}
#contrast-panel h3 {
  font-size: 0.85rem;
  font-weight: 600;
  color: #374151;
  margin-bottom: 0.6rem;
}

/* Verdict badge */
.verdict-banner {
  text-align: center;
  padding: 0.75rem;
  border-radius: 8px;
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  margin-bottom: 0.75rem;
}
.verdict-positive { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
.verdict-negative { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
.verdict-uncertain { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
.verdict-detail {
  font-size: 0.82rem;
  color: #6b7280;
  text-align: center;
  margin-bottom: 0.75rem;
}
.dominant-label {
  font-size: 0.85rem;
  text-align: center;
  margin-bottom: 0.75rem;
  color: #374151;
}
#contrast-results { display: none; }
#contrast-chart-container { height: 300px; margin-bottom: 0.75rem; }
```

- [ ] **Step 2: Add contrast HTML sections**

Add the contrast label section after the existing label section (after `<p class="hint" ...>` and before `<label for="prompt-input">`):

```html
    <!-- Contrast Label Section (shown only in contrast mode) -->
    <div id="contrast-label-section">
      <label>Labels</label>
      <div class="contrast-panels">
        <div class="contrast-panel positive">
          <h4>Positive Labels</h4>
          <input type="text" id="pos-labels-input" placeholder="Comma-separated labels">
          <div class="contrast-csv-row">
            <label class="csv-btn" for="pos-csv-input">Import CSV</label>
            <input type="file" id="pos-csv-input" accept=".csv,.txt" style="display:none">
            <button type="button" id="pos-csv-clear" class="csv-btn danger" style="display:none">Clear</button>
          </div>
          <div class="contrast-chips" id="pos-chips"></div>
          <div class="contrast-counter" id="pos-counter">0 / 50</div>
        </div>
        <div class="contrast-panel negative">
          <h4>Negative Labels</h4>
          <input type="text" id="neg-labels-input" placeholder="Comma-separated labels">
          <div class="contrast-csv-row">
            <label class="csv-btn" for="neg-csv-input">Import CSV</label>
            <input type="file" id="neg-csv-input" accept=".csv,.txt" style="display:none">
            <button type="button" id="neg-csv-clear" class="csv-btn danger" style="display:none">Clear</button>
          </div>
          <div class="contrast-chips" id="neg-chips"></div>
          <div class="contrast-counter" id="neg-counter">0 / 50</div>
        </div>
      </div>
      <p class="hint">Enter positive and negative labels. 1–50 per group.</p>
    </div>
```

Add `"contrast"` option to the aggregation select dropdown:

```html
        <select id="agg-select">
          <option value="mean">mean</option>
          <option value="max">max</option>
          <option value="temporal">temporal</option>
          <option value="contrast">contrast</option>
        </select>
```

Add contrast parameter panel after the temporal panel:

```html
    <!-- Contrast Parameter Panel -->
    <div id="contrast-panel">
      <h3>Contrast Parameters</h3>
      <div class="slider-row">
        <div class="slider-header">
          <label for="contrast-threshold-slider">Contrast Threshold</label>
          <span class="tooltip-icon">?
            <span class="tooltip-card">Minimum margin between positive and negative group scores to render a verdict. Higher = stricter, more uncertain results.</span>
          </span>
          <span class="slider-value" id="contrast-threshold-value">0.15</span>
          <span class="slider-default-tag" id="contrast-threshold-default-tag">(model default)</span>
        </div>
        <input type="range" id="contrast-threshold-slider" min="0" max="1" step="0.01" value="0.15">
      </div>
      <div class="slider-row">
        <div class="slider-header">
          <label for="contrast-reduce-select">Reduction</label>
          <span class="tooltip-icon">?
            <span class="tooltip-card">How per-frame margins are combined. 'mean' averages all frames. 'top_k_mean' focuses on the 10% strongest frames. 'max' takes the single strongest. 'quantile' uses 90th/10th percentile.</span>
          </span>
        </div>
        <select id="contrast-reduce-select" style="margin-bottom:0">
          <option value="mean">mean (whole-video)</option>
          <option value="top_k_mean">top_k_mean (sparse events)</option>
          <option value="max">max (single strongest)</option>
          <option value="quantile">quantile (robust)</option>
        </select>
      </div>
    </div>
```

Add contrast results section inside the results card, after temporal-results and before standard-results:

```html
    <!-- Contrast Results (shown only in contrast mode) -->
    <div id="contrast-results">
      <div class="verdict-banner" id="verdict-banner"></div>
      <div class="verdict-detail" id="verdict-detail"></div>
      <div class="dominant-label" id="dominant-label"></div>
      <div id="contrast-chart-container">
        <canvas id="contrast-chart"></canvas>
      </div>
    </div>
```

- [ ] **Step 3: Add aggregation mode toggle JS**

Update the `aggSelect` change event listener to handle all three panels and label sections:

```javascript
  var standardLabelSection = document.querySelector('.card:nth-child(2) > label:first-of-type');
  var contrastLabelSection = document.getElementById('contrast-label-section');
  var contrastPanel = document.getElementById('contrast-panel');
  var contrastResults = document.getElementById('contrast-results');
  // Wrap existing label elements to show/hide
  var standardLabelEls = [labelChips.parentElement === undefined ? labelChips : labelChips, document.querySelector('.label-row'), labelsHint];

  aggSelect.addEventListener('change', function () {
    var mode = aggSelect.value;
    temporalPanel.style.display = mode === 'temporal' ? 'block' : 'none';
    contrastPanel.style.display = mode === 'contrast' ? 'block' : 'none';

    // Toggle label sections
    if (mode === 'contrast') {
      // Hide standard labels
      labelChips.style.display = 'none';
      document.querySelector('.label-row').style.display = 'none';
      labelsHint.style.display = 'none';
      document.querySelector('.card:nth-child(2) > label:first-of-type').style.display = 'none';
      contrastLabelSection.style.display = 'block';
    } else {
      labelChips.style.display = '';
      document.querySelector('.label-row').style.display = '';
      labelsHint.style.display = '';
      document.querySelector('.card:nth-child(2) > label:first-of-type').style.display = '';
      contrastLabelSection.style.display = 'none';
    }
  });
```

Note: Remove or update the old `aggSelect` event listener.

- [ ] **Step 4: Add contrast label management JS**

Add contrast label state variables and chip rendering:

```javascript
  var posLabels = [];
  var negLabels = [];
  var posCsvLabels = [];
  var negCsvLabels = [];
  var posChips = document.getElementById('pos-chips');
  var negChips = document.getElementById('neg-chips');
  var posCounter = document.getElementById('pos-counter');
  var negCounter = document.getElementById('neg-counter');
  var posInput = document.getElementById('pos-labels-input');
  var negInput = document.getElementById('neg-labels-input');
  var posCsvInput = document.getElementById('pos-csv-input');
  var negCsvInput = document.getElementById('neg-csv-input');
  var posCsvClear = document.getElementById('pos-csv-clear');
  var negCsvClear = document.getElementById('neg-csv-clear');
  var contrastThresholdSlider = document.getElementById('contrast-threshold-slider');
  var contrastThresholdValue = document.getElementById('contrast-threshold-value');
  var contrastThresholdDefaultTag = document.getElementById('contrast-threshold-default-tag');
  var contrastThresholdDirty = false;
  var contrastReduceSelect = document.getElementById('contrast-reduce-select');

  function renderContrastChips(container, labels, cls) {
    while (container.firstChild) container.removeChild(container.firstChild);
    labels.forEach(function (lbl, idx) {
      var chip = document.createElement('span');
      chip.className = 'contrast-chip ' + cls;
      chip.textContent = lbl + ' ';
      var x = document.createElement('span');
      x.className = 'remove-chip';
      x.textContent = '×';
      x.addEventListener('click', function () {
        labels.splice(idx, 1);
        renderContrastChips(container, labels, cls);
        updateContrastCounters();
      });
      chip.appendChild(x);
      container.appendChild(chip);
    });
  }

  function updateContrastCounters() {
    posCounter.textContent = posLabels.length + ' / 50';
    negCounter.textContent = negLabels.length + ' / 50';
  }

  function addContrastLabels(input, labelArr, container, cls) {
    var text = input.value.trim();
    if (!text) return;
    var newLabels = text.split(',').map(function (s) { return s.trim(); }).filter(Boolean);
    newLabels.forEach(function (lbl) {
      if (labelArr.length < 50 && labelArr.indexOf(lbl) === -1) labelArr.push(lbl);
    });
    input.value = '';
    renderContrastChips(container, labelArr, cls);
    updateContrastCounters();
  }

  posInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') { e.preventDefault(); addContrastLabels(posInput, posLabels, posChips, 'pos'); }
  });
  negInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') { e.preventDefault(); addContrastLabels(negInput, negLabels, negChips, 'neg'); }
  });

  function handleContrastCsv(fileInput, labelArr, container, cls, clearBtn) {
    fileInput.addEventListener('change', function () {
      var file = fileInput.files[0];
      if (!file) return;
      var reader = new FileReader();
      reader.onload = function (e) {
        var lines = e.target.result.split(/\r?\n/).map(function (s) { return s.trim(); }).filter(Boolean);
        lines.forEach(function (lbl) {
          if (labelArr.length < 50 && labelArr.indexOf(lbl) === -1) labelArr.push(lbl);
        });
        renderContrastChips(container, labelArr, cls);
        updateContrastCounters();
        clearBtn.style.display = '';
      };
      reader.readAsText(file);
      fileInput.value = '';
    });
    clearBtn.addEventListener('click', function () {
      labelArr.length = 0;
      renderContrastChips(container, labelArr, cls);
      updateContrastCounters();
      clearBtn.style.display = 'none';
    });
  }

  handleContrastCsv(posCsvInput, posLabels, posChips, 'pos', posCsvClear);
  handleContrastCsv(negCsvInput, negLabels, negChips, 'neg', negCsvClear);

  contrastThresholdSlider.addEventListener('input', function () {
    contrastThresholdValue.textContent = parseFloat(contrastThresholdSlider.value).toFixed(2);
    if (!contrastThresholdDirty) {
      contrastThresholdDirty = true;
      contrastThresholdDefaultTag.style.display = 'none';
    }
  });
```

- [ ] **Step 5: Update classify button to build contrast FormData**

In the `classifyBtn` click handler, update the label collection and form building. Replace the existing label collection block with a mode-aware version:

```javascript
    var isContrast = aggSelect.value === 'contrast';

    if (isContrast) {
      addContrastLabels(posInput, posLabels, posChips, 'pos');
      addContrastLabels(negInput, negLabels, negChips, 'neg');
      if (posLabels.length < 1 || negLabels.length < 1) {
        classifyError.textContent = 'Provide at least 1 positive and 1 negative label.';
        return;
      }
      if (posLabels.length > 50 || negLabels.length > 50) {
        classifyError.textContent = 'Maximum 50 labels per group.';
        return;
      }
    } else {
      // Keep the existing chipLabels + customLabels + dedup logic unchanged.
      // This is the block from `var chipLabels = ...` through `if (labels.length < 1 || labels.length > 50)`.
      var chipLabels = defaultLabels.concat(csvLabels).filter(function (lbl) {
        return selectedLabels[lbl];
      });
      var customLabels = labelsInput.value.trim()
        ? labelsInput.value.split(',').map(function (s) { return s.trim(); }).filter(Boolean)
        : [];
      var seen = {};
      var labels = chipLabels.concat(customLabels).filter(function (lbl) {
        if (seen[lbl]) return false;
        seen[lbl] = true;
        return true;
      });
      if (labels.length < 1 || labels.length > 50) {
        classifyError.textContent = 'Please provide between 1 and 50 labels.';
        return;
      }
    }

    var form = new FormData();
    form.append('video', file);
    form.append('prompt_template', promptInput.value);
    form.append('fps', fpsInput.value);
    form.append('aggregation', aggSelect.value);

    if (isContrast) {
      form.append('positive_labels', JSON.stringify(posLabels));
      form.append('negative_labels', JSON.stringify(negLabels));
      if (contrastThresholdDirty) {
        form.append('threshold', contrastThresholdSlider.value);
      }
      form.append('contrast_reduce', contrastReduceSelect.value);
    } else {
      form.append('labels', JSON.stringify(labels));
      if (aggSelect.value === 'temporal') {
        if (thresholdDirty) form.append('threshold', thresholdSlider.value);
        form.append('gap_tolerance', gapSlider.value);
        form.append('min_duration', mindurSlider.value);
      }
    }
```

- [ ] **Step 6: Update renderResults for contrast mode**

In the `renderResults` function, add contrast handling:

```javascript
  function renderResults(data) {
    if (activeChart) {
      try { activeChart.destroy(); } catch (e) {}
      activeChart = null;
    }

    var isTemporal = data.temporal && aggSelect.value === 'temporal';
    var isContrast = data.contrast && aggSelect.value === 'contrast';

    temporalResults.style.display = 'none';
    contrastResults.style.display = 'none';
    standardResults.style.display = 'none';

    if (isContrast) {
      contrastResults.style.display = 'block';
      renderContrast(data.contrast);
    } else if (isTemporal) {
      temporalResults.style.display = 'block';
      renderTemporal(data.temporal, data.scores);
    } else {
      standardResults.style.display = 'block';
      renderStandard(data);
    }

    // Keep existing metadata rendering below (the `var m = data.metadata; ...` block through `resultsSection.style.display = 'block';`)
    var m = data.metadata;
    while (metaInfo.firstChild) { metaInfo.removeChild(metaInfo.firstChild); }
    metaInfo.appendChild(document.createTextNode(
      'Frames: ' + m.frames_analyzed +
      ' | Duration: ' + m.video_duration_seconds.toFixed(1) + 's' +
      ' | Model: ' + m.model +
      ' | Device: ' + m.device +
      ' | Aggregation: ' + m.aggregation +
      ' | Time: ' + m.processing_time_seconds.toFixed(2) + 's'
    ));
    resultsSection.style.display = 'block';
  }
```

- [ ] **Step 7: Implement renderContrast function**

```javascript
  function renderContrast(contrast) {
    var banner = document.getElementById('verdict-banner');
    var detail = document.getElementById('verdict-detail');
    var domLabel = document.getElementById('dominant-label');

    // Verdict banner
    while (banner.firstChild) banner.removeChild(banner.firstChild);
    banner.textContent = contrast.verdict.toUpperCase();
    banner.className = 'verdict-banner verdict-' + contrast.verdict;

    // Detail line
    while (detail.firstChild) detail.removeChild(detail.firstChild);
    detail.textContent =
      'Positive: ' + contrast.positive.mean_group_score.toFixed(3) +
      ' | Negative: ' + contrast.negative.mean_group_score.toFixed(3) +
      ' | Margin: ' + contrast.difference.toFixed(3) +
      ' | Threshold: ±' + contrast.threshold.toFixed(2) +
      ' | Reduce: ' + contrast.contrast_reduce;

    // Dominant label
    while (domLabel.firstChild) domLabel.removeChild(domLabel.firstChild);
    if (contrast.dominant_label) {
      var strong = document.createElement('strong');
      strong.textContent = 'Dominant: ';
      domLabel.appendChild(strong);
      domLabel.appendChild(document.createTextNode(contrast.dominant_label));
    }

    // Grouped horizontal bar chart
    var chartContainer = document.getElementById('contrast-chart-container');
    while (chartContainer.firstChild) chartContainer.removeChild(chartContainer.firstChild);
    var canvas = document.createElement('canvas');
    canvas.id = 'contrast-chart';
    chartContainer.appendChild(canvas);

    if (typeof Chart === 'undefined') return;

    var posLabels = contrast.positive.labels.slice().sort(function (a, b) { return b.score - a.score; });
    var negLabels = contrast.negative.labels.slice().sort(function (a, b) { return b.score - a.score; });
    var allLabels = posLabels.concat(negLabels);
    var chartLabels = allLabels.map(function (l) { return l.label; });
    var chartData = allLabels.map(function (l) { return parseFloat((l.score * 100).toFixed(2)); });
    var chartColors = posLabels.map(function () { return '#22c55e'; })
      .concat(negLabels.map(function () { return '#ef4444'; }));

    activeChart = new Chart(canvas, {
      type: 'bar',
      data: {
        labels: chartLabels,
        datasets: [{
          data: chartData,
          backgroundColor: chartColors,
          borderWidth: 0,
          barThickness: 18
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: {
          x: {
            min: 0,
            max: 100,
            ticks: { font: { size: 11 }, callback: function (v) { return v + '%'; } }
          },
          y: {
            ticks: { font: { size: 11 } }
          }
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) { return ctx.parsed.x.toFixed(1) + '%'; }
            }
          }
        }
      }
    });
  }
```

- [ ] **Step 8: Update fetchTemporalDefaults to also set contrast defaults**

In `fetchTemporalDefaults`, after setting temporal defaults, also set contrast defaults:

```javascript
  function fetchTemporalDefaults() {
    fetch('/api/v1/models/active')
      .then(function (r) { if (!r.ok) throw new Error('no active model'); return r.json(); })
      .then(function (data) {
        temporalDefaults = data.temporal_defaults || null;
        applyTemporalDefaults();
        // Contrast defaults
        var cd = data.contrast_defaults || null;
        if (cd) {
          contrastThresholdSlider.value = cd.threshold;
          contrastThresholdValue.textContent = cd.threshold.toFixed(2);
          contrastThresholdDirty = false;
          contrastThresholdDefaultTag.style.display = '';
        }
      })
      .catch(function () {
        temporalDefaults = null;
      });
  }
```

Also reset contrast dirty state when loading a new model (in `loadBtn` click handler):

```javascript
        contrastThresholdDirty = false;
        contrastThresholdDefaultTag.style.display = '';
```

- [ ] **Step 9: Test manually in browser**

Run: `ALLOW_UNAUTHENTICATED=true uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000 --reload`

Test:
1. Select "contrast" in aggregation dropdown → verify dual label panels appear, standard labels hide
2. Add positive and negative labels via text input and CSV
3. Verify chip display, counter, removal
4. Upload a video and classify → verify verdict banner, bar chart, detail line
5. Switch back to "mean" → verify standard labels reappear
6. Verify temporal mode still works normally

- [ ] **Step 10: Commit**

```bash
git add app/static/index.html
git commit -m "feat: add contrast mode UI — dual label panels, verdict display, bar chart"
```

---

### Task Summary

| Task | Description | Dependencies |
|------|-------------|-------------|
| 1 | Response Schema | none |
| 2 | Contrast Policy | none |
| 3 | Temporal Reduction Functions | none |
| 4 | aggregate_contrast() | 1, 2, 3 |
| 5 | Error Types + API Validation | none |
| 6 | Wire Contrast Into Pipeline | 4, 5 |
| 7 | Active Model Endpoint | 2 |
| 8 | UI | 6, 7 |

Tasks 1, 2, 3, and 5 can run in parallel. Task 4 depends on 1+2+3. Task 6 depends on 4+5. Task 7 depends on 2. Task 8 depends on 6+7.
