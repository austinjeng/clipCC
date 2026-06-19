from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.services.scoring import ScoringContext


@dataclass
class FrameRef:
    frame_index: int
    timestamp_seconds: float
    score: float


@dataclass
class GatedLabel:
    label: str
    label_idx: int
    score: float


def per_label_scores(ctx: ScoringContext, aggregation: str) -> list[float]:
    """One score per label used for gating and display."""
    if aggregation == "max":
        vals = ctx.confidence.max(dim=0).values
    elif aggregation == "mean":
        vals = ctx.confidence.mean(dim=0)
    else:
        raise ValueError(f"hybrid aggregation must be 'max' or 'mean', got {aggregation!r}")
    return [round(v.item(), 6) for v in vals]


def gate_and_rank_labels(
    ctx: ScoringContext, scores: list[float], threshold: float, max_verified_labels: int
) -> tuple[list[GatedLabel], int, int]:
    """Keep labels at/above threshold, ranked by score, capped to max_verified_labels.
    Returns (selected, n_above_threshold, n_truncated)."""
    above = [
        GatedLabel(label=ctx.labels[i], label_idx=i, score=scores[i])
        for i in range(len(ctx.labels))
        if scores[i] >= threshold
    ]
    above.sort(key=lambda g: g.score, reverse=True)
    selected = above[:max_verified_labels]
    return selected, len(above), len(above) - len(selected)


def select_topk_spread(
    ctx: ScoringContext, label_idx: int, k: int, min_gap_seconds: float = 0.5
) -> list[FrameRef]:
    """Top-k frames for one label by score, rejecting any frame within
    min_gap_seconds of an already-picked frame. Returned ranked best-first;
    the caller sorts chronologically before sending to Gemma."""
    col = ctx.confidence[:, label_idx]
    order = col.argsort(descending=True).tolist()
    picked: list[FrameRef] = []
    for fi in order:
        if len(picked) >= k:
            break
        ts = ctx.frames[fi].approx_timestamp_seconds
        if all(abs(ts - p.timestamp_seconds) >= min_gap_seconds for p in picked):
            picked.append(FrameRef(frame_index=fi, timestamp_seconds=ts, score=round(col[fi].item(), 6)))
    return picked


def thumbnail_data_uri(path: Path, max_px: int) -> str:
    """Downscale a frame to a small base64 JPEG data URI for inline display."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
