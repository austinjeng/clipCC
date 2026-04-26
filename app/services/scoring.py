import torch
from app.schemas.response import BestMatch, ScoreItem
from app.services.video import FrameSample


def compute_frame_scores(
    cosine_sim: torch.Tensor, logit_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_similarity = cosine_sim.clone()
    scaled_logits = cosine_sim * logit_scale
    confidence = torch.softmax(scaled_logits, dim=-1)
    return confidence, raw_similarity


def aggregate_mean(
    confidence: torch.Tensor, raw_sim: torch.Tensor,
    labels: list[str], frames: list[FrameSample],
) -> list[ScoreItem]:
    mean_conf = confidence.mean(dim=0)
    mean_raw = raw_sim.mean(dim=0)
    return [
        ScoreItem(label=labels[i], confidence=round(mean_conf[i].item(), 6),
                  raw_similarity=round(mean_raw[i].item(), 6))
        for i in range(len(labels))
    ]


def aggregate_max(
    confidence: torch.Tensor, raw_sim: torch.Tensor,
    labels: list[str], frames: list[FrameSample],
) -> list[ScoreItem]:
    max_conf, max_indices = confidence.max(dim=0)
    return [
        ScoreItem(
            label=labels[i], confidence=round(max_conf[i].item(), 6),
            raw_similarity=round(raw_sim[max_indices[i].item(), i].item(), 6),
            peak_frame_index=max_indices[i].item(),
            approx_timestamp_seconds=frames[max_indices[i].item()].approx_timestamp_seconds,
        )
        for i in range(len(labels))
    ]


def build_response_scores(
    confidence: torch.Tensor, raw_sim: torch.Tensor,
    labels: list[str], frames: list[FrameSample], aggregation: str,
) -> tuple[list[ScoreItem], BestMatch]:
    if aggregation == "max":
        scores = aggregate_max(confidence, raw_sim, labels, frames)
    else:
        scores = aggregate_mean(confidence, raw_sim, labels, frames)
    best = max(scores, key=lambda s: s.confidence)
    best_match = BestMatch(label=best.label, confidence=best.confidence)
    return scores, best_match
