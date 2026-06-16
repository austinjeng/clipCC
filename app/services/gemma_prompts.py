from __future__ import annotations

import json
import re

from app.schemas.gemma import GemmaScoreItem

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)

# Token budget: fixed overhead + per-label allowance ({"id": N, "score": 0.XX},
# plus evidence strings for the top-k items).
_BUDGET_BASE = 64
_BUDGET_PER_LABEL = 24
# Ceiling sized to the 50-label cap: 64 + 24*50 = 1264 must fit un-truncated
_BUDGET_MAX = 1280


def label_scores_token_budget(n_labels: int) -> int:
    return min(_BUDGET_BASE + _BUDGET_PER_LABEL * n_labels, _BUDGET_MAX)


# The default leading instruction. Users may override it (label-scores mode),
# but the behaviors block and JSON contract below are always appended.
DEFAULT_LABEL_SCORES_INSTRUCTION = (
    "You are analyzing frames sampled from a video, in chronological order.\n"
    "Score how strongly each numbered behavior below is visible anywhere in these frames."
)


def label_scores_contract(evidence_top_k: int) -> str:
    # The locked JSON-output contract appended after the (editable) instruction
    # and the behaviors block. Single source of truth so the UI can preview the
    # exact final prompt without drifting from what is actually sent.
    return (
        "Respond with ONLY a JSON array, no other text. One object per behavior id:\n"
        '[{"id": <behavior number>, "score": <number from 0.0 to 1.0>}]\n'
        f"For the {evidence_top_k} highest-scoring behaviors only, add an "
        '"evidence" field: one short sentence describing what you saw.\n'
        "Scores must be JSON numbers, not strings. Include every id exactly once."
    )


def build_label_scores_prompt(
    labels: list[str], evidence_top_k: int, instruction: str | None = None
) -> str:
    # Blank/None instruction falls back to the default, so the default prompt is
    # byte-identical to before. Only the leading instruction is user-editable;
    # the behaviors block and the strict JSON contract are always appended.
    instr = (instruction or "").strip() or DEFAULT_LABEL_SCORES_INSTRUCTION
    numbered = "\n".join(f"{i + 1}: {label}" for i, label in enumerate(labels))
    return (
        f"{instr}\n\n"
        f"Behaviors:\n{numbered}\n\n"
        f"{label_scores_contract(evidence_top_k)}"
    )


def build_qa_prompt(user_prompt: str) -> str:
    return (
        "You are analyzing frames sampled from a video, in chronological order. "
        "Answer the following question about the video concisely.\n\n"
        f"Question: {user_prompt}"
    )


def parse_label_scores(text: str, labels: list[str]) -> list[GemmaScoreItem]:
    """Strict ID-keyed parse (spec §4/§5): reject unknown ids, duplicate ids,
    non-numeric or out-of-range scores. Missing ids become score=None.
    Raises ValueError on any violation (route does one bounded retry)."""
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    # Tolerate prose preamble/postamble around the array — slice to the
    # outermost brackets so a structurally-correct output doesn't burn a retry.
    start, end = cleaned.find("["), cleaned.rfind("]")
    if start != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON: {e}") from e
    if not isinstance(data, list):
        raise ValueError("expected a JSON array")

    by_id: dict[int, dict] = {}
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("array items must be objects")
        item_id = item.get("id")
        if not isinstance(item_id, int) or isinstance(item_id, bool):
            raise ValueError(f"id must be an integer, got {item_id!r}")
        if item_id < 1 or item_id > len(labels):
            raise ValueError(f"unknown id {item_id} (valid: 1..{len(labels)})")
        if item_id in by_id:
            raise ValueError(f"duplicate id {item_id}")
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"score for id {item_id} must be a JSON number, got {score!r}")
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"score {score} for id {item_id} out of range [0, 1]")
        evidence = item.get("evidence")
        if evidence is not None and not isinstance(evidence, str):
            raise ValueError(f"evidence for id {item_id} must be a string")
        by_id[item_id] = {"score": float(score), "evidence": evidence}

    return [
        GemmaScoreItem(
            label=label,
            score=by_id.get(i + 1, {}).get("score"),
            evidence=by_id.get(i + 1, {}).get("evidence"),
        )
        for i, label in enumerate(labels)
    ]
