from app.errors.handlers import GemmaOutputParseError, InvalidGemmaParamsError


def test_gemma_output_parse_error_is_502():
    err = GemmaOutputParseError("bad json")
    assert err.status_code == 502
    assert "bad json" in err.detail


def test_invalid_gemma_params_error_is_422():
    err = InvalidGemmaParamsError("prompt too long")
    assert err.status_code == 422
    assert err.detail == "prompt too long"


import pytest

from app.services.gemma_prompts import (
    build_label_scores_prompt,
    build_qa_prompt,
    label_scores_token_budget,
    parse_label_scores,
)

LABELS = ["texting while driving", "sleeping", "eating"]


def test_prompt_numbers_labels_from_1():
    p = build_label_scores_prompt(LABELS, evidence_top_k=3)
    assert "1: texting while driving" in p
    assert "3: eating" in p
    assert '"id"' in p and '"score"' in p


def test_token_budget_scales_with_label_count():
    assert label_scores_token_budget(1) < label_scores_token_budget(16)
    assert label_scores_token_budget(16) <= 640


def test_parse_happy_path():
    text = '[{"id": 1, "score": 0.9, "evidence": "phone visible"}, {"id": 2, "score": 0.1}, {"id": 3, "score": 0.2}]'
    items = parse_label_scores(text, LABELS)
    assert items[0].label == "texting while driving"
    assert items[0].score == 0.9
    assert items[0].evidence == "phone visible"
    assert items[1].evidence is None


def test_parse_strips_code_fences():
    text = '```json\n[{"id": 1, "score": 0.5}]\n```'
    items = parse_label_scores(text, ["only label"])
    assert items[0].score == 0.5


def test_parse_tolerates_prose_preamble():
    text = 'Sure! Here is the result:\n```json\n[{"id": 1, "score": 0.5}]\n```'
    items = parse_label_scores(text, ["only label"])
    assert items[0].score == 0.5


def test_parse_tolerates_space_after_fence():
    text = '``` json\n[{"id": 1, "score": 0.5}]\n```'
    items = parse_label_scores(text, ["only label"])
    assert items[0].score == 0.5


def test_parse_unknown_id_rejected():
    with pytest.raises(ValueError):
        parse_label_scores('[{"id": 9, "score": 0.5}]', LABELS)


def test_parse_duplicate_id_rejected():
    with pytest.raises(ValueError):
        parse_label_scores('[{"id": 1, "score": 0.5}, {"id": 1, "score": 0.6}]', LABELS)


def test_parse_score_out_of_range_rejected():
    with pytest.raises(ValueError):
        parse_label_scores('[{"id": 1, "score": 1.5}]', LABELS)


def test_parse_score_nan_rejected():
    with pytest.raises(ValueError):
        parse_label_scores('[{"id": 1, "score": NaN}]', LABELS)


def test_parse_score_as_string_rejected():
    with pytest.raises(ValueError):
        parse_label_scores('[{"id": 1, "score": "0.5"}]', LABELS)


def test_parse_score_as_bool_rejected():
    with pytest.raises(ValueError):
        parse_label_scores('[{"id": 1, "score": true}]', LABELS)


def test_parse_missing_ids_yield_null_scores():
    items = parse_label_scores('[{"id": 2, "score": 0.7}]', LABELS)
    by_label = {i.label: i.score for i in items}
    assert by_label["sleeping"] == 0.7
    assert by_label["texting while driving"] is None
    assert by_label["eating"] is None


def test_parse_non_list_rejected():
    with pytest.raises(ValueError):
        parse_label_scores('{"id": 1, "score": 0.5}', LABELS)


def test_qa_prompt_embeds_user_text():
    p = build_qa_prompt("what is the driver doing?")
    assert "what is the driver doing?" in p
