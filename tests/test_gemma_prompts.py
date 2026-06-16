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
    # 50 labels (the cap) must fit un-truncated: 64 + 24*50 = 1264
    assert label_scores_token_budget(50) == 1264
    assert label_scores_token_budget(100) <= 1280


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


def test_default_instruction_used_when_omitted():
    from app.services.gemma_prompts import DEFAULT_LABEL_SCORES_INSTRUCTION

    p = build_label_scores_prompt(LABELS, evidence_top_k=3)
    assert DEFAULT_LABEL_SCORES_INSTRUCTION in p
    # explicit None must match the omitted default exactly (backward compatible)
    assert build_label_scores_prompt(LABELS, evidence_top_k=3, instruction=None) == p


def test_custom_instruction_replaces_preamble_but_keeps_contract():
    from app.services.gemma_prompts import DEFAULT_LABEL_SCORES_INSTRUCTION

    custom = "Be a harsh, skeptical judge of each behavior."
    p = build_label_scores_prompt(LABELS, evidence_top_k=3, instruction=custom)
    assert custom in p
    assert DEFAULT_LABEL_SCORES_INSTRUCTION not in p
    # behaviors block + JSON contract are appended regardless of the instruction
    assert "1: texting while driving" in p
    assert '"id"' in p and "Include every id exactly once." in p


def test_blank_instruction_falls_back_to_default():
    p_blank = build_label_scores_prompt(LABELS, evidence_top_k=3, instruction="   ")
    p_default = build_label_scores_prompt(LABELS, evidence_top_k=3)
    assert p_blank == p_default


def test_label_scores_contract_is_the_locked_json_block():
    from app.services.gemma_prompts import label_scores_contract

    c = label_scores_contract(3)
    assert "Respond with ONLY a JSON array" in c
    assert "3 highest-scoring" in c
    assert '"id"' in c and '"score"' in c
    assert "Include every id exactly once." in c


def test_builder_ends_with_the_contract():
    from app.services.gemma_prompts import label_scores_contract

    # the exposed contract must be exactly the tail of the real prompt so a
    # client-side preview can reconstruct the final message without drift
    p = build_label_scores_prompt(LABELS, evidence_top_k=3)
    assert p.endswith(label_scores_contract(3))
