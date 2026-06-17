import pytest

from app.services.gemma_prompts import build_verdict_prompt, parse_verdict, VERDICT_LITERALS


def test_prompt_names_label_and_demands_json_object():
    p = build_verdict_prompt("texting while driving")
    assert "texting while driving" in p
    assert "JSON object" in p
    assert '"verdict"' in p and '"explanation"' in p


def test_prompt_uses_custom_instruction():
    p = build_verdict_prompt("eating", instruction="Look only at the hands.")
    assert "Look only at the hands." in p


def test_parse_valid_object():
    out = parse_verdict('{"verdict": "present", "explanation": "phone in hand"}')
    assert out == {"verdict": "present", "explanation": "phone in hand"}


def test_parse_tolerates_prose_and_fences():
    out = parse_verdict('Sure!\n```json\n{"verdict":"not_present","explanation":"nothing seen"}\n```')
    assert out["verdict"] == "not_present"


def test_parse_rejects_unknown_verdict():
    with pytest.raises(ValueError):
        parse_verdict('{"verdict": "maybe", "explanation": "x"}')


def test_parse_rejects_missing_explanation():
    with pytest.raises(ValueError):
        parse_verdict('{"verdict": "present"}')


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_verdict("not json at all")


def test_literals_are_the_three_values():
    assert VERDICT_LITERALS == ("present", "not_present", "uncertain")
