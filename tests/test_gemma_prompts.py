from app.errors.handlers import GemmaOutputParseError, InvalidGemmaParamsError


def test_gemma_output_parse_error_is_502():
    err = GemmaOutputParseError("bad json")
    assert err.status_code == 502
    assert "bad json" in err.detail


def test_invalid_gemma_params_error_is_422():
    err = InvalidGemmaParamsError("prompt too long")
    assert err.status_code == 422
    assert err.detail == "prompt too long"
