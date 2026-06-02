import json
from tools.android_assets.manifest import ModelBundleManifest, FileRef, SCHEMA_VERSION


def _sample() -> ModelBundleManifest:
    return ModelBundleManifest(
        model_id="siglip2-base-patch16-256",
        hf_repo="google/siglip2-base-patch16-256",
        hf_revision="a" * 40,
        onnx_source="onnx-community",
        onnx_source_repo="onnx-community/siglip2-base-patch16-256-ONNX",
        onnx_source_revision="b" * 40,
        display_name="SigLIP2 Base (256px)",
        params="0.4B",
        resolution=256,
        precision="fp32",
        ram_budget_mb=1600,
        transformers_version="5.0.0",
        logit_scale=4.7654,
        logit_bias=-16.53,
        vision=FileRef(file="vision_model.onnx", data_file=None, bytes=10, sha256="x", data_sha256=None),
        text=FileRef(file="text_model.onnx", data_file="text_model.onnx_data", bytes=10, sha256="y", data_sha256="yy"),
        tokenizer_sha256="z",
    )


def test_round_trip_preserves_fields_and_schema_version():
    m = _sample()
    blob = m.to_json()
    parsed = json.loads(blob)
    assert parsed["schema_version"] == SCHEMA_VERSION == 1
    assert parsed["profile"] == "benchmark-v1"
    assert parsed["score_semantics"] == "siglip2_pairwise_sigmoid"
    assert parsed["preprocess"]["resample"] == "bicubic"
    assert parsed["frame_pipeline"]["prescale"] == "none"
    assert parsed["tokenizer"]["lowercase_applied_by"] == "unknown"
    assert parsed["tokenizer"]["padding_side"] == "right"
    assert parsed["ram_budget_mb"] == 1600
    assert parsed["onnx_source_revision"] == "b" * 40
    assert parsed["text"]["data_sha256"] == "yy"
    back = ModelBundleManifest.from_json(blob)
    assert back == m


def test_from_json_rejects_wrong_schema_version():
    blob = _sample().to_json().replace('"schema_version": 1', '"schema_version": 99')
    try:
        ModelBundleManifest.from_json(blob)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "schema_version" in str(e)
