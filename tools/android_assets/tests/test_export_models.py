from tools.android_assets.export_models import onnx_community_repo, PROFILE_MODELS


def test_profile_has_exactly_four_models():
    assert list(PROFILE_MODELS.keys()) == [
        "siglip2-base-patch16-256",
        "siglip2-base-patch16-384",
        "siglip2-large-patch16-384",
        "siglip2-so400m-patch14-384",
    ]


def test_onnx_community_repo_maps_google_repo():
    assert (
        onnx_community_repo("google/siglip2-base-patch16-256")
        == "onnx-community/siglip2-base-patch16-256-ONNX"
    )
    assert (
        onnx_community_repo("google/siglip2-so400m-patch14-384")
        == "onnx-community/siglip2-so400m-patch14-384-ONNX"
    )


import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
from tools.android_assets.export_models import to_fp16, save_onnx


def _identity_fp32_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    w = helper.make_tensor("w", TensorProto.FLOAT, [4], np.ones(4, np.float32))
    node = helper.make_node("Mul", ["x", "w"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y], initializer=[w])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_to_fp16_runs_and_keeps_shape(tmp_path):
    m32 = _identity_fp32_model()
    m16 = to_fp16(m32)
    p = tmp_path / "m16.onnx"
    save_onnx(m16, p)
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"x": np.array([[1, 2, 3, 4]], np.float32)})[0]
    assert out.shape == (1, 4)


def test_save_onnx_external_data_when_forced(tmp_path):
    m = _identity_fp32_model()
    p = tmp_path / "ext.onnx"
    save_onnx(m, p, force_external=True)
    assert p.exists()
    assert (tmp_path / "ext.onnx_data").exists()


import torch
from tools.android_assets.export_models import extract_logit_params


class _StubModel:
    def __init__(self):
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.7654))
        self.logit_bias = torch.nn.Parameter(torch.tensor(-16.53))


def test_extract_logit_params_reads_raw_scalars():
    scale, bias = extract_logit_params(_StubModel())
    assert round(scale, 4) == 4.7654
    assert round(bias, 2) == -16.53
