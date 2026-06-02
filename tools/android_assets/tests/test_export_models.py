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
