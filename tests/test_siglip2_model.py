import pytest
import torch
from PIL import Image
from app.models.base_model import BaseModel, ScoreBatch
from app.models.siglip2_model import SigLip2Model


@pytest.fixture
def dummy_images():
    return [Image.new("RGB", (256, 256), color=(i * 30, 100, 200)) for i in range(3)]


@pytest.fixture
def model(temp_dir):
    return SigLip2Model(
        hf_repo="google/siglip2-base-patch16-256",
        cache_dir=str(temp_dir / "models"),
    )


class TestSigLip2ModelInterface:
    def test_is_base_model(self, model):
        assert isinstance(model, BaseModel)
        assert model.model_type == "siglip2"
        assert model.max_token_length == 64
        assert model.device in ("cpu", "cuda")

    def test_score_batch_shape(self, model, dummy_images):
        texts = ["a cat", "a dog", "a bird"]
        batch = model.score_batch(dummy_images, texts)
        assert isinstance(batch, ScoreBatch)
        assert batch.confidence.shape == (3, 3)
        assert batch.raw_similarity.shape == (3, 3)
        assert batch.logits.shape == (3, 3)
        assert batch.semantics == "siglip2_pairwise_sigmoid"

    def test_score_batch_uses_sigmoid(self, model, dummy_images):
        texts = ["a cat", "a dog", "a bird"]
        batch = model.score_batch(dummy_images, texts)
        assert (batch.confidence >= 0).all()
        assert (batch.confidence <= 1).all()

    def test_encode_images(self, model, dummy_images):
        features = model.encode_images(dummy_images)
        assert features.shape[0] == 3
        assert features.shape[1] > 0
        norms = features.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_validate_prompts_normal(self, model):
        counts = model.validate_prompts(["this is a photo of a cat", "this is a photo of a dog"])
        assert len(counts) == 2
        assert all(0 < c <= 64 for c in counts)

    def test_validate_prompts_overflow(self, model):
        long_prompt = "this is a photo of " + "very " * 100 + "long description"
        counts = model.validate_prompts([long_prompt])
        assert counts[0] > 64

    def test_tokenize_raw_lowercases(self, model):
        upper = model.tokenize_raw(["A Cat"])
        lower = model.tokenize_raw(["a cat"])
        assert torch.equal(upper[0], lower[0])

    def test_tokenize_for_inference_shape(self, model):
        result = model.tokenize_for_inference(["a cat", "a dog"])
        assert "input_ids" in result
        assert result["input_ids"].shape[1] == 64
