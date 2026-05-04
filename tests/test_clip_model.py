import pytest
import torch
from PIL import Image
from app.models.clip_model import ClipModel
from app.models.model_spec import ModelSpec

@pytest.fixture
def dummy_images():
    return [Image.new("RGB", (224, 224), color=(i * 30, 100, 200)) for i in range(3)]

@pytest.fixture
def model(temp_dir):
    spec = ModelSpec(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", cache_dir=str(temp_dir / "models"))
    return ClipModel(spec)

def test_load_model(model):
    assert model.model is not None
    assert model.preprocess is not None
    assert model.tokenizer is not None
    assert model.device in ("cpu", "cuda")

def test_encode_text(model):
    features = model.encode_text(["a photo of a cat", "a photo of a dog"])
    assert features.shape[0] == 2
    assert features.shape[1] > 0

def test_encode_images(model, dummy_images):
    features = model.encode_images(dummy_images)
    assert features.shape[0] == 3

def test_compute_similarities(model, dummy_images):
    texts = ["red image", "green image", "blue image"]
    similarities, logit_scale = model.compute_similarities(dummy_images, texts)
    assert similarities.shape == (3, 3)
    assert logit_scale > 0

def test_tokenize_and_check(model):
    prompts = ["a video of driving", "a video of parking"]
    token_counts = model.tokenize_and_check(prompts, max_tokens=77)
    assert len(token_counts) == 2
    assert all(0 < c <= 77 for c in token_counts)

def test_tokenize_detects_long_prompt(model):
    long_prompt = "a video of " + "very " * 100 + "long description"
    token_counts = model.tokenize_and_check([long_prompt], max_tokens=77)
    assert token_counts[0] > 77

def test_tokenize_raw(model):
    prompts = ["a video of driving", "a video of parking"]
    result = model.tokenize_raw(prompts)
    assert len(result) == 2
    assert all(isinstance(t, torch.Tensor) for t in result)


from app.models.base_model import BaseModel, ScoreBatch


def test_clip_model_is_base_model(model):
    assert isinstance(model, BaseModel)
    assert model.model_type == "clip"
    assert model.max_token_length == 77


def test_score_batch(model, dummy_images):
    texts = ["red image", "green image", "blue image"]
    batch = model.score_batch(dummy_images, texts)
    assert isinstance(batch, ScoreBatch)
    assert batch.confidence.shape == (3, 3)
    assert batch.raw_similarity.shape == (3, 3)
    assert batch.logits.shape == (3, 3)
    assert batch.semantics == "clip_relative_softmax"
    # Softmax rows should sum to ~1
    row_sums = batch.confidence.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(3), atol=1e-5)


def test_validate_prompts(model):
    counts = model.validate_prompts(["a video of driving", "a video of parking"])
    assert len(counts) == 2
    assert all(0 < c <= 77 for c in counts)


def test_validate_prompts_detects_overflow(model):
    long_prompt = "a video of " + "very " * 100 + "long description"
    counts = model.validate_prompts([long_prompt])
    assert counts[0] > 77
