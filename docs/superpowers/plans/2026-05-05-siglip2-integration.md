# SigLIP2 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hot-swappable SigLIP2 model support with a web UI for model selection and video classification.

**Architecture:** Adapter pattern with `BaseModel` ABC, `ClipModel` and `SigLip2Model` implementations, a `ModelManager` with condition-based lease concurrency, and a vanilla HTML/JS frontend served by FastAPI.

**Tech Stack:** Python 3.11, FastAPI, HuggingFace transformers, open_clip, PyTorch, asyncio, vanilla HTML/JS

---

## Pre-work: Branch Setup

- [ ] **Step 1: Create and switch to the SigLip2 branch**

```bash
git checkout -b SigLip2
```

- [ ] **Step 2: Add dependencies to requirements.txt**

Add to the end of `requirements.txt`:

```
transformers>=4.50.0
sentencepiece
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add transformers and sentencepiece dependencies"
```

---

### Task 1: BaseModel ABC and ScoreBatch

**Files:**
- Create: `app/models/base_model.py`
- Test: `tests/test_base_model.py`

- [ ] **Step 1: Write the test for ScoreBatch and BaseModel interface**

Create `tests/test_base_model.py`:

```python
import pytest
import torch
from app.models.base_model import BaseModel, ScoreBatch


def test_score_batch_dataclass():
    batch = ScoreBatch(
        confidence=torch.tensor([[0.8, 0.2]]),
        raw_similarity=torch.tensor([[0.6, 0.3]]),
        logits=torch.tensor([[1.5, -0.5]]),
        semantics="clip_relative_softmax",
    )
    assert batch.confidence.shape == (1, 2)
    assert batch.semantics == "clip_relative_softmax"


def test_base_model_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseModel()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_base_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models.base_model'`

- [ ] **Step 3: Implement BaseModel and ScoreBatch**

Create `app/models/base_model.py`:

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image


@dataclass
class ScoreBatch:
    confidence: torch.Tensor
    raw_similarity: torch.Tensor
    logits: torch.Tensor
    semantics: str


class BaseModel(ABC):
    model_type: str
    device: str
    max_token_length: int

    @abstractmethod
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        ...

    @abstractmethod
    def score_batch(self, images: list[Image.Image], texts: list[str]) -> ScoreBatch:
        ...

    @abstractmethod
    def validate_prompts(self, prompts: list[str]) -> list[int]:
        ...

    @abstractmethod
    def tokenize_for_inference(self, prompts: list[str]) -> Any:
        ...

    @abstractmethod
    def tokenize_raw(self, prompts: list[str]) -> list[torch.Tensor]:
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_base_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/models/base_model.py tests/test_base_model.py
git commit -m "feat: add BaseModel ABC and ScoreBatch dataclass"
```

---

### Task 2: Refactor ClipModel to Extend BaseModel

**Files:**
- Modify: `app/models/clip_model.py`
- Modify: `tests/test_clip_model.py`

- [ ] **Step 1: Write tests for new BaseModel-conformant methods**

Add to end of `tests/test_clip_model.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_clip_model.py::test_clip_model_is_base_model tests/test_clip_model.py::test_score_batch tests/test_clip_model.py::test_validate_prompts tests/test_clip_model.py::test_validate_prompts_detects_overflow -v`
Expected: FAIL (ClipModel not an instance of BaseModel, no score_batch method)

- [ ] **Step 3: Refactor ClipModel to extend BaseModel**

Replace contents of `app/models/clip_model.py`:

```python
import open_clip
import torch
from PIL import Image
from app.models.base_model import BaseModel, ScoreBatch
from app.models.model_spec import ModelSpec


class ClipModel(BaseModel):
    model_type = "clip"
    max_token_length = 77

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            spec.model_name, pretrained=spec.pretrained, cache_dir=spec.cache_dir,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(spec.model_name)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda"):
                    return self.model.encode_text(tokens, normalize=True)
            return self.model.encode_text(tokens, normalize=True)

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda"):
                    return self.model.encode_image(batch, normalize=True)
            return self.model.encode_image(batch, normalize=True)

    def score_batch(self, images: list[Image.Image], texts: list[str]) -> ScoreBatch:
        text_features = self.encode_text(texts)
        image_features = self.encode_images(images)
        logit_scale = self.model.logit_scale.exp().item()
        raw_similarity = image_features @ text_features.T
        logits = raw_similarity * logit_scale
        confidence = torch.softmax(logits, dim=-1)
        return ScoreBatch(
            confidence=confidence,
            raw_similarity=raw_similarity,
            logits=logits,
            semantics="clip_relative_softmax",
        )

    def validate_prompts(self, prompts: list[str]) -> list[int]:
        counts = []
        for prompt in prompts:
            if hasattr(self.tokenizer, "encode"):
                raw = self.tokenizer.encode(prompt)
                counts.append(len(raw))
            else:
                tokens = self.tokenizer([prompt])[0]
                nonzero = (tokens != 0).sum().item()
                counts.append(nonzero)
        return counts

    def tokenize_for_inference(self, prompts: list[str]) -> torch.Tensor:
        return self.tokenizer(prompts).to(self.device)

    def tokenize_raw(self, prompts: list[str]) -> list[torch.Tensor]:
        return [self.tokenizer([p])[0] for p in prompts]

    def compute_similarities(
        self, images: list[Image.Image], texts: list[str]
    ) -> tuple[torch.Tensor, float]:
        text_features = self.encode_text(texts)
        image_features = self.encode_images(images)
        logit_scale = self.model.logit_scale.exp().item()
        cosine_sim = image_features @ text_features.T
        return cosine_sim, logit_scale

    def tokenize_and_check(self, prompts: list[str], max_tokens: int = 77) -> list[int]:
        return self.validate_prompts(prompts)
```

Note: `compute_similarities` and `tokenize_and_check` kept as aliases for backward compatibility with `app/main.py` until Task 7 updates it.

- [ ] **Step 4: Run all ClipModel tests**

Run: `pytest tests/test_clip_model.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add app/models/clip_model.py app/models/base_model.py tests/test_clip_model.py
git commit -m "refactor: ClipModel extends BaseModel with score_batch and validate_prompts"
```

---

### Task 3: SigLip2Model Implementation

**Files:**
- Create: `app/models/siglip2_model.py`
- Create: `tests/test_siglip2_model.py`

- [ ] **Step 1: Write tests for SigLip2Model**

Create `tests/test_siglip2_model.py`:

```python
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
        # Sigmoid outputs are in (0, 1) but rows do NOT sum to 1
        assert (batch.confidence >= 0).all()
        assert (batch.confidence <= 1).all()

    def test_encode_images(self, model, dummy_images):
        features = model.encode_images(dummy_images)
        assert features.shape[0] == 3
        assert features.shape[1] > 0
        # Should be L2-normalized
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
        assert result["input_ids"].shape[1] == 64  # max_length padding
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_siglip2_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models.siglip2_model'`

- [ ] **Step 3: Implement SigLip2Model**

Create `app/models/siglip2_model.py`:

```python
from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from app.models.base_model import BaseModel, ScoreBatch


class SigLip2Model(BaseModel):
    model_type = "siglip2"
    max_token_length = 64

    def __init__(self, hf_repo: str, cache_dir: str):
        self.hf_repo = hf_repo
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(hf_repo, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(
            hf_repo,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda"):
                    features = self.model.get_image_features(**inputs)
            else:
                features = self.model.get_image_features(**inputs)
        return features / features.norm(p=2, dim=-1, keepdim=True)

    def score_batch(self, images: list[Image.Image], texts: list[str]) -> ScoreBatch:
        inputs = self.processor(
            text=texts,
            images=images,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda"):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        logits = outputs.logits_per_image
        confidence = torch.sigmoid(logits)

        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        raw_similarity = image_embeds @ text_embeds.T

        return ScoreBatch(
            confidence=confidence,
            raw_similarity=raw_similarity,
            logits=logits,
            semantics="siglip2_pairwise_sigmoid",
        )

    def validate_prompts(self, prompts: list[str]) -> list[int]:
        counts = []
        for prompt in prompts:
            tokens = self.processor.tokenizer(
                prompt.lower(), truncation=False, padding=False
            )
            counts.append(len(tokens["input_ids"]))
        return counts

    def tokenize_for_inference(self, prompts: list[str]) -> dict:
        return self.processor(
            text=prompts,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )

    def tokenize_raw(self, prompts: list[str]) -> list[torch.Tensor]:
        result = []
        for prompt in prompts:
            tokens = self.processor.tokenizer(
                prompt.lower(), truncation=False, padding=False, return_tensors="pt"
            )
            result.append(tokens["input_ids"][0])
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_siglip2_model.py -v`
Expected: All PASS (requires network to download model on first run, about 800MB for base)

Note: If running in CI without network, these tests should be marked `@pytest.mark.slow` or skipped. For local dev, first run will be slow.

- [ ] **Step 5: Commit**

```bash
git add app/models/siglip2_model.py tests/test_siglip2_model.py
git commit -m "feat: add SigLip2Model with score_batch, validate_prompts, sigmoid scoring"
```

---

### Task 4: ModelManager with Condition-Based Concurrency

**Files:**
- Create: `app/models/model_manager.py`
- Create: `tests/test_model_manager.py`

- [ ] **Step 1: Write tests for ModelManager**

Create `tests/test_model_manager.py`:

```python
import asyncio
import pytest
from unittest.mock import patch, MagicMock
from app.models.model_manager import ModelManager, NoModelLoadedError, SIGLIP2_REGISTRY


@pytest.fixture
def manager(temp_dir):
    return ModelManager(cache_dir=str(temp_dir / "models"))


class TestRegistry:
    def test_registry_has_models(self, manager):
        models = manager.list_models()
        assert len(models) >= 6
        assert any(m["model_id"] == "siglip2-base-patch16-256" for m in models)

    def test_list_models_includes_status(self, manager):
        models = manager.list_models()
        for m in models:
            assert "model_id" in m
            assert "display_name" in m
            assert "loaded" in m
            assert "cached" in m


class TestAcquire:
    @pytest.mark.asyncio
    async def test_acquire_raises_when_no_model(self, manager):
        with pytest.raises(NoModelLoadedError):
            async with manager.acquire(timeout=1.0):
                pass

    @pytest.mark.asyncio
    async def test_acquire_returns_model_after_load(self, manager):
        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_instance = MagicMock()
            mock_instance.model_type = "siglip2"
            MockModel.return_value = mock_instance
            await manager.load_model("siglip2-base-patch16-256")

        async with manager.acquire(timeout=5.0) as lease:
            assert lease.model is mock_instance

    @pytest.mark.asyncio
    async def test_concurrent_leases_allowed(self, manager):
        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            await manager.load_model("siglip2-base-patch16-256")

        async def hold_lease(duration):
            async with manager.acquire(timeout=5.0) as lease:
                await asyncio.sleep(duration)
                return lease.model

        results = await asyncio.gather(
            hold_lease(0.1), hold_lease(0.1), hold_lease(0.1)
        )
        assert all(r is mock_instance for r in results)


class TestLoadModel:
    @pytest.mark.asyncio
    async def test_load_sets_active(self, manager):
        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            await manager.load_model("siglip2-base-patch16-256")

        assert manager.active_model_id == "siglip2-base-patch16-256"
        assert manager.active_model is mock_instance

    @pytest.mark.asyncio
    async def test_load_same_model_noop(self, manager):
        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            await manager.load_model("siglip2-base-patch16-256")
            await manager.load_model("siglip2-base-patch16-256")

        assert MockModel.call_count == 1

    @pytest.mark.asyncio
    async def test_load_waits_for_leases_to_drain(self, manager):
        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_a = MagicMock()
            mock_b = MagicMock()
            MockModel.side_effect = [mock_a, mock_b]
            await manager.load_model("siglip2-base-patch16-256")

        order = []

        async def hold_then_release():
            async with manager.acquire(timeout=5.0) as lease:
                await asyncio.sleep(0.2)
                order.append("lease_released")

        async def swap_model():
            await asyncio.sleep(0.05)
            with patch("app.models.model_manager.SigLip2Model") as MockModel2:
                mock_new = MagicMock()
                MockModel2.return_value = mock_new
                await manager.load_model("siglip2-base-patch16-384")
                order.append("swap_complete")

        await asyncio.gather(hold_then_release(), swap_model())
        assert order == ["lease_released", "swap_complete"]

    @pytest.mark.asyncio
    async def test_load_failure_recovers(self, manager):
        with patch("app.models.model_manager.SigLip2Model") as MockModel:
            mock_instance = MagicMock()
            MockModel.return_value = mock_instance
            await manager.load_model("siglip2-base-patch16-256")

        with patch("app.models.model_manager.SigLip2Model", side_effect=RuntimeError("download failed")):
            with pytest.raises(RuntimeError):
                await manager.load_model("siglip2-base-patch16-384")

        # Manager is still usable -- swapping flag cleared
        assert manager._swapping is False
        # No model loaded (old was cleared before attempt)
        assert manager.active_model is None

    @pytest.mark.asyncio
    async def test_invalid_model_id_raises(self, manager):
        with pytest.raises(KeyError):
            await manager.load_model("nonexistent-model")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model_manager.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ModelManager**

Create `app/models/model_manager.py`:

```python
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

import torch

from app.models.base_model import BaseModel
from app.models.siglip2_model import SigLip2Model


class NoModelLoadedError(Exception):
    pass


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    display_name: str
    model_type: str
    hf_repo: str
    params: str
    resolution: int | str


@dataclass
class ModelLease:
    model: BaseModel


SIGLIP2_REGISTRY: dict[str, ModelConfig] = {
    "siglip2-base-patch16-256": ModelConfig(
        model_id="siglip2-base-patch16-256",
        display_name="SigLIP2 Base (256px)",
        model_type="siglip2",
        hf_repo="google/siglip2-base-patch16-256",
        params="0.4B",
        resolution=256,
    ),
    "siglip2-base-patch16-384": ModelConfig(
        model_id="siglip2-base-patch16-384",
        display_name="SigLIP2 Base (384px)",
        model_type="siglip2",
        hf_repo="google/siglip2-base-patch16-384",
        params="0.4B",
        resolution=384,
    ),
    "siglip2-large-patch16-256": ModelConfig(
        model_id="siglip2-large-patch16-256",
        display_name="SigLIP2 Large (256px)",
        model_type="siglip2",
        hf_repo="google/siglip2-large-patch16-256",
        params="0.9B",
        resolution=256,
    ),
    "siglip2-large-patch16-384": ModelConfig(
        model_id="siglip2-large-patch16-384",
        display_name="SigLIP2 Large (384px)",
        model_type="siglip2",
        hf_repo="google/siglip2-large-patch16-384",
        params="0.9B",
        resolution=384,
    ),
    "siglip2-so400m-patch14-384": ModelConfig(
        model_id="siglip2-so400m-patch14-384",
        display_name="SigLIP2 SO400M (384px)",
        model_type="siglip2",
        hf_repo="google/siglip2-so400m-patch14-384",
        params="1B",
        resolution=384,
    ),
    "siglip2-so400m-patch16-512": ModelConfig(
        model_id="siglip2-so400m-patch16-512",
        display_name="SigLIP2 SO400M (512px)",
        model_type="siglip2",
        hf_repo="google/siglip2-so400m-patch16-512",
        params="1B",
        resolution=512,
    ),
}


class ModelManager:
    def __init__(self, cache_dir: str):
        self.registry = dict(SIGLIP2_REGISTRY)
        self.active_model: BaseModel | None = None
        self.active_model_id: str | None = None
        self.cache_dir = cache_dir
        self._condition = asyncio.Condition()
        self._swapping = False
        self._active_leases = 0

    @asynccontextmanager
    async def acquire(self, timeout: float) -> AsyncGenerator[ModelLease, None]:
        async with asyncio.timeout(timeout):
            async with self._condition:
                await self._condition.wait_for(lambda: not self._swapping)
                if self.active_model is None:
                    raise NoModelLoadedError()
                self._active_leases += 1
                model_ref = self.active_model

        try:
            yield ModelLease(model=model_ref)
        finally:
            async with self._condition:
                self._active_leases -= 1
                if self._active_leases == 0:
                    self._condition.notify_all()

    async def load_model(self, model_id: str) -> None:
        config = self.registry[model_id]

        async with self._condition:
            await self._condition.wait_for(lambda: not self._swapping)
            if self.active_model_id == model_id:
                return
            self._swapping = True
            self._condition.notify_all()
            await self._condition.wait_for(lambda: self._active_leases == 0)
            old_model = self.active_model
            self.active_model = None
            self.active_model_id = None

        try:
            del old_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            new_model = SigLip2Model(
                hf_repo=config.hf_repo, cache_dir=self.cache_dir
            )
        except Exception:
            async with self._condition:
                self._swapping = False
                self._condition.notify_all()
            raise

        async with self._condition:
            self.active_model = new_model
            self.active_model_id = model_id
            self._swapping = False
            self._condition.notify_all()

    def list_models(self) -> list[dict]:
        result = []
        for config in self.registry.values():
            cached = self._is_cached(config.hf_repo)
            result.append({
                "model_id": config.model_id,
                "display_name": config.display_name,
                "model_type": config.model_type,
                "params": config.params,
                "resolution": config.resolution,
                "loaded": self.active_model_id == config.model_id,
                "cached": cached,
            })
        return result

    def _is_cached(self, hf_repo: str) -> bool:
        cache_path = Path(self.cache_dir) / f"models--{hf_repo.replace('/', '--')}"
        return cache_path.exists()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model_manager.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add app/models/model_manager.py tests/test_model_manager.py
git commit -m "feat: add ModelManager with condition-based lease concurrency"
```

---

### Task 5: Scoring Service Adaptation

**Files:**
- Modify: `app/services/scoring.py`
- Modify: `tests/test_scoring.py`

- [ ] **Step 1: Write test for new aggregate_frame_scores function**

Add to end of `tests/test_scoring.py`:

```python
from app.models.base_model import ScoreBatch
from app.services.scoring import aggregate_frame_scores


def test_aggregate_frame_scores_mean():
    batch1 = ScoreBatch(
        confidence=torch.tensor([[0.8, 0.1, 0.1], [0.6, 0.2, 0.2]]),
        raw_similarity=torch.tensor([[0.5, 0.3, 0.2], [0.4, 0.3, 0.3]]),
        logits=torch.tensor([[2.0, -1.0, -1.0], [1.0, -0.5, -0.5]]),
        semantics="clip_relative_softmax",
    )
    frames = [
        FrameSample(path=Path("/tmp/f1.jpg"), approx_timestamp_seconds=0.0),
        FrameSample(path=Path("/tmp/f2.jpg"), approx_timestamp_seconds=1.0),
    ]
    labels = ["cat", "dog", "bird"]

    scores, best = aggregate_frame_scores([batch1], labels, frames, "mean")
    assert len(scores) == 3
    assert best.label == "cat"
    assert best.confidence > 0


def test_aggregate_frame_scores_max():
    batch1 = ScoreBatch(
        confidence=torch.tensor([[0.3, 0.9, 0.1]]),
        raw_similarity=torch.tensor([[0.2, 0.7, 0.1]]),
        logits=torch.tensor([[0.0, 2.0, -1.0]]),
        semantics="siglip2_pairwise_sigmoid",
    )
    batch2 = ScoreBatch(
        confidence=torch.tensor([[0.8, 0.2, 0.1]]),
        raw_similarity=torch.tensor([[0.6, 0.1, 0.05]]),
        logits=torch.tensor([[1.5, -0.5, -1.0]]),
        semantics="siglip2_pairwise_sigmoid",
    )
    frames = [
        FrameSample(path=Path("/tmp/f1.jpg"), approx_timestamp_seconds=0.0),
        FrameSample(path=Path("/tmp/f2.jpg"), approx_timestamp_seconds=1.0),
    ]
    labels = ["cat", "dog", "bird"]

    scores, best = aggregate_frame_scores([batch1, batch2], labels, frames, "max")
    assert len(scores) == 3
    # dog has highest peak (0.9 in batch1)
    assert best.label == "dog"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scoring.py::test_aggregate_frame_scores_mean tests/test_scoring.py::test_aggregate_frame_scores_max -v`
Expected: FAIL with `ImportError: cannot import name 'aggregate_frame_scores'`

- [ ] **Step 3: Add aggregate_frame_scores to scoring.py**

Add to end of `app/services/scoring.py`:

```python
from app.models.base_model import ScoreBatch


def aggregate_frame_scores(
    batches: list[ScoreBatch],
    labels: list[str],
    frames: list[FrameSample],
    aggregation: str,
) -> tuple[list[ScoreItem], BestMatch]:
    all_confidence = torch.cat([b.confidence for b in batches], dim=0)
    all_raw_sim = torch.cat([b.raw_similarity for b in batches], dim=0)
    return build_response_scores(all_confidence, all_raw_sim, labels, frames, aggregation)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scoring.py -v`
Expected: All PASS (both old and new tests)

- [ ] **Step 5: Commit**

```bash
git add app/services/scoring.py tests/test_scoring.py
git commit -m "feat: add aggregate_frame_scores for ScoreBatch-based pipeline"
```

---

### Task 6: Config and Response Schema Updates

**Files:**
- Modify: `app/config.py`
- Modify: `app/schemas/response.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write test for new config field**

Add to `tests/test_config.py`:

```python
def test_default_model_id_default():
    s = Settings(allow_unauthenticated=True)
    assert s.default_model_id == "siglip2-base-patch16-256"


def test_default_model_id_from_env(monkeypatch):
    monkeypatch.setenv("DEFAULT_MODEL_ID", "siglip2-large-patch16-384")
    s = Settings(allow_unauthenticated=True)
    assert s.default_model_id == "siglip2-large-patch16-384"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_default_model_id_default -v`
Expected: FAIL (no `default_model_id` field)

- [ ] **Step 3: Add default_model_id to Settings**

In `app/config.py`, add to the `Settings` class after `temp_dir`:

```python
    default_model_id: str = "siglip2-base-patch16-256"
```

- [ ] **Step 4: Update ClassifyMetadata schema**

In `app/schemas/response.py`, add two fields to `ClassifyMetadata`:

```python
class ClassifyMetadata(BaseModel):
    frames_analyzed: int
    video_duration_seconds: float
    model: str
    device: str
    aggregation: str
    processing_time_seconds: float
    disclaimer: str
    model_type: str = ""
    score_semantics: str = ""
```

- [ ] **Step 5: Run all config and schema tests**

Run: `pytest tests/test_config.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add app/config.py app/schemas/response.py tests/test_config.py
git commit -m "feat: add default_model_id config and score_semantics to response"
```

---

### Task 7: Update main.py -- ModelManager, New Endpoints, Static Serving

**Files:**
- Modify: `app/main.py`
- Modify: `app/middleware.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write tests for new model endpoints**

Add to `tests/test_api.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport
from app.main import create_app
from app.config import Settings


@pytest.fixture
def app():
    settings = Settings(allow_unauthenticated=True, temp_dir="/tmp/clipcc_test")
    return create_app(settings)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_list_models(client):
    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 6
    assert all("model_id" in m for m in data)


@pytest.mark.asyncio
async def test_load_model_invalid(client):
    resp = await client.post("/api/v1/models/load", json={"model_id": "nonexistent"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_active_model_none_at_start(client):
    resp = await client.get("/api/v1/models/active")
    # Could be 200 (if background loaded fast) or 404
    assert resp.status_code in (200, 404)


@pytest.mark.asyncio
async def test_models_endpoint_requires_auth():
    settings = Settings(api_key="secret123", temp_dir="/tmp/clipcc_test")
    app = create_app(settings)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/models")
        assert resp.status_code == 401
        resp = await client.get("/api/v1/models", headers={"X-API-Key": "secret123"})
        assert resp.status_code == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api.py::test_list_models -v`
Expected: FAIL (endpoint does not exist)

- [ ] **Step 3: Update middleware to auth model endpoints**

In `app/middleware.py`, insert before the `# All other paths: pass through` comment (before line 155):

```python
        # /api/v1/models*: auth required, no body size or concurrency gates
        if path.startswith("/api/v1/models"):
            if not self._check_auth(scope):
                await _send_json_response(
                    send,
                    401,
                    {"detail": "Invalid or missing API key. Provide a valid key in the X-API-Key header."},
                )
                return
            await self._app(scope, receive, send)
            return
```

- [ ] **Step 4: Rewrite main.py with ModelManager integration**

Replace `app/main.py` with the updated version. Key changes:
- Creates `ModelManager` in lifespan instead of directly loading ClipModel
- Background task auto-loads default model
- New endpoints: `GET /api/v1/models`, `POST /api/v1/models/load`, `GET /api/v1/models/active`
- Serves static `index.html` at `GET /`
- `/classify` uses `manager.acquire()` and `model.score_batch()`
- `/ready` uses ModelManager state

```python
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel as PydanticBaseModel

from app.config import Settings
from app.errors.handlers import (
    DuplicateTokensError,
    InferenceConcurrencyError,
    InferenceTimeoutError,
    InvalidAggregationError,
    InvalidFpsError,
    InvalidLabelsError,
    InvalidPromptTemplateError,
    TokenTruncationError,
    UnsupportedFormatError,
)
from app.inference_runner import InferenceRunner
from app.middleware import RequestGateMiddleware
from app.models.model_manager import ModelManager, NoModelLoadedError
from app.resource_gates import ResourceGates
from app.schemas.response import (
    ClassifyMetadata,
    ClassifyResponse,
    HealthResponse,
    ReadyResponse,
)
from app.services.scoring import aggregate_frame_scores
from app.services.video import FrameExtractor, probe_video, validate_video_constraints
from app.temp_store import TempStore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

DISCLAIMER_MEAN = (
    "Scores are relative to the supplied labels, not calibrated probabilities. "
    "Not suitable for safety-critical decisions."
)
DISCLAIMER_MAX = (
    "Scores are relative to the supplied labels, not calibrated probabilities. "
    "Max-mode scores are independent peaks per label and do not sum to 1. "
    "Not suitable for safety-critical decisions."
)

STATIC_DIR = Path(__file__).parent / "static"


class LoadModelRequest(PydanticBaseModel):
    model_id: str


def create_app(settings: Optional[Settings] = None) -> RequestGateMiddleware:
    if settings is None:
        settings = Settings()

    settings.validate_auth_config()

    state: dict = {"manager": None, "temp_store": None, "gates": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager = ModelManager(cache_dir=settings.clip_cache_dir)
        temp_store = TempStore(settings.temp_dir)
        temp_store.run_janitor()
        gates = ResourceGates(
            max_upload_concurrency=settings.effective_upload_concurrency,
            max_inference_concurrency=settings.max_concurrent_requests,
        )

        state["manager"] = manager
        state["temp_store"] = temp_store
        state["gates"] = gates

        async def _auto_load():
            try:
                await manager.load_model(settings.default_model_id)
                logger.info(f"Auto-loaded model: {settings.default_model_id}")
            except Exception as e:
                logger.error(f"Failed to auto-load model: {e}")

        asyncio.create_task(_auto_load())

        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/live", response_model=HealthResponse)
    async def live():
        return HealthResponse(status="ok")

    @app.get("/ready")
    async def ready():
        manager: Optional[ModelManager] = state.get("manager")
        if manager is None or manager.active_model is None:
            return JSONResponse(status_code=503, content={"detail": "Model not loaded"})
        return ReadyResponse(
            status="ready",
            model=manager.active_model_id or "",
            pretrained=manager.active_model_id or "",
            device=manager.active_model.device,
        )

    @app.get("/api/v1/models")
    async def list_models():
        manager: ModelManager = state["manager"]
        return manager.list_models()

    @app.post("/api/v1/models/load")
    async def load_model_endpoint(request: LoadModelRequest):
        manager: ModelManager = state["manager"]
        if request.model_id not in manager.registry:
            return JSONResponse(
                status_code=400,
                content={"detail": f"Unknown model_id: {request.model_id}"},
            )
        try:
            await manager.load_model(request.model_id)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"detail": f"Failed to load model: {str(e)}"},
            )
        return {"status": "loaded", "model_id": request.model_id}

    @app.get("/api/v1/models/active")
    async def active_model():
        manager: ModelManager = state["manager"]
        if manager.active_model is None:
            return JSONResponse(status_code=404, content={"detail": "No model loaded"})
        config = manager.registry[manager.active_model_id]
        return {
            "model_id": config.model_id,
            "display_name": config.display_name,
            "model_type": config.model_type,
            "params": config.params,
            "resolution": config.resolution,
            "device": manager.active_model.device,
        }

    @app.get("/")
    async def serve_ui():
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/api/v1/classify", response_model=ClassifyResponse)
    async def classify(
        video: UploadFile,
        labels: str = Form(...),
        prompt_template: str = Form(default="This is a photo of {}."),
        fps: float = Form(default=1.0),
        aggregation: str = Form(default="mean"),
    ):
        manager: Optional[ModelManager] = state.get("manager")
        temp_store: Optional[TempStore] = state.get("temp_store")
        gates: Optional[ResourceGates] = state.get("gates")

        filename = video.filename or ""
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(ext if ext else filename)

        if fps < 0.1 or fps > 5.0:
            raise InvalidFpsError(fps)

        if aggregation not in ("mean", "max"):
            raise InvalidAggregationError(aggregation)

        try:
            parsed_labels = json.loads(labels)
        except (json.JSONDecodeError, ValueError):
            raise InvalidLabelsError("labels must be a valid JSON array of strings.")

        if not isinstance(parsed_labels, list) or not all(
            isinstance(lb, str) for lb in parsed_labels
        ):
            raise InvalidLabelsError("labels must be a valid JSON array of strings.")

        if len(parsed_labels) < 3 or len(parsed_labels) > 10:
            raise InvalidLabelsError(
                "Number of labels must be between 3 and 10 (inclusive)."
            )

        for lb in parsed_labels:
            if not lb.strip():
                raise InvalidLabelsError("Each label must be a non-empty string.")
            if len(lb) > 200:
                raise InvalidLabelsError(
                    f"Label '{lb[:50]}...' exceeds the maximum length of 200 characters."
                )

        seen: set[str] = set()
        for lb in parsed_labels:
            if lb in seen:
                raise InvalidLabelsError(f"Duplicate label: '{lb}'.")
            seen.add(lb)

        brace_count = prompt_template.count("{}")
        if brace_count != 1:
            raise InvalidPromptTemplateError(
                "prompt_template must contain exactly one '{}' placeholder."
            )
        if len(prompt_template) > 500:
            raise InvalidPromptTemplateError(
                "prompt_template must be 500 characters or fewer."
            )

        prompts = [prompt_template.replace("{}", lb) for lb in parsed_labels]

        try:
            async with manager.acquire(timeout=settings.request_timeout_seconds) as lease:
                model = lease.model

                token_counts = model.validate_prompts(prompts)
                for prompt, count in zip(prompts, token_counts):
                    if count > model.max_token_length:
                        raise TokenTruncationError(prompt, count)

                raw_tokens = model.tokenize_raw(prompts)
                for i in range(len(raw_tokens)):
                    for j in range(i + 1, len(raw_tokens)):
                        if torch.equal(raw_tokens[i], raw_tokens[j]):
                            raise DuplicateTokensError(parsed_labels[i], parsed_labels[j])

                request_id = str(uuid.uuid4())
                start_time = time.monotonic()

                try:
                    stored = temp_store.save_upload(request_id, video.file)

                    video_info = probe_video(stored.path, timeout=settings.ffmpeg_timeout_seconds)
                    validate_video_constraints(video_info, settings, fps)

                    async with gates.inference_admission():
                        runner = InferenceRunner(timeout_seconds=settings.request_timeout_seconds)

                        def pipeline(cancel_event, runner_ref):
                            frame_dir = temp_store.create_frame_dir(request_id)
                            extractor = FrameExtractor(ffmpeg_timeout=settings.ffmpeg_timeout_seconds)
                            frame_samples = extractor.extract(
                                video_path=stored.path,
                                fps=fps,
                                max_frames=settings.max_frames,
                                frame_dir=frame_dir,
                                cancel_event=cancel_event,
                            )

                            all_batches = []
                            all_frames = []

                            for batch_start in range(0, len(frame_samples), settings.batch_size):
                                if cancel_event.is_set():
                                    break
                                batch = frame_samples[batch_start: batch_start + settings.batch_size]
                                images = [Image.open(fs.path).convert("RGB") for fs in batch]

                                score_batch = model.score_batch(images, prompts)
                                all_batches.append(score_batch)
                                all_frames.extend(batch)

                                for fs in batch:
                                    try:
                                        fs.path.unlink(missing_ok=True)
                                    except Exception:
                                        pass

                            return all_batches, all_frames

                        result = await runner.run(pipeline)

                        if result is None:
                            raise InferenceTimeoutError(settings.request_timeout_seconds)

                        all_batches, all_frames = result

                    scores, best_match = aggregate_frame_scores(
                        all_batches, parsed_labels, all_frames, aggregation
                    )

                    processing_time = time.monotonic() - start_time
                    disclaimer = DISCLAIMER_MAX if aggregation == "max" else DISCLAIMER_MEAN
                    semantics = all_batches[0].semantics if all_batches else ""

                    return ClassifyResponse(
                        best_match=best_match,
                        scores=scores,
                        metadata=ClassifyMetadata(
                            frames_analyzed=len(all_frames),
                            video_duration_seconds=video_info.duration,
                            model=manager.active_model_id or "",
                            device=model.device,
                            aggregation=aggregation,
                            processing_time_seconds=round(processing_time, 3),
                            disclaimer=disclaimer,
                            model_type=model.model_type,
                            score_semantics=semantics,
                        ),
                    )

                except InferenceConcurrencyError:
                    return JSONResponse(
                        status_code=429,
                        content={"detail": "Too many inference requests in progress. Please retry in a moment."},
                    )
                finally:
                    temp_store.cleanup(request_id)

        except NoModelLoadedError:
            return JSONResponse(
                status_code=503,
                content={"detail": "No model loaded. Load a model first via POST /api/v1/models/load."},
            )
        except TimeoutError:
            raise InferenceTimeoutError(settings.request_timeout_seconds)

    return RequestGateMiddleware(
        app=app,
        gates=ResourceGates(
            max_upload_concurrency=settings.effective_upload_concurrency,
            max_inference_concurrency=settings.max_concurrent_requests,
        ),
        api_key=settings.api_key,
        max_body_bytes=settings.max_file_size_bytes,
    )
```

- [ ] **Step 5: Run API tests**

Run: `pytest tests/test_api.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add app/main.py app/middleware.py tests/test_api.py
git commit -m "feat: integrate ModelManager, model endpoints, and lease-based classify"
```

---

### Task 8: Web UI

**Files:**
- Create: `app/static/index.html`

- [ ] **Step 1: Create the static directory**

```bash
mkdir -p app/static
```

- [ ] **Step 2: Create index.html**

Create `app/static/index.html` with the minimal UI:
- Model dropdown populated from `GET /api/v1/models`
- Load Model button with spinner
- Video upload + labels input + prompt template + FPS + aggregation
- Classify button (disabled until model loaded)
- Results section with horizontal bars

The HTML uses inline CSS (no external dependencies) and vanilla JavaScript:
- `fetchModels()` on page load
- `loadModel()` posts to `/api/v1/models/load`
- `classify()` posts multipart form to `/api/v1/classify`
- `renderResults()` draws sorted confidence bars
- Labels input supports comma-separated or JSON array via `JSON.parse()`

- [ ] **Step 3: Verify UI serves correctly**

Start the dev server and open `http://localhost:8000/` in a browser:

```bash
ALLOW_UNAUTHENTICATED=true uvicorn app.main:create_app --factory --reload
```

Expected: Page loads, model dropdown populated, "No model loaded" status shown.

- [ ] **Step 4: Commit**

```bash
git add app/static/index.html
git commit -m "feat: add minimal web UI for model selection and video classification"
```

---

### Task 9: Docker and Compose Updates

**Files:**
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`

- [ ] **Step 1: Update Dockerfile -- remove model bake step**

Replace the model bake section in `Dockerfile`. Remove these lines:

```dockerfile
ARG MODEL_NAME=ViT-L-14
ARG PRETRAINED=laion2b_s32b_b82k

ENV CLIP_CACHE_DIR=/app/models

RUN python -c "import json, open_clip; \
open_clip.create_model_and_transforms('${MODEL_NAME}', pretrained='${PRETRAINED}', cache_dir='/app/models'); \
json.dump({'model_name': '${MODEL_NAME}', 'pretrained': '${PRETRAINED}', 'cache_dir': '/app/models'}, open('/app/.baked_model', 'w'))"
```

Replace with:

```dockerfile
ENV CLIP_CACHE_DIR=/app/models
RUN mkdir -p /app/models
```

- [ ] **Step 2: Update docker-compose.yml**

Replace contents of `docker-compose.yml`:

```yaml
services:
  clipcc-cpu:
    build:
      context: .
      args:
        TORCH_VARIANT: cpu
    ports:
      - "8000:8000"
    volumes:
      - clipcc-models:/app/models
    environment:
      - MAX_FILE_SIZE_MB=500
      - MAX_DURATION_SECONDS=300
      - MAX_FRAMES=300
      - DEFAULT_FPS=1.0
      - MAX_CONCURRENT_REQUESTS=2
      - ALLOW_UNAUTHENTICATED=true
      - DEFAULT_MODEL_ID=siglip2-base-patch16-256
    profiles: ["cpu"]

  clipcc-gpu:
    build:
      context: .
      args:
        TORCH_VARIANT: cu121
    ports:
      - "8000:8000"
    volumes:
      - clipcc-models:/app/models
    environment:
      - MAX_FILE_SIZE_MB=500
      - MAX_DURATION_SECONDS=300
      - MAX_FRAMES=300
      - DEFAULT_FPS=1.0
      - MAX_CONCURRENT_REQUESTS=1
      - ALLOW_UNAUTHENTICATED=true
      - DEFAULT_MODEL_ID=siglip2-so400m-patch14-384
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    profiles: ["gpu"]

volumes:
  clipcc-models:
```

- [ ] **Step 3: Verify Docker build**

```bash
docker compose --profile cpu build
```

Expected: Build completes without errors.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: Docker with volume-based model cache, no baked weights"
```

---

### Task 10: Integration Test -- Full Flow

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

Create `tests/test_integration.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
import torch
from httpx import AsyncClient, ASGITransport
from app.main import create_app
from app.config import Settings
from app.models.base_model import ScoreBatch


@pytest.fixture
def mock_siglip2():
    with patch("app.models.model_manager.SigLip2Model") as MockModel:
        instance = MagicMock()
        instance.model_type = "siglip2"
        instance.device = "cpu"
        instance.max_token_length = 64
        instance.validate_prompts.return_value = [10, 10, 10]
        instance.tokenize_raw.return_value = [
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 2, 4]),
            torch.tensor([1, 2, 5]),
        ]
        instance.score_batch.return_value = ScoreBatch(
            confidence=torch.tensor([[0.8, 0.1, 0.1]]),
            raw_similarity=torch.tensor([[0.6, 0.2, 0.1]]),
            logits=torch.tensor([[1.5, -1.0, -1.5]]),
            semantics="siglip2_pairwise_sigmoid",
        )
        MockModel.return_value = instance
        yield instance


@pytest.fixture
def app(mock_siglip2):
    settings = Settings(
        allow_unauthenticated=True,
        temp_dir="/tmp/clipcc_integration_test",
        clip_cache_dir="/tmp/clipcc_integration_models",
    )
    return create_app(settings)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_full_flow_load_then_classify(client, small_video, mock_siglip2):
    # Load model
    resp = await client.post("/api/v1/models/load", json={"model_id": "siglip2-base-patch16-256"})
    assert resp.status_code == 200

    # Verify active
    resp = await client.get("/api/v1/models/active")
    assert resp.status_code == 200
    assert resp.json()["model_id"] == "siglip2-base-patch16-256"

    # Classify
    import json
    with open(small_video, "rb") as f:
        resp = await client.post(
            "/api/v1/classify",
            files={"video": ("test.mp4", f, "video/mp4")},
            data={
                "labels": json.dumps(["driving", "parking", "reversing"]),
                "prompt_template": "This is a photo of {}.",
                "fps": "1.0",
                "aggregation": "mean",
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "best_match" in data
    assert "scores" in data
    assert data["metadata"]["model_type"] == "siglip2"
    assert data["metadata"]["score_semantics"] == "siglip2_pairwise_sigmoid"
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for full load then classify flow"
```

---

### Task 11: Manual E2E Verification

- [ ] **Step 1: Start dev server**

```bash
ALLOW_UNAUTHENTICATED=true uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
```

- [ ] **Step 2: Open browser at http://localhost:8000/**

Verify:
- Model dropdown shows 6 SigLIP2 models
- Status shows "Loading..." then "Loaded" (after background auto-load)

- [ ] **Step 3: Test model switching**

- Select a different model from dropdown
- Click "Load Model"
- Verify spinner shows during load, then status updates

- [ ] **Step 4: Test classification**

- Select a video file (.mp4)
- Enter labels: `driving, parking, reversing`
- Click Classify
- Verify results bars appear with confidence values

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v --ignore=tests/test_siglip2_model.py
```

(Ignore SigLIP2 model tests if no network access for model download)

Expected: All pass.

- [ ] **Step 6: Final commit if any fixups needed**

```bash
git add -A
git commit -m "fix: address issues found during manual E2E testing"
```

---

## File Map Summary

| File | Task | Action |
|---|---|---|
| `requirements.txt` | Pre-work | Modify -- add transformers, sentencepiece |
| `app/models/base_model.py` | 1 | Create -- ABC + ScoreBatch |
| `app/models/clip_model.py` | 2 | Modify -- extend BaseModel, add score_batch |
| `app/models/siglip2_model.py` | 3 | Create -- SigLIP2 via HuggingFace transformers |
| `app/models/model_manager.py` | 4 | Create -- registry, condition-based leases, hot-swap |
| `app/services/scoring.py` | 5 | Modify -- add aggregate_frame_scores |
| `app/config.py` | 6 | Modify -- add default_model_id |
| `app/schemas/response.py` | 6 | Modify -- add model_type, score_semantics |
| `app/main.py` | 7 | Rewrite -- ModelManager, new endpoints, lease-based classify |
| `app/middleware.py` | 7 | Modify -- auth for /api/v1/models* |
| `app/static/index.html` | 8 | Create -- minimal web UI |
| `Dockerfile` | 9 | Modify -- remove bake, add models dir |
| `docker-compose.yml` | 9 | Modify -- add volume, DEFAULT_MODEL_ID |
| `tests/test_base_model.py` | 1 | Create |
| `tests/test_clip_model.py` | 2 | Modify -- add BaseModel conformance tests |
| `tests/test_siglip2_model.py` | 3 | Create |
| `tests/test_model_manager.py` | 4 | Create |
| `tests/test_scoring.py` | 5 | Modify -- add aggregate_frame_scores tests |
| `tests/test_config.py` | 6 | Modify -- add default_model_id tests |
| `tests/test_api.py` | 7 | Modify -- add model endpoint tests |
| `tests/test_integration.py` | 10 | Create |
