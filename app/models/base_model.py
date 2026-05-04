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
