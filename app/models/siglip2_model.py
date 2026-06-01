from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from app.models.base_model import BaseModel, ScoreBatch
from app.services.temporal_policy import ScoreSemantics


class SigLip2Model(BaseModel):
    model_type = "siglip2"
    max_token_length = 64

    def __init__(
        self,
        hf_repo: str,
        cache_dir: str,
        revision: str | None = None,
        offline: bool = False,
    ):
        self.hf_repo = hf_repo
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(
            hf_repo,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=offline,
        )
        self.model = AutoModel.from_pretrained(
            hf_repo,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=offline,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast("cuda"):
                    outputs = self.model.get_image_features(**inputs)
            else:
                outputs = self.model.get_image_features(**inputs)
        features = outputs.pooler_output
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
            semantics=ScoreSemantics.SIGLIP2_SIGMOID,
        )

    def validate_prompts(self, prompts: list[str]) -> list[int]:
        # Tokenize the original-case text so the token-count check matches what
        # score_batch/tokenize_for_inference actually feed the model. The Gemma
        # tokenizer is case-sensitive; lowercasing here would validate a
        # different token sequence than inference uses.
        counts = []
        for prompt in prompts:
            tokens = self.processor.tokenizer(
                prompt, truncation=False, padding=False
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
        # Original case: duplicate-label detection must match inference, which
        # is case-sensitive. Lowercasing here would wrongly flag e.g. "Car" and
        # "car" as identical even though the model scores them differently.
        result = []
        for prompt in prompts:
            tokens = self.processor.tokenizer(
                prompt, truncation=False, padding=False, return_tensors="pt"
            )
            result.append(tokens["input_ids"][0])
        return result
