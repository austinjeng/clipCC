import open_clip
import torch
from PIL import Image
from app.models.model_spec import ModelSpec


class ClipModel:
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

    def compute_similarities(
        self, images: list[Image.Image], texts: list[str]
    ) -> tuple[torch.Tensor, float]:
        text_features = self.encode_text(texts)
        image_features = self.encode_images(images)
        logit_scale = self.model.logit_scale.exp().item()
        cosine_sim = image_features @ text_features.T
        return cosine_sim, logit_scale

    def tokenize_and_check(self, prompts: list[str], max_tokens: int = 77) -> list[int]:
        counts = []
        for prompt in prompts:
            # Use the underlying encode() to get raw token count before truncation
            if hasattr(self.tokenizer, "encode"):
                raw = self.tokenizer.encode(prompt)
                counts.append(len(raw))
            else:
                tokens = self.tokenizer([prompt])[0]
                nonzero = (tokens != 0).sum().item()
                counts.append(nonzero)
        return counts

    def tokenize_raw(self, prompts: list[str]) -> list[torch.Tensor]:
        return [self.tokenizer([p])[0] for p in prompts]
