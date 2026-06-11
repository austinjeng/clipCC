from __future__ import annotations

import threading

import torch
from PIL import Image
from transformers import StoppingCriteria, StoppingCriteriaList


class CancelStoppingCriteria(StoppingCriteria):
    """Reads the SAME threading.Event that InferenceRunner sets on deadline.

    Consulted between decode steps only — blind during prefill. This makes the
    request timeout a soft deadline (spec §3); max_new_tokens is the real bound.
    """

    def __init__(self, cancel_event: threading.Event):
        super().__init__()
        self._cancel_event = cancel_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self._cancel_event.is_set()


def pick_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    # bf16 on x86 CPU falls back to slow kernels or silently upcasts; fp32 is
    # the honest CPU dtype (≈22GB resident — CPU is demo-only per spec).
    return "cpu", torch.float32


def build_messages(frames: list[Image.Image], prompt: str) -> list[dict]:
    content: list[dict] = [{"type": "image", "image": img} for img in frames]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


class GemmaVLM:
    """Thin wrapper around Gemma 4 E2B via transformers.

    NOTE for implementer: exact apply_chat_template / processor kwargs must be
    verified against the installed transformers version using the model card
    examples (https://huggingface.co/google/gemma-4-E2B-it) in the gated
    integration test. enable_thinking=False is REQUIRED — the E2B/E4B
    "ghost thought channel" token leak breaks JSON parsing otherwise (spec §5).
    """

    def __init__(self, hf_repo: str, cache_dir: str, image_token_budget: int = 280):
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.device, self.dtype = pick_device_and_dtype()
        # Not yet wired into the processor call: the exact kwarg name for the
        # per-image token budget is confirmed against the installed transformers
        # in the gated integration test (Task 14), then plumbed through here.
        self.image_token_budget = image_token_budget
        self.processor = AutoProcessor.from_pretrained(hf_repo, cache_dir=cache_dir)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            hf_repo,
            cache_dir=cache_dir,
            dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        frames: list[Image.Image],
        prompt: str,
        max_new_tokens: int,
        cancel_event: threading.Event,
    ) -> str:
        messages = build_messages(frames, prompt)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([CancelStoppingCriteria(cancel_event)]),
            )
        new_tokens = output_ids[:, inputs["input_ids"].shape[-1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
