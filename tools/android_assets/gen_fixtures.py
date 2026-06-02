from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def gen_all(hf_repo: str, frames_dir: Path, labels: list[str], out_dir: Path) -> dict:
    """Generate golden fixtures from the HF torch reference (the source of truth).

    The reference is the fp32 PyTorch model + AutoProcessor loaded from `hf_repo` — NOT the
    exported ONNX bundle (which has no torch weights). On-device ONNX output is later compared
    against these fixtures within tolerance.
    """
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(hf_repo)
    model = AutoModel.from_pretrained(hf_repo).eval()

    # --- tokenizer golden ---
    tok_rows = []
    for text in labels:
        ids = proc(text=[text], padding="max_length", max_length=64,
                   truncation=True, return_tensors="np")["input_ids"][0].tolist()
        tok_rows.append({"text": text, "input_ids": ids})
    (out_dir / "tokenizer_golden.json").write_text(json.dumps(tok_rows, indent=2))

    # --- lossless frames: PNG -> AutoProcessor image path (bilinear square, normalize) ---
    frame_paths = sorted(frames_dir.glob("*.png"))
    images = [Image.open(p).convert("RGB") for p in frame_paths]
    pixel_values = proc(images=images, return_tensors="np")["pixel_values"]  # [F,3,R,R]
    np.savez(out_dir / "preprocess_golden.npz", pixel_values=pixel_values)

    # --- scores golden (fp32 reference) ---
    enc = proc(text=labels, images=images, padding="max_length", max_length=64,
               truncation=True, return_tensors="pt")
    with torch.inference_mode():
        o = model(**enc)
    confidence = torch.sigmoid(o.logits_per_image)            # [F,L]
    ie = o.image_embeds / o.image_embeds.norm(p=2, dim=-1, keepdim=True)
    te = o.text_embeds / o.text_embeds.norm(p=2, dim=-1, keepdim=True)
    cosine = ie @ te.T                                        # [F,L]
    (out_dir / "scores_golden.json").write_text(json.dumps({
        "labels": labels,
        "frames": [p.name for p in frame_paths],
        "confidence": confidence.tolist(),
        "cosine": cosine.tolist(),
    }, indent=2))

    # --- exact resample/normalize contract for the Android resampler target (M16) ---
    # SigLIP2 uses resample=2 (PIL BILINEAR) for all 4 profile models. PIL bilinear is
    # convolution-based and ANTIALIASES on downscale, so Android's plain Bitmap.createScaledBitmap
    # does NOT match — Plan 1 ports a custom separable-triangle resampler in Kotlin.
    # resample_contract.json is the authoritative per-model value. (antialias=null here is a red
    # herring: it gated the legacy LANCZOS path, not PIL's always-on triangle prefilter.)
    ip = proc.image_processor
    contract = {
        "do_resize": getattr(ip, "do_resize", True),
        "size": dict(getattr(ip, "size", {})),
        "resample": getattr(ip, "resample", 2),  # PIL.Image.BILINEAR == 2 (SigLIP2 default)
        "antialias": getattr(ip, "antialias", None),
        "do_rescale": getattr(ip, "do_rescale", True),
        "rescale_factor": getattr(ip, "rescale_factor", 1 / 255),
        "do_normalize": getattr(ip, "do_normalize", True),
        "image_mean": list(getattr(ip, "image_mean", [0.5, 0.5, 0.5])),
        "image_std": list(getattr(ip, "image_std", [0.5, 0.5, 0.5])),
        "do_convert_rgb": getattr(ip, "do_convert_rgb", True),
    }
    (out_dir / "resample_contract.json").write_text(json.dumps(contract, indent=2))
    return {"out_dir": str(out_dir)}
