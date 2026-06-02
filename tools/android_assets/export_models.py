from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    hf_repo: str
    display_name: str
    params: str
    resolution: int
    precision: str       # target device precision
    ram_budget_mb: int   # rough estimate; so400m refined by Spike 0d (Task 12)


PROFILE_MODELS: dict[str, ModelSpec] = {
    "siglip2-base-patch16-256": ModelSpec(
        "siglip2-base-patch16-256", "google/siglip2-base-patch16-256",
        "SigLIP2 Base (256px)", "0.4B", 256, "fp32", 1600),
    "siglip2-base-patch16-384": ModelSpec(
        "siglip2-base-patch16-384", "google/siglip2-base-patch16-384",
        "SigLIP2 Base (384px)", "0.4B", 384, "fp32", 1900),
    "siglip2-large-patch16-384": ModelSpec(
        "siglip2-large-patch16-384", "google/siglip2-large-patch16-384",
        "SigLIP2 Large (384px)", "0.9B", 384, "fp16", 2600),
    "siglip2-so400m-patch14-384": ModelSpec(
        "siglip2-so400m-patch14-384", "google/siglip2-so400m-patch14-384",
        "SigLIP2 SO400M (384px)", "1.0B", 384, "fp16", 3600),
}


def onnx_community_repo(hf_repo: str) -> str:
    """google/siglip2-X -> onnx-community/siglip2-X-ONNX."""
    name = hf_repo.split("/", 1)[1]
    return f"onnx-community/{name}-ONNX"


def _tower_basenames(precision: str) -> tuple[str, str]:
    """Prebuilt onnx-community filenames for the target precision."""
    if precision == "fp16":
        return "vision_model_fp16.onnx", "text_model_fp16.onnx"
    return "vision_model.onnx", "text_model.onnx"


def _place(src: Path, dst: Path) -> str | None:
    """Copy an .onnx to a canonical path; copy its co-located .onnx_data sibling if present.
    Returns the canonical data-file name when one was copied, else None."""
    import shutil
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    src_data = src.parent / (src.name + "_data")
    if src_data.exists():
        dst_data = dst.parent / (dst.name + "_data")
        shutil.copy(src_data, dst_data)
        return dst_data.name
    return None


def acquire_onnx(hf_repo: str, precision: str, out_dir: Path) -> dict:
    """Acquire prebuilt onnx-community towers for the target precision (else optimum fallback).

    Downloads the matching .onnx (+ any external .onnx_data sibling) + tokenizer.json, then
    normalizes to canonical bundle paths: out_dir/vision_model.onnx[+_data],
    out_dir/text_model.onnx[+_data], out_dir/tokenizer.json. No in-process precision conversion
    for the prebuilt path (onnx-community ships both fp32 and *_fp16 variants).

    Returns: {vision, text, tokenizer: Path, vision_data, text_data: str|None,
              source, source_repo, source_revision}. Network + disk heavy — integration only.
    """
    from huggingface_hub import snapshot_download, model_info
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    out_dir.mkdir(parents=True, exist_ok=True)
    repo = onnx_community_repo(hf_repo)
    v_name, t_name = _tower_basenames(precision)
    vision = out_dir / "vision_model.onnx"
    text = out_dir / "text_model.onnx"
    tok = out_dir / "tokenizer.json"
    try:
        snap = Path(snapshot_download(repo, allow_patterns=[
            f"onnx/{v_name}", f"onnx/{v_name}_data",
            f"onnx/{t_name}", f"onnx/{t_name}_data",
            "tokenizer.json",
        ]))
        v_data = _place(snap / "onnx" / v_name, vision)
        t_data = _place(snap / "onnx" / t_name, text)
        import shutil
        shutil.copy(snap / "tokenizer.json", tok)
        return {"vision": vision, "text": text, "tokenizer": tok,
                "vision_data": v_data, "text_data": t_data,
                "source": "onnx-community", "source_repo": repo,
                "source_revision": model_info(repo).sha}
    except (EntryNotFoundError, RepositoryNotFoundError):
        export_dir = out_dir / "optimum_export"
        subprocess.run(
            ["optimum-cli", "export", "onnx",
             "--model", hf_repo, "--task", "zero-shot-image-classification",
             str(export_dir)],
            check=True,
        )
        produced = {p.name for p in export_dir.glob("*.onnx")}
        assert "vision_model.onnx" in produced and "text_model.onnx" in produced, (
            f"optimum export missing towers; produced: {sorted(produced)}")
        v_data = _place(export_dir / "vision_model.onnx", vision)
        t_data = _place(export_dir / "text_model.onnx", text)
        import shutil
        shutil.copy(export_dir / "tokenizer.json", tok)
        return {"vision": vision, "text": text, "tokenizer": tok,
                "vision_data": v_data, "text_data": t_data,
                "source": "optimum", "source_repo": "", "source_revision": ""}
