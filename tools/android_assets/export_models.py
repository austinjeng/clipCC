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


EXTERNAL_DATA_THRESHOLD = 2_000_000_000  # under the 2 GiB protobuf limit


def to_fp16(model):
    import onnx
    from onnxconverter_common import float16
    return float16.convert_float_to_float16(model, keep_io_types=True)


def _exceeds_protobuf_limit(model) -> bool:
    """True if the model would exceed the 2 GB protobuf serialization ceiling.

    `ModelProto.ByteSize()` itself RAISES `ValueError(... exceeds maximum protobuf size of
    2GB ...)` for >2 GB models — so catch that and treat it as 'too big' rather than
    serialize-then-compare (which can never return True for the models that need external data).
    """
    try:
        return model.ByteSize() >= EXTERNAL_DATA_THRESHOLD
    except ValueError:
        return True


def save_onnx(model, path, force_external: bool | None = None) -> None:
    """Save ONNX; externalize tensors when the model is large (or forced).

    NOTE: only the **optimum-fallback** path serializes a ModelProto. The normal prebuilt path
    relocates files by copy (see `_place`) and never round-trips through the 2 GB serializer.
    """
    import onnx

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    use_external = force_external if force_external is not None else _exceeds_protobuf_limit(model)
    if use_external:
        # onnx only externalizes tensors with raw_data; convert non-raw initializers first.
        from onnx import numpy_helper
        for init in model.graph.initializer:
            if not init.raw_data:
                arr = numpy_helper.to_array(init)
                init.raw_data = arr.tobytes()
                for field in ("float_data", "int32_data", "int64_data", "double_data",
                              "uint64_data"):
                    init.ClearField(field)
        onnx.save_model(
            model, str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=path.name + "_data",
            size_threshold=0,
        )
    else:
        onnx.save_model(model, str(path))


# Set by Spike 0a (Task 7). "unknown" until then; "tokenizer_json" if the Rust normalizer
# already lowercases, else "kotlin_wrapper" (Android must lowercase before encoding).
LOWERCASE_APPLIED_BY = "kotlin_wrapper"


def extract_logit_params(model) -> tuple[float, float]:
    """Read raw logit_scale and logit_bias (HF SiglipModel applies exp(scale) at runtime)."""
    return float(model.logit_scale.detach().item()), float(model.logit_bias.detach().item())


def build_manifest(spec: "ModelSpec", *, hf_revision: str, transformers_version: str,
                   logit_scale: float, logit_bias: float,
                   vision_path: Path, text_path: Path, tokenizer_path: Path,
                   onnx_source: str, onnx_source_repo: str, onnx_source_revision: str):
    from tools.android_assets.manifest import ModelBundleManifest, FileRef
    from tools.android_assets.hashing import sha256_file

    def _ref(p: Path) -> "FileRef":
        data = p.parent / (p.name + "_data")
        has_data = data.exists()
        return FileRef(
            file=p.name,
            data_file=(data.name if has_data else None),
            bytes=p.stat().st_size,
            sha256=sha256_file(p),
            data_sha256=(sha256_file(data) if has_data else None),
        )

    return ModelBundleManifest(
        model_id=spec.model_id, hf_repo=spec.hf_repo, hf_revision=hf_revision,
        onnx_source=onnx_source, onnx_source_repo=onnx_source_repo,
        onnx_source_revision=onnx_source_revision,
        display_name=spec.display_name, params=spec.params, resolution=spec.resolution,
        precision=spec.precision, ram_budget_mb=spec.ram_budget_mb,
        transformers_version=transformers_version,
        logit_scale=logit_scale, logit_bias=logit_bias,
        vision=_ref(vision_path), text=_ref(text_path),
        tokenizer_sha256=sha256_file(tokenizer_path),
        tokenizer_lowercase_applied_by=LOWERCASE_APPLIED_BY,
    )


def export_one(model_id: str, out_root: Path) -> Path:
    from transformers import AutoModel
    import transformers as _t
    from huggingface_hub import model_info

    spec = PROFILE_MODELS[model_id]
    out_dir = out_root / model_id

    # Acquire prebuilt towers at the target precision (+ tokenizer.json), normalized to
    # canonical bundle paths. No precision conversion / serializer round-trip for prebuilt.
    acquired = acquire_onnx(spec.hf_repo, spec.precision, out_dir)

    # logit_scale / logit_bias come from the torch source model (downloaded to HF cache).
    model = AutoModel.from_pretrained(spec.hf_repo)
    scale, bias = extract_logit_params(model)

    manifest = build_manifest(
        spec,
        hf_revision=model_info(spec.hf_repo).sha,
        transformers_version=_t.__version__,
        logit_scale=scale, logit_bias=bias,
        vision_path=acquired["vision"], text_path=acquired["text"],
        tokenizer_path=acquired["tokenizer"],
        onnx_source=acquired["source"], onnx_source_repo=acquired["source_repo"],
        onnx_source_revision=acquired["source_revision"],
    )
    (out_dir / "manifest.json").write_text(manifest.to_json())
    return out_dir / "manifest.json"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("build/android_assets"))
    ap.add_argument("--models", default=",".join(PROFILE_MODELS))
    args = ap.parse_args()
    for mid in [m.strip() for m in args.models.split(",")]:
        print(f"[{mid}] exporting...")
        print("  manifest:", export_one(mid, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
