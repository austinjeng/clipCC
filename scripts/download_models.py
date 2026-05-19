#!/usr/bin/env python3
"""Pre-download SigLip2 models into a local cache directory.

Usage:
    python scripts/download_models.py --preset dev
    python scripts/download_models.py --preset all
    python scripts/download_models.py --models siglip2-base-patch16-256,siglip2-giant-opt-patch16-384
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.model_manager import SIGLIP2_REGISTRY

DEV_MODELS = [
    "siglip2-base-patch16-256",
    "siglip2-large-patch16-512",
    "siglip2-so400m-patch14-384",
    "siglip2-giant-opt-patch16-384",
]


def resolve_default_cache_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "models"


def get_models_for_preset(preset: str) -> list[str]:
    if preset == "dev":
        return list(DEV_MODELS)
    if preset == "all":
        return list(SIGLIP2_REGISTRY.keys())
    raise ValueError(f"Unknown preset: {preset}")


def write_validated_marker(
    model_dir: Path,
    model_id: str,
    hf_repo: str,
    revision: str,
) -> None:
    marker_data = {
        "schema_version": 1,
        "model_id": model_id,
        "hf_repo": hf_repo,
        "revision": revision,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    marker_path = model_dir / ".validated"
    fd, tmp_path = tempfile.mkstemp(dir=str(model_dir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(marker_data, f, indent=2)
        os.replace(tmp_path, str(marker_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_manifest(cache_dir: Path, records: dict) -> None:
    manifest_path = cache_dir / "manifest.json"
    fd, tmp_path = tempfile.mkstemp(dir=str(cache_dir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(records, f, indent=2)
        os.replace(tmp_path, str(manifest_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def resolve_revision(hf_repo: str) -> str:
    from huggingface_hub import model_info
    info = model_info(hf_repo)
    return info.sha


def download_model(
    model_id: str,
    hf_repo: str,
    cache_dir: Path,
) -> str:
    from transformers import AutoModel, AutoProcessor

    print(f"  Downloading processor for {model_id}...")
    AutoProcessor.from_pretrained(hf_repo, cache_dir=str(cache_dir))

    print(f"  Downloading model for {model_id}...")
    AutoModel.from_pretrained(hf_repo, cache_dir=str(cache_dir))

    revision = resolve_revision(hf_repo)

    print(f"  Validating with local_files_only=True...")
    AutoProcessor.from_pretrained(
        hf_repo, cache_dir=str(cache_dir), local_files_only=True, revision=revision,
    )
    AutoModel.from_pretrained(
        hf_repo, cache_dir=str(cache_dir), local_files_only=True, revision=revision,
    )

    model_dir = cache_dir / f"models--{hf_repo.replace('/', '--')}"
    write_validated_marker(model_dir, model_id, hf_repo, revision)

    return revision


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download SigLip2 models for local development"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory (default: <repo-root>/models)",
    )
    parser.add_argument(
        "--preset",
        choices=["dev", "all"],
        default="dev",
        help="Model set to download (default: dev)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model IDs (overrides --preset)",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or resolve_default_cache_dir()
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.models:
        model_ids = [m.strip() for m in args.models.split(",")]
        for mid in model_ids:
            if mid not in SIGLIP2_REGISTRY:
                print(f"ERROR: Unknown model_id: {mid}")
                print(f"Available: {', '.join(SIGLIP2_REGISTRY.keys())}")
                return 1
    else:
        model_ids = get_models_for_preset(args.preset)

    print(f"Cache directory: {cache_dir}")
    print(f"Models to download: {len(model_ids)}")
    print()

    manifest_records: dict = {}
    failed: list[str] = []

    for model_id in model_ids:
        config = SIGLIP2_REGISTRY[model_id]
        print(f"[{model_id}] ({config.params}, {config.resolution}px)")
        try:
            revision = download_model(model_id, config.hf_repo, cache_dir)
            manifest_records[model_id] = {
                "revision": revision,
                "hf_repo": config.hf_repo,
                "validated_at": datetime.now(timezone.utc).isoformat(),
            }
            print(f"  OK (revision: {revision[:12]}...)")
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(model_id)
        print()

    if manifest_records:
        write_manifest(cache_dir, manifest_records)
        print(f"Manifest written to {cache_dir / 'manifest.json'}")

    print()
    print(f"Results: {len(manifest_records)} succeeded, {len(failed)} failed")
    if failed:
        print(f"Failed models: {', '.join(failed)}")
        return 1

    print()
    print("To use in development:")
    print(f"  export CLIP_CACHE_DIR={cache_dir}")
    print("  export CLIPCC_OFFLINE=1")

    return 0


if __name__ == "__main__":
    sys.exit(main())
