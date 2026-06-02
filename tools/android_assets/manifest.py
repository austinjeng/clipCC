from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Optional

SCHEMA_VERSION = 1
PROFILE = "benchmark-v1"
SCORE_SEMANTICS = "siglip2_pairwise_sigmoid"


@dataclass
class FileRef:
    file: str
    data_file: Optional[str]
    bytes: int
    sha256: str
    data_sha256: Optional[str] = None   # sha256 of the external .onnx_data, when present


@dataclass
class ModelBundleManifest:
    model_id: str
    hf_repo: str
    hf_revision: str                 # google/ source-repo sha (logit_scale provenance)
    onnx_source: str                 # "onnx-community" | "optimum"
    onnx_source_repo: str            # repo the shipped ONNX bytes came from
    onnx_source_revision: str        # sha of that repo ("" for optimum local export)
    display_name: str
    params: str
    resolution: int
    precision: str                   # "fp32" | "fp16"
    ram_budget_mb: int
    transformers_version: str
    logit_scale: float
    logit_bias: float
    vision: FileRef
    text: FileRef
    tokenizer_sha256: str
    tokenizer_padding_side: str = "right"
    tokenizer_lowercase_applied_by: str = "unknown"  # "tokenizer_json" | "kotlin_wrapper"

    def to_json(self) -> str:
        doc = {
            "schema_version": SCHEMA_VERSION,
            "profile": PROFILE,
            "model_id": self.model_id,
            "hf_repo": self.hf_repo,
            "hf_revision": self.hf_revision,
            "onnx_source": self.onnx_source,
            "onnx_source_repo": self.onnx_source_repo,
            "onnx_source_revision": self.onnx_source_revision,
            "display_name": self.display_name,
            "params": self.params,
            "resolution": self.resolution,
            "precision": self.precision,
            "ram_budget_mb": self.ram_budget_mb,
            "transformers_version": self.transformers_version,
            "score_semantics": SCORE_SEMANTICS,
            "logit_scale": self.logit_scale,
            "logit_bias": self.logit_bias,
            "vision": asdict(self.vision),
            "text": asdict(self.text),
            "tokenizer": {
                "file": "tokenizer.json",
                "sha256": self.tokenizer_sha256,
                "max_length": 64,
                "pad_id": 0,
                "padding": "max_length",
                "padding_side": self.tokenizer_padding_side,
                "truncation": True,
                "lowercase_applied_by": self.tokenizer_lowercase_applied_by,
            },
            "preprocess": {
                "resize": "stretch_square",
                "resample": "bicubic",
                "rescale": 0.00392156862745098,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
                "layout": "CHW",
            },
            "frame_pipeline": {
                "fps": 1.0,
                "max_frames": 300,
                "prescale": "none",
                "intermediate_codec": "none",
                "resample": "bicubic",
            },
        }
        return json.dumps(doc, indent=2)

    @classmethod
    def from_json(cls, blob: str) -> "ModelBundleManifest":
        d = json.loads(blob)
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {d.get('schema_version')!r}; expected {SCHEMA_VERSION}"
            )
        tok = d["tokenizer"]
        return cls(
            model_id=d["model_id"],
            hf_repo=d["hf_repo"],
            hf_revision=d["hf_revision"],
            onnx_source=d["onnx_source"],
            onnx_source_repo=d["onnx_source_repo"],
            onnx_source_revision=d["onnx_source_revision"],
            display_name=d["display_name"],
            params=d["params"],
            resolution=d["resolution"],
            precision=d["precision"],
            ram_budget_mb=d["ram_budget_mb"],
            transformers_version=d["transformers_version"],
            logit_scale=d["logit_scale"],
            logit_bias=d["logit_bias"],
            vision=FileRef(**d["vision"]),
            text=FileRef(**d["text"]),
            tokenizer_sha256=tok["sha256"],
            tokenizer_padding_side=tok["padding_side"],
            tokenizer_lowercase_applied_by=tok["lowercase_applied_by"],
        )
