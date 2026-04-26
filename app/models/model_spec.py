from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    pretrained: str
    cache_dir: str

    @classmethod
    def from_baked_metadata(cls, path: str | Path) -> "ModelSpec":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            model_name=data["model_name"],
            pretrained=data["pretrained"],
            cache_dir=data["cache_dir"],
        )

    @classmethod
    def default(cls) -> "ModelSpec":
        return cls(
            model_name="ViT-L-14",
            pretrained="laion2b_s32b_b82k",
            cache_dir="/app/models",
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "model_name": self.model_name,
                "pretrained": self.pretrained,
                "cache_dir": self.cache_dir,
            }
        )
