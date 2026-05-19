from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

import psutil
import torch

from app.models.base_model import BaseModel
from app.models.siglip2_model import SigLip2Model


class NoModelLoadedError(Exception):
    pass


class ModelNotCachedError(Exception):
    """Raised when loading an uncached model in offline mode."""
    pass


class InsufficientResourcesError(Exception):
    """Raised when host RAM/VRAM is below model's minimum requirements."""
    pass


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    display_name: str
    model_type: str
    hf_repo: str
    params: str
    resolution: int | str
    revision: str | None = None
    min_ram_gb: int | None = None
    min_vram_gb: int | None = None


@dataclass
class ModelLease:
    model: BaseModel


SIGLIP2_REGISTRY: dict[str, ModelConfig] = {
    "siglip2-base-patch16-256": ModelConfig(
        model_id="siglip2-base-patch16-256",
        display_name="SigLIP2 Base (256px)",
        model_type="siglip2",
        hf_repo="google/siglip2-base-patch16-256",
        params="0.4B",
        resolution=256,
    ),
    "siglip2-base-patch16-384": ModelConfig(
        model_id="siglip2-base-patch16-384",
        display_name="SigLIP2 Base (384px)",
        model_type="siglip2",
        hf_repo="google/siglip2-base-patch16-384",
        params="0.4B",
        resolution=384,
    ),
    "siglip2-large-patch16-256": ModelConfig(
        model_id="siglip2-large-patch16-256",
        display_name="SigLIP2 Large (256px)",
        model_type="siglip2",
        hf_repo="google/siglip2-large-patch16-256",
        params="0.9B",
        resolution=256,
    ),
    "siglip2-large-patch16-384": ModelConfig(
        model_id="siglip2-large-patch16-384",
        display_name="SigLIP2 Large (384px)",
        model_type="siglip2",
        hf_repo="google/siglip2-large-patch16-384",
        params="0.9B",
        resolution=384,
    ),
    "siglip2-so400m-patch14-384": ModelConfig(
        model_id="siglip2-so400m-patch14-384",
        display_name="SigLIP2 SO400M (384px)",
        model_type="siglip2",
        hf_repo="google/siglip2-so400m-patch14-384",
        params="1B",
        resolution=384,
    ),
    "siglip2-so400m-patch16-512": ModelConfig(
        model_id="siglip2-so400m-patch16-512",
        display_name="SigLIP2 SO400M (512px)",
        model_type="siglip2",
        hf_repo="google/siglip2-so400m-patch16-512",
        params="1B",
        resolution=512,
    ),
    "siglip2-large-patch16-512": ModelConfig(
        model_id="siglip2-large-patch16-512",
        display_name="SigLIP2 Large (512px)",
        model_type="siglip2",
        hf_repo="google/siglip2-large-patch16-512",
        params="0.9B",
        resolution=512,
        min_ram_gb=4,
    ),
    "siglip2-giant-opt-patch16-384": ModelConfig(
        model_id="siglip2-giant-opt-patch16-384",
        display_name="SigLIP2 Giant-Opt (384px)",
        model_type="siglip2",
        hf_repo="google/siglip2-giant-opt-patch16-384",
        params="~2B",
        resolution=384,
        min_ram_gb=10,
    ),
}


class ModelManager:
    def __init__(self, cache_dir: str, offline: bool = False):
        self.registry = dict(SIGLIP2_REGISTRY)
        self.active_model: BaseModel | None = None
        self.active_model_id: str | None = None
        self.cache_dir = cache_dir
        self._offline = offline
        self._condition = asyncio.Condition()
        self._swapping = False
        self._active_leases = 0

    @asynccontextmanager
    async def acquire(self, timeout: float) -> AsyncGenerator[ModelLease, None]:
        async with asyncio.timeout(timeout):
            async with self._condition:
                await self._condition.wait_for(lambda: not self._swapping)
                if self.active_model is None:
                    raise NoModelLoadedError()
                self._active_leases += 1
                model_ref = self.active_model

        try:
            yield ModelLease(model=model_ref)
        finally:
            async with self._condition:
                self._active_leases -= 1
                if self._active_leases == 0:
                    self._condition.notify_all()

    async def load_model(self, model_id: str) -> None:
        config = self.registry[model_id]
        self._preflight_check(config)

        async with self._condition:
            await self._condition.wait_for(lambda: not self._swapping)
            if self.active_model_id == model_id:
                return
            self._swapping = True
            self._condition.notify_all()
            await self._condition.wait_for(lambda: self._active_leases == 0)
            old_model = self.active_model
            self.active_model = None
            self.active_model_id = None

        try:
            del old_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            new_model = SigLip2Model(
                hf_repo=config.hf_repo, cache_dir=self.cache_dir
            )
        except Exception:
            async with self._condition:
                self._swapping = False
                self._condition.notify_all()
            raise

        async with self._condition:
            self.active_model = new_model
            self.active_model_id = model_id
            self._swapping = False
            self._condition.notify_all()

    def _preflight_check(self, config: ModelConfig) -> None:
        self._check_resources(config)

    def _check_resources(self, config: ModelConfig) -> None:
        if config.min_ram_gb is None:
            return

        mem = psutil.virtual_memory()
        required_bytes = config.min_ram_gb * 1.2 * 1e9

        if required_bytes > mem.total:
            raise InsufficientResourcesError(
                f"Model {config.model_id} requires ~{config.min_ram_gb}GB RAM "
                f"(with 1.2x headroom), but system total is "
                f"{mem.total / 1e9:.1f}GB"
            )

        estimated_available = mem.available
        if self.active_model_id:
            active_config = self.registry.get(self.active_model_id)
            if active_config and active_config.min_ram_gb:
                estimated_available += active_config.min_ram_gb * 1e9

        if required_bytes > estimated_available:
            raise InsufficientResourcesError(
                f"Model {config.model_id} requires ~{config.min_ram_gb}GB RAM "
                f"(with 1.2x headroom), but estimated post-unload available is "
                f"{estimated_available / 1e9:.1f}GB"
            )

        if config.min_vram_gb and torch.cuda.is_available():
            free_vram, total_vram = torch.cuda.mem_get_info()
            vram_required = config.min_vram_gb * 1.2 * 1e9
            if vram_required > total_vram:
                raise InsufficientResourcesError(
                    f"Model {config.model_id} requires ~{config.min_vram_gb}GB VRAM, "
                    f"but GPU total is {total_vram / 1e9:.1f}GB"
                )

    def list_models(self) -> list[dict]:
        result = []
        for config in self.registry.values():
            cached = self._is_cached(config)
            result.append({
                "model_id": config.model_id,
                "display_name": config.display_name,
                "model_type": config.model_type,
                "params": config.params,
                "resolution": config.resolution,
                "loaded": self.active_model_id == config.model_id,
                "cached": cached,
            })
        return result

    def _is_cached(self, config: ModelConfig) -> bool:
        cache_path = Path(self.cache_dir) / f"models--{config.hf_repo.replace('/', '--')}"
        marker_path = cache_path / ".validated"
        if not marker_path.exists():
            return False
        try:
            marker = json.loads(marker_path.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        if marker.get("model_id") != config.model_id:
            return False
        if marker.get("hf_repo") != config.hf_repo:
            return False
        if config.revision and marker.get("revision") != config.revision:
            return False
        return True
