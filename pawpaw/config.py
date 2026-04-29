"""Compile-time options. Pure dataclasses — no I/O."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Tuple

DEFAULT_GENERATION_CONFIG = {
    "max_new_tokens": 256,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 50,
}

DEFAULT_N_CTX = 1024


def auto_n_threads() -> int:
    """Return a sensible default thread count for llama.cpp on this machine.

    Uses physical cores (not logical/hyperthreaded) to avoid contention.
    Falls back to half of logical cores if physical count is unavailable.
    Respects the PAWPAW_N_THREADS env var if set.
    """
    env = os.environ.get("PAWPAW_N_THREADS")
    if env is not None:
        return int(env)
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        if physical and physical > 0:
            return min(physical, 8)
    except ImportError:
        pass
    logical = os.cpu_count() or 4
    return max(1, min(logical // 2, 8))


@dataclass(frozen=True)
class SynthConfig:
    n_per_category: int = 30
    dedup_threshold: float = 0.85
    min_examples: int = 100
    taxonomy_prompt_version: str = "v2"
    examples_prompt_version: str = "v2"
    llm_model_path: str | None = None
    llm_seed: int = 42
    llm_n_threads: int | None = None
    llm_n_batch: int = 512
    llm_n_gpu_layers: int | None = None

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]


_QUALITY_PRESETS = {
    "draft": {"lora_rank": 4, "epochs": 1, "per_device_batch_size": 2, "gradient_accumulation_steps": 8},
    "standard": {"lora_rank": 16, "epochs": 3, "per_device_batch_size": 4, "gradient_accumulation_steps": 4},
    "production": {"lora_rank": 32, "epochs": 5, "per_device_batch_size": 4, "gradient_accumulation_steps": 4},
}


@dataclass(frozen=True)
class TrainConfig:
    lora_rank: int = 16
    lora_alpha: int | None = None
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    seed: int = 42
    val_fraction: float = 0.1
    max_length: int = 1024

    @property
    def effective_alpha(self) -> int:
        if self.lora_alpha is not None:
            return self.lora_alpha
        return 2 * self.lora_rank

    @classmethod
    def preset(cls, name: str, **overrides) -> TrainConfig:
        """Create a TrainConfig from a named preset: 'draft', 'standard', or 'production'.

        Any field can be overridden:
            TrainConfig.preset("draft", epochs=2)
        """
        if name not in _QUALITY_PRESETS:
            raise ValueError(f"unknown preset {name!r}; choose from {sorted(_QUALITY_PRESETS)}")
        return cls(**{**_QUALITY_PRESETS[name], **overrides})


@dataclass(frozen=True)
class CompileOptions:
    base_model: str = "Qwen/Qwen3-0.6B"
    synth: SynthConfig = field(default_factory=SynthConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
