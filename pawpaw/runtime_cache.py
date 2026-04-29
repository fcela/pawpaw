"""Local cache for base-model GGUF files used by pawpaw.runtime.

Bundles ship with `adapter.gguf` but NOT the base model — that would bloat each
program by 600 MB+. The base GGUF is downloaded once on first use and cached
under ~/.cache/pawpaw/base_models/.

A small registry maps HuggingFace model IDs to known-good GGUF quantizations.
Users can override with environment variables or by passing an explicit path.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

from huggingface_hub import hf_hub_download


class GGUFRef(NamedTuple):
    repo_id: str
    filename: str


_AVAILABLE_QUANTS = ("Q4_K_M", "Q6_K", "Q8_0")
_preferred_quant: str | None = None


def set_preferred_quant(quant: str) -> None:
    """Set the preferred quantization level for base model downloads.

    Must be one of: Q4_K_M, Q6_K, Q8_0.
    """
    if quant not in _AVAILABLE_QUANTS:
        raise ValueError(f"unknown quant {quant!r}; choose from {_AVAILABLE_QUANTS}")
    global _preferred_quant
    _preferred_quant = quant


def _effective_quant() -> str:
    env = os.environ.get("PAWPAW_BASE_QUANT")
    if env:
        if env not in _AVAILABLE_QUANTS:
            raise ValueError(f"PAWPAW_BASE_QUANT={env!r}; choose from {_AVAILABLE_QUANTS}")
        return env
    if _preferred_quant:
        return _preferred_quant
    return "Q6_K"


_KNOWN_GGUFS: dict[str, dict[str, GGUFRef]] = {
    "Qwen/Qwen3-0.6B": {
        "Q4_K_M": GGUFRef("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf"),
        "Q6_K": GGUFRef("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q6_K.gguf"),
        "Q8_0": GGUFRef("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"),
    },
    "Qwen/Qwen3-1.7B": {
        "Q4_K_M": GGUFRef("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf"),
        "Q6_K": GGUFRef("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q6_K.gguf"),
        "Q8_0": GGUFRef("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q8_0.gguf"),
    },
    "Qwen/Qwen3-4B": {
        "Q4_K_M": GGUFRef("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
        "Q6_K": GGUFRef("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q6_K.gguf"),
        "Q8_0": GGUFRef("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf"),
    },
}

KNOWN_GGUFS: dict[str, GGUFRef] = {
    model: quants["Q6_K"] for model, quants in _KNOWN_GGUFS.items()
}


def _cache_root() -> Path:
    return Path(os.environ.get("PAWPAW_CACHE", Path.home() / ".cache" / "pawpaw"))


def base_models_dir() -> Path:
    d = _cache_root() / "base_models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_base_model_gguf(interpreter: str, *, override_path: str | None = None) -> Path:
    """Return a local path to a GGUF file for the named interpreter.

    Resolution order:
    1. `override_path` if provided.
    2. `PAWPAW_BASE_MODEL_<NAME>` env var (NAME is interpreter with non-alphanumerics → '_').
    3. _KNOWN_GGUFS registry (respects preferred quant) → download via huggingface_hub.
    """
    if override_path:
        p = Path(override_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"override base model not found: {p}")
        return p

    env_key = "PAWPAW_BASE_MODEL_" + "".join(c if c.isalnum() else "_" for c in interpreter).upper()
    env_path = os.environ.get(env_key)
    if env_path:
        p = Path(env_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"{env_key} → {p} does not exist")
        return p

    if interpreter not in _KNOWN_GGUFS:
        raise KeyError(
            f"no GGUF mirror registered for interpreter {interpreter!r}. "
            f"Set {env_key} to a local GGUF path, or add to pawpaw.runtime_cache.KNOWN_GGUFS."
        )

    quant = _effective_quant()
    quants = _KNOWN_GGUFS[interpreter]
    if quant not in quants:
        raise KeyError(
            f"quant {quant!r} not available for {interpreter!r}; "
            f"available: {sorted(quants.keys())}"
        )

    ref = quants[quant]
    return Path(hf_hub_download(repo_id=ref.repo_id, filename=ref.filename, cache_dir=base_models_dir()))
