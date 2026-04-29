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
    repo_id: str  # HF repo containing the GGUF
    filename: str  # filename within the repo


# Known GGUF mirrors. Keys are canonical interpreter names (HF base model IDs);
# values are pre-quantized GGUFs that match upstream's quality/size tradeoff
# (Q6_K for Qwen3 per upstream ADR-001).
KNOWN_GGUFS: dict[str, GGUFRef] = {
    "Qwen/Qwen3-0.6B": GGUFRef("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q6_K.gguf"),
    "Qwen/Qwen3-1.7B": GGUFRef("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q6_K.gguf"),
    "Qwen/Qwen3-4B": GGUFRef("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q6_K.gguf"),
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
      3. KNOWN_GGUFS registry → download via huggingface_hub.
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

    if interpreter not in KNOWN_GGUFS:
        raise KeyError(
            f"no GGUF mirror registered for interpreter {interpreter!r}. "
            f"Set {env_key} to a local GGUF path, or add to pawpaw.runtime_cache.KNOWN_GGUFS."
        )

    ref = KNOWN_GGUFS[interpreter]
    return Path(hf_hub_download(repo_id=ref.repo_id, filename=ref.filename, cache_dir=base_models_dir()))
