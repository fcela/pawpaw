"""Single-file .paw writer. Reads a PEFT adapter dir and writes the v2 binary."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as load_safetensors

from pawpaw import format as paw_format
from pawpaw.train.prompt_template import INPUT_PLACEHOLDER

PEFT_PREFIXES = ("base_model.model.", "base_model.")
_WEIGHT_SUFFIX = ".weight"


def strip_peft_prefix(name: str) -> str:
    """Strip PEFT-specific prefixes so tensor keys are portable."""
    for prefix in PEFT_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _normalize_lora_name(name: str) -> str:
    """Strip PEFT-specific prefixes/suffixes so tensor keys are portable."""
    clean = strip_peft_prefix(name)
    if clean.endswith(_WEIGHT_SUFFIX):
        clean = clean[: -len(_WEIGHT_SUFFIX)]
    return clean


def _load_peft_dir(peft_dir: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cfg_path = peft_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {peft_dir}")
    config = json.loads(cfg_path.read_text())

    weights_path = peft_dir / "adapter_model.safetensors"
    bin_path = peft_dir / "adapter_model.bin"
    if weights_path.exists():
        raw = load_safetensors(str(weights_path))
    elif bin_path.exists():
        raw = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"no adapter weights found in {peft_dir}")

    return config, {_normalize_lora_name(k): v for k, v in raw.items()}


def write_paw_file(
    *,
    out_path: Path,
    peft_dir: Path,
    spec: str,
    prompt_template: str,
    examples: list[dict],
    interpreter_model: str,
    description: str = "",
    author: str = "",
    tags: list[str] | None = None,
    generation_config: dict | None = None,
) -> Path:
    """Write a v2 .paw container directly from a PEFT adapter directory."""
    config, lora_weights = _load_peft_dir(peft_dir)

    lora_config: dict[str, Any] = {
        "rank": config.get("r", config.get("rank", 0)),
        "alpha": config.get("lora_alpha", config.get("r", 0)),
        "target_modules": config.get("target_modules", []),
        "peft_type": config.get("peft_type", "LORA"),
        "task_type": config.get("task_type", ""),
    }

    pseudo_program = prompt_template.replace(INPUT_PLACEHOLDER, "")

    paw_format.save_program(
        filepath=str(out_path),
        spec=spec,
        base_model=interpreter_model,
        lora_weights=lora_weights,
        lora_config=lora_config,
        pseudo_program=pseudo_program,
        generation_config=generation_config,
        description=description,
        author=author,
        tags=tags,
        examples=examples,
        source="peft",
        source_info={
            "training_method": "peft",
            "pipeline": "pawpaw",
            "base_model": interpreter_model,
            "adapter_path": str(peft_dir),
        },
    )
    return Path(out_path)
