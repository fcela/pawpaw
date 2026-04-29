"""Writes the runtime-loadable directory bundle: adapter.gguf + prompt_template.txt + meta.json.

The schema mirrors what programasweights/runtime_llamacpp.py reads.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from pawpaw.config import DEFAULT_GENERATION_CONFIG
from pawpaw.train.prompt_template import INPUT_PLACEHOLDER


@dataclass(frozen=True)
class BundleMeta:
    spec: str
    interpreter_model: str
    spec_hash: str
    pipeline_version: str
    lora_rank: int
    lora_alpha: int
    target_modules: tuple[str, ...]
    examples: list[dict]


def _meta_json(meta: BundleMeta) -> dict:
    return {
        "spec": meta.spec,
        "interpreter_model": meta.interpreter_model,
        "format_version": 2,
        "kind": "neural_program",
        "source": "peft",
        "source_info": {
            "training_method": "peft",
            "pipeline": "pawpaw",
            "pipeline_version": meta.pipeline_version,
            "spec_hash": meta.spec_hash,
        },
        "lora_config": {
            "rank": meta.lora_rank,
            "alpha": meta.lora_alpha,
            "target_modules": list(meta.target_modules),
        },
        "generation_config": DEFAULT_GENERATION_CONFIG,
        "examples": meta.examples,
    }


def write_directory(
    *,
    out_dir: Path,
    gguf_path: Path,
    prompt_template: str,
    meta: BundleMeta,
) -> Path:
    if INPUT_PLACEHOLDER not in prompt_template:
        raise ValueError(f"prompt_template missing placeholder {INPUT_PLACEHOLDER}")
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(gguf_path, out_dir / "adapter.gguf")
    (out_dir / "prompt_template.txt").write_text(prompt_template, encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(_meta_json(meta), indent=2), encoding="utf-8")
    return out_dir
