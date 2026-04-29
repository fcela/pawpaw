from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from pawpaw import format as paw_format
from pawpaw.pack.paw_file import write_paw_file


def _make_fake_peft_dir(tmp_path: Path) -> Path:
    d = tmp_path / "peft"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj"],
        "base_model_name_or_path": "Qwen/Qwen3-0.6B",
    }))
    weights = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(8, 4),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(4, 8),
    }
    save_file(weights, str(d / "adapter_model.safetensors"))
    return d


def test_write_paw_file_round_trip(tmp_path):
    peft = _make_fake_peft_dir(tmp_path)
    out = tmp_path / "out.paw"

    write_paw_file(
        out_path=out,
        peft_dir=peft,
        spec="test spec",
        prompt_template="prefix {INPUT_PLACEHOLDER} suffix",
        examples=[{"input": "i", "output": "o"}],
        interpreter_model="Qwen/Qwen3-0.6B",
    )

    assert paw_format._is_paw_file(out)
    tensors, metadata = paw_format.load(out)
    assert metadata["spec"] == "test spec"
    assert metadata["interpreter_model"] == "Qwen/Qwen3-0.6B"
    assert metadata["source"] == "peft"
    assert any(k.startswith("lora_") for k in tensors)


def test_write_paw_file_strips_placeholder_from_pseudo_program(tmp_path):
    peft = _make_fake_peft_dir(tmp_path)
    out = tmp_path / "out.paw"
    write_paw_file(
        out_path=out,
        peft_dir=peft,
        spec="s",
        prompt_template="prefix {INPUT_PLACEHOLDER} suffix",
        examples=[],
        interpreter_model="Qwen/Qwen3-0.6B",
    )
    _, metadata = paw_format.load(out)
    assert "{INPUT_PLACEHOLDER}" not in metadata.get("pseudo_program", "")
