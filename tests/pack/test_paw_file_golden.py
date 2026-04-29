"""Golden test: a tiny .paw built from a frozen LoRA tensor dict round-trips to the same metadata.

We assert structural equality (metadata + tensor names + shapes), not byte-equality, because
the safetensors blob includes a JSON header whose dict ordering is implementation-defined.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from pawpaw import format as paw_format
from pawpaw.pack.paw_file import write_paw_file


def _frozen_peft_dir(tmp_path: Path) -> Path:
    d = tmp_path / "peft"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "r": 4, "lora_alpha": 8, "target_modules": ["q_proj"],
        "base_model_name_or_path": "Qwen/Qwen3-0.6B",
    }, sort_keys=True))
    weights = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.full((4, 8), 0.5),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.full((8, 4), 0.25),
    }
    save_file(weights, str(d / "adapter_model.safetensors"))
    return d


def test_paw_file_golden_roundtrip(tmp_path):
    peft = _frozen_peft_dir(tmp_path)
    out = tmp_path / "golden.paw"
    write_paw_file(
        out_path=out,
        peft_dir=peft,
        spec="frozen-spec",
        prompt_template="P {INPUT_PLACEHOLDER} S",
        examples=[{"input": "i", "output": "o"}],
        interpreter_model="Qwen/Qwen3-0.6B",
    )

    tensors, metadata = paw_format.load(out)

    assert any("lora_A" in k for k in tensors)
    assert any("lora_B" in k for k in tensors)
    a_keys = [k for k in tensors if "lora_A" in k]
    b_keys = [k for k in tensors if "lora_B" in k]
    assert tensors[a_keys[0]].shape == (4, 8)
    assert tensors[b_keys[0]].shape == (8, 4)
    assert torch.allclose(tensors[a_keys[0]], torch.full((4, 8), 0.5))

    assert metadata["spec"] == "frozen-spec"
    assert metadata["interpreter_model"] == "Qwen/Qwen3-0.6B"
    assert metadata["source"] == "peft"
    assert metadata["lora_config"]["rank"] == 4
    assert metadata["lora_config"]["alpha"] == 8
    assert metadata["pseudo_program"] == "P  S"
    assert metadata["has_lora"] is True
