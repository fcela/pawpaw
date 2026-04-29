from __future__ import annotations

import json

import pytest

from pawpaw.config import CompileOptions, SynthConfig, TrainConfig
from pawpaw.pipeline import compile_spec, PipelineHooks
from pawpaw.synth.examples import Pair


class StubLLM:
    """Deterministic stub that returns canned JSON for taxonomy and examples calls."""

    def __init__(self):
        self.taxonomy = json.dumps({
            "categories": [{"name": "a", "description": "d", "weight": 1.0}]
        })
        self.examples = json.dumps({
            "pairs": [{"input": f"in{i}", "output": f"out{i}"} for i in range(120)]
        })

    def complete(self, prompt, *, max_tokens=1024, temperature=0.0):
        if "Enumerate" in prompt or "categories" in prompt.lower():
            return self.taxonomy
        return self.examples


def _stub_train(*, base_model, template, pairs, config, output_dir, max_length=1024):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "r": config.lora_rank, "lora_alpha": config.effective_alpha,
        "target_modules": list(config.target_modules),
        "base_model_name_or_path": base_model,
    }))
    import torch
    from safetensors.torch import save_file
    save_file(
        {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(config.lora_rank, 4),
         "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(4, config.lora_rank)},
        str(output_dir / "adapter_model.safetensors"),
    )
    return output_dir


def _stub_gguf(peft_dir, *, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "adapter.gguf"
    p.write_bytes(b"FAKEGGUF")
    return p


def test_compile_spec_end_to_end_with_stubs(tmp_path):
    options = CompileOptions(
        synth=SynthConfig(n_per_category=120, dedup_threshold=0.95, min_examples=10),
        train=TrainConfig(epochs=1, per_device_batch_size=1, gradient_accumulation_steps=1),
    )
    hooks = PipelineHooks(
        make_llm=lambda opts: StubLLM(),
        train_lora=_stub_train,
        peft_to_gguf=_stub_gguf,
    )

    out_paw = tmp_path / "out.paw"
    bundle_dir = tmp_path / "bundle"
    result = compile_spec(
        spec="classify",
        options=options,
        out_paw_path=out_paw,
        bundle_dir=bundle_dir,
        cache_root=tmp_path / "cache",
        hooks=hooks,
    )
    assert out_paw.exists()
    assert (bundle_dir / "adapter.gguf").exists()
    assert (bundle_dir / "prompt_template.txt").exists()
    assert (bundle_dir / "meta.json").exists()
    assert result.paw_path.exists()


def test_pipeline_retries_train_on_oom(tmp_path):
    """First train_lora call raises OOM; pipeline retries with halved batch / doubled accum."""

    import torch

    seen_configs: list[TrainConfig] = []

    def flaky_train(*, base_model, template, pairs, config, output_dir, max_length=1024):
        seen_configs.append(config)
        if len(seen_configs) == 1:
            raise RuntimeError("CUDA out of memory")
        return _stub_train(
            base_model=base_model, template=template, pairs=pairs,
            config=config, output_dir=output_dir, max_length=max_length,
        )

    options = CompileOptions(
        synth=SynthConfig(n_per_category=120, dedup_threshold=0.95, min_examples=10),
        train=TrainConfig(epochs=1, per_device_batch_size=8, gradient_accumulation_steps=2),
    )
    hooks = PipelineHooks(
        make_llm=lambda opts: StubLLM(),
        train_lora=flaky_train,
        peft_to_gguf=_stub_gguf,
    )
    compile_spec(
        spec="classify",
        options=options,
        out_paw_path=tmp_path / "out.paw",
        bundle_dir=tmp_path / "bundle",
        cache_root=tmp_path / "cache",
        hooks=hooks,
    )
    assert len(seen_configs) == 2
    assert seen_configs[1].per_device_batch_size == 4
    assert seen_configs[1].gradient_accumulation_steps == 4


def test_pipeline_aborts_on_second_oom(tmp_path):
    def always_oom(**kwargs):
        raise RuntimeError("CUDA out of memory")

    options = CompileOptions(
        synth=SynthConfig(n_per_category=120, dedup_threshold=0.95, min_examples=10),
        train=TrainConfig(epochs=1, per_device_batch_size=8, gradient_accumulation_steps=2),
    )
    hooks = PipelineHooks(
        make_llm=lambda opts: StubLLM(),
        train_lora=always_oom,
        peft_to_gguf=_stub_gguf,
    )
    with pytest.raises(RuntimeError, match="out of memory"):
        compile_spec(
            spec="classify",
            options=options,
            out_paw_path=tmp_path / "out.paw",
            bundle_dir=tmp_path / "bundle",
            cache_root=tmp_path / "cache",
            hooks=hooks,
        )


def test_compile_spec_aborts_on_too_few_examples(tmp_path):
    options = CompileOptions(
        synth=SynthConfig(n_per_category=2, dedup_threshold=0.5, min_examples=1000),
    )
    hooks = PipelineHooks(
        make_llm=lambda opts: StubLLM(),
        train_lora=_stub_train,
        peft_to_gguf=_stub_gguf,
    )
    with pytest.raises(ValueError, match="too few"):
        compile_spec(
            spec="classify",
            options=options,
            out_paw_path=tmp_path / "out.paw",
            bundle_dir=tmp_path / "bundle",
            cache_root=tmp_path / "cache",
            hooks=hooks,
        )
