"""End-to-end smoke test. Requires:
- RUN_GPU_TESTS=1
- A local Qwen3-0.6B base model (HF cache)
- A small local synthesis LLM at PAW_TEST_LLM_GGUF
"""
from __future__ import annotations

import os

import pytest

from pawpaw.config import CompileOptions, SynthConfig, TrainConfig
from pawpaw.pipeline import compile_spec


@pytest.mark.skipif(
    "PAW_TEST_LLM_GGUF" not in os.environ,
    reason="set PAW_TEST_LLM_GGUF to a local synth model path",
)
def test_compile_toy_spec(tmp_path):

    from pawpaw import load as paw_load

    spec = (
        "Classify each input as either 'red' (warm/fire/sun things) or 'blue' "
        "(cool/water/sky things). Output exactly 'red' or 'blue', no other text."
    )
    options = CompileOptions(
        synth=SynthConfig(
            n_per_category=20,
            min_examples=80,
            llm_model_path=os.environ["PAW_TEST_LLM_GGUF"],
        ),
        train=TrainConfig(lora_rank=4, lora_alpha=8, epochs=1, per_device_batch_size=2),
    )
    result = compile_spec(
        spec=spec,
        options=options,
        out_paw_path=tmp_path / "redblue.paw",
        bundle_dir=tmp_path / "bundle",
        cache_root=tmp_path / "cache",
    )
    assert result.paw_path.exists()
    fn = paw_load(str(result.paw_path))
    out = fn("the sun was setting in flames")
    assert out.strip().lower() in {"red", "blue"}
