from __future__ import annotations

import json

import pytest

from pawpaw.pack.bundle import BundleMeta, write_directory


def test_write_directory_creates_expected_files(tmp_path):
    fake_gguf = tmp_path / "adapter.gguf"
    fake_gguf.write_bytes(b"\x47\x47\x55\x46")

    out = tmp_path / "bundle"
    meta = BundleMeta(
        spec="classify",
        interpreter_model="Qwen/Qwen3-0.6B",
        spec_hash="hh",
        pipeline_version="0.1.0",
        lora_rank=16,
        lora_alpha=32,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        examples=[{"input": "x", "output": "y"}],
    )
    write_directory(out_dir=out, gguf_path=fake_gguf, prompt_template="prefix {INPUT_PLACEHOLDER} suffix", meta=meta)

    assert (out / "adapter.gguf").read_bytes() == b"\x47\x47\x55\x46"
    template_text = (out / "prompt_template.txt").read_text()
    assert "{INPUT_PLACEHOLDER}" in template_text

    meta_json = json.loads((out / "meta.json").read_text())
    assert meta_json["spec"] == "classify"
    assert meta_json["interpreter_model"] == "Qwen/Qwen3-0.6B"
    assert meta_json["lora_config"]["rank"] == 16
    assert meta_json["source"] == "peft"


def test_write_directory_rejects_template_without_placeholder(tmp_path):
    fake_gguf = tmp_path / "adapter.gguf"
    fake_gguf.write_bytes(b"x")
    meta = BundleMeta(spec="s", interpreter_model="m", spec_hash="h", pipeline_version="0.1.0",
                      lora_rank=8, lora_alpha=16, target_modules=("q_proj",), examples=[])
    with pytest.raises(ValueError, match="placeholder"):
        write_directory(out_dir=tmp_path / "b", gguf_path=fake_gguf, prompt_template="no marker", meta=meta)
