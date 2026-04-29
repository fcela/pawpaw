"""Convert a PEFT/LoRA adapter directory to a GGUF f16 adapter file.

Pure-Python implementation built on the `gguf` PyPI package — no llama.cpp
source or build step required. Output is a fp16 GGUF adapter that the
llama-cpp-python runtime can load via `llama_adapter_lora_init`.

Adapter quantization (e.g. Q4_0) is intentionally skipped; bundles use ~5 MB
f16 adapters which is small enough that quantization adds complexity without
meaningful payoff for classifier-sized programs. The base model GGUF is
separately quantized (Q6_K by default, see runtime_cache.py).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from gguf import GGUFWriter, MODEL_ARCH, TensorNameMap
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors


from pawpaw.pack.paw_file import strip_peft_prefix


class GGUFConvertError(RuntimeError):
    pass


_LORA_RE = re.compile(r"^(?P<base>.+?)\.lora_(?P<ab>[AB])\.weight$")


def _arch_for(model_type: str) -> MODEL_ARCH:
    try:
        return MODEL_ARCH[model_type.upper()]
    except KeyError as e:
        raise GGUFConvertError(f"unsupported base architecture: {model_type!r}") from e


def _resolve_base_config(adapter_config: dict) -> dict:
    base_id = adapter_config.get("base_model_name_or_path")
    if not base_id:
        raise GGUFConvertError("adapter_config.json missing base_model_name_or_path")
    base_dir = Path(snapshot_download(base_id, allow_patterns=["config.json"]))
    return json.loads((base_dir / "config.json").read_text())


def _split_lora_name(name: str) -> tuple[str, str]:
    """Split 'model.layers.5.self_attn.q_proj.lora_A.weight' →
    ('model.layers.5.self_attn.q_proj.weight', 'lora_a')."""
    m = _LORA_RE.match(name)
    if not m:
        raise GGUFConvertError(f"unexpected LoRA tensor name: {name!r}")
    return f"{m['base']}.weight", f"lora_{m['ab'].lower()}"


def _add_arch_metadata(writer: GGUFWriter, arch: str, base_cfg: dict) -> None:
    """Embed enough base-model architecture metadata for llama.cpp's adapter loader."""
    writer.add_uint32(f"{arch}.context_length", base_cfg.get("max_position_embeddings", 8192))
    writer.add_uint32(f"{arch}.embedding_length", base_cfg["hidden_size"])
    writer.add_uint32(f"{arch}.feed_forward_length", base_cfg["intermediate_size"])
    writer.add_uint32(f"{arch}.attention.head_count", base_cfg["num_attention_heads"])
    writer.add_uint32(f"{arch}.attention.head_count_kv",
                      base_cfg.get("num_key_value_heads", base_cfg["num_attention_heads"]))
    writer.add_uint32(f"{arch}.block_count", base_cfg["num_hidden_layers"])
    writer.add_float32(f"{arch}.attention.layer_norm_rms_epsilon",
                       base_cfg.get("rms_norm_eps", 1e-6))
    writer.add_float32(f"{arch}.rope.freq_base", base_cfg.get("rope_theta", 10000.0))
    head_dim = base_cfg.get("head_dim", base_cfg["hidden_size"] // base_cfg["num_attention_heads"])
    writer.add_uint32(f"{arch}.attention.key_length", head_dim)
    writer.add_uint32(f"{arch}.attention.value_length", head_dim)


def peft_to_gguf(peft_dir: Path, *, out_dir: Path) -> Path:
    """Convert a PEFT/LoRA adapter directory to a single f16 GGUF file.

    Returns the path to the produced `adapter.gguf`.
    """
    peft_dir = Path(peft_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "adapter.gguf"

    cfg_path = peft_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise GGUFConvertError(f"adapter_config.json not found in {peft_dir}")
    adapter_cfg = json.loads(cfg_path.read_text())

    weights_path = peft_dir / "adapter_model.safetensors"
    if not weights_path.exists():
        raise GGUFConvertError(f"adapter_model.safetensors not found in {peft_dir}")
    raw_weights = load_safetensors(str(weights_path))

    base_cfg = _resolve_base_config(adapter_cfg)
    arch_enum = _arch_for(base_cfg["model_type"])
    arch_name = arch_enum.name.lower()
    name_map = TensorNameMap(arch_enum, n_blocks=base_cfg["num_hidden_layers"])

    writer = GGUFWriter(str(out_path), arch=arch_name)
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora.alpha", float(adapter_cfg.get("lora_alpha", 16)))
    writer.add_string("general.name", "pawpaw_adapter")

    _add_arch_metadata(writer, arch_name, base_cfg)

    n_added = 0
    for raw_name, tensor in raw_weights.items():
        clean = strip_peft_prefix(raw_name)
        base_weight_name, lora_suffix = _split_lora_name(clean)
        # base_weight_name still has '.weight'; gguf's TensorNameMap drops it for us
        gguf_base = name_map.get_name(base_weight_name[: -len(".weight")])
        if gguf_base is None:
            raise GGUFConvertError(
                f"could not map PEFT tensor to GGUF name: {raw_name} → {base_weight_name}"
            )
        gguf_name = f"{gguf_base}.weight.{lora_suffix}"

        f16 = tensor.detach().to(torch.float16).contiguous().cpu().numpy()
        writer.add_tensor(gguf_name, f16)
        n_added += 1

    if n_added == 0:
        raise GGUFConvertError(f"no LoRA tensors found in {peft_dir}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    return out_path
