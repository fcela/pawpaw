"""Binary format reader/writer for `.paw` v2 files.

Binary-compatible with the upstream programasweights .paw v2 format:

    [4 bytes] Magic: b"PAW\\x02"
    [4 bytes] Version: uint32 little-endian
    [4 bytes] Metadata length: uint32 little-endian
    [N bytes] Metadata (JSON, UTF-8)
    [M bytes] Tensors (safetensors blob)

The metadata schema mirrors what runtime expects:
  - format_version, kind, interpreter_model, base_model, spec
  - prefix_type, prefix_steps, num_layers, has_lora
  - lora_config (rank, alpha, target_modules)
  - generation_config
  - source ("compiled" | "finetuned" | "peft" | "custom"), source_info
  - examples, tags, description, author
  - pseudo_program (optional discrete prompt prefix)
  - prompt_token_ids (optional)
"""
from __future__ import annotations

import json
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file

from pawpaw.config import DEFAULT_GENERATION_CONFIG

MAGIC = b"PAW\x02"
VERSION = 2


def _is_paw_file(path: str | Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == MAGIC
    except OSError:
        return False


def save(filepath: str | Path, tensors: dict[str, Any], metadata: dict[str, Any]) -> None:
    """Write a .paw v2 file: header + metadata JSON + safetensors blob."""
    metadata_bytes = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        save_file(tensors, tmp_path)
        with open(tmp_path, "rb") as f:
            tensors_blob = f.read()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    with open(filepath, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(metadata_bytes)))
        f.write(metadata_bytes)
        f.write(tensors_blob)


def load(filepath: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a .paw v2 file. Returns (tensors_dict, metadata)."""
    with open(filepath, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"Not a .paw file: bad magic {magic!r}")
        (version,) = struct.unpack("<I", f.read(4))
        if version != VERSION:
            raise ValueError(f"Unsupported .paw version: {version}")
        (metadata_len,) = struct.unpack("<I", f.read(4))
        metadata = json.loads(f.read(metadata_len).decode("utf-8"))
        tensors_blob = f.read()

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(tensors_blob)
        tmp_path = tmp.name
    try:
        tensors = load_file(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return tensors, metadata


def save_program(
    filepath: str | Path,
    *,
    spec: str = "",
    base_model: str = "",
    lora_weights: dict[str, Any] | None = None,
    lora_config: dict[str, Any] | None = None,
    pseudo_program: str = "",
    generation_config: dict[str, Any] | None = None,
    description: str = "",
    author: str = "",
    tags: list[str] | None = None,
    examples: list[dict[str, str]] | None = None,
    source: str = "peft",
    source_info: dict[str, Any] | None = None,
) -> None:
    """Convenience helper that builds the standard metadata + tensors layout."""
    metadata: dict[str, Any] = {
        "format_version": VERSION,
        "kind": "neural_program",
        "interpreter_model": base_model,
        "base_model": base_model,
        "spec": spec,
        "pseudo_program": pseudo_program,
        "prefix_type": "kv_cache",
        "prefix_steps": 0,
        "num_layers": 0,
        "has_lora": bool(lora_weights),
        "description": description,
        "author": author,
        "tags": tags or [],
        "examples": examples or [],
        "source": source,
        "source_info": source_info or {},
        "generation_config": generation_config or DEFAULT_GENERATION_CONFIG,
    }
    if lora_config:
        metadata["lora_config"] = lora_config

    tensors: dict[str, Any] = {}
    if lora_weights:
        for name, t in lora_weights.items():
            tensors[f"lora_{name}"] = t

    save(filepath, tensors, metadata)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]


def validate(filepath: str | Path, *, max_size_mb: int = 500, max_lora_rank: int = 128) -> ValidationResult:
    """Sanity-check a .paw file. Returns errors found, never raises (except IO)."""
    errors: list[str] = []
    p = Path(filepath)
    if not p.exists():
        return ValidationResult(False, ["file not found"])
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        errors.append(f"file too large: {size_mb:.1f} MB (max {max_size_mb})")

    try:
        tensors, metadata = load(p)
    except Exception as e:
        return ValidationResult(False, [f"load failed: {e}"])

    if metadata.get("format_version") != VERSION:
        errors.append(f"unsupported format_version: {metadata.get('format_version')}")
    if not (metadata.get("interpreter_model") or metadata.get("base_model")):
        errors.append("missing interpreter_model / base_model")
    if metadata.get("has_lora"):
        rank = metadata.get("lora_config", {}).get("rank", 0)
        if rank > max_lora_rank:
            errors.append(f"lora rank too large: {rank} (max {max_lora_rank})")
        if not any(k.startswith("lora_") for k in tensors):
            errors.append("has_lora=true but no lora_ tensors present")

    return ValidationResult(not errors, errors)
