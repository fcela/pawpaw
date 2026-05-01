"""Cache layout under ~/.cache/pawpaw/<spec_hash>/, with dataset reuse.

Performance notes:
- get_dataset uses chunked reading for large files
- put_dataset uses buffered writing and atomic rename
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


from pawpaw.config import CompileOptions
from pawpaw.version import PIPELINE_VERSION
from pawpaw.runtime_cache import _cache_root


def _default_root() -> Path:
    return _cache_root()


@dataclass(frozen=True)
class CacheLayout:
    root: Path

    def dir_for(self, spec_hash_value: str) -> Path:
        return self.root / spec_hash_value

    def dataset_path(self, spec_hash_value: str) -> Path:
        return self.dir_for(spec_hash_value) / "dataset.jsonl"

    def peft_dir(self, spec_hash_value: str) -> Path:
        return self.dir_for(spec_hash_value) / "peft_adapter"

    def synth_failure_path(self, spec_hash_value: str) -> Path:
        return self.dir_for(spec_hash_value) / "synth_failure.txt"


def default_layout() -> CacheLayout:
    return CacheLayout(root=_default_root())


def spec_hash(spec: str, options: CompileOptions) -> str:
    payload = json.dumps(
        {
            "spec": spec,
            "pipeline_version": PIPELINE_VERSION,
            "base_model": options.base_model,
            "synth_fingerprint": options.synth.fingerprint(),
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_dataset(layout: CacheLayout, spec_hash_value: str) -> list[dict] | None:
    """Load dataset from cache, using line-by-line parsing for memory efficiency."""
    path = layout.dataset_path(spec_hash_value)
    if not path.exists():
        return None

    out: list[dict] = []
    # Read line by line to handle large files without loading entirely into memory
    try:
        with open(path, "r", encoding="utf-8", buffering=8192) as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
    except (json.JSONDecodeError, OSError):
        # If file is corrupted, return None to trigger regeneration
        return None
    return out


def put_dataset(layout: CacheLayout, spec_hash_value: str, records: Iterable[dict]) -> None:
    """Write dataset to cache with atomic rename for safety."""
    d = layout.dir_for(spec_hash_value)
    d.mkdir(parents=True, exist_ok=True)

    dest = layout.dataset_path(spec_hash_value)
    tmp = d / f"dataset.jsonl.tmp.{os.getpid()}"

    try:
        # Use larger buffer for better I/O performance
        with open(tmp, "w", encoding="utf-8", buffering=16384) as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")
                f.flush()
        # Atomic rename ensures readers never see partial files
        tmp.replace(dest)
    except Exception:
        # Clean up temp file on error
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
