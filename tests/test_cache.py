from __future__ import annotations

import json

from pawpaw.cache import (
    CacheLayout,
    get_dataset,
    put_dataset,
    spec_hash,
)
from pawpaw.config import CompileOptions, SynthConfig


def test_spec_hash_is_stable():
    h1 = spec_hash("classify text", CompileOptions())
    h2 = spec_hash("classify text", CompileOptions())
    assert h1 == h2
    assert len(h1) == 64


def test_spec_hash_changes_with_spec():
    a = spec_hash("classify text", CompileOptions())
    b = spec_hash("classify email", CompileOptions())
    assert a != b


def test_spec_hash_changes_with_synth_config():
    base = CompileOptions()
    bumped = CompileOptions(synth=SynthConfig(n_per_category=99))
    assert spec_hash("x", base) != spec_hash("x", bumped)


def test_dataset_roundtrip(tmp_path):
    layout = CacheLayout(root=tmp_path)
    h = "abc123"
    assert get_dataset(layout, h) is None
    put_dataset(layout, h, [{"input": "i", "output": "o"}])
    out = get_dataset(layout, h)
    assert out == [{"input": "i", "output": "o"}]


def test_dataset_jsonl_format(tmp_path):
    layout = CacheLayout(root=tmp_path)
    put_dataset(layout, "h", [{"input": "i1", "output": "o1"}, {"input": "i2", "output": "o2"}])
    text = (tmp_path / "h" / "dataset.jsonl").read_text()
    lines = [line for line in text.splitlines() if line]
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"input": "i1", "output": "o1"}
