from __future__ import annotations

import json

import pytest

from pawpaw.synth.examples import Pair, generate_for_category, generate_all
from pawpaw.synth.taxonomy import Category
from pawpaw.synth.llm import LLM


class StubLLM(LLM):
    def __init__(self, *responses: str):
        self._responses = list(responses)

    def complete(self, prompt, *, max_tokens=1024, temperature=0.0):
        return self._responses.pop(0)


def _payload(n: int) -> str:
    return json.dumps({"pairs": [{"input": f"in{i}", "output": f"out{i}"} for i in range(n)]})


def test_generate_for_category_returns_pairs_with_metadata():
    cat = Category("mood", "Mood inputs", 1.0)
    pairs = generate_for_category("classify", cat, StubLLM(_payload(3)), n_examples=3)
    assert len(pairs) == 3
    assert pairs[0] == Pair(input="in0", output="out0", category="mood", length_bucket="short")


def test_generate_for_category_assigns_length_buckets():
    short = "x"
    medium = " ".join(["w"] * 15)
    long_ = " ".join(["w"] * 60)
    payload = json.dumps({"pairs": [
        {"input": short, "output": "a"},
        {"input": medium, "output": "b"},
        {"input": long_, "output": "c"},
    ]})
    cat = Category("c", "d", 1.0)
    pairs = generate_for_category("s", cat, StubLLM(payload), n_examples=3)
    buckets = [p.length_bucket for p in pairs]
    assert buckets == ["short", "medium", "long"]


def test_generate_for_category_accepts_common_alternate_keys():
    payload = json.dumps({"pairs": [{"request": "Can you fix this?", "label": "code"}]})
    cat = Category("c", "d", 1.0)
    with pytest.raises(KeyError):
        generate_for_category("s", cat, StubLLM(payload), n_examples=1)


def test_generate_all_iterates_categories_and_scales_by_weight():
    cats = [Category("a", "x", 1.0), Category("b", "y", 2.0)]
    llm = StubLLM(_payload(10), _payload(20))
    pairs = generate_all("spec", cats, llm, n_per_category=10, batch_size=1)
    assert len(pairs) == 30
    assert {p.category for p in pairs} == {"a", "b"}


def test_generate_all_skips_bad_categories():
    cats = [Category("bad", "x", 1.0), Category("good", "y", 1.0)]
    pairs = generate_all("spec", cats, StubLLM(_payload(0), _payload(3)), n_per_category=3, batch_size=1)
    assert len(pairs) == 3
    assert {p.category for p in pairs} == {"good"}


def test_generate_for_category_rejects_empty():
    cat = Category("c", "d", 1.0)
    with pytest.raises(ValueError, match="no training pairs"):
        generate_for_category("s", cat, StubLLM(_payload(0)), n_examples=5)
