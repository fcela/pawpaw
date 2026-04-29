from __future__ import annotations

import json

import pytest

from pawpaw.synth.taxonomy import Category, enumerate_categories
from pawpaw.synth.llm import LLM


class StubLLM(LLM):
    def __init__(self, *responses: str):
        self._responses = list(responses)

    def complete(self, prompt, *, max_tokens=1024, temperature=0.0):
        return self._responses.pop(0)


def test_enumerate_categories_parses_response():
    payload = json.dumps({
        "categories": [
            {"name": "explicit_positive", "description": "Strong positive", "weight": 1.0},
            {"name": "adversarial", "description": "Confusing inputs", "weight": 2.0},
        ]
    })
    cats = enumerate_categories("classify sentiment", StubLLM(payload), n_categories=2)
    assert len(cats) == 2
    assert cats[0] == Category(name="explicit_positive", description="Strong positive", weight=1.0)


def test_enumerate_categories_rejects_empty():
    payload = json.dumps({"categories": []})
    with pytest.raises(ValueError, match="no categories"):
        enumerate_categories("classify", StubLLM(payload), n_categories=5)


def test_enumerate_categories_normalizes_missing_weight():
    payload = json.dumps({"categories": [{"name": "x", "description": "y"}]})
    cats = enumerate_categories("z", StubLLM(payload), n_categories=1)
    assert cats[0].weight == 1.0
