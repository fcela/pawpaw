from __future__ import annotations

import pytest

from pawpaw.synth.llm import LLM, parse_json_strict, complete_json_with_retry


class StubLLM(LLM):
    def __init__(self, *responses: str):
        self._responses = list(responses)
        self.calls: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0)


def test_parse_json_strict_ok():
    assert parse_json_strict('{"a": 1}') == {"a": 1}


def test_parse_json_strict_extracts_fenced_block():
    text = "Some preamble.\n```json\n{\"a\": 2}\n```\n"
    assert parse_json_strict(text) == {"a": 2}


def test_parse_json_strict_raises_on_garbage():
    with pytest.raises(ValueError):
        parse_json_strict("not json at all")


def test_complete_json_with_retry_succeeds_first_try():
    llm = StubLLM('{"x": 1}')
    out = complete_json_with_retry(llm, "do thing")
    assert out == {"x": 1}
    assert len(llm.calls) == 1


def test_complete_json_with_retry_succeeds_second_try():
    llm = StubLLM("garbage", '{"x": 2}')
    out = complete_json_with_retry(llm, "do thing")
    assert out == {"x": 2}
    assert len(llm.calls) == 2
    assert "valid JSON" in llm.calls[1]


def test_complete_json_with_retry_raises_after_two_failures():
    llm = StubLLM("garbage", "still garbage")
    with pytest.raises(ValueError):
        complete_json_with_retry(llm, "do thing")
