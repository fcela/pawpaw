from __future__ import annotations

from pawpaw.synth.examples import Pair
from pawpaw.train.dataset import build_train_records, train_val_split


class StubTokenizer:
    """Whitespace tokenizer with a fixed vocab."""

    def __init__(self):
        self._vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1}

    def _id(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = len(self._vocab)
        return self._vocab[tok]

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        ids = [self._id(t) for t in text.split()]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    @property
    def eos_token_id(self) -> int:
        return 1


def _p(i, o):
    return Pair(input=i, output=o, category="c", length_bucket="short")


def test_build_train_records_masks_prompt_tokens():
    template = "PREFIX {INPUT_PLACEHOLDER} SUFFIX "
    pair = _p("hello world", "good")
    tok = StubTokenizer()

    records = build_train_records(template, [pair], tokenizer=tok, max_length=64)
    assert len(records) == 1
    rec = records[0]
    assert "input_ids" in rec and "labels" in rec and "attention_mask" in rec
    assert len(rec["input_ids"]) == len(rec["labels"]) == len(rec["attention_mask"])
    n_prompt = len(tok("PREFIX hello world SUFFIX")["input_ids"])
    assert all(label == -100 for label in rec["labels"][:n_prompt])
    assert rec["labels"][n_prompt:n_prompt + 1] == [tok._vocab["good"]]


def test_train_val_split_is_deterministic():
    pairs = [_p(f"i{i}", f"o{i}") for i in range(20)]
    a = train_val_split(pairs, val_fraction=0.1, seed=42)
    b = train_val_split(pairs, val_fraction=0.1, seed=42)
    assert a == b
    assert len(a[1]) == 2
    assert len(a[0]) == 18


def test_train_val_split_minimum_one_val():
    pairs = [_p(f"i{i}", f"o{i}") for i in range(5)]
    train, val = train_val_split(pairs, val_fraction=0.1, seed=0)
    assert len(val) >= 1


def test_build_train_records_rejects_oversize(caplog):
    template = "PREFIX {INPUT_PLACEHOLDER} SUFFIX "
    long_input = " ".join(["w"] * 100)
    pair = _p(long_input, "out")
    tok = StubTokenizer()
    records = build_train_records(template, [pair], tokenizer=tok, max_length=20)
    assert records == []
