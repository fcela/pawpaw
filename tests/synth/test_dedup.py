from pawpaw.synth.dedup import dedup
from pawpaw.synth.examples import Pair


def _p(text: str, bucket: str = "short", category: str = "a") -> Pair:
    return Pair(input=text, output="o", category=category, length_bucket=bucket)


def test_dedup_removes_near_duplicates():
    pairs = [
        _p("the quick brown fox jumps over the lazy dog"),
        _p("the quick brown fox jumps over the lazy dog!"),
        _p("a completely different sentence about turtles"),
    ]
    out = dedup(pairs, threshold=0.85)
    assert len(out) == 2
    assert out[0].input.startswith("the quick")
    assert out[1].input.startswith("a completely")


def test_dedup_keeps_distinct():
    pairs = [_p(f"unique sentence number {i}") for i in range(5)]
    out = dedup(pairs, threshold=0.85)
    assert len(out) == 5


def test_dedup_handles_empty():
    assert dedup([], threshold=0.85) == []


def test_dedup_preserves_first_occurrence_order():
    pairs = [_p("alpha", "short"), _p("beta beta beta beta", "medium"), _p("alpha!", "short")]
    out = dedup(pairs, threshold=0.85)
    assert [p.input for p in out] == ["alpha", "beta beta beta beta"]
