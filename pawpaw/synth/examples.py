"""Stage B: per-category example generation."""
from __future__ import annotations

from dataclasses import dataclass

from pawpaw.synth.llm import LLM, complete_json_with_retry
from pawpaw.synth.prompts import build_examples_prompt
from pawpaw.synth.taxonomy import Category


@dataclass(frozen=True, slots=True)
class Pair:
    input: str
    output: str
    category: str
    length_bucket: str


def _length_bucket(text: str) -> str:
    n = len(text.split())
    if n < 10:
        return "short"
    if n <= 30:
        return "medium"
    return "long"


def generate_for_category(
    spec: str,
    category: Category,
    llm: LLM,
    *,
    n_examples: int,
) -> list[Pair]:
    prompt = build_examples_prompt(
        spec=spec,
        category_name=category.name,
        category_description=category.description,
        n_examples=n_examples,
    )
    payload = complete_json_with_retry(llm, prompt, max_tokens=4096)
    raw = payload.get("pairs", [])
    if not raw:
        raise ValueError(
            f"LLM returned no training pairs for category '{category.name}'. "
            f"This usually means the synthesis LLM produced invalid output. "
            f"Try a larger or different model, or re-run."
        )
    return [
        Pair(
            input=str(item["input"]),
            output=str(item["output"]),
            category=category.name,
            length_bucket=_length_bucket(str(item["input"])),
        )
        for item in raw
    ]


def generate_all(
    spec: str,
    categories: list[Category],
    llm: LLM,
    *,
    n_per_category: int,
) -> list[Pair]:
    """Generate examples for every category, scaling target count by category weight."""
    out: list[Pair] = []
    for cat in categories:
        target = max(1, int(round(n_per_category * cat.weight)))
        out.extend(generate_for_category(spec, cat, llm, n_examples=target))
    return out
