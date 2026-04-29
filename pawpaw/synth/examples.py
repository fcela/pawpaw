"""Stage B: per-category example generation."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from pawpaw.synth.llm import LLM, complete_json_with_retry
from pawpaw.synth.prompts import build_batch_examples_prompt, build_examples_prompt
from pawpaw.synth.taxonomy import Category

logger = logging.getLogger(__name__)


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


def _generate_batch(
    spec: str,
    categories: list[Category],
    llm: LLM,
    *,
    n_per_category: int,
) -> list[Pair]:
    """Generate examples for multiple categories in a single LLM call."""
    cat_dicts = [
        {"name": c.name, "description": c.description, "n": max(1, int(round(n_per_category * c.weight)))}
        for c in categories
    ]
    prompt = build_batch_examples_prompt(spec=spec, categories=cat_dicts, n_per_category=n_per_category)
    payload = complete_json_with_retry(llm, prompt, max_tokens=8192)
    raw = payload.get("pairs", [])
    if not raw:
        raise ValueError(
            "LLM returned no training pairs in batch. "
            "Try a larger or different model, or re-run."
        )
    out: list[Pair] = []
    for item in raw:
        cat = str(item.get("category", categories[0].name))
        out.append(Pair(
            input=str(item["input"]),
            output=str(item["output"]),
            category=cat,
            length_bucket=_length_bucket(str(item["input"])),
        ))
    return out


def generate_all(
    spec: str,
    categories: list[Category],
    llm: LLM,
    *,
    n_per_category: int,
    batch_size: int = 4,
) -> list[Pair]:
    """Generate examples for every category, scaling target count by category weight.

    Uses batched generation: combines up to `batch_size` categories into a
    single LLM call to reduce total round-trips. Falls back to per-category
    generation if a batch returns no results.
    """
    out: list[Pair] = []
    i = 0
    while i < len(categories):
        batch = categories[i:i + batch_size]
        if len(batch) == 1:
            cat = batch[0]
            target = max(1, int(round(n_per_category * cat.weight)))
            try:
                out.extend(generate_for_category(spec, cat, llm, n_examples=target))
            except ValueError:
                logger.warning("skipping category %r: LLM returned no pairs", cat.name)
        else:
            try:
                batch_pairs = _generate_batch(spec, batch, llm, n_per_category=n_per_category)
                if batch_pairs:
                    out.extend(batch_pairs)
                else:
                    for cat in batch:
                        target = max(1, int(round(n_per_category * cat.weight)))
                        out.extend(generate_for_category(spec, cat, llm, n_examples=target))
            except ValueError:
                for cat in batch:
                    target = max(1, int(round(n_per_category * cat.weight)))
                    out.extend(generate_for_category(spec, cat, llm, n_examples=target))
        i += batch_size
    return out
