"""Stage A: spec → list of input categories."""
from __future__ import annotations

from dataclasses import dataclass

from pawpaw.synth.llm import LLM, complete_json_with_retry
from pawpaw.synth.prompts import build_taxonomy_prompt


@dataclass(frozen=True)
class Category:
    name: str
    description: str
    weight: float = 1.0


def enumerate_categories(spec: str, llm: LLM, *, n_categories: int = 10) -> list[Category]:
    prompt = build_taxonomy_prompt(spec, n_categories=n_categories)
    payload = complete_json_with_retry(llm, prompt, max_tokens=2048)
    raw = payload.get("categories", [])
    if not raw:
        raise ValueError(
            "LLM returned no categories. This usually means the synthesis LLM "
            "produced invalid output. Try a larger or different model, or re-run."
        )
    out: list[Category] = []
    for item in raw:
        out.append(Category(
            name=str(item["name"]),
            description=str(item["description"]),
            weight=float(item.get("weight", 1.0)),
        ))
    return out
