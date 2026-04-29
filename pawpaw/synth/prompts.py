"""Versioned synthesis prompt templates. Bump SynthConfig version fields when changing wording."""
from __future__ import annotations


def build_taxonomy_prompt(spec: str, *, n_categories: int = 10) -> str:
    return f"""You are designing the input distribution for a small classifier.

Specification:
\"\"\"{spec}\"\"\"

Enumerate {n_categories} distinct categories of inputs that this classifier must
handle well. Cover the obvious cases AND failure modes: edge cases, adversarial
inputs, multilingual variants, unusual formats, ambiguous inputs.

Output ONLY valid JSON in this shape:
{{
  "categories": [
    {{
      "name": "snake_case_id",
      "description": "one sentence description",
      "weight": 1.0
    }}
  ]
}}

Weights should sum to roughly {n_categories}.0 — give edge cases and failure modes
proportionally more weight if they're underrepresented in naive data.
"""


def build_examples_prompt(
    *,
    spec: str,
    category_name: str,
    category_description: str,
    n_examples: int = 30,
) -> str:
    return f"""You are generating training examples for a classifier.

Specification:
\"\"\"{spec}\"\"\"

Category: {category_name}
Description: {category_description}

Generate exactly {n_examples} (input, output) pairs for this category. Vary along
ALL of these axes within the {n_examples} examples:
- length: short (<10 words), medium (10-30 words), long (>30 words)
- register: formal, casual, technical, terse
- format: plain prose, list, JSON, code-like, transcript

Each output must obey the specification exactly. Do not add explanations.

Output ONLY valid JSON in this shape:
{{
  "pairs": [
    {{"input": "...", "output": "..."}},
    ...
  ]
}}
"""
