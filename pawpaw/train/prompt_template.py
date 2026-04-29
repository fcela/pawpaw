"""Builds the prompt_template.txt expected by programasweights.runtime_llamacpp.

The template uses Qwen3 chat-template tokens and ends with the assistant turn opener
so the model continues from the right place.
"""
from __future__ import annotations

from typing import Sequence

from pawpaw.synth.examples import Pair

INPUT_PLACEHOLDER = "{INPUT_PLACEHOLDER}"


def build_prompt_template(spec: str, *, demos: Sequence[Pair]) -> str:
    """Build a Qwen3 chat template with reasoning disabled.

    Two mechanisms force a no-reasoning, single-turn answer:
    1. `/no_think` in the system message — Qwen3's native control token to skip reasoning.
    2. An empty `<think></think>` block already opened in the assistant turn — the model
       sees reasoning as already-completed and generates the bare answer directly.

    Both training targets and inference run against this template, so the LoRA learns
    to emit just the answer (no `<think>...` block of its own).
    """
    demo_block = "\n\n".join(
        f"Input: {d.input}\nOutput: {d.output}" for d in demos
    )
    return (
        f"<|im_start|>system\n"
        f"{spec}\n\n"
        f"/no_think\n\n"
        f"Examples:\n"
        f"{demo_block}\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{INPUT_PLACEHOLDER}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n\n</think>\n\n"
    )


def split_template(template: str) -> tuple[str, str]:
    if INPUT_PLACEHOLDER not in template:
        raise ValueError(f"Template missing {INPUT_PLACEHOLDER}")
    prefix, suffix = template.split(INPUT_PLACEHOLDER, 1)
    return prefix, suffix


def render_for_training(template: str, pair: Pair) -> str:
    """Substitute {INPUT_PLACEHOLDER} with the pair's input and append the expected output."""
    return template.replace(INPUT_PLACEHOLDER, pair.input) + pair.output
