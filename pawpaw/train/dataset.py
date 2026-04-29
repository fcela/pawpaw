"""Build tokenized training records with a label mask that ignores prompt tokens.

Performance notes:
- Uses list comprehension for faster list creation
- Pre-computes prompt lengths to avoid repeated len() calls
- Avoids building temporary lists when possible
"""
from __future__ import annotations

import random
from typing import Any, Sequence

from pawpaw.synth.examples import Pair
from pawpaw.train.prompt_template import INPUT_PLACEHOLDER, render_for_training

LABEL_IGNORE_INDEX = -100


def _make_labels(full_ids: list[int], prompt_len: int) -> list[int]:
    """Create labels with prompt tokens masked."""
    return [LABEL_IGNORE_INDEX] * prompt_len + full_ids[prompt_len:]


def build_train_records(
    template: str,
    pairs: Sequence[Pair],
    *,
    tokenizer: Any,
    max_length: int,
) -> list[dict]:
    """Tokenize each (template-rendered prompt + output) pair into a Trainer-ready dict.

    Labels for prompt tokens are set to LABEL_IGNORE_INDEX so the loss is only computed
    on output tokens. Records exceeding max_length after tokenization are dropped.
    """
    if INPUT_PLACEHOLDER not in template:
        raise ValueError(f"Template missing {INPUT_PLACEHOLDER}")

    eos_id = getattr(tokenizer, "eos_token_id", None)
    out: list[dict] = []

    for pair in pairs:
        prompt_text = template.replace(INPUT_PLACEHOLDER, pair.input)
        full_text = render_for_training(template, pair)

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        if eos_id is not None:
            full_ids = full_ids + [eos_id]

        if len(full_ids) > max_length:
            continue

        prompt_len = len(prompt_ids)
        labels = _make_labels(full_ids, prompt_len)

        attention_mask = [1] * len(full_ids)
        out.append({
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        })

    return out


def train_val_split(
    pairs: Sequence[Pair],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[Pair], list[Pair]]:
    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(pairs) * val_fraction)))
    val_idx = set(indices[:n_val])
    train = [pairs[i] for i in range(len(pairs)) if i not in val_idx]
    val = [pairs[i] for i in range(len(pairs)) if i in val_idx]
    return train, val
