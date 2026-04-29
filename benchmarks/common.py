"""Shared utilities for benchmarks."""
from __future__ import annotations

import platform
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence


@dataclass
class TimedRun:
    label: str
    n: int
    elapsed_s: float

    @property
    def per_call_ms(self) -> float:
        return 1000.0 * self.elapsed_s / self.n if self.n else 0.0


def time_calls(label: str, fn: Callable[[str], str], inputs: Sequence[str], *, warmup: int = 1) -> TimedRun:
    """Run fn on each input; first `warmup` calls are not counted."""
    for w in inputs[:warmup]:
        fn(w)
    start = time.perf_counter()
    for x in inputs[warmup:]:
        fn(x)
    elapsed = time.perf_counter() - start
    return TimedRun(label=label, n=max(0, len(inputs) - warmup), elapsed_s=elapsed)


def host_info() -> str:
    parts = [platform.system(), platform.machine(), f"py{platform.python_version()}"]
    try:
        import torch
        parts.append(f"torch{torch.__version__}")
        if torch.cuda.is_available():
            parts.append(f"cuda:{torch.cuda.get_device_name(0)}")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            parts.append("metal")
        else:
            parts.append("cpu")
    except Exception:
        pass
    return " | ".join(parts)


def normalize_label(text: str) -> str:
    """Strip whitespace, punctuation, lowercase. Useful for exact-match comparison."""
    return "".join(c for c in text.strip().lower() if c.isalnum())


def accuracy(predictions: Iterable[str], labels: Iterable[str]) -> float:
    preds = [normalize_label(p) for p in predictions]
    refs = [normalize_label(l) for l in labels]
    if not preds:
        return 0.0
    correct = sum(1 for p, r in zip(preds, refs) if p == r)
    return correct / len(preds)
