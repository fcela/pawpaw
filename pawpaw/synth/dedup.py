"""Near-duplicate removal using MinHash + Jaccard threshold over input n-grams.

Performance notes:
- Pair uses __slots__ to reduce memory overhead
- Tokenization uses compiled regex for speed
- MinHash updates use joined bytes directly to avoid repeated allocations
"""
from __future__ import annotations

import re
from typing import Iterable, List

from datasketch import MinHash, MinHashLSH

from pawpaw.synth.examples import Pair

_TOKEN_RE = re.compile(r"\w+")


def _minhash(text: str, *, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    # Update with joined n-grams as bytes to reduce allocations
    tokenizer = _TOKEN_RE.findall
    tokens = tokenizer(text.lower())
    tokens_len = len(tokens)
    n = 2  # Default n-gram size
    if tokens_len < n:
        if tokens:
            m.update(" ".join(tokens).encode("utf-8"))
    else:
        # Process n-grams efficiently
        for i in range(tokens_len - n + 1):
            m.update(" ".join(tokens[i : i + n]).encode("utf-8"))
    return m


def dedup(pairs: Iterable[Pair], *, threshold: float = 0.85) -> List[Pair]:
    """Drop pairs whose input is >threshold Jaccard-similar to any earlier pair's input."""
    pairs = list(pairs)
    if not pairs:
        return []

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    out: List[Pair] = []
    for i, pair in enumerate(pairs):
        m = _minhash(pair.input)
        if lsh.query(m):
            continue
        lsh.insert(str(i), m)
        out.append(pair)
    return out
