"""LLM protocol + llama.cpp implementation + JSON helpers used by synthesis stages.

Tests pass a stub implementing the LLM protocol — no model file required.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class LLM(Protocol):
    def complete(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        ...


_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


def _parse_json_candidate(text: str) -> Any:
    candidate = text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            value, _end = decoder.raw_decode(text, i)
            return value
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("no JSON object or array found", text, 0)


def parse_json_strict(text: str) -> Any:
    """Parse JSON, tolerating fenced blocks and small-model leading/trailing chatter."""
    candidates = [text]
    match = _FENCE_RE.search(text)
    if match:
        candidates.insert(0, match.group(1))
    last_err: Exception | None = None
    for candidate in candidates:
        try:
            return _parse_json_candidate(candidate)
        except json.JSONDecodeError as e:
            last_err = e
    raise ValueError(
        f"Could not parse JSON from model output after retry: {last_err}\n"
        f"This usually means the synthesis LLM is too small or confused. "
        f"Try a larger model (4B+ recommended) or re-run.\n"
        f"---\n{text[:500]}"
    )


def complete_json_with_retry(
    llm: LLM,
    prompt: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> Any:
    """One bounded retry: if first response is not valid JSON, reprompt asking for valid JSON.

    Raises ValueError on the second failure.
    """
    first = llm.complete(prompt, max_tokens=max_tokens, temperature=temperature)
    try:
        return parse_json_strict(first)
    except ValueError:
        logger.debug("first LLM response was not valid JSON, retrying")

    repair_prompt = (
        f"{prompt}\n\n"
        f"Your previous response was not valid JSON:\n{first[:1500]}\n\n"
        f"Output ONLY valid JSON. No prose, no markdown fences, no commentary."
    )
    second = llm.complete(repair_prompt, max_tokens=max_tokens, temperature=temperature)
    return parse_json_strict(second)


class LlamaCppLLM:
    """Wraps llama-cpp-python. Lazy-loads the model on first call."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 4096,
        seed: int = 42,
        n_threads: int | None = None,
        n_batch: int = 512,
        n_gpu_layers: int | None = None,
        flash_attn: bool = True,
    ):
        self._model_path = str(model_path)
        self._n_ctx = n_ctx
        self._seed = seed
        self._n_threads = n_threads
        self._n_batch = n_batch
        self._n_gpu_layers = n_gpu_layers
        self._flash_attn = flash_attn
        self._llama: Any | None = None

    def _ensure_loaded(self) -> None:
        if self._llama is not None:
            return
        from llama_cpp import Llama
        from pawpaw.config import auto_n_threads

        n_threads = self._n_threads if self._n_threads is not None else auto_n_threads()
        n_gpu = self._n_gpu_layers if self._n_gpu_layers is not None else 0

        self._llama = Llama(
            model_path=self._model_path,
            n_ctx=self._n_ctx,
            seed=self._seed,
            n_threads=n_threads,
            n_batch=self._n_batch,
            n_gpu_layers=n_gpu,
            flash_attn=self._flash_attn,
            verbose=False,
        )

    def complete(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        self._ensure_loaded()
        assert self._llama is not None
        out = self._llama.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            top_k=1 if temperature == 0.0 else 50,
            stop=[],
        )
        return out["choices"][0]["text"]
