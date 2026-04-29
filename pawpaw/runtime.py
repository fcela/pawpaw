"""Local execution runtime for compiled .paw programs.

This module provides two primary entry points:

pawpaw.load("triage.paw")       # load a saved program (fast)
program("some user input")      # apply it (very fast)

Many programs can be loaded into the same process. They share one Llama
context behind the scenes (keyed by base-model + n_ctx + GPU offload), so
loading N programs costs roughly the same memory as loading one. Switching
from one program to another swaps the LoRA adapter and restores the
prefix KV cache for that program.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import llama_cpp
from llama_cpp import Llama

from pawpaw.runtime_cache import _cache_root, ensure_base_model_gguf

logger = logging.getLogger(__name__)

PLACEHOLDER = "{INPUT_PLACEHOLDER}"
DEFAULT_N_CTX = 4096


# -----------------------------------------------------------------------------
# Shared Llama session keyed by (base_model_path, n_ctx, n_gpu_layers).
# Many programs that share a base model reuse the same Llama instance.
# -----------------------------------------------------------------------------

_session_lock = threading.RLock()
_sessions: weakref.WeakValueDictionary[tuple[str, int, int], _Session] = weakref.WeakValueDictionary()


def _auto_n_gpu_layers() -> int:
    try:
        return -1 if llama_cpp.llama_supports_gpu_offload() else 0
    except Exception:
        return 0


def _silence_stderr_during(fn, verbose: bool) -> Any:
    if verbose:
        return fn()
    fd = sys.stderr.fileno()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(fd)
    try:
        os.dup2(devnull, fd)
        return fn()
    finally:
        os.dup2(saved, fd)
        os.close(devnull)
        os.close(saved)


@dataclass
class _Session:
    """Holds a Llama instance shared across multiple programs.

    `active_program_id` lets callers skip redundant adapter/KV switches when
    the same program is invoked back-to-back.
    """
    llama: Any
    n_ctx: int
    n_gpu_layers: int
    base_model_path: str
    active_program_id: int | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)


def _get_or_create_session(
    base_model_path: Path,
    *,
    n_ctx: int,
    n_gpu_layers: int,
    verbose: bool,
) -> _Session:
    key = (str(base_model_path), n_ctx, n_gpu_layers)
    with _session_lock:
        sess = _sessions.get(key)
        if sess is not None:
            return sess

        def _load():
            return Llama(
                model_path=str(base_model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose,
            )

        llama = _silence_stderr_during(_load, verbose)
        sess = _Session(llama=llama, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, base_model_path=str(base_model_path))
        _sessions[key] = sess
        return sess


def clear_session_cache() -> None:
    """Release shared Llama instances. Mostly for tests."""
    with _session_lock:
        _sessions.clear()


# ---------------------------------------------------------------------------
# .paw extraction: unpack a .paw file into a bundle directory for loading.
# ---------------------------------------------------------------------------


def _paw_cache_dir() -> Path:
    d = _cache_root() / "paw_bundles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _unpack_paw(paw_path: Path) -> Path:
    """Extract a .paw file into a bundle directory under the cache.

    The bundle directory is keyed by the file's SHA-256, so the same .paw
    is only unpacked once. Returns the bundle directory path.
    """
    file_hash = hashlib.sha256(paw_path.read_bytes()).hexdigest()[:16]
    bundle_dir = _paw_cache_dir() / file_hash
    if bundle_dir.exists() and (bundle_dir / "meta.json").exists():
        logger.debug("reusing cached bundle for %s at %s", paw_path.name, bundle_dir)
        return bundle_dir

    from pawpaw.format import load as paw_load

    tensors, metadata = paw_load(paw_path)

    tmp_dir = bundle_dir.with_suffix(".tmp")
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Write prompt template
        pseudo = metadata.get("pseudo_program", "")
        prompt_template = pseudo + PLACEHOLDER if pseudo else PLACEHOLDER
        (tmp_dir / "prompt_template.txt").write_text(prompt_template, encoding="utf-8")

        # Write metadata
        (tmp_dir / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # Convert safetensors tensors to GGUF adapter
        import numpy as np
        import torch
        from gguf import GGUFWriter, MODEL_ARCH, TensorNameMap
        from huggingface_hub import snapshot_download

        model_id = metadata.get("interpreter_model") or metadata.get("base_model", "")
        base_cfg_path = Path(snapshot_download(model_id, allow_patterns=["config.json"]))
        base_cfg = json.loads((base_cfg_path / "config.json").read_text())
        arch_enum = MODEL_ARCH[base_cfg["model_type"].upper()]
        arch_name = arch_enum.name.lower()
        name_map = TensorNameMap(arch_enum, n_blocks=base_cfg["num_hidden_layers"])

        lora_cfg = metadata.get("lora_config", {})
        writer = GGUFWriter(str(tmp_dir / "adapter.gguf"), arch=arch_name)
        writer.add_string("general.type", "adapter")
        writer.add_string("adapter.type", "lora")
        writer.add_float32("adapter.lora.alpha", float(lora_cfg.get("alpha", 16)))
        writer.add_string("general.name", "pawpaw_adapter")

        writer.add_uint32(f"{arch_name}.context_length", base_cfg.get("max_position_embeddings", 8192))
        writer.add_uint32(f"{arch_name}.embedding_length", base_cfg["hidden_size"])
        writer.add_uint32(f"{arch_name}.feed_forward_length", base_cfg["intermediate_size"])
        writer.add_uint32(f"{arch_name}.attention.head_count", base_cfg["num_attention_heads"])
        writer.add_uint32(f"{arch_name}.attention.head_count_kv",
                          base_cfg.get("num_key_value_heads", base_cfg["num_attention_heads"]))
        writer.add_uint32(f"{arch_name}.block_count", base_cfg["num_hidden_layers"])
        writer.add_float32(f"{arch_name}.attention.layer_norm_rms_epsilon",
                           base_cfg.get("rms_norm_eps", 1e-6))
        writer.add_float32(f"{arch_name}.rope.freq_base", base_cfg.get("rope_theta", 10000.0))
        head_dim = base_cfg.get("head_dim", base_cfg["hidden_size"] // base_cfg["num_attention_heads"])
        writer.add_uint32(f"{arch_name}.attention.key_length", head_dim)
        writer.add_uint32(f"{arch_name}.attention.value_length", head_dim)

        import re
        _LORA_RE = re.compile(r"^(?P<base>.+?)\.lora_(?P<ab>[AB])$")
        for name, tensor in tensors.items():
            if not name.startswith("lora_"):
                continue
            clean = name[5:]  # strip "lora_" prefix added by save_program
            m = _LORA_RE.match(clean)
            if not m:
                continue
            gguf_base = name_map.get_name(m["base"])
            if gguf_base is None:
                continue
            gguf_name = f"{gguf_base}.weight.lora_{m['ab'].lower()}"
            if isinstance(tensor, torch.Tensor):
                arr = tensor.detach().to(torch.float16).contiguous().cpu().numpy()
            elif isinstance(tensor, np.ndarray):
                arr = tensor.astype(np.float16)
            else:
                arr = np.array(tensor, dtype=np.float16)
            writer.add_tensor(gguf_name, arr)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        # Atomic rename
        tmp_dir.replace(bundle_dir)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    logger.info("unpacked %s → %s", paw_path.name, bundle_dir)
    return bundle_dir


def _resolve_bundle(path: Path) -> tuple[Path, dict, str]:
    """Resolve a path to a bundle directory, returning (bundle_dir, meta, prompt_template).

    Accepts:
    - A .paw file (extracted on the fly, cached)
    - A bundle directory containing meta.json + prompt_template.txt + adapter.gguf
    """
    p = Path(path)

    if p.is_file() or (p.exists() and p.suffix == ".paw"):
        bundle = _unpack_paw(p)
    elif p.is_dir() and (p / "meta.json").exists():
        bundle = p
    else:
        candidates = [p, p.with_suffix(""), p.with_suffix(".bundle"), p.parent / (p.stem + "_bundle")]
        bundle = next((c for c in candidates if c.is_dir() and (c / "meta.json").exists()), None)
        if bundle is None:
            raise FileNotFoundError(
                f"Could not find a pawpaw program at {path}. "
                f"Provide a .paw file or a directory containing meta.json."
            )

    meta = json.loads((bundle / "meta.json").read_text())
    template = (bundle / "prompt_template.txt").read_text(encoding="utf-8")
    return bundle, meta, template


# ---------------------------------------------------------------------------
# Program: a compiled .paw program that you can call.
# ---------------------------------------------------------------------------


class Program:
    """A compiled neural program loaded from a .paw file or bundle directory.

    Created by `pawpaw.load(path)`; call it like a function:

    triage = pawpaw.load("triage.paw")
    triage("How are you doing today?")  # → "trivial"

    Many Programs can coexist in one process; they share a Llama instance for
    their common base model.
    """

    _next_id: int = 0
    _id_lock = threading.Lock()

    def __init__(
        self,
        path: str | Path,
        *,
        n_ctx: int = DEFAULT_N_CTX,
        n_gpu_layers: int | str = "auto",
        verbose: bool = False,
        base_model_path: str | Path | None = None,
    ):
        self._bundle_dir, self._meta, self._template = _resolve_bundle(Path(path))
        self._verbose = verbose

        with Program._id_lock:
            Program._next_id += 1
            self._id = Program._next_id

        if n_gpu_layers == "auto":
            n_gpu_layers = _auto_n_gpu_layers()

        interpreter = self._meta.get("interpreter_model") or self._meta.get("interpreter") or "Qwen/Qwen3-0.6B"
        base_path = ensure_base_model_gguf(
            interpreter,
            override_path=str(base_model_path) if base_model_path else None,
        )

        self._session = _get_or_create_session(
            base_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=verbose,
        )

        adapter_path = self._bundle_dir / "adapter.gguf"
        if not adapter_path.exists():
            raise FileNotFoundError(f"missing adapter.gguf in {self._bundle_dir}")

        adapter = _silence_stderr_during(
            lambda: llama_cpp.llama_adapter_lora_init(
                self._session.llama.model, str(adapter_path).encode("utf-8"),
            ),
            verbose,
        )
        if adapter is None:
            raise RuntimeError(f"failed to load LoRA adapter at {adapter_path}")
        self._adapter = adapter

        if PLACEHOLDER in self._template:
            prefix_text, self._suffix_text = self._template.split(PLACEHOLDER, 1)
        else:
            prefix_text, self._suffix_text = self._template, ""

        self._prefix_tokens: list[int] = self._session.llama.tokenize(
            prefix_text.encode("utf-8"), add_bos=False, special=True,
        )
        self._n_prefix = len(self._prefix_tokens)
        self._prefix_kv_path = self._bundle_dir / "prefix_kv_cache.bin"

    # ----- lifecycle ----------------------------------------------------

    def __del__(self):
        self._adapter = None

    @property
    def spec(self) -> str:
        return self._meta.get("spec", "")

    @property
    def interpreter(self) -> str:
        return self._meta.get("interpreter_model") or self._meta.get("interpreter") or ""

    @property
    def bundle_dir(self) -> Path:
        return self._bundle_dir

    def __repr__(self) -> str:
        s = self.spec
        preview = s[:60] + "…" if len(s) > 60 else s
        return f"Program({preview!r})"

    # ----- inference ----------------------------------------------------

    def _activate(self) -> None:
        """Make this program the active one on the shared Llama: swap adapter, restore prefix KV.

        Skipped if this program is already active and the prefix is in place.
        Uses disk cache when available (fast), falls back to evaluating from scratch.
        """
        sess = self._session
        if sess.active_program_id == self._id:
            return

        # 1. Swap LoRA adapter.
        if hasattr(llama_cpp, "llama_set_adapters_lora"):
            adapters = (llama_cpp.llama_adapter_lora_p_ctypes * 1)(self._adapter)
            scales = (ctypes.c_float * 1)(1.0)
            llama_cpp.llama_set_adapters_lora(sess.llama.ctx, adapters, 1, scales)
        else:
            llama_cpp.llama_set_adapter_lora(sess.llama.ctx, self._adapter, 1.0)

        # 2. Restore the prefix KV cache from disk or evaluate from scratch
        token_array = (llama_cpp.llama_token * self._n_prefix)(*self._prefix_tokens)

        # Try disk cache first (fast path after first call)
        if self._prefix_kv_path.exists():
            try:
                n_token_count = ctypes.c_size_t(0)
                n_loaded = llama_cpp.llama_state_seq_load_file(
                    sess.llama.ctx,
                    str(self._prefix_kv_path).encode("utf-8"),
                    0,
                    token_array,
                    self._n_prefix,
                    ctypes.byref(n_token_count),
                )
                if n_loaded > 0:
                    sess.llama.n_tokens = self._n_prefix
                    sess.llama.input_ids[: self._n_prefix] = self._prefix_tokens
                    sess.active_program_id = self._id
                    return
            except Exception:
                logger.debug("KV cache load failed for %s, falling back to cold start", self._prefix_kv_path)

        # Cold start: evaluate the prefix and save to disk for next time
        sess.llama.reset()
        sess.llama.eval(self._prefix_tokens)
        try:
            llama_cpp.llama_state_seq_save_file(
                sess.llama.ctx,
                str(self._prefix_kv_path).encode("utf-8"),
                0,
                token_array,
                self._n_prefix,
            )
        except Exception:
            logger.debug("KV cache save failed for %s", self._prefix_kv_path)

        sess.active_program_id = self._id

    def _generate(self, gen_limit: int, temperature: float) -> str:
        """Token generation loop shared by __call__ and batch_call."""
        out_tokens: list[int] = []
        llama = self._session.llama
        eos = llama.token_eos()
        sample_temp = temperature if temperature > 0 else 0
        llama_sample = llama.sample
        llama_eval = llama.eval
        append = out_tokens.append

        for _ in range(gen_limit):
            tok = llama_sample(temp=sample_temp)
            if tok == eos:
                break
            append(tok)
            llama_eval([tok])

        return llama.detokenize(out_tokens).decode("utf-8", errors="replace").strip()

    def _tokenize_and_check(self, input_text: str, max_tokens: int | None) -> int:
        """Tokenize input, evaluate it, and return gen_limit. Must hold session lock."""
        sess = self._session
        sess.llama.n_tokens = self._n_prefix
        full_input = (input_text + self._suffix_text).encode("utf-8")
        input_tokens = sess.llama.tokenize(full_input, add_bos=False, special=True)
        used = self._n_prefix + len(input_tokens)
        remaining = sess.n_ctx - used
        if remaining <= 0:
            raise ValueError(
                f"input too long: {used} tokens used (prefix={self._n_prefix}, input={len(input_tokens)}); "
                f"context window is {sess.n_ctx}. Reload with a larger n_ctx."
            )
        gen_limit = remaining if max_tokens is None else min(max_tokens, remaining)
        sess.llama.eval(input_tokens)
        return gen_limit

    def __call__(
        self,
        input_text: str,
        *,
        max_tokens: int | None = 32,
        temperature: float = 0.0,
    ) -> str:
        """Run the program on `input_text` and return the (stripped) output string.

        `max_tokens` defaults to 32 so classifiers don't run away if the model
        ever fails to emit EOS. Pass `max_tokens=None` to use the full remaining
        context window.
        """
        with self._session.lock:
            self._activate()
            gen_limit = self._tokenize_and_check(input_text, max_tokens)
            return self._generate(gen_limit, temperature)

    def batch_call(
        self,
        inputs: list[str],
        *,
        max_tokens: int | None = 32,
        temperature: float = 0.0,
    ) -> list[str]:
        """Run the program on multiple inputs efficiently.

        This amortizes the activation cost across multiple inputs by keeping
        the adapter loaded and reusing the prefix KV cache.

        Args:
            inputs: List of input strings to process.
            max_tokens: Max tokens per output (default 32).
            temperature: Sampling temperature (default 0.0 for greedy).

        Returns:
            List of output strings.

        Example:
            >>> classifier = pawpaw.load("sentiment.paw")
            >>> inputs = ["Great!", "Terrible.", "Okay."]
            >>> classifier.batch_call(inputs)
            ['positive', 'negative', 'neutral']
        """
        if not inputs:
            return []

        results = []
        with self._session.lock:
            self._activate()
            for input_text in inputs:
                gen_limit = self._tokenize_and_check(input_text, max_tokens)
                results.append(self._generate(gen_limit, temperature))

        return results
