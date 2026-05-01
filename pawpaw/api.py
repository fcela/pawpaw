"""High-level convenience API.

Most users only need two functions: `build` and `load`.

# Build (slow, do once)
pawpaw.build(
    "Classify the user message as 'trivial' or 'substantive'.",
    save_to="programs/triage.paw",
    examples=[("How are you?", "trivial"), ("Why does my code crash?", "substantive")],
)

# Load (fast, do once at process start)
triage = pawpaw.load("programs/triage.paw")

# Apply (very fast, do many times)
triage("Hi there!")  # → "trivial"

For agentic workflows you typically load several programs at startup and call
them many times each. They share a base model in memory, so the per-program
cost is essentially just the LoRA adapter (a few MB).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

ExamplePair = tuple[str, str] | dict[str, str]


def _normalize_examples(examples: ExamplePair | Iterable[ExamplePair] | None) -> list[dict]:
    if not examples:
        return []
    out: list[dict] = []
    for e in examples:
        if isinstance(e, tuple) and len(e) == 2:
            out.append({"input": str(e[0]), "output": str(e[1])})
        elif isinstance(e, dict) and "input" in e and "output" in e:
            if not isinstance(e["input"], str) or not isinstance(e["output"], str):
                raise TypeError(f"example dict values must be strings: {e!r}")
            out.append({"input": e["input"], "output": e["output"]})
        else:
            raise TypeError(f"examples entries must be (input, output) tuples or dicts: {e!r}")
    return out


def build(
    spec: str,
    *,
    save_to: str | Path,
    examples: Iterable[ExamplePair] | None = None,
    base_model: str = "Qwen/Qwen3-0.6B",
    base_quant: str | None = None,
    rank: int = 16,
    epochs: int = 3,
    n_per_category: int = 30,
    min_examples: int = 100,
    llm_model_path: str | Path | None = None,
    llm_n_threads: int | None = None,
    llm_n_batch: int = 512,
    llm_n_gpu_layers: int | None = None,
    force: bool = False,
) -> "CompileResult":
    """Build a pawpaw program from a natural-language spec.

    The output is a `.paw` file (a single portable artifact ~5 MB) that you
    can copy between machines, ship with your service, or hash-pin in CI.
    A sibling bundle directory is also created alongside it for internal use.

    Args:
        spec: Natural-language description of what the program should do.
        save_to: Output path for the `.paw` file (e.g. `"programs/triage.paw"`).
            A bundle directory is created at the same location without the
            `.paw` suffix for internal use by the runtime.
        examples: Optional seed (input, output) pairs to guide training.
            If you provide enough (>= min_examples), synthesis is skipped
            entirely and no `llm_model_path` is needed.
        base_model: HF model id for the interpreter. Default Qwen3-0.6B.
        base_quant: Base model quantization: "Q4_K_M", "Q6_K", or "Q8_0".
            Default "Q6_K". "Q4_K_M" saves ~25% memory with minimal quality
            loss for classifiers.
        rank: LoRA rank. 16 is a reasonable default; raise for harder tasks.
        epochs: Training epochs. More epochs can improve accuracy but take longer.
        n_per_category: Number of synthetic examples to generate per category.
            Only used when `llm_model_path` is provided for synthesis.
        min_examples: Minimum examples required after deduplication. If fewer
            are produced, the build fails with a suggestion to lower
            `dedup_threshold` or raise `n_per_category`.
        llm_model_path: Path to a local GGUF file for the synthesis LLM.
            Required when you don't provide enough seed `examples`.
        llm_n_threads: CPU threads for synthesis LLM. Defaults to physical
            core count (capped at 8). Set PAWPAW_N_THREADS env var for global override.
        llm_n_batch: Prompt eval batch size for synthesis LLM. Default 512.
        llm_n_gpu_layers: GPU layers for synthesis LLM. "auto" uses GPU if
            available, 0 forces CPU-only. Default "auto".
        force: If True, ignore cached datasets and adapters, forcing a fresh
            synthesis + training run. Useful when you've changed settings but
            the cache would otherwise reuse old results.

    Returns:
        A CompileResult with the `.paw` path and example counts.

    Raises:
        ValueError: If spec is empty/too long, or too few examples after dedup.
        RuntimeError: If CUDA/MPS OOM even after auto-retry at batch_size=1.
    """
    from pawpaw.config import CompileOptions, SynthConfig, TrainConfig
    from pawpaw.pipeline import CompileResult, compile_spec

    if base_quant is not None:
        from pawpaw.runtime_cache import set_preferred_quant
        set_preferred_quant(base_quant)

    paw_path = Path(save_to)
    if paw_path.suffix != ".paw":
        paw_path = paw_path.with_suffix(".paw")
    bundle_dir = paw_path.with_suffix("")

    extra_examples = _normalize_examples(examples)

    synth_n_gpu = llm_n_gpu_layers
    if synth_n_gpu == "auto":
        try:
            import llama_cpp
            synth_n_gpu = -1 if llama_cpp.llama_supports_gpu_offload() else 0
        except (ImportError, AttributeError, OSError):
            synth_n_gpu = 0

    options = CompileOptions(
        base_model=base_model,
        synth=SynthConfig(
            n_per_category=n_per_category,
            min_examples=min_examples,
            llm_model_path=str(llm_model_path) if llm_model_path else None,
            llm_n_threads=llm_n_threads,
            llm_n_batch=llm_n_batch,
            llm_n_gpu_layers=synth_n_gpu,
        ),
        train=TrainConfig(
            lora_rank=rank,
            epochs=epochs,
        ),
    )

    hooks = None

    if extra_examples and not llm_model_path:
        from pawpaw.cache import CacheLayout, default_layout, put_dataset, spec_hash
        from pawpaw.pipeline import PipelineHooks

        layout = default_layout()
        h = spec_hash(spec, options)
        records = [
            {"input": e["input"], "output": e["output"], "category": "seed", "length_bucket": "any"}
            for e in extra_examples
        ]
        if not records or len(records) < min_examples:
            raise ValueError(
                f"need at least min_examples={min_examples} seed examples when no llm_model_path is set "
                f"(got {len(records)}). Provide more examples or pass llm_model_path."
            )
        put_dataset(layout, h, records)

        class _Unreachable:
            def complete(self, *_a, **_kw):
                raise AssertionError("LLM should not be invoked when seed examples cover min_examples")

        hooks = PipelineHooks(make_llm=lambda opts: _Unreachable())

    return compile_spec(
        spec=spec,
        options=options,
        out_paw_path=paw_path,
        bundle_dir=bundle_dir,
        hooks=hooks,
        force=force,
    )


def load(
    path: str | Path,
    *,
    n_ctx: int = 1024,
    n_gpu_layers: int | str = "auto",
    n_threads: int | None = None,
    n_batch: int = 512,
    n_ubatch: int | None = None,
    use_mlock: bool = False,
    use_mmap: bool = True,
    numa: bool = False,
    flash_attn: bool = True,
    verbose: bool = False,
    base_model_path: str | Path | None = None,
    base_quant: str | None = None,
) -> "Program":
    """Load a pawpaw program from a `.paw` file or bundle directory.

    The first call for a given base model may take a few seconds to load the
    model into memory. Subsequent programs sharing the same base model reuse
    it for free.

    Args:
        path: Path to a `.paw` file (recommended) or a bundle directory.
            `.paw` files are extracted on first load and cached under
            `~/.cache/pawpaw/paw_bundles/` for fast subsequent loads.
        n_ctx: Context window size in tokens. Increase if you hit
            "input too long" errors on long inputs. Default 1024.
        n_gpu_layers: Number of layers to offload to GPU. "auto" uses GPU
            if available, 0 forces CPU-only.
        n_threads: Number of CPU threads for llama.cpp. Defaults to
            physical core count (capped at 8) to avoid contention.
            Set PAWPAW_N_THREADS env var for a global override.
        n_batch: Prompt eval batch size. Larger values speed up prefix
            processing but use more memory. Default 512.
        n_ubatch: Micro-batch size for prompt eval. Defaults to n_batch.
        use_mlock: Lock model weights in RAM to prevent paging. Recommended
            on CPU-only servers to avoid latency spikes after idle.
        use_mmap: Memory-map the model file. Slightly slower startup but
            lower RAM usage. Default True; set False if use_mlock=True.
        numa: Enable NUMA-aware allocation for multi-socket CPU servers.
        flash_attn: Enable Flash Attention in llama.cpp. Significantly
            speeds up prompt eval (~10x) and overall inference (~1.3x)
            on CPU. Default True.
        verbose: If True, show llama.cpp debug output during loading.
        base_model_path: Override the base model GGUF path. By default, the
            model is looked up from the KNOWN_GGUFS registry or downloaded
            from HuggingFace (~400 MB on first use).
        base_quant: Base model quantization level: "Q4_K_M", "Q6_K", or
            "Q8_0". Default "Q6_K". "Q4_K_M" saves ~25% memory with minimal
            quality loss for classifiers.

    Returns:
        A Program object that you call like a function.

    Example:
        >>> triage = pawpaw.load("triage.paw")
        >>> triage("How are you?")
        'trivial'
    """
    from pawpaw.runtime import Program

    if base_quant is not None:
        from pawpaw.runtime_cache import set_preferred_quant
        set_preferred_quant(base_quant)

    return Program(
        path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, n_threads=n_threads,
        n_batch=n_batch, n_ubatch=n_ubatch, use_mlock=use_mlock, use_mmap=use_mmap,
        numa=numa, flash_attn=flash_attn, verbose=verbose,
        base_model_path=base_model_path,
    )


def clear_cache() -> None:
    """Clear all pawpaw caches: compiled datasets, trained adapters, and unpacked .paw bundles.

    This does NOT delete downloaded base models (those are managed by huggingface_hub).
    After clearing, the next `build()` call will re-synthesize and re-train from scratch.
    """
    import shutil
    from pawpaw.cache import default_layout
    from pawpaw.runtime import _paw_cache_dir

    layout = default_layout()
    if layout.root.exists():
        shutil.rmtree(layout.root, ignore_errors=True)
    paw_bundles = _paw_cache_dir()
    if paw_bundles.exists():
        shutil.rmtree(paw_bundles, ignore_errors=True)
