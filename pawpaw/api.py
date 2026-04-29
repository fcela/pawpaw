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

ExamplePair = tuple[str, str] | dict


def _normalize_examples(examples: ExamplePair | Iterable[ExamplePair] | None) -> list[dict]:
    if not examples:
        return []
    out: list[dict] = []
    for e in examples:
        if isinstance(e, tuple) and len(e) == 2:
            out.append({"input": str(e[0]), "output": str(e[1])})
        elif isinstance(e, dict) and "input" in e and "output" in e:
            out.append({"input": str(e["input"]), "output": str(e["output"])})
        else:
            raise TypeError(f"examples entries must be (input, output) tuples or dicts: {e!r}")
    return out


def build(
    spec: str,
    *,
    save_to: str | Path,
    examples: Iterable[ExamplePair] | None = None,
    base_model: str = "Qwen/Qwen3-0.6B",
    rank: int = 16,
    epochs: int = 3,
    n_per_category: int = 30,
    min_examples: int = 100,
    llm_model_path: str | Path | None = None,
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
        rank: LoRA rank. 16 is a reasonable default; raise for harder tasks.
        epochs: Training epochs. More epochs can improve accuracy but take longer.
        n_per_category: Number of synthetic examples to generate per category.
            Only used when `llm_model_path` is provided for synthesis.
        min_examples: Minimum examples required after deduplication. If fewer
            are produced, the build fails with a suggestion to lower
            `dedup_threshold` or raise `n_per_category`.
        llm_model_path: Path to a local GGUF file for the synthesis LLM.
            Required when you don't provide enough seed `examples`.
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

    paw_path = Path(save_to)
    if paw_path.suffix != ".paw":
        paw_path = paw_path.with_suffix(".paw")
    bundle_dir = paw_path.with_suffix("")

    extra_examples = _normalize_examples(examples)

    options = CompileOptions(
        base_model=base_model,
        synth=SynthConfig(
            n_per_category=n_per_category,
            min_examples=min_examples,
            llm_model_path=str(llm_model_path) if llm_model_path else None,
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
    n_ctx: int = 4096,
    n_gpu_layers: int | str = "auto",
    verbose: bool = False,
    base_model_path: str | Path | None = None,
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
            "input too long" errors on long inputs. Default 4096.
        n_gpu_layers: Number of layers to offload to GPU. "auto" uses GPU
            if available, 0 forces CPU-only.
        verbose: If True, show llama.cpp debug output during loading.
        base_model_path: Override the base model GGUF path. By default, the
            model is looked up from the KNOWN_GGUFS registry or downloaded
            from HuggingFace (~400 MB on first use).

    Returns:
        A Program object that you call like a function.

    Example:
        >>> triage = pawpaw.load("triage.paw")
        >>> triage("How are you?")
        'trivial'
    """
    from pawpaw.runtime import DEFAULT_N_CTX, Program

    if n_ctx == 4096:
        n_ctx = DEFAULT_N_CTX
    return Program(path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=verbose,
                   base_model_path=base_model_path)


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
