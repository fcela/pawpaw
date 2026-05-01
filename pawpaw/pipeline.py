"""Pipeline orchestrator: spec → .paw file.

External effects (LLM init, training, GGUF conversion) are injected via PipelineHooks
so unit tests can stub them out.
"""
from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field, replace as dc_replace
from pathlib import Path
from typing import Callable

from pawpaw.version import PIPELINE_VERSION

logger = logging.getLogger(__name__)


def _default_make_llm(options) -> "LLM":
    from pawpaw.synth.llm import LlamaCppLLM
    if not options.synth.llm_model_path:
        raise RuntimeError(
            "llm_model_path must be set to a local llama.cpp .gguf model for synthesis. "
            "For example: --llm-model ~/.cache/pawpaw/models/qwen3-4b-instruct-q4_k_m.gguf"
        )
    return LlamaCppLLM(
        model_path=options.synth.llm_model_path,
        seed=options.synth.llm_seed,
        n_threads=options.synth.llm_n_threads,
        n_batch=options.synth.llm_n_batch,
        n_gpu_layers=options.synth.llm_n_gpu_layers,
    )


def _default_train(*args, **kwargs) -> Path:
    from pawpaw.train.trainer import train_lora
    return train_lora(*args, **kwargs)


def _default_gguf(*args, **kwargs) -> Path:
    from pawpaw.pack.gguf_convert import peft_to_gguf
    return peft_to_gguf(*args, **kwargs)


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        exc.__class__.__name__ == "OutOfMemoryError"
        or "out of memory" in msg
        or "out-of-memory" in msg
        or "outofmemoryerror" in msg
    )


def _has_adapter_files(peft_dir: Path) -> bool:
    return (
        (peft_dir / "adapter_config.json").is_file()
        and (
            (peft_dir / "adapter_model.safetensors").is_file()
            or (peft_dir / "adapter_model.bin").is_file()
        )
    )


def _train_with_oom_retry(hook, *, base_model, template, pairs, config, output_dir) -> Path:
    try:
        return hook(base_model=base_model, template=template, pairs=pairs, config=config, output_dir=output_dir)
    except (RuntimeError, MemoryError) as e:
        if not _is_oom(e):
            raise
        if config.per_device_batch_size <= 1:
            raise RuntimeError(
                "CUDA/MPS out of memory at per_device_batch_size=1. Try a smaller base model "
                "or reduce LoRA rank."
            ) from e
        retry_config = dc_replace(
            config,
            per_device_batch_size=max(1, config.per_device_batch_size // 2),
            gradient_accumulation_steps=config.gradient_accumulation_steps * 2,
        )
        logger.warning("OOM during training, retrying with batch_size=%d (was %d)",
                        retry_config.per_device_batch_size, config.per_device_batch_size)
        return hook(base_model=base_model, template=template, pairs=pairs, config=retry_config, output_dir=output_dir)


def _validate_outputs(*, paw_path: Path, bundle_dir: Path, holdout_pair: dict | None) -> None:
    """Validate single-file .paw; optional bundle smoke-test gated on PAWPAW_SMOKE_TEST=1."""
    from pawpaw.format import validate
    result = validate(paw_path)
    if not result.ok:
        raise RuntimeError(f".paw validation failed: {result.errors}")
    if os.environ.get("PAWPAW_SMOKE_TEST") != "1" or holdout_pair is None:
        return
    from pawpaw.runtime import Program
    fn = Program(bundle_dir)
    out = fn(holdout_pair["input"])
    if not isinstance(out, str) or not out:
        raise RuntimeError(f"bundle smoke test produced empty output for input {holdout_pair['input']!r}")


@dataclass
class PipelineHooks:
    make_llm: Callable = field(default=_default_make_llm)
    train_lora: Callable[..., Path] = field(default=_default_train)
    peft_to_gguf: Callable[..., Path] = field(default=_default_gguf)


@dataclass
class CompileResult:
    paw_path: Path
    bundle_dir: Path
    n_train_examples: int
    n_val_examples: int


def compile_spec(
    *,
    spec: str,
    options: "CompileOptions",
    out_paw_path: Path,
    bundle_dir: Path,
    cache_root: Path | None = None,
    hooks: PipelineHooks | None = None,
    force: bool = False,
) -> CompileResult:
    from pawpaw.cache import CacheLayout, default_layout, get_dataset, put_dataset, spec_hash
    from pawpaw.config import CompileOptions
    from pawpaw.pack.bundle import BundleMeta, write_directory
    from pawpaw.pack.paw_file import write_paw_file
    from pawpaw.synth.dedup import dedup
    from pawpaw.synth.examples import Pair, generate_all
    from pawpaw.synth.llm import LLM
    from pawpaw.synth.taxonomy import enumerate_categories
    from pawpaw.train.prompt_template import build_prompt_template

    if not spec or len(spec.encode("utf-8")) > 8 * 1024:
        raise ValueError("spec must be non-empty and < 8 KB")

    hooks = hooks or PipelineHooks()
    layout = CacheLayout(root=cache_root) if cache_root else default_layout()
    h = spec_hash(spec, options)

    cached = get_dataset(layout, h)
    if cached is not None and not force:
        logger.info("reusing cached dataset for spec_hash=%s (use force=True to re-synthesize)", h[:12])
    else:
        if force and cached is not None:
            logger.info("force=True: re-synthesizing dataset (ignoring cache for spec_hash=%s)", h[:12])
        else:
            logger.info("synthesizing dataset for spec_hash=%s", h[:12])
        llm = hooks.make_llm(options)
        cats = enumerate_categories(spec, llm, n_categories=10)
        raw = generate_all(spec, cats, llm, n_per_category=options.synth.n_per_category)
        deduped = dedup(raw, threshold=options.synth.dedup_threshold)
        if len(deduped) < options.synth.min_examples:
            raise ValueError(
                f"too few examples after dedup: {len(deduped)} < min_examples={options.synth.min_examples}. "
                f"Lower dedup_threshold, raise n_per_category, or expand the spec."
            )
        records = [
            {"input": p.input, "output": p.output, "category": p.category, "length_bucket": p.length_bucket}
            for p in deduped
        ]
        put_dataset(layout, h, records)
        cached = records

    pairs = [
        Pair(input=r["input"], output=r["output"], category=r["category"], length_bucket=r["length_bucket"])
        for r in cached
    ]

    rng = random.Random(options.train.seed)
    demos = rng.sample(pairs, k=min(2, len(pairs)))
    template = build_prompt_template(spec, demos=demos)

    peft_dir = layout.peft_dir(h)
    if not force and _has_adapter_files(peft_dir):
        logger.info("reusing cached adapter for spec_hash=%s (use force=True to re-train)", h[:12])
    else:
        if force and peft_dir.exists():
            logger.info("force=True: re-training adapter (ignoring cache for spec_hash=%s)", h[:12])
        _train_with_oom_retry(
            hooks.train_lora,
            base_model=options.base_model,
            template=template,
            pairs=pairs,
            config=options.train,
            output_dir=peft_dir,
        )

    gguf_path = hooks.peft_to_gguf(peft_dir, out_dir=layout.dir_for(h))

    examples_for_meta = [{"input": p.input, "output": p.output} for p in pairs[:5]]

    bundle_meta = BundleMeta(
        spec=spec,
        interpreter_model=options.base_model,
        spec_hash=h,
        pipeline_version=PIPELINE_VERSION,
        lora_rank=options.train.lora_rank,
        lora_alpha=options.train.effective_alpha,
        target_modules=options.train.target_modules,
        examples=examples_for_meta,
    )
    write_directory(out_dir=bundle_dir, gguf_path=gguf_path, prompt_template=template, meta=bundle_meta)

    write_paw_file(
        out_path=out_paw_path,
        peft_dir=peft_dir,
        spec=spec,
        prompt_template=template,
        examples=examples_for_meta,
        interpreter_model=options.base_model,
    )

    holdout = {"input": pairs[-1].input, "output": pairs[-1].output} if pairs else None
    _validate_outputs(paw_path=out_paw_path, bundle_dir=bundle_dir, holdout_pair=holdout)

    return CompileResult(
        paw_path=out_paw_path,
        bundle_dir=bundle_dir,
        n_train_examples=len(pairs),
        n_val_examples=max(1, int(len(pairs) * options.train.val_fraction)),
    )
