"""CLI: `pawpaw <spec.txt> --save-to programs/myprog.paw [--llm-model PATH]`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from pawpaw.api import build as build_api


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pawpaw", description="Build a .paw program from a natural-language spec.")
    p.add_argument("spec_path", help="Path to a text file containing the spec")
    p.add_argument("--save-to", "-o", required=True, help="Output path for the .paw file")
    p.add_argument("--llm-model", default=None, help="Path to a local llama.cpp .gguf for the synth LLM")
    p.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--base-quant", default=None, choices=["Q4_K_M", "Q6_K", "Q8_0"],
                   help="Base model quantization (default Q6_K)")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--n-per-category", type=int, default=30)
    p.add_argument("--min-examples", type=int, default=100)
    p.add_argument("--n-threads", type=int, default=None,
                   help="CPU threads for llama.cpp (default: physical cores, capped at 8)")
    p.add_argument("--n-batch", type=int, default=512, help="Prompt eval batch size for synth LLM")
    p.add_argument("--llm-n-gpu-layers", type=int, default=None,
                   help="GPU layers for synth LLM (default: 0 / CPU-only)")
    p.add_argument("--force", action="store_true", help="Force re-synthesis and re-training, ignoring cache")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    spec_path = Path(args.spec_path)
    if not spec_path.exists():
        print(f"error: spec file not found: {spec_path}", file=sys.stderr)
        return 2
    spec = spec_path.read_text(encoding="utf-8").strip()

    try:
        result = build_api(
            spec,
            save_to=args.save_to,
            base_model=args.base_model,
            base_quant=args.base_quant,
            rank=args.rank,
            epochs=args.epochs,
            n_per_category=args.n_per_category,
            min_examples=args.min_examples,
            llm_model_path=args.llm_model,
            llm_n_threads=args.n_threads,
            llm_n_batch=args.n_batch,
            llm_n_gpu_layers=args.llm_n_gpu_layers,
            force=args.force,
        )
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    paw_size = result.paw_path.stat().st_size / 1024 / 1024
    print(f"built: {result.paw_path} ({paw_size:.1f} MB)")
    print(f"examples: {result.n_train_examples} train, {result.n_val_examples} val")
    return 0


if __name__ == "__main__":
    sys.exit(main())
