"""Accuracy benchmark on SST-2 (Stanford Sentiment Treebank, 2-class).

Usage:
    # 1. Compile pawpaw program (once)
    python -m benchmarks.sst2_accuracy compile --bundle ./sst2_pawpaw_bundle \\
        --llm-model ~/.cache/pawpaw/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf

    # 2. Evaluate on the SST-2 validation set
    python -m benchmarks.sst2_accuracy eval --bundle ./sst2_pawpaw_bundle --n 500

    # 3. Optional: evaluate an upstream-compiled program against the same set
    python -m benchmarks.sst2_accuracy eval --upstream-program-id <ID> --n 500

The validation set is 872 examples; we default to the first 500 for speed.

Reports exact-match accuracy and a confusion matrix.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

from benchmarks.common import accuracy, host_info, normalize_label

SPEC = (
    "Classify the sentiment of the input text as either 'positive' or 'negative'. "
    "Output exactly one word: positive or negative."
)


def _load_sst2_validation(n: int) -> Tuple[List[str], List[str]]:
    """Returns (texts, labels) where labels are 'positive'/'negative' strings."""
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2", split="validation")
    if n:
        ds = ds.select(range(min(n, len(ds))))
    label_map = {0: "negative", 1: "positive"}
    texts = [row["sentence"].strip() for row in ds]
    labels = [label_map[row["label"]] for row in ds]
    return texts, labels


def cmd_compile(args) -> int:
    from pawpaw.config import CompileOptions, SynthConfig, TrainConfig
    from pawpaw.pipeline import compile_spec

    out_paw = Path(args.out_paw or (Path(args.bundle).parent / (Path(args.bundle).name + ".paw")))
    options = CompileOptions(
        synth=SynthConfig(
            n_per_category=args.n_per_category,
            min_examples=args.min_examples,
            llm_model_path=args.llm_model,
        ),
        train=TrainConfig(
            lora_rank=args.rank,
            lora_alpha=2 * args.rank,
            epochs=args.epochs,
            per_device_batch_size=args.batch_size,
        ),
    )
    print(f"Compiling SST-2 sentiment program to {args.bundle}/")
    result = compile_spec(
        spec=SPEC,
        options=options,
        out_paw_path=out_paw,
        bundle_dir=Path(args.bundle),
    )
    print(f"  paw:      {result.paw_path}  ({result.paw_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  bundle:   {result.bundle_dir}")
    print(f"  examples: {result.n_train_examples} train, {result.n_val_examples} val")
    return 0


def cmd_eval(args) -> int:
    print(f"Host: {host_info()}\n")
    texts, labels = _load_sst2_validation(args.n)
    print(f"Loaded {len(texts)} SST-2 validation examples.")

    import pawpaw

    if args.bundle:
        print(f"Evaluating pawpaw bundle: {args.bundle}")
        fn = pawpaw.load(args.bundle, n_gpu_layers="auto", n_ctx=args.n_ctx)
        program_label = f"pawpaw bundle ({Path(args.bundle).name})"
    else:
        if not args.upstream_program_id:
            print("error: --bundle or --upstream-program-id required", file=sys.stderr)
            return 2
        import programasweights as paw
        print(f"Downloading upstream program {args.upstream_program_id}...")
        handle = paw.function(args.upstream_program_id, n_gpu_layers=0, n_ctx=args.n_ctx)
        bundle = handle._program_dir
        del handle
        fn = pawpaw.load(bundle, n_gpu_layers="auto", n_ctx=args.n_ctx)
        program_label = f"upstream {args.upstream_program_id} (run via pawpaw)"

    predictions: List[str] = []
    start = time.perf_counter()
    for i, text in enumerate(texts):
        # Cap output to avoid runaway generation on confused inputs.
        out = fn(text, max_tokens=8)
        predictions.append(out)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(texts)} ({(i + 1) / (time.perf_counter() - start):.1f}/s)")

    elapsed = time.perf_counter() - start
    acc = accuracy(predictions, labels)

    # Confusion matrix
    cm = {("positive", "positive"): 0, ("positive", "negative"): 0,
          ("negative", "positive"): 0, ("negative", "negative"): 0}
    invalid = 0
    for p, r in zip(predictions, labels):
        pn, rn = normalize_label(p), normalize_label(r)
        if pn not in {"positive", "negative"}:
            invalid += 1
            continue
        cm[(pn, rn)] += 1

    print(f"\n=== {program_label} ===")
    print(f"Examples evaluated: {len(texts)}")
    print(f"Total time:         {elapsed:.1f}s ({len(texts)/elapsed:.1f}/s)")
    print(f"Exact-match acc:    {acc:.4f}  ({int(acc*len(texts))}/{len(texts)})")
    print(f"Invalid outputs:    {invalid}")
    print(f"Confusion (pred → label):")
    print(f"  pred=pos, label=pos: {cm[('positive','positive')]}")
    print(f"  pred=pos, label=neg: {cm[('positive','negative')]}")
    print(f"  pred=neg, label=pos: {cm[('negative','positive')]}")
    print(f"  pred=neg, label=neg: {cm[('negative','negative')]}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("compile")
    pc.add_argument("--bundle", required=True, help="Output bundle directory")
    pc.add_argument("--out-paw", default=None, help="Single-file .paw output (default: <bundle>.paw)")
    pc.add_argument("--llm-model", required=True, help="Path to local synth LLM (.gguf)")
    pc.add_argument("--n-per-category", type=int, default=30)
    pc.add_argument("--min-examples", type=int, default=80)
    pc.add_argument("--rank", type=int, default=16)
    pc.add_argument("--epochs", type=int, default=3)
    pc.add_argument("--batch-size", type=int, default=4)
    pc.set_defaults(func=cmd_compile)

    pe = sub.add_parser("eval")
    grp = pe.add_mutually_exclusive_group(required=True)
    grp.add_argument("--bundle", help="pawpaw-compiled bundle directory to evaluate")
    grp.add_argument("--upstream-program-id", help="Upstream program ID to download and evaluate")
    pe.add_argument("--n", type=int, default=500, help="Number of validation examples (max 872)")
    pe.add_argument("--n-ctx", type=int, default=512)
    pe.set_defaults(func=cmd_eval)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
