"""Speed benchmark: measure inference latency for pawpaw vs upstream programasweights.

Both bundles are loaded via pawpaw.load (binary-compatible .paw v2 format), so
the comparison isolates differences in the compiled program itself (LoRA size,
prompt template length) rather than runtime implementation differences.

Usage:
    python -m benchmarks.speed \\
        --pawpaw-bundle ./sentiment_bundle \\
        --upstream-program-id 3894309a9eac5c584899 \\
        --n 100

Outputs a markdown table of per-call latencies for various inputs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from benchmarks.common import TimedRun, host_info, time_calls


SAMPLE_INPUTS = [
    "I absolutely loved the new restaurant downtown!",
    "The customer service was terrible and I will never go back.",
    "The movie was somewhat okay, but mostly disappointing.",
    "Nothing about this product worked as advertised.",
    "Honestly the best pizza I've had in years.",
    "The hotel room was clean and quiet, perfect for a long stay.",
    "Long lines, bad food, and rude staff - skip it.",
    "Service was slow but the dessert made up for it.",
    "Crashed three times in the first hour. Refund requested.",
    "Solid build quality, snappy software, and a great battery.",
]


def _load_pawpaw(bundle_dir: Path):
    import pawpaw
    return pawpaw.load(bundle_dir, n_gpu_layers="auto", n_ctx=512)


def _load_upstream(program_id: str, *, use_pawpaw_runtime: bool):
    """Download via paw.function then optionally re-open the bundle in pawpaw.runtime."""
    import programasweights as paw
    handle = paw.function(program_id, n_gpu_layers=0, n_ctx=512)
    bundle_dir = handle._program_dir
    del handle
    if use_pawpaw_runtime:
        import pawpaw
        return pawpaw.load(bundle_dir, n_gpu_layers="auto", n_ctx=512), bundle_dir
    return paw.function(program_id, n_gpu_layers=-1, n_ctx=512), bundle_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pawpaw-bundle", type=Path, required=True, help="Path to a pawpaw-compiled bundle dir")
    parser.add_argument("--upstream-program-id", default=None, help="Upstream program ID to compare against")
    parser.add_argument("--n", type=int, default=50, help="Total iterations (first is warmup)")
    parser.add_argument("--also-upstream-runtime", action="store_true",
                        help="Also benchmark upstream's runtime to verify the runtime itself isn't the bottleneck")
    args = parser.parse_args()

    print(f"Host: {host_info()}\n")

    inputs: List[str] = (SAMPLE_INPUTS * (args.n // len(SAMPLE_INPUTS) + 1))[: args.n]

    runs: List[TimedRun] = []

    print("Loading pawpaw bundle...")
    pawpaw_fn = _load_pawpaw(args.pawpaw_bundle)
    runs.append(time_calls("pawpaw / pawpaw runtime", pawpaw_fn, inputs))

    if args.upstream_program_id:
        print("Loading upstream bundle (executed via pawpaw runtime)...")
        upstream_fn, bundle = _load_upstream(args.upstream_program_id, use_pawpaw_runtime=True)
        runs.append(time_calls("upstream / pawpaw runtime", upstream_fn, inputs))
        del upstream_fn

        if args.also_upstream_runtime:
            print("Loading upstream bundle (executed via upstream runtime)...")
            up_fn, _ = _load_upstream(args.upstream_program_id, use_pawpaw_runtime=False)
            runs.append(time_calls("upstream / upstream runtime", up_fn, inputs))

    print()
    print("| program | runtime | n | total (s) | per-call (ms) |")
    print("|---|---|---:|---:|---:|")
    for r in runs:
        prog, _, rt = r.label.partition(" / ")
        print(f"| {prog} | {rt} | {r.n} | {r.elapsed_s:.2f} | {r.per_call_ms:.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
