# pawpaw benchmarks

Two scripts that compare a `pawpaw`-compiled program against an upstream
`programasweights`-compiled program on the same task. Both can be loaded via
`pawpaw.load` (binary-compatible `.paw` v2 bundles), so the comparisons
isolate the *compilation pipeline*, not the runtime.

## Speed

Per-call inference latency on a small sample of inputs.

```bash
# After compiling a pawpaw program (e.g. via the tutorial):
python -m benchmarks.speed \
    --pawpaw-bundle ./sentiment_bundle \
    --upstream-program-id 3894309a9eac5c584899 \
    --n 50
```

Reports a markdown table of per-call latency for each (program, runtime) combo.
Add `--also-upstream-runtime` to verify upstream's own runtime isn't the bottleneck.

## Accuracy on SST-2

Compile a pawpaw sentiment program and evaluate on the SST-2 validation set
(Stanford Sentiment Treebank, 872 examples).

```bash
# 1. Compile (uses your local synth LLM)
python -m benchmarks.sst2_accuracy compile \
    --bundle ./sst2_pawpaw_bundle \
    --llm-model ~/.cache/pawpaw/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
    --n-per-category 30 --min-examples 80 --rank 16 --epochs 3

# 2. Evaluate
python -m benchmarks.sst2_accuracy eval \
    --bundle ./sst2_pawpaw_bundle \
    --n 500

# 3. (optional) Compare against an upstream-compiled program for the same task
python -m benchmarks.sst2_accuracy eval \
    --upstream-program-id <UPSTREAM_ID> \
    --n 500
```

Reports exact-match accuracy + a confusion matrix.

The SST-2 dataset is loaded via HuggingFace `datasets` from
`stanfordnlp/sst2`; first run downloads ~70 MB.

## What "competitive" means here

For a small classification task with `rank=16` and a few hundred synthetic
examples, you should expect:

- **Accuracy:** within 1-3 points of upstream on SST-2 validation (upstream
  trains on more synthesized data with their proprietary compiler model).
- **Speed:** within 2x of upstream when run via the same llama.cpp settings.
  Faster than upstream is achievable when your prompt template is shorter than
  upstream's pseudo-program (your KV-cache prefix is cheaper).

If accuracy or speed lags by more than that, check:

- Is the synth LLM small/weak? (4B-Instruct is the floor; smaller models hurt
  data quality enough to drop a few points of accuracy.)
- Is the adapter file valid f16 GGUF? Check `file <bundle>/adapter.gguf`.
  (Adapters are intentionally f16 — too small to benefit from quantization.)
- Is `n_gpu_layers="auto"` actually using GPU? Set `verbose=True` on `pawpaw.load`
  and look for "Metal" or "CUDA" in the load log.
