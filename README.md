# pawpaw

Build a sentence of English into a small LoRA adapter that runs locally and
returns a one-word answer in milliseconds. Useful for the routing decisions an
agent makes a thousand times a day:

- *Is this a trivial question or something substantive?*
- *Is the user frustrated?*
- *Is this about code, writing, or math?*

In the spirit of [ProgramAsWeights](https://programasweights.com), but
completely self-contained.

Each one becomes a `.paw` file (~5 MB) you load once and call many times.
Several of them in the same process share a base model, so you pay the GPU
memory cost once.

## Install

```bash
pip install pawpaw
```

That's it — no submodules, no C++ build step. Pure Python on top of
[llama-cpp-python](https://github.com/abetlen/llama-cpp-python). Works on
macOS (Metal), Linux + CUDA, and CPU-only hosts.

For build-only or inference-only installs, use extras:

```bash
pip install "pawpaw[build]"   # includes torch, transformers, peft for training
pip install "pawpaw[load]"    # includes torch, gguf for .paw loading only
pip install "pawpaw[all]"     # everything + psutil for auto thread detection
```

## The whole API

```python
import pawpaw

# Once, offline:
pawpaw.build(
    "Decide if the user message is 'trivial' (small talk / one-line lookup) "
    "or 'substantive' (needs reasoning or domain knowledge). Output one word.",
    save_to="programs/triage.paw",
    examples=[
        ("How are you doing today?", "trivial"),
        ("What time is it in Tokyo?", "trivial"),
        ("Why is my React component re-rendering?", "substantive"),
        ("Explain how attention heads work.", "substantive"),
        # … more seed examples, or pass llm_model_path=... to synthesize
    ],
)

# In your service / agent:
triage = pawpaw.load("programs/triage.paw")
mood = pawpaw.load("programs/mood.paw") # shares the base model with triage
intent = pawpaw.load("programs/intent.paw") # shares the base model too

label = triage(user_message) # e.g. "trivial"
if label == "substantive" and mood(user_message) == "frustrated":
    escalate(user_message)
```

A `.paw` is a single file you can copy between machines, ship with your
service, or hash-pin in CI. `pawpaw.load()` accepts a `.paw` file directly
— no unpacking needed.

## Performance tuning

`pawpaw.load()` and `pawpaw.build()` accept several parameters for
optimizing CPU performance:

```python
program = pawpaw.load(
    "programs/triage.paw",
    n_ctx=1024,          # context window (default 1024; classifiers rarely need more)
    n_threads=4,         # CPU threads (default: physical cores, capped at 8)
    n_batch=512,         # prompt eval batch size
    flash_attn=True,     # flash attention — ~10x faster prompt eval on CPU (default)
    use_mlock=True,      # lock weights in RAM (prevents paging on servers)
    numa=True,           # NUMA-aware allocation for multi-socket servers
    base_quant="Q4_K_M", # lighter base model (~25% less memory, minimal quality loss)
)
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_threads` | physical cores (≤8) | Avoids contention from oversubscription. Override with `PAWPAW_N_THREADS` env var. |
| `flash_attn` | `True` | ~10x faster prompt eval, ~2x lower p50 latency on CPU |
| `n_ctx` | `1024` | Reduces memory vs the old default of 4096; classifiers use ~110-250 tokens |
| `base_quant` | `"Q6_K"` | `"Q4_K_M"` saves ~25% memory; `"Q8_0"` gives higher quality. Override with `PAWPAW_BASE_QUANT` env var. |
| `use_mlock` | `False` | Prevents OS paging out model weights — avoids latency spikes after idle |

`pawpaw.build()` also accepts `llm_n_threads`, `llm_n_batch`, and
`llm_n_gpu_layers` to control the synthesis LLM.

## CPU bfloat16

On CPUs with AVX512_BF16 or AMX instructions (Sapphire Rapids, Zen 4+, or
Apple M2+), `pawpaw` automatically uses bfloat16 for training — roughly 2x
faster than float32. Override with the `PAWPAW_CPU_BF16=0` or `=1` env var.

## How it works

Building a program has three stages:

1. **Synthesize** training examples from the spec using a local LLM
   (`llama-cpp-python`). Or skip it by passing seed `examples`.
2. **Train a LoRA adapter** against the base model (`Qwen/Qwen3-0.6B` by
   default) using HuggingFace `peft` + `transformers`.
3. **Pack** the adapter into a `.paw` file containing the LoRA weights,
   prompt template, and metadata.

Execution loads the base GGUF once, attaches the program-specific LoRA, and
caches the prompt-prefix KV state on disk so cold-starts after the first call
are free. Multiple programs that share a base model also share its in-memory
context — loading three classifiers costs roughly the same memory as loading
one.

The `.paw` v2 binary container is documented in `pawpaw/format.py` and is
compatible with the
[programasweights](https://github.com/programasweights/programasweights-python)
runtime for ease of testing and benchmarking.

## Caching

`pawpaw.build()` caches synthesized datasets and trained adapters under
`~/.cache/pawpaw/`. If you re-run with the same spec and settings, the cached
results are reused silently. Use `force=True` to ignore the cache and
re-synthesize/re-train from scratch:

```python
pawpaw.build(spec, save_to="programs/triage.paw", force=True)
```

To clear all caches (datasets, adapters, unpacked bundles — but not base
models):

```python
pawpaw.clear_cache()
```

Set the `PAWPAW_CACHE` environment variable to use a different cache location.

## CLI

```bash
# Build a .paw program
pawpaw build spec.txt --save-to programs/triage.paw --llm-model /path/to/model.gguf

# Run a .paw program
pawpaw run programs/triage.paw "How are you doing today?"
```

## Tutorial

See [`tutorial.ipynb`](https://github.com/fcela/pawpaw/blob/main/tutorial.ipynb)
for a walkthrough covering build + save, load + call, multi-program
agents sharing a base model, long prompts, a speed comparison vs upstream, and
hardware auto-detection. [`how_it_works.ipynb`](https://github.com/fcela/pawpaw/blob/main/how_it_works.ipynb) provides a more detailed explanation of the methodology aimed at readers not familiar with transformers, LoRA adapters or PEFT.

## Benchmarks

`benchmarks/sst2_accuracy.py` builds a sentiment classifier with pawpaw and
evaluates exact-match accuracy on the SST-2 validation split (872 examples).
`benchmarks/speed.py` measures per-call latency for `.paw` programs.
See [benchmarks/README.md](https://github.com/pawpaw-dev/pawpaw/blob/main/benchmarks/README.md) for details.

## Development

```bash
git clone https://github.com/pawpaw-dev/pawpaw
cd pawpaw
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](https://github.com/pawpaw-dev/pawpaw/blob/main/LICENSE).
