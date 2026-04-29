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
mood = pawpaw.load("programs/mood.paw")    # shares the base model with triage
intent = pawpaw.load("programs/intent.paw") # shares the base model too

label = triage(user_message) # e.g. "trivial"
if label == "substantive" and mood(user_message) == "frustrated":
    escalate(user_message)
```

A `.paw` is a single file you can copy between machines, ship with your
service, or hash-pin in CI. `pawpaw.load()` accepts a `.paw` file directly
— no unpacking needed.

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

## Tutorial

See [`tutorial.ipynb`](https://github.com/pawpaw-dev/pawpaw/blob/main/tutorial.ipynb)
for a 7-section walkthrough covering build + save, load + call, multi-program
agents sharing a base model, long prompts, a speed comparison vs upstream, and
hardware auto-detection.

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
