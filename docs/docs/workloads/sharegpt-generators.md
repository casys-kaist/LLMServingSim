---
sidebar_position: 3
title: ShareGPT generator
---

# ShareGPT generator

ShareGPT is the de facto standard inference benchmark, a curated
dataset of real human ↔ ChatGPT conversations spanning a wide range
of prompt lengths and use cases. The bundled generator turns ShareGPT
(or any compatible Hugging Face text dataset) into the JSONL format
the simulator consumes, with proper tokenization for prefix caching.

## Quick run

From the vLLM Docker container at `/workspace`:

```bash
python -m workloads.generators sharegpt \
  --model meta-llama/Llama-3.1-8B \
  --source shibing624/sharegpt_gpt4 \
  --num-reqs 300 --sps 10 --seed 42 \
  --output workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl
```

That produces a flat-format workload with 300 requests arriving at
10 sessions/second on average, tokenized with the Llama-3.1-8B
tokenizer.

`workloads/examples/` contains ready-to-edit templates for the
bundled models, copy and tweak:

```bash
ls workloads/examples/
# gen-llama-3.1-8b.sh
# gen-qwen3-30b-a3b.sh
# gen-qwen3-32b.sh
```

## Why use the vLLM container

The generator imports `transformers` for tokenization and (optionally)
`vllm` for free-generation mode. Both are pre-installed in the vLLM
Docker image. Run from inside `scripts/docker-vllm.sh` and you don't
need to manage Python deps yourself.

For gated models (Llama 3.x, etc.), set `HF_TOKEN` before launching
the container, see
**[Installation → vLLM setup](/docs/getting-started/installation/vllm)**.

## Options, grouped

### Source and model

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | (required) | HuggingFace model id; used for tokenization (and optionally free-generation) |
| `--source` | `shibing624/sharegpt_gpt4` | HF dataset id or local path. Any dataset with `conversations` field works |
| `--output` | (required) | Output JSONL path |

### Sampling

| Flag | Default | Meaning |
| --- | --- | --- |
| `--num-reqs` | (required) | How many requests / sessions to emit |
| `--sps` | (required) | Sessions per simulated second (Poisson arrival) |
| `--seed` | `42` | RNG seed for sampling and arrival times |
| `--first-arrival-sec` | `0` | Offset for the first request's arrival time |

### Length filters

Drop requests outside these ranges from the source dataset:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--min-input-toks` | `0` | Minimum prompt tokens (after tokenization) |
| `--max-input-toks` | `16384` | Maximum prompt tokens |
| `--min-output-toks` | `0` | Minimum output tokens |
| `--max-output-toks` | `16384` | Maximum output tokens |
| `--max-kv-toks` | `16384` | Cap `input + output` tokens (KV-cache footprint) |
| `--max-sessions` | `5000` | Cap the number of source sessions sampled before filtering |

A reasonable starting point: `--min-input-toks 256 --min-output-toks
512` filters out very short conversations that aren't representative
of real serving traffic.

### Fixed-length mode

For controlled stress tests, fix the prompt and output lengths:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--fix-len` | off | Enable fixed-length mode |
| `--fix-input-length` | `128` | Prompt tokens |
| `--fix-output-length` | `512` | Output tokens |

In this mode, the generator still pulls real conversations from the
source dataset for prefix-cache realism, but truncates / pads each
to the fixed lengths.

### Pulse arrival pattern

A burst-mode arrival pattern that approximates the "everyone hits
the API at the top of the hour" production phenomenon:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--pulse` | off | Enable pulse mode |
| `--pulse-n` | `10` | Number of requests per pulse |
| `--pulse-delay-sec` | `60` | Time between pulses |
| `--pulse-poisson` | off | Within each pulse, use Poisson arrivals at the configured `--sps` instead of all-at-once |

Without `--pulse-poisson`, pulse arrivals all fire at the start of
each pulse window, useful for testing the simulator's burst-handling
behavior.

### vLLM free-generation mode (optional)

Instead of using the source dataset's response field for output
tokens, **regenerate** outputs with vLLM. This produces outputs
matching the model you'll run in the simulator:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--use-vllm` | off | Use vLLM to free-generate outputs |
| `--vllm-tp` | `1` | `tensor_parallel_size` for the offline engine |
| `--vllm-dtype` | `bfloat16` | vLLM weight dtype |
| `--vllm-max-num-seqs` | `1024` | `max_num_seqs`. Deliberately high — this is a throughput job, not a latency measurement |
| `--vllm-max-num-batched-tokens` | `16384` | `max_num_batched_tokens`, likewise high for throughput |
| `--vllm-max-model-len` | model's max | Override `max_model_len` |
| `--vllm-temperature` | `0.0` | Sampling temperature. `0` = greedy, which makes generation reproducible for a given seed |
| `--vllm-repetition-penalty` | `1.1` | Down-weights already-emitted tokens. `1.0` disables it |

:::note[Why the repetition penalty defaults above 1.0]
Free generation at `1.0` tends to ramble past the natural stopping
point, so EOS never fires and every request runs to
`--max-output-toks`. `1.1` keeps natural EOS landing at typical
ShareGPT lengths (~500–1000 tokens), which is what makes the resulting
output-length distribution look like the real dataset's.
:::

The five `--vllm-*` engine knobs above are throughput settings for the
generation job itself. They have **no** bearing on the simulated run:
the generator only records token counts and ids, so `--vllm-max-num-seqs
1024` here does not mean the workload should be simulated at
`--max-num-seqs 1024`.

With `--use-vllm`:

1. The prompt is taken from ShareGPT.
2. vLLM generates a fresh response with that model.
3. Both prompt and response are tokenized; `input_tok_ids` and
   `output_tok_ids` are populated.

Without `--use-vllm`:

- Both prompt and response come from the ShareGPT entry as text.
- Only the prompt is re-tokenized with `--model`'s tokenizer for
  `input_tok_ids`.

Use `--use-vllm` when you specifically want output token IDs to
match what the model would actually produce. For most simulator
runs this isn't needed (the simulator doesn't generate text, it
just counts tokens), but it's useful for downstream evaluation or if
you want fully self-consistent traces.

## Output format

The generator writes one JSONL line per request:

```json
{"input_toks": 1472, "output_toks": 133, "arrival_time_ns": 4059740, "input_tok_ids": [...], "output_tok_ids": [...]}
```

Always **flat format**: ShareGPT entries don't have dependency
chains. For agentic workloads see
**[Agentic sessions](./agentic-sessions)**.

The output filename convention is
`sharegpt-<model-short>-<n>-sps<rate>.jsonl` (matches the bundled
files).

## Tips

1. **Tokenize with the same model the simulator will run.** Otherwise
   the prefix-cache hit rate in the simulator won't match what
   production would see. The bundled JSONL files use this convention
   and are paired with their respective models.
2. **`--max-sessions` caps the source sample, not the output.**
   Increase it if you're applying tight length filters and not
   getting enough surviving requests. Default 5000 is enough for most
   `--num-reqs` values.
3. **Pulse mode is great for sanity tests.** A clean burst pattern
   exposes scheduler behavior that smooth Poisson arrivals can hide
   (queue buildup, fairness, head-of-line blocking).
4. **Generation speed.** Without `--use-vllm`, generation is
   tokenization-bound and finishes in seconds. With `--use-vllm`, you
   pay real vLLM inference cost, minutes to hours depending on
   `--num-reqs`. Cache the output JSONL.
5. **Reuse JSONL across simulator runs.** Generate once, simulate
   many times. The file is small (~MB) and self-contained.

## What's next

- **[JSONL format](./jsonl-format)**: schema reference for what the
  generator produces.
- **[Agentic sessions](./agentic-sessions)**: for closed-loop
  workloads. The ShareGPT generator only produces flat workloads.
