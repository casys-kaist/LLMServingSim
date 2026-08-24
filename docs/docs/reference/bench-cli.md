---
sidebar_position: 5
title: Bench CLI
---

# `python -m bench` CLI flags

Complete reference for the vLLM benchmark harness. `bench` has two
subcommands: `run` replays a workload through real vLLM, `validate`
compares a finished run against simulator output.

Both must run inside the **vLLM container**
(`scripts/docker-vllm.sh`), not the simulator container. See
**[Installation → vLLM environment](/docs/getting-started/installation/vllm)**.

For the resulting accuracy numbers, see
**[Validation](/docs/validation)**.

## `python -m bench run`

Strict replay: the runner reads a LLMServingSim-format JSONL workload
— the same format `python -m workloads.generators` emits and
`python -m serving --dataset` consumes — and pins every request's
`input_tok_ids` and `output_toks` via
`SamplingParams(min_tokens=N, max_tokens=N, ignore_eos=True)`. The
vLLM run therefore processes exactly the prompts the simulator sees,
in the same order.

### Required

| Flag | Type | Description |
| --- | --- | --- |
| `--model` | string | HF model id, passed verbatim to `vllm.AsyncLLM`. Unlike the simulator, this is a real load: weights are downloaded and placed on GPU |
| `--dataset` | path | LLMServingSim-format JSONL workload. See **[Workloads → JSONL format](/docs/workloads/jsonl-format)** |
| `--output-dir` | path | Where to write `meta.json` / `requests.jsonl` / `timeseries.csv`. Conventionally `bench/results/<run_id>/` |

### Parallelism

These are vLLM's own engine arguments, forwarded unchanged. They are
the bench-side counterpart of a cluster config's `tp_size` / `ep_size`
/ `dp_group`.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--tensor-parallel-size` | int | `1` | vLLM `tensor_parallel_size` |
| `--data-parallel-size` | int | `1` | vLLM `data_parallel_size` (DP across engines) |
| `--enable-expert-parallel` | flag | off | vLLM `enable_expert_parallel`, for MoE models |

### Scheduler and precision

Match these to the simulator run you intend to compare against, or the
comparison is not apples-to-apples.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--max-num-seqs` | int | `128` | vLLM `max_num_seqs`, the per-engine running cap |
| `--max-num-batched-tokens` | int | `2048` | vLLM `max_num_batched_tokens` |
| `--max-model-len` | int | `None` | vLLM `max_model_len`. `None` uses the model's own maximum |
| `--dtype` | string | `bfloat16` | Model dtype |
| `--kv-cache-dtype` | string | `auto` | vLLM `kv_cache_dtype` |
| `--seed` | int | `42` | Sampling seed |

:::note[Defaults differ from `python -m serving`]
`bench run` defaults `--dtype` to `bfloat16` outright, where the
simulator resolves it from the model config's `torch_dtype`. `bench`
also has no `--block-size`: vLLM picks the KV block size itself, and
the value it settled on is recorded in `meta.json` under
`kv_cache.block_size`. Read it back and pass it to the simulator as
`--block-size` if you want the two to line up.
:::

### Workload and output

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--num-reqs` | int | `0` | Cap on requests taken from the dataset. `0` = replay all |
| `--tick-seconds` | float | `1.0` | Stat-logger downsample interval, i.e. row spacing in `timeseries.csv`. Matches the simulator's `--log-interval` |
| `--log-level` | choice | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

### Output

```
<output-dir>/
  meta.json          run metadata plus what vLLM *resolved*: kv_cache
                     (num_gpu_blocks, block_size, num_kv_tokens,
                     gpu_memory_utilization), hardware (device name,
                     total memory, CUDA / torch versions), and
                     resolved_config -- the whole VllmConfig, one key
                     per sub-config
  requests.jsonl     per request: request_id, input_toks, output_toks,
                     arrival_time, queued_ts, scheduled_ts,
                     first_token_ts, last_token_ts
  timeseries.csv     per tick: t, prompt_throughput, gen_throughput,
                     running, waiting, kv_cache_pct
```

`meta.json`'s `kv_cache.num_gpu_blocks` is the number worth reading
first: it is the KV capacity the simulator has to match, and the only
place vLLM's activation peak shows up, since the rest of its memory
budget is known up front. The simulator does not model that peak, so
its capacity at the same `mem_util` is an upper bound. See
**[KV cache and memory](/docs/simulator/scheduling/kv-cache-and-memory)**.

The dataset is never modified — generation lives in
`workloads/generators`.

## `python -m bench validate`

Loads the bench artifacts plus the simulator's per-request CSV and log
for the same workload, derives TTFT / TPOT / end-to-end latency on both
sides under matched definitions, and writes plots and a numeric summary
into a subdirectory of the bench run.

### Required

| Flag | Type | Description |
| --- | --- | --- |
| `--bench-dir` | path | A finished `bench run` output directory |
| `--sim-csv` | path | Simulator per-request CSV, i.e. whatever you passed to `python -m serving --output` |
| `--sim-log` | path | Simulator log, parsed for per-tick running / waiting counts. Capture it by redirecting the simulator's stdout |

### Optional

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--output-subdir` | string | `validation` | Subdirectory under `--bench-dir` for plots and summary |
| `--prefix` | string | `""` | Filename prefix for the generated files |
| `--title` | string | `vLLM vs LLMServingSim` | Plot title suffix |
| `--log-level` | choice | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

### Output

```
<bench-dir>/<output-subdir>/
  <prefix>_throughput.png     prompt + generation throughput, both sides
  <prefix>_requests.png       running / waiting counts over time
  <prefix>_latency.png        TTFT / TPOT / latency CDFs
  <prefix>_summary.txt        mean and P50 / P90 / P95 / P99 per metric, with diff%
```

### Matched metric definitions

Both sides compute the same three quantities from the same reference
points, so `diff%` is meaningful:

| Metric | Definition |
| --- | --- |
| TTFT | `first_token_ts - arrival_time` (queueing included) |
| TPOT | `(last_token_ts - first_token_ts) / max(1, output_toks - 1)` |
| Latency | `last_token_ts - arrival_time` |

The simulator's CSV exposes `arrival`, `end_time`, and a per-token ITL
list directly; bench derives the same fields from vLLM's
`RequestStateStats`.

:::caution[`prompt_throughput` is not comparable tick-for-tick]
vLLM counts a prompt once, at prefill completion, so its
`prompt_throughput` series cannot show preemption and recomputation.
The simulator's per-tick prompt tokens can. Compare the per-request
metrics above, and read the throughput plots as shape rather than as
matched values.
:::

## Shell wrappers

Two host-side wrappers set the flags for you. Both are meant to be
edited in place or driven by environment variables.

### `bench/bench.sh`

Every knob is an environment variable with a default:

```bash
MODEL=Qwen/Qwen3-32B \
DATASET=workloads/sharegpt-qwen3-32b-300-sps10.jsonl \
TP=2 DP=1 EXPERT_PARALLEL=0 \
MAX_NUM_SEQS=128 MAX_NUM_BATCHED_TOKENS=2048 \
./bench/bench.sh
```

| Variable | Flag it sets | Default |
| --- | --- | --- |
| `MODEL` | `--model` | `Qwen/Qwen3-32B` |
| `DATASET` | `--dataset` | `workloads/sharegpt-qwen3-32b-300-sps10.jsonl` |
| `RUN_ID` | (names the output dir) | `$(date +%Y%m%d-%H%M%S)` |
| `OUTPUT_DIR` | `--output-dir` | `bench/results/$RUN_ID` |
| `TP` | `--tensor-parallel-size` | `2` |
| `DP` | `--data-parallel-size` | `1` |
| `EXPERT_PARALLEL` | `--enable-expert-parallel` when `1` | `0` |
| `MAX_NUM_SEQS` | `--max-num-seqs` | `128` |
| `MAX_NUM_BATCHED_TOKENS` | `--max-num-batched-tokens` | `2048` |
| `MAX_MODEL_LEN` | `--max-model-len`, omitted when blank | blank |
| `DTYPE` | `--dtype` | `bfloat16` |
| `KV_CACHE_DTYPE` | `--kv-cache-dtype` | `auto` |
| `SEED` | `--seed` | `42` |
| `TICK_SECONDS` | `--tick-seconds` | `1.0` |
| `NUM_REQS` | `--num-reqs` | `0` |
| `LOG_LEVEL` | `--log-level` | `INFO` |

Note `TP=2` — the wrapper's default is not vLLM's `1`.

### `bench/validate.sh`

Positional, with three environment overrides:

```bash
./bench/validate.sh <bench_dir> <sim_csv> <sim_log> [prefix]
```

| Position / variable | Flag it sets | Default |
| --- | --- | --- |
| `$1` | `--bench-dir` | required |
| `$2` | `--sim-csv` | required |
| `$3` | `--sim-log` | required |
| `$4` | `--prefix`, omitted when blank | blank |
| `OUTPUT_SUBDIR` | `--output-subdir` | `validation` |
| `TITLE` | `--title` | `vLLM vs LLMServingSim` |
| `LOG_LEVEL` | `--log-level` | `INFO` |

## Committed examples

`bench/examples/` holds four end-to-end runs, keyed
`<hardware>/<model>` — a dense single-GPU baseline, a TP=2 dense run and
a DP+EP MoE run on RTXPRO6000, plus the same dense baseline on an
RTX 4090 — each bundling its cluster `config.json`, the vLLM artifacts,
the simulator output, and the resulting validation summary and plots.
`bench/examples/run.sh <hardware>/<model>` re-runs the simulator side
and `bench/examples/validate.sh <hardware>/<model>` re-runs the
comparison; both take every example when given no argument. The headline
numbers are on **[Validation](/docs/validation)**.

## What's next

- **[Validation](/docs/validation)**: what these runs actually measured.
- **[For Contributors → Validating your changes](/docs/contributor/validating-changes)**:
  the regression workflow that uses this harness.
- **[Workloads → ShareGPT generators](/docs/workloads/sharegpt-generators)**:
  producing a dataset for `--dataset`.
