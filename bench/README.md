# bench

End-to-end vLLM benchmark + simulator validation. Runs a real vLLM
serving workload, captures per-request timing and per-tick scheduler
state, and compares the result against the simulator's output for the
same dataset.

## Layout

```
bench/                          Python package — `python -m bench ...`
├── __init__.py                 package marker + module map
├── __main__.py                 CLI dispatch (run / validate)
├── core/                       internals
│   ├── runner.py               AsyncLLM driver, captures RequestStateStats
│   ├── recorder.py             writes meta.json / requests.jsonl / timeseries.csv
│   ├── stat_logger.py          custom vLLM StatLoggerBase that fills timeseries
│   ├── validate.py             bench-vs-sim comparison entry point
│   ├── plots.py                throughput / running-waiting / latency-CDF helpers
│   └── logger.py               Rich-based logger + stdio capture
├── bench.sh                    host-side ``python -m bench run`` wrapper
├── validate.sh                 host-side ``python -m bench validate`` wrapper
├── examples/                   canonical end-to-end runs (committed artifacts)
│   ├── configs/<model>.json    cluster config used by the simulator side
│   ├── <model>/vllm/           vLLM bench artifacts (meta.json, requests.jsonl, timeseries.csv)
│   ├── <model>/outputs/        simulator output (sim.csv, sim.log)
│   ├── <model>/validation/     `bench validate` output (PDFs + summary.txt)
│   ├── run.sh                  rerun the simulator side for any/all examples
│   └── validate.sh             rerun the validation step for any/all examples
└── results/                    output root for ad-hoc runs: bench/results/<run_id>/
```

## Usage

`bench run` — strict replay of an existing dataset

The runner reads a LLMServingSim-format JSONL (the same format
`python -m workloads.generators` produces and `python -m serving --dataset`
consumes). Each request's `input_tok_ids` and `output_toks` are pinned via
`SamplingParams(min_tokens=N, max_tokens=N, ignore_eos=True)`, so the
vLLM run is bit-for-bit comparable to the simulator's view of the same
workload.

```bash
# Inside the vLLM container (scripts/docker-vllm.sh).
./bench/bench.sh
# or invoke the module directly with explicit args:
python -m bench run \
    --model <hf-id-or-path> \
    --dataset workloads/<workload>.jsonl \
    --output-dir bench/results/<run_id> \
    --tensor-parallel-size 1 --data-parallel-size 1 \
    --max-num-seqs 128 --max-num-batched-tokens 2048 \
    --dtype bfloat16 --kv-cache-dtype auto
```

All 15 flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--model` | required | HF id, passed verbatim to `vllm.AsyncLLM` |
| `--dataset` | required | LLMServingSim-format JSONL |
| `--output-dir` | required | run output root |
| `--tensor-parallel-size` | `1` | vLLM `tensor_parallel_size` |
| `--data-parallel-size` | `1` | vLLM `data_parallel_size` |
| `--enable-expert-parallel` | off | vLLM `enable_expert_parallel`, MoE only |
| `--max-num-seqs` | `128` | vLLM `max_num_seqs` |
| `--max-num-batched-tokens` | `2048` | vLLM `max_num_batched_tokens` |
| `--max-model-len` | model max | vLLM `max_model_len` |
| `--dtype` | `bfloat16` | note: not inferred from the model config, unlike `python -m serving` |
| `--kv-cache-dtype` | `auto` | vLLM `kv_cache_dtype` |
| `--seed` | `42` | sampling seed |
| `--tick-seconds` | `1.0` | `timeseries.csv` row spacing; the simulator's `--log-interval` |
| `--num-reqs` | `0` | cap on requests from the dataset, `0` = all |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

There is no `--block-size`: vLLM picks the KV block size itself and records
what it chose in `meta.json` under `kv_cache.block_size`. Pass that value to
the simulator as `--block-size` to line the two up.

`bench validate` — compare a finished bench run against simulator output

Loads the bench artifacts plus the simulator's `sim.csv` / `sim.log`
for the same workload, computes TTFT / TPOT / end-to-end latency on
both sides under matched definitions, and writes plots + a numeric
summary into a subdirectory of the bench run.

```bash
./bench/validate.sh \
    bench/results/<run_id> \
    outputs/<sim-run>/sim.csv \
    outputs/<sim-run>/sim.log \
    [prefix]
```

`validate.sh` sets `--output-subdir`, `--title` and `--log-level` from the
`OUTPUT_SUBDIR` / `TITLE` / `LOG_LEVEL` environment variables. The module
itself takes all seven directly:

| Flag | Default | Notes |
| --- | --- | --- |
| `--bench-dir` | required | a finished `bench run` output directory |
| `--sim-csv` | required | simulator `--output` CSV |
| `--sim-log` | required | simulator stdout, parsed for per-tick running / waiting |
| `--output-subdir` | `validation` | subdirectory under `--bench-dir` |
| `--prefix` | `""` | filename prefix for plots and summary |
| `--title` | `vLLM vs LLMServingSim` | plot title suffix |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

## Output schema (one bench run)

```
bench/results/<run_id>/
  meta.json            run metadata (model, vLLM version, engine kwargs,
                       dataset hash, wall-clock start/end) plus what vLLM
                       *resolved*: kv_cache (num_gpu_blocks, block_size,
                       num_kv_tokens, gpu_memory_utilization), hardware
                       (device name, total memory, CUDA/torch), and
                       resolved_config (the whole VllmConfig, one key per
                       sub-config). num_gpu_blocks is the KV capacity a
                       simulator has to match; the rest of that budget is
                       known up front, so it is the only place vLLM's
                       activation peak shows up.
  requests.jsonl       per-request timing — request_id, input_toks,
                       output_toks, arrival_time, queued_ts, scheduled_ts,
                       first_token_ts, last_token_ts
  timeseries.csv       per-tick aggregates — t, prompt_throughput,
                       gen_throughput, running, waiting, kv_cache_pct
  validation/          (created by `bench validate`)
    <prefix>_throughput.png
    <prefix>_requests.png
    <prefix>_latency.png
    <prefix>_summary.txt
```

## Latency definitions (sim ↔ bench)

Both sides report TTFT, TPOT, and end-to-end latency from the same
reference points so diff% is meaningful:

| Metric | Definition |
| --- | --- |
| `TTFT`     | `first_token_ts - arrival_time` (incl. queueing) |
| `TPOT`     | `(last_token_ts - first_token_ts) / max(1, output_toks - 1)` |
| `Latency`  | `last_token_ts - arrival_time` |

The simulator's `sim.csv` exposes `arrival`, `end_time`, and a per-token
ITL list directly; bench computes the same fields from vLLM's
`RequestStateStats` (`vllm/v1/metrics/stats.py`).

## Canonical examples (`bench/examples/`)

Four end-to-end validation runs are committed under `bench/examples/`,
keyed by `<hardware>/<model>`: a dense single-GPU baseline, a TP=2 dense
run, and a DP+EP MoE run on RTXPRO6000, plus the same dense baseline on
an RTX 4090. Each example bundles its cluster `config.json`, the vLLM
bench artifacts, the simulator output, and the resulting
`bench validate` summary + plots.

| Example | Parallelism | Workload (300 reqs) | TTFT mean | TPOT mean | Latency mean |
| --- | --- | --- | --- | --- | --- |
| `RTX4090/Llama-3.1-8B`                   | TP=1 dense     | `sharegpt-llama-3.1-8b-300-sps10.jsonl`  | +0.6% | +0.2% | +0.5% |
| `RTXPRO6000/Llama-3.1-8B`                | TP=1 dense     | `sharegpt-llama-3.1-8b-300-sps10.jsonl`  | -4.0% | -1.0% | -1.8% |
| `RTXPRO6000/Qwen3-32B`                   | TP=2 dense     | `sharegpt-qwen3-32b-300-sps10.jsonl`     | +1.3% | +0.8% | +1.0% |
| `RTXPRO6000/Qwen3-30B-A3B-Instruct-2507` | DP=2, EP=2 MoE | `sharegpt-qwen3-30b-a3b-300-sps10.jsonl` | -13.6% | -1.7% | -2.2% |

Diff% is `(sim - vLLM) / vLLM × 100`. All runs use `bf16` weights,
`max_num_batched_tokens=2048` and `block_size=16`; the RTXPRO6000 runs
use `max_num_seqs=128` and the RTX 4090 run `max_num_seqs=256`. The
workloads are generated by
`python -m workloads.generators` (ShareGPT, single-turn, vLLM
free-generation mode). Per-percentile breakdowns
(P50 / P90 / P95 / P99) live in each
`bench/examples/<hardware>/<model>/validation/summary.txt`.

Reproducing a canonical example:

```bash
# Inside the simulator container:
./bench/examples/run.sh                       # all four examples
./bench/examples/run.sh RTXPRO6000/Qwen3-30B-A3B-Instruct-2507   # single example

# Then validate against the committed vLLM artifacts:
./bench/examples/validate.sh
./bench/examples/validate.sh RTXPRO6000/Qwen3-30B-A3B-Instruct-2507
```

`run.sh` reads each example's `meta.json` (engine kwargs + dataset path)
and its own `config.json`, so the
simulator runs against the exact same workload and engine configuration
as the original vLLM bench. To regenerate the vLLM side from scratch,
use `bench/bench.sh` (or `python -m bench run`) from inside the vLLM
container.
