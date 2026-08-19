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

Three end-to-end validation runs are committed under `bench/examples/`,
covering a dense single-GPU baseline, a TP=2 dense run, and a DP+EP MoE
run. Each example bundles the vLLM bench artifacts, the simulator
output, and the resulting `bench validate` summary + plots.

| Example | Parallelism | Workload (300 reqs) | TTFT mean | TPOT mean | Latency mean |
| --- | --- | --- | --- | --- | --- |
| `Llama-3.1-8B`                | TP=1 dense              | `sharegpt-llama-3.1-8b-300-sps10.jsonl`     | -2.8% | -0.3% | -1.0% |
| `Qwen3-32B`                   | TP=2 dense              | `sharegpt-qwen3-32b-300-sps10.jsonl`        | -0.7% | -0.3% | -0.4% |
| `Qwen3-30B-A3B-Instruct-2507` | DP=2, EP=2 MoE          | `sharegpt-qwen3-30b-a3b-300-sps10.jsonl`    | -2.9% | +0.6% | +0.4% |

Diff% is `(sim - vLLM) / vLLM × 100`. All three runs are on RTXPRO6000
with `bf16` weights, `max_num_seqs=128`, `max_num_batched_tokens=2048`,
`block_size=16`, and the workloads are generated by
`python -m workloads.generators` (ShareGPT, single-turn, vLLM
free-generation mode). Per-percentile breakdowns
(P50 / P90 / P95 / P99) live in each
`bench/examples/<model>/validation/summary.txt`.

Reproducing a canonical example:

```bash
# Inside the simulator container:
./bench/examples/run.sh                       # all three examples
./bench/examples/run.sh Qwen3-30B-A3B-Instruct-2507   # single example

# Then validate against the committed vLLM artifacts:
./bench/examples/validate.sh
./bench/examples/validate.sh Qwen3-30B-A3B-Instruct-2507
```

`run.sh` reads each example's `meta.json` (engine kwargs + dataset path)
and the matching cluster config under `bench/examples/configs/`, so the
simulator runs against the exact same workload and engine configuration
as the original vLLM bench. To regenerate the vLLM side from scratch,
use `bench/bench.sh` (or `python -m bench run`) from inside the vLLM
container.
