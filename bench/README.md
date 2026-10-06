# bench

End-to-end vLLM benchmark + simulator validation. Runs a real vLLM
serving workload, captures per-request timing and per-tick scheduler
state, and compares the result against the simulator's output for the
same dataset.

## Layout

```text
bench/                         Python package: python -m bench
├── __init__.py                 package marker + module map
├── __main__.py                 CLI dispatch (run / validate)
├── core/                       internals
│   ├── runner.py               AsyncLLM driver, captures RequestStateStats
│   ├── recorder.py             writes meta.json / requests.jsonl / timeseries.csv
│   ├── stat_logger.py          custom vLLM StatLoggerBase that fills timeseries
│   ├── validate.py             bench-vs-sim comparison entry point
│   ├── plots.py                throughput / running-waiting / latency-CDF helpers
│   └── logger.py               Rich-based logger + stdio capture
├── bench.sh                    python -m bench run wrapper
├── validate.sh                 python -m bench validate wrapper
├── examples/                   canonical end-to-end runs (committed artifacts)
│   ├── <hw>/<model>/           one recorded configuration and workload
│   │   ├── config.json        simulator cluster config
│   │   ├── vllm/              recorded vLLM metadata, requests and timeseries
│   │   ├── outputs/           simulator output (sim.csv, sim.log)
│   │   └── validation/        comparison plots and summary.txt
│   ├── run.sh                  rerun the simulator side for any/all examples
│   └── validate.sh             rerun the validation step for any/all examples
└── results/                    output root for ad-hoc runs: bench/results/<run_id>/
```

## Usage

`bench run` needs the vLLM GPU environment. `bench validate` only reads recorded
files and runs on CPU in the simulator environment; it does not boot vLLM.
The committed examples validate their stored profile bundles, not automatically
a new acquisition made with the latest profiler. Check both profile acquisition
metadata and benchmark engine settings before claiming equivalent coverage.

### Record a workload: `bench run`

The runner reads a LLMServingSim-format JSONL (the same format
`python -m workloads.generators` produces and `python -m serving --dataset`
consumes). Each request's `input_tok_ids` and `output_toks` are pinned via
`SamplingParams(min_tokens=N, max_tokens=N, ignore_eos=True)`, so the
vLLM run uses the same prompt token IDs and requested output lengths as the
simulator. Generated token content and execution timing are not bit-identical.

The communication baseline is NCCL: the runner explicitly sets
`disable_custom_all_reduce=True` and records it in `engine_kwargs`, alongside
the resolved parallel configuration. It also disables the independent Torch
symmetric-memory and FlashInfer all-reduce paths before loading vLLM; the
effective overrides are saved in `meta.json` under
`hardware.all_reduce_environment`. Non-NCCL all-reduce/RMS and asynchronous
GEMM/communication fusions are disabled through recorded compilation overrides.
Ordinary compute compilation and CUDA graphs remain enabled by default.
Historical artifacts retain their original
settings; inspect their metadata before comparing them.

`--record-gate-stats --enforce-eager` observes routing between explicit workload
start/end markers. A concentrated gate is valid data, not a warmup signature.
The resulting curve describes the recorded weights and workload; the observer's
latencies are marked diagnostic and rejected by `bench validate`. Unmarked or
incomplete raw logs need a new recording, while existing `gate_stats.json`
curves remain readable. Normal runs without this flag are unchanged.

Simulator examples include local CUDA graph padding before DP synchronization.
Use per-instance `cudagraph` settings to describe the effective target worker
mode and capture grid; the profiler's eager configuration is not that target.
A simulator-side refresh preserves the recorded vLLM truth.

The RTXPRO6000 Qwen3-32B TP2 and Qwen3-30B DP2/EP2 examples use NCCL-only
references under the calibrated host interconnect. Both omit link overrides
and inherit the hardware bundle's per-operation bandwidths and common latency;
the MoE example also includes native component measurements. Keep recorded
references and their interconnect calibration matched when refreshing examples.

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

Run options (also available through `python -m bench run --help`):

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
| `--kv-cache-memory-bytes` | unset | Positive per-GPU KV cache budget; otherwise vLLM profiles available memory automatically |
| `--seed` | `42` | sampling seed |
| `--load-format` | `auto` | vLLM weight loader; `dummy` is for controlled diagnostics, not automatically equivalent ground truth |
| `--skip-tokenizer-init` | off | Replay token IDs without loading a tokenizer |
| `--enforce-eager` | off | Disable compilation and CUDA graphs; match this mode to the simulation |
| `--record-gate-stats` | off | Diagnostic routing observation; requires `--enforce-eager`, and its timings are rejected by validation |
| `--resolve-only` | off | Boot and save resolved settings, then exit without replay or request/timeseries files |
| `--tick-seconds` | `1.0` | `timeseries.csv` row spacing; the simulator's `--log-interval` |
| `--num-reqs` | `0` | cap on requests from the dataset, `0` = all |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

There is no `--block-size`: vLLM picks the KV block size itself and records
what it chose in `meta.json` under `kv_cache.block_size`. Pass that value to
the simulator as `--block-size` to line the two up.

`bench.sh` is an editable run template, not a passthrough for every CLI flag.
Use the module directly for `--kv-cache-memory-bytes` or `--record-gate-stats`.
Its default TP is `2`, while the module defaults to `1`; see the
[wrapper variable reference](../docs/docs/reference/bench-cli.md#shell-wrappers).

### Compare recorded results: `bench validate`

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

## Output schema

One completed bench run and its optional validation output:

```text
bench/results/<run_id>/
├── engine_start.json   startup snapshot; not a completed benchmark by itself
├── meta.json           run identity, engine settings, hardware and KV capacity
├── requests.jsonl      token counts, arrival and per-request engine timestamps
├── timeseries.csv      throughput, running/waiting requests and KV utilization
└── validation/         created by bench validate (default filenames below)
    ├── throughput.png
    ├── requests.png
    ├── latency.png
    └── summary.txt
```

`--prefix NAME` prepends `NAME_` to each validation filename. `meta.json`
includes the dataset hash, vLLM version, requested engine kwargs, wall-clock
start/end, resolved `VllmConfig`, device placement and `kv_cache` fields.
Match `kv_cache.num_gpu_blocks` and `block_size` to the simulator; equal
memory-utilization settings alone do not account for vLLM's activation budget.

## Latency definitions (sim ↔ bench)

Resolved engine settings are captured before shutdown. Each serving run uses a
fresh output directory: an existing `engine_start.json` is not overwritten.
Validation rejects initialization-only, explicitly aborted and known paced or
synchronized diagnostic runs. Legacy request/timeseries inputs remain readable.
Equal memory-utilization settings do not guarantee equal KV block counts; check
`meta.json` even when using an explicit cache budget. Dummy weights are useful
for controlled diagnostics, not automatically equivalent end-to-end truth.

Both sides report TTFT, TPOT, and end-to-end latency from the same
reference points so diff% is meaningful:

`arrival` below is the dataset's wall-clock arrival shifted into the engine's
monotonic clock domain by the validator. Do not subtract raw `arrival_time`
from an engine timestamp, or substitute `queued_ts`: the latter drops the
frontend-to-engine delay from TTFT. See the [metric definitions](../docs/docs/validation.md#metric-definitions).

| Metric | Definition |
| --- | --- |
| `TTFT`     | `first_token_ts - arrival` (incl. queueing) |
| `TPOT`     | `(last_token_ts - first_token_ts) / (output_toks - 1)`, only when `output_toks > 1` |
| `Latency`  | `last_token_ts - arrival` |

The simulator's `sim.csv` exposes `arrival`, `end_time`, and a per-token
ITL list directly; bench computes the same fields from vLLM's
`RequestStateStats` (`vllm/v1/metrics/stats.py`).

Do not confuse these request metrics or `Total clocks (ns)` with the simulator
log's `Total simulation time`: the latter measures host execution, not modeled
serving latency. Compare it with the same timing boundary and CPU environment;
the reproduction wrapper also incurs startup and teardown time.

Non-speculative head rows count real requests, not CUDA graph padding; idle
DP forwards omit logits and sampling. A scheduling-delay change need not
change the latency summary, so inspect per-request outputs too. Regenerate
the example's validation artifacts whenever its simulation output changes.

## Canonical examples (`bench/examples/`)

Four headline end-to-end validation runs are committed under `bench/examples/`,
keyed by `<hardware>/<model>`: a dense single-GPU baseline, a TP=2 dense
run, and a DP+EP MoE run on RTXPRO6000, plus the same dense baseline on
an RTX 4090. Each example bundles its cluster `config.json`, the vLLM
bench artifacts, the simulator output, and the resulting
`bench validate` summary + plots.

| Example | Parallelism | Workload (300 reqs) | TTFT mean | TPOT mean | Latency mean |
| --- | --- | --- | --- | --- | --- |
| `RTX4090/Llama-3.1-8B` | TP=1 dense | `sharegpt-llama-3.1-8b-300-sps10.jsonl` | +0.5% | +0.4% | +0.5% |
| `RTXPRO6000/Llama-3.1-8B` | TP=1 dense | `sharegpt-llama-3.1-8b-300-sps10.jsonl` | +1.0% | +0.1% | +0.3% |
| `RTXPRO6000/Qwen3-32B` | TP=2 dense | `sharegpt-qwen3-32b-300-sps10.jsonl` | -1.0% | -0.7% | -0.8% |
| `RTXPRO6000/Qwen3-30B-A3B-Instruct-2507` | DP=2, EP=2 MoE | `sharegpt-qwen3-30b-a3b-300-sps10.jsonl` | -1.1% | +0.2% | +0.1% |

The dense RTXPRO6000 bundles include broader measured skew geometry and
reference-aligned calibration. Check all fifteen statistics, not just these
means: all headline examples are within 5% on every statistic, with the MoE
example's largest displayed absolute error at 2.6%. Additional stored diagnostics
include a reduced DeepSeek model with larger errors. See the public validation page
for those results and the limits of these workload-specific comparisons.

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
./bench/examples/run.sh                       # all discovered example configs
./bench/examples/run.sh RTXPRO6000/Qwen3-30B-A3B-Instruct-2507   # single example

# Then validate against the committed vLLM artifacts:
./bench/examples/validate.sh
./bench/examples/validate.sh RTXPRO6000/Qwen3-30B-A3B-Instruct-2507
```

`run.sh` reads each example's `meta.json` (engine kwargs + dataset path)
and its own `config.json`. Both wrappers discover `*/*/config.json` rather
than a fixed model list; additional stored diagnostics are included too.
`BLOCK_SIZE` overrides the recorded `kv_cache.block_size`; otherwise the
recorded value is passed to the simulator. If it is missing, the wrapper
omits `--block-size` and lets the simulator resolve its default. Thus the
simulator runs against the exact same workload and engine configuration
as the original vLLM bench. To regenerate the vLLM side from scratch,
use `bench/bench.sh` (or `python -m bench run`) from inside the vLLM
container.
