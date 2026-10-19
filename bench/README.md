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
│   ├── kv_cache_metadata.py     qualified per-engine startup READY capacities
│   ├── execution_metadata.py   completed worker graph observations
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
The shared launcher targets vLLM 0.31.0 / CUDA 13. Committed reference runs
retain the versions in their metadata; changing the launcher does not refresh them.
The committed examples validate their stored profile bundles, not automatically
a new acquisition made with the latest profiler. Check both profile acquisition
metadata and benchmark engine settings before claiming equivalent coverage.
Profile CSVs contain coordinates and timings; per-category method and runtime
observations are recorded in profile metadata. A version label alone neither
proves equivalence nor vetoes execution: compare the actual backend, precision,
geometry and workload. Bounded Kimi/GLM profiling and duration replay do not
establish native serving capacity, TP2 support or benchmark accuracy.

### Record a workload: `bench run`

The runner reads a LLMServingSim-format JSONL (the same format
`python -m workloads.generators` produces and `python -m serving --dataset`
consumes). Each request's `input_tok_ids` and `output_toks` are pinned via
`SamplingParams(min_tokens=N, max_tokens=N, ignore_eos=True)`, so the
vLLM run uses the same prompt token IDs and requested output lengths as the
simulator. Generated token content and execution timing are not bit-identical.

Benchmark replay supports **flat request rows with nonempty `input_tok_ids`**
only. Rows containing `sub_requests` are skipped; session dependencies and
tool delays are not replayed. Use a flat-only dataset when comparing against
the simulator, since a mixed file would otherwise represent different workloads.
The benchmark does not tokenize count-only requests.

Communication backend selection follows vLLM's defaults and the caller's
environment. The runner does not force NCCL or overwrite all-reduce environment
variables. It retains `fuse_allreduce_rms=False` and `fuse_gemm_comms=False` to
keep the benchmark's separate compute/communication decomposition; ordinary
compute compilation and CUDA graphs remain enabled. These compiler settings
do not disable model-native fused implementations. `--disable-custom-all-reduce` is an
explicit opt-in (off by default); it disables that implementation only, not
other optimized or fused communication paths. The choice is recorded in
`engine_kwargs` together with the two compilation overrides, environment values
in `hardware.all_reduce_environment`, and
resolved parallel/compiler settings in `resolved_config`. An unset environment
value means vLLM chooses its default, not that the path is disabled.
Historical artifacts retain their original settings; a new default run is not
automatically equivalent to a committed NCCL-only reference.

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

The RTXPRO6000 headline examples pair refreshed vLLM 0.31.0 profiles with
recorded references and explicit target graph settings. Qwen3-32B TP2 records
custom all-reduce disabled; Llama TP1 and Qwen3-30B TP1/DP2/EP2 retain the
default setting and do not exercise the TP custom-all-reduce path. The Qwen
examples inherit the hardware bundle's primitive-calibrated per-operation
bandwidths and common latency. Keep each reference's recorded communication
settings and interconnect calibration matched.

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
| `--disable-custom-all-reduce` | off | Explicitly disable custom all-reduce; does not enforce NCCL-only |
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
what it chose in `meta.json`. For a new DP run, use the matched engine's block
size under complete `kv_cache.capacity.per_engine` observations. Historical
single-engine recordings retain `kv_cache.block_size`; the raw DP client value
can describe only the last engine. Pass the matched value to the simulator as
`--block-size` or a per-instance override.

`bench.sh` is an editable run template, not a passthrough for every CLI flag.
`DISABLE_CUSTOM_ALL_REDUCE=1` enables the same explicit opt-in; its default is `0`.
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
New runs also retain `execution`: a read-only worker startup observation of the
effective runner, graph mode and completed capture sizes, qualified by model,
precision and deployment limits. Unsupported or incomplete observations remain
`unknown`; frontend configuration alone is not evidence of completed captures.
The observer runs before workload submission and changes no sampling or graph
settings. Capture lists are not truncated by the generic metadata serializer.
Use `profiler import-execution` to copy a compatible serving observation into a
profile bundle; this does not change its measured CSVs.
For new runs, match each simulator instance to its rank in
`kv_cache.capacity.per_engine` only when `capacity.status` is `complete`.
The reviewed vLLM 0.31 client sums connected DP engines in the raw
`kv_cache.num_gpu_blocks` and retains the last READY `block_size`; their
`num_kv_tokens` product is not a general aggregate token capacity. Never divide
the total by DP to invent per-engine budgets. Equal memory-utilization settings
alone do not account for vLLM's activation budget.

The bounded observer retains native READY payloads during engine construction
and restores the callback before decoding or request submission. It adds no
request instrumentation. Unsupported callback/response APIs, replaced callbacks,
client/topology mismatches and incomplete observations record `status: unknown`,
`per_engine: null` and a reason. Observed versions and callback/response ABI facts
are recorded; compatibility follows the actual callback and response interfaces.
These are startup scheduler-pool facts, not physical GPU memory or later elastic
changes. Historical artifacts keep their original fields and version semantics.

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
| `RTXPRO6000/Llama-3.1-8B` | TP=1 dense | `sharegpt-llama-3.1-8b-300-sps10.jsonl` | +0.7% | +0.3% | +0.4% |
| `RTXPRO6000/Qwen3-32B` | TP=2 dense | `sharegpt-qwen3-32b-300-sps10.jsonl` | -0.2% | +0.2% | +0.1% |
| `RTXPRO6000/Qwen3-30B-A3B-Instruct-2507` | DP=2, EP=2 MoE | `sharegpt-qwen3-30b-a3b-300-sps10.jsonl` | -0.2% | -0.3% | -0.3% |

Check all fifteen statistics, not just these means. The largest absolute
errors for the RTXPRO6000 Llama, Qwen3-32B and Qwen3-30B references are
0.682%, 0.456% and 1.197%, respectively. These results cover the recorded
workloads and deployment scopes, not every acquired TP or MoE deployment.
Additional stored diagnostics
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
omits `--block-size` and lets the simulator resolve its default. Published
examples use repository-relative dataset paths and equivalent canonical model
identifiers in place of machine-local paths. The instance config supplies
the target graph policy and memory settings; the wrapper does not reconstruct
every engine option from metadata. Llama and Qwen3-32B use `mem_util: 0.92`;
Qwen3-30B matches 38,256 recorded KV blocks per rank with
`mem_util: 0.9043190002441406`, while its benchmark retains the recorded 0.92.
To record a new vLLM reference, use `bench/bench.sh` or `python -m bench run`
inside the vLLM environment. These commands do not reconstruct a committed
recording's settings. The CLI exposes only a subset of vLLM options and has
no `--gpu-memory-utilization`; the RTXPRO6000 references record 0.92, while
a new run inherits the installed vLLM default. Match the effective engine
settings, communication policy and observed KV capacity before treating a new
recording as equivalent. `--kv-cache-memory-bytes` can set an explicit budget,
but its resolved capacity must still be verified.

Apart from portable dataset and local model path identifiers, adopted outputs
retain their emitted values and rows, including measurements, engine settings,
hashes, validation results and logs.
