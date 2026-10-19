---
sidebar_position: 5
title: Bench CLI
---

# `python -m bench` CLI flags

Complete reference for the vLLM benchmark harness. `bench` has two
subcommands: `run` replays a workload through real vLLM, `validate`
compares a finished run against simulator output.

`run` needs the **vLLM GPU environment** (`scripts/docker-vllm.sh`); see
**[Installation → vLLM environment](/docs/getting-started/installation/vllm)**.
`validate` only reads recorded files and runs on CPU in the simulator
environment. It neither starts a vLLM engine nor requires a GPU.

For the resulting accuracy numbers, see
**[Validation](/docs/validation)**.

The `bench/examples/run.sh` and `validate.sh` wrappers discover stored example
configs under `bench/examples/*/*/config.json`. With no example argument they
include diagnostic configs as well as headline benchmarks. Reproduction uses
`BLOCK_SIZE` if set, otherwise the recorded `kv_cache.block_size`; absent
metadata leaves the simulator's block-size resolution in control.

## `python -m bench run`

Strict replay: the runner reads a LLMServingSim-format JSONL workload
— the same format `python -m workloads.generators` emits and
`python -m serving --dataset` consumes — and pins every request's
`input_tok_ids` and `output_toks` via
`SamplingParams(min_tokens=N, max_tokens=N, ignore_eos=True)`. The
vLLM run therefore processes exactly the prompts the simulator sees,
in the same order.

Replay supports **flat requests with nonempty `input_tok_ids`** only. The
loader skips rows containing `sub_requests`, so agentic-session dependencies
and tool delays are not benchmarked. Use a flat-only dataset for a matched
simulator comparison; mixed files would run different workloads on the two
sides. Count-only requests cannot be tokenized by this runner.

### Required

| Flag | Type | Description |
| --- | --- | --- |
| `--model` | string | HF model id or local model directory, passed verbatim to `vllm.AsyncLLM`. The default loader reads checkpoint weights; `--load-format dummy` instead initializes random weights for diagnostics |
| `--dataset` | path | Flat LLMServingSim-format JSONL requests with nonempty `input_tok_ids`; session rows are skipped. See **[Workloads → JSONL format](/docs/workloads/jsonl-format)** |
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
| `--disable-custom-all-reduce` | flag | off | Disable vLLM custom all-reduce explicitly; other optimized or fused communication paths remain available |

Communication backend selection defaults to vLLM's own choice, **not NCCL-only**.
The runner preserves caller environment values. Separately, it retains
`compilation_config.pass_config.fuse_allreduce_rms=False` and
`fuse_gemm_comms=False`, keeping the benchmark's separate compute/communication
decomposition. Ordinary compute compilation and CUDA graphs stay enabled.
These two compiler switches do not disable model-native fused implementations.
`--disable-custom-all-reduce` disables only vLLM custom all-reduce; FlashInfer,
symmetric-memory or model-native fused paths may still run. Do not use that flag
alone as proof of an NCCL-only recording. The `bench.sh` equivalent is
`DISABLE_CUSTOM_ALL_REDUCE=1` (default `0`).

`meta.json` and `engine_start.json` record the custom-all-reduce argument and
the two compilation overrides under `engine_kwargs`, and resolved
parallel/compiler settings under `resolved_config`. `meta.json` also records
the inherited `VLLM_ALLREDUCE_USE_FLASHINFER`,
`VLLM_ALLREDUCE_USE_SYMM_MEM` and `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` values
under `hardware.all_reduce_environment`. A null value means unset: the installed
vLLM resolves its default. These settings describe configuration, not a measured
per-collective dispatch trace; check runtime backend selection when establishing
a communication baseline. Preserve each reference's recorded backend settings;
a custom-all-reduce-disabled reference must not be relabelled as a measurement
of default backend selection. TP1 does not exercise the TP custom-all-reduce path.

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
| `--kv-cache-memory-bytes` | int | `None` | Explicit per-GPU KV cache budget for capacity-matched controls. `None` retains automatic memory profiling. Verify complete `kv_cache.capacity.per_engine` observations; equal utilization does not guarantee equal blocks, and a DP client scalar can be an aggregate |
| `--seed` | int | `42` | Sampling seed |
| `--load-format` | string | `auto` | vLLM `load_format`. `dummy` skips checkpoint reads and initializes random weights. For dense models, fixed-input/fixed-output-length replay makes this useful for controlled performance diagnostics, not automatically interchangeable end-to-end truth. Verify the resolved backend and KV block count on every boot. MoE routing can depend on real weights and activations, so dummy weights need a separate routing control. The load format is recorded in `meta.json` |
| `--skip-tokenizer-init` | flag | off | Boot without loading a tokenizer; replay supplies token IDs directly. For a local config-only diagnostic, create a separate model directory containing `config.json`, pass that directory as `--model`, and use `--load-format dummy`. The benchmark does not materialize the repository's named JSON files into that layout for you. Real-weight runs still require checkpoint weights |
| `--enforce-eager` | flag | off | Run vLLM eager, with `torch.compile` and CUDA graphs off. Leave it off when comparing with the default compiled target. An eager reference helps isolate execution-mode effects because ordinary layerwise acquisition requires module boundaries, while native MoE component tables can include graph timings. Record and match the target mode; graph padding alone does not reproduce graph kernel fusion. Saved in `meta.json` |
| `--record-gate-stats` | flag | off | Observe distinct-expert counts between explicit workload start/end markers and write `gate_stats.json` for `--expert-routing-policy CUSTOM`. Concentrated routing is retained; counts do not identify warmup. Requires the `VLLM_MOE_ACTIVATED_LOG` source patch from `scripts/docker-vllm.sh` and **`--enforce-eager`**, because `.unique()` is not graph-capturable. The curve describes the recorded weights and inputs, not all deployments. Instrumented latency is marked diagnostic and rejected by `bench validate`. Dense models produce no curve and warn |
| `--resolve-only` | flag | off | Boot the engine, write resolved settings and KV capacity metadata, then exit without replay. Match complete per-engine observations to simulator instances; the raw DP client count need not describe one engine. Dummy loading avoids checkpoint reads but does not certify a real-weight capacity. No `requests.jsonl` or `timeseries.csv` is written |

:::note[Defaults differ from `python -m serving`]
`bench run` defaults `--dtype` to `bfloat16` outright, where the
simulator resolves it from the model config's `torch_dtype`. `bench`
also has no `--block-size`: vLLM picks the KV block size itself, and
values it settled on are recorded in `meta.json`. For new DP runs use the
corresponding complete `kv_cache.capacity.per_engine` entry, including its
block size. Historical single-engine references retain `kv_cache.block_size`.
Pass the matched value to the simulator as `--block-size` or an instance override.
:::

The benchmark CLI does not expose every vLLM engine option. In particular,
`gpu_memory_utilization` inherits the installed vLLM default; there is no
`--gpu-memory-utilization` flag. The committed RTXPRO6000 references record
0.92, so invoking the current wrapper is not an exact reconstruction of their
settings. Verify effective settings and observed KV capacity for every new
reference, including when using an explicit `--kv-cache-memory-bytes` budget.

### Workload and output

Normal serving runs save `engine_start.json` before submitting requests,
including workload identity, resolved settings and KV capacity. This is not a
completion marker. Use a fresh output directory: an existing startup snapshot
is not overwritten. Final results require `meta.json`, `requests.jsonl` and
`timeseries.csv`; resolved settings are captured while the engine is still alive.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--num-reqs` | int | `0` | Cap on requests taken from the dataset. `0` = replay all |
| `--tick-seconds` | float | `1.0` | Stat-logger downsample interval, i.e. row spacing in `timeseries.csv`. Matches the simulator's `--log-interval` |
| `--log-level` | choice | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

### Output

```
<output-dir>/
  engine_start.json  normal replay startup snapshot, written before submission;
                     not a completion marker
  meta.json          run metadata plus what vLLM *resolved*: kv_cache
                     (num_gpu_blocks, block_size, num_kv_tokens,
                     gpu_memory_utilization, qualified capacity), hardware (device name,
                     total memory, UUID, CPU affinity, allowed NUMA nodes,
                     CUDA / torch versions), and
                     resolved_config -- the whole VllmConfig, one key
                     per sub-config; execution -- effective worker graph
                     policy after capture, or unknown with a reason
  requests.jsonl     per request: request_id, input_toks, output_toks,
                     arrival_time, queued_ts, scheduled_ts,
                     first_token_ts, last_token_ts
  timeseries.csv     per tick: t, prompt_throughput,
                     prompt_compute_throughput, prompt_cached_throughput,
                     gen_throughput, running, waiting, preempted, kv_cache_pct
  gate_stats.json    --record-gate-stats only: the measured curve
                     (tokens -> mean distinct experts, samples) plus
                     model / num_experts / num_experts_per_tok, so a
                     reader can refuse a curve from another checkpoint
  moe_activated.jsonl  --record-gate-stats only: phase markers and raw per-call log
                     gate_stats.json was reduced from
```

`preempted` counts preemption events during the tick, summed over DP engines.
The prompt compute/cache throughput columns distinguish computed prompt tokens
from prefix-cache hits. Older recordings can omit these three columns; inspect
the CSV header before using them.

### Serving graph metadata

After startup and before request submission, a read-only worker RPC observes
the effective graph mode and complete FULL/PIECEWISE capture grids, along with
model, device, precision and deployment scope. All TP/PP/DP ranks must agree;
the DP client is queried per engine to retain every response. Unsupported or
incomplete observations remain `unknown` without failing the benchmark.
Frontend `resolved_config.compilation_config.cudagraph_capture_sizes` also keeps
its full list, but a requested list is not evidence of completed worker capture.
Use [profiler import-execution](/docs/profiler/output-bundle#serving_execution)
to explicitly attach a compatible observation to a profile bundle. It changes
neither recorded timings nor the profiler's acquisition mode.

### KV capacity metadata

`kv_cache.capacity` schema 1 qualifies the preserved raw client fields. On the
reviewed vLLM 0.31 startup path, `num_gpu_blocks` sums the connected DP engines
and `block_size` is the last engine's READY value. The legacy `num_kv_tokens`
product is arithmetic, not a general aggregate token capacity.

When `capacity.status` is `complete`, `capacity.per_engine` contains directly
observed DP ranks, scheduler block counts, block sizes, native token capacities
and parallelism. Match each simulator instance to its rank; do not divide a
client total by DP or assume equal hardware guarantees equal budgets. These are
scheduler-pool counts, not tensor traffic or physical per-GPU memory measurements.
The simulator does not model the activation peak, so equal `mem_util` values do
not ensure equal capacity; see **[KV cache and memory](/docs/simulator/scheduling/kv-cache-and-memory)**.

The observer retains bounded native READY payloads only while `AsyncLLM`
constructs, restoring its callback before decoding or request submission. No
worker RPC or request instrumentation is added. Unsupported callback/response
APIs, replaced callbacks, client/topology mismatches and incomplete observations
produce `status: unknown`, `per_engine: null` and a reason. The observer records
the runtime version, callback signature and response fields; exact version equality
is not an admission requirement. Resolve-only
uses the same contract. Startup facts do not track later elastic changes;
historical runs keep their original fields and recorded-version semantics.

The dataset is never modified — generation lives in
`workloads/generators`.

## `python -m bench validate`

Initialization-only, explicitly aborted and known paced, synchronized or failed
diagnostic runs are rejected as end-to-end ground truth. Legacy directories
without metadata remain readable when their request and timeseries files exist.

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

Both sides compute the same three quantities from aligned reference points.
Here `arrival` is the dataset arrival converted from epoch seconds to the
engine's monotonic clock domain, not raw `arrival_time` or `queued_ts`.
Use the validator rather than subtracting mixed-clock timestamps manually;
see [metric definitions](/docs/validation#metric-definitions).

| Metric | Definition |
| --- | --- |
| TTFT | `first_token_ts - arrival` (queueing included) |
| TPOT | `(last_token_ts - first_token_ts) / (output_toks - 1)`, only when `output_toks > 1` |
| Latency | `last_token_ts - arrival` |

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

Two shell wrappers set the flags for you. Run `bench.sh` in the vLLM GPU
environment and `validate.sh` in the CPU simulator environment. Both can be
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
| `DISABLE_CUSTOM_ALL_REDUCE` | `--disable-custom-all-reduce` when `1` | `0` |
| `MAX_NUM_SEQS` | `--max-num-seqs` | `128` |
| `MAX_NUM_BATCHED_TOKENS` | `--max-num-batched-tokens` | `2048` |
| `MAX_MODEL_LEN` | `--max-model-len`, omitted when blank | blank |
| `DTYPE` | `--dtype` | `bfloat16` |
| `KV_CACHE_DTYPE` | `--kv-cache-dtype` | `auto` |
| `SEED` | `--seed` | `42` |
| `LOAD_FORMAT` | `--load-format` | `auto` |
| `RESOLVE_ONLY` | `--resolve-only` when `1` | `0` |
| `SKIP_TOKENIZER_INIT` | `--skip-tokenizer-init` when `1` | `0` |
| `ENFORCE_EAGER` | `--enforce-eager` when `1` | `0` |
| `TICK_SECONDS` | `--tick-seconds` | `1.0` |
| `NUM_REQS` | `--num-reqs` | `0` |
| `LOG_LEVEL` | `--log-level` | `INFO` |

Note `TP=2` — the wrapper's default is not vLLM's `1`.
`--kv-cache-memory-bytes` and `--record-gate-stats` are module-only options:
invoke `python -m bench run` directly to use them. This wrapper does not
forward arbitrary extra positional arguments.

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

`bench/examples/` holds four headline end-to-end runs plus stored diagnostics, keyed
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
