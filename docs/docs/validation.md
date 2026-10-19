---
title: Validation
sidebar_position: 3
description: How LLMServingSim's output compares against real vLLM
---

# Validation

LLMServingSim is validated end-to-end against real vLLM on the
**bundled `(hardware, model)` combos**. The numbers below come from
running a 300-request ShareGPT replay through both real vLLM and the
simulator, then comparing the per-request and per-tick metrics with
`python -m bench validate`. The tables reproduce values from the committed
`bench/examples/<hardware>/<model>/validation/summary.txt`; the example wrappers
regenerate the comparisons on CPU without starting vLLM.

:::caution[Validation belongs to the recorded deployment]
The RTXPRO6000 Llama and both Qwen bundles contain refreshed vLLM 0.31.0
TP1 and TP2 acquisition data. Accuracy here covers Llama TP1, Qwen3-32B TP2
and Qwen3-30B TP1/DP2/EP2. Generated metadata retains measured values and
settings; only environment-dependent path identifiers are made portable.
Other acquired TP and MoE deployment scopes
require their own end-to-end validation. The RTX 4090 example and retained
historical MoE fallback data keep their original provenance.
:::

**The largest absolute errors across all fifteen statistics for the RTXPRO6000
Llama, Qwen3-32B and Qwen3-30B examples are 0.682%, 0.456% and 1.197%.**

The additional stored DeepSeek diagnostic has larger errors and is reported
separately below; it is not included in the four headline configurations.

> **Want to validate your own change?** See
> **[For Contributors → Validating your changes](/docs/contributor/validating-changes)**
> for the regression workflow.

## Setup

| Knob | Value |
| --- | --- |
| **Workload** | 300 ShareGPT-derived requests, ~10 sps Poisson arrivals |
| **Hardware** | RTXPRO6000 and RTX 4090, single node (profile bundles in `profiler/perf/<hardware>/`) |
| **vLLM version** | `v0.31.0` for the three RTXPRO6000 headline references; `v0.19.0` for the retained RTX 4090 reference |
| **Block size** | 16 for the headline examples; 64 for the additional DeepSeek diagnostic |
| **Engine flags** | Recorded in each example's `vllm/meta.json`; match its effective settings rather than assuming current CLI defaults |
| **Cluster configs** | `bench/examples/<hardware>/<model>/config.json` |
| **Interconnect** | Qwen3-32B and Qwen3-30B inherit primitive-calibrated per-operation BW and common latency from `profiler/perf/RTXPRO6000/hardware.yaml`; recorded backend settings remain in each reference |
| **Target graphs** | Explicit V2 `FULL_AND_PIECEWISE` policy and capture grid in the RTXPRO6000 example configs |
| **KV capacity** | Llama and Qwen3-32B use `mem_util: 0.92`; Qwen3-30B and RTX 4090 match recorded block counts as described below |

The strict-replay path fixes each request's input token IDs and requested output
length, and schedules it at the dataset's arrival offset. Generated output token
content and per-request timing are measured outcomes, not pinned values. These
comparisons use flat request workloads; the benchmark skips agentic-session rows.

:::caution[Match capacity from recorded engine observations]
`npu_mem.mem_util` sizes the simulated KV cache. The simulator does not model
vLLM's activation peak or CUDA context, so equal utilization fractions need
not produce equal block counts. The RTX 4090 example sets `mem_util` to
reproduce the capacity in its historical single-engine `meta.json`:

```json
"kv_cache": { "num_gpu_blocks": 2588, "block_size": 16, "num_kv_tokens": 41408 }
```

The example uses `mem_util: 0.833919` to reproduce those 2,588 blocks.
This matches a recorded capacity, not a latency target; its current errors
are listed in the result tables below.

That JSON is the historical single-engine reference. For new DP benchmarks,
use complete `kv_cache.capacity.per_engine` observations and match each
instance to its native rank. vLLM 0.31's raw client block count sums connected
engines; do not use it as one pool or divide it to infer unobserved capacities.
See the [capacity metadata contract](/docs/reference/bench-cli#kv-capacity-metadata).

Qwen3-30B records 38,256 blocks for each of its two DP engines. Its simulator
instances use `mem_util: 0.9043190002441406` to match those observations;
the benchmark metadata retains its actual utilization of 0.92. Llama and
Qwen3-32B use 0.92 without claiming an observed per-engine capacity match.

Check preemption in the truth's `timeseries.csv` and the simulator log when
assessing memory pressure. Occupancy percentages have different definitions:
the simulator's heartbeat includes cached-but-free blocks, while vLLM's
`kv_cache_pct` counts pinned blocks. Compare block capacities and scheduling
behavior before interpreting a difference in those percentages.
:::

## Metric definitions

Use `python -m bench validate` for the comparison. Its arrival timestamp is
converted from wall-clock epoch seconds to the engine's monotonic domain using
the run's minimum `queued_ts - arrival_time` offset. Raw timestamps from those
two domains cannot be subtracted directly. Using `queued_ts` as the arrival
instead would omit frontend-to-engine waiting from TTFT; legacy records use the
validator's documented fallback when arrival information is missing.

| Metric | Definition in the aligned clock domain |
| --- | --- |
| TTFT | `first_token_ts - arrival` |
| TPOT | `(last_token_ts - first_token_ts) / (output_toks - 1)`, for requests with more than one output token |
| Latency | `last_token_ts - arrival` |

Report mean, median, P90, P95 and P99 for each metric. Benchmark accuracy is
separate from simulator execution speed: `Total clocks (ns)` is the modeled
makespan, while the log's `Total simulation time` is elapsed host time inside
the simulator. A wrapper's wall time also includes startup and teardown.
Compare wall times under the same CPU placement, dependencies, storage and
logging settings; identical simulated results need not take identical host time.

## Headline numbers

Mean error vs. real vLLM, per metric, on the four configurations summarized here:

| Hardware | Model | Parallelism | TTFT mean | TPOT mean | Latency mean | worst of 15 |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 4090 | Llama-3.1-8B | TP=1 dense | +0.5% | +0.4% | +0.5% | +1.2% |
| RTXPRO6000 | Llama-3.1-8B | TP=1 dense | +0.7% | +0.3% | +0.4% | +0.7% |
| RTXPRO6000 | Qwen3-32B | TP=2 dense | -0.2% | +0.2% | +0.1% | +0.5% |
| RTXPRO6000 | Qwen3-30B-A3B-Instruct-2507 | DP=2 x EP=2 MoE | -0.2% | -0.3% | -0.3% | +1.2% |

These workloads were used during development, including skew-axis selection.
Their results are useful regression checks, not an independent estimate of
generalization to other models, hardware or request distributions.

The last column is the largest absolute error across all fifteen metrics
(TTFT / TPOT / latency x mean / median / P90 / P95 / P99), which is the honest
summary of a run: a mean can be small because two errors cancelled. Small
errors on these references do not cover arbitrary deployments or eliminate
execution-to-execution TTFT variation.

Queueing can amplify small step-cost differences into larger TTFT errors.
The signs of aggregate errors do not identify a unique kernel-level cause or
establish generalization. Target CUDA graph padding applies before DP
synchronization; these examples use measured kernel tables without a fitted
per-step graph-time correction.

Per-percentile numbers are in the same `summary.txt` files under
[`bench/examples/`](https://github.com/casys-kaist/LLMServingSim/tree/main/bench/examples).

:::caution[One recorded run does not establish generalization]
The simulator is deterministic, while real DP batch pairing and queueing can
vary between executions. A stored truth is a reproducible reference, not a
guarantee of the same error on another run. Report all fifteen statistics
against each fixed repeat when repeated references are available; TTFT and
failing percentiles remain part of the accuracy assessment.
:::

## Per-configuration results

### RTX 4090 — Llama-3.1-8B (TP=1 dense)

Throughput timeline, vLLM (blue) vs. simulator (orange, dashed):

![RTX 4090 Llama-3.1-8B throughput](/img/validation/rtx4090-llama-3.1-8b-throughput.png)

| Metric | vLLM | Sim | Diff |
| --- | --- | --- | --- |
| TTFT mean | 65.49 s | 65.81 s | **+0.5%** |
| TTFT median | 60.13 s | 60.46 s | +0.6% |
| TTFT P99 | 137.36 s | 138.15 s | +0.6% |
| TPOT mean | 32.4 ms | 32.6 ms | **+0.4%** |
| TPOT P99 | 56.0 ms | 56.7 ms | +1.2% |
| Latency mean | 86.61 s | 87.02 s | **+0.5%** |
| Latency P99 | 153.63 s | 154.52 s | +0.6% |

The 24 GB configuration saturates its KV cache. Its memory utilization
setting matches the recorded KV block count, so memory-pressure behavior
can be compared at the same capacity. Compute latencies are supplied by
the profile bundle rather than fitted to the end-to-end results.

This is a retained vLLM `v0.19.0` reference, not a v0.28 validation. Its
profile bundle is the matching 0.19 one and its
cluster config carries `link_bw` / `link_latency` explicitly, since there is no
interconnect measurement to inherit. Those explicit link values are configuration
assumptions, not measured interconnect parameters; this TP=1 example does not
validate collective transport.

### RTXPRO6000 — Llama-3.1-8B (TP=1 dense)

![Llama-3.1-8B throughput](/img/validation/rtxpro6000-llama-3.1-8b-throughput.png)

| Metric | vLLM (ms) | Sim (ms) | Diff |
| --- | --- | --- | --- |
| TTFT mean | 6515.4 | 6559.8 | **+0.7%** |
| TTFT median | 8366.7 | 8415.1 | +0.6% |
| TTFT P90 | 15304.4 | 15408.5 | +0.7% |
| TTFT P95 | 17076.9 | 17088.9 | +0.1% |
| TTFT P99 | 18387.1 | 18501.1 | +0.6% |
| TPOT mean | 31.4 | 31.5 | **+0.3%** |
| TPOT median | 32.6 | 32.7 | +0.3% |
| TPOT P90 | 35.6 | 35.7 | +0.3% |
| TPOT P95 | 35.9 | 36.0 | +0.3% |
| TPOT P99 | 36.4 | 36.6 | +0.5% |
| Latency mean | 26946.4 | 27057.9 | **+0.4%** |
| Latency median | 28357.9 | 28429.1 | +0.3% |
| Latency P90 | 33840.9 | 33950.1 | +0.3% |
| Latency P95 | 35091.3 | 35251.9 | +0.5% |
| Latency P99 | 36064.9 | 36158.1 | +0.3% |

This TP1 scope uses refreshed vLLM 0.31.0 eager CUDA measurements with
the target serving graph policy declared separately. Local graph padding
changes forward rows without inventing requests or KV history. The reference
records `disable_custom_all_reduce: false`; TP1 has no TP collective to exercise.
A good result on this stored workload does not establish the same accuracy
for another model, GPU or execution contract.

### RTXPRO6000 — Qwen3-32B (TP=2 dense)

![Qwen3-32B throughput](/img/validation/rtxpro6000-qwen3-32b-throughput.png)

| Metric | vLLM (ms) | Sim (ms) | Diff |
| --- | --- | --- | --- |
| TTFT mean | 27758.1 | 27707.5 | **-0.2%** |
| TTFT median | 31314.2 | 31323.0 | +0.0% |
| TTFT P90 | 65070.1 | 65063.4 | -0.0% |
| TTFT P95 | 67848.2 | 67823.0 | -0.0% |
| TTFT P99 | 72188.3 | 72151.1 | -0.1% |
| TPOT mean | 67.5 | 67.7 | **+0.2%** |
| TPOT median | 70.5 | 70.8 | +0.5% |
| TPOT P90 | 78.3 | 78.5 | +0.2% |
| TPOT P95 | 79.6 | 79.8 | +0.2% |
| TPOT P99 | 80.0 | 80.2 | +0.3% |
| Latency mean | 72714.0 | 72769.0 | **+0.1%** |
| Latency median | 79224.6 | 79287.4 | +0.1% |
| Latency P90 | 94933.6 | 94847.6 | -0.1% |
| Latency P95 | 99410.6 | 99249.0 | -0.2% |
| Latency P99 | 102554.6 | 102303.0 | -0.2% |

TP=2 exercises the decoder all-reduces, plus the shared
embedding all-reduce and logits all-gather. Head tensor sizes now use the
same per-sequence row count as the profile lookup, with padded vocabulary
shards and full-vocabulary sampler inputs. This deployment uses refreshed
vLLM 0.31.0 TP2 measurements and the required TP1 stable-entry source.

Profile coverage alone does not establish generalization. Communication parameters and profiled
compute times are not fitted to these results. This dense TP example does
not validate MoE dispatch/combine or alternative heads.

This reference records custom all-reduce disabled for its NCCL communication
baseline with the calibrated interconnect; the general bench default is unchanged.
Its config omits `link_bw`, `link_latency` and `collective_links`, inheriting
the hardware bundle's AllReduce and AllGather bandwidths and common latency.
No benchmark-fitted bandwidth or model-specific override is applied.

### RTXPRO6000 — Qwen3-30B-A3B-Instruct-2507 (DP=2 × EP=2 MoE)

![Qwen3-30B-A3B throughput](/img/validation/rtxpro6000-qwen3-30b-a3b-throughput.png)

| Metric | vLLM (ms) | Sim (ms) | Diff |
| --- | --- | --- | --- |
| TTFT mean | 929.3 | 927.1 | **-0.2%** |
| TTFT median | 158.0 | 157.7 | -0.2% |
| TTFT P90 | 4202.2 | 4252.5 | +1.2% |
| TTFT P95 | 7163.7 | 7098.1 | -0.9% |
| TTFT P99 | 8796.7 | 8888.1 | +1.0% |
| TPOT mean | 45.9 | 45.7 | **-0.3%** |
| TPOT median | 47.3 | 47.3 | -0.1% |
| TPOT P90 | 50.5 | 50.7 | +0.3% |
| TPOT P95 | 51.1 | 51.0 | -0.1% |
| TPOT P99 | 51.6 | 51.7 | +0.2% |
| Latency mean | 31244.6 | 31157.3 | **-0.3%** |
| Latency median | 30449.0 | 30338.4 | -0.4% |
| Latency P90 | 37224.3 | 37200.7 | -0.1% |
| Latency P95 | 38401.0 | 38417.5 | +0.0% |
| Latency P99 | 42605.4 | 42462.8 | -0.3% |

This is DP+EP, not prefill/decode disaggregation: two data-parallel
members share experts and execute wave-synchronized collectives. The
simulator first resolves each member's local graph padding and then the
common DP mode. The head and attention request geometry remain separate
from padded forward rows.

This example uses refreshed vLLM 0.31.0 eager native MoE component tables
for TP1/DP2/EP2, with target serving graph padding resolved separately.
Its reference retains `disable_custom_all_reduce: false`; TP1 does not invoke
the TP custom-all-reduce path. Hidden states, top-k weights and IDs have
separate collective payloads. Transport bandwidths come from primitive NCCL
measurements, not these end-to-end timings; the calibration retains the existing
common latency. The bundle also contains TP2 and native TP1/DP1/EP1
acquisition data; this benchmark does not validate those deployments.
Its retained whole-block `moe.csv` is historical fallback data.

The transport path still uses a worst-rank Ring approximation for unequal
contributions and does not reproduce grouped NCCL kernel execution exactly.
The measured native contract covers this TP1/DP2/EP2 deployment, not arbitrary
TP/DP degrees, routing histograms, quantized backends or other model families.

### Additional diagnostic: DeepSeek-V3.2-Exp-16L64E

This stored dummy-weight checkpoint reduces depth and expert count so the
sparse model fits one card. It is not a full-size DeepSeek deployment. Its
simulator-side refresh uses the recorded block size of 64.

| Metric | vLLM | Sim | Diff |
| --- | --- | --- | --- |
| TTFT mean | 29.55 s | 24.83 s | **-16.0%** |
| TTFT median | 32.09 s | 27.18 s | -15.3% |
| TTFT P99 | 64.81 s | 60.55 s | -6.6% |
| TPOT mean | 55.7 ms | 57.3 ms | **+2.8%** |
| TPOT P99 | 77.3 ms | 71.6 ms | -7.4% |
| Latency mean | 66.80 s | 63.03 s | **-5.6%** |

It remains an accuracy limitation even when its deterministic regression
digest matches. All fifteen statistics are retained in its `summary.txt`.

## Reproducing locally

The bench module ships with reproduction scripts that re-run the
simulator side and re-run the comparison against the committed vLLM
artifacts:

Published examples use repository-relative dataset paths and equivalent
canonical model identifiers instead of machine-local paths. All other emitted
values and rows remain unchanged, including measurement settings, validation
results and logs.

```bash
# Sim side: writes bench/examples/<hardware>/<model>/outputs/sim.csv
./bench/examples/run.sh                       # all discovered examples
./bench/examples/run.sh RTX4090/Llama-3.1-8B  # or one at a time

# Compare: writes bench/examples/<hardware>/<model>/validation/{summary.txt, *.png}
./bench/examples/validate.sh
./bench/examples/validate.sh RTX4090/Llama-3.1-8B
```

Both scripts take `<hardware>/<model>` and discover the examples from
the directory layout, so every number on this page comes back from the
committed artifacts without editing a script.

The validation step regenerates the throughput / latency / requests
plots and the headline summary. To rerun vLLM itself (instead of
reusing the committed artifacts under
`bench/examples/<hardware>/<model>/vllm/`), use `python -m bench run` from
inside the vLLM container; see
[`bench/README.md`](https://github.com/casys-kaist/LLMServingSim/blob/main/bench/README.md)
for the full layout.

## What's next

- **[For Contributors → Validating your changes](/docs/contributor/validating-changes)**:
  `./serving/validate.sh` — the check you run before opening a PR, and
  how to report a number that moved.
- **[Simulator → Reading the output](/docs/simulator/reading-output)**:
  what every column in the per-request CSV means and how to derive
  your own metrics from it.
