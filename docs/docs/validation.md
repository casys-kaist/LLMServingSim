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

:::caution[Recorded-bundle validation is not fresh-profiler validation]
These results use the checked-in profiles. Updating the profiler does not
remeasure those files: acquisition protocols and coverage must be checked in
each bundle's metadata. Reproducing these examples does not establish accuracy
using only new-protocol measurements, or validate every supported architecture.
:::

**The largest displayed absolute error across the three dense configurations is 1.5%.
The bundled DP+EP MoE example is within 2.6% across all fifteen statistics.**

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
| **vLLM version** | `v0.28.0` for the three RTXPRO6000 headline truths, matching the bench container pin. The retained RTX 4090 reference uses `v0.19.0` |
| **Block size** | 16 for the headline examples; 64 for the additional DeepSeek diagnostic |
| **Engine flags** | Recorded in each example's `vllm/meta.json`; match its effective settings rather than assuming current CLI defaults |
| **Cluster configs** | `bench/examples/<hardware>/<model>/config.json` |
| **Interconnect** | Qwen3-32B and Qwen3-30B use matched NCCL-only references and inherit per-operation BW and common latency from `profiler/perf/RTXPRO6000/hardware.yaml` |
| **KV capacity** | `mem_util` `0.9`, except the RTX 4090 example which is calibrated to the measured block count (see below) |

Inputs and outputs (vLLM token IDs, sampling params, per-request
timings) are pinned via `bench`'s strict-replay path so both runs
process exactly the same prompts in the same order.

:::caution[Match `mem_util` to the real run whenever the KV cache saturates]
`npu_mem.mem_util` sizes the KV cache, and KV cache size only shows up in the
results once a run actually **fills** it — below that nothing is preempted and
the capacity is invisible. Of the four configurations here, only the RTX 4090
one is in that regime: 24 GB, pinned at its ceiling for most of the run. It
therefore sets `mem_util` so the simulator's block count equals the one vLLM
resolved, read out of that run's own `meta.json`:

```json
"kv_cache": { "num_gpu_blocks": 2588, "block_size": 16, "num_kv_tokens": 41408 }
```

That matters because the simulator does not model vLLM's activation peak or
CUDA context, so the default `mem_util: 0.9` yields *more* KV cache than vLLM
gets at the same fraction — less preemption, an early finish, and every
latency metric moving with it.

The example uses `mem_util: 0.833919` to reproduce those 2,588 blocks.
This matches a recorded capacity, not a latency target; its current errors
are listed in the result tables below.

The three RTXPRO6000 configurations stay at `0.9`, and the reason is not that
they are far from the ceiling — Qwen3-32B's pool reaches 97% of its budget.
It is that **neither side preempts a single request** on any of the three, so
no scheduling decision depends on where the ceiling is; on the RTX 4090 run the
simulator preempts 198 times. Calibrating `mem_util` on a run that never
preempts changes nothing.

Two cautions if you validate against your own vLLM run. Check preemption
first — `preempted` in the truth's `timeseries.csv` and the simulator's own
counter — because that, not a percentage, is what says the capacity is
load-bearing. And do not compare the two occupancy percentages directly: the
simulator's heartbeat counts cached-but-free blocks in `Each NPU Memory Usage`
while vLLM's `kv_cache_pct` counts only pinned ones, so the same run reads 71%
on one side and 28% on the other. Admission uses free blocks on both, so the
difference is cosmetic — but it is not a 2.5x discrepancy to chase.
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
| RTXPRO6000 | Llama-3.1-8B | TP=1 dense | +1.0% | +0.1% | +0.3% | +1.5% |
| RTXPRO6000 | Qwen3-32B | TP=2 dense | -1.0% | -0.7% | -0.8% | -1.1% |
| RTXPRO6000 | Qwen3-30B-A3B-Instruct-2507 | DP=2 x EP=2 MoE | -1.1% | +0.2% | +0.1% | -2.6% |

These workloads were used during development, including skew-axis selection.
Their results are useful regression checks, not an independent estimate of
generalization to other models, hardware or request distributions.

The last column is the largest absolute error across all fifteen metrics
(TTFT / TPOT / latency x mean / median / P90 / P95 / P99), which is the honest
summary of a run: a mean can be small because two errors cancelled. The native
MoE example remains inside 3% on this reference; that does not cover arbitrary
deployments or eliminate execution-to-execution TTFT variation.

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

| Metric | vLLM | Sim | Diff |
| --- | --- | --- | --- |
| TTFT mean | 6.55 s | 6.61 s | **+1.0%** |
| TTFT median | 8.43 s | 8.41 s | -0.2% |
| TTFT P99 | 18.49 s | 18.65 s | +0.9% |
| TPOT mean | 31.5 ms | 31.6 ms | **+0.1%** |
| TPOT P99 | 36.4 ms | 37.0 ms | +1.5% |
| Latency mean | 27.04 s | 27.13 s | **+0.3%** |
| Latency P99 | 36.17 s | 36.22 s | +0.2% |

This bundle includes measured heterogeneous attention geometry and
reference-aligned offline weighted-median skew calibration. Local graph
padding changes forward rows without inventing requests or KV history.
A good result on this stored workload does not establish the same accuracy
for another model, GPU or execution contract.

### RTXPRO6000 — Qwen3-32B (TP=2 dense)

![Qwen3-32B throughput](/img/validation/rtxpro6000-qwen3-32b-throughput.png)

| Metric | vLLM (ms) | Sim (ms) | Diff |
| --- | --- | --- | --- |
| TTFT mean | 27585.4 | 27319.5 | **-1.0%** |
| TTFT median | 31163.9 | 30868.5 | -0.9% |
| TTFT P90 | 64888.6 | 64235.3 | -1.0% |
| TTFT P95 | 67643.3 | 66927.8 | -1.1% |
| TTFT P99 | 71965.6 | 71183.8 | -1.1% |
| TPOT mean | 67.5 | 67.0 | **-0.7%** |
| TPOT median | 70.5 | 70.1 | -0.6% |
| TPOT P90 | 78.3 | 77.8 | -0.6% |
| TPOT P95 | 79.5 | 79.1 | -0.6% |
| TPOT P99 | 79.9 | 79.4 | -0.6% |
| Latency mean | 72511.3 | 71916.3 | **-0.8%** |
| Latency median | 79037.3 | 78440.3 | -0.8% |
| Latency P90 | 94694.1 | 93715.9 | -1.0% |
| Latency P95 | 99169.2 | 98128.8 | -1.0% |
| Latency P99 | 102326.2 | 101175.3 | -1.1% |

TP=2 exercises the decoder all-reduces, plus the previously omitted shared
embedding all-reduce and logits all-gather. Head tensor sizes now use the
same per-sequence row count as the profile lookup, with padded vocabulary
shards and full-vocabulary sampler inputs. These results include broader TP2
skew measurements and the same calibration rule used for Llama.

More profile rows and a shared calibration rule are not by themselves
evidence of improved generalization. Communication parameters and profiled
compute times are not fitted to these results. This dense TP example does
not validate MoE dispatch/combine or alternative heads.

This reference uses NCCL-only communication with the calibrated interconnect.
Its config omits `link_bw`, `link_latency` and `collective_links`, inheriting
the hardware bundle's AllReduce and AllGather bandwidths and common latency.
No benchmark-fitted bandwidth or model-specific override is applied.

### RTXPRO6000 — Qwen3-30B-A3B-Instruct-2507 (DP=2 × EP=2 MoE)

![Qwen3-30B-A3B throughput](/img/validation/rtxpro6000-qwen3-30b-a3b-throughput.png)

| Metric | vLLM (ms) | Sim (ms) | Diff |
| --- | --- | --- | --- |
| TTFT mean | 1046.2 | 1034.8 | **-1.1%** |
| TTFT median | 162.4 | 158.2 | -2.6% |
| TTFT P90 | 4964.9 | 4967.9 | +0.1% |
| TTFT P95 | 8051.6 | 7858.3 | -2.4% |
| TTFT P99 | 9453.6 | 9624.1 | +1.8% |
| TPOT mean | 46.8 | 46.9 | **+0.2%** |
| TPOT median | 48.4 | 48.5 | +0.2% |
| TPOT P90 | 51.4 | 51.9 | +0.8% |
| TPOT P95 | 51.9 | 52.3 | +0.7% |
| TPOT P99 | 52.6 | 53.1 | +0.8% |
| Latency mean | 32002.7 | 32037.1 | **+0.1%** |
| Latency median | 31103.0 | 31233.4 | +0.4% |
| Latency P90 | 38059.2 | 38132.0 | +0.2% |
| Latency P95 | 39208.5 | 39297.2 | +0.2% |
| Latency P99 | 43455.0 | 43532.0 | +0.2% |

This is DP+EP, not prefill/decode disaggregation: two data-parallel
members share experts and execute wave-synchronized collectives. The
simulator first resolves each member's local graph padding and then the
common DP mode. The head and attention request geometry remain separate
from padded forward rows.

This example uses NCCL-only communication and deployment-matched native MoE
tables, with local routing, gathered experts and local finalization measured
separately in eager and graph modes. Hidden states, top-k weights and IDs have
separate collective payloads. Transport bandwidths come from primitive NCCL
measurements, not these end-to-end timings; the calibration retains the existing
common latency. Attention and skew tables are unchanged by this refresh.

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
