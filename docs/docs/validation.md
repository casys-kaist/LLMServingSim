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
`python -m bench validate`. Every figure on this page is read out of the
committed `bench/examples/<hardware>/<model>/validation/summary.txt`, so it is
reproducible rather than quoted.

**The largest displayed absolute error across the three dense configurations is 1.5%.
The DP+EP MoE configuration reaches +5.9% and remains a separate accuracy target.**

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
| **vLLM version** | `v0.28.0` for the three RTXPRO6000 headline truths, which the bench container pins. The RTX 4090 truth stays on `v0.19.0`: that card is no longer in the machine, so it cannot be re-recorded — see the note under its section |
| **Block size** | 16 for the headline examples; 64 for the additional DeepSeek diagnostic |
| **Engine flags** | Defaults except where the cluster config dictates otherwise |
| **Cluster configs** | `bench/examples/<hardware>/<model>/config.json` |
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
## Headline numbers

Mean error vs. real vLLM, per metric, on the four configurations summarized here:

| Hardware | Model | Parallelism | TTFT mean | TPOT mean | Latency mean | worst of 15 |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 4090 | Llama-3.1-8B | TP=1 dense | +0.5% | +0.4% | +0.5% | +1.2% |
| RTXPRO6000 | Llama-3.1-8B | TP=1 dense | +1.0% | +0.1% | +0.3% | +1.5% |
| RTXPRO6000 | Qwen3-32B | TP=2 dense | -1.0% | -0.2% | -0.5% | -1.0% |
| RTXPRO6000 | Qwen3-30B-A3B-Instruct-2507 | DP=2 x EP=2 MoE | +4.8% | +0.9% | +1.0% | +5.9% |

These workloads were used during development, including skew-axis selection.
Their results are useful regression checks, not an independent estimate of
generalization to other models, hardware or request distributions.

The last column is the largest absolute error across all fifteen metrics
(TTFT / TPOT / latency x mean / median / P90 / P95 / P99), which is the honest
summary of a run: a mean can be small because two errors cancelled. The MoE TTFT tail remains outside the target even though its TPOT and latency
means are much closer.

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

Throughput timeline, vLLM (orange) vs. simulator (blue):

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

It is also the one truth still recorded on vLLM `v0.19.0`, because the card has
since left the machine. Its profile bundle is the matching 0.19 one and its
cluster config carries `link_bw` / `link_latency` explicitly, since there is no
interconnect measurement to inherit — a reader can see those two numbers are
the author's choice rather than measured.

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

| Metric | vLLM | Sim | Diff |
| --- | --- | --- | --- |
| TTFT mean | 35.62 s | 35.28 s | **-1.0%** |
| TTFT median | 39.47 s | 39.21 s | -0.7% |
| TTFT P99 | 89.43 s | 88.63 s | -0.9% |
| TPOT mean | 77.4 ms | 77.2 ms | **-0.2%** |
| TPOT P99 | 94.3 ms | 93.8 ms | -0.5% |
| Latency mean | 87.17 s | 86.70 s | **-0.5%** |
| Latency P99 | 120.69 s | 120.52 s | -0.1% |

TP=2 exercises the decoder all-reduces, plus the previously omitted shared
embedding all-reduce and logits all-gather. Head tensor sizes now use the
same per-sequence row count as the profile lookup, with padded vocabulary
shards and full-vocabulary sampler inputs. These results include broader TP2
skew measurements and the same calibration rule used for Llama.

More profile rows and a shared calibration rule are not by themselves
evidence of improved generalization. Communication parameters and profiled
compute times are not fitted to these results. This dense TP example does
not validate MoE dispatch/combine or alternative heads.

### RTXPRO6000 — Qwen3-30B-A3B-Instruct-2507 (DP=2 × EP=2 MoE)

![Qwen3-30B-A3B throughput](/img/validation/rtxpro6000-qwen3-30b-a3b-throughput.png)

| Metric | vLLM | Sim | Diff |
| --- | --- | --- | --- |
| TTFT mean | 1.11 s | 1.16 s | **+4.8%** |
| TTFT median | 0.17 s | 0.18 s | +3.3% |
| TTFT P99 | 9.78 s | 10.36 s | +5.9% |
| TPOT mean | 47.2 ms | 47.6 ms | **+0.9%** |
| TPOT P99 | 53.1 ms | 54.3 ms | +2.4% |
| Latency mean | 32.29 s | 32.63 s | **+1.0%** |
| Latency P99 | 43.78 s | 44.10 s | +0.7% |

This is DP+EP, not prefill/decode disaggregation: two data-parallel
members share experts and execute wave-synchronized collectives. The
simulator first resolves each member's local graph padding and then the
common DP mode. The head and attention request geometry remain separate
from padded forward rows.

The current aggregate dispatch/combine representation uses a worst-rank
analytical approximation for unequal contributions. It does not reproduce
native grouped multi-tensor NCCL timing exactly. This configuration remains
an accuracy limitation; graph-shape alignment alone does not establish a
complete MoE execution model.

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
