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

**The three dense configurations stay within 1.6% across their 15 metrics.
The DP+EP MoE configuration reaches +5.6% and remains a separate accuracy target.**

> **Want to validate your own change?** See
> **[For Contributors → Validating your changes](/docs/contributor/validating-changes)**
> for the regression workflow.

## Setup

| Knob | Value |
| --- | --- |
| **Workload** | 300 ShareGPT-derived requests, ~10 sps Poisson arrivals |
| **Hardware** | RTXPRO6000 and RTX 4090, single node (profile bundles in `profiler/perf/<hardware>/`) |
| **vLLM version** | `v0.28.0` for the three RTXPRO6000 truths, which the bench container pins. The RTX 4090 truth stays on `v0.19.0`: that card is no longer in the machine, so it cannot be re-recorded — see the note under its section |
| **Block size** | 16 |
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
latency metric moving with it:

| | KV tokens | blocks | TTFT mean | TPOT mean | Latency mean |
| --- | --- | --- | --- | --- | --- |
| `mem_util: 0.9` (default) | 54,400 | 3,400 | -20.7% | +12.9% | -12.5% |
| `mem_util: 0.833919` (matched) | 41,408 | 2,588 | **+0.6%** | **+0.2%** | **+0.5%** |

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
| RTX 4090 | Llama-3.1-8B | TP=1 dense | +0.3% | +0.2% | +0.3% | +1.1% |
| RTXPRO6000 | Llama-3.1-8B | TP=1 dense | +0.9% | +0.1% | +0.3% | +1.5% |
| RTXPRO6000 | Qwen3-32B | TP=2 dense | -1.0% | -0.3% | -0.6% | -1.0% |
| RTXPRO6000 | Qwen3-30B-A3B-Instruct-2507 | DP=2 x EP=2 MoE | +3.6% | +0.7% | +0.8% | +5.6% |

These workloads were used during development, including skew-axis selection.
Their results are useful regression checks, not an independent estimate of
generalization to other models, hardware or request distributions.

The last column is the largest absolute error across all fifteen metrics
(TTFT / TPOT / latency x mean / median / P90 / P95 / P99), which is the honest
summary of a run: a mean can be small because two errors cancelled. TPOT and
end-to-end latency means are inside 1% on all four, but the MoE TTFT tail is not.

Queueing can amplify a small step-cost error into a larger TTFT error.
The current Llama reference over-predicts mean TTFT by 0.9%, while
Qwen3-32B is at -1.0% TTFT and -0.3% TPOT. These aggregate
signs do not identify a unique kernel-level cause or establish generalization.
The default tables are measured profiles; no end-to-end benchmark coefficient
is fitted for these endpoint corrections.

Per-percentile numbers are in the same `summary.txt` files under
[`bench/examples/`](https://github.com/casys-kaist/LLMServingSim/tree/main/bench/examples).

:::caution[The DP+EP row is a single draw, and its tail has a wide error bar]
The MoE configuration's TTFT tail is not reproducible **on the engine side**.
Twelve runs of it with identical flags, weights and workload spread 16.9% on
TTFT mean and 22.1% on TTFT P90, because which data-parallel member's batch
pairs with which in a round depends on arrival timing vLLM does not control.
Scoring one fixed simulator output against all twelve gives a TTFT mean
anywhere from -10.2% to +5.0%.

That spread is vLLM's, not the simulator's — the simulator is deterministic and
returns the same clock every time. So the committed truth is chosen for
**representativeness**, by distance to the twelve runs' own median, and not for
agreement: the runs that agree best with the simulator are the least
representative ones, which is exactly how a compensating error gets published.
TPOT, end-to-end latency and the run *span* are far tighter (the span is
deterministic to 0.05% on both sides) and are the metrics to trust here.
:::

## Per-configuration results

### RTX 4090 — Llama-3.1-8B (TP=1 dense)

Throughput timeline, vLLM (orange) vs. simulator (blue):

![RTX 4090 Llama-3.1-8B throughput](/img/validation/rtx4090-llama-3.1-8b-throughput.png)

| Metric | vLLM | Sim | Diff |
| --- | --- | --- | --- |
| TTFT mean | 65.49 s | 65.70 s | **+0.3%** |
| TTFT median | 60.13 s | 60.36 s | +0.4% |
| TTFT P99 | 137.36 s | 137.91 s | +0.4% |
| TPOT mean | 32.4 ms | 32.5 ms | **+0.2%** |
| TPOT P99 | 56.0 ms | 56.6 ms | +1.1% |
| Latency mean | 86.61 s | 86.87 s | **+0.3%** |
| Latency P99 | 153.63 s | 154.22 s | +0.4% |

The tightest configuration in the set: all fifteen metrics land between
+0.2% and +1.1%. Two things make it the cleanest comparison available.
The 24 GB card genuinely saturates its KV cache — the simulator preempts
198 times here and zero times on the other three — so the scheduler is
under real memory pressure on both sides rather than running with slack;
and its `mem_util` is calibrated to the block count vLLM actually
resolved, so the two are working from the same capacity. The latency
model itself is untouched — profiled latencies go in as measured — so
TPOT at +0.2% is a free prediction rather than a fit.

It is also the one truth still recorded on vLLM `v0.19.0`, because the card has
since left the machine. Its profile bundle is the matching 0.19 one and its
cluster config carries `link_bw` / `link_latency` explicitly, since there is no
interconnect measurement to inherit — a reader can see those two numbers are
the author's choice rather than measured.

### RTXPRO6000 — Llama-3.1-8B (TP=1 dense)

![Llama-3.1-8B throughput](/img/validation/rtxpro6000-llama-3.1-8b-throughput.png)

| Metric | vLLM | Sim | Diff |
| --- | --- | --- | --- |
| TTFT mean | 6.55 s | 6.60 s | **+0.9%** |
| TTFT median | 8.43 s | 8.39 s | -0.5% |
| TTFT P99 | 18.49 s | 18.64 s | +0.8% |
| TPOT mean | 31.5 ms | 31.6 ms | **+0.1%** |
| TPOT P99 | 36.4 ms | 37.0 ms | +1.5% |
| Latency mean | 27.04 s | 27.11 s | **+0.3%** |
| Latency P99 | 36.17 s | 36.19 s | +0.0% |

This bundle includes broader measured skew geometry and uses reference-aligned,
offline weighted-median calibration. Its fifteen absolute errors improve over
the previous bundled skew fit. This does not show that the acquisition defaults
reproduce the same accuracy on a fresh model or GPU.

The same model and parallelism on a 96 GB card. It never preempts, though its
block pool does reach 79% of budget. On one controlled uniform-KV shape, the
simulator charges a decode step **1.6% above production** (production 26.04 ms
against the simulator's 26.47 at n=128, kv 1190). The
saturated queue can amplify a step-level discrepancy. This single shape does
not identify the entire run's residual or establish a universal correction.

Two observed measurement sensitivities are token layout and profiled depth.
`dense.csv` records only the **prefill** token layout, because the profiler
packs a shot's tokens into one request, and the same token count spread over
decode sequences measures 2-4% cheaper on the projections (`down_proj` at
0.998-1.002 is the control that says this is a layout effect, not a scale
factor). And the per-layer figures come from a 1-layer boot, which reads
attention 1.1% high against a real 32-layer depth. Both are properties of how a
bundle is measured rather than of the simulator, and both are still open.

### RTXPRO6000 — Qwen3-32B (TP=2 dense)

![Qwen3-32B throughput](/img/validation/rtxpro6000-qwen3-32b-throughput.png)

| Metric | vLLM | Sim | Diff |
| --- | --- | --- | --- |
| TTFT mean | 35.62 s | 35.28 s | **-1.0%** |
| TTFT median | 39.47 s | 39.21 s | -0.7% |
| TTFT P99 | 89.43 s | 88.63 s | -0.9% |
| TPOT mean | 77.4 ms | 77.1 ms | **-0.3%** |
| TPOT P99 | 94.3 ms | 93.8 ms | -0.5% |
| Latency mean | 87.17 s | 86.68 s | **-0.6%** |
| Latency P99 | 120.69 s | 120.36 s | -0.3% |

TP=2 exercises the decoder all-reduces, plus the previously omitted shared
embedding all-reduce and logits all-gather. Head tensor sizes now use the
same per-sequence row count as the profile lookup, with padded vocabulary
shards and full-vocabulary sampler inputs. These results include broader TP2
skew measurements and the same calibration rule used for Llama.

All fifteen errors remain below 1% in absolute value, but only two improve
over the previous bundled fit; thirteen worsen. More profile rows and a shared
estimator are not by themselves evidence of improved generalization. The communication
parameters and profiled compute times were not fitted to these results.
This checkpoint does not validate MoE dispatch/combine or alternative heads.

### RTXPRO6000 — Qwen3-30B-A3B-Instruct-2507 (DP=2 × EP=2 MoE)

![Qwen3-30B-A3B throughput](/img/validation/rtxpro6000-qwen3-30b-a3b-throughput.png)

| Metric | vLLM | Sim | Diff |
| --- | --- | --- | --- |
| TTFT mean | 1.11 s | 1.15 s | **+3.6%** |
| TTFT median | 0.17 s | 0.18 s | +3.4% |
| TTFT P99 | 9.78 s | 10.29 s | +5.2% |
| TPOT mean | 47.2 ms | 47.5 ms | **+0.7%** |
| TPOT P99 | 53.1 ms | 54.3 ms | +2.3% |
| Latency mean | 32.29 s | 32.54 s | **+0.8%** |
| Latency P99 | 43.78 s | 44.00 s | +0.5% |

The disaggregated path: data-parallel across two instances, expert-parallel
within each, with wave-synchronized collectives. TPOT and end-to-end latency
means stay below 1%. The skew table was recompiled with the common rule;
this migration does not change or validate MoE communication modelling.

Its TTFT tail (+5.6% at P90 and +5.2% at P99) is the widest error on the page, and it is also
the one number here that a single vLLM run cannot pin down: see the caution
under the headline table for the 22.1% engine-side spread across twelve
identical runs, and why the committed truth is chosen for representativeness
rather than for agreement. This run also has the most memory headroom of the
four (59% of budget), so nothing about it is a capacity effect.

Three modelling details it exercises that the dense runs do not, each measured
rather than assumed: a DP round is padded only while it fits the CUDA-graph
capture range; the EP dispatch/combine is **ragged**, so its cost is
`gathered - min` rather than the average rank's share; and the number of
distinct experts a batch activates is read from a measured gate curve
(`--gate-stats`) rather than from the uniform closed form, which runs 0.87x of
it through the middle of the range.

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
