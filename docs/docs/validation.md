---
title: Validation
sidebar_position: 3
description: How LLMServingSim's output compares against real vLLM
---

# Validation

LLMServingSim is validated end-to-end against real vLLM on the
**bundled `(hardware, model)` combos**. The numbers below come from
running a 300-request ShareGPT replay through both vLLM v0.19.0 and
the simulator on RTXPRO6000, then comparing the per-request and
per-tick metrics with `python -m bench validate`.

> **Want to validate your own change?** See
> **[For Contributors → Validating your changes](/docs/contributor/validating-changes)**
> for the regression workflow.

## Setup

| Knob | Value |
| --- | --- |
| **Workload** | 300 ShareGPT-derived requests, ~10 sps Poisson arrivals |
| **Hardware** | RTXPRO6000 (single node, profile bundle in `profiler/perf/RTXPRO6000/`) |
| **vLLM version** | `v0.19.0` (the pin used by the bench container) |
| **Block size** | 16 |
| **Engine flags** | Defaults except where the cluster config dictates otherwise |
| **Cluster configs** | `bench/examples/configs/<model>.json` |

Inputs and outputs (vLLM token IDs, sampling params, per-request
timings) are pinned via `bench`'s strict-replay path so both runs
process exactly the same prompts in the same order.

## Headline numbers

Mean error vs. real vLLM, per metric, on the three currently bundled
configurations:

| Model | Parallelism | TTFT mean | TPOT mean | Latency mean |
| --- | --- | --- | --- | --- |
| Llama-3.1-8B                | TP=1 dense       | -2.6% | -0.4% | -1.0% |
| Qwen3-32B                   | TP=2 dense       | +1.8% | +1.2% | +1.5% |
| Qwen3-30B-A3B-Instruct-2507 | DP=2 x EP=2 MoE  | -5.8% | -0.2% | -0.4% |

Every number on this page is read out of the committed
`bench/examples/<model>/validation/summary.txt` files, so it is
reproducible rather than quoted.

**TPOT and end-to-end latency means land within 1.5% on all three
configurations**, and the DP+EP MoE path tracks vLLM as tightly as the
dense TP path on both. TTFT means are looser: -2.6% on Llama and -5.8%
on the MoE run.

That TTFT spread is expected rather than alarming, and it is worth
knowing why before reading the tables. The two sides do not define the
measurement identically — the simulator stops the clock when the first
token's *computation* finishes, vLLM when the client *receives* it — and
TTFT is dominated by queueing, so a small scheduling difference early in
the run moves the mean a lot. On the MoE run the median TTFT is 139 ms
against 111 ms, so a 28 ms absolute difference reads as -20%. Judge
TTFT by its absolute error and its tail, not by percentage of a very
small number: P90 through P99 land within 5.4% on every configuration.

Per-percentile numbers (median / P90 / P95 / P99) are in the same
`summary.txt` files under
[`bench/examples/`](https://github.com/casys-kaist/LLMServingSim/tree/main/bench/examples).

## Per-model results

### Llama-3.1-8B (TP=1 dense)

Throughput timeline, vLLM (orange) vs. simulator (blue):

![Llama-3.1-8B throughput](/img/validation/llama-3.1-8b-throughput.png)

Headline error vs. vLLM:

| Metric | vLLM | Sim | Diff |
| --- | --- | --- | --- |
| TTFT mean     |    7.10 s |    6.92 s | **-2.6%** |
| TTFT P99      |   19.76 s |   19.63 s | -0.6% |
| TPOT mean     |   32.5 ms |   32.3 ms | **-0.4%** |
| TPOT P99      |   37.3 ms |   37.8 ms | +1.2% |
| Latency mean  |   28.20 s |   27.92 s | **-1.0%** |
| Latency P99   |   37.64 s |   37.38 s | -0.7% |

Single-instance dense Llama is the simplest configuration, and the
tightest: TPOT within 0.4% and end-to-end latency within 1.0% at every
percentile. TTFT mean is -2.6%, and its median -7.0% — the simulator
gets first tokens out slightly early, which matters proportionally more
on the metric with the smallest absolute values.

### Qwen3-32B (TP=2 dense)

Throughput timeline:

![Qwen3-32B throughput](/img/validation/qwen3-32b-throughput.png)

Headline error vs. vLLM:

| Metric | vLLM | Sim | Diff |
| --- | --- | --- | --- |
| TTFT mean     |   36.91 s |   37.59 s | **+1.8%** |
| TTFT P99      |   93.35 s |   94.68 s | +1.4% |
| TPOT mean     |   80.3 ms |   81.3 ms | **+1.2%** |
| TPOT P99      |   97.1 ms |   98.8 ms | +1.8% |
| Latency mean  |   90.41 s |   91.73 s | **+1.5%** |
| Latency P99   |  126.34 s |  128.45 s | +1.7% |

TP=2 exercises the dense ALLREDUCE collective on `o_proj` /
`down_proj`. This is the most uniformly accurate of the three: every
metric at every percentile lands between +1.2% and +2.3%, and every one
of them is positive — the simulator consistently over-predicts by a
little rather than scattering. A systematic bias of that shape points at
a small fixed per-iteration overhead, not at a modelling error in any
one component.

### Qwen3-30B-A3B-Instruct-2507 (DP=2 × EP=2 MoE)

Throughput timeline:

![Qwen3-30B-A3B-Instruct-2507 throughput](/img/validation/qwen3-30b-a3b-throughput.png)

Headline error vs. vLLM:

| Metric | vLLM | Sim | Diff |
| --- | --- | --- | --- |
| TTFT mean     |    1.09 s |    1.02 s | **-5.8%** |
| TTFT P99      |    9.59 s |   10.10 s | +5.4% |
| TPOT mean     |   47.3 ms |   47.2 ms | **-0.2%** |
| TPOT P99      |   53.3 ms |   54.0 ms | +1.4% |
| Latency mean  |   32.34 s |   32.22 s | **-0.4%** |
| Latency P99   |   43.90 s |   43.73 s | -0.4% |

This is the disaggregated path: data-parallel across two instances,
expert-parallel within each instance, with wave-synchronized ALLTOALL
on the 2D ASTRA-Sim topology. TPOT and latency are the best of the
three — within 0.2% and 0.4% on the mean, and within 1.4% at P99.

TTFT is the outlier at -5.8% on the mean and -19.9% on the median, and
the absolute numbers explain it: the median is 139 ms against 111 ms, a
28 ms difference. This workload arrives at 0.2 sessions/s, so most
prefills are scheduled immediately and TTFT is nearly pure compute with
almost no queueing to average over. The tail, where queueing does
appear, lands within 5.4%.

## Reproducing locally

The bench module ships with reproduction scripts that re-run the
simulator side and re-run the comparison against the committed vLLM
artifacts:

```bash
# Sim side: writes bench/examples/<model>/outputs/sim.csv
./bench/examples/run.sh Llama-3.1-8B
./bench/examples/run.sh Qwen3-32B
./bench/examples/run.sh Qwen3-30B-A3B-Instruct-2507

# Compare: writes bench/examples/<model>/validation/{summary.txt, *.png}
./bench/examples/validate.sh Llama-3.1-8B
./bench/examples/validate.sh Qwen3-32B
./bench/examples/validate.sh Qwen3-30B-A3B-Instruct-2507
```

The validation step regenerates the throughput / latency / requests
plots and the headline summary. To rerun vLLM itself (instead of
reusing the committed artifacts under
`bench/examples/<model>/vllm/`), use `python -m bench run` from
inside the vLLM container; see
[`bench/README.md`](https://github.com/casys-kaist/LLMServingSim/blob/main/bench/README.md)
for the full layout.

## What's next

- **[For Contributors → Validating your changes](/docs/contributor/validating-changes)**:
  the three-tier check (smoke → scenario → bench validate) you run
  before opening a PR, plus what regression to flag.
- **[Simulator → Reading the output](/docs/simulator/reading-output)**:
  what every column in the per-request CSV means and how to derive
  your own metrics from it.
