---
sidebar_position: 3
title: Quickstart
---

# Quickstart

Run your first end-to-end simulation in under a minute.

This walkthrough assumes you've finished
[Installation → Simulator setup](./installation/simulator) and you're
inside the simulator container at `/app/LLMServingSim`.

## Run the example

```bash
python -m serving \
  --cluster-config 'configs/cluster/single_node_single_instance.json' \
  --block-size 16 \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/example_single_run.csv' \
  --log-interval 1.0
```

That's the whole thing. The simulator will:

1. Load the cluster topology from
   `configs/cluster/single_node_single_instance.json` (a single
   RTXPRO6000 GPU running Llama-3.1-8B at TP=1).
2. Stream requests from `workloads/example_trace.jsonl` according
   to their arrival times.
3. Step ASTRA-Sim each scheduling iteration to get cycle counts.
4. Write per-request latency metrics to
   `outputs/example_single_run.csv`.

You should see throughput, memory, and power lines printed roughly
once per second. After the run finishes:

```bash
head outputs/example_single_run.csv
```

shows the per-request output. The columns are `instance id`,
`request id`, `model`, `input`, `output`, `arrival`, `end_time`,
`latency`, `queuing_delay`, `TTFT`, `TPOT`, `ITL` — see
**[Reading the output](/docs/simulator/reading-output)**.

## What the flags mean

| Flag | What it does |
| --- | --- |
| `--cluster-config` | Cluster topology + hardware. Generates ASTRA-Sim input files automatically. |
| `--block-size` | KV-cache block size in tokens. Omit it and the simulator uses whatever the profile bundle records vLLM settling on, which is what vLLM does — it derives a block size rather than accepting one. |
| `--dataset` | JSONL file of requests (or agentic sessions). |
| `--output` | Where to write per-request metrics. |
| `--log-interval` | How often to print the throughput / memory / power summary line (seconds). |

There is no dtype flag. Weight and KV-cache precision are read from the
model config, along with three more cache dtypes a modern checkpoint
carries — see **[CLI flags → Precision](/docs/reference/cli-flags)**.

The full flag list lives at
[Reference → CLI flags](/docs/reference/cli-flags).

## Try a different scenario

`serving/run.sh` ships a few worked examples, multi-instance,
prefill/decode disaggregation, MoE with EP, prefix caching, CXL
memory, PIM offload, and sub-batch interleaving:

```bash
./serving/run.sh
```

Each block in that script is self-contained and ready to copy into
your own scripts. Browse the cluster configs that drive them:

```bash
ls configs/cluster/
```

## What's next

- **[Simulator → Architecture overview](/docs/simulator/architecture)**
  to understand how the simulator runs internally.
- **[Simulator → Reading the output](/docs/simulator/reading-output)**
  to understand the metrics in `*.csv`.
- **[Workloads → JSONL format](/docs/workloads/jsonl-format)**
  to drive the simulator with your own traces.
- **[Profiler overview](/docs/profiler/overview)** if you want to
  add new hardware or models.
