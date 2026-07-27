---
title: Pipeline parallel (PP)
sidebar_position: 2
---

# Pipeline parallel (PP)

> **What this demonstrates:** splitting a model's decoder layers
> across GPUs (one stage per GPU) so each iteration streams as
> micro-batches through the pipeline.

PP is the orthogonal axis to TP: TP shards weights *within* a
layer, PP shards layers *across* devices. Each GPU runs a
contiguous stretch of the decoder block stack and hands the
intermediate activations to the next stage. The scheduler caps
in-flight batches at `pp_size`, and Chakra splits each iteration's
layer list across stage NPUs with send/recv at the boundaries.

## Prerequisites

- Simulator container set up
- Bundled RTXPRO6000 profile for `meta-llama/Llama-3.1-8B`

## Cluster config

There's no PP-specific bundled config; pipeline-parallel runs by
flipping `pp_size` on a multi-GPU instance. Drop a
`single_node_pp_instance.json` next to the others:

```json title="configs/cluster/single_node_pp_instance.json"
{
  "num_nodes": 1,
  "link_bw": 16,
  "link_latency": 20000,
  "nodes": [
    {
      "num_instances": 1,
      "cpu_mem": {"mem_size": 512, "mem_bw": 256, "mem_latency": 0},
      "instances": [
        {
          "model_name": "meta-llama/Llama-3.1-8B",
          "hardware": "RTXPRO6000",
          "npu_mem": {"mem_size": 96, "mem_bw": 1597, "mem_latency": 0},
          "num_npus": 2,
          "tp_size": 1,
          "pp_size": 2,
          "pd_type": null
        }
      ]
    }
  ]
}
```

The two fields that matter:

- `num_npus: 2`, `tp_size: 1`, `pp_size: 2`: invariant is
  `num_npus = tp_size * pp_size`, so the simulator splits the model
  into two pipeline stages (each on its own GPU) with no TP within a
  stage.
- For combined TP × PP (e.g., 4 GPUs as `tp=2, pp=2`), set
  `num_npus: 4, tp_size: 2, pp_size: 2`.

## Run

```bash
python -m serving \
  --cluster-config 'configs/cluster/single_node_pp_instance.json' \
  --dtype bfloat16 --block-size 16 \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/pp2_run.csv' \
  --log-interval 1.0
```

No new CLI flag, the parallelism degree is fully driven by the
cluster config.

## Expected output

The throughput log looks like a standard single-instance run:

```text
[INFO] step=20 batch=8 prompt_t=1.4k tok/s decode_t=540 tok/s npu_mem=44.0 GB
[INFO] step=21 batch=8 prompt_t=1.5k tok/s decode_t=560 tok/s npu_mem=44.1 GB
```

Two things to notice vs. the TP=1 baseline:

- **`npu_mem` is roughly halved** (each GPU holds half the
  decoder layers, so weights + KV cache per device shrink).
- **`batch` may saturate at lower values** during short bursts
  because the scheduler stops issuing once `inflight == pp_size`,
  this is the back-pressure that prevents over-injecting work into
  the pipeline.

## What's interesting

- **Memory split is real.** Each stage holds only its slice of
  decoder layers, so per-GPU weight + KV-cache footprint shrinks
  roughly 1/`pp_size`. PP=2 lets you fit a model that doesn't fit
  on TP=1.
- **Inter-stage activation shipment is real.** Bumping
  `link_bw` / `link_latency` in the cluster config visibly moves
  iteration time, because the send/recv nodes Chakra inserts
  between stages route through the simulated network just like any
  other collective. Use this to study how interconnect choice
  affects PP scaling.
- **Pipeline depth caps in-flight batches.** `inflight ≤ pp_size`
  is the PP-driven scheduling constraint. With `pp_size=2` and a
  token budget that allows 6 batches, you'll see the scheduler
  queue at most 2 batches in the pipeline at once. Steady-state
  pipeline overlap (batch *k+1* on stage 0 while batch *k* is on
  stage 1) emerges naturally from ASTRA-Sim executing each stage's
  `.et` file independently.
- **What's not modeled.** Within a single iteration the batch is
  a single unit traversing stages in order — there's no
  micro-batch split *inside* one iteration, and no choice of
  pipeline schedule (1F1B, interleaved, etc.). The fill/drain
  bubbles you'd see in those schedules therefore don't appear; the
  pipelining benefit comes entirely from overlapping consecutive
  iterations up to `pp_size`.

## Related examples

- **[Tensor parallel](./tensor-parallel)**: the within-layer
  counterpart. TP × PP combinations are valid and common at
  large scale.
- **[Multi-instance LOAD routing](../disaggregated/multi-instance)**:
  the next-level-up scaling — replicate whole TP × PP groups
  across instances.

## Where to learn more

- **[Simulator → Parallelism mechanics](/docs/simulator/parallelism-mechanics)**:
  how `num_npus`, `tp_size`, and `pp_size` are validated and
  threaded through the scheduler / trace generator.
- The PP `inflight` list lives in `serving/core/scheduler.py`; the
  per-stage layer split and send/recv insertion live in
  `astra-sim/extern/graph_frontend/chakra/src/converter/llm_converter.py`
  (`convert_common` / `convert_prefill`).
