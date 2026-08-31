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
in-flight batches at `pp_size`, the trace generator picks the stage
boundaries (always on a transformer-block edge, the same split rule
vLLM's `get_pp_indices` uses), and Chakra emits one `.et` per stage
NPU with send/recv at those boundaries.

## Prerequisites

- Simulator container set up
- Bundled RTXPRO6000 profile for `meta-llama/Llama-3.1-8B`

## Cluster config

Pipeline parallelism is driven entirely by `pp_size` on a multi-GPU
instance. Three bundled configs cover it:
`single_node_pp_instance.json` (4 GPUs as `pp=4`),
`single_node_tp_pp_instance.json` (`tp=2 x pp=2`) and
`single_node_moe_pp_instance.json` (MoE, `tp=2 x pp=2` with `ep=2`).
The first one:

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
          "num_npus": 4,
          "tp_size": 1,
          "pp_size": 4,
          "pd_type": null
        }
      ]
    }
  ]
}
```

The fields that matter:

- `num_npus: 4`, `tp_size: 1`, `pp_size: 4`: the invariant is
  `num_npus = tp_size * pp_size`, so the simulator splits
  Llama-3.1-8B's 32 decoder blocks into four stages of eight, one
  per GPU, with no TP within a stage.
- For combined TP × PP set `num_npus: 4, tp_size: 2, pp_size: 2`
  (that is `single_node_tp_pp_instance.json`): two stages of 16
  blocks, each stage sharded across two GPUs.
- `pp_size` may not exceed the model's `num_hidden_layers` — a
  pipeline stage with no decoder block is rejected up front.

## Run

```bash
python -m serving \
  --cluster-config 'configs/cluster/single_node_pp_instance.json' \
  --block-size 16 \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/example_pp_run.csv' \
  --num-reqs 10
```

No new CLI flag, the parallelism degree is fully driven by the
cluster config. Swap in `single_node_tp_pp_instance.json` or
`single_node_moe_pp_instance.json` to combine PP with TP and EP.

## Expected output

The throughput log looks like a standard single-instance run:

```text
[20.0s] Avg prompt throughput: 1436.0 tokens/s, Avg generation throughput: 540.0 tokens/s
        ├─Running Instance[0]: 8 reqs, Waiting: 0 reqs, Total # 4 NPUs, Each NPU Memory Usage 44032.19 MB (44.774 % Used), Prefix Cache Hit ratio 1.02 %, (1424 / 139612)
        └─Node[0]: Total CPU Memory Usage 0.00 MB, 0.000 % Used
[21.0s] Avg prompt throughput: 1502.0 tokens/s, Avg generation throughput: 560.0 tokens/s
        ├─Running Instance[0]: 8 reqs, Waiting: 0 reqs, Total # 4 NPUs, Each NPU Memory Usage 44118.19 MB (44.861 % Used), Prefix Cache Hit ratio 1.01 %, (1424 / 141114)
        └─Node[0]: Total CPU Memory Usage 0.00 MB, 0.000 % Used
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
- **`--enable-sub-batch-interleaving` is refused with `pp_size > 1`.**
  An interleaved trace leaves both sub-batches mid-block at every
  group edge, so a stage has no single activation to hand on.

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
  stage boundaries are chosen in `serving/core/trace_generator.py`
  (`_pp_stage_boundaries`) and consumed, along with the send/recv
  insertion, in
  `astra-sim/extern/graph_frontend/chakra/src/converter/llm_converter.py`
  (`get_stage_edges`, `convert_common` / `convert_prefill`).
