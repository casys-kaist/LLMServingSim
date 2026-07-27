---
title: Sub-batch interleaving
sidebar_position: 2
---

# Sub-batch interleaving

> **What this demonstrates:** splitting each batch in half and
> running GPU dense layers on one half while PIM attention runs on
> the other, so neither device sits idle.

[PIM attention offload](../disaggregated/pim-attention-offload) by
itself often regresses prefill TTFT: the GPU finishes its dense
layers and waits for PIM to catch up on attention. Sub-batch
interleaving fixes this. The scheduler chops the batch into two
halves (`BATCH_1` and `BATCH_2`), and the trace generator alternates
GPU work on one half with PIM work on the other. Both devices stay
busy; total iteration time drops to roughly the slower of the two.

This is the natural follow-on to PIM offload, **don't enable it
without `--enable-attn-offloading`**.

## Prerequisites

- Simulator container set up
- Bundled RTXPRO6000 profile for `meta-llama/Llama-3.1-8B`
- A PIM device config (`configs/pim/DDR4_8GB_3200_pim/`); the
  bundled `single_node_pim_instance.json` already references it

## Cluster config

Same config as
[PIM attention offload](../disaggregated/pim-attention-offload) -
`configs/cluster/single_node_pim_instance.json`. No config changes
are needed; sub-batch interleaving is a runtime CLI flag.

## Run

```bash
python -m serving \
  --cluster-config 'configs/cluster/single_node_pim_instance.json' \
  --dtype bfloat16 --block-size 16 \
  --enable-attn-offloading \
  --enable-sub-batch-interleaving \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/pim_sub_batch_run.csv' \
  --log-level WARNING
```

The two flags work together:

- `--enable-attn-offloading` swaps the GPU attention kernel for
  the PIM kernel inside the trace.
- `--enable-sub-batch-interleaving` then splits each iteration's
  batch into two halves and emits an interleaved trace where one
  half's GPU dense layers overlap with the other half's PIM
  attention.

## Expected output

The throughput log shows both devices loaded:

```text
[INFO] step=10 batch=8 prompt_t=1.4k tok/s decode_t=620 tok/s
       npu_mem=63.4 GB pim_busy=78% gpu_busy=82%
[INFO] step=11 batch=8 prompt_t=1.4k tok/s decode_t=640 tok/s
       npu_mem=63.4 GB pim_busy=80% gpu_busy=80%
```

Compare against the pure-PIM run (without `--enable-sub-batch-interleaving`):
the GPU previously had long idle stretches while waiting on PIM;
now both `pim_busy` and `gpu_busy` plateau in the high 70s / 80s.

`outputs/pim_sub_batch_run.csv` has the same per-request schema as
any other run; what changes is the per-iteration latency, not the
column set.

## What's interesting

- **Prefill TTFT recovers.** Pure PIM offload regresses prefill
  (PIM's compute-per-channel is narrower than the GPU's parallel
  attention units). With interleaving the GPU's dense work hides
  most of the PIM prefill cost.
- **Decode is mostly unchanged.** Decode attention is already
  memory-bound and PIM-friendly, so sub-batch interleaving doesn't
  add much for decode-heavy workloads. The win is concentrated on
  the prefill side.
- **Half-batch granularity is the only knob.** The scheduler
  always splits 50/50. If a batch has only 1 request,
  interleaving silently no-ops (you can't split a single request
  into two halves without breaking the per-request semantics).
- **Trace tags.** If you read the generated trace file
  (`astra-sim/inputs/runs/<run_id>/trace/...`), each layer carries a `BATCH_1`
  or `BATCH_2` misc tag instead of the usual `NONE`. Confirms
  interleaving is actually emitted.

## Related examples

- **[PIM attention offload](../disaggregated/pim-attention-offload)** -
  the prerequisite. Sub-batch interleaving is the recovery layer
  on top of it.
- **[Power modeling](./power-modeling)**: turning on the `power:`
  block alongside this example shows how interleaving redistributes
  energy across NPU active and PIM compute.

## Where to learn more

- **[Simulator → PIM offload](/docs/simulator/specialized/pim-offload)**:
  the PIM device model and how the trace generator emits
  `PIM {channel}` / `PIM END` markers. Sub-batch interleaving
  sits on top of these.
- **[Reference → Trace format](/docs/reference/trace-format)**:
  the `BATCH_1` / `BATCH_2` misc tag semantics.
