---
title: Power modeling
sidebar_position: 1
---

# Power modeling

> **What this demonstrates:** turning on the per-node power model so
> the simulator emits live wattage in the throughput log and a
> per-component energy breakdown at the end of the run.

The power model is opt-in: a node only tracks power when its config
includes a `power:` block. The bundled
`single_node_power_instance.json` is a ready-to-run example.

## Prerequisites

- Simulator container set up
- Bundled RTXPRO6000 profile for `meta-llama/Llama-3.1-8B`
  (no extra profiling needed)

## Cluster config

`configs/cluster/single_node_power_instance.json` adds a `power:`
block to the node alongside the usual `instances`:

```json title="configs/cluster/single_node_power_instance.json (excerpt)"
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
          "pd_type": null,
          "tp_size": 1
        }
      ],
      "power": {
        "base_node_power": 60,
        "npu": {
          "RTXPRO6000": {
            "idle_power": 35,
            "standby_power": 300,
            "active_power": 600,
            "standby_duration": 18
          }
        },
        "cpu":     {"idle_power": 10, "active_power": 200, "util": 0.15},
        "dram":    {"dimm_size": 32,  "idle_power": 2.0,   "energy_per_bit": 6.0},
        "link":    {"num_links": 1,   "idle_power": 5,     "energy_per_bit": 4.0},
        "nic":     {"num_nics": 1,    "idle_power": 20},
        "storage": {"num_devices": 2, "idle_power": 5}
      }
    }
  ]
}
```

The `npu.<hardware>` key looks up power coefficients by the
instance's `hardware` field, so multi-hardware clusters list one
entry per hardware type.

For the field-by-field schema (`base_node_power`, `idle_power`,
`standby_duration`, `energy_per_bit`, ...), see
[Cluster config → power](/docs/reference/cluster-config).

## Run

```bash
python -m serving \
  --cluster-config 'configs/cluster/single_node_power_instance.json' \
  --dtype bfloat16 --block-size 16 \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/power_run.csv' \
  --log-interval 1.0
```

No new CLI flag is needed. The presence of the `power:` block in
the cluster config is the trigger; remove the block for a baseline
run that doesn't track power.

## Expected output

The throughput log gains a `power=` field (in watts):

```text
[INFO] step=42 batch=8 prompt_t=1.2k tok/s decode_t=420 tok/s
       npu_mem=88.4 GB power=712 W
[INFO] step=43 batch=8 prompt_t=1.1k tok/s decode_t=440 tok/s
       npu_mem=88.4 GB power=698 W
```

`power` is the **instantaneous** total node power summed across
NPU / CPU / DRAM / link / NIC / storage / base.

When the run ends, the simulator prints a per-component energy
breakdown:

```text
─────── Power summary (node 0) ───────
   NPU active     :   12,453 J  (78%)
   NPU standby    :    1,012 J   (6%)
   NPU idle       :       89 J   (1%)
   CPU            :    1,233 J   (8%)
   DRAM           :      442 J   (3%)
   Link           :      388 J   (2%)
   Base + NIC + storage : 332 J  (2%)
   ─────────────────────────────────
   Total energy   :   15,949 J
```

The breakdown is the actionable output. A run dominated by
`NPU active` is compute-bound; one with significant `NPU idle` is
under-utilized; one with disproportionate `Link` energy is
ALLREDUCE-bound (worth checking when `tp_size > 1`).

## What's interesting

- **Throughput vs. wattage trade-offs.** Bumping `--max-num-seqs`
  raises throughput and `NPU active` / `standby` time together, but
  the slope differs by workload — energy-per-token improves on
  decode-heavy loads and degrades on prefill-heavy ones.
- **Standby vs. idle gap.** `standby_duration` (ns after a kernel
  finishes) determines how often the NPU drops back to
  `idle_power`. Bursty workloads spend more time in `idle`;
  steady-state workloads stay in `standby` / `active`.
  `NPU idle > NPU standby` usually means the workload doesn't
  saturate the GPU.
- **Base-node power is constant.** The host-side draw
  (`base_node_power`) doesn't depend on what the simulator is
  doing; it's the always-on overhead that energy-efficiency
  comparisons need to factor in.

## Related examples

- **[Sub-batch interleaving](./sub-batch-interleaving)** — pairs
  cleanly with the power model. Overlapping PIM attention with GPU
  compute changes both throughput and the energy breakdown.
- **[CXL memory](../memory-tiers/cxl-memory)** — adding a `cxl_mem`
  device and per-device placement rules adds a `cxl_mem=...` field
  to the throughput log; the energy summary then includes CXL
  transfer energy.

## Where to learn more

- **[Simulator → Power model](/docs/simulator/specialized/power-model)**:
  per-component math, NPU state machine, and how
  `standby_duration` factors in.
- The implementation lives in `serving/core/power_model.py`.
