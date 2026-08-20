---
sidebar_position: 1
title: Cluster config
---

# Cluster config schema

Formal field-by-field schema for the JSON file passed via
`--cluster-config`. For a guided walkthrough with examples, see
**[Examples → Cluster config explained](/docs/examples/cluster-config-explained)**.
This page is the **lookup reference**: every field, every type,
every default.

## File location

Configs live at `configs/cluster/<name>.json`. The simulator reads
the file once at startup and `serving/core/config_builder.py`
generates derived ASTRA-Sim input files (`network.yml`,
`system.json`, `memory_expansion.json`).

## Top-level

```json
{
  "num_nodes": 1,
  "link_bw": 16,
  "link_latency": 20000,
  "nodes": [...],
  "cxl_mem": {...}
}
```

| Field | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `num_nodes` | int | ✓ |  | Number of physical nodes in the cluster |
| `link_bw` | float or float[] | ✓ |  | ASTRA-Sim topology link bandwidth in **GB/s**. Scalars apply to every topology dimension; arrays must match the final `network.yml::npus_count` rank |
| `link_latency` | float or float[] | ✓ |  | ASTRA-Sim topology link latency in **ns**. Scalars apply to every topology dimension; arrays must match the final `network.yml::npus_count` rank |
| `nodes` | array | ✓ |  | Length must equal `num_nodes` |
| `cxl_mem` | object | optional | absent | CXL memory expansion (see below) |

Example: if `network.yml` will end up with `npus_count: [4, 2]`, you may set
`link_bw: [900, 100]` and `link_latency: [0, 20000]` to assign different
bandwidth/latency per topology dimension.

## `cxl_mem` (top-level, optional)

```json
"cxl_mem": {
  "mem_size": 1024,
  "mem_bw": 60,
  "mem_latency": 250,
  "num_devices": 4
}
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `mem_size` | float | ✓ | Capacity per device in **GB** |
| `mem_bw` | float | ✓ | Bandwidth per device in **GB/s** |
| `mem_latency` | float | ✓ | Access latency in **ns** |
| `num_devices` | int | ✓ | Number of CXL devices (`cxl:0` through `cxl:N-1`) |

When present, instances can reference `cxl:N` in their `placement`
field.

## Per-node (`nodes[i]`)

```json
{
  "num_instances": 2,
  "cpu_mem": {"mem_size": 512, "mem_bw": 256, "mem_latency": 0},
  "instances": [...],
  "power": {...},
  "cpu_mem.pim_config": "DDR4_8GB_3200_pim"
}
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `num_instances` | int | ✓ | Number of serving instances on this node |
| `cpu_mem` | object | ✓ | Host CPU memory config (see below) |
| `instances` | array | ✓ | Length must equal `num_instances` |
| `power` | object | optional | Power model config (see below) |

### `cpu_mem`

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `mem_size` | float | ✓ | Host CPU memory capacity in **GB** |
| `mem_bw` | float | ✓ | CPU memory bandwidth in **GB/s** |
| `mem_latency` | float | ✓ | CPU memory latency in **ns** |
| `pim_config` | string | optional | Name of a PIM device config in `configs/pim/`. See **[PIM config](./pim-config)** |

### `power` (optional)

Enables the power model on this node. See **[Examples → Power
modeling](/docs/examples/advanced/power-modeling)** for the full
schema. Top-level structure:

```json
"power": {
  "base_node_power": 60,
  "npu": {"<hardware>": {...}},
  "cpu": {...},
  "dram": {...},
  "link": {...},
  "nic": {...},
  "storage": {...}
}
```

| Sub-field | Required | Description |
| --- | --- | --- |
| `base_node_power` | ✓ | Always-on host platform power in **W** |
| `npu.<hardware>.idle_power` | ✓ | NPU idle wattage |
| `npu.<hardware>.standby_power` | ✓ | NPU post-compute standby wattage |
| `npu.<hardware>.active_power` | ✓ | NPU active compute wattage |
| `npu.<hardware>.standby_duration` | ✓ | Time to stay in standby after compute, in **ns** |
| `cpu.idle_power`, `cpu.active_power`, `cpu.util` | ✓ | CPU baseline + utilization fraction |
| `dram.dimm_size`, `dram.idle_power`, `dram.energy_per_bit` | ✓ | DIMM size, idle power, per-bit energy |
| `link.num_links`, `link.idle_power`, `link.energy_per_bit` | ✓ | Network link power |
| `nic.num_nics`, `nic.idle_power` | ✓ | NIC count and baseline |
| `storage.num_devices`, `storage.idle_power` | ✓ | Storage devices |

## Per-instance (`instances[i]`)

```json
{
  "model_name": "Qwen/Qwen3-32B",
  "hardware": "RTXPRO6000",
  "npu_mem": {"mem_size": 96, "mem_bw": 1597, "mem_latency": 0},
  "num_npus": 2,
  "tp_size": 2,
  "pp_size": 1,
  "ep_size": 1,
  "dp_group": null,
  "pd_type": null,
  "max_num_seqs": 128,
  "max_num_batched_tokens": 2048,
  "placement": {...}
}
```

### Required fields

| Field | Type | Description |
| --- | --- | --- |
| `model_name` | string | HF id. Must match a config at `configs/model/<model_name>.json` (see **[Model config](./model-config)**) |
| `hardware` | string | Hardware label. Must match `profiler/perf/<hardware>/` |
| `npu_mem.mem_size` | float | Per-GPU NPU memory in **GB** |
| `npu_mem.mem_bw` | float | Per-GPU NPU memory bandwidth in **GB/s** |
| `npu_mem.mem_latency` | float | Per-GPU NPU memory latency in **ns** |
| `pd_type` | string \| null | `"prefill"`, `"decode"`, or `null` (combined) |

### Parallelism (at least one of `num_npus` / `tp_size`)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `num_npus` | int | inferred from `tp_size * pp_size` | Total GPUs for this instance |
| `tp_size` | int | inferred from `num_npus // pp_size` | Tensor-parallel degree |
| `pp_size` | int | `1` | Pipeline-parallel degree |
| `ep_size` | int | `tp_size` (MoE) / `1` (dense) | Expert-parallel degree |
| `dp_group` | string \| null | `null` | Group ID. Instances with the same string share experts via cross-instance ALLTOALL |

**Constraints:**

- `num_npus == tp_size * pp_size` (always)
- Without `dp_group`: `ep_size <= tp_size`
- For MoE: `ep_size` must divide `num_local_experts`

### Runtime overrides (optional)

These fields override the matching `python -m serving` CLI flag for this
instance only. Omitted fields keep the CLI value; for `dtype`, an omitted CLI
value still falls back to the model config's `torch_dtype`.

| Field | Type | CLI fallback | Description |
| --- | --- | --- | --- |
| `max_num_seqs` | int | `--max-num-seqs` | Max active sequences for this instance. `0` means unlimited |
| `max_num_batched_tokens` | int | `--max-num-batched-tokens` | Per-iteration token budget for this instance. `0` means unlimited |
| `long_prefill_token_threshold` | int | `--long-prefill-token-threshold` | Per-request chunk cap for chunked prefill |
| `block_size` | int | `--block-size` | KV-cache block size in tokens |
| `dtype` | string | `--dtype` | Weight/profile dtype for this instance |
| `kv_cache_dtype` | string | `--kv-cache-dtype` | KV-cache dtype for memory accounting and profile variant selection |
| `enable_chunked_prefill` | bool | `--enable-chunked-prefill` | Enable chunked prefill in this instance's scheduler |
| `enable_prefix_caching` | bool | `--enable-prefix-caching` | Enable this instance's local prefix cache |
| `npu_mem.mem_util` | float | `--npu-memory-utilization` | Fraction of `npu_mem.mem_size` usable for weights plus KV cache. KV capacity is `mem_size * mem_util - model weight`, divided into `--block-size` blocks |
| `reserve_full_isl` | bool | `--reserve-full-isl` | Admit only if the request's whole sequence fits, not just its first chunk |
| `enable_local_offloading` | bool | `--enable-local-offloading` | Emit graph conversion with local offloading for this instance |
| `enable_attn_offloading` | bool | `--enable-attn-offloading` | Emit PIM attention offload for this instance |
| `enable_sub_batch_interleaving` | bool | `--enable-sub-batch-interleaving` | Enable sub-batch interleaving for this instance |
| `enable_block_copy` | bool | `--enable-block-copy` | Reuse one block trace across repeated transformer blocks |

### `placement` (optional)

Per-layer / per-block weight + KV-cache placement rules. See
**[Examples → CXL extended memory](/docs/examples/memory-tiers/cxl-memory)**
for a worked example.

```json
"placement": {
  "default": {"weights": "npu", "kv_loc": "npu", "kv_evict_loc": "cpu"},
  "blocks": [
    {"blocks": "0-3", "weights": "cxl:0", "kv_loc": "npu", "kv_evict_loc": "cpu"}
  ],
  "layers": {
    "embedding": {"weights": "cxl:1", "kv_loc": "npu", "kv_evict_loc": "cpu"}
  }
}
```

| Sub-field | Type | Required | Description |
| --- | --- | --- | --- |
| `default` | object | ✓ | Catch-all rule for layers / blocks not in `blocks` or `layers` |
| `blocks` | array | optional | Per-decoder-block-range overrides |
| `layers` | object | optional | Per-named-layer overrides |

Each rule object has three string fields:

| Field | Allowed values | Description |
| --- | --- | --- |
| `weights` | `npu` / `cpu` / `cxl:<id>` | Where this layer's weights live |
| `kv_loc` | `npu` / `cpu` / `cxl:<id>` | Where active KV blocks live (attention layers only) |
| `kv_evict_loc` | `npu` / `cpu` / `cxl:<id>` | Where evicted KV blocks spill |

`blocks` strings are dash-and-comma-separated ranges:
`"0-3"`, `"4-7"`, `"8,9,10"`, `"11-23"`. Layer-name keys must match
canonical layer names from the architecture YAML.

## Validation rules

- `num_nodes == len(nodes)` and per-node `num_instances == len(instances)`.
- Per-instance `weight_per_gpu * num_npus <= npu_mem.mem_size *
  num_npus` (otherwise startup OOM).
- Hardware folder must exist at `profiler/perf/<hardware>/<model_name>/<variant>/tp<tp_size>/`.
- `dp_group` must be a valid string or `null`.
- All instances within the same `dp_group` must share the same
  `ep_size` and `tp_size`.

## What's next

- **[Model config](./model-config)**: schema for the file
  `model_name` resolves to.
- **[PIM config](./pim-config)**: schema for the file
  `cpu_mem.pim_config` resolves to.
