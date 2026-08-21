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
| `num_devices` | int | optional (default `1`) | Number of CXL devices (`cxl:0` through `cxl:N-1`) |

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

Three rules the table cannot show:

- **Power modeling is all-or-nothing across the cluster.** If *any*
  node omits `power`, `config_builder.py` disables power modeling for
  **every** node, silently. There is no per-node opt-in.
- **`npu` needs one entry per distinct `hardware` on that node.** The
  key is the instance's `hardware` string, and every instance on the
  node must find its own key. A heterogeneous node needs one block per
  hardware label.
- **`dram.dimm_size` and `dram.idle_power` become optional under
  `--enable-attn-offloading`.** With PIM on, both are supplied by the
  PIM config instead (`dimm_size` from the derived per-channel
  capacity, `idle_power` from the INI's `idle_power`), and only
  `dram.energy_per_bit` stays required. See
  **[PIM config](./pim-config)**.

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
| `dp_group` | string \| null | `null` | Group ID. Instances with the same string form one data-parallel group, wave-synchronized per iteration; for MoE they also share experts across the group |

**Constraints:**

- `num_npus == tp_size * pp_size` (always)
- `pp_size <= num_hidden_layers`: pipeline stages are cut on
  transformer-block boundaries, so a stage cannot be empty
- Without `dp_group`: `ep_size <= tp_size`
- For MoE: `ep_size` must divide `num_local_experts`
- All members of a `dp_group` must agree on `tp_size`, `pp_size` and `ep_size`
- For a **MoE** model in a `dp_group`, `ep_size` is the total EP degree across
  the group: it must be divisible by `dp_group_size`, and
  `ep_size / dp_group_size <= tp_size`. For a **dense** model `ep_size` is not a
  degree to spread over the group, so neither check applies — plain data
  parallelism over a dense model is supported (`single_node_dp_instance.json`)

### Runtime overrides (optional)

Exactly **14** of the `python -m serving` flags can be re-specified per
instance, letting one cluster run heterogeneous instances — a prefill
instance with a tight `max_num_seqs` next to a decode instance with a
wide one, or two instances at different `mem_util`. Every one of them
is resolved in `_build_instance_runtime_configs()` in
`serving/__main__.py`.

**Precedence** is one level deep, no merging:

```
instances[i].<field>   >   --<field> on the CLI   >   built-in default
```

The lookup is literally `instance.get("<field>", args.<field>)`, so a
field present in the cluster config wins for that instance and every
other instance keeps the CLI value.

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
| `npu_mem.mem_util` | float | `--npu-memory-utilization` | Fraction of `npu_mem.mem_size` usable for weights plus KV cache. KV capacity is `mem_size * mem_util - model weight`, divided into `block_size` blocks |
| `reserve_full_isl` | bool | `--reserve-full-isl` | Admit only if the request's whole sequence fits, not just its first chunk |
| `enable_local_offloading` | bool | `--enable-local-offloading` | Emit graph conversion with local offloading for this instance |
| `enable_attn_offloading` | bool | `--enable-attn-offloading` | Emit PIM attention offload for this instance |
| `enable_sub_batch_interleaving` | bool | `--enable-sub-batch-interleaving` | Enable sub-batch interleaving for this instance |
| `enable_block_copy` | bool | `--enable-block-copy` | Reuse one block trace across repeated transformer blocks |

#### `npu_mem.mem_util` is the one nested override

The other 13 are plain keys on the instance object. `mem_util` sits
**inside** the `npu_mem` block, because its only job is to scale
`mem_size` and it follows that block's `mem_*` naming:

```json
{
  "model_name": "meta-llama/Llama-3.1-8B",
  "hardware": "RTXPRO6000",
  "npu_mem": {"mem_size": 96, "mem_bw": 1597, "mem_latency": 0, "mem_util": 0.8},
  "tp_size": 1,
  "pd_type": null,
  "max_num_seqs": 64
}
```

It must be a number in `(0, 1]` — it is a *fraction*, so `0.9`, never
`90`. Anything else raises at startup rather than being clamped.

#### `0` means unlimited, with one caveat

`max_num_seqs` and `max_num_batched_tokens` route through a
`_runtime_limit()` helper that maps `0` to infinity:

- `max_num_seqs: 0` — genuinely unbounded concurrency.
- `max_num_batched_tokens: 0` — **not** unbounded in practice. The
  scheduler then computes
  `min(max_num_batched_tokens, max_position_embeddings)`, so the
  effective budget becomes the model's context length from
  **[Model config](./model-config)**. On
  `microsoft/Phi-mini-MoE-instruct` that is 4096, not infinity.

No other numeric override treats `0` specially:
`long_prefill_token_threshold: 0` means *disabled* (no per-request
cap), matching the CLI flag, and `block_size: 0` is simply invalid.

#### `dtype` resolution is three levels, not two

`dtype` is the one override with a fallback below the CLI:

```
instances[i].dtype   >   --dtype   >   model config torch_dtype   >   bfloat16
```

The resolved value must be one of `float16` / `bfloat16` / `float32` /
`fp8` / `int8`, and it selects the profile **variant folder**, so the
matching `profiler/perf/<hardware>/<model>/<variant>/tp<N>/` bundle has
to exist. `kv_cache_dtype` is validated per instance too — only `auto`
or `fp8`.

#### Validation gates

Two combinations are rejected at config-load time, per instance:

| Rejected | Error | Why |
| --- | --- | --- |
| `enable_sub_batch_interleaving: true` without `enable_attn_offloading: true` | `RuntimeError` | There is nothing to overlap the NPU sub-batch against |
| `enable_sub_batch_interleaving: true` with `pp_size > 1` | `RuntimeError` | An interleaved trace leaves both sub-batches mid-block at every stage edge, so a pipeline stage has no single hidden state to pass on |

Both gates read the *effective* values, so inheriting
`--enable-sub-batch-interleaving` from the CLI onto an instance that
locally disables `enable_attn_offloading` fails just the same.

#### Flags that are **not** per-instance

The remaining 15 CLI flags are cluster-wide. Setting them inside an
instance object is silently ignored — nothing reads the key:

| Scope | Flags |
| --- | --- |
| Cluster / backend | `--cluster-config`, `--network-backend` |
| Router (cross-instance by definition) | `--request-routing-policy`, `--expert-routing-policy` |
| Shared lower KV tier | `--enable-prefix-sharing`, `--prefix-storage` |
| Workload (one per run) | `--dataset`, `--num-reqs`, `--skip-prefill` |
| Run plumbing | `--output`, `--run-id`, `--inputs-root`, `--cleanup-inputs`, `--log-interval`, `--log-level` |

#### Worked example

`configs/cluster/single_node_pd_per_instance_config.json` splits
prefill and decode with different scheduler limits, and
`configs/cluster/single_node_heterogeneous.json` pairs them with
different chunked-prefill settings. See
**[Examples → Cluster config explained](/docs/examples/cluster-config-explained#per-instance-runtime-overrides)**
for the annotated walkthrough.

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

Structural, in `config_builder.py`:

- `num_nodes == len(nodes)` and per-node `num_instances == len(instances)`.
- `link_bw` and `link_latency` must both be present at top level.
- Every instance needs `model_name`, `hardware`, `npu_mem`, and
  `pd_type`; `npu_mem` needs `mem_size`, `mem_bw`, `mem_latency`. Same
  three keys are required in `cpu_mem` and, if present, `cxl_mem`.
- `num_npus == tp_size * pp_size`, and `pp_size <= num_hidden_layers`.
- `dp_group` must be a string or `null`, and all instances sharing one
  `dp_group` must agree on `tp_size`, `pp_size` **and** `ep_size`.
- Hardware folder must exist at
  `profiler/perf/<hardware>/<model_name>/<variant>/tp<tp_size>/`.

Memory, in `memory_model.py`, evaluated **per GPU** (weights are
already sharded by `tp_size` / `ep_size`):

- `weight_per_gpu <= npu_mem.mem_size`, ignoring `mem_util`. Failing
  this raises `Model size ...GB exceeds total NPU memory ...GB`.
- `npu_mem.mem_size * mem_util - weight_per_gpu` must leave room for at
  least **one** KV block of `block_size` tokens. This is the tighter of
  the two and the one `mem_util` actually gates: dropping `mem_util`
  far enough fails here, with a message naming the requested bytes,
  the weight bytes, and the shortfall.

Runtime, per instance, in `serving/__main__.py`:

- `dtype` must be one of the five supported values and
  `kv_cache_dtype` one of `auto` / `fp8`.
- `npu_mem.mem_util` must be a number in `(0, 1]`.
- The two sub-batch-interleaving gates above.

## What's next

- **[Model config](./model-config)**: schema for the file
  `model_name` resolves to.
- **[PIM config](./pim-config)**: schema for the file
  `cpu_mem.pim_config` resolves to.
