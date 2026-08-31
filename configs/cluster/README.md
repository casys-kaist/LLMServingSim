# configs/cluster

This directory contains cluster configuration files that define the hardware topology,
instance layout, memory hierarchy, and interconnect parameters for LLMServingSim.

Pass a config file to `python -m serving` via `--cluster-config configs/cluster/{name}.json`.

## Configuration format

```json
{
  "num_nodes": 1,
  "link_bw": 16,
  "link_latency": 0,
  "nodes": [
    {
      "num_instances": 1,
      "cpu_mem": {
        "mem_size": 512,
        "mem_bw": 256,
        "mem_latency": 0
      },
      "instances": [
        {
          "model_name": "Qwen/Qwen3-32B",
          "hardware": "RTXPRO6000",
          "npu_mem": {
            "mem_size": 96,
            "mem_bw": 1597,
            "mem_latency": 0
          },
          "num_npus": 2,
          "tp_size": 2,
          "pd_type": null
        }
      ]
    }
  ]
}
```

### Top-level fields

| Field | Type | Description |
| --- | --- | --- |
| `num_nodes` | Integer | Number of nodes in the cluster |
| `link_bw` | Float or Array<Float> | ASTRA-Sim topology link bandwidth in GB/s. A scalar is broadcast to all topology dimensions; an array must match the final `npus_count` rank |
| `link_latency` | Float or Array<Float> | ASTRA-Sim topology link latency in ns. A scalar is broadcast to all topology dimensions; an array must match the final `npus_count` rank |

### Per-node fields

| Field | Type | Description |
| --- | --- | --- |
| `num_instances` | Integer | Number of instances on this node |
| `cpu_mem.mem_size` | Float | CPU memory capacity in GB |
| `cpu_mem.mem_bw` | Float | CPU memory bandwidth in GB/s |
| `cpu_mem.mem_latency` | Float | CPU memory latency in ns |

### Per-instance fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `model_name` | String | Yes | HuggingFace model identifier (must match `configs/model/`) |
| `hardware` | String | Yes | Hardware name matching `profiler/perf/{hardware}/` |
| `npu_mem` | Object | Yes | NPU memory config (`mem_size` in GB, `mem_bw` in GB/s, `mem_latency` in ns; optional `mem_util`, see below) |
| `pd_type` | String/null | Yes | `"prefill"`, `"decode"`, or `null` for combined |
| `num_npus` | Integer | * | Total GPUs for this instance (inferred from `tp_size * pp_size` if omitted) |
| `tp_size` | Integer | * | Tensor parallel degree (inferred from `num_npus // pp_size` if omitted) |
| `pp_size` | Integer | No | Pipeline parallel degree (default: 1) |
| `ep_size` | Integer | No | Expert parallel degree (default: `tp_size` for MoE, 1 for dense) |
| `dp_group` | String/null | No | DP group ID. Instances with the same string form one data-parallel group, wave-synchronized per iteration; for MoE they also share experts across the group |
| `max_num_seqs` | Integer | No | Per-instance override for `--max-num-seqs` (`0` = unlimited) |
| `max_num_batched_tokens` | Integer | No | Per-instance override for `--max-num-batched-tokens` (`0` = capped at the model's `max_position_embeddings`, see below) |
| `long_prefill_token_threshold` | Integer | No | Per-instance override for `--long-prefill-token-threshold` |
| `block_size` | Integer | No | Per-instance override for `--block-size` |
| `enable_chunked_prefill` | Boolean | No | Per-instance override for `--enable-chunked-prefill` |
| `enable_prefix_caching` | Boolean | No | Per-instance override for `--enable-prefix-caching` |
| `npu_mem.mem_util` | Float | No | Fraction of `npu_mem.mem_size` an instance may use for weights plus KV cache. Per-instance override for `--npu-memory-utilization` (default `0.9`) |
| `reserve_full_isl` | Boolean | No | Admit only if the request's whole sequence fits, not just its first chunk. Per-instance override for `--reserve-full-isl` (default on) |
| `enable_local_offloading` | Boolean | No | Per-instance override for `--enable-local-offloading` |
| `enable_attn_offloading` | Boolean | No | Per-instance override for `--enable-attn-offloading` |
| `enable_sub_batch_interleaving` | Boolean | No | Per-instance override for `--enable-sub-batch-interleaving` |
| `enable_block_copy` | Boolean | No | Per-instance override for `--enable-block-copy`. Reuses a block's built rows across layers of the same shape; a generation-speed knob, not a trace-content one |

\* At least one of `num_npus` or `tp_size` must be provided. The other is inferred.

### Per-instance runtime overrides

The 14 runtime fields listed above (`max_num_seqs`, `max_num_batched_tokens`, etc.) support **per-instance overrides** in the cluster config. This enables heterogeneous deployments where different instances in the same cluster use different scheduler limits.

**Precedence rule:**
```
per-instance value (from cluster config) > global CLI value (from --flag)
```

For each field, the runtime reads `instance.get("<field>", args.<field>)` — if the field is present in the cluster config, it takes precedence; otherwise the global CLI value is used.

**Unlimited semantics:**
Setting either batching limit to `0` maps it to infinity via the `_runtime_limit` helper:
- `max_num_seqs: 0` → no limit on concurrent sequences
- `max_num_batched_tokens: 0` → **not** actually unbounded. The scheduler then takes
  `min(max_num_batched_tokens, max_position_embeddings)`, so the effective budget is the
  model's context length from its `configs/model/` entry (4096 on
  `microsoft/Phi-mini-MoE-instruct`, not infinity).

`long_prefill_token_threshold: 0` means *disabled* (no per-request cap), matching the CLI
flag — it is not an "unlimited" sentinel.

**Dtypes are not overridable.** There is no `dtype` or `kv_cache_dtype` field
and no CLI flag below it: a modern checkpoint carries five cache dtypes decided
in four different places, so every one is read from the model config. The
weight dtype still picks the profile variant folder, so
`profiler/perf/{hardware}/{model}/{variant}/tp{N}/` must exist — but the folder
name follows from the config alone, so a missing bundle means *profile this
model*. Serving two precisions of one model is two **model configs**.

> **Calibrate `mem_util` when the KV cache saturates.** It sizes the KV cache,
> and that only affects results once a run actually fills it — below the
> ceiling nothing is preempted and the capacity is invisible, so the default
> `0.9` is fine. When a run does hit the ceiling the default is wrong: the
> simulator does not model vLLM's activation peak or CUDA context, so `0.9`
> here buys more KV cache than `0.9` does in vLLM. Read
> `kv_cache.num_gpu_blocks` from the bench run's `meta.json` and pick the
> `mem_util` whose startup "KV Cache Initialization" line reports the same
> block count. On the bundled RTX 4090 example that is `0.833919`, and it is
> the difference between -20.7% and +0.6% on TTFT mean.

**`npu_mem.mem_util` is the one nested override.** The other 13 are plain instance keys;
`mem_util` lives inside `npu_mem` because its only job is to scale `mem_size`. It must be a
number in `(0, 1]` — a fraction, so `0.9`, never `90`.

**Validation gates:**
- `enable_sub_batch_interleaving: true` requires `enable_attn_offloading: true`
- `enable_sub_batch_interleaving: true` requires `pp_size == 1`: an interleaved trace leaves
  both sub-batches mid-block at every stage edge, so a pipeline stage has no single hidden
  state to pass on

Both are enforced at config load time, against the *effective* values — so inheriting
`--enable-sub-batch-interleaving` from the CLI onto an instance that locally sets
`enable_attn_offloading: false` fails just the same.

**Example: heterogeneous P/D instances**

`single_node_pd_per_instance_config.json` gives the prefill instance a tight
concurrency and a large token budget, and the decode instance the reverse
(hardware fields elided):

```json
{
  "instances": [
    {
      "pd_type": "prefill",
      "tp_size": 1,
      "max_num_seqs": 32,
      "max_num_batched_tokens": 8192,
      "long_prefill_token_threshold": 2048,
      "enable_chunked_prefill": true,
      "block_size": 16,
      "dtype": "bfloat16",
      "kv_cache_dtype": "auto"
    },
    {
      "pd_type": "decode",
      "tp_size": 1,
      "max_num_seqs": 256,
      "max_num_batched_tokens": 256,
      "enable_chunked_prefill": true,
      "block_size": 16,
      "dtype": "bfloat16",
      "kv_cache_dtype": "auto"
    }
  ]
}
```

`single_node_heterogeneous.json` is the Qwen3-32B / TP=2 variant, and it does
use `max_num_batched_tokens: 0` plus `enable_chunked_prefill: false` on the
decode instance.

### Parallelism rules:
- `num_npus = tp_size * pp_size`
- TP and EP share the same GPUs: non-MoE layers use TP (ALLREDUCE), MoE layers use EP (ALLTOALL)
- DP is achieved via multiple instances with the same `dp_group`
- Without `dp_group`: `ep_size <= tp_size`
- For MoE models: `ep_size` must divide `num_local_experts`
- All members of a `dp_group` must agree on `tp_size`, `pp_size` and `ep_size`
- For a **MoE** model in a `dp_group`, `ep_size` is the total EP degree across
  the group, so it must be divisible by the group size, and
  `ep_size / dp_group_size <= tp_size`. For a **dense** model `ep_size` is not a
  degree to spread over the group (there are no experts to shard), so neither
  check applies and plain data parallelism works — see
  `single_node_dp_instance.json`

### DP topology:
When `dp_group` is set, `config_builder.py` generates a multi-dimensional
ASTRA-Sim topology, innermost dimension first: `[tp_size, dp_group_size]`, or
`[tp_size, pp_size, dp_group_size]` when `pp_size > 1`. This mirrors vLLM's rank
layout (`all_ranks.reshape(-1, dp, pp, pcp, tp)`); `pp_size` is omitted when it
is 1 so existing DP+TP configs keep their 2-D topology. Collectives are routed
per dimension via `involved_dim`: TP-ALLREDUCE on the TP dim only, and EP on the
DP and TP dims but never PP (vLLM's EP group pins the pipeline stage). All
instances in a DP group share one ASTRA-Sim process with wave-synchronized
scheduling. MoE expert weights are sharded by `ep_size` (each instance holds
`num_local_experts // ep_size` experts).

### Optional fields

| Field | Scope | Type | Description |
| --- | --- | --- | --- |
| `placement` | instance | Object | Per-layer placement rules for weights and KV cache location |
| `power` | node | Object | Power model config (NPU idle/standby/active, CPU, DRAM, link, NIC, storage) |
| `cxl_mem` | top-level | Object | CXL memory expansion parameters (`mem_size`, `mem_bw`, `mem_latency`, `num_devices`) |
| `pim_config` | node cpu_mem | String | Name of a PIM device config in `configs/pim/` |

## Provided configurations

| File | Description |
| --- | --- |
| `single_node_single_instance.json` | Single node, Llama-3.1-8B on one GPU (the default config) |
| `single_node_single_instance_H100.json` | Single node, Llama-3.1-70B on H100 with TP=4 |
| `single_node_multi_instance.json` | Single node, two instances |
| `single_node_pd_instance.json` | Single node with prefill/decode disaggregation |
| `single_node_pd_per_instance_config.json` | P/D disaggregation with prefill/decode-specific runtime limits |
| `single_node_moe_single_instance.json` | Single node, Qwen3-MoE with TP=2 EP=2 |
| `single_node_moe_dp_ep_instance.json` | Single node, two MoE instances in one DP group sharing experts over EP=2 |
| `single_node_dp_instance.json` | Single node, DP=2 x TP=2 dense model (4 GPUs) |
| `rtx4090_single_instance.json` | RTX 4090 (24 GB), Llama-3.1-8B TP=1. `mem_util` calibrated to the validated bench run |
| `rtx4090_tp2_instance.json` | Two RTX 4090s as TP=2, Llama-3.1-8B. A template, not runnable as shipped: only `tp1` is profiled for RTX4090, so it raises `FileNotFoundError` until you profile the card with `TP_DEGREES=2`. `mem_util` is left at the default because there is no validated run |
| `rtx4090_multi_instance.json` | Two independent TP=1 RTX 4090 instances behind the router |
| `single_node_moe_dp_tp_instance.json` | Single node, DP=2 x TP=2 MoE (EP=2, 4 GPUs) |
| `single_node_moe_dp_pp_instance.json` | Single node, DP=2 x PP=2 MoE (EP=2, 4 GPUs) |
| `single_node_moe_dp_tp_pp_instance.json` | Single node, DP=2 x TP=2 x PP=2 MoE (EP=4, 8 GPUs) |
| `single_node_4_instance_2TP.json` | Single node, four TP=2 instances |
| `single_node_heterogeneous.json` | Single node, P/D pair with different per-instance runtime settings |
| `single_node_moe_multi_instance.json` | Single node, two MoE instances |
| `single_node_moe_pd_instance.json` | Single node, MoE with P/D disaggregation |
| `single_node_cxl_instance.json` | Single node with CXL memory expansion |
| `single_node_memory_instance.json` | Single node with weight/KV placement control |
| `single_node_pim_instance.json` | Single node with PIM-enabled memory + power model |
| `single_node_power_instance.json` | Single node with power modeling enabled |
| `single_node_pp_instance.json` | Single node, 4 GPUs as `pp=4` (one pipeline stage each) |
| `single_node_tp_pp_instance.json` | Single node, 4 GPUs as `tp=2 x pp=2` |
| `single_node_moe_pp_instance.json` | Single node, MoE on 4 GPUs as `tp=2 x pp=2` with `ep=2` |
| `dual_node_multi_instance.json` | Two nodes, two instances each |
| `dual_node_moe_dp_ep_intra_inter_instance.json` | Two-node MoE DP+EP example with per-dimension intra/inter link settings |
