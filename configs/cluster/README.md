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
| `hardware` | String | Yes | Hardware name matching `profiler/perf_models/{hardware}/` |
| `npu_mem` | Object | Yes | NPU memory config (`mem_size` in GB, `mem_bw` in GB/s, `mem_latency` in ns) |
| `pd_type` | String/null | Yes | `"prefill"`, `"decode"`, or `null` for combined |
| `num_npus` | Integer | * | Total GPUs for this instance (inferred from `tp_size * pp_size` if omitted) |
| `tp_size` | Integer | * | Tensor parallel degree (inferred from `num_npus // pp_size` if omitted) |
| `pp_size` | Integer | No | Pipeline parallel degree (default: 1) |
| `ep_size` | Integer | No | Expert parallel degree (default: `tp_size` for MoE, 1 for dense) |
| `dp_group` | String/null | No | DP group ID. Instances with the same string share experts via cross-instance ALLTOALL |
| `max_num_seqs` | Integer | No | Per-instance override for `--max-num-seqs` (`0` = unlimited) |
| `max_num_batched_tokens` | Integer | No | Per-instance override for `--max-num-batched-tokens` (`0` = unlimited) |
| `long_prefill_token_threshold` | Integer | No | Per-instance override for `--long-prefill-token-threshold` |
| `block_size` | Integer | No | Per-instance override for `--block-size` |
| `dtype` | String | No | Per-instance override for `--dtype` |
| `kv_cache_dtype` | String | No | Per-instance override for `--kv-cache-dtype` |
| `enable_chunked_prefill` | Boolean | No | Per-instance override for `--enable-chunked-prefill` |
| `enable_prefix_caching` | Boolean | No | Per-instance override for `--enable-prefix-caching` |
| `npu_mem.mem_util` | Float | No | Fraction of `npu_mem.mem_size` an instance may use for weights plus KV cache. Per-instance override for `--npu-memory-utilization` (default `0.9`) |
| `reserve_full_isl` | Boolean | No | Admit only if the request's whole sequence fits, not just its first chunk. Per-instance override for `--reserve-full-isl` (default on) |
| `enable_local_offloading` | Boolean | No | Per-instance override for `--enable-local-offloading` |
| `enable_attn_offloading` | Boolean | No | Per-instance override for `--enable-attn-offloading` |
| `enable_sub_batch_interleaving` | Boolean | No | Per-instance override for `--enable-sub-batch-interleaving` |
| `enable_block_copy` | Boolean | No | Per-instance override for `--enable-block-copy` |

\* At least one of `num_npus` or `tp_size` must be provided. The other is inferred.

### Per-instance runtime overrides

The 13 runtime fields listed above (`max_num_seqs`, `max_num_batched_tokens`, etc.) support **per-instance overrides** in the cluster config. This enables heterogeneous deployments where different instances in the same cluster use different scheduler limits.

**Precedence rule:**
```
per-instance value (from cluster config) > global CLI value (from --flag)
```

For each field, the runtime reads `instance.get("<field>", args.<field>)` — if the field is present in the cluster config, it takes precedence; otherwise the global CLI value is used.

**Unlimited semantics:**
Setting a numeric field to `0` means "unlimited" (via the `_runtime_limit` helper). For example:
- `max_num_seqs: 0` → no limit on concurrent sequences
- `max_num_batched_tokens: 0` → no limit on batched tokens

**Validation gates:**
- `enable_sub_batch_interleaving: true` requires `enable_attn_offloading: true` (enforced at config load time)

**Example: heterogeneous P/D instances**

See `single_node_pd_per_instance_config.json` for a concrete example where the prefill instance uses `max_num_seqs: 32` (tight concurrency) and the decode instance uses `max_num_seqs: 256` (high throughput):

```json
{
  "instances": [
    {
      "pd_type": "prefill",
      "max_num_seqs": 32,
      "max_num_batched_tokens": 8192,
      "enable_chunked_prefill": true
    },
    {
      "pd_type": "decode",
      "max_num_seqs": 256,
      "max_num_batched_tokens": 0,
      "enable_chunked_prefill": false
    }
  ]
}
```

### Parallelism rules:
- `num_npus = tp_size * pp_size`
- TP and EP share the same GPUs: non-MoE layers use TP (ALLREDUCE), MoE layers use EP (ALLTOALL)
- DP is achieved via multiple instances with the same `dp_group`
- Without `dp_group`: `ep_size <= tp_size`
- For MoE models: `ep_size` must divide `num_local_experts`

### DP+EP topology:
When `dp_group` is set, `config_builder.py` generates a 2D ASTRA-Sim topology
`[tp_size, dp_group_size]` with per-dimension collective routing via `involved_dim`.
ALLREDUCE (TP) runs on dim 0 only, ALLTOALL (EP) runs on dim 1. All instances in a
DP group share one ASTRA-Sim process with wave-synchronized scheduling. MoE expert
weights are sharded by `ep_size` (each instance holds `num_local_experts // ep_size` experts).

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
