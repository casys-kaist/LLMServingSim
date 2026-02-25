# cluster_config

This directory contains cluster configuration files that define the hardware topology,
instance layout, memory hierarchy, and interconnect parameters for LLMServingSim.

Pass a config file to `main.py` via `--cluster-config cluster_config/{name}.json`.

## Configuration format

```json
{
  "num_nodes": 1,
  "link_bw": 112,
  "link_latency": 0,
  "nodes": [
    {
      "num_instances": 1,
      "cpu_mem": {
        "mem_size": 128,
        "mem_bw": 256,
        "mem_latency": 0
      },
      "instances": [
        {
          "model_name": "meta-llama/Llama-3.1-8B",
          "hardware": "A6000",
          "npu_mem": {
            "mem_size": 40,
            "mem_bw": 768,
            "mem_latency": 0
          },
          "npu_num": 1,
          "npu_group": 1,
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
| `link_bw` | Float | Inter-node link bandwidth in GB/s |
| `link_latency` | Float | Inter-node link latency in ns |

### Per-node fields

| Field | Type | Description |
| --- | --- | --- |
| `num_instances` | Integer | Number of instances on this node |
| `cpu_mem.mem_size` | Float | CPU memory capacity in GB |
| `cpu_mem.mem_bw` | Float | CPU memory bandwidth in GB/s |
| `cpu_mem.mem_latency` | Float | CPU memory latency in ns |

### Per-instance fields

| Field | Type | Description |
| --- | --- | --- |
| `model_name` | String | HuggingFace model identifier |
| `hardware` | String | Hardware target matching a profile in `llm_profile/perf_models/` |
| `npu_mem.mem_size` | Float | NPU memory capacity in GB |
| `npu_mem.mem_bw` | Float | NPU memory bandwidth in GB/s |
| `npu_mem.mem_latency` | Float | NPU memory latency in ns |
| `npu_num` | Integer | Number of NPUs in this instance |
| `npu_group` | Integer | NPU group size for tensor parallelism |
| `pd_type` | String or null | `"prefill"`, `"decode"`, or `null` for combined |

### Optional per-instance fields

| Field | Type | Description |
| --- | --- | --- |
| `placement` | Object | Per-layer placement rules for weights, KV cache, and experts |
| `pim_config` | String | Path to a PIM device INI file in `pim_config/` |
| `power` | Object | Power configuration for the power model |
| `cxl_mem` | Object | CXL memory expansion parameters (`mem_size`, `mem_bw`, `mem_latency`) |

## Provided configurations

| File | Description |
| --- | --- |
| `single_node_single_instance.json` | Single node, single instance (default) |
| `single_node_single_instance_H100.json` | Single node, single instance on H100 |
| `single_node_multi_instance.json` | Single node, multiple instances |
| `single_node_pd_instance.json` | Single node with P/D disaggregation |
| `single_node_moe_single_instance.json` | Single node, single MoE instance |
| `single_node_moe_multi_instance.json` | Single node, multiple MoE instances |
| `single_node_moe_pd_instance.json` | Single node, MoE with P/D disaggregation |
| `single_node_cxl_instance.json` | Single node with CXL memory expansion |
| `single_node_pim_instance.json` | Single node with PIM-enabled memory |
| `single_node_power_instance.json` | Single node with power modeling enabled |
| `single_node_memory_instance.json` | Single node memory hierarchy configuration |
| `dual_node_multi_instance.json` | Dual node, multiple instances |
