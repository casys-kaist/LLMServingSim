---
sidebar_position: 1
title: CLI flags
---

# `python -m serving` CLI flags

Complete reference for every command-line flag accepted by
`python -m serving`. For the conceptual side of each flag (what it
*does* internally), see **[Simulator](/docs/simulator/architecture)**.

## Cluster topology

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--cluster-config` | path | `configs/cluster/single_node_single_instance.json` | Path to a cluster-config JSON. See **[Cluster config](./cluster-config)** |
| `--network-backend` | choice | `analytical` | Network simulation backend. `analytical` (fast) or `ns3` (detailed, WIP) |

## Batching and scheduling

These flags are deployment defaults. A cluster config can override the
matching runtime knobs per `instances[i]`; see
**[Cluster config](./cluster-config#runtime-overrides-optional)**.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--max-num-seqs` | int | `128` | Max sequences in a batch. `0` = unlimited |
| `--max-num-batched-tokens` | int | `2048` | Max tokens per iteration across all requests (token budget) |
| `--long-prefill-token-threshold` | int | `0` | Per-request token cap per step for chunked prefill. `0` = disabled |
| `--enable-chunked-prefill` | bool | `True` | Split long prefill across iterations. Use `--no-enable-chunked-prefill` to disable |
| `--gpu-memory-utilization` | float | `0.9` | Fraction of NPU memory usable for weights plus KV cache, as in vLLM. KV capacity is `npu_mem * this - model weight` |
| `--block-size` | int | `16` | KV cache block size in tokens |
| `--skip-prefill` | flag | off | Skip prefill, run decode only |

## Routing

| Flag | Choices | Default | Description |
| --- | --- | --- | --- |
| `--request-routing-policy` | `LOAD` / `RR` / `RAND` / `CUSTOM` | `LOAD` | Cross-instance request routing |
| `--expert-routing-policy` | `BALANCED` / `RR` / `RAND` / `CUSTOM` | `BALANCED` | MoE expert token routing |
| `--enable-block-copy` | bool | `True` | Replay one block's trace across layers (set False for per-layer EP variance) |

## Precision

| Flag | Choices | Default | Description |
| --- | --- | --- | --- |
| `--dtype` | `float16` / `bfloat16` / `float32` / `fp8` / `int8` | model's `torch_dtype`, fallback `bfloat16` | Model weight dtype |
| `--kv-cache-dtype` | `auto` / `fp8` | `auto` (inherits dtype) | KV cache dtype. `fp8` halves KV memory and selects a `*-kvfp8` profile variant |

## Prefix caching and offloading

| Flag | Default | Description |
| --- | --- | --- |
| `--enable-prefix-caching` | `True` | Prefix caching over a per-tier block pool with chained block hashes. Use `--no-enable-prefix-caching` to disable |
| `--enable-prefix-sharing` | off | Second-tier prefix pool shared across instances within a node |
| `--prefix-storage` | `None` | Where the second-tier pool lives. `None` / `CPU` / `CXL` |
| `--enable-local-offloading` | off | Weight offloading to NPU (counts weight reads in profiling) |
| `--enable-attn-offloading` | off | Attention computation offloading to PIM |
| `--enable-sub-batch-interleaving` | off | Overlap GPU compute with PIM attention. Requires `--enable-attn-offloading` |

## Dataset and output

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--dataset` | path | `None` | JSONL workload file. See **[Workloads → JSONL format](/docs/workloads/jsonl-format)** |
| `--num-reqs` | int | `0` | Entries to load from the dataset (`0` = all). For agentic, each entry is a session |
| `--output` | path | `None` | Per-request CSV output path. Stdout only if `None`. The literal `{run_id}` is replaced with the active run id |

## Run isolation

Each invocation writes ASTRA-Sim intermediates under a run-specific input
root so parallel simulations do not overwrite each other's generated
configs, traces, or Chakra workloads. Generated text traces are removed
after Chakra conversion by default, and the run-specific input root is
removed after a successful simulation by default.

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--run-id` | string | auto-generated | Path-safe id for this simulation run. Used in `astra-sim/inputs/runs/<run-id>` and the `{run_id}` output placeholder |
| `--inputs-root` | path | `astra-sim/inputs/runs/<run-id>` | Override the generated ASTRA-Sim input root, for example to place intermediates on local SSD or tmpfs |
| `--cleanup-inputs` / `--no-cleanup-inputs` | bool | `true` | Remove generated trace files after Chakra conversion and remove the generated run directory after a successful simulation. Use `--no-cleanup-inputs` to preserve traces, Chakra workloads, and input configs for debugging |

## Logging

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--log-interval` | float | `1.0` | Seconds between throughput / memory log lines |
| `--log-level` | choice | `WARNING` | `WARNING` (default) / `INFO` / `DEBUG` |

## Quick reference: which flag for which feature

| Feature | Flag(s) |
| --- | --- |
| Multi-instance (parallelism via cluster config) | (cluster config `num_instances`) |
| Tensor parallel | (cluster config `tp_size`) |
| MoE expert parallel | (cluster config `ep_size`) |
| DP+EP MoE | (cluster config `dp_group`) |
| Prefix caching | `--enable-prefix-caching` (default on), `--enable-prefix-sharing`, `--prefix-storage` |
| Chunked prefill | `--enable-chunked-prefill` (default on), `--long-prefill-token-threshold` |
| PIM attention offload | `--enable-attn-offloading` (cluster config sets `pim_config`) |
| FP8 KV cache | `--kv-cache-dtype fp8` |
| ns3 backend | `--network-backend ns3` |

For the full conceptual treatment of each feature, browse the
**[Simulator](/docs/simulator/architecture)** section. For runnable
examples, see **[Examples](/docs/examples)**.
