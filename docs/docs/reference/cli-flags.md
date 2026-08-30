---
sidebar_position: 1
title: CLI flags
---

# `python -m serving` CLI flags

Complete reference for every command-line flag accepted by
`python -m serving`. For the conceptual side of each flag (what it
*does* internally), see **[Simulator](/docs/simulator/architecture)**.

:::tip[14 of these can be set per instance]
Flags marked **(per-instance)** below can also be written into an
individual `instances[i]` object in the cluster config, which wins over
the CLI value for that instance only. That is how one run serves
heterogeneous instances. The other 15 flags are cluster-wide. See
**[Cluster config → Runtime overrides](./cluster-config#runtime-overrides-optional)**.
:::

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
| `--max-num-seqs` **(per-instance)** | int | `128` | Max sequences in a batch. `0` = unlimited |
| `--max-num-batched-tokens` **(per-instance)** | int | `2048` | Max tokens per iteration across all requests (token budget). Clamped to the model config's `max_position_embeddings`, so `0` ("unlimited") resolves to the context length, not infinity |
| `--long-prefill-token-threshold` **(per-instance)** | int | `0` | Per-request token cap per step for chunked prefill. `0` = disabled |
| `--enable-chunked-prefill` **(per-instance)** | bool | `True` | Split long prefill across iterations. Use `--no-enable-chunked-prefill` to disable |
| `--npu-memory-utilization` **(per-instance,** as `npu_mem.mem_util`**)** | float | `0.9` | Fraction of NPU memory usable for weights plus KV cache. Corresponds to vLLM's `--gpu-memory-utilization`; KV capacity is `npu_mem.mem_size * this - model weight`. Override per instance with `npu_mem.mem_util` |
| `--reserve-full-isl` / `--no-reserve-full-isl` **(per-instance)** | flag | on | Admit a request only if its whole sequence fits, not merely its first chunk. Mirrors vLLM's `scheduler_reserve_full_isl`; without it chunked prefill over-admits and thrashes the KV cache |
| `--block-size` **(per-instance)** | int | `16` | KV cache block size in tokens |
| `--skip-prefill` | flag | off | Skip prefill, run decode only |

## Routing

| Flag | Choices | Default | Description |
| --- | --- | --- | --- |
| `--request-routing-policy` | `LOAD` / `RR` / `RAND` / `CUSTOM` | `LOAD` | Cross-instance request routing |
| `--expert-routing-policy` | `BALANCED` / `RR` / `RAND` / `CUSTOM` | `BALANCED` | MoE expert token routing |
| `--enable-block-copy` **(per-instance)** | bool | `True` | Replay one block's trace across layers (set False for per-layer EP variance) |

## Precision

| Flag | Choices | Default | Description |
| --- | --- | --- | --- |
| `--dtype` **(per-instance)** | `float16` / `bfloat16` / `float32` / `fp8` / `int8` | the model config's declared weight dtype, fallback `bfloat16` | Model weight dtype. The default reads `quantization_config.quant_method` first, then `torch_dtype` / `dtype` — on a quantized checkpoint the dtype fields describe the *activation* dtype, so DeepSeek-V3.2 (`quant_method: fp8`, `torch_dtype: bfloat16`) defaults to `fp8`. Same rule as the profiler's, because it also picks which `perf/.../<variant>/` folder is read |
| `--num-speculative-tokens` **(per-instance)** | int | `0` (off) | Draft length N, vLLM's own flag name. Omit `--spec-acceptance-rate` to take the model's published N and acceptance from `configs/spec_decode.json` |
| `--spec-acceptance-rate` **(per-instance)** | float | the model's published value | Fraction of drafted tokens the target accepts, so the mean accept length is `1 + rate * N`. **Marginal**, which is what every published source reports — not Leviathan's conditional per-position alpha. A model with no published figure must be given one |
| `--spec-acceptance-policy` **(per-instance)** | `FIXED` / `DECAY` / `CUSTOM` | `FIXED` | How the accepted count is drawn. `DECAY` uses per-position rates, which fall with draft position — same mean, different spread |
| `--kv-cache-dtype` **(per-instance)** | `auto` / `fp8` | `auto` (inherits dtype) | KV cache dtype. `fp8` halves KV memory and selects a `*-kvfp8` profile variant |

## Prefix caching and offloading

| Flag | Default | Description |
| --- | --- | --- |
| `--enable-prefix-caching` **(per-instance)** | `True` | Prefix caching over a per-tier block pool with chained block hashes. Use `--no-enable-prefix-caching` to disable |
| `--enable-prefix-sharing` | off | Second-tier prefix pool shared across instances within a node |
| `--prefix-storage` | `None` | Where the second-tier pool lives. `None` / `CPU` / `CXL` |
| `--enable-local-offloading` **(per-instance)** | off | Weight offloading to NPU (counts weight reads in profiling) |
| `--enable-attn-offloading` **(per-instance)** | off | Attention computation offloading to PIM |
| `--enable-sub-batch-interleaving` **(per-instance)** | off | Overlap GPU compute with PIM attention. Requires `--enable-attn-offloading` |

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
| `--save-trace-text` / `--no-save-trace-text` | bool | `false` | Write each batch's trace as text, for inspection. Nothing in the pipeline reads it — the Chakra converter takes the trace rows directly — so it is produced only on request, and it is the only human-readable form of what the simulator emitted. Implies `--keep-inputs` |
| `--keep-inputs` / `--no-keep-inputs` | bool | `false` | Keep the generated ASTRA-Sim inputs under `astra-sim/inputs/runs/<run-id>` after a successful simulation: the Chakra `.et` workloads and the generated network / system / memory configs, so a run can be replayed through ASTRA-Sim by hand |

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
| Heterogeneous instances in one run | (cluster config per-instance overrides; see the tip above) |

For the full conceptual treatment of each feature, browse the
**[Simulator](/docs/simulator/architecture)** section. For runnable
examples, see **[Examples](/docs/examples)**.
