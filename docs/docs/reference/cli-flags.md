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
| `--block-size` **(per-instance)** | int | the profiled value, else `16` | KV cache block size in tokens. vLLM treats this as a **floor and an alignment unit**, not the answer: it takes `max(backend minimum, your value)` as the alignment, derives the smallest multiple of that whose attention page covers one mamba page, and raises the block size to it — never lowering it. On Qwen3.8-27B, asking for 16 gives **784** and asking for 64 gives **832**. The profiler records what the engine settled on in `meta.yaml::engine_resolved.per_tp[tp]` — **per TP degree**, since both the mamba page and the attention page scale with the rank's shard — and the simulator reads back the entry for the instance's `tp_size`, so lookups match the block size the latencies were measured at. An explicit value that disagrees is allowed but warned about. Bundles profiled before the field existed do not carry it and fall back to `16`; ones written before it was split by TP carry a flat value, which is read as a fallback |
| `--skip-prefill` | flag | off | Skip prefill, run decode only |

## Routing

| Flag | Choices | Default | Description |
| --- | --- | --- | --- |
| `--request-routing-policy` | `LOAD` / `RR` / `RAND` / `CUSTOM` | `LOAD` | Cross-instance request routing |
| `--expert-routing-policy` | `BALANCED` / `RR` / `RAND` / `CUSTOM` | `BALANCED` | MoE expert token routing |
| `--enable-block-copy` **(per-instance)** | bool | `True` | Replay one block's trace across layers (set False for per-layer EP variance) |

## Precision

**There are no precision flags.** Every dtype is read from the model config,
because that is where each one is actually decided:

| Cache | Config field | Rule |
| --- | --- | --- |
| Weights | `quantization_config.quant_method`, then `torch_dtype` / `dtype` | On a quantized checkpoint the dtype fields describe the *activation* dtype, so DeepSeek-V3.2 (`quant_method: fp8`, `torch_dtype: bfloat16`) is fp8, not bf16. Same rule as the profiler's, because it also picks which `perf/.../<variant>/` folder is read |
| KV cache | `quantization_config.kv_cache_scheme` (compressed-tensors) or `kv_cache_quant_algo` (ModelOpt) | Either one present means fp8, otherwise the weight dtype. This is vLLM's own promotion at `attention.py:281`, and the direction its source states for itself: *"kv cache dtype should be specified in the FP8 checkpoint config and become the 'auto' behavior"* |
| Mamba conv state | `mamba_cache_dtype` | `auto` falls back to the weight dtype |
| Mamba recurrent state | `mamba_ssm_dtype` | `auto` falls back to the conv dtype. Qwen3.8-27B declares `float32`, so its recurrent state is 4 bytes where its conv state is 2 |
| Sparse-indexer side cache | none — fixed by the model | DeepSeek/GLM store fp8 keys plus fp32 scales as uint8; MiniMax-M3 stores bf16. Neither follows the KV cache dtype |

A dtype is a property of the checkpoint, and once a model carries five of
them a flag per dtype is both unusable and unfaithful — it describes a model
nobody can serve. To simulate a different precision, profile it: the
profiler's `--dtype` / `--kv-cache-dtype` / `--variant` write a separate
`perf/.../<variant>/` bundle, and the simulator reads the one the checkpoint
names.

| Flag | Choices | Default | Description |
| --- | --- | --- | --- |
| `--num-speculative-tokens` **(per-instance)** | int | `0` (off) | Draft length N, vLLM's own flag name. Omit `--spec-acceptance-rate` to take the model's published N and acceptance from `configs/spec_decode.json` |
| `--spec-acceptance-rate` **(per-instance)** | float | the model's published value | Fraction of drafted tokens the target accepts, so the mean accept length is `1 + rate * N`. **Marginal**, which is what every published source reports — not Leviathan's conditional per-position alpha. A model with no published figure must be given one |
| `--spec-acceptance-policy` **(per-instance)** | `FIXED` / `DECAY` / `CUSTOM` | `FIXED` | How the accepted count is drawn. `DECAY` uses per-position rates, which fall with draft position — same mean, different spread |

**The drafter's time is not charged yet, and a model that drafts with
itself refuses to run.** vLLM runs the drafter **N times per step** —
once, then `num_speculative_tokens - 1` more — each a decode-shaped
forward over a norm pair, an `eh_proj`, a full decoder layer, `lm_head`
and the sampler. Reporting that as free would claim a speedup no engine
can deliver, so a model with MTP modules (`num_nextn_predict_layers`,
`num_mtp_modules`, `mtp_num_hidden_layers`) raises until its
architecture catalog has an `mtp:` block to price them from — and then
until a profiled `mtp.csv` exists to read. **All four modern families
now have the catalog block**; what is still missing is the measurement,
which needs a profile run with `--profile-mtp N`.

A model with **no** MTP modules drafts with a separate model or with
n-gram — a serving choice rather than a checkpoint property, and the
simulator has no second model to charge. That case warns instead of
refusing, and says plainly that the reported speedup is an upper bound.

The drafter's **KV cache** is charged either way: an MTP module wraps a
real decoder layer, so it publishes a cache spec of its own. That is
+1.6% bytes/token on DeepSeek-V3.2's one module, +11.7% on
MiniMax-M3's seven, +6.2% on Qwen3.8-27B.

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
| FP8 KV cache | (model config `quantization_config.kv_cache_scheme`) |
| ns3 backend | `--network-backend ns3` |
| Heterogeneous instances in one run | (cluster config per-instance overrides; see the tip above) |

For the full conceptual treatment of each feature, browse the
**[Simulator](/docs/simulator/architecture)** section. For runnable
examples, see **[Examples](/docs/examples)**.
