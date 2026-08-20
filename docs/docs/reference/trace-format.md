---
sidebar_position: 4
title: Trace file format
---

# Trace file format

The simulator's `trace_generator.py` writes a per-batch text trace
that the Chakra converter then reads to produce the `.et` file
ASTRA-Sim consumes. This page is the **field-by-field spec** of that
text trace.

For the *internals* of how this trace is produced, see
**[Simulator → Trace generation](/docs/simulator/trace-generation)**.

## File location

```
astra-sim/inputs/runs/<run_id>/trace/<hardware>/<model>/instance_<i>_batch_<b>.txt
```

One file per (instance × batch), under the run-specific ASTRA-Sim input
root. Regenerated every iteration and removed after Chakra conversion by
default. Use `--no-cleanup-inputs` to preserve generated traces.

## File structure

```
COLOCATED		model_parallel_NPU_group: {pp_size}		pp_stage_boundaries: 73,145,217
{num_layers}
Layername    comp_time    input_loc    input_size    weight_loc    weight_size    output_loc    output_size    comm_type    comm_size    misc
embedding_0    5621    REMOTE:0    40    LOCAL    1050673152    LOCAL    81920    NONE    0    NONE
layernorm_0    1240    LOCAL    81920    LOCAL    8192    LOCAL    81920    NONE    0    NONE
qkv_proj_0    8324    LOCAL    81920    LOCAL    25165824    LOCAL    245760    NONE    0    NONE
...
sampler_291    25933    LOCAL    2565120    LOCAL    0    REMOTE:0    40    NONE    0    NONE
```

### Header (lines 1–3)

| Line | Content | Meaning |
| --- | --- | --- |
| 1 | `COLOCATED\tmodel_parallel_NPU_group: {pp_size}` + optional `\tpp_stage_boundaries: {i1},{i2},…` | Trace mode marker followed by `key: value` pairs. `model_parallel_NPU_group` is the pipeline-parallel degree. `pp_stage_boundaries` is written only when `pp_size > 1`: the `pp_size - 1` layer-row indices at which each stage after the first begins, counted after any leading `kv_load`/`kv_evict` rows |
| 2 | `{num_layers}` | Number of layer rows that follow |
| 3 | column header (tab-separated) | Field names |

### Layer rows

Each row has 11 tab-separated fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `Layername` | string | Canonical layer name + index (e.g., `qkv_proj_0`, `attention_31`) |
| `comp_time` | int | Computation latency in **nanoseconds** |
| `input_loc` | enum | Where the input tensor lives (see [memory locations](#memory-locations)) |
| `input_size` | int | Input tensor size in bytes |
| `weight_loc` | enum | Where the layer's weights live |
| `weight_size` | int | Weight size in bytes |
| `output_loc` | enum | Where the output tensor will be written |
| `output_size` | int | Output tensor size in bytes |
| `comm_type` | enum | Collective type after this layer (see [communication](#communication-types)) |
| `comm_size` | int | Collective message size in bytes (`0` if `comm_type` is `NONE`) |
| `misc` | string | Misc tag (sub-batch interleaving, etc.; usually `NONE`) |

## Memory locations

The `input_loc`, `weight_loc`, and `output_loc` fields use one of:

| Value | Meaning | Backed by |
| --- | --- | --- |
| `LOCAL` | NPU memory | per-instance NPU |
| `REMOTE:{node_id}` | CPU memory on the named node | per-node `cpu_mem` |
| `CXL:{device_id}` | CXL device memory | top-level `cxl_mem` block |
| `STORAGE` | Storage tier (used by power model only) | (none) |

The numeric IDs match the C++ enum in
`astra-sim/astra-sim/system/AstraMemoryAPI.hh`:

| Symbol | Value |
| --- | --- |
| `LOCAL` | 1 |
| `REMOTE` | 2 |
| `CXL` | 3 |
| `STORAGE` | 4 |

These must stay in sync between the trace and the C++ enum;
mismatches cause silent miscounting.

### First and last layer must use REMOTE

The Chakra converter emits a `MEM_LOAD_NODE` from the **first**
layer's `input_loc` and a `MEM_STORE_NODE` from the **last** layer's
`output_loc`. Both must be `REMOTE:{node_id}` (CPU side): the
simulator models the request entering / leaving the NPU as a
host-side transfer.

This is why `embedding_0` has `input_loc=REMOTE:0` and `sampler_*`
has `output_loc=REMOTE:0` in the example above. The `MEM_STORE_NODE`
is sized from the last row's `output_size` — the sampled token ids,
4 bytes per sequence — not from its `input_size`, which is the logits
tensor the sampler consumed on the NPU.

## Communication types

The `comm_type` field selects the collective ASTRA-Sim runs after
this layer:

| Value | Meaning | When emitted |
| --- | --- | --- |
| `NONE` | No collective | Most layers |
| `ALLREDUCE` | All-reduce across the involved dim | After `o_proj` and `down_proj` (TP > 1) |
| `ALLTOALL` | All-to-all dispatch / combine | Around the MoE block (EP-aware) |

### Dimension scoping

For multi-dimensional ASTRA-Sim topologies (DP+EP layouts), the
`comm_type` can include a **dimension scope suffix**:

| Suffix | Meaning |
| --- | --- |
| `ALLREDUCE` | Default, all dims involved |
| `ALLREDUCE:1,0` | Dim 0 = involved (`True`), dim 1 = not (`False`). i.e., TP-only ALLREDUCE in a 2D `[tp, dp]` topology |
| `ALLTOALL:0,1` | Dim 0 = not involved, dim 1 = involved. i.e., EP-only ALLTOALL across the DP group |

The Chakra converter parses these via `_parse_comm_type` and writes
the `involved_dim` BoolList into the `.et` file. ASTRA-Sim's
`Workload::issue_comm()` reads the BoolList and routes the collective
on the named dimensions.

## Special markers

Some layers are wrapped by markers:

### `EXPERT {i}` / `EXPERT END` (MoE)

Wrap the per-rank expert compute:

```
EXPERT 0
moe_expert_local_3_rank0    1842    LOCAL    524288    LOCAL    9437184    LOCAL    524288    ALLTOALL    524288    NONE
EXPERT END
EXPERT 1
moe_expert_local_3_rank1    1804    LOCAL    524288    LOCAL    9437184    LOCAL    524288    ALLTOALL    524288    NONE
EXPERT END
```

ASTRA-Sim runs each `EXPERT {i}` block on rank `i` in parallel,
synchronizing at the surrounding ALLTOALLs.

### `PIM {channel}` / `PIM END` (PIM offload)

Wrap PIM-side attention compute:

```
PIM 0
pim_attention_3    4126    LOCAL    245760    LOCAL    0    LOCAL    245760    NONE    0    NONE
PIM END
```

Multiple `PIM <channel>` blocks can appear back-to-back to model
multi-channel parallel attention.

## Sub-batch interleaving (`misc`)

When `--enable-sub-batch-interleaving` is on, layers carry a batch
tag in `misc`:

```
qkv_proj_3    4128    ...    NONE    0    BATCH_1
pim_attention_3    8264    ...    NONE    0    BATCH_2
o_proj_3    3845    ...    NONE    0    BATCH_1
```

`BATCH_1` and `BATCH_2` halves run in parallel, typically GPU
compute on one half while PIM attention runs on the other.

## Sample full trace (single instance, TP=1, dense model)

```
COLOCATED		model_parallel_NPU_group: 1
292
Layername	comp_time	input_loc	input_size	weight_loc	weight_size	output_loc	output_size	comm_type	comm_size	misc
embedding_0	5386	REMOTE:0	40	LOCAL	1050673152	LOCAL	81920	NONE	0	NONE
layernorm_1	2416	LOCAL	81920	LOCAL	8192	LOCAL	81920	NONE	0	NONE
qkv_proj_2	36000	LOCAL	81920	LOCAL	50331648	LOCAL	122880	NONE	0	NONE
rotary_emb_3	2795	LOCAL	102400	LOCAL	0	LOCAL	102400	NONE	0	NONE
attention_4	7985	LOCAL	81920	LOCAL	0	LOCAL	81920	NONE	0	NONE
o_proj_5	25611	LOCAL	81920	LOCAL	33554432	LOCAL	81920	NONE	0	NONE
layernorm_6	2416	LOCAL	81920	LOCAL	8192	LOCAL	81920	NONE	0	NONE
gate_up_proj_7	159157	LOCAL	81920	LOCAL	234881024	LOCAL	573440	NONE	0	NONE
act_fn_8	2965	LOCAL	573440	LOCAL	0	LOCAL	286720	NONE	0	NONE
... (decoder blocks 1..31 elided) ...
down_proj_288	81014	LOCAL	286720	LOCAL	117440512	LOCAL	81920	NONE	0	NONE
final_layernorm_289	2624	LOCAL	81920	LOCAL	8192	LOCAL	81920	NONE	0	NONE
lm_head_290	714006	LOCAL	81920	LOCAL	1050673152	LOCAL	2565120	NONE	0	NONE
sampler_291	24746	LOCAL	2565120	LOCAL	0	REMOTE:0	40	NONE	0	NONE
```

A layer's `output_size` is **not** in general the next layer's `input_size`:
`qkv_proj` emits Q+K+V while `rotary_emb` only declares Q+K, and `attention`
reads K/V from the KV cache rather than from the activation. The two agree at
transformer-block boundaries (`layernorm` in, `down_proj`/`moe` out — both the
hidden state), which is why pipeline stages may only be cut there.

## How the Chakra converter consumes this

The Chakra converter (`astra-sim/extern/graph_frontend/chakra/src/converter/llm_converter.py`)
walks the trace and emits Chakra protobuf nodes:

| Trace row | Chakra node |
| --- | --- |
| First layer | `MEM_LOAD_NODE` for the input transfer |
| Each compute row | `COMP_NODE` keyed by `comp_time` |
| Last layer | `MEM_STORE_NODE` for the output transfer |
| `comm_type != NONE` | `COMM_COLL_NODE` with optional `involved_dim` BoolList |
| `EXPERT {i}` block | Sub-graph run on rank `i` |
| `PIM <channel>` block | Sub-graph routed to the PIM device |

The `.et` file is what `controller.write_flush` then sends to
ASTRA-Sim.

## Gotchas

1. **`comp_time` is nanoseconds in the trace** but the underlying
   profile CSVs use microseconds. The conversion happens in
   `_load_perf_db()` at simulator startup.
2. **Tab-separated, not space.** Mixing tabs and spaces breaks the
   Chakra parser silently.
3. **Don't hand-edit production traces.** They're regenerated every
   iteration; manual edits get clobbered. To inject custom timings,
   modify the profile CSVs or the trace generator.
4. **`comm_size` is the total payload, not per-rank.** ASTRA-Sim
   divides by the number of nodes in the ring internally.

## What's next

- **[Simulator → Trace generation](/docs/simulator/trace-generation)**
  how each row is produced.
- **[Cluster config](./cluster-config)**: `placement` rules
  determine `weight_loc` and `kv_loc`.
