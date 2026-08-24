---
sidebar_position: 4
title: Trace file format
---

# Trace file format

The simulator's `trace_generator.py` builds a per-batch trace that the
Chakra converter turns into the `.et` file ASTRA-Sim consumes. This page
is the **field-by-field spec** of that trace.

The trace is normally handed to the converter **in memory**, as one field
tuple per layer, and never becomes text: the converter runs inside the
simulator process, so formatting the fields into padded columns only to
split them apart again was pure overhead. The text form below is still
exactly what the fields mean, and it is still what gets written when you
ask for it with `--save-trace-text` — so it remains the format to read
when inspecting what the simulator emitted.

For the *internals* of how this trace is produced, see
**[Simulator → Trace generation](/docs/simulator/trace-generation)**.

## File location

```
astra-sim/inputs/runs/<run_id>/trace/<hardware>/<model>/instance_<i>_batch_<b>.txt
```

One file per (instance × batch), under the run-specific ASTRA-Sim input
root — written only when `--save-trace-text` is passed. By default no
text file is produced at all; the rows go straight to the converter.

That includes the event handler's trace (`event_handler.txt`), which is
built from rows like any other.

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
| 1 | `{mode}\t\tmodel_parallel_NPU_group: {pp_size}` + optional `\t\tpp_stage_boundaries: {i1},{i2},…` | Mode marker followed by `key: value` pairs, separated by double tabs. `model_parallel_NPU_group` is the pipeline-parallel degree. `pp_stage_boundaries` is written only when `pp_size > 1`: the `pp_size - 1` layer-row indices at which each stage after the first begins, counted after any leading `kv_load`/`kv_evict` rows |
| 2 | `{num_layers}` | Number of rows that follow, **including** any `kv_load` / `kv_evict` rows |
| 3 | column header | Field names |

The mode marker comes from the instance's `pd_type`:

| Marker | `pd_type` | Converter path |
| --- | --- | --- |
| `COLOCATED` | `null` | combined prefill + decode |
| `PREFILL` | `"prefill"` | adds the per-layer KV SEND to the paired decode NPU |
| `DECODE` | `"decode"` | adds the matching RECV |

Any other `pd_type` raises `ValueError: Unknown instance type` at
trace-generation time.

### Layer rows

Each row has 11 fields, written as **left-aligned columns** by
`serving/core/utils.py::_FMT` — a 30-character minimum for `Layername`,
15 for each of the rest, **plus an explicit single space after every
field but the last**. Nothing is tab-separated.

That trailing space is load-bearing. `{:<15}` pads a value shorter than
the column but emits nothing for one that already fills it, so a
15-character field would butt straight against the next and the readers
would see the two merged into one. `ALLREDUCE:1,0,0` — a `comm_type`
with three-dimensional `involved_dim` — is 15 characters exactly. Treat
the widths as a minimum column, not a guarantee. Both readers (`trace_generator`'s own re-read and the
Chakra converter) split on runs of arbitrary whitespace
(`re.findall(r'\S+', line)` and `line.strip().split()` respectively),
so the column widths are for human legibility only and no field may
contain a space.

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
| `comm_size` | int | Collective message size in bytes. Usually `0` when `comm_type` is `NONE`, but **not always**: on a `PREFILL` trace, `qkv_proj` carries the per-layer P/D KV transfer here while keeping `comm_type` at `NONE` (see [below](#comm_size-without-a-collective)) |
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

## `comm_size` without a collective

On a `PREFILL` trace, every `qkv_proj` row carries a non-zero
`comm_size` while its `comm_type` stays `NONE`. This is not an
inconsistency: the converter's prefill path emits a point-to-point
SEND after each layer's KV projection rather than a collective, and a
SEND needs only a size, a source, a destination, and a tag — there is
no collective type to name.

The value is the **per-layer, per-rank K+V** byte count, honouring
`kv_cache_dtype`. It is deliberately *not* the layer's `output_size`,
which is the whole QKV activation: reading that shipped Q as well and
overstated the transfer by `(q_dim + 2 * kv_dim) / (2 * kv_dim)` — 3x
on Llama-3.1-8B.

Everywhere else, `comm_size` is `0` when `comm_type` is `NONE`.

## Special markers

Some layers are wrapped by markers:

### `kv_load` / `kv_evict` (tiered KV recall)

When a lower KV tier is configured (`--prefix-storage CPU` or `CXL`), a
step that recalls blocks from it prepends up to two rows **before** the
first real layer:

```
kv_load    0    LOCAL    0    REMOTE:0    8388608    LOCAL    0    NONE    0    NONE
kv_evict   0    LOCAL    0    REMOTE:0    2097152    LOCAL    0    NONE    0    NONE
```

They are not compute: `comp_time` is `0` and the byte count sits in
`weight_size`, so the converter charges them as a memory transfer
against the tier named in `weight_loc` — which is the instance's
`placement` `kv_evict_loc`, not `kv_loc`.

Each row is emitted only when its byte count is non-zero, so a step can
have both, one, or neither. Without `--prefix-storage` there is no
lower tier to recall from and both counts are always `0`, so neither
row ever appears. `batch.evict` is `0` in every mode: eviction off the
NPU costs nothing, because the data is either a finished request's
cache or was already written down off the critical path.

Two consequences worth knowing:

- The `{num_layers}` count on header line 2 includes these rows.
- `pp_stage_boundaries` indices are counted **after** they are
  stripped, so they stay stable whether or not a step recalled
  anything.

### Layer-name suffixes

Each row's `Layername` gets `_{i}` appended, where `i` is the row's
index in the whole file — *including* any `kv_load` / `kv_evict` rows.
So the same layer of the same model can carry different suffixes on
different iterations, and the suffix is an identifier, not a layer
number. `EXPERT` and `PIM` marker rows are the exception: they are
written verbatim with no suffix.

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

Reproduced at `_FMT`'s real column widths, so this is byte-for-byte
what the generator writes (scroll right for the full row):

```
COLOCATED		model_parallel_NPU_group: 1
292
Layername                      comp_time       input_loc       input_size      weight_loc      weight_size     output_loc      output_size     comm_type       comm_size       misc
embedding_0                    5386            REMOTE:0        40              LOCAL           1050673152      LOCAL           81920           NONE            0               NONE
layernorm_1                    2416            LOCAL           81920           LOCAL           8192            LOCAL           81920           NONE            0               NONE
qkv_proj_2                     36000           LOCAL           81920           LOCAL           50331648        LOCAL           122880          NONE            0               NONE
rotary_emb_3                   2795            LOCAL           102400          LOCAL           0               LOCAL           102400          NONE            0               NONE
attention_4                    7985            LOCAL           81920           LOCAL           0               LOCAL           81920           NONE            0               NONE
o_proj_5                       25611           LOCAL           81920           LOCAL           33554432        LOCAL           81920           NONE            0               NONE
... (decoder blocks 1..31 elided) ...
final_layernorm_289            2624            LOCAL           81920           LOCAL           8192            LOCAL           81920           NONE            0               NONE
lm_head_290                    714006          LOCAL           81920           LOCAL           1050673152      LOCAL           2565120         NONE            0               NONE
sampler_291                    24746           LOCAL           2565120         LOCAL           0               REMOTE:0        40              NONE            0               NONE
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
2. **Column alignment does not matter.** Both readers split on
   arbitrary whitespace, so tabs, single spaces, and `_FMT`'s padding
   are equivalent. What *does* matter is that no field contains a
   space, since that would read as two fields.
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
