---
title: Parallelism mechanics
sidebar_position: 5
---

# Parallelism mechanics

This page is the **runtime** side of parallelism: when a batch hits
ASTRA-Sim, what collectives fire, where, and how multi-instance DP
groups synchronize. The cluster-config angle (which fields turn each
of these on) is on
**[Examples → Cluster config explained](/docs/examples/cluster-config-explained)**.

## What the simulator can model

| Style | What's parallelized | Collective | Where it fires |
| --- | --- | --- | --- |
| **TP** (tensor) | Linear weights split along head dim | ALLREDUCE | After `o_proj` and `down_proj` |
| **PP** (pipeline) | Decoder layers split across GPU groups | (point-to-point in `inflight` queue) | At stage boundaries |
| **EP** (expert) | MoE experts split across ranks | ALLTOALL | Around the MoE block |
| **DP+EP** | EP across multiple instances | ALLTOALL | Same, but across instance boundaries with wave-sync |

TP and EP can share the same GPUs. DP requires a `dp_group`
identifier on the cluster config — for a dense model that is plain data
parallelism, and for MoE it also spreads experts across the group.

## TP, ALLREDUCE on every dense layer

```mermaid
flowchart LR
    subgraph INST["Instance (TP=2)"]
        direction LR
        G0["GPU 0<br/>head 0..N/2"]
        G1["GPU 1<br/>head N/2..N"]
    end
    G0 <-->|"ALLREDUCE<br/>(after o_proj, after down_proj)"| G1
```

When `tp_size > 1`, the trace generator attaches an ALLREDUCE
`COMM_COLL_NODE` after each TP-aware dense linear:

- `o_proj` (attention output projection)
- `down_proj` (MLP output projection)

These are the two layers where each TP rank holds a different head
slice of the output and needs to sum across ranks.

The `comm_size` on each ALLREDUCE is the full output tensor size
(not per-rank, ASTRA-Sim divides internally based on
`nodes_in_ring`).

`qkv_proj`, `gate_up_proj`, etc. don't need ALLREDUCE because they
*split* the input along the head dim, those layers' output is
already correctly sharded for the next layer. TP's collective cost
is bound by `o_proj` + `down_proj`, two ALLREDUCEs per decoder block.

## PP, pipeline stages and `inflight`

```mermaid
flowchart LR
    subgraph S0["Stage 0 (.et on GPU 0)"]
        direction TB
        L0a["embedding"]
        L0b["decoder layers<br/>0 .. n/pp − 1"]
        L0a --> L0b
    end
    subgraph S1["Stage 1 (.et on GPU 1)"]
        direction TB
        L1a["decoder layers<br/>n/pp .. 2n/pp − 1"]
    end
    subgraph SN["Stage pp−1 (.et on last GPU)"]
        direction TB
        LNa["decoder layers<br/>(pp−1)·n/pp .. n−1"]
        LNb["lm_head + sampler"]
        LNa --> LNb
    end
    S0 -->|"COMM_SEND / COMM_RECV<br/>comm_size = activation"| S1
    S1 -.->|"… more stages …"| SN
```

When `pp_size > 1`, the scheduler keeps an `inflight` list capped at
`pp_size` entries. When the pipeline is full, `schedule()` returns
`None` and waits for ASTRA-Sim to drain a stage, the same
back-pressure pattern as Megatron-style 1F1B.

The trace header is stamped with `model_parallel_NPU_group: {pp_size}`
plus `pp_stage_boundaries`, the layer-row indices at which each stage
after the first begins. `trace_generator.py` computes them from the
transformer-block starts it just wrote, using the same partitioning
rule as vLLM's `get_pp_indices`: blocks split evenly, with any
remainder going to the stages *before* the last one, since the last
stage also carries `final_layernorm` / `lm_head` / `sampler`. Chakra's
`llm_converter.py` reads the boundaries and emits one `.et` per NPU. At
each stage boundary it pairs a `COMM_SEND_NODE` on the upstream NPU with
a matching `COMM_RECV_NODE` on the downstream one, sized by the boundary
activation tensor.

Stages are cut **only** on transformer-block boundaries. That is the
one place where the upstream layer's `output_size` and the downstream
layer's `input_size` are the same tensor — the hidden state, since a
block runs `layernorm` → … → `down_proj`/`moe`. Inside a block they
differ (`qkv_proj` emits Q+K+V, `rotary_emb` declares only Q+K), and
ASTRA-Sim's analytical backend keys its send/recv callback tracker on
`(tag, src, dst, chunk_size, chunk_id)` — so a size disagreement never
matches and the downstream NPU waits forever instead of raising. Cutting
the raw line count evenly used to land boundaries mid-block, which is
what made only some `pp_size` values hang.

`--enable-sub-batch-interleaving` is rejected with `pp_size > 1`: an
interleaved trace leaves both sub-batches mid-block at every group edge,
so a stage has no single hidden state to hand on.

Inter-stage P2P latency (link bandwidth, hop count, contention) is
therefore part of the reported iteration time, and pipeline overlap
between in-flight batches falls out from each NPU's independent `.et`
schedule.

## EP, ALLTOALL around the MoE block

```mermaid
flowchart LR
    INPUT[Input residue] --> DISP["Dispatch<br/>ALLTOALL"]
    subgraph EXP["Expert compute (parallel ranks)"]
        direction TB
        E0["Rank 0<br/>experts 0..N/2"]
        E1["Rank 1<br/>experts N/2..N"]
    end
    DISP --> E0
    DISP --> E1
    E0 --> COMB["Combine<br/>ALLTOALL"]
    E1 --> COMB
    COMB --> OUTPUT[Output residue]
```

For MoE models, `trace_generator` wraps the MoE block with two
ALLTOALL collectives:

```
... → MoE dispatch ALLTOALL → expert compute → MoE combine ALLTOALL → ...
```

The dispatch ALLTOALL routes each token to its assigned expert's
rank. The combine ALLTOALL gathers expert outputs back to the
originating ranks. Both are scoped to the EP dimension.

Each EP rank gets a per-rank latency from
`profiler/perf/<hw>/<model>/<variant>/tp1/moe.csv` keyed on its
**local** token count (after dispatch) and the **activated experts**
per token. Ranks execute in parallel and synchronize at the ALLTOALL
barrier, slower ranks gate the others.

Token routing decisions come from `gate_function.py`. See
**[MoE expert routing](./moe-expert-routing)** for the policies.

## DP+EP, wave synchronization

```mermaid
flowchart TB
    subgraph DPGROUP["DP group A (2D topology, [tp_size=1, dp_size=2])"]
        direction LR
        subgraph I1["Instance 1"]
            G1["GPU 0<br/>experts 0..63"]
        end
        subgraph I2["Instance 2"]
            G2["GPU 0<br/>experts 64..127"]
        end
        G1 <-->|"EP-ALLTOALL<br/>(involved_dim = [F, T])"| G2
    end
```

```mermaid
sequenceDiagram
    autonumber
    participant I1 as Instance 1
    participant I2 as Instance 2
    participant DPB as Python<br/>dp_pending barrier
    participant A as ASTRA-Sim
    I1->>I1: scheduler.schedule()
    I1->>DPB: dp_pending["A"][0] = batch
    Note over I2: scheduling on its own pace
    I2->>I2: scheduler.schedule()
    I2->>DPB: dp_pending["A"][1] = batch
    Note over DPB: All members ready
    DPB->>I1: emit trace (comm_size = max)
    DPB->>I2: emit trace (comm_size = max)
    I1->>A: workload_dp_A.et
    I2->>A: workload_dp_A.et
    Note over A: Matching stream IDs<br/>block at ALLTOALL
    A-->>I1: cycle count
    A-->>I2: cycle count
```

This is where the simulator gets clever. When two or more instances
share a `dp_group`, they form a single coordinated wave. Two
synchronization mechanisms work together:

### 1. Python-side `dp_pending` barrier

In `__main__.py`, a `dp_pending` dict tracks which DP-group members
have scheduled their batches for the current wave. Trace generation
is **deferred** until all members have scheduled. When the last
member arrives:

- The simulator takes `max_total_len` across the group and pads every
  member's batch up to it, matching CUDA-graph DP padding in production
  serving.
- The MoE collective size is anchored to that same `max_total_len` — *not*
  `max x dp_group_size`. That calibrates the AllGather/ReduceScatter
  bandwidth model against the same `link_bw` that already matches
  AllReduce.
- All members generate their traces with the same `comm_size`, even
  if their per-instance `total_len` differs.

If one DP member has no pending requests, the scheduler synthesizes a
**dummy batch** (1 decode token) so the wave still runs. When
all of one member's real requests have finished but the others
haven't, the dummy batches keep flowing until the whole group is
done.

### 2. ASTRA-Sim ALLTOALL barrier

All DP-group instances' `.et` files share the same workload folder
(`dp_<group>_batch<bid>/llm.et`) and use **matching stream IDs** on
the ALLTOALL collectives. ASTRA-Sim's runtime sees the matching IDs
and blocks until both NPUs reach the collective, naturally
implementing the wave-sync at the network layer.

So both halves of the sync, Python deferral on submission, ASTRA-Sim
blocking on the collective, together produce a deterministic
wave-synchronous schedule.

## Multi-dimensional ASTRA-Sim topology and `involved_dim`

`config_builder` generates a multi-dimensional ASTRA-Sim network when DP
groups are present, innermost dimension first:
`npus_count: [tp_size, dp_group_size]`, or
`[tp_size, pp_size, dp_group_size]` when `pp_size > 1`. This mirrors
vLLM's rank layout, `all_ranks.reshape(-1, dp, pp, pcp, tp)`; the
`pp_size` dimension is omitted when it is 1, so DP+TP configs keep their
2-D topology. Collectives are scoped per dimension via the
`involved_dim` BoolList on each `COMM_COLL_NODE`:

- **TP-ALLREDUCE:** the TP dim only — `[True, False]`, or
  `[True, False, False]` with PP.
- **EP:** the DP dim, plus the TP dim when EP spans past one instance's
  GPUs — `[False, True]` / `[True, True]`, or `[False, False, True]` /
  `[True, False, True]` with PP. The PP dim is **never** involved:
  vLLM's EP group is `all_ranks.transpose(1, 2).reshape(-1, dp*pcp*tp)`,
  whose transpose pins the pipeline stage, so experts are sharded across
  the DP x TP ranks of one stage.

The `involved_dim` is encoded in the trace's `comm_type` field with
a `:dim0,dim1` suffix:

```
ALLREDUCE:1,0     # TP only
ALLTOALL:0,1      # EP across DP only
```

The Chakra converter parses this via `_parse_comm_type` and writes
the BoolList into the `.et` file. ASTRA-Sim's `Workload::issue_comm`
reads it and dispatches the collective only on the involved dims.

The `system.json` collective implementations need one entry per
topology dim, `config_builder` generates this automatically:
`"all-to-all-implementation": ["ring", "ring"]` for 2D.

## Communication sizes (ASTRA-Sim semantics)

Every `comm_size` in the trace is the **total** data size, not
per-NPU. ASTRA-Sim divides internally by the number of nodes in the
ring (`msg_size = data_size / nodes_in_ring`).

So:

- ALLREDUCE on `o_proj`: pass the **full output tensor size**
  (`total_len * hidden_size * fp_size`).
- ALLTOALL for MoE: pass the **full activation tensor size**
  (`total_len * hidden_size * fp_size`).

If you see surprisingly fast collectives in your trace logs, check
that you're not accidentally passing per-rank sizes, that's a
common mistake when extending the trace generator.

## When to use which

A rough decision tree (the *configuration* angle is on
[Examples → Cluster config explained](/docs/examples/cluster-config-explained)):

- **Single GPU fits the model:** TP=1. Done.
- **Need more GPUs for memory:** start with TP. ALLREDUCE cost grows
  with `tp_size`, so going past 4-8 is rarely worth it.
- **Multiple replicas for throughput:** add `num_instances` (no
  `dp_group`). Independent instances behind a router.
- **MoE model, single instance:** add `ep_size = tp_size`. Same GPUs,
  EP-ALLTOALL replaces TP-ALLREDUCE on the MoE block.
- **MoE, want to scale experts past one instance's GPUs:** DP+EP
  with `dp_group` set. EP spans instances via wave-sync.
- **Dense model, want data-parallel replicas:** `dp_group` set and no
  `ep_size`. The replicas are wave-synchronized but share no experts.

## Gotchas

1. **`ep_size > tp_size` requires `dp_group`.** Otherwise the cluster
   config builder rejects the spec. EP needs the DP dimension of the
   topology to scale beyond a single instance's GPU count.
2. **Dummy batches are real ASTRA-Sim work.** A DP group with one
   idle instance still pays the ALLTOALL cost on the dummy batch.
   This is what production looks like, wave-sync is wave-sync.
3. **`comm_size` is synchronized to the max.** Even if one DP
   member's batch is much smaller, the ALLTOALL message size matches
   the largest member's. This is *correct* (matches production
   padding) but worth knowing.
4. **PP models inter-stage forwarding via send/recv, not via
   micro-batch splitting inside an iteration.** Activation shipment
   between stages goes through ASTRA-Sim send/recv (so link bandwidth
   and contention show up in the result), but a single iteration is
   not chunked into multiple micro-batches — the overlap benefit
   comes from running up to `pp_size` consecutive iterations
   simultaneously. There's also no knob to pick a pipeline schedule
   (1F1B, interleaved, etc.).

## What's next

- **[MoE expert routing](./moe-expert-routing)**: how tokens get
  distributed across EP ranks before the dispatch ALLTOALL.
- **[Examples → DP+EP MoE](/docs/examples/parallelism/dp-ep-moe)** -
  a worked-out config that exercises this whole machinery.
