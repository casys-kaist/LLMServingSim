---
title: MoE expert routing
sidebar_position: 6
---

# MoE expert routing

For Mixture-of-Experts models, every token visiting an MoE layer
needs an answer to two questions: **which experts do I activate** and
**which EP rank holds them**. The first is the model's gate
function; the second is determined by how the simulator assigns
experts to ranks. This page is about both.

With a matching native DP+EP profile installed, the component consumer uses
the recorded rank placement and balanced assignment surface. See
[Native MoE components](../profiler/native-moe-components) for local/gathered
token domains, supported policies and warnings. The legacy whole-block
`moe.csv` path remains the fallback; its reduced-top-k acquisition is not
equivalent to the native distributed expert kernel.

> Configuration angle (`--expert-routing-policy` flag, when to use
> which) is on **[Examples → Expert parallel](/docs/examples/parallelism/expert-parallel)**.
> This page is the internal mechanics.

## The piece that does it: `GateRouter`

`serving/core/gate_function.py` defines `GateRouter`. The trace
generator instantiates one per generated trace; on every built MoE block it
calls:

```python
router = GateRouter(
    node_id=0, instance_id=0,
    num_local_experts=N,           # total experts in the model
    num_experts_per_tok=K,         # top-K activations per token
    routing_policy='BALANCED',     # one of 4 policies, see below
    seed=42,
    block_copy=True,
)

result = router.route_ep(layer_num=0, batch_id="0", total_len=T, ep_size=EP)
# → RoutingResult(local_tokens=[...], activated_experts=[...], source_tokens=[...])
```

`local_tokens[i]` counts unique tokens with at least one expert on global
EP rank `i`; `activated_experts[i]` counts its distinct active experts.
For the default all-gather/reduce-scatter backend, MoE lookup uses the
**gathered input token count**, not `local_tokens[i]`, and that rank's
activated-expert count. The source-token split in this routing result is
synthetic; actual DP communication sizes come from the scheduled DP round.

## Four policies

```mermaid
flowchart LR
    subgraph BAL["BALANCED (default)"]
        TB["Token count"] --> ASB["Equal expected counts<br/>per rank<br/>(deterministic)"]
    end
    subgraph RR["RR"]
        TR["8 tokens"] --> ASR["1, 1, 1, 1, 1, 1, 1, 1<br/>(positional)"]
    end
    subgraph RND["RAND"]
        TN["8 tokens"] --> ASN["3, 1, 0, 4<br/>(seeded uniform)"]
    end
    subgraph CST["CUSTOM"]
        TC["8 tokens"] --> ASC["the count a real<br/>gate was measured<br/>to reach"]
    end
```

| Policy | Determinism | What it models | When to use |
| --- | --- | --- | --- |
| **BALANCED** (default) | Deterministic | Idealized load-balanced gate (post-aux-loss training) | Most research baselines |
| **RR** | Deterministic | Pure round-robin assignment | Sanity / null-baseline runs |
| **RAND** | Seeded random | Uniform random per token | Variation under a uniform gate assumption |
| **CUSTOM** | Deterministic | The **measured** distinct-expert count of a real trained gate | Comparing against a real vLLM run |

### BALANCED, closed-form pigeonhole

BALANCED estimates counts under a uniform-routing assumption; it does not
construct an exact assignment histogram. With `E` experts, top-`K`, and
`T` tokens, its distinct-expert expectation per rank is
`(E / EP) * (1 - (1 - K / E)**T)`, rounded and bounded for lookup.
All ranks receive the same analytical count. This is deterministic and
allows block copy, but training with load-balancing losses does not establish
that a deployed model has equal rank loads or this exact distribution.

It also estimates **how many EP ranks one token reaches**, which sets the
unique local-token count for a dispatching backend. The default AgRs path
uses all gathered input rows and does not size its collective from this
local-token estimate. `_hit_probs` computes the hit probability
exactly, by
a DP over which groups the token selected rather than by sampling, so
the simulator stays deterministic. It draws **without replacement**,
matching `torch.topk`: a token picks `k` *distinct* experts. An earlier
closed form, `1 - ((ep-1)/ep)**k`, modelled `k` independent draws and
read about 1% low (Qwen3-30B at EP=8: 0.6564 against the exact 0.6674).

### RR, round-robin

For unrestricted routing, token `t` selects the `K` distinct experts
`(t + j) % E`, for `j = 0..K-1`. The ordinal advances across the full
gathered/replicated batch, not separately within each source partition.
Grouped routing applies the same ordinal to the configured group selection.

Each layer invocation starts at token zero; token content does not affect
the result. RR is a deterministic sanity-check policy, not the trained gate
or the independent uniform-draw law used by BALANCED. Their distinct-expert
counts need not agree, especially on short batches.

### RAND, random

Per-token uniform random selection of distinct experts (using `seed=42`
by default for reproducibility). Ranks can receive different loads, but
uniform random draws are neither a learned gate nor a worst-case bound.
The router is recreated for each trace, so identical shapes with the
same seed repeat their draws.

### CUSTOM, the measured curve

BALANCED's closed form asks how many distinct experts a *uniform* gate
would reach. A trained gate concentrates on popular experts, so it
reaches fewer, and the concentration lives in the weights where no
closed form can get at it. Measured on Qwen3-30B-A3B over 110,640 real
gate calls: **0.87x** of the uniform model through the middle of the
range, **0.94x** at a saturated decode, and exactly **1.000** at one
token, where both readings are just `k`.

CUSTOM reads that measurement instead of deriving it. Record it with
[`bench run --record-gate-stats`](/docs/reference/bench-cli), which logs
what the real gate did and reduces it to one curve, then point the
simulator at the file:

```bash
python -m serving \
  --cluster-config configs/cluster/single_node_moe_dp_ep_instance.json \
  --dataset workloads/sharegpt-qwen3-30b-a3b-300-sps10.jsonl \
  --expert-routing-policy CUSTOM \
  --gate-stats bench/results/<run_id>/gate_stats.json
```

The recorder marks workload start and end explicitly. Calls outside that
interval are excluded; a workload that routes many tokens to the same top-k
experts remains valid data. Raw logs without both boundaries must be recorded
again rather than classified by their expert counts. Existing reduced
`gate_stats.json` files remain readable, but their recorded scope still matters.
The observer requires eager execution and synchronization, so its latency is
diagnostic, not an end-to-end reference. Validate against a separate
uninstrumented run. A measured count is an explicit workload/weight input to
CUSTOM, not a latency coefficient or proof of accuracy for other checkpoints,
batching, precision or expert-load histograms.

Against a real DP=1 vLLM run of Qwen3-30B-A3B on RTX PRO 6000, holding
everything else fixed:

| | TTFT mean | TTFT p50 | TPOT mean | span |
| --- | --- | --- | --- | --- |
| BALANCED (uniform closed form) | +8.3% | +6.5% | +6.0% | +4.8% |
| **CUSTOM (measured curve)** | **+2.8%** | **+2.1%** | **+1.9%** | **+0.3%** |

Nothing is fitted and nothing is interpolated across models. The curve
is linear between the batch sizes the recording run visited and clamped
at both ends: below the first point there is nothing under one token,
and above the last the count is bounded by `E` and the curve is already
flat there. `num_experts` and `num_experts_per_tok` are recorded in the
file and checked on load, because a distinct count means nothing without
them -- a curve from another checkpoint is refused rather than rescaled,
and the closed form takes over.

**A missing, unreadable or mismatched file falls back to BALANCED** with
one warning. That is deliberate: a guessed concentration is worse than
the closed form, which at least knows the right `E`.

Per-token call sites (`GateRouter.route`, used at EP=1 by nothing in the
simulator today) still go through `_custom_gate_function`, which is the
plug-in hook for driving routing from something else entirely.

:::note[Why not force the *truth* to be uniform instead]
A bench mode that replaced the real gate's assignment was tried, and it
does not answer this question. On a non-EP configuration under
cudagraphs it reads **0.700x** of the real gate's TPOT -- flattening the
per-expert histogram lands the grouped GEMM on a *different kernel
variant*. An in-situ profile of 8 real decode steps at matched batch
shape shows a `MoeFCGemm` template instantiation appearing with 144
calls and another dropping to zero, with attention unchanged as the
control. It cannot hold "everything but the count" fixed, so it measures
the schedule rather than the count. Measuring the count directly has no
such coupling.
:::

## Group-limited routing

DeepSeek-V3/V3.2 and GLM restrict a token's experts to `topk_group` of
`n_group` groups. `deepseek_v2.py` passes `num_expert_group=config.n_group`
and `topk_group=config.topk_group`, both defaulting to 1, and `GateRouter`
reads the same two fields off the checkpoint.

It matters because it narrows the set of ranks one token can reach, and
so the per-rank MoE token count and the collective that surrounds the
block:

| Model | `E` / top-`k` | `n_group` / `topk_group` | P(token reaches a given rank) at EP=8 |
| --- | --- | --- | --- |
| DeepSeek-V3.2-Exp | 256 / 8 | **8 / 4** | **0.454** — against GLM-5's 0.662 at the same `E` and `k`, a 31% cut |
| GLM-5 | 256 / 8 | 1 / 1 | 0.662 |
| Qwen3-30B-A3B | 128 / 8 | — | 0.667 |
| Mixtral-8x7B | 8 / 2 | — | 0.250 |
| Phi-mini-MoE | 16 / 2 | — | 0.242 |

Only DeepSeek-V3.2 actually restricts. GLM-5 ships `n_group: 1`, which
is the unrestricted case spelled out, and the other three do not
declare the fields at all — their spread comes from `E` and `k`, not
from grouping. The expression is verified
against a 400k-trial Monte Carlo on seven
`(E, k, n_group, topk_group, ep)` points and agrees within 0.0011,
which is the Monte Carlo's own noise.

The grouped and ungrouped cases go through the **same** expression on
purpose. Keeping the old approximation for `n_group == 1` would have
left two answers to one question.

Grouping changes the reachable set, not the batch's total work: a
balanced gate still spreads `T * K` expert-token pairs over every
expert, so the **activated experts per rank** is unchanged. Only
`local_tokens` moves.

## Expert-to-rank assignment

Whatever policy decides "token T goes to expert E", the simulator
also has to know "expert E lives on which rank". This uses **even
partitioning**:

```
GateFunction.expert_owner(e, ep_size, num_experts)
    = min(e * ep_size // num_experts, ep_size - 1)
```

So with 128 experts and `ep_size=2`, experts 0–63 live on rank 0 and
64–127 on rank 1. With `ep_size=4`, each rank holds 32 experts. The
`min(..., ep_size - 1)` is defensive — for a valid expert id the
division already lands below `ep_size` — but it keeps an out-of-range id
from indexing past the last rank.

The `GateRouter.route_ep()` result covers the **whole EP group**. A DP member
selects its own slice:

```python
global_ep_rank = dp_rank * local_ep + local_rank
```

`dp_rank` is the member's zero-based position within its DP group, not its
instance ID. For TP2/DP2 with full EP, the first member selects ranks 0/1
and the second selects 2/3. PP stages have separate EP groups and do not
add another offset. Both ordinary and idle-member DP completion paths use
this mapping, including interleaved trace generation.

The trace's `EXPERT {local_rank}` markers stay **instance-local** for the
converter. Global indexing applies to routing lookup, not marker numbering.
This distinction matters for asymmetric rank loads; equal BALANCED vectors
produce the same values on every member.

## `block_copy`: what it means and when it's safe

By default, `block_copy=True`. It is a **trace-generation** optimization, not
a trace-size one: the generator *builds* a block's rows once per distinct block
shape and appends that same list once per layer that shares it. The emitted
trace is the full per-layer thing either way — a 48-layer Qwen3-30B-A3B run
emits 583 trace lines with block copy on and 583 with it off. There is no
`block_copy` instruction in the trace or in Chakra.

What it saves is the per-layer latency lookups and size computations, which
dominate trace-generation time on a deep model.

This is **exact** for:

- Dense models (no MoE, every layer of a given block shape is identical).
- MoE with `BALANCED` (every block routes the same way, since
  BALANCED is deterministic and stateless).
- MoE with `RR` (the same gathered batch shape starts at token zero
  at every layer, so the routing is deterministic).

It's an **approximation** for:

- MoE with `RAND` (per-block randomness produces variance the copy
  can't capture).
- MoE with a `CUSTOM` per-token `_custom_gate_function` (depends
  entirely on what you wrote). The measured curve is deterministic in
  the batch's token count, so `block_copy` is exact for it, as it is
  for BALANCED.

For research where per-block variance matters, set
`--no-enable-block-copy` on the simulator. Trace generation runs more slowly and
every layer is built from its own router draw.

A heterogeneous stack never shares a block across differing layers, block copy
or not: the reuse is keyed on the layer's resolved block shape, so Qwen3.5's
gated-DeltaNet and full-attention layers are built separately even with the
optimization on.

## Per-rank latency lookup

Every rank's MoE block latency comes from
`profiler/perf/<hw>/<model>/<variant>/tp1/moe.csv` keyed on:

| Key | Meaning |
| --- | --- |
| `ep` | The instance's **total** EP degree — picks which grid to read |
| `tokens` | Gathered input rows, or replicated rows without dispatch |
| `activated_experts` | Number of *distinct* experts this rank touches |

Profiled at TP=1 (expert weights shard by `ep_size`, not `tp_size`) but
**once per EP degree**, because a rank runs a slice of the block rather than
the block: `E/ep` local experts and `k/ep` of a token's k assignments. The
simulator does 2D linear interpolation across `tokens` and
`activated_experts` inside the chosen grid, and falls back to the nearest
profiled `ep` with a one-shot warning rather than interpolating across it.

This matters most in decode. `activated_experts` is the only lookup axis whose
minimum is a positive number the runtime can go under — it floors at `top_k`,
since a token cannot activate fewer experts than it selects — and the lookup
clamps below the grid instead of extrapolating. A GLM-5 EP=2 run asked for
`activated=4` against a floor of 8 in **98.4%** of its MoE lookups. Reading the
ep=1 grid there over-charges the MoE term by 1.7x to 4.8x depending on the
model and degree.

The full MoE block latency is then **max(rank_latencies)** because
ranks execute in parallel and synchronize at the surrounding
collectives. Whichever rank gets the most tokens × experts dominates.

## What the EP all-to-all costs

MoE dispatch/combine is implemented with the configured
`allgather_reducescatter` backend, not one `ALLTOALL` trace node.
Which tensors are sent depends on the measured execution contract.

With a matching native component table, the order is local gate/routing,
three ordered AllGathers, gathered experts, ReduceScatter, then local
finalization. Dispatch retains separate hidden-state, top-k-weight and
top-k-ID tensors, using the contract's dtypes. Sequence-parallel wrappers
restore TP output with an AllGather; non-SP wrappers use a TP AllReduce when
needed. See [native components](../profiler/native-moe-components) and the
[payload formulas](./parallelism-mechanics#deployment-matched-native-components).

Without native coverage, the retained `moe.csv` fallback approximates dispatch
as one AllGather of hidden states plus full router logits. That legacy payload
is not the modular path's three tensors. At DP=1 the input is replicated:
there is no dispatch/combine pair, and local expert contributions are reduced
over TP when necessary.

Both paths use an analytical Ring envelope for unequal rank counts.
After local graph resolution and DP coordination, let `G` be the dispatch
group size and `remote_rows = sum(dispatch_rows) - min(dispatch_rows)`.
An AllGather tensor of width `W` bytes uses
`ceil(remote_rows * W / (G - 1))` as its equivalent local chunk.
ReduceScatter uses the hidden-state chunk multiplied by `G`.
Uniform counts recover the ordinary local chunk; graph padding does not imply
every prefill or mixed round has uniform counts.

This envelope does not reproduce grouped NCCL launch or arbitrary ragged
scheduling exactly. Operation-specific bandwidths change its effective time,
not its tensor geometry or dependency order. Consult
[DP+EP wave synchronization](./parallelism-mechanics#dpep-wave-synchronization)
for the padded row domains and measured coverage limits.

## Gotchas

1. **`block_copy` defaults to True** and suppresses per-layer randomness
   for RAND or a varying custom per-token policy. If you're studying load
   imbalance specifically, disable it. It does **not** change the trace for a
   deterministic router — until v1.2.0 disabling it emitted a single
   transformer block instead of `num_hidden_layers`, understating the clock
   by 3.1x on a 48-layer model.
2. **`activated_experts` is per-rank, not per-token.** A rank with
   100 tokens hitting 8 distinct experts reports `activated_experts
   = 8`, not 800. The latency lookup expects this convention.
3. **Native and legacy coverage differ.** Native component lookup requires
   a matching TP/DP/EP/rank contract under `tp<N>/`. Legacy `moe.csv` uses
   EP slices from the TP1 bundle and warns when substituting an unmeasured
   EP degree. Neither establishes native coverage for another deployment.
4. **`num_experts_per_tok` (top-K)** is read from the model's HF
   config. Deviating from the trained value is OK at simulation time
   but won't match the real model's behavior.
5. **Dummy batches in DP groups still route through the gate.**
   1-token dummy batches go through routing exactly like real
   batches, so DP+EP results are consistent across waves.

## What's next

- **[Parallelism mechanics](./parallelism-mechanics)**: what the
  collectives around the MoE block look like at the network level.
- **[Examples → Expert parallel](/docs/examples/parallelism/expert-parallel)** -
  the configuration angle (when to use which `ep_size`).
- **[Examples → DP+EP MoE](/docs/examples/parallelism/dp-ep-moe)** -
  multi-instance MoE.
