---
title: Sparse attention
sidebar_position: 2
---

# Sparse attention (DSA and block-sparse MSA)

> **What this demonstrates:** three model families whose attention reads
> only part of the KV cache — and the fact that "sparse attention" is not
> one mechanism. The selection granularity differs, and it is the
> granularity that maps onto a block pool.

## Two shapes, three checkpoints

| | DeepSeek-V3.2-Exp, GLM-5 | MiniMax-M3 |
| --- | --- | --- |
| Mechanism | DeepSeek Sparse Attention (DSA) | block-sparse MSA |
| Selects | top **2048 tokens** | top **16 blocks of 128 tokens** |
| Over | MLA (one latent, no separate V) | plain GQA |
| Index cache | `index_head_dim` fp8 keys + fp32 scales, packed `uint8` | one `sparse_index_dim` vector per token, **bf16** |
| Historical attention components | `attention` (MLA), `indexer` | `attention`, `sparse_attention`, `indexer` |
| `model_type` | `deepseek_v32`, `glm_moe_dsa` | `minimax_m3_vl` |

DeepSeek-V3.2 and GLM-5 share **one catalog** for their recorded vLLM 0.28
`deepseek_v2.py` decomposition. The catalog lists both rotary-embedding classes
used by those checkpoints. This does not assert that later vLLM versions execute
the same modules: vLLM 0.31 uses a native DeepSeek-V3.2 path under
`vllm.models.deepseek_v32.nvidia.model`. Stored bundles remain readable, but new
acquisitions require matching live coverage and execution contracts. The native
DeepSeek representation is described by its
[four-table acquisition contract](../../profiler/output-bundle.md#native-dsa-acquisition):
it uses a combined, full-context `dsa_attention_tail` and explicit local duration
replay only. This does not extend native acquisition to GLM-5.

MiniMax-M3's new runtime-checked vLLM 0.31 Triton BF16 acquisition records a
[per-TP attention contract](../../profiler/output-bundle.md#native-attention-lookup-contract):
the configured selected-block count and reviewed 128-token block layout determine
the sparse-prefill lookup cap (2048 for the shipped configuration). The
historical unmarked tables retain 2176. Raw producer coordinates, indexer
history and decode lookup remain unchanged; selecting a native cap does not
certify a full grid, distributed deployment, KV capacity or benchmark accuracy.

:::caution[The `attention` category holds more than one kernel, and they are not interchangeable]
Every lookup is keyed by kernel name, and so is the skew alpha. On
MiniMax-M3 the same batch fits alphas of 0.24 / 0.74 / -0.01 for
`attention` / `indexer` / `sparse_attention`. There is deliberately no
pooled table: there was one, every lookup took it, and a sparse layer
got the dense kernel's latency — 2.1x per layer on M3.
:::

## What the KV cache actually holds

The capacity and deployment examples below describe the historical simulator
path. They do not admit the new native DSA tables for serving: their FP8 main
cache is 656 bytes plus a 132-byte index cache, whereas the legacy FP8 formula
counts 576+132. Native duration replay does not perform capacity admission;
ordinary serving refuses these new tables.

Sizing these as GQA is the single biggest error available here. Per
token per layer, in bf16:

| Shape | Bytes | Sharded by TP? |
| --- | --- | --- |
| MLA latent (`kv_lora_rank + qk_rope_head_dim`) | 1,152 | **no** — `num_kv_heads` is 1, so every rank holds the same bytes |
| + DSA indexer cache | 132 | no |
| M3 GQA (`2 * kv_head * head_dim`) | 2,048 | yes |
| + M3 indexer cache | 256 | no |

Reading DeepSeek-V3.2 with the GQA formula gives 1,748,992 bytes per
token where it actually caches 78,324 — a 22x overstatement of the thing
that decides how many requests fit.

The indexer caches follow **neither** the weight dtype nor
`--kv-cache-dtype`: DeepSeek's is constructed `uint8` outright, M3's asks
vLLM for `resolve_indexer_kv_dtype("bf16")`. An fp8-KV run must not
shrink them.

## Placement: expert weights decide everything

These are all MoE models, and the split is lopsided:

| Model | dense | experts | expert share |
| --- | --- | --- | --- |
| DeepSeek-V3.2-Exp (fp8) | 14 GB | 611 GB | 97.7% |
| GLM-5 (bf16) | 30 GB | 1355 GB | 97.8% |
| MiniMax-M3 (bf16) | 18 GB | 776 GB | 97.7% |

Expert weights shard by **EP**, not TP, so TP buys almost nothing here —
dense is 20-30 GB whatever you do. Maximising DP is the natural layout:
with N GPUs and `pp_size 1`, `tp_size = 1` and `dp_group_size = N` forces
`ep_size = N` (the constraint is `ep % dp == 0` and `ep / dp <= tp`), so
each instance is one GPU holding 1/N of the experts.

Per-rank weight on 96 GB cards, that layout:

| Model | N=8 | N=16 | N=32 | N=64 |
| --- | --- | --- | --- | --- |
| DeepSeek-V3.2 | 91 GB ✗ | **52 GB** | 33 GB | 24 GB |
| MiniMax-M3 | 115 GB ✗ | **67 GB** | 42 GB | 30 GB |
| GLM-5 | 200 GB ✗ | 115 GB ✗ | **73 GB** | 51 GB |

**Every instance runs at `tp_size = 1`, so only `tp1/` is ever read.**
That is not a coincidence: MoE latency is profiled once at tp=1 and
looked up per rank by local token count, and `ep_size` never appears in
a profile path — it feeds the gate and the collective sizes.

## Group-limited routing

DeepSeek-V3/V3.2 and GLM restrict a token's experts to `topk_group` of
`n_group` groups. It changes how many EP ranks one token reaches, and
therefore the per-rank MoE work and the size of the collective around the
block:

| Model | `E` / top-`k` | `n_group` / `topk_group` | P(token reaches a rank) at EP=8 |
| --- | --- | --- | --- |
| DeepSeek-V3.2-Exp | 256 / 8 | **8 / 4** | **0.454** — a 31% cut |
| GLM-5 | 256 / 8 | 1 / 1 | 0.662 |
| MiniMax-M3 | 128 / 4 | — | 0.250 |

Only DeepSeek-V3.2 actually restricts; GLM-5 ships `n_group: 1`, the
unrestricted case spelled out, and M3 does not declare the fields.

## Profiling

These are acquisition templates, not a guarantee that a historical catalog
matches the installed runtime. Verify the selected model's coverage and component
boundaries first; do not publish a table with missing or double-counted work.

```bash
# Native DeepSeek-V3.2 — fresh admitted tables, local duration replay only
python -m profiler profile deepseek-ai/DeepSeek-V3.2-Exp \
  --hardware RTXPRO6000 --tp 1 --ep 1 --out-root profiler/perf-native

# MiniMax-M3 — block size 128 is REQUIRED
python -m profiler profile MiniMaxAI/MiniMax-M3 --hardware RTXPRO6000 \
  --tp 1 --block-size 128 --out-root profiler/perf-native
```

Inspect coverage before a sweep to check active bindings and uncovered CUDA
work. Runtime admission and category publication also validate required
components; a coverage report alone does not establish a complete bundle:

```bash
python -m profiler coverage MiniMaxAI/MiniMax-M3 --hardware RTXPRO6000 --block-size 128
```

:::caution[Two per-family requirements that fail loudly if missed]
- **MiniMax-M3 needs `--block-size 128`.** Its sparse selection works in
  `sparse_block_size: 128` blocks, and the platform default of 16 fails
  outright with `No common block size for 16`.
- **`configs/model/MiniMaxAI/MiniMax-M3.json` must stay nested**, keeping
  its `text_config` key. vLLM's `MiniMaxM3Config` is a wrapper that builds
  its backbone from that key and sends everything else to `**kwargs`, so a
  flattened file silently yields an all-defaults backbone — 60 layers, 128
  experts, none of your values. Qwen3.5's config class is the opposite and
  wants a flat file; `stack.text_config` handles both.
:::

Ordinary historical TP profiling is single-GPU — per-rank shapes are
emulated by dividing the declared sharded dimensions, including attention
head counts and MLP intermediate widths, through `hf_overrides`;
`hidden_size` stays unchanged. Collectives are left to ASTRA-Sim. So a 671B model
profiles on one card: the profiler shrinks the stack to the 4 layers
needed to reach every block type (3 dense MLP + 1 MoE, sparse throughout).
The native DSA acquisition contract currently permits only actual/logical TP1
and EP1; it does not certify emulated wider TP or sliced expert geometry.

## Run

This serving example requires a compatible historical bundle; the new native
DSA representation is restricted to the explicit duration-replay API.

```bash
python -m serving \
  --cluster-config '<your max-DP config>' \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/sparse_run.csv' \
  --num-reqs 10
```

There is no sparse-attention flag: which layers run a selection branch
comes from the checkpoint (`sparse_attention_config.sparse_attention_freq`
for M3, `index_topk` plus `index_topk_pattern` / `index_topk_freq` for
DeepSeek/GLM), resolved per layer by `profiler/core/stack.py`.

## Caveats

:::note[The top-k caps what attention reads, not what is stored]
Both mechanisms select at read time. The KV cache still holds every
token, and the indexer's side cache is an *addition* to it, not a
replacement — M3 stores a GQA cache **and** an index cache. Proposals
that offload the unselected majority to a lower tier are research and
framework work, not behavior modeled by this acquisition and simulation path.
:::

:::note[Profile coverage is not full-model accuracy validation]
The repository includes RTXPRO6000 profile bundles for DeepSeek-V3.2, GLM-5
and MiniMax-M3. Inspect each bundle's metadata for measured precision,
parallelism, coverage and acquisition protocol; catalog coverage alone does
not establish end-to-end accuracy. A reduced, dummy-weight DeepSeek diagnostic
is reported on the [validation page](/docs/validation#additional-diagnostic-deepseek-v32-exp-16l64e)
and retains significant error. It does not validate full-size DeepSeek,
GLM-5 or MiniMax-M3 deployments.
:::

## Where to learn more

- **[MoE expert routing](/docs/simulator/moe-expert-routing)**: the gate,
  group-limited routing, and the EP collectives.
- **[KV cache & memory](/docs/simulator/scheduling/kv-cache-and-memory)**:
  every KV shape and how each shards.
- **[DP+EP MoE](/docs/examples/parallelism/dp-ep-moe)**: the wave
  synchronisation a max-DP layout depends on.
