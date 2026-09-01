---
title: Hybrid linear attention
sidebar_position: 1
---

# Hybrid linear attention (gated DeltaNet)

> **What this demonstrates:** a stack that is not N identical blocks.
> Qwen3.8-27B interleaves 48 gated-DeltaNet layers with 16 full-attention
> ones, and almost every part of the simulator has to know which is which.

## Why this family is different

A gated-DeltaNet layer caches **nothing per token**. It holds a rolling
convolution state plus a recurrent state whose size does not depend on
sequence length — which is exactly why its *cost* does not either — but
they are held for as long as the sequence lives. Three consequences,
each of which the simulator models separately:

**1. It bounds concurrency, not context.** On Qwen3.8-27B the state is
3.21 MB per sequence per layer, and 48 of its 64 layers are GDN:
**153.9 MB per concurrent sequence**. Its KV per token is 65,536 bytes,
from the 16 full-attention layers only.

The two halves do **not** share a dtype. The conv state follows
`mamba_cache_dtype` and the recurrent state `mamba_ssm_dtype`, which
`auto` resolves to the *conv* dtype rather than the weight dtype.
Qwen3.8 declares `float32` there, and the recurrent state is 98% of the
total — reading that wrong halves the figure.

**2. The block size is not 16.** vLLM raises the attention block size
until one attention page covers one mamba page, then pads the mamba page
to match:

```
alignment = max(min(backend supported block sizes), your --block-size)
block_size = alignment * cdiv(mamba_page_size, alignment * attn_page_1_token)
           = 16 * cdiv(3,207,168, 16 * 4,096)
           = 784
```

So one GDN state occupies exactly one pool block, and
`MambaSpec.max_memory_usage_bytes` decides how many: `1 + N` per layer
with prefix caching off, `2 + N` with it on — one page for the state
being written this step, one for the last checkpoint at a block
boundary. On Qwen3.8 that is 48 × 2 = 96 pages, i.e. **6 pool blocks
per request**, which the run logs:

```text
NPU: 48 mamba layer(s) x 2 page(s) = 96 page(s) per request (align mode) -> 6 block(s)
```

**3. Prefill chunks must end on a block boundary.** A state slot holds
the state after exactly `(p + 1) * block_size` tokens and state is
written at a chunk end, so with prefix caching on
(`mamba_cache_mode: "align"`) the scheduler floors each chunk to a
boundary. At `block_size 784` with a 2048-token budget a chunk is **784
or 1568, never 2048** — batch composition differs from a dense model on
the same workload.

## Prerequisites

- Simulator container set up
- A profiled bundle for the model. **The repo does not ship one**, so
  profile it first (see below).

## Profiling it

```bash
# inside the vLLM container
MODEL="Qwen/Qwen3.8-27B" HARDWARE=RTXPRO6000 ./profiler/profile.sh
```

Three things happen automatically and are worth knowing about:

- **The layer count resolves to 4, not 1.** A uniform stack is shrunk
  to one layer; a hybrid needs the smallest *prefix* that instantiates
  every distinct block, and Qwen3.8's `layer_types` runs GDN three
  times before the first full-attention layer. The run logs
  `-> profiling 4 layers to reach every block type`.
- **The engine overrides your block size.** Pass 16 and the log reports
  `block_size=784`. The bundle records it in
  `meta.yaml::engine_resolved.per_tp[tp]` — per TP degree, because both
  pages scale with the rank's shard — and the simulator reads back the
  entry for the instance's `tp_size`.
- **`linear_attention.csv` appears**, keyed `(prefill_tokens, n_decode)`.

:::caution[A hybrid costs roughly 4x a uniform stack to profile]
Not because the grid is bigger — measured, the attention sweep is 8,643
shots either way. Because every shot's forward runs all four layers,
and the catalog binds 24 entries against a dense Qwen3's 14, so the
layerwise profiler attributes proportionally more nodes per forward.
The bottleneck is that single-threaded attribution, not the GPU. Reach
for `--attention-chunk-factor` / `--attention-kv-factor` to coarsen the
grid, or `--measurement-iterations 1` to drop the 3x averaging at the
cost of 15-25% per-shot noise.
:::

Check the catalog binds every kernel before committing to a long run —
this is the only check that catches an entry naming a real class and
measuring **nothing**:

```bash
python -m profiler coverage Qwen/Qwen3.8-27B --hardware RTXPRO6000
```

## Run

```bash
python -m serving \
  --cluster-config 'configs/cluster/single_node_single_instance.json' \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/hybrid_run.csv' \
  --num-reqs 10
```

Omitting `--block-size` is deliberate: the bundle's recorded 784 is
what the latencies were measured at.

## The part that is easy to get wrong

A GDN block runs a **different set of kernels** depending on the batch
mix — not the same kernel at a different size. Measured on Qwen3.8-27B:

| kernel | prefill | decode | mixed |
| --- | --- | --- | --- |
| `gdn_conv_prefill`, `gdn_post_conv`, `gdn_prefill` | yes | — | yes |
| `gdn_conv_decode`, `gdn_decode` | — | yes | — |
| `gdn_decode_mixed` | — | — | yes |
| `gdn_in_proj`, `gdn_out_proj`, `gdn_norm`, `gdn_glue` | yes | yes | yes |

The regime-dependent six live in the `linear_attention` category, whose
key includes the mix; a kernel that does not fire in a regime simply has
**no rows** for it, and the simulator emits nothing for it there. The
always-on four live in `dense`, which has no notion of regime. Putting a
regime-dependent kernel in `dense` charges a decode convolution on a
pure prefill and a prefill convolution on a pure decode.

`python -m profiler coverage` reports per regime, which is how that
split was determined in the first place.

## Caveats

:::caution[Prefix caching requires chunked prefill here]
`align` mode exists so a state checkpoint lands on a block boundary, and
the only lever for that is where a chunk ends. vLLM asserts
`enable_chunked_prefill` in the same block that selects `align`, and the
simulator refuses the combination for the same reason. Turn prefix
caching off instead and the mode becomes `none`, which checkpoints
nothing.
:::

:::note[No real-system validation yet]
The rules above are transcriptions of vLLM's, verified against its
source and unit-tested — the chunk-alignment split against vLLM's over
14,310 input combinations, the page arithmetic against
`MambaSpec.max_memory_usage_bytes`. What has *not* happened is an
end-to-end comparison against a real vLLM run of this model, which is
what `python -m bench` is for.
:::

## Where to learn more

- **[KV cache & memory](/docs/simulator/scheduling/kv-cache-and-memory)**:
  the per-sequence state, page counting, and the `none`/`align` table.
- **[Continuous batching](/docs/simulator/scheduling/continuous-batching)**:
  the block-aligned chunk split.
- **[Profiler → Adding a model architecture](/docs/profiler/adding-model-architecture)**:
  how the `blocks:` axes describe a heterogeneous stack.
