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
- A compatible profiled bundle. The repository includes historical RTXPRO6000
  Qwen3.8-27B BF16 tables at TP 1 and 2, recorded with vLLM 0.28. Their
  metadata does not establish acquisition compatibility with a newer runtime.

## Profiling it

```bash
# inside the vLLM container
python -m profiler profile Qwen/Qwen3.8-27B --hardware RTXPRO6000 \
  --tp 1 --out-root profiler/perf-native
```

Use a separate output root for a new runtime's acquisition. The committed
historical bundle remains available to the serving example below. Three
things to check in a new acquisition are:

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

:::caution[Hybrid stack depth increases acquisition cost]
Every attention shot must execute the layer prefix that exposes both block
types, so a hybrid can require substantially more attribution work than a
uniform one-layer stack. Grid counts also depend on resolved cache capacity
and feasibility; neither a fixed shot count nor a universal 4x runtime follows.
Use `--attention-chunk-factor` / `--attention-kv-factor` to coarsen the grid,
or reduce `--measurement-iterations` with an explicit repeatability check.
Neither runtime nor uncertainty is determined by that flag alone.
:::

Inspect coverage before a long run to check active bindings, alternative
layouts and uncovered CUDA work. Runtime admission and publication perform
additional required-component checks; coverage alone is not completion:

```bash
python -m profiler coverage Qwen/Qwen3.8-27B --hardware RTXPRO6000
```

## Run

```bash
python -m serving \
  --cluster-config 'configs/cluster/single_node_hybrid_instance.json' \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/hybrid_run.csv' \
  --num-reqs 10
```

Omitting `--block-size` is deliberate: the bundle's recorded 784 is
what the latencies were measured at.

## The part that is easy to get wrong

A GDN block can run a **different set of kernels** depending on the batch mix.
The historical Qwen3.8-27B decomposition is:

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

### Packed-core traces

An engine exposing the packed GDN operator can instead use `gdn_core`, including
conv, recurrence and gated norm. Projections and outer glue remain separate.
Positive runtime coverage selects the representation; new `dense` and
`linear_attention` tables must be acquired together under the
[layout contract](../../profiler/output-bundle#runtime-selected-component-layouts).
Stored historical timings are not rewritten.

The core's input is packed qkvz plus ba; its output is gated-normalized value rows.
Its owned weights are conv, dt_bias, float32 A_log and the replicated norm scale.
Other values use model activation dtype, even with fp8 linear weights. Persistent
recurrent-state capacity remains per sequence, not per token. Invalid dimensions,
nondivisible TP head counts and unverified quantized BA projection layouts are
refused; fp8 acceptance here is not certification of an entire quantized model.

Both historical and packed layouts attach the declared TP reduction after
`gdn_out_proj`, using the full hidden-state output at activation dtype. This can
change simulated communication without changing stored latency rows. The
MoE sibling's attention-wide TP+DP sequence-parallel path remains
[explicitly unsupported](../../simulator/parallelism-mechanics#attention-wide-sequence-parallelism);
component coverage alone does not certify that deployment.

## Caveats

:::caution[Prefix caching requires chunked prefill here]
`align` mode exists so a state checkpoint lands on a block boundary, and
the only lever for that is where a chunk ends. vLLM asserts
`enable_chunked_prefill` in the same block that selects `align`, and the
simulator refuses the combination for the same reason. Turn prefix
caching off instead and the mode becomes `none`, which checkpoints
nothing.
:::

:::note[Profile coverage is not end-to-end validation]
The model's profile bundle and deterministic simulator scenarios exercise the
hybrid stack, state sizing and chunk alignment. They do not establish real-system
accuracy: no end-to-end Qwen3.8 reference is included on the
[validation page](/docs/validation). Match the actual deployment's resolved
block size, state dtypes and graph configuration before recording a comparison.
:::

## Where to learn more

- **[KV cache & memory](/docs/simulator/scheduling/kv-cache-and-memory)**:
  the per-sequence state, page counting, and the `none`/`align` table.
- **[Continuous batching](/docs/simulator/scheduling/continuous-batching)**:
  the block-aligned chunk split.
- **[Profiler → Adding a model architecture](/docs/profiler/adding-model-architecture)**:
  how the `blocks:` axes describe a heterogeneous stack.
