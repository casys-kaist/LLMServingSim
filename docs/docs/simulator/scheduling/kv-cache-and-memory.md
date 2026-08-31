---
title: KV cache & memory
sidebar_position: 3
---

# KV cache & memory

Each `Scheduler` owns a `MemoryModel` that tracks how many bytes of
NPU and CPU (and optionally CXL) memory are in use at any moment.
This is what tells the scheduler when to stop accepting new requests
and what triggers prefix-cache evictions.

> Looking for memory-tier *configuration*? See
> **[Examples → CXL extended memory](/docs/examples/memory-tiers/cxl-memory)**
> for placement rules and
> **[Examples → Prefix caching](/docs/examples/memory-tiers/prefix-caching)**
> for the second-tier pool. This page is the byte-accounting side.

## Memory tiers

```mermaid
flowchart LR
    subgraph NPU["NPU memory (per-instance)"]
        WEIGHTS[Weights<br/>per-rank]
        ACTIVE[Active KV<br/>blocks]
        NPUCACHE[Prefix<br/>cache - tier 1]
    end
    subgraph CPU["CPU memory (per-node)"]
        CPUPOOL[Prefix pool<br/>tier 2]
        CPUKV[Spilled<br/>KV blocks]
    end
    subgraph CXL["CXL memory (per-node, optional)"]
        CXLW[Placed<br/>weights]
        CXLPOOL[Prefix pool<br/>tier 2]
    end
    NPUCACHE -->|"evict"| CPUPOOL
    NPUCACHE -.->|"evict (CXL pool)"| CXLPOOL
    ACTIVE -->|"swap-out"| CPUKV
    CPUPOOL -->|"hit"| NPUCACHE
    CXLW -.->|"weight load"| WEIGHTS
```

Three tiers, each represented by a separate counter on the
`MemoryModel`:

| Tier | Object | Capacity from | Holds |
| --- | --- | --- | --- |
| **NPU** | `npu_used` | `npu_mem.mem_size` × `num_npus` | Weights (per-rank), active KV cache, NPU prefix cache |
| **CPU** | `cpu_used` | `cpu_mem.mem_size` (per node) | CPU prefix pool, evicted KV blocks, model weight staging |
| **CXL** *(optional)* | `cxl_used[device_id]` | `cxl_mem.mem_size` × `num_devices` | CXL-resident weights / KV / prefix pool depending on placement |

Capacity comes from the cluster config; usage is tracked at runtime.
Exceeding capacity at startup (e.g., `weight_per_gpu > npu_mem`) is
a fatal error. Exceeding it at runtime triggers eviction (for the
prefix cache) or scheduler back-pressure (for active KV).

## What's in NPU memory

Two big consumers, in this order of priority:

### 1. Model weights (per-GPU)

Computed at scheduler init via
`MemoryModel.get_weight()`. The size is the model's full parameter
count divided by `tp_size` (and for MoE: experts further divided by
`ep_size`), times the dtype byte size:

```
weight_bytes_per_gpu = (
    dense_params / tp_size
    + moe_params / ep_size  # if MoE
) * fp
```

`fp` is 2 bytes for `bfloat16` / `float16`, 4 for `float32`,
1 for `int8` and `fp8`.

`get_weight()` gets the parameter count by **walking the architecture yaml's
blocks and the checkpoint's own per-layer composition**, summing
`calculate_sizes()` over every canonical layer each decoder layer emits. So a
heterogeneous stack is weighed layer by layer: DeepSeek-V3.2's first three
layers carry a dense MLP and its other 58 carry 256 experts. It is one built
weight per distinct block *shape*, not per layer, so the walk stays cheap.

Some layers carry parameters that a Llama-shaped block has nowhere to put —
MLA's `mla_qkv_a_proj` is replicated rather than sharded, its `q_b_proj` and
`kv_b_proj` read low-rank latents, and DeepSeek's sparse indexer adds three
more projections. That is why the weight is read from the catalog rather than
from a fixed list of six layer names.

:::tip[Check a new family against its published parameter count]
It is the one number that catches a wrong tensor shape anywhere in the stack,
and it is public. Summing the size formulas over DeepSeek-V3.2-Exp's layer
composition gives 671.878B; subtracting the sparse indexer's 0.852B leaves
671.026B, which is DeepSeek-V3's published 671B — and the difference is exactly
what V3.2 adds over V3.
:::

This bytes amount is reserved on every NPU at startup and never
freed. If `weight_per_gpu > npu_mem.mem_size`, the simulator exits
with a clear error message, the typical fix is to bump `tp_size`,
add CXL placement rules, or pick a smaller model.

### 2. Active KV cache

Per-request KV cache, tracked at block granularity. The block size
is `--block-size` tokens (default 16):

```
bytes_per_block = (
    2                                         # K and V
    * num_layers
    * num_key_value_heads
    * head_dim
    * block_size
    * kv_fp
) / num_npus                                  # = tp_size * pp_size
```

The divisor is `num_npus`, i.e. **`tp_size * pp_size`**, not `tp_size`
alone (`MemoryModel.get_kv()`). Both factors belong there: KV per layer
shards across TP ranks, and the layers themselves are split across PP
stages, so a rank in a `tp=2 x pp=2` instance holds a quarter of the
model's KV, not a half.

Note also that `head_dim` is read explicitly from the model config, not
derived — see **[Model config](/docs/reference/model-config)**. On Qwen3
`hidden_size / num_attention_heads` gives the wrong answer.

### KV is not one shape

That formula is grouped-query attention, and it is what most families use. It
is not universal, and `kv_bytes_per_token_per_layer()` resolves which shape a
layer has from the checkpoint:

| Attention | Bytes per token per layer | Sharded by TP? |
| --- | --- | --- |
| GQA | `2 * num_key_value_heads * head_dim * kv_fp` | yes |
| **MLA** (DeepSeek-V3.2, GLM-5) | `(kv_lora_rank + qk_rope_head_dim) * kv_fp` — one latent, no separate V | **no** — `num_kv_heads` is 1, so every rank holds the same bytes |
| **+ sparse indexer** (the same two) | plus `index_head_dim + index_head_dim/128 * 4` — a **second** cache beside the latent, fp8 keys with one fp32 scale per 128 elements | no |
| **Linear attention** (Qwen3.5/3.8 gated DeltaNet) | 0 — its state is fixed *per sequence*, not per token | n/a |

The MLA row is the reason this matters rather than being a detail: sizing
DeepSeek-V3.2 with the GQA formula reads 1,748,992 bytes per token where it
actually caches 78,324, a 22x overstatement of the thing that decides how many
requests fit.

### 3. Per-sequence layer state

A linear-attention layer caches nothing per token, but it does hold a fixed
state for as long as the sequence lives — a rolling convolution state plus a
recurrent state. Their size does not depend on the sequence length, which is
why the layer's *cost* does not either, and it is not small:

```
conv_state = (2*key_head_dim*num_key_heads + value_head_dim*num_value_heads)
             * (conv_kernel_dim - 1)
ssm_state  = num_value_heads * value_head_dim * key_head_dim
```

The two do **not** share a dtype. The conv state follows
`mamba_cache_dtype` (`auto` → the weight dtype) and the recurrent state
follows `mamba_ssm_dtype` (`auto` → the *conv* dtype, not the weight dtype),
which is vLLM's `MambaStateDtypeCalculator._mamba_state_dtype`. Qwen3.8-27B
declares `mamba_ssm_dtype: float32`, so its recurrent state is 4 bytes per
element against the conv state's 2 — and it is the large half, 786,432 elements
per layer against 30,720. That comes to 3.21 MB per sequence per layer, and 48
of its 64 layers are gated DeltaNet: **153.9 MB per concurrent sequence**. Its
KV per token is 65,536 bytes, from the 16 full-attention layers only.

So it bounds **concurrency** the way a KV cache bounds context.

#### It is counted in pages, not in bytes

vLLM picks the attention block size so that one attention page covers one mamba
page, then pads the mamba page up to it:

```
attn_block_size       = alignment * cdiv(mamba_page_size,
                                         alignment * attn_page_size_1_token)
mamba_page_size_padded = attn_page_size
```

so a layer's whole state occupies **exactly one page** and the padding is
really allocated. On Qwen3.8-27B the mamba page is 3,207,168 bytes against an
attention page of 3,211,264 at `block_size 784` — and that is where 784 comes
from: `16 * cdiv(3,207,168, 16 * 4,096) = 784`. That is why the block size is
read out of the profile bundle rather than defaulted, though the four bundles
shipped in this repo predate the `engine_resolved` field and still fall back to
16.

How many pages per layer is `MambaSpec.max_memory_usage_bytes`, and it depends
on the cache mode:

| `mamba_cache_mode` | Pages per mamba layer | When |
| --- | --- | --- |
| `none` | `1 + N` | prefix caching **off** |
| `align` | `2 + N` | prefix caching **on** — the default |
| `all` | `cdiv(max_model_len, block_size)` | opt-in, not modelled |

`align` holds two because one page carries the state being written this step
and the other the last checkpoint committed at a block boundary — which is what
a later prefix hit resumes from. `N` is `--num-speculative-tokens`, one extra
page per draft token, and it also **widens the conv state itself**
(`conv_kernel_size - 1 + N`).

On Qwen3.8-27B at `block_size 784` that is 48 layers × 2 pages = 96 pages, or
**6 pool blocks per request** with prefix caching on against 3 with it off.
Charging one page per layer — the caching-off figure — understated the default
configuration by exactly 2x.

Those blocks live in a separate list from the token blocks, because the token
list is positional — block `i` backs tokens `[i*block_size, (i+1)*block_size)`
— and a state block backs no tokens, so nothing may hash or index it.

Models whose layers all cache per token charge 0 here, which is every family
except the gated-DeltaNet hybrids.

### 4. The drafter's own KV cache

With `--num-speculative-tokens` on, a model that drafts with itself runs MTP
modules, and an MTP module wraps a **real decoder layer**
(`DeepseekV2DecoderLayer`, `Glm4DecoderLayer`, `MiniMaxM3DecoderLayer`,
`Qwen3_5DecoderLayer`), so it publishes a KV cache spec of its own and vLLM
allocates for it:

| Model | MTP modules | Extra bytes per token |
| --- | --- | --- |
| DeepSeek-V3.2-Exp | 1 | +1.6% |
| MiniMax-M3 | 7 | +11.7% |
| Qwen3.8-27B | 1 | +6.2% |

The drafter's attention is **full attention** whatever the target's layers are
— Qwen3.5's MTP builds its block with `layer_type="full_attention"` explicitly
— so a hybrid's drafter carries a KV cache but no recurrent state. That is why
Qwen3.8's 6.2% is larger than DeepSeek's 1.6% despite both having one module:
only 16 of Qwen3.8's 64 layers cache per token, so one more is 1/16.

Where `kv_fp` is:

- 2 bytes for `bfloat16` / `float16` — the checkpoint's weight dtype,
  which the KV cache inherits unless the config says otherwise; 4 for
  `float32`.
- **1 byte when the checkpoint declares an fp8 KV cache**
  (`quantization_config.kv_cache_scheme` or `kv_cache_quant_algo`),
  which halves KV memory against
  a 16-bit weight dtype.

How many blocks exist is fixed at startup, the way vLLM sizes its cache:

```
requested   = npu_mem.mem_size * npu_mem.mem_util      # per rank
kv_bytes    = requested - model weight
num_blocks  = kv_bytes / bytes_per_block
```

`npu_mem.mem_util` defaults to `--npu-memory-utilization` (`0.9`), the
analogue of vLLM's `--gpu-memory-utilization`. vLLM additionally subtracts
its activation peak and CUDA context, which the simulator does not model,
so this capacity is an **upper bound** on vLLM's at the same fraction. The
run prints the resulting figure per instance under **KV Cache
Initialization** at startup:

```
  • Instance [0] : 585248 tokens / 36578 blocks (71.44 GiB/rank at util 0.90)
```

:::caution[Calibrate `mem_util` when the KV cache is the binding constraint]
`mem_util` only changes behaviour once a run actually **saturates** the KV
cache. Below that, the pool never runs out, nothing is preempted, and the
capacity you configured is invisible in the results. On a card with plenty of
headroom the default `0.9` is perfectly fine.

When a run does hit the ceiling, the number matters a great deal — and the
default is not the right one. Because the simulator does not model vLLM's
activation peak or CUDA context, `0.9` here buys noticeably more KV cache than
`0.9` does in vLLM. More cache means less preemption, so the simulated run
finishes early and every latency metric skews with it.

Set it from the measured run in that case. `python -m bench run` records what
vLLM actually resolved in `meta.json`:

```json
"kv_cache": { "num_gpu_blocks": 2588, "block_size": 16, "num_kv_tokens": 41408 }
```

Then pick the `mem_util` whose **KV Cache Initialization** line reports that
same block count. The bundled RTX 4090 / Llama-3.1-8B example is exactly this
case — 24 GB, pinned at its ceiling for most of the run — and the matched value
is `0.833919`, not `0.9`:

| | KV tokens | blocks | TTFT mean | TPOT mean | Latency mean |
| --- | --- | --- | --- | --- | --- |
| vLLM (measured) | 41,408 | 2,588 | — | — | — |
| `mem_util: 0.9` | 54,400 | 3,400 | -20.7% | +12.9% | -12.5% |
| `mem_util: 0.833919` | 41,408 | 2,588 | **+0.6%** | **+0.2%** | **+0.5%** |

Same profile bundle, same workload, same engine flags — only the capacity
differs. The bundled RTXPRO6000 examples, by contrast, peak at 58-97% of their
budget, so they are left at `0.9`: there is nothing for a calibration to
change. Read the peak `Each NPU Memory Usage` percentage off the heartbeat to
tell which case you are in — remember its ceiling is `mem_util * 100`, not 100.
:::

The scheduler takes `ceil(tokens / block_size)` blocks per active request
from the pool's free list, and returns them when the request finishes or is
preempted.

### 5. NPU prefix cache

Not a separate allocation. A block that becomes full is indexed under a
**chained hash** of its tokens, and it keeps that index entry after its
request returns it to the free list — so it is simultaneously reusable and
still findable. A later request whose prefix hashes the same way claims it
back instead of recomputing.

Eviction is therefore a **side effect of allocation**: when the pool hands
out a block from the head of its free list and that block still carries a
hash, the hash is dropped. There is no separate eviction pass and no
"evictable size" to estimate. With `--prefix-storage`, the block also has a
copy on the CPU or CXL tier, so a later request can still find it there.

Full mechanics: **[Prefix caching](./prefix-caching)**.

## What's in CPU / CXL memory

Per-node CPU memory (and per-device CXL memory) hold:

- The shared **second-tier prefix cache** if
  `--enable-prefix-sharing` is on.
- **Spilled KV blocks** from NPU evictions (when offloading is
  enabled).
- **Weights placed there explicitly** via the `placement` field in
  the cluster config (e.g., `"weights": "cxl:0"` for some decoder
  blocks on CXL device 0). See
  **[Examples → CXL memory](/docs/examples/memory-tiers/cxl-memory)**
  for the placement rule syntax.

Unlike NPU memory, CPU/CXL accounting is **per node**, not per
instance. Multiple instances on the same node share the same
`cpu_used` counter.

## How the scheduler uses this

The scheduler does not estimate. It asks the pool, and the pool either
hands over the blocks or reports failure in the same call:

```python
blocks = kv.allocate_slots(request, tokens_to_run_this_step)
if blocks is None:
    ...   # nothing was mutated; the caller decides what to do
```

`num_free_blocks` is exact, so a queued block *is* allocatable. That
all-or-nothing property is what makes the two callers behave differently
and correctly:

- **A running request** that cannot get a block causes a preemption: the
  scheduler drops the tail of its running set, returning that request's
  blocks, and retries.
- **A waiting request** that cannot get a block is simply not admitted this
  step. Admission never preempts, and it stops at the first failure.

Admission also refuses a request whose *whole* sequence would not fit, not
merely its first chunk (`--reserve-full-isl`, on by default, mirroring
vLLM's `scheduler_reserve_full_isl`). Checking only the first chunk lets
chunked prefill admit a request that then grows past capacity, which turns
into a preemption later.

Memory therefore sits at the pool ceiling once a workload has filled it,
rather than oscillating. A block in the free list that still carries a hash
is reusable data, not waste.

## Per-instance vs per-node accounting (gotcha)

The NPU block pool is **per-instance**. Two instances on the same node
have completely separate NPU accounting, even though they're on the same
physical GPU. `npu_used` is derived from the pool rather than tracked
alongside it, so there is exactly one ledger per tier.

`cpu_used` is **per-node**. Two instances on the same node share one
CPU memory budget. If both have spilled prefix blocks to CPU, they
compete for the same `cpu_mem.mem_size` capacity.

This matters for multi-instance configs: `num_instances: 4` with
each instance reserving 60 GB of NPU memory implies each instance
gets its own GPU; but they all share the node's `cpu_mem.mem_size`
GB of host memory.

## Reading memory in the throughput log

The heartbeat block emitted every `--log-interval` simulated seconds
carries one branch per instance and then one per node:

```text
[1.0s] Avg prompt throughput: 9069.0 tokens/s, Avg generation throughput: 224.0 tokens/s
        ├─Running Instance[0]: 9 reqs, Waiting: 0 reqs, Total # 1 NPUs, Each NPU Memory Usage 16486.51 MB (16.771 % Used), Prefix Cache Hit ratio 0.00 %, (0 / 9069)
        └─Node[0]: Total CPU Memory Usage 0.00 MB, 0.000 % Used
```

Reading the memory fields specifically:

- **`Each NPU Memory Usage`** is per rank, in **MB**, and covers weights
  plus active KV plus indexed prefix blocks — the whole `npu_used`
  ledger. The percentage is against `npu_mem.mem_size`, so it tops out
  near `mem_util * 100`, not 100.
- **`Node[i]: Total CPU Memory Usage`** is the node's lower-tier
  total. On a multi-instance node it is followed by each instance's
  share of that total, e.g.
  `(Instance[0]: 61.20 %, Instance[1]: 38.80 %)`.
- With `--prefix-storage CXL` a `CXL[...]` branch is added, one per
  device under `--enable-prefix-sharing` and one per prefix-caching
  instance otherwise.

Full field-by-field reference, including the multi-instance and CXL
variants: **[Reading the output](/docs/simulator/reading-output#heartbeat-block)**.

## Gotchas

1. **OOM at startup** is always a weights-vs-NPU-capacity issue. The
   error message points at exact byte counts; bump `tp_size` or
   reduce model size.
2. **OOM mid-run** is unusual but possible if CXL placement is
   misconfigured. Check the `CXL[...]` branch of the heartbeat block,
   which appears only with `--prefix-storage CXL`.
3. **`block_size` matters for memory granularity, not throughput.**
   Smaller blocks = finer accounting but more overhead per request.
   Default 16 is what vLLM uses.
4. **FP8 KV cache halves the KV byte budget**, but you also need a
   profile bundle for the `*-kvfp8` variant
   (e.g., `bf16-kvfp8`). Without it, the simulator errors out with a
   variant-not-found message.
5. **Weight memory is fixed for the run.** It doesn't grow when you
   add more requests; only KV cache does. The "weight ceiling"
   visible at the top of the throughput log line stays constant.

## What's next

- **[Trace generation](../trace-generation)**: how the latency for
  each iteration is calculated *given* a memory state.
- **[Examples → Prefix caching](/docs/examples/memory-tiers/prefix-caching)**
  and **[CXL memory](/docs/examples/memory-tiers/cxl-memory)**: the
  configuration angle.
