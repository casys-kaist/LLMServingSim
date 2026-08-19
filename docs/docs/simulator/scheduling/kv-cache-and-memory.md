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
) * fp_size_in_bytes
```

`fp_size` is 2 bytes for `bfloat16` / `float16`, 4 for `float32`,
1 for `int8` and `fp8`. Actually loading is done via `get_weight()`,
which reads the model config and accounts for shared embeddings,
tied weights, etc.

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
    * num_key_value_heads / tp_size           # GQA shards by TP
    * head_dim
    * block_size
    * kv_fp_size
)
```

Where `kv_fp_size` is:

- 2 bytes for `--kv-cache-dtype auto` (inherits from `--dtype`).
- **1 byte for `--kv-cache-dtype fp8`**: halves KV memory.

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
  • Instance [0]                  : 3400 blocks / 54400 tokens (6.64 GiB per rank) at NPU memory utilization 0.90
```

The scheduler takes `ceil(tokens / block_size)` blocks per active request
from the pool's free list, and returns them when the request finishes or is
preempted.

### 3. NPU prefix cache

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

The throughput log line emitted every `--log-interval` seconds shows
running memory usage:

```
[INFO] step=42 batch=8 prompt_t=1.2k tok/s decode_t=420 tok/s
       npu_mem=88.4 GB cpu_mem=12.4 GB
```

For multi-instance setups it lists per-instance NPU usage:

```
       npu_mem=[88.4 GB, 87.9 GB] cpu_mem=24.8 GB
```

If you're using CXL:

```
       npu_mem=12.4 GB cxl_mem=[3.2 GB, 3.1 GB, 3.1 GB, 3.2 GB]
```

(`12.4 GB` is the surviving NPU active KV + cache, with weights now on
CXL.)

## Gotchas

1. **OOM at startup** is always a weights-vs-NPU-capacity issue. The
   error message points at exact byte counts; bump `tp_size` or
   reduce model size.
2. **OOM mid-run** is unusual but possible if CXL placement is
   misconfigured. Check the per-device CXL counter in the throughput
   log.
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
