---
title: Trace generation
sidebar_position: 4
---

# Trace generation

`trace_generator.generate_trace(...)` is the bridge between the
**profiled latency database** (CSV files produced by the profiler)
and the **per-batch execution trace** that ASTRA-Sim consumes.

It's the page where "the model has 32 decoder blocks, each block has
qkv + attention + o_proj + mlp" turns into "this batch takes
1.78 ms".

> Looking for the trace file format spec? See
> **[Reference → Trace file format](/docs/reference/trace-format)**.
> Looking for how the profiler *produces* the latency database in the
> first place? See **[Profiler → Output bundle](/docs/profiler/output-bundle)**.
> This page is about how the simulator *consumes* it.

```mermaid
flowchart LR
    subgraph PROF["Profile DB (loaded once)"]
        DENSE[dense.csv]
        PERSEQ[per_sequence.csv]
        ATTN[attention.csv]
        MOE[moe.csv]
        SKEW[skew_fit.csv]
    end
    BATCH[Batch] --> WALK[Walk architecture YAML]
    WALK --> DISPATCH{Layer category?}
    DISPATCH -->|dense| DENSE
    DISPATCH -->|per_sequence| PERSEQ
    DISPATCH -->|attention| ATTN
    DISPATCH -->|moe| MOE
    ATTN --> SKEWBLEND["Blend t_mean and t_max<br/>with alpha"]
    SKEW --> SKEWBLEND
    DENSE --> EMIT[Emit trace row]
    PERSEQ --> EMIT
    SKEWBLEND --> EMIT
    MOE --> EMIT
    EMIT --> ROWS["TraceData<br/>(field tuple per layer)"]
    ROWS --> GRAPH["Chakra converter<br/>(in-process)"]
    ROWS -.->|--save-trace-text| TRACEFILE["trace .txt<br/>(for inspection)"]
```

## The data the simulator consumes

The profiler writes per-category CSVs at:

```
profiler/perf/<hardware>/<model>/<variant>/tp<N>/{
  dense.csv,
  per_sequence.csv,
  attention.csv,
  moe.csv,           # MoE models only
  skew.csv,          # if heterogeneous-decode sweep is on
  skew_fit.csv       # ditto, the fitted alpha table
}
meta.yaml
```

Where `<variant>` encodes the dtype combination, e.g., `bf16` or
`bf16-kvfp8` or `fp8-kvfp8`. The simulator resolves it with
`resolve_variant(model_config)` — a pure function of the model config,
taking no dtype argument.

The CSVs hold `time_us` (microseconds). The simulator multiplies by
1000 and rounds to ns at load time, every internal latency is in ns.

## Loading the perf DB

`_load_perf_db(hardware, model, variant)` is called once per
unique `(hardware, model, variant)` triple over the simulator's
lifetime; results are cached in `_perf_db_cache`. Calling it on every
batch would be way too slow.

On first load, the simulator also:

1. Reads `meta.yaml` and compares the runtime's
   `--max-num-batched-tokens` and `--max-num-seqs` against the
   profiled sweep bounds. If you exceed them, you get a one-shot
   warning that lookups will **extrapolate** rather than clamp.
2. Hydrates the skew_fit table (`alpha_by_bucket` map) from
   `skew_fit.csv`.

## Per-category lookup

Each layer in the model's architecture YAML is tagged with a
**category**: dense, per_sequence, attention, or moe. Each category
has its own lookup function:

| Category | Lookup function | Key | Interpolation |
| --- | --- | --- | --- |
| `dense` | `_lookup_dense` | `total_len` (sum of tokens in batch) | 1D linear |
| `per_sequence` | `_lookup_per_sequence` | `num_requests` | 1D linear |
| `attention` | `_lookup_attention` | `(prefill_chunk, kv_prefill, n_decode, kv_decode)` | 4D linear (bracket + blend on each axis) |
| `moe` | `_lookup_moe` | `(local_tokens, activated_experts)` (per rank, profiled at TP=1) | 2D linear |

Every axis is bracketed by its two neighbouring profiled values and
blended on a linear scale.

All lookups **extrapolate** outside the profiled grid (via linear
extension), so a runtime value larger than the largest profiled
sample doesn't fail, it produces a (less reliable) extrapolated
latency. The startup warning above tells you when this is happening.

The `time_us` value at each grid point is converted to ns at load
time, so lookups directly yield ns.

## Variant resolution

`resolve_variant(model_config)` mirrors the profiler's
`effective_variant`, but reads only the checkpoint — there is no dtype
flag on the simulator to read instead:

```
dtype           config_weight_dtype(config)
                  quantization_config.quant_method, else torch_dtype / dtype

kv_cache_dtype  config_kv_cache_dtype(config)
                  'fp8' if quantization_config declares kv_cache_scheme
                  or kv_cache_quant_algo, else 'auto'

variant         f"{short(dtype)}"                            # kv_cache_dtype == 'auto'
                f"{short(dtype)}-kv{short(kv_cache_dtype)}"  # otherwise
```

So one model config names exactly one folder:

- Llama-3.1-8B (`torch_dtype: bfloat16`) → `bf16`
- DeepSeek-V3.2-Exp (`quant_method: fp8`) → `fp8`
- a checkpoint declaring `kv_cache_scheme` → `bf16-kvfp8`

The profiler can still *write* other folders for the same model — its
`--variant`, `--dtype` and `--kv-cache-dtype` are how a deliberate
second precision gets measured and kept beside the first. The simulator
simply never asks for one.

If the resolved folder doesn't exist under `profiler/perf/...`, the
simulator raises a clear `FileNotFoundError` pointing at the missing
variant. Profile that model with the profiler's defaults, which name the
same folder.

## Heterogeneous-decode skew correction

FlashAttention's varlen kernel pays tile-padding and SM-imbalance
costs when a decode batch has non-uniform KV lengths. The plain
attention grid can't see that, it's profiled with uniform
`kv_decode` per shot. So the profiler runs a **second sweep** on
bimodal batches (`skew.csv`) and fits a per-bucket
**alpha** that says how far along the mean→max line a skewed batch
lands:

```
alpha = (t_skew - t_mean) / (t_max - t_mean)
```

At runtime, `_lookup_attention_with_skew` looks the batch up at its
`kv_decode_mean` and blends toward a second lookup at `kv_decode_max`:

```
t_attention = t_mean + alpha * (t_max - t_mean)
```

The second lookup only happens when a non-zero alpha applies. A batch
with one decode, or with every decode at the same length, or resolving
to `alpha = 0`, returns `t_mean` directly.

The bucket key is built from five axes:
`pc | n_label | skew_rate_label | kv_big_label | kp_label`

- `pc`: prefill chunk size (bucket per profiled value).
- `n_label`: `n_decode` value (bucket per profiled value).
- `skew_rate_label`: normalized skew rate, fixed [0,1] scheme.
- `kv_big_label`: log-4× bins of the long KV.
- `kp_label`: `kv_prefill` value (bucket per profiled value).

The bucket axis definitions live in
`meta.yaml::skew_fit.bucket_axes`, so widening the profile sweep
lights up finer resolution without any simulator code change.

If the skew sweep wasn't run (`SKIP_SKEW=1` at profile time), the
simulator applies **no** correction (`alpha = 0`, i.e. `t_mean`).
Profile skew if you need it. The profile angle
of skew correction is documented on
**[Profiler → Skew & alpha fit](/docs/profiler/skew-alpha-fit)**.

## Walking the architecture YAML

Each model has an architecture YAML at
`profiler/models/<model_type>.yaml` — or at a YAML that lists its
`model_type` under `model_types:`, since one file serves a whole family
(e.g., `llama.yaml`, `qwen3.yaml` for both `qwen3` and `qwen3_moe`). The YAML
has:

- A `catalog:` mapping canonical layer names (e.g., `qkv_proj`,
  `attention`, `moe`) to vLLM class names.
- A `blocks:` describing what one decoder layer emits, keyed by axis, plus a
  `shared:` for what runs once per iteration:
  `shared.prologue → (attn.<type>.pre_attn → attn.<type>.post_attn →
  mlp.<dense|moe>) x num_hidden_layers → shared.head`.

Which block a given layer runs comes from the **checkpoint's own config**, not
from the YAML: `layer_types` decides the attention, `first_k_dense_replace` /
`decoder_sparse_step` / `moe_layer_freq` the MLP, `sparse_attention_freq` /
`index_topk_pattern` whether a sparse-selection branch applies.
`profiler/core/stack.py` owns those rules and both the profiler and the
simulator read it, so the two cannot disagree about a hybrid stack.

Blocks are built once per distinct block *shape* and replayed for every layer
that shares it, so trace generation stays O(1) in depth for a uniform model
while a heterogeneous one still gets the right block per layer.

`trace_generator._emit_sequence` walks a block's layer list and emits
one trace row per layer. It also:

- Attaches **TP-ALLREDUCE** after `o_proj` and `down_proj` when
  `tp_size > 1`.
- Wraps the MoE block with **EP-ALLTOALL** markers when MoE is
  active.
- Swaps in PIM attention before the NPU attention kernel when
  `--enable-attn-offloading` is on.
- One-shot-warns when a sequence layer is missing from the profile
  CSVs (so you know to extend the profile).

## Where DP groups change things

When instances are in a `dp_group`, trace generation is **deferred**
until all DP members have scheduled their batches for the current
iteration. The simulator collects each member's `total_len`, takes
the **max** across the group, and uses that for the EP-ALLTOALL
`comm_size`:

```
comm_size_alltoall = max(total_len_per_member) * hidden_size * fp_size
```

Each member's trace still uses its own per-instance `total_len` for
the dense and attention kernels, only the ALLTOALL is synchronized.
This matches what production MoE serving does (vLLM CUDA-graph
padding to the max in the wave).

The full DP+EP wave-sync mechanics live on
**[Parallelism mechanics](./parallelism-mechanics)**.

## Block copy optimization

Layers that resolve to the same block shape produce identical trace rows — the
rows carry the canonical layer name, and the writer numbers the lines — so
building each one separately is wasted work. By default
`enable_block_copy=True`:

- **Build** a block's rows once per distinct block shape.
- Append that same list once per layer sharing the shape.

The emitted trace is unchanged: it still has every layer's rows. This is a
trace-*generation* optimization, saving the per-layer latency lookups and size
computations, and there is no `block_copy` instruction in the trace or in
Chakra. A 48-layer Qwen3-30B-A3B run emits 583 trace lines either way.

The reuse key is the layer's resolved block shape, so a heterogeneous stack
gets one built block per shape rather than one for the whole model — Qwen3.5's
gated-DeltaNet and full-attention layers are never shared.

Exact for dense models and for MoE with `--expert-routing-policy BALANCED` (the
default), which is deterministic, so every layer produces the same
`(local_tokens, activated_experts)` pair. For `RR` / `RAND`, per-layer variance
is small once the batch saturates, so block copy remains a harmless
approximation; `CUSTOM` policies that need per-layer variance can disable it
via `block_copy=False` in the gate router constructor.

## Per-rank latency for MoE

MoE uses `EXPERT {i}` / `EXPERT END` markers in the trace, with one
`COMP_NODE` per EP rank. Each rank's latency comes from the MoE CSV
keyed on its **local** token count and activated experts (profiled
at TP=1). Ranks execute in parallel and synchronize at the
ALLTOALL barrier.

Expert-to-rank assignment uses even partitioning:
`expert_id * ep_size // num_experts`.

## Gotchas

1. **`time_us` in CSV is microseconds.** The simulator converts to
   ns at load time. If you're cross-referencing a CSV row against
   a simulator log line, multiply by 1000.
2. **No calibration scaling.** Profiled latencies are used directly,
   not rescaled. If your profiles look off, re-profile rather than
   tweaking a "scale factor", there isn't one.
3. **First-load is slow** (perf DB parsing); subsequent loads hit
   `_perf_db_cache`. Restarting the simulator pays the parse cost
   again.
4. **Variant folder must exist.** A model config whose bundle was never
   profiled → `FileNotFoundError`. The profiler's defaults name the same
   folder the simulator asks for, so profiling the model is the fix.
5. **Skew correction only fires when the skew sweep was profiled.**
   Otherwise you get a single pooled alpha, which is correct on
   average but loses heterogeneity sensitivity.

## What's next

- **[Parallelism mechanics](./parallelism-mechanics)**: what
  TP-ALLREDUCE / EP-ALLTOALL actually look like in the trace.
- **[Reference → Trace file format](/docs/reference/trace-format)**
  the field-by-field spec of the text trace this page produces.
