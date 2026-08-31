---
title: FP8 KV cache
sidebar_position: 3
---

# FP8 KV cache

> **What this demonstrates:** halving KV cache memory by storing keys
> and values in 8-bit floats (1 byte / element) instead of bf16/fp16
> (2 bytes) — and, more importantly, **where that decision comes
> from**, which is the checkpoint rather than a flag.

## There is no `--kv-cache-dtype`

The simulator takes no dtype input at all. A modern checkpoint carries
**five** cache dtypes decided in four different places, so a flag per
dtype would be both unusable and unfaithful — it would describe a model
nobody can serve. Every one is read from the model config instead; see
**[Reference → CLI flags → Precision](/docs/reference/cli-flags)** for
the full table.

For the KV cache specifically, the rule is vLLM's own
(`attention.py:281`): a checkpoint whose `quantization_config` declares
`kv_cache_scheme` (compressed-tensors) or `kv_cache_quant_algo`
(ModelOpt) gets an fp8 cache; anything else inherits the weight dtype.
vLLM's source states this as the direction it is heading in as well —
*"kv cache dtype should be specified in the FP8 checkpoint config and
become the 'auto' behavior"*.

So an FP8-KV run is an FP8-KV **checkpoint**, and switching it on means
pointing the simulator at one.

## What it changes

1. **Trace generator** resolves the variant folder to
   `<dtype>-kvfp8` (e.g. `bf16-kvfp8`) instead of `<dtype>`, so
   attention latency comes from the FP8-KV profile bundle.
   `resolve_variant(model_config)` is a pure function of the config —
   one checkpoint names exactly one folder.
2. **Memory model** charges 1 byte per KV element instead of 2, so the
   same `npu_mem` holds roughly twice as many tokens.

Two caches deliberately do **not** follow it:

- A **mamba** conv or recurrent state follows `mamba_cache_dtype` /
  `mamba_ssm_dtype`, not the KV dtype.
- A **sparse-attention indexer's** side cache is fixed by the model —
  DeepSeek/GLM store fp8 keys plus fp32 scales as `uint8`, MiniMax-M3
  stores bf16. Shrinking those with the KV cache would understate
  M3's cache by 10%.

## Prerequisites

- Simulator container set up
- A checkpoint config under `configs/model/` that declares a quantized
  KV cache
- A profile bundle with the matching **`-kvfp8` variant** for your
  `(hardware, model)` pair

> ⚠️ **Nothing bundled today declares one.** Every config under
> `configs/model/` leaves `quantization_config.kv_cache_scheme` unset,
> and every bundle under `profiler/perf/` ships the `bf16` variant
> only:
>
> | Hardware | Model | Variants shipped |
> | --- | --- | --- |
> | `RTX4090` | `meta-llama/Llama-3.1-8B` | `bf16` |
> | `RTXPRO6000` | `meta-llama/Llama-3.1-8B` | `bf16` |
> | `RTXPRO6000` | `Qwen/Qwen3-32B` | `bf16` |
> | `RTXPRO6000` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | `bf16` |
>
> This example therefore describes the mechanism rather than a run you
> can reproduce from the repo as shipped.

## Doing it

**1. Use a checkpoint that declares an fp8 KV cache.** Copy the
model's real `config.json` into `configs/model/<org>/<name>.json`. The
fields that matter:

```json title="configs/model/<org>/<name>.json (excerpt)"
{
  "torch_dtype": "bfloat16",
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "kv_cache_scheme": {"type": "float", "num_bits": 8}
  }
}
```

`config_weight_dtype` reads `quant_method` first (on a quantized
checkpoint the dtype fields describe the *activations*, not the
weights), and `config_kv_cache_dtype` reads `kv_cache_scheme`. Check
what the pair resolves to before profiling:

```python
from serving.core.trace_generator import resolve_variant
from serving.core.utils import get_config
resolve_variant(get_config("<org>/<name>"))    # -> e.g. 'bf16-kvfp8'
```

**2. Profile that variant.** The profiler still takes the flags — it
is what *writes* the bundles, and measuring a second precision beside
the first is exactly what they are for:

```bash
KV_CACHE_DTYPE=fp8 ./profiler/profile.sh
```

See **[Profiler → Adding hardware](/docs/profiler/adding-hardware)**.

**3. Run.** No new flag — the checkpoint already said what it is:

```bash
python -m serving \
  --cluster-config 'configs/cluster/single_node_single_instance.json' \
  --block-size 16 \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/fp8_kv_run.csv' \
  --log-interval 1.0
```

If the bundle is missing the simulator stops at startup with a
`FileNotFoundError` naming the folder it wanted and the variants that
do exist.

## What's interesting

- **Throughput rises on KV-bound workloads.** Long-context decode is
  dominated by KV cache memory; halving it roughly doubles the batch
  that fits at the same `npu_mem`, and decode throughput follows.
- **TTFT changes slightly.** Prefill attention reads the FP8-KV
  profile, whose per-token cost differs (the kernel converts dtype on
  the fly) — usually a small win on long prefills, neutral on short
  ones.
- **No accuracy claim.** This is a latency and memory model, not a
  numerics model. The simulator charges the right bytes and the
  measured latencies; whether fp8 KV produces acceptable outputs is
  vLLM's question, not this one.

## Related examples

- **[Prefix caching](./prefix-caching)**: orthogonal and often
  combined — fewer bytes per token plus fewer tokens recomputed.
- **[CXL memory](./cxl-memory)**: the other way at memory pressure,
  spilling to a second tier instead of compressing in place.

## Where to learn more

- **[Reference → CLI flags](/docs/reference/cli-flags)**: the whole
  five-dtype table and where each is decided.
- **[Simulator → KV cache & memory](/docs/simulator/scheduling/kv-cache-and-memory)**:
  the per-block byte formula and how the KV dtype reaches the
  scheduler's memory check.
- **[Profiler → Output bundle](/docs/profiler/output-bundle)**:
  variant naming (`bf16`, `bf16-kvfp8`, `fp8`, `fp8-kvfp8`) and how
  the profiler emits each.
