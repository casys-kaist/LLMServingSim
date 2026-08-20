---
title: FP8 KV cache
sidebar_position: 3
---

# FP8 KV cache

> **What this demonstrates:** halving KV cache memory consumption by
> storing keys and values in 8-bit floats (1 byte / element) instead
> of bf16/fp16 (2 bytes). Frees NPU memory for larger batches or
> longer contexts.

`--kv-cache-dtype fp8` is the flag. It does two things:

1. **Trace generator** swaps the variant folder lookup from
   `<dtype>` (e.g., `bf16`) to `<dtype>-kvfp8` (e.g., `bf16-kvfp8`),
   so attention latency comes from the FP8-KV profile bundle.
2. **Memory model** halves the per-block KV cache byte count
   (`bytes_per_block` uses `kv_fp = 1` instead of `2`), so the
   scheduler can fit roughly 2× as many active tokens at the same
   `npu_mem`.

## Prerequisites

- Simulator container set up
- A profile bundle with the **`-kvfp8` variant** for your
  `(hardware, model)` combo. The bundled RTXPRO6000 perf data ships
  the `bf16` variant only — see the box below.

> ⚠️ **You need the FP8-KV profile bundle.** If
> `profiler/perf/<hardware>/<model>/<variant>-kvfp8/` doesn't exist,
> the simulator exits at startup with a clear `FileNotFoundError`
> pointing at the missing folder. Bundled today:
>
> | Hardware | Model | Variants shipped |
> | --- | --- | --- |
> | `RTXPRO6000` | `meta-llama/Llama-3.1-8B` | `bf16` |
> | `RTXPRO6000` | `Qwen/Qwen3-32B` | `bf16` |
> | `RTXPRO6000` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | `bf16` |
>
> To use this example today, profile the `-kvfp8` variant first
> with `KV_CACHE_DTYPE=fp8 ./profiler/profile.sh` (see
> **[Profiler → Adding hardware](/docs/profiler/adding-hardware)**)
> and rerun.

## Cluster config

Any single-instance cluster config works; FP8 KV is a runtime CLI
flag, not a config field. Example using the bundled simple config:

```json title="configs/cluster/single_node_single_instance.json"
{
  "num_nodes": 1,
  "link_bw": 16,
  "link_latency": 20000,
  "nodes": [
    {
      "num_instances": 1,
      "cpu_mem": {"mem_size": 512, "mem_bw": 256, "mem_latency": 0},
      "instances": [
        {
          "model_name": "meta-llama/Llama-3.1-8B",
          "hardware": "RTXPRO6000",
          "npu_mem": {"mem_size": 96, "mem_bw": 1597, "mem_latency": 0},
          "num_npus": 1,
          "tp_size": 1,
          "pd_type": null
        }
      ]
    }
  ]
}
```

## Run

```bash
python -m serving \
  --cluster-config 'configs/cluster/single_node_single_instance.json' \
  --dtype bfloat16 --kv-cache-dtype fp8 --block-size 16 \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/fp8_kv_run.csv' \
  --log-interval 1.0
```

The two dtype flags compose:

- `--dtype bfloat16`: weights still in bf16 (chosen by the
  weights-side profile variant).
- `--kv-cache-dtype fp8`: KV cache in fp8. The variant resolver
  appends `-kvfp8` to the weights variant, so this run reads
  attention latency from
  `profiler/perf/RTXPRO6000/meta-llama/Llama-3.1-8B/bf16-kvfp8/`.

## Expected output

The throughput log looks unchanged in shape, but the memory
footprint at the same batch size is much smaller:

```text
[42.0s] Avg prompt throughput: 2416.0 tokens/s, Avg generation throughput: 860.0 tokens/s
        ├─Running Instance[0]: 16 reqs, Waiting: 0 reqs, Total # 1 NPUs, Each NPU Memory Usage 68204.26 MB (69.362 % Used), Prefix Cache Hit ratio 3.18 %, (4096 / 128704)
        └─Node[0]: Total CPU Memory Usage 0.00 MB, 0.000 % Used
```

For comparison, the same workload on the same machine with
`--kv-cache-dtype auto` (= bf16) at `batch=16` would either OOM or
produce a much smaller batch under memory pressure. The KV-cache
half of the per-token memory cost is gone.

## What's interesting

- **Throughput rises on KV-bound workloads.** Long-context decode
  is dominated by KV cache memory; halving it doubles the
  effective batch size at the same `npu_mem`. Decode throughput
  follows.
- **TTFT changes slightly.** Prefill attention reads the FP8 KV
  profile, which has slightly different per-token cost (the
  attention kernel does dtype-conversion on the fly). Usually a
  small win on long prefills, neutral on short ones.
- **No accuracy claim from the simulator.** Like every other
  knob, `--kv-cache-dtype fp8` is a *latency / memory* knob, not a
  numerical-accuracy knob. The simulator doesn't validate vs. real
  vLLM that FP8 KV produces the right outputs; that's vLLM's
  problem. The simulator just charges the right bytes and
  latencies.

## Related examples

- **[Prefix caching](./prefix-caching)**: orthogonal, often
  combined. Halving KV per token plus reusing prefix blocks
  compounds the memory savings.
- **[CXL memory](./cxl-memory)**: another way to attack memory
  pressure, by spilling to a second tier instead of compressing
  in place.

## Where to learn more

- **[Simulator → KV cache & memory](/docs/simulator/scheduling/kv-cache-and-memory)**:
  the `bytes_per_block` formula and how `kv_fp` flows into
  the scheduler's memory check.
- **[Profiler → Output bundle](/docs/profiler/output-bundle)**:
  variant naming (`bf16` vs. `bf16-kvfp8` vs. `fp8` vs.
  `fp8-kvfp8`) and how the profiler emits each.
