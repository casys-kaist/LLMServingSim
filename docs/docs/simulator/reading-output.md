---
title: Reading the output
sidebar_position: 8
---

# Reading the output

The simulator produces three kinds of output:

1. **Per-request CSV** at the path passed via `--output`.
2. **Throughput log line** printed every `--log-interval` seconds.
3. **Final power summary** (only if the cluster config has a
   `power:` block).

This page covers what each one means and how to read them.

## Per-request CSV

When you pass `--output outputs/foo.csv`, the simulator writes one
row per finished request:

```csv
instance id,request id,model,input,output,arrival,end_time,latency,queuing_delay,TTFT,TPOT,ITL
0,0,Qwen/Qwen3-30B-A3B-Instruct-2507,1472,133,4059740,1082836204,1078776464,0,51162321,7784955,"[7780422, 7779379, 7779523, ...]"
0,3,meta-llama/Llama-3.1-8B,4,16,570907776,711600111,140692335,3739551,15137413,11414083,"[11043655, 11381158, ...]"
...
```

The bundled `outputs/example_*_run.csv` files (one per scenario in
`serving/run.sh`) are good examples to skim.

### Column reference

| Column | Type | Meaning |
| --- | --- | --- |
| `instance id` | int | Which serving instance ran this request |
| `request id` | int | Monotonic id assigned by the router |
| `model` | string | Model name (e.g., `meta-llama/Llama-3.1-8B`) |
| `input` | int | Prompt tokens (full input length, including any prefix-cache hits) |
| `output` | int | Decode tokens generated (i.e., total length minus `input`) |
| `arrival` | int (ns) | When the request arrived (simulator clock) |
| `end_time` | int (ns) | When the last generated token completed |
| `latency` | int (ns) | End-to-end latency: `end_time - arrival` |
| `queuing_delay` | int (ns) | From arrival to first scheduling step |
| `TTFT` | int (ns) | Time-to-first-token: first-token-completion minus `arrival` |
| `TPOT` | int (ns) | Mean time-per-output-token: `(latency - TTFT) // (output - 1)` (or `0` when `output == 1`) |
| `ITL` | string | Inter-token latencies, ns. Serialized Python list, e.g. `"[7780422, 7779379, ...]"` |

All times are in **nanoseconds**. Divide by `1e9` for seconds, `1e6`
for milliseconds. Column names use spaces, not underscores; quote
them in pandas (`df["instance id"]`).

> **Note:** `Request` objects internally also carry `session_id` /
> `sub_request_index` (for agentic workloads) and per-tier prefix-
> cache hit counters (`prefix_cache_hit`, `npu_cache_hit`,
> `storage_cache_hit`). These are tracked in memory and surfaced in
> the throughput log line, but are **not** written to the per-request
> CSV today. Use the throughput log (with `--log-interval`) to see
> aggregate prefix-hit rates; for per-request agentic accounting,
> read the `Request` objects directly or extend `Scheduler.save_output`.

### Common derived metrics

```python
import pandas as pd
df = pd.read_csv("outputs/foo.csv")

# Wall-clock TTFT in milliseconds
df["TTFT_ms"] = df["TTFT"] / 1e6

# TPOT in milliseconds (already a per-token mean; divide for ms)
df["TPOT_ms"] = df["TPOT"] / 1e6

# End-to-end latency in seconds
df["latency_s"] = df["latency"] / 1e9

# Throughput across the whole run (tokens / second)
total_tokens = (df["input"] + df["output"]).sum()
sim_duration_s = (df["end_time"].max() - df["arrival"].min()) / 1e9
throughput = total_tokens / sim_duration_s

# Per-instance distribution
per_inst = df.groupby("instance id").agg(
    requests=("request id", "count"),
    p50_TTFT_ms=("TTFT", lambda x: x.quantile(0.5) / 1e6),
    p99_TTFT_ms=("TTFT", lambda x: x.quantile(0.99) / 1e6),
)

# Inter-token latency: parse the ITL string back into a list per row
import ast
df["ITL_list"] = df["ITL"].apply(ast.literal_eval)
df["ITL_p50_ms"] = df["ITL_list"].apply(lambda xs: pd.Series(xs).quantile(0.5) / 1e6)
```

## Standard output (log levels)

The simulator's `--log-level` flag controls how much detail lands on
stdout while a run is in progress:

| Level | What you see |
| --- | --- |
| `WARNING` (default) | The throughput log line every `--log-interval` seconds, plus warnings (variant fallback, runtime exceeds profiler sweep, MoE config mismatch, etc.) |
| `INFO` | Adds per-iteration scheduler decisions (which requests entered the batch, prefix-cache hits per request) and the request lifecycle (arrival / first token / completion). Useful for debugging routing and scheduling. |
| `DEBUG` | Adds per-layer memory load / store activity, full `Batch` / `Request` dumps, and prefix-cache hit-ratio snapshots. Generates a lot of output; pipe to a file. |

Independently of the level, the simulator always emits:

- A startup banner with the resolved `(hardware, model, variant)`
  and the engine_effective comparison vs. `meta.yaml`.
- The final summary on shutdown (Total requests, mean TTFT / TPOT,
  throughput, plus the **power summary** below if `power:` is
  configured).

The throughput log line itself is identical regardless of level,
the only difference is what surrounds it.

## Throughput log line

Every `--log-interval` seconds the simulator prints a one-line
status update. The format adapts to which features are enabled:

### Single-instance baseline

```text
[INFO] step=42 batch=8 prompt_t=1.2k tok/s decode_t=420 tok/s npu_mem=88.4 GB
```

| Field | Meaning |
| --- | --- |
| `step` | Iteration number this interval ended on |
| `batch` | Batch size in requests |
| `prompt_t` | Prompt-side throughput (input tokens/sec, includes prefix hits) |
| `decode_t` | Decode-side throughput (generated tokens/sec) |
| `npu_mem` | NPU memory footprint at this moment |

### Multi-instance

```text
[INFO] step=21 inst0_batch=6 inst1_batch=4 prompt_t=2.5k tok/s decode_t=860 tok/s
       npu_mem=[63.2 GB, 63.2 GB]
```

`inst0_batch` / `inst1_batch` are per-instance batch sizes; `npu_mem`
is per-instance.

### Prefill / decode split

```text
[INFO] step=15 P=8 D=12 prompt_t=3.1k tok/s decode_t=620 tok/s
       npu_mem=[55.4 GB, 71.2 GB]
```

`P=` and `D=` are batch sizes on the prefill and decode instances.

### With prefix sharing

```text
[INFO] step=20 inst0_batch=6 inst1_batch=4 prompt_t=2.4k tok/s decode_t=820 tok/s
       prefix_hit=78% (npu=42%, cpu=36%)
```

The `prefix_hit` field shows the cache hit rate across the interval,
broken down by tier.

### With DP+EP MoE

```text
[INFO] step=8 batch=4+4 prompt_t=1.4k tok/s decode_t=520 tok/s
       npu_mem=[81.2 GB, 81.2 GB] alltoall=512 KB
```

`batch=4+4` shows per-DP-member batches. `alltoall` is the
wave-synchronized ALLTOALL message size.

### With PIM offload

```text
[INFO] step=10 batch=8 prompt_t=1.1k tok/s decode_t=520 tok/s
       npu_mem=63.4 GB pim_busy=72%
```

`pim_busy` is the fraction of the interval the PIM device was active.
At ~100% PIM is your bottleneck.

### With CXL memory

```text
[INFO] step=10 batch=4 prompt_t=620 tok/s decode_t=180 tok/s
       npu_mem=12.4 GB cxl_mem=[3.2 GB, 3.1 GB, 3.1 GB, 3.2 GB]
```

`cxl_mem` is per-device usage; `npu_mem` drops because weights are
on CXL.

### With power model

```text
[INFO] step=42 batch=8 prompt_t=1.2k tok/s decode_t=420 tok/s
       npu_mem=88.4 GB power=712 W
```

`power` is the **current** total system power.

## Final power summary

When `--output` is set and the cluster config has a `power:` block,
the simulator emits a per-node energy breakdown at the end:

```text
─────── Power summary (node 0) ───────
   NPU active     :   12,453 J  (78%)
   NPU standby    :    1,012 J   (6%)
   NPU idle       :       89 J   (1%)
   CPU            :    1,233 J   (8%)
   DRAM           :      442 J   (3%)
   Link           :      388 J   (2%)
   Base + NIC + storage : 332 J  (2%)
   ─────────────────────────────────
   Total energy   :   15,949 J
```

For multi-node runs you get one block per node plus a cluster total.
The breakdown is what makes power numbers actionable for
energy-efficiency research, you can see which component dominates.

## Common patterns to look for

### High waiting count, low NPU memory

The throughput log shows large `batch` counts but `npu_mem` is far
below the cluster config's `npu_mem.mem_size`. Likely cause: the
token budget (`--max-num-batched-tokens`) is the bottleneck, not
memory. Bump it.

### Decode TPOT spikes during prefill bursts

A prefill-heavy moment lands in the same batch as ongoing decodes,
the budget gets eaten by prefill, and decode latency stretches.

Mitigations:
- `--enable-chunked-prefill` (default) splits long prefills.
- `--long-prefill-token-threshold N` caps prefill tokens per
  step.
- `--gpu-memory-utilization` sets how much NPU memory the KV cache
  gets; lowering it raises memory pressure and therefore preemptions.

### Prefix hit rate near 0%

Either the workload genuinely has no shared prefixes, or you forgot
to pre-tokenize. Check that `input_tok_ids` is populated in the
JSONL (see [Workloads → JSONL format](/docs/workloads/jsonl-format)).

### MoE per-rank latency varies wildly

Set `--expert-routing-policy BALANCED` (default). RR or RAND can
produce uneven loads on small batches. With BALANCED, per-rank
latency should be uniform within ~1%.

### CXL latency dominates TPOT

Weights placed on CXL pay the round-trip on every decode step. If
TPOT looks far worse than expected, check the `placement` block -
moving cold layers (embedding, lm_head) to CXL helps; moving every
decoder block hurts.

## Validation against known references

LLMServingSim is validated end-to-end against real vLLM with sub-3%
error on TTFT / TPOT / throughput on the bundled hardware × model
combos. The validation methodology and per-model results live in
**[bench/](https://github.com/casys-kaist/LLMServingSim/tree/main/bench)**
on GitHub.

## What's next

- **[Reference → CLI flags](/docs/reference/cli-flags)**: every
  flag that affects the output.
- **[Examples](/docs/examples)**: worked configurations to compare
  your output against.
