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
| `WARNING` (default) | The heartbeat block every `--log-interval` seconds, plus warnings (variant fallback, runtime exceeding the profiler sweep, MoE config mismatch, and so on) |
| `INFO` | Adds per-iteration scheduler detail and request-lifecycle events (resume notices, per-node power logs) |
| `DEBUG` | Adds per-layer memory load / store activity and full `Batch` / `Request` dumps. Generates a lot of output; pipe to a file |

Independently of the level, a run always prints a startup banner, a
KV-cache sizing block, the periodic heartbeat, and the final results.
`bench/examples/<model>/outputs/sim.log` holds complete real examples of
all of it; every sample on this page is copied from there.

### Startup banner

```text
──────────────────────────── LLMServingSim2.0 ────────────────────────────
                              Input configuration

  • Cluster config             : bench/examples/configs/Llama-3.1-8B.json
  • Run ID                     : run_1787203264816023_112338
  • ASTRA-Sim inputs root      : /app/LLMServingSim/astra-sim/inputs/runs/run_1787203264816023_112338
  • Dataset                    : workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl
  • Max num seqs               : 128
  • Max batched tokens         : 2048
  • Block size (tokens)        : 16
  • Request routing            : LOAD
  • Expert routing             : BALANCED
  • Prefix caching             : ENABLED
  • Chunked prefill            : ENABLED
  • Prefix caching scheme      : xPU-Only
  • Centralized prefix caching : DISABLED
  • Offload attention to PIM   : DISABLED
  • Sub-batch interleaving     : DISABLED
  • Network backend            : analytical
  • Log interval (s)           : 1.0
  • Log level                  : WARNING
──────────────────────────────────────────────────────────────────────────
                          KV Cache Initialization

  • Instance [0] : 585248 tokens / 36578 blocks (71.44 GiB/rank at util 0.90)
```

The **Run ID** and inputs root are what you need to find a run's
intermediate ASTRA-Sim files, and they only survive if you passed
`--no-cleanup-inputs`.

The **KV Cache Initialization** line is the one to read first when
comparing against real vLLM: it reports the capacity the simulator
derived from `npu_mem.mem_size * mem_util - weight`. vLLM also
subtracts its activation peak and CUDA context, which the simulator
does not model, so this is an upper bound on vLLM's capacity at the
same utilization. `bench run`'s `meta.json::kv_cache.num_gpu_blocks`
is the number to compare it against.

Note the banner reports the **CLI** values for `Max num seqs` and
`Max batched tokens`. Per-instance overrides are not echoed here, so on
a heterogeneous config this block does not tell you what each instance
actually got. See
**[Cluster config → Runtime overrides](/docs/reference/cluster-config#runtime-overrides-optional)**.

## Heartbeat block

Every `--log-interval` simulated seconds the simulator prints a
throughput line followed by an indented tree, one branch per instance
and then one per node:

```text
[1.0s] Avg prompt throughput: 9069.0 tokens/s, Avg generation throughput: 224.0 tokens/s
        ├─Running Instance[0]: 9 reqs, Waiting: 0 reqs, Total # 1 NPUs, Each NPU Memory Usage 16486.51 MB (16.771 % Used), Prefix Cache Hit ratio 0.00 %, (0 / 9069)
        └─Node[0]: Total CPU Memory Usage 0.00 MB, 0.000 % Used
```

The leading `[1.0s]` is the **simulated** clock, not wall-clock.

| Field | Meaning |
| --- | --- |
| `Avg prompt throughput` | Input tokens/s over the interval, **including** prefix-cache hits |
| `Avg generation throughput` | Generated tokens/s over the interval |
| `Running Instance[i]` | `len(scheduler.running)` — the persistent running set, not the size of this step's batch. This is the analogue of vLLM's `num_running_reqs`, which is what `bench validate` compares against |
| `Waiting` | Waiting requests that have already **arrived** (future arrivals are excluded) |
| `Total # N NPUs` | The instance's `num_npus` |
| `Each NPU Memory Usage` | Weights plus KV per rank, and that as a percentage of `npu_mem.mem_size` — so the ceiling is `mem_util * 100`, not 100 |
| `Prefix Cache Hit ratio` | Cumulative since the run started, not per interval, with `(hit tokens / requested tokens)` after it. Present only when the instance has prefix caching on |
| `Node[i]` | Host memory used by that node's lower KV tier |

### Multiple instances

One `├─Running Instance[i]` branch per instance, and the `Node` line
gains a per-instance split of the host total:

```text
[4.0s] Avg prompt throughput: 9063.0 tokens/s, Avg generation throughput: 1074.0 tokens/s
        ├─Running Instance[0]: 23 reqs, Waiting: 0 reqs, Total # 1 NPUs, Each NPU Memory Usage 32634.88 MB (33.198 % Used), Prefix Cache Hit ratio 0.00 %, (0 / 20285)
        ├─Running Instance[1]: 22 reqs, Waiting: 0 reqs, Total # 1 NPUs, Each NPU Memory Usage 32981.38 MB (33.550 % Used), Prefix Cache Hit ratio 0.00 %, (0 / 24147)
        └─Node[0]: Total CPU Memory Usage 0.00 MB, 0.000 % Used (Instance[0]: 0.00 %, Instance[1]: 0.00 %)
```

The trailing per-instance percentages are each instance's share of the
node's CPU usage, not of node capacity — they sum to 100% when anything
is resident. They are omitted when the node has a single instance, and
also when `--enable-prefix-sharing --prefix-storage CPU` makes the pool
node-wide rather than per-instance; in that case the `Node` line
carries a shared `Prefix Cache Hit ratio` instead.

Nothing in this block identifies which instance is prefill and which is
decode. Read the instance order from your cluster config.

### With CXL as the lower tier

`--prefix-storage CXL` adds a `CXL` branch after the node lines. With
`--enable-prefix-sharing` it is one branch per device:

```text
        ├─CXL[0]: Total CXL Device Memory Usage 3276.80MB, 3.200 % Used
```

and without sharing, one per prefix-caching instance:

```text
        ├─CXL[0]/Instance[0]: Total CXL Device Memory Usage 3276.80 MB, 3.200 % Used
```

### With the power model

A `power:` block on the node appends a final branch:

```text
        └─Avg power consumption: 712.4 W
```

The label is right and misleading at once. It *is* an average — over the
log interval, computed as energy accrued since the previous heartbeat
divided by elapsed simulated time — but it is also a single
**cluster-wide** figure summed across every node, so on a multi-node run
this one line cannot be attributed per node. The per-node split appears
only in the final summary.

## Final results

On shutdown the simulator prints several ruled sections. Trimmed from a
real 300-request run:

```text
▶ Simulation results...

Total simulation time: 0h 2m 40.611s
─────────────────────────── Throughput Results ───────────────────────────
Total requests:                                                     300
Total clocks (ns):                                                  65049871040
Total latency (s):                                                  65.050
Total input tokens:                                                 257239
Total generated tokens:                                             195753
Request throughput (req/s):                                         4.61
Average prompt throughput (tok/s):                                  3954.49
Average generation throughput (tok/s):                              3009.28
Total token throughput (tok/s):                                     6963.76
Throughput per 1.0 sec ([prompt_throughput], [gen_throughput]): [(9069.0, 224.0), (13535.0, 493.0), ...]
──────────────────────── Prefix Caching Results ──────────────────────────
Total requested prompt tokens:                                      257239
NPU prefix hit prompt tokens:                                       19520
NPU prefix hit ratio (%):                                           7.59
Total prefix hit ratio (%):                                         7.59
───────────────────────────── Instance [0] ───────────────────────────────
─────────────────────────── Time to First Token ──────────────────────────
Mean TTFT (ms):                                                     6915.26
Median TTFT (ms):                                                   8783.06
P99 TTFT (ms):                                                      19627.55
───────────────── Time per Output Token (excl. 1st token) ────────────────
Mean TPOT (ms):                                                     32.33
Median TPOT (ms):                                                   33.59
P99 TPOT (ms):                                                      37.80
─────────────────────────── Inter-token Latency ──────────────────────────
Mean ITL (ms) :                                                     32.24
Median ITL (ms) :                                                   27.68
P99 ITL (ms) :                                                      111.39
──────────────────────────────────────────────────────────────────────────
```

Notes on individual fields:

- **`Total simulation time`** is wall-clock: how long the simulation
  took to run, not the workload's simulated duration. That is
  `Total latency (s)`.
- **`Total clocks (ns)`** is the simulated makespan in ASTRA-Sim
  cycles. Because the simulator is deterministic, this is an exact
  regression signal — the same config and workload reproduce it
  bit-for-bit.
- **`Total input tokens`** is summed from each request's
  `original_input`, deliberately *not* derived by subtracting the
  recompute counter. A request preempted again mid-recompute is charged
  its full remaining work each time it is re-admitted, so the two are
  not complements.
- **`Preemptions`** and **`Recomputed prompt tokens (preemption)`**
  lines appear only when non-zero. Their absence means the run never
  preempted.
- **`Throughput per N sec`** is the full per-interval series, the same
  numbers as the heartbeat lines, as a list of
  `(prompt, generation)` pairs.
- The **Prefix Caching Results** block appears only when some instance
  has prefix caching on. With `--prefix-storage CPU` or `CXL` it gains
  a `<tier> prefix hit prompt tokens` / `<tier> prefix hit ratio` pair,
  and `Total prefix hit ratio` covers both tiers.
- The per-**Instance** blocks report **milliseconds**, unlike the CSV,
  which is nanoseconds throughout.

:::note TTFT is not measured the way vLLM measures it
The simulator stops the clock when the first token's *computation*
completes. vLLM stops it when the client *receives* the token, so real
vLLM TTFT is higher. `bench validate` puts both sides on matched
definitions; see **[Bench CLI](/docs/reference/bench-cli)**.
:::

### Power modeling results

With a `power:` block configured, an extra section lands between
throughput and the per-instance blocks:

```text
──────────────────────── Power Modeling Results ──────────────────────────
Total energy consumption (kJ):                                      15.95
──────────────────────────────────────────────────────────────────────────
Node 0 total energy consumption (kJ):                               15.95
├─ NPU energy consumption (J):        12453.00
├─ CPU energy consumption (J):         1233.00
├─ DRAM energy consumption (J):         442.00
├─ Link energy consumption (J):         388.00
└─ ...
──────────────────────────────────────────────────────────────────────────
Power per 1.0 sec (W): [712.4, 698.1, ...]
```

One node block per node, each listing its per-device energy, followed
by the full per-interval power series. Per-device figures include that
device's always-on base power times the run duration, so they sum to
the node total.

## Common patterns to look for

### High waiting count, low NPU memory

The heartbeat shows a large `Waiting` count while
`Each NPU Memory Usage` sits well under `mem_util * 100 %`. Likely
cause: the token budget (`--max-num-batched-tokens`) or
`--max-num-seqs` is the bottleneck, not memory. Bump whichever binds.

If `Running` is pinned at exactly `--max-num-seqs`, that is the
binding constraint. If `Running` moves but throughput does not, the
token budget is.

### Decode TPOT spikes during prefill bursts

A prefill-heavy moment lands in the same batch as ongoing decodes,
the budget gets eaten by prefill, and decode latency stretches.

Mitigations:
- `--enable-chunked-prefill` (default) splits long prefills.
- `--long-prefill-token-threshold N` caps prefill tokens per
  step.
- `--npu-memory-utilization` sets how much NPU memory weights plus
  KV may use. **Raising** it enlarges the KV cache and admits more
  concurrent requests; lowering it shrinks capacity and therefore
  raises preemptions.

### Prefix hit rate near 0%

Either the workload genuinely has no shared prefixes, or it is not
pre-tokenized. A request with no `input_tok_ids` gets an empty hash
chain, which disables prefix caching for it outright — so an
untokenized workload reports exactly 0.00% with the feature enabled.
Check the JSONL first (see
[Workloads → JSONL format](/docs/workloads/jsonl-format#why-token-ids-matter)).

Note the heartbeat's ratio is **cumulative** from the start of the run,
so it climbs slowly and is near zero for the first few intervals even
on a workload with heavy sharing.

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

LLMServingSim is validated end-to-end against real vLLM. On the three
bundled configurations, TPOT and end-to-end latency means land within
1.5%; TTFT means run -2.6% to -5.8%, which is a small absolute
difference on a metric with small absolute values. Numbers and plots are on
**[Validation](/docs/validation)**; the harness that produces them is
**[Bench CLI](/docs/reference/bench-cli)**. Complete real logs for
those runs are committed under
**[bench/examples/](https://github.com/casys-kaist/LLMServingSim/tree/main/bench/examples)**,
which is the best place to see what a healthy run looks like
end-to-end.

## What's next

- **[Reference → CLI flags](/docs/reference/cli-flags)**: every
  flag that affects the output.
- **[Examples](/docs/examples)**: worked configurations to compare
  your output against.
