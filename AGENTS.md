# AGENTS.md

Guidelines for AI coding agents (Claude Code, Cursor, Copilot, etc.) working in this repository.

## Project Context

LLMServingSim 2.0 is a cycle-level LLM serving simulator. It combines a Python frontend
(`serving/`, run as `python -m serving`) with ASTRA-Sim (C++ analytical network simulator)
as the backend. The profiling pipeline (`profiler/`) generates per-hardware latency data
that drives the simulation, and the bench module (`bench/`) runs vLLM end-to-end to
validate the simulator against ground truth.

### Repository structure

```
LLMServingSim/
├── serving/                    # Simulator (`python -m serving`)
│   ├── __main__.py             # Simulation entry point + main loop
│   ├── core/                   # Internals
│   │   ├── scheduler.py        # vLLM-style continuous batching scheduler
│   │   ├── trace_generator.py  # Builds execution traces from profiled latencies
│   │   ├── memory_model.py     # Memory tracking, KV cache, tensor sizes
│   │   ├── graph_generator.py  # Chakra protobuf graph generation
│   │   ├── controller.py       # IPC with ASTRA-Sim subprocess
│   │   ├── router.py           # Request routing across instances
│   │   ├── gate_function.py    # MoE expert token routing
│   │   ├── config_builder.py   # Cluster config → ASTRA-Sim input files
│   │   ├── power_model.py      # Power/energy estimation
│   │   ├── pim_model.py        # PIM device model
│   │   ├── request.py          # Request/Batch data classes
│   │   ├── block_pool.py       # Per-tier KV block pool + prefix-cache index
│   │   ├── kv_cache_manager.py # Tiered KV cache manager (block hashing, allocation)
│   │   ├── logger.py           # Rich-based logger + stdio capture
│   │   └── utils.py            # Model config loading, formatting
│   └── run.sh                  # Example invocations across cluster configs
├── configs/
│   ├── cluster/                # Cluster topology configs (hardware, memory, instances)
│   ├── model/                  # Model architecture configs (subset of HF config.json)
│   └── pim/                    # PIM device configs (DRAMSim3 INI format)
├── workloads/                   # Request trace datasets (.jsonl)
│   └── generators/             # ShareGPT/etc → JSONL workload generators
├── profiler/                   # vLLM-based layerwise profiler (`python -m profiler`)
│   ├── __main__.py             # CLI dispatch (profile / slice)
│   ├── core/                   # internals
│   │   ├── runner.py           # Orchestration (spin_up → categories → spin_down)
│   │   ├── config.py           # Architecture / ProfileArgs / engine defaults
│   │   ├── engine.py           # vLLM lifecycle (tmpdir-based local config load)
│   │   ├── categories.py       # Dense / PerSequence / Attention / Expert
│   │   ├── skew.py             # Heterogeneous-decode skew sweep
│   │   ├── fit_alpha.py        # 5-axis weighted-LS alpha fit
│   │   ├── writer.py           # CSV + meta.yaml writer, TP-stable replication
│   │   ├── logger.py           # Rich-based logger + stdio capture
│   │   └── hooks/              # vLLM-internal-API touchpoints (worker ext, MoE patch, etc.)
│   ├── models/                 # Architecture yamls, one per HF `model_type`
│   ├── power/                  # nvidia-smi / IPMI power-logging helpers
│   ├── perf/                   # Output: perf/<hw>/<model>/<variant>/tp<N>/{dense,per_sequence,attention,moe,skew,skew_fit}.csv
│   ├── v0/                     # Legacy (pre-rewrite) profiler, kept for reference
│   ├── profile.sh              # Editable user template (MODEL / HARDWARE / TP_DEGREES / …)
│   └── profile-all.sh          # Helper: sweeps several MODELs × TP degrees
├── bench/                      # vLLM end-to-end benchmark + sim validation (`python -m bench`)
│   ├── __main__.py             # CLI dispatch (run / validate)
│   ├── core/                   # internals
│   │   ├── runner.py           # AsyncLLM driver, captures RequestStateStats
│   │   ├── recorder.py         # writes meta.json / requests.jsonl / timeseries.csv
│   │   ├── stat_logger.py      # custom vLLM StatLoggerBase that fills timeseries
│   │   ├── validate.py         # bench-vs-sim comparison entry point
│   │   ├── plots.py            # throughput / running-waiting / latency-CDF plot helpers
│   │   └── logger.py           # Rich-based logger + stdio capture
│   ├── results/                # output: bench/results/<run_id>/
│   ├── bench.sh                # host-side wrapper for `python -m bench run`
│   └── validate.sh             # host-side wrapper for `python -m bench validate`
├── scripts/                    # Shared shell entry points (env / build, not module-specific)
│   ├── docker-vllm.sh          # vLLM container (profiler + bench)
│   ├── docker-sim.sh           # simulator container
│   ├── install-vllm.sh         # bare-metal vLLM install (uv venv)
│   └── compile.sh              # ASTRA-Sim + Chakra build
└── astra-sim/                  # ASTRA-Sim C++ backend (submodule)
    ├── inputs/                 # Generated configs (network, memory, system)
    └── extern/graph_frontend/chakra/  # Chakra trace converter
```

Per-paper artifact evaluation scripts (the previous `evaluation/`
directory) live on dedicated branches (`ispass26-artifact`, etc.) and
are not part of the main branch's tree.

### Simulation flow

1. `serving/__main__.py` parses CLI args and cluster config
2. `config_builder.py` generates ASTRA-Sim input files (network.yml, system.json, memory_expansion.json)
3. ASTRA-Sim subprocess is launched
4. Per iteration:
   - `scheduler.py` forms a batch under memory and token budget constraints
   - `trace_generator.py` looks up profiled latencies and emits a text trace
   - `graph_generator.py` converts the trace to a Chakra protobuf graph
   - `controller.py` feeds the graph path to ASTRA-Sim, reads back cycle count
   - `scheduler.py` updates request state, marks completions
5. Results are printed and optionally saved to CSV

### Key data flow

```
profile.csv (profiled latencies)
    ↓ _load_perf_db() + _lookup_latency_ns()
trace_generator.py → text trace file
    ↓ Chakra converter
graph_generator.py → .et protobuf file
    ↓ stdin/stdout IPC
ASTRA-Sim (C++) → cycle count
    ↓
scheduler.py → next iteration
```

## Code Style & Formatting

- **Python**: 4-space indentation, snake_case for functions/variables, PascalCase for classes
- **No enforced formatter** — match surrounding code style in the file you're editing
- **CLI flags**: use hyphens (`--cluster-config`, `--max-num-seqs`)
- **Internal Python**: use underscores (`max_num_seqs`, `enable_chunked_prefill`)
- **JSON config filenames**: descriptive snake_case (`single_node_pim_instance.json`)
- **Imports**: keep minimal and consistent; `serving/` modules use relative imports
- **Comments**: use English only — no Korean or other non-English text in comments, docstrings, or log messages

## Architecture Patterns

### Profiler (`profiler/`)
The profiler uses vLLM's built-in `layerwise_profile()` via a worker extension class to
capture per-layer CUDA kernel timings from real vLLM execution paths. Architecture is
dispatched by the HF config's `model_type` field against YAML catalogs under
`profiler/models/<model_type>.yaml`, which bind canonical layer names (dense /
per-sequence / attention / moe) to vLLM class names.

Every TP degree is profiled on a **single GPU**: the engine is always booted with
`tensor_parallel_size=1`, and per-rank shapes are emulated by dividing `SHARD_FIELDS`
(e.g. `hidden_size`, `num_attention_heads`) by TP via `hf_overrides`. Collective
timings are left to ASTRA-Sim. The model's full `config.json` (read from
`configs/model/<org>/<name>.json`, or auto-fetched from the HF Hub on first run)
is written to a tmpdir at spin-up so vLLM never needs Hub access.

Attribution: the base layerwise-profile methodology (worker-extension hook into
vLLM's `layerwise_profile()`, single-GPU TP emulation via `hf_overrides`) is
adapted from [@waneon](https://github.com/waneon). The unified 4D attention
sweep, the heterogeneous-decode skew sweep in `profiler/core/skew.py`, and
the 5-axis weighted-LS alpha fit in `profiler/core/fit_alpha.py` are
developed in this repo.

Each run produces a per-category CSV bundle:

```
perf/<hw>/<model>/<variant>/
  meta.yaml                              profiler/vLLM version, effective engine kwargs, GPU,
                                         timestamps, compact sweep specs, skew_fit summary
  tp<N>/
    dense.csv                            layer, tokens, time_us
    per_sequence.csv                     layer, sequences, time_us
    attention.csv                        prefill_chunk, kv_prefill, n_decode, kv_decode, time_us
    moe.csv                              tokens, activated_experts, time_us   (MoE only)
    skew.csv                             raw heterogeneous-decode shots        (skew enabled)
    skew_fit.csv                         fitted per-bucket alpha table         (skew enabled)
```

`<variant>` is auto-derived from weight + KV dtype (e.g. `bf16`, `bf16-kvfp8`,
`fp8-kvfp8`) unless `--variant` is set. Times are in **microseconds**. Layers marked
`tp_stable: true` in the yaml (layernorms, sampler) are profiled once at TP=1 and
replicated into other `tp<N>/` folders by the writer.

The profiler Docker uses **vLLM v0.19.0** (`vllm/vllm-openai:v0.19.0` or
`v0.19.0-cu130` for CUDA 13.x). The MoE hook patches `FusedMoE.forward_native` for
forced expert routing — method name is version-specific.

### Skew profiling & alpha fit
FlashAttention's varlen kernel pays tile-padding + SM-imbalance costs when a
decode batch has non-uniform kv lengths. The uniform attention grid can't see
that (every shot uses a single kv_decode value), so `skew.py` runs a second
sweep on bimodal batches and measures three latencies per case — `t_mean`
(all decodes at the batch mean), `t_max` (all at the max), and `t_skew` (the
actual bimodal mix). The normalised alpha ∈ [0, 1],
`alpha = (t_skew − t_mean) / (t_max − t_mean)`, tells the simulator how far
along the mean→max line a skewed batch lands.

- **Sweep structure**: Tier 1 is a factorial over `(n, ratio, pc, kp, kvs)`
  at `_SKEW_REP = 4.0`; Tier 2 adds a skew-axis sweep at a handful of anchor
  pivots (`skew ∈ {1.5, 2, 4, 8, 16}`). Any CLI `SKEW_<axis>_FACTOR`
  (default 2.0) coarsens that axis geometrically — higher = faster, lower
  = denser. Factors and grid specs land in `meta.yaml::skew_profile`.
- **Fit**: `fit_alpha.py` groups rows by the 5-axis key
  `pc | n_label | skew_rate_label | kv_big_label | kp_label` and runs a
  weighted least-squares fit per cell. Axis ablation on the widened
  ~13k-sample dataset picked the 5-axis scheme (test p50/p90 ≈ 2.7% / 14.8%
  on TP=1 vs 3.5% / 16.4% for the previous 3-axis fit).
- **Data-driven bucket axes**: `n` and `kp` get one bucket per unique
  profiled value (`kp=0` sentinel + overflow), `kv_big` uses log-4x bins
  extended to the observed max, `skew_rate` is a fixed normalised [0, 1]
  scheme, and `pc` is keyed raw. Derived axes are written to
  `meta.yaml::skew_fit.bucket_axes`; the simulator reads them from there
  so widening the profile sweep lights up finer resolution without any
  simulator code change.
- **Storage**: the full (bucket → alpha) mapping spills to
  `tp<N>/skew_fit.csv` with columns `pc, n_label, skew_rate_label,
  kv_big_label, kp_label, alpha, n_samples`. `meta.yaml::skew_fit.per_tp[tp]`
  keeps only a summary (`method`, `n_samples`, `alpha_default`,
  `rel_err_p50/p90/p99`, `signed_mean`, `bucket_table` pointer). This
  turns meta.yaml from ~3100 lines into ~100 lines per variant. The
  simulator hydrates the CSV back into memory on `_load_perf_db()`.
- **Disable**: `SKIP_SKEW=1` skips the sweep entirely (simulator falls
  back to a pooled constant alpha). `ONLY_SKEW=1` skips every other
  category and refreshes just `skew.csv` + `skew_fit.csv`.

### Feasibility bounds shared by attention and skew
Both the uniform attention sweep and the skew sweep cap `n_reqs > max_num_seqs`
(strict `>`, not `>=`) so that `n = MSQ` **pure** cases (no prefill chunk) fit.
This uses vLLM V1's `input_batch` buffer exactly up to `MSQ`. Mixed cases at
`n = MSQ` need `MSQ + 1` requests and are still filtered. If a runtime workload
needs mixed-regime data at `n = X`, profile with `MAX_NUM_SEQS ≥ X + 1`.

### Canonical layer names (simulator ↔ profiler, unified)
The simulator consumes the profiler's per-category CSVs directly. Canonical
layer names match vLLM's own attribute names. `trace_generator` walks the
`sequence:` section of `profiler/models/<model_type>.yaml`; the table below
lists where each layer appears in the profiler CSVs and how the simulator keys
the lookup.

| Layer | Category (CSV) | Key semantics |
|-------|----------------|---------------|
| `embedding` | dense | `tokens = total_len` |
| `layernorm` | dense (tp_stable) | `tokens = total_len` |
| `qkv_proj` | dense | `tokens = total_len` |
| `qk_norm` | dense (tp_stable; Qwen3 only) | `tokens = total_len` |
| `rotary_emb` | dense | `tokens = total_len` |
| `attention` | attention | `(prefill_chunk, kv_prefill, n_decode, kv_decode)` |
| `o_proj` | dense + ALLREDUCE after (TP>1) | `tokens = total_len` |
| `gate_up_proj` | dense | `tokens = total_len` |
| `act_fn` | dense | `tokens = total_len` |
| `down_proj` | dense + ALLREDUCE after (TP>1) | `tokens = total_len` |
| `final_layernorm` | dense (tp_stable) | `tokens = total_len` |
| `lm_head` | per_sequence | `sequences = num_requests` |
| `sampler` | per_sequence (tp_stable) | `sequences = num_requests` |
| `moe` | moe (always profiled at tp=1; wrapped in EP ALLTOALL) | `(local_tokens, activated_experts)` |

### Trace generator structure
`trace_generator.py` walks the architecture yaml's `sequence:` section to emit
each iteration. Composable helpers:
- `resolve_variant()` / `_load_perf_db()` / `_load_architecture()` — resolve
  the variant folder, load meta.yaml, load per-category CSVs, and attach the
  architecture catalog + sequence.
- `_lookup_dense()` / `_lookup_per_sequence()` / `_lookup_attention()` /
  `_lookup_moe()` — category-specific lookups. Attention uses 4D lookup
  (nearest-neighbour on `prefill_chunk, n_decode`, bilinear on
  `kv_prefill, kv_decode`).
- `_emit_sequence()` — walks a list of canonical names from the yaml, attaches
  TP ALLREDUCE to `o_proj`/`down_proj`, swaps in PIM attention before the
  NPU attention kernel when offloading is enabled, and one-shot-warns when a
  sequence layer is missing from the profile CSVs.
- `_emit_prologue()` / `_emit_pre_attn_layers()` / `_emit_post_attn_layers()` /
  `_emit_final_layers()` — thin wrappers over `_emit_sequence`.
- `_synthesize_interleaved_trace()` — alternates two `BatchCtx` objects for
  sub-batch interleaving.
- `_emit_final_layers()` — final_layernorm → lm_head → sampler (sampler output goes to REMOTE)

### Trace file format
Each trace is a tab-separated text file consumed by the Chakra converter:

```
COLOCATED		model_parallel_NPU_group: {npu_group}
{num_layers}
Layername    comp_time    input_loc    input_size    weight_loc    weight_size    output_loc    output_size    comm_type    comm_size    misc
embedding_0  5621         REMOTE:0     40            LOCAL         1050673152     LOCAL         81920          NONE         0            NONE
...
sampler_291  25933        LOCAL        2565120       LOCAL         0              REMOTE:0      40             NONE         0            NONE
```

- `comp_time`: latency in nanoseconds (from profile.csv, converted at load time)
- `input_loc`/`weight_loc`/`output_loc`: `LOCAL` (NPU), `REMOTE:{node_id}` (CPU), `CXL:{id}`
- `comm_type`: `NONE`, `ALLREDUCE`, `ALLTOALL`, or with dimension scoping `ALLREDUCE:1,0`, `ALLTOALL:0,1`
  (the `:dim0,dim1` suffix maps to ASTRA-Sim's `involved_dim` BoolList for multi-dimensional topologies)
- `comm_size` on `qkv_proj` carries the **P/D KV transfer amount** (per layer, per rank,
  K+V only, honouring `kv_cache_dtype`), and is 0 unless `pd_type == "prefill"`.
  `convert_prefill` reads it for the per-layer SEND/RECV to the paired decode NPU. It used
  to read the layer's `output_size`, i.e. the whole QKV activation, which shipped Q as well
  and overstated the transfer by `(q_dim + 2*kv_dim) / (2*kv_dim)` — 3x for Llama-3.1-8B.
  `comm_type` stays `NONE` there: a SEND/RECV pair only needs comm_size/src/dst/tag
- `misc`: `NONE` or batch tag for sub-batch interleaving (`BATCH_1`, `BATCH_2`)
- First layer (embedding) input comes from `REMOTE` (CPU → NPU), last layer (sampler) output goes to `REMOTE` (NPU → CPU)
- MoE uses `EXPERT {i}` / `EXPERT END` markers (comm_type on EXPERT line can include dimension scoping)
- PIM uses `PIM {channel}` / `PIM END` markers

### Performance DB and latency lookup
The simulator loads per-category CSVs via `_load_perf_db()` and dispatches
lookups by catalog category: `_lookup_dense` (1D linear over tokens),
`_lookup_per_sequence` (1D linear over sequences), `_lookup_attention` (4D:
nearest-neighbour on `(prefill_chunk, n_decode)` + bilinear on `(kv_prefill,
kv_decode)`), and `_lookup_moe` (2D over `(tokens, activated_experts)`,
profiled at tp=1). All lookups extrapolate (time_us is linearly
extended) rather than clamping. Latencies are stored as microseconds in the
CSVs and converted to nanoseconds at load time. No calibration scaling —
profiled latencies are used directly.

Attention with skew correction: `_lookup_attention_with_skew` does two 4D
lookups (at `kv_decode_mean` and `kv_decode_max`) and blends them using
`alpha` resolved from `meta.yaml::skew_fit` by `_skew_alpha`. The bucket key
is `pc={pc}|{n_label}|{sr_label}|{kvb_label}|{kp_label}`, built against
`skew_fit.bucket_axes` from the meta (falling back to module defaults for
older profiles). `_hydrate_skew_fit_tables()` reads each TP's `skew_fit.csv`
into the in-memory `alpha_by_bucket` map on first load.

Profile CSV path: `profiler/perf/<hardware>/<model>/<variant>/tp<N>/{dense,
per_sequence,attention,moe,skew,skew_fit}.csv` (resolved as
`../profiler/perf/...` from the `astra-sim/` working directory).

Variant resolution: `trace_generator.resolve_variant(dtype, kv_cache_dtype,
model_config)` mirrors the profiler's `effective_variant` — weight dtype is
the CLI value or `torch_dtype` from the model config (default `bfloat16`),
KV dtype appends a `-kv<short>` suffix when not `auto`. Runtime lookups verify
the resulting folder exists; a mismatch raises a clear `FileNotFoundError`
pointing at the missing variant.

FP8 KV cache (`--kv-cache-dtype fp8`) resolves to a `<dtype>-kvfp8` variant
folder (e.g. `bf16-kvfp8`). The `kv_cache_dtype` parameter is threaded through
`generate_trace` → `resolve_variant` → `_load_perf_db`. In `memory_model.py`,
`kv_fp` is 1 byte for fp8 (vs `fp` for others), halving KV cache memory usage.

Runtime vs. profiled warnings: on first load of a `(hardware, model, variant)`,
the simulator compares the CLI's `--max-num-batched-tokens` and `--max-num-seqs`
against `meta.yaml`'s `engine_effective` values and logs a one-shot warning
when the runtime exceeds the profiler's sweep bounds (lookups will extrapolate).

### Agentic session support (dependency chains)
The simulator supports closed-loop agentic workloads (SWE-bench, tool-calling agents)
where LLM calls within a session form a dependency chain interleaved with tool calls.

**Dataset format:** Each JSONL line is a session with `sub_requests[]`. Each sub-request
has `input_toks`, `output_toks`, `tool_duration_ns` (wait time after this LLM call before
the next can start). Flat requests (no `sub_requests` key) are also supported for backward
compatibility. Both formats can coexist in the same file.

**Router dependency tracking** (`router.py`):
- `load_requests()` auto-detects flat vs agentic format. For agentic sessions, only the
  first sub-request is queued; the rest are stored in `_deferred_sessions`
- `notify_request_completed(request_id, completion_time_ns)` releases the next sub-request
  at `completion_time + tool_duration_ns` and inserts it sorted into `_pending_requests`
- `has_deferred_sessions()` prevents premature simulation exit while sessions are in-flight
- `scheduler.add_request()` uses `bisect.insort` (not `append`) to maintain arrival-time
  sort order when dynamically released sub-requests enter the queue

**Time advancement:** When all instances are idle but deferred sub-requests have future
arrival times (tool calls still running), `serving/__main__.py` advances `current` to the next pending
arrival time to avoid busy-looping.

### Scheduler and memory model
- `scheduler.py` follows vLLM V1's `schedule()` shape: a persistent `self.running` set is
  served first (phase A), preempting only from its own tail, then `self.waiting` is admitted
  (phase B) while budget and slots remain. Phase B **never** preempts, and it is skipped
  entirely on any step that preempted — that anti-thrash rule is load-bearing
- There is **one** `schedule()` for prefix caching on and off. The pool handles
  `enable_caching=False` the way vLLM does (allocate through the same free list, never
  index), so do not reintroduce a second scheduler
- Token budget controlled by `--max-num-batched-tokens` (default 2048) and `--max-num-seqs` (default 128)
- `--long-prefill-token-threshold` caps per-request tokens per step for chunked prefill
- **There is no prefill phase or decode phase.** A request just catches up to
  `num_tokens_reached`, so `num_new = num_tokens_reached - num_computed_tokens` — 1 in
  steady-state decode, the whole sequence for a resumed request. `Request.is_prefill()` is
  gone on purpose: it read `original_input` and would misread a resumed request's
  recomputation as decoding. The trace classifies by **scheduled token count**
  (>1 = prefill chunk, ==1 = decode)
- `num_computed_tokens` is advanced at **schedule** time, as in vLLM's
  `_update_after_schedule`. `Batch.scheduled_tokens` is the snapshot `add_done` works from.
  Advancing at completion instead lets `pp_size > 1` schedule the same tokens twice
- `num_tokens_reached` (prompt + generated) must never be derived from
  `num_computed_tokens`: preemption resets the latter to 0
- Preemption is vLLM verbatim, including `num_computed_tokens = 0`. That is **not**
  re-prefill — `free_blocks` keeps the blocks' hashes, so on re-admission
  `get_computed_blocks` finds whatever is still resident, a lower tier returns what was
  written down, and only the remainder is recomputed. Do not add a "preserve the decode
  state" special case; the tiers are what preserve it
- The three modes map onto three real vLLM configurations: `--no-enable-prefix-caching` =
  vLLM with prefix caching off (a resume recomputes the whole sequence);
  `--enable-prefix-caching` = default vLLM; plus `--prefix-storage CPU/CXL` = vLLM with
  LMCache / `OffloadingConnector` attached
- KV capacity is `npu_mem * gpu_memory_utilization - weight`, divided into `--block-size`
  blocks. vLLM also subtracts the activation peak and CUDA context, which are not modelled,
  so the simulator's capacity is an upper bound at the same utilization
- `block_pool.py` / `kv_cache_manager.py` own everything about which blocks exist and where.
  `memory_model.py` keeps the static sizing math and a byte-level view; `npu_used` is a
  property over the pool, so there is one ledger per tier
- `calculate_sizes(parallel=)` computes per-layer tensor sizes — `parallel` is TP for dense
  layers and EP for MoE experts. Uses `head_dim`, `q_dim`, `kv_dim`
- MoE expert weights are sharded by `ep_size` (not `tp_size`)
- Prompt throughput (`prompt_t` in `add_done()`) includes prefix cache hit tokens,
  not just actually computed prefill tokens. This matches vLLM's reported prompt
  throughput which counts all input tokens including cached ones
- `kv_load` / `kv_evict` trace layers fire **only** with `--prefix-storage`: without a lower
  tier there is nothing to recall from, so both byte counts are 0. `batch.evict` is 0 in
  every mode — eviction from the NPU costs nothing, because the data is either a finished
  request's cache or already written down off the critical path

### CLI argument conventions
CLI flags follow vLLM naming where applicable:
- `--dtype` (`float16`, `bfloat16`, `float32`, `int8`) — model weight precision
- `--skip-prefill` — skip the prefill phase (decode only)
- `--request-routing-policy` (`LOAD`, `RR`, `RAND`, `CUSTOM`) — request routing across instances
- `--expert-routing-policy` (`BALANCED`, `RR`, `RAND`, `CUSTOM`) — expert token routing for MoE
  (block-copy optimization is controlled separately via `--enable-block-copy`, default on)
- Boolean flags use `argparse.BooleanOptionalAction` (e.g., `--enable-prefix-caching` /
  `--no-enable-prefix-caching`)

### Head dimension
Some models (e.g., Qwen3) have `head_dim != hidden_size // num_attention_heads`. Always use:
```python
head_dim = config.get('head_dim', n_embd // n_head)
q_dim = n_head * head_dim        # NOT n_embd
kv_dim = kv_head * head_dim      # NOT n_embd // group
```

### Model configs
Model architecture configs live in `configs/model/{org}/{model}.json`. These are subsets
of HuggingFace `config.json` containing fields the simulator needs (`hidden_size`,
`num_attention_heads`, `num_hidden_layers`, `num_key_value_heads`, `intermediate_size`,
`vocab_size`, `head_dim`, `num_local_experts`, `num_experts_per_tok`).

The simulator loads these via `get_config(model_name)` in `utils.py`.

### Cluster configs
Cluster configs in `configs/cluster/` define hardware topology. Key instance fields:
- `hardware`: must match a directory name in `profiler/perf/<hardware>/`
- `model_name`: must match a config in `configs/model/{model_name}.json`
- `num_npus`: total GPUs for the instance (optional, inferred from `tp_size * pp_size`)
- `tp_size`: tensor parallel degree (required or inferred)
- `pp_size`: pipeline parallel degree (optional, default 1)
- `ep_size`: expert parallel degree (optional, default `tp_size` for MoE, 1 for dense)
- `dp_group`: DP group ID string (optional, instances with same string share experts)
- `npu_mem.mem_bw`: NPU memory bandwidth (also set as `local-mem-bw` in system.json)
- `cpu_mem.mem_bw`: CPU memory bandwidth (set as remote memory in memory_expansion.json)
- `link_bw`: inter-node bandwidth in GB/s (set in network.yml)
- `link_latency`: inter-node link latency in ns

Parallelism inference: users may provide partial info (e.g., `num_npus=4, tp_size=2`)
and `config_builder.py` infers the rest (`pp_size=2`). Validation ensures
`num_npus = tp_size * pp_size` and `ep_size` divides `num_local_experts`.

TP and EP share the same GPUs: non-MoE layers use TP (ALLREDUCE), MoE layers use EP
(ALLTOALL). DP is achieved via multiple instances with the same `dp_group`.

`config_builder.py` reads the cluster config and generates three ASTRA-Sim input files:
- `astra-sim/inputs/network/network.yml` — topology and bandwidth
- `astra-sim/inputs/system/system.json` — scheduling policy and memory bandwidth
- `astra-sim/inputs/memory/memory_expansion.json` — remote (CPU) memory config

### Working directory
`serving/__main__.py` changes cwd to `astra-sim/` early in execution. All relative paths in the simulator
resolve from `astra-sim/`, not the repo root. Paths to `configs/`, `workloads/`, `profiler/`
are relative to the repo root and prefixed with `../` in code.

### Communication sizes for ASTRA-Sim
ASTRA-Sim expects the **total** data size for collectives (not per-NPU). It divides by N
internally (`msg_size = data_size / nodes_in_ring`).
- ALLREDUCE on `o_proj` and `down_proj`: pass full output tensor size
- ALLTOALL for MoE: pass full activation tensor size

### Multi-dimensional topology and `involved_dim`
For DP+EP configurations, the network topology is 2D: `npus_count: [tp_size, dp_group_size]`.
Collectives are scoped to specific dimensions via the `involved_dim` BoolList attribute
on COMM_COLL_NODE protobuf nodes:
- ALLREDUCE (TP): `involved_dim=[True, False]` — dim 0 only
- ALLTOALL (EP): `involved_dim=[False, True]` — dim 1 only (or `[True, True]` if EP spans TP+DP)

The `involved_dim` is encoded in the trace `comm_type` field as `ALLTOALL:0,1` (parsed by
the Chakra converter's `_parse_comm_type`). ASTRA-Sim's `Workload::issue_comm()` reads this
and passes it to `generate_all_to_all()`, which skips dimensions where `involved_dim` is false.

The `system.json` collective implementations must have one entry per topology dimension
(e.g., `"all-to-all-implementation": ["ring", "ring"]` for 2D). `config_builder.py`
generates this automatically based on whether DP groups are present.

### MoE expert blocks
Expert blocks use `EXPERT {i}` / `EXPERT END` markers for ASTRA-Sim. Each EP rank
gets a per-rank latency from profiled data based on its local token count and activated
experts (`key_0=local_tokens, key_1=activated_experts`, profiled at tp=1). Ranks execute
in parallel and sync at the ALLTOALL barrier. Expert-to-rank assignment uses even
partitioning: `expert_id * ep_size // num_experts`.

### DP+EP wave synchronization
For DP groups (instances with the same `dp_group`), wave synchronization is achieved
through two mechanisms:
1. **Python-side dp_pending barrier**: trace generation is deferred until all DP group
   members have scheduled their batches. The ALLTOALL `comm_size` is synchronized to
   `max(total_len) * hidden_size * fp` across the group.
2. **ASTRA-Sim ALLTOALL barrier**: all DP group instances' `.et` files are placed in a
   shared workload folder. The ALLTOALL collectives in both files have matching stream
   IDs, causing ASTRA-Sim to block until both NPUs reach the collective.

When one DP instance is idle (no requests), a dummy batch (1 decode token) is created
so it can participate in the ALLTOALL sync. When one instance finishes all requests,
it continues generating dummy batches until all DP group members are done.

### Chakra graph converter
The Chakra converter (`astra-sim/extern/graph_frontend/chakra/src/converter/llm_converter.py`)
transforms text traces into protobuf `.et` files. It creates:
- `MEM_LOAD_NODE` for the first layer's input (from REMOTE/CPU memory)
- `COMP_NODE` for each computation layer
- `MEM_STORE_NODE` for the last layer's output (to REMOTE/CPU memory)
- `COMM_COLL_NODE` for ALLREDUCE/ALLTOALL (with optional `involved_dim` BoolList attribute)

The converter parses `comm_type` strings like `ALLTOALL:0,1` via `_parse_comm_type()`,
splitting into `comm_type="ALLTOALL"` and `involved_dim=[False, True]`.

The MEM_STORE node uses the **last layer's** `output_memory_loc`. This is why the sampler
(not lm_head) must have `output_loc=REMOTE:{node_id}`.

Memory location types: `LOCAL` (NPU) = 1, `REMOTE` (CPU) = 2, `CXL` = 3, `STORAGE` = 4.
These must match the C++ enum in `astra-sim/astra-sim/system/AstraMemoryAPI.hh`.

### Docker environments
- **vLLM container** (used by `python -m profiler`, `python -m bench`, and
  `python -m workloads.generators`): `vllm/vllm-openai:v0.19.0` (or
  `v0.19.0-cu130` for CUDA 13.x)
  - Launched via `scripts/docker-vllm.sh`
  - Mounts the **LLMServingSim repo root** as `/workspace`; container cwd
    is `/workspace`, so `python -m profiler …` etc. work directly
  - Pre-installs `datasets` and `matplotlib` on first start (extra deps
    used by the workload generator and bench plots; vLLM brings the rest)
  - Set `HF_TOKEN` in `scripts/docker-vllm.sh` for gated-config auto-download
- **Simulator container**: `astrasim/tutorial-micro2024` + Python deps
  - Launched via `scripts/docker-sim.sh`
  - Mounts the repo root at `/app/LLMServingSim`; ASTRA-Sim + Chakra are
    built inside via `scripts/compile.sh` on first use

## README and docs split

The repo has two documentation surfaces with deliberate scope:

- **`README.md`** — minimal front door. About / Getting Started / Publications /
  Citation only. Logo + link bar (Website / Documentation / Contribute /
  Contact / Changelog) point everything else out to the website. **Do not
  re-add detailed content (CLI flag tables, dataset schema, profiler
  walkthroughs, validation plots, etc.) to the README** — it lives on the
  website now.
- **`docs/`** — the public docs site (Docusaurus 3, deployed at
  `https://llmservingsim.ai`). All long-form content lives here. See
  `docs/AGENTS.md` for site-specific conventions.

When you add a new feature with user-visible behavior, document it on the
website (not the README).

## Commit & Pull Request Guidelines

- Short imperative commit messages: `Fix incorrect evict_size accumulation`,
  `Add Qwen3 model support`
- Keep commits focused — one logical change per commit
- Include the exact command used for validation and note any output CSV path in PRs
- Describe which simulation mode is affected and the config/dataset used

## Testing & Validation

No dedicated unit-test suite. Validate by:
1. Running the smallest relevant `python -m serving …` scenario and inspecting
   the per-request CSV.
2. For end-to-end accuracy checks against real vLLM, use `python -m bench run`
   followed by `python -m bench validate` (see `bench/README.md`).
3. For profiler changes: edit `MODEL` / `HARDWARE` in `profiler/profile.sh`
   and run `./profiler/profile.sh` from the repo root inside the vLLM container.

## Common Pitfalls

- **Don't edit `astra-sim/`** unless the change targets simulator integration
  (e.g., `llm_converter.py`, `Workload.cc`, input configs)
- **Don't commit large files**: generated traces, output CSVs, `.et` files are gitignored
- **Don't use machine-specific absolute paths** in configs or code — use relative paths
  rooted at the repo
- **Don't add `getattr` fallbacks** for Request attributes — initialize all attributes
  in `Request.__init__` and access directly
- **Don't reintroduce `is_prefill()` or a prefill/decode branch in the scheduler.** vLLM has
  neither; a request just catches up to `num_tokens_reached`. Classify for the trace by
  scheduled token count instead
- **Don't add a "preserve the decode state on preemption" special case.** `num_computed_tokens
  = 0` is vLLM's own behaviour and is not re-prefill — recovery comes from the block hashes
  surviving in the pool and from the lower tier. Two earlier attempts to special-case this
  cost 375 preemptions / 293k recomputed tokens and 41,569 preemptions / 6 TB of swap
- **Don't derive sequence length from `num_computed_tokens`** — preemption resets it.
  `num_tokens_reached` is the independent counter, mirroring vLLM's `len(_all_token_ids)`
- **Don't assume `hidden_size == num_heads * head_dim`** — use explicit `head_dim` from config
- **Use canonical vLLM layer names** (`qkv_proj`, `o_proj`, `gate_up_proj`,
  `act_fn`, `down_proj`, `rotary_emb`, `qk_norm`, `attention`, `layernorm`,
  `final_layernorm`, `embedding`, `lm_head`, `sampler`, `moe`). Every name the
  simulator emits must also appear in the architecture yaml's catalog.
- **Profiler CSVs store microseconds** (`time_us` column) — the simulator
  multiplies by 1000 and rounds to nanoseconds at load time
- **First and last trace layers must use REMOTE** — the Chakra converter creates a MEM_LOAD
  node from the first layer's input_loc and a MEM_STORE node from the last layer's output_loc;
  if either is LOCAL without local_mem configured, ASTRA-Sim crashes
- **memory_expansion.json only has remote_mem by default** — local_mem is not configured unless
  `--enable-local-offloading` is used; weight loads from LOCAL go through compute time, not memory
- **`config_builder.py` regenerates ASTRA-Sim inputs on every run** — don't manually edit
  `astra-sim/inputs/` files expecting them to persist
