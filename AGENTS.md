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
│   ├── run.sh                  # One runnable example per feature (a menu, not a suite)
│   ├── validate.sh             # every scenario vs recorded clocks + bench/examples digests
│   └── validate-baselines.txt  # the recorded values; refresh with validate.sh --update
├── configs/
│   ├── cluster/                # Cluster topology configs (hardware, memory, instances)
│   ├── model/                  # Model architecture configs (subset of HF config.json)
│   └── pim/                    # PIM device configs (DRAMSim3 INI format)
├── workloads/                   # Request trace datasets (.jsonl)
│   └── generators/             # ShareGPT/etc → JSONL workload generators
├── profiler/                   # vLLM-based layerwise profiler (`python -m profiler`)
│   ├── __main__.py             # CLI dispatch (profile / slice / coverage)
│   ├── core/                   # internals
│   │   ├── runner.py           # Orchestration (spin_up → categories → spin_down)
│   │   ├── config.py           # Architecture / ProfileArgs / engine defaults
│   │   ├── engine.py           # vLLM lifecycle (tmpdir-based local config load)
│   │   ├── categories.py       # Dense / PerSequence / Attention / LinearAttention / Expert
│   │   ├── skew.py             # Heterogeneous-decode skew sweep
│   │   ├── fit_alpha.py        # per-kernel 5-axis weighted-LS alpha fit
│   │   ├── writer.py           # CSV + meta.yaml writer, TP-stable replication
│   │   ├── stack.py            # per-layer block composition from the HF config (shared with serving/)
│   │   ├── catalog_path.py     # model_type → yaml resolution (shared with serving/)
│   │   ├── logger.py           # Rich-based logger + stdio capture
│   │   └── hooks/              # vLLM-internal-API touchpoints (worker ext, MoE patch, etc.)
│   ├── models/                 # Architecture yamls, one per HF `model_type`
│   ├── power/                  # nvidia-smi / IPMI power-logging helpers
│   ├── perf/                   # Output: perf/<hw>/<model>/<variant>/tp<N>/{dense,per_sequence,attention,linear_attention,moe,skew,skew_fit}.csv
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
│   ├── results/                # output: bench/results/<run_id>/ (gitignored)
│   ├── examples/               # committed end-to-end runs, keyed <hardware>/<model>/
│   │   ├── <hw>/<model>/config.json   # the cluster config the example runs
│   │   ├── <hw>/<model>/vllm/         # ground truth: meta.json, requests.jsonl, timeseries.csv
│   │   ├── <hw>/<model>/outputs/      # sim.csv, sim.log
│   │   ├── <hw>/<model>/validation/   # summary.txt + plots
│   │   ├── run.sh              # re-run the simulator side: run.sh <hardware>/<model>
│   │   └── validate.sh         # re-run the comparison: validate.sh <hardware>/<model>
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
profiler/perf/<hw>/<model>/<variant>/tp<N>/*.csv (profiled latencies)
    ↓ _load_perf_db() + _lookup_{dense,per_sequence,attention,moe}()
trace_generator.py → per-layer field tuples (TraceData)
    ↓ Chakra converter, in-process, via LLMConverter.convert_rows()
graph_generator.py → .et protobuf file (reused when the rows repeat)
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
per-sequence / attention / linear-attention / moe) to vLLM class names.

**Catalog naming rule.** The file is named after the `model_type` it primarily
serves, spelled **verbatim** — including where vendors disagree: DeepSeek
writes V3.2 as `deepseek_v32`, Qwen writes 3.5 as `qwen3_5`, so the files are
`deepseek_v32.yaml` and `qwen3_5.yaml`. Copying upstream's spelling is what
keeps the rule mechanical; normalising it would mean the filename matches no
`model_type` and every lookup falls through to the directory scan. `model_type` is read from the top level of the config handed to the
profiler, so for a wrapped (VL) checkpoint the convention is to store the
**text tower flattened to top level** with `architectures` set to the text-only
class — that makes `qwen3_5_text`, not the wrapper's `qwen3_5`, the recorded
name. When several `model_type` values are the same implementation (GLM-5's
`glm_moe_dsa` and DeepSeek-V3.2's `deepseek_v32` both run vLLM's `deepseek_v2`
path), one file lists them all under `model_types:` rather than the catalog
being duplicated or symlinked. Two files claiming one `model_type` is an error,
not first-wins.

**One layer-order form: `blocks:` + `shared:`.** `blocks:` is keyed by **axis**
(`blocks.attn.<layer_types value>`, `blocks.sparse_attn.<same>` as an overlay,
`blocks.mlp.dense|moe`) rather than by block name, because a layer's identity
is a tuple and enumerating combinations explodes. `shared:` holds `prologue`
and `head`, which run once per iteration. A uniform stack is the degenerate
case — one entry per axis.

There was a second form, `sequence:`, for uniform stacks. **It is gone; do not
reintroduce it.** It was this one flattened (`pre_attn`/`post_attn` = the single
implicit `attn.full_attention`, `mlp_dense`/`mlp_moe` = the `mlp` axis with the
value in the key name), and two forms meant two code paths — the simulator only
implemented the flat one, so Qwen3.5 and MiniMax-M3 could not be simulated at
all. The flattening also encoded a false claim: with the axis in the key there
is no per-layer question left, and the MLP was resolved once per *model*, which
modelled DeepSeek-V3.2's and GLM-5's first three dense layers as MoE.

Which block a given layer runs comes from the *checkpoint's* config
(`layer_types`, `first_k_dense_replace`, `decoder_sparse_step`,
`moe_layer_freq`, `sparse_attention_freq`, `index_topk_pattern`), never from
the yaml. `profiler/core/stack.py` owns those rules and **both** the profiler
and the simulator import it — the profiler to decide how many layers to
instantiate, the simulator to decide which block each layer emits. Two
implementations would drift, and the rules disagree on the off-by-one in
opposite directions between vendors. Profiling a hybrid needs
`--num-hidden-layers` raised to the smallest count that instantiates every
block type — 4 for Qwen3.8-27B — since the default of 1 only ever reaches one;
the profiler resolves that itself.

**Write a catalog from a live profile dump, not from vLLM's source.** The
module tree and the profile tree differ both ways: `rotary_emb`, `q_norm`,
`k_norm` and `RMSNormGated` are real modules that never become profile nodes
(binding them silently measures nothing), while raw CUDA kernels that are not
modules *can* be bound by name (`vllm: _causal_conv1d_fwd_kernel`) and are the
only way to reach gated DeltaNet's conv and decode recurrence. Kernel identity
also changes with the batch regime — a GDN block runs one set for pure prefill,
another for pure decode, a third for a mixed batch — so a catalog written from
a single shot binds the wrong kernel for the others.

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
    linear_attention.csv                 layer, prefill_tokens, n_decode, time_us  (mamba/GDN only)
    moe.csv                              tokens, activated_experts, time_us   (MoE only)
    skew.csv                             raw heterogeneous-decode shots        (skew enabled)
    skew_fit.csv                         fitted per-bucket alpha table         (skew enabled)
```

`attention.csv`, `skew.csv` and `skew_fit.csv` are keyed by **`layer`** as well:
a sparse-attention model has two or three kernels in that category and they
share neither a latency curve nor an alpha (on MiniMax-M3 the same batch fits
0.24 / 0.74 / -0.01 for `attention` / `indexer` / `sparse_attention`). A bundle
profiled before those columns existed holds one kernel and it is `attention`.

`<variant>` is auto-derived from weight + KV dtype (e.g. `bf16`, `bf16-kvfp8`,
`fp8-kvfp8`) unless `--variant` is set. Times are in **microseconds**. Layers marked
`tp_stable: true` in the yaml (layernorms, sampler) are profiled once at TP=1 and
replicated into other `tp<N>/` folders by the writer.

The profiler Docker uses **vLLM v0.28.0** (`vllm/vllm-openai:v0.28.0`, or
`v0.28.0-cu129` on a CUDA 12.9 host). The MoE hook forges expert routing by
patching `_compute_routing` on the live router instance — every symbol under
`profiler/core/hooks/` is a vLLM *internal* API and is version-specific. v0.28
restructured MoE substantially: the old `FusedMoE` module is gone, replaced by
`FusedMoEFactory` returning a `MoERunner` that owns a `router` (`BaseRouter`)
and a `RoutedExperts`.

### Skew profiling & alpha fit
FlashAttention's varlen kernel pays tile-padding + SM-imbalance costs when a
decode batch has non-uniform kv lengths. The uniform attention grid can't see
that (every shot uses a single kv_decode value), so `skew.py` runs a second
sweep on bimodal batches and measures three latencies per case — `t_mean`
(all decodes at the batch mean), `t_max` (all at the max), and `t_skew` (the
actual bimodal mix). The alpha
`alpha = (t_skew − t_mean) / (t_max − t_mean)` tells the simulator how far
along the mean→max line a skewed batch lands. It is **not** clamped to
[0, 1], in the fit or in `_skew_alpha`: measured p50 is 0.07–0.13, but
14–20% of rows are negative (endpoint gap inside measurement noise) and
2–5% exceed 1 (a skewed mix genuinely costing more than uniform-max,
which tile padding and SM imbalance do not bound). Rows with
`t_max <= t_mean` are recorded `nan` and dropped.

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
- **Disable**: `SKIP_SKEW=1` skips the sweep entirely, and the simulator
  then applies **no** skew correction (`_ATTN_SKEW_ALPHA_FALLBACK = 0`,
  i.e. `t_mean`). Deliberately not a borrowed constant: the endpoint gap
  `(t_max - t_mean)` is a large fraction of an iteration, so alpha has to
  be known to ~±0.02 to be worth applying. Buckets with no samples
  *inside* a real fit still fall back to that fit's own pooled
  `alpha_default`, measured on the same GPU. `ONLY_SKEW=1` skips every
  other category and refreshes just `skew.csv` + `skew_fit.csv`.

### Feasibility bounds shared by attention and skew
Both the uniform attention sweep and the skew sweep cap `n_reqs > max_num_seqs`
(strict `>`, not `>=`) so that `n = MSQ` **pure** cases (no prefill chunk) fit.
This uses vLLM V1's `input_batch` buffer exactly up to `MSQ`. Mixed cases at
`n = MSQ` need `MSQ + 1` requests and are still filtered. If a runtime workload
needs mixed-regime data at `n = X`, profile with `MAX_NUM_SEQS ≥ X + 1`.

### Canonical layer names (simulator ↔ profiler, unified)
The simulator consumes the profiler's per-category CSVs directly. Canonical
layer names match vLLM's own attribute names. `trace_generator` walks the
`blocks:` section of `profiler/models/<model_type>.yaml`; the table below
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

The **`linear_attention`** category is keyed on `(prefill_tokens, n_decode)`,
and that key is what makes it regime-aware: a gated-DeltaNet block runs a
*different set of kernels* for a pure prefill, a pure decode and a mixed batch,
so a kernel simply has no rows for a regime it does not fire in, and the
simulator emits nothing for it there. Measured on Qwen3.8-27B:

| kernel | prefill | decode | mixed |
|---|---|---|---|
| `gdn_conv_prefill`, `gdn_post_conv`, `gdn_prefill` | yes | — | yes |
| `gdn_conv_decode`, `gdn_decode` | — | yes | — |
| `gdn_decode_mixed` | — | — | yes |
| `gdn_in_proj`, `gdn_out_proj`, `gdn_norm`, `gdn_glue` | yes | yes | yes |

Put a regime-dependent kernel in `dense` or `per_sequence` and it is emitted on
**every** batch, because neither has a notion of regime — that charged a decode
conv on a pure prefill and a prefill conv on a pure decode. The always-on
layers stay in `dense`, which is why the split is per layer rather than per
block. Use `python -m profiler coverage` to find out which is which; it reports
per regime.

The `attention` category holds **more than one kernel** on a sparse model, and
they are not interchangeable — the lookup takes a layer name
(`attention_by_layer`) and so does the skew alpha. MiniMax-M3 profiles
`attention` (its non-sparse layers), `sparse_attention` and `indexer`;
DeepSeek/GLM profile `attention` (MLA) and `indexer`. There is deliberately no
pooled `tables["attention"]` shortcut: there was one, every lookup took it, and
a sparse layer got the dense kernel's latency (2.1x per layer on M3).

Names beyond this table are per-family and live in the catalogs
(`gdn_*`, `mla_*`, `indexer_*`, `sparse_*`, `*_glue`). **`calculate_sizes` in
`memory_model.py` raises on a name it does not know**, and it knows only the 16
above — so a new family needs its tensor-size formulas added there before it
can be simulated, which is outstanding for Qwen3.5 (12 names), DeepSeek/GLM
(10) and MiniMax-M3 (6).

### Trace generator structure
`trace_generator.py` walks the architecture yaml's `blocks:` section to emit
each iteration. Composable helpers:
- `resolve_variant()` / `_load_perf_db()` / `_load_architecture()` — resolve
  the variant folder, load meta.yaml, load per-category CSVs, attach the
  architecture catalog, and resolve the checkpoint's per-layer block list once
  (`perf_db["layer_stack"]`, via `profiler/core/stack.py`).
- `_lookup_dense()` / `_lookup_per_sequence()` / `_lookup_attention()` /
  `_lookup_moe()` — category-specific lookups. Attention is a 4D lookup:
  each axis is bracketed by its two neighbouring profiled values and
  blended **linearly** (`_axis_bracket`).
- `_shared_layers()` / `_layer_spec()` / `_block_layers()` — the block walk.
  `_block_layers(perf_db, layer_num, part)` answers "what does *this* layer
  emit for `pre_attn` / `post_attn` / `mlp`", consulting
  `blocks.sparse_attn` first and falling through to `blocks.attn`.
- `_emit_sequence()` — walks a list of canonical names from a block, attaches
  TP ALLREDUCE to `o_proj`/`down_proj`, swaps in PIM attention before the
  NPU attention kernel when offloading is enabled, and one-shot-warns when a
  layer is missing from the profile CSVs.
- `_emit_prologue()` / `_emit_pre_attn_layers()` / `_emit_post_attn_layers()` /
  `_emit_final_layers()` — thin wrappers over `_emit_sequence`.
- `_block_copy_key()` — the reuse key for a built block: `None` when it must be
  rebuilt (block mode, or a non-deterministic MoE router), otherwise the
  layers' own `LayerSpec`s. Build once per distinct block *shape*, replay for
  every layer that shares it.
- `_synthesize_interleaved_trace()` — alternates two `BatchCtx` objects for
  sub-batch interleaving.
- `_emit_final_layers()` — final_layernorm → lm_head → sampler (sampler output goes to REMOTE)

### Trace file format
A trace is a list of per-layer field tuples, handed to the Chakra converter
in memory. `--save-trace-text` also writes it as a whitespace-aligned text
file (fixed-width columns from `utils.py::_FMT`), which is the only
human-readable form of what the simulator emitted — nothing in the pipeline
reads it. The field spec below describes both representations; the text
columns are what `indexed_cols()` and `_write_trace()` agree on. The sample below is illustrative, not byte-accurate — see
`docs/docs/reference/trace-format.md` for the real widths. Every field except the
last carries an **explicit trailing space** in `_FMT`: `{:<15}` pads a short value
but emits nothing for one that already fills the column, so a 15-character value
like `ALLREDUCE:1,0,0` (3-D `involved_dim`) would otherwise run into the next
field and the row would come back one short. `header()` goes through `_FMT` too, so
the whole file has one layout:

```
COLOCATED		model_parallel_NPU_group: {pp_size}		pp_stage_boundaries: 73,145,217
{num_layers}
Layername    comp_time    input_loc    input_size    weight_loc    weight_size    output_loc    output_size    comm_type    comm_size    misc
embedding_0  5621         REMOTE:0     40            LOCAL         1050673152     LOCAL         81920          NONE         0            NONE
...
sampler_291  25933        LOCAL        2565120       LOCAL         0              REMOTE:0      40             NONE         0            NONE
```

- Line 1 carries `key: value` pairs after the mode marker.
  `model_parallel_NPU_group` is `pp_size`. `pp_stage_boundaries` appears only when
  `pp_size > 1` and lists the `pp_size - 1` line indices at which each stage after
  the first begins, counted **after** any leading `kv_load`/`kv_evict` rows (the
  converter strips those before partitioning). The frontend places them on
  transformer-block starts using vLLM's `get_pp_indices` rule — blocks split
  evenly, remainder to the stages *before* the last, which also carries
  `lm_head`
- `comp_time`: latency in nanoseconds (from the per-category CSVs, whose `time_us` is converted at load time)
- `input_loc`/`weight_loc`/`output_loc`: `LOCAL` (NPU), `REMOTE:{node_id}` (CPU), `CXL:{id}`
- `comm_type`: `NONE`, `ALLREDUCE`, `ALLTOALL`, `ALLGATHER`, `REDUCESCATTER`, or with
  dimension scoping `ALLREDUCE:1,0`, `ALLTOALL:0,1`, `ALLREDUCE:1,0,0` (the
  `:dim0,...` suffix maps to ASTRA-Sim's `involved_dim` BoolList for
  multi-dimensional topologies, one entry per topology dimension)
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
`_lookup_per_sequence` (1D linear over sequences), `_lookup_attention` (4D
linear over `(prefill_chunk, kv_prefill, n_decode, kv_decode)`), and
`_lookup_moe` (2D over `(tokens, activated_experts)`, profiled at tp=1).
All lookups extrapolate (time_us is linearly extended) rather than
clamping.

`_axis_bracket` blends on a **linear** scale, not in log space, even
though the profiler sweeps every axis geometrically. Grid spacing decides
where the kernel is sampled; the blend decides how two samples are
combined; the kernel is linear in each axis. Profiled decode attention
fits `time_us = a + b * (n_decode * kv_decode)` with R^2 = 1.0000 (RTX
4090 / Llama-3.1-8B, implied 953 GB/s = 95% of spec — a pure
KV-bandwidth read). Log blending of a per-axis-linear function is
convex-biased upward by up to +6.0% per axis on a doubling grid, and
leave-one-out over the measured grid puts it at +11.6% to +14.4% mean
error against +2.3% to +3.7% for linear, across every bundle in
`profiler/perf/`. Don't "restore" log space because the sweep is
geometric. Latencies are stored as microseconds in the
CSVs and converted to nanoseconds at load time. No calibration scaling —
profiled latencies are used directly.

Attention with skew correction: `_lookup_attention_with_skew` looks up at
`kv_decode_mean` and, only when a non-zero `alpha` applies, blends toward a
second lookup at `kv_decode_max`. `alpha` is resolved from
`meta.yaml::skew_fit` by `_skew_alpha`, and the function returns `t_mean`
after a single lookup for `n_decode <= 1`, for a batch whose decode kv
lengths are all equal, or for `alpha == 0` (the default with no skew
profile). The bucket key
is `pc={pc}|{n_label}|{sr_label}|{kvb_label}|{kp_label}`, built against
`skew_fit.bucket_axes` from the meta (falling back to module defaults for
older profiles). `_hydrate_skew_fit_tables()` reads each TP's `skew_fit.csv`
into the in-memory `alpha_by_bucket` map on first load.

Profile CSV path: `profiler/perf/<hardware>/<model>/<variant>/tp<N>/{dense,
per_sequence,attention,moe,skew,skew_fit}.csv` (resolved as
`../profiler/perf/...` from the `astra-sim/` working directory).

Variant resolution: `trace_generator.resolve_variant(model_config)` mirrors
the profiler's `effective_variant`, and is a **pure function of the model
config** — it takes no dtype argument, because the simulator has no dtype
input. Weight dtype is `utils.config_weight_dtype` (`quantization_config`
first, then `torch_dtype` / `dtype`), KV dtype is `utils.config_kv_cache_dtype`
and appends a `-kv<short>` suffix when not `auto`. Runtime lookups verify the
resulting folder exists; a miss raises a clear `FileNotFoundError` pointing at
the missing variant. The profiler can still *write* other variants for the same
model (`--variant`, `--dtype`, `--kv-cache-dtype`), which is how a deliberate
second precision is measured and kept beside the first; the simulator just
never asks for one.

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
- Admission also refuses a request whose *whole* sequence would not fit, not just
  its first chunk (`--reserve-full-isl`, on by default, per-instance
  `reserve_full_isl`). Mirrors vLLM's `scheduler_reserve_full_isl`, `True` there
  too; checking only the first chunk lets chunked prefill over-admit
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
- KV capacity is `npu_mem.mem_size * npu_mem.mem_util - weight`, divided into `--block-size`
  blocks. vLLM also subtracts the activation peak and CUDA context, which are not modelled,
  so the simulator's capacity is an upper bound at the same utilization. That only matters
  when a run **saturates** the KV cache: below the ceiling nothing is preempted and the
  configured capacity is invisible in the results. **For a run that does saturate,
  calibrate `mem_util` so the block count matches `kv_cache.num_gpu_blocks` in the bench
  run's `meta.json` before comparing latency at all.** On the bundled RTX 4090 example
  (24 GB, pinned at its ceiling) the matched value is `0.833919`, which moves TTFT mean
  from -20.7% to +0.6% and TPOT mean from +12.9% to +0.2% against the same vLLM run. The
  RTXPRO6000 examples peak at 58-97% of budget on a 96 GB card and stay at `0.9` --
  calibrating them would change nothing, so do not treat `mem_util` as a general
  explanation for validation error
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

### Speculative decoding
`--num-speculative-tokens N` turns it on. Which draft tokens the target accepts
is the one thing a simulator cannot compute -- it needs the draft's and the
target's distributions over real tokens -- so acceptance is a **policy** chosen
the way MoE expert routing is, with the default taken from what the model's own
authors published (`configs/spec_decode.json`, one entry per model, each
carrying its source).

**The rate is `accepted / drafted`, and it is marginal**, so

    mean_accept_length = 1 + rate * N

That identity reproduces all nine published (rate, length) pairs to within 0.01
tokens. It is deliberately **not** Leviathan's per-position alpha (ICML 2023),
which is *conditional* -- position i is reached only if 1..i-1 were accepted --
and gives the capped geometric `(1 - a^(N+1))/(1 - a)`. Passing a published
rate to that formula under-predicts the published accept length by 25-30%,
because real acceptance is front-loaded rather than i.i.d. The check that
settles the reading: Qwen's published per-position decline of 95% at p1 to 60%
at p5 averages 0.775 read as marginals against a published 0.779, and 0.621
read as conditionals. **Don't reintroduce the capped-geometric formula.**

A model with no published figure gets no default -- rates run from 0.39 to 0.78
across the four modern families, so there is nothing defensible to guess.

Scheduling follows vLLM exactly, including its framing: `num_tokens_with_spec =
num_tokens + spec_tokens`, a request just catches up to it, and rejection rolls
back with `num_computed_tokens -= num_rejected`. Three details that are easy to
get wrong:

- **Roll back before caching the prefix.** A block holding a rejected token
  must never be hashed, or a later request hits on text the model never emitted
- **Classify a spec step by why it has many tokens, not by the count.**
  `num_new > 1` files a verification step as a prefill chunk; the `1 + N`
  queries of one sequence share that sequence's KV read, a prefill chunk of the
  same size does not
- **Clamp the overshoot.** A step commits `1 + accepted` at once and can run
  past `output`; vLLM stops at max_tokens and discards the excess, so the
  overshoot is not throughput

The verification forward needs the **fifth attention axis**, `decode_q_len`
(query tokens per decode sequence, `1 + N`). It is opt-in in the profiler
(`--attention-decode-q-lens`, default `1`) because it multiplies the grid. An
unprofiled value falls back to the nearest with a one-shot warning rather than
interpolating: query length changes the kernel's tile shape, not just its size,
and unlike the other four axes there is no measurement saying a value between
two profiled ones lies between their costs.

### Linear-attention state, prefix caching and the drafter
Three things a hybrid or speculative run costs that a per-token KV model does
not see. All three are vLLM's rules, and none is guesswork.

**State pages, not state bytes.** vLLM picks the attention block size so one
attention page covers one mamba page — `platforms/interface.py`:
`attn_block_size = alignment * cdiv(mamba_page_size, alignment *
attn_page_size_1_token)` — then sets `mamba_page_size_padded = attn_page_size`,
so a layer's whole recurrent state occupies exactly one page and the padding is
really allocated. Qwen3.8-27B: mamba page 3,207,168 B against an attention page
of 3,211,264 at `block_size 784`, which is how 784 gets chosen. How many pages
per layer is `MambaSpec.max_memory_usage_bytes`:

| cache mode | pages per mamba layer | when |
|---|---|---|
| `none` | `1 + N` | prefix caching off |
| `align` | `2 + N` | prefix caching on — **the default** |
| `all` | `cdiv(max_model_len, block_size)` | opt-in, not modelled |

`align` holds two because one page carries the state being written this step
and the other the last checkpoint at a block boundary, which is what a later
prefix hit resumes from. `N` is `num_speculative_tokens`. Charging one page per
layer understates a prefix-caching run by exactly 2x. Speculative decoding also
**widens the conv state itself** (`conv_kernel_size - 1 + N`), which is the
small half and moves ~2% at N=4.

**Chunk ends must be block-aligned under `align`.** State slot *p* holds the
state after exactly `(p + 1) * block_size` tokens and state is written at chunk
ends, so `Scheduler._mamba_block_aligned_split` floors a prefill chunk to a
block boundary (exempting the prompt's last chunk), stops a mid-block chunk at
the next boundary, and never runs past the last block-aligned position. It can
legitimately return **0** — vLLM's "insufficient budget for a block-aligned
chunk" — and that is not the scheduler deadlock the `num_new <= 0` guard
catches: the split only floors to zero when `block_size <= max_prefill_tokens`,
so a fresh step's budget does cover a block. With `block_size 784` and a 2048
budget a chunk is 784 or 1568, never 2048, so this changes batch composition on
every hybrid run.

**The drafter is not free.** vLLM runs it **N times per step** — once, then
`num_speculative_tokens - 1` more in `llm_base_proposer.py`'s loop — each a
decode-shaped forward at `max_query_len = 1`. A model that drafts with itself
runs an MTP module per pass: two norms, an `eh_proj`, one full decoder layer of
its own family (`DeepseekV2DecoderLayer`, `Glm4DecoderLayer`,
`MiniMaxM3DecoderLayer`, `Qwen3_5DecoderLayer`), then a norm, `lm_head` and the
sampler. The block is **full attention** whatever the target's layers are —
Qwen3.5's MTP passes `layer_type="full_attention"` explicitly — so a hybrid's
drafter carries no recurrent state, but it does carry a KV cache: `+1.6%`
bytes/token on DeepSeek-V3.2's one module, `+11.7%` on MiniMax-M3's seven,
`+6.2%` on Qwen3.8 (one more full-attention layer out of its 16).

Draft **time** is not charged yet, and `_require_drafter_cost` **refuses** a
speculative run on a model with MTP modules until its catalog has an `mtp:`
block, rather than reporting a speedup no engine can deliver. That block has to
come from a live profile dump like every other one. A model with no MTP modules
drafts externally (a second model, or n-gram); that is a serving choice rather
than a checkpoint property, so it warns instead of refusing.

### Dtypes come from the model config, never from an input
There is no `--dtype` and no `--kv-cache-dtype`, and no cluster-config
`dtype` / `kv_cache_dtype` either. A model carries **five** cache dtypes and
they are decided in four different places, so a flag per dtype is both
unusable and unfaithful — the checkpoint already says what it is, and saying
otherwise describes a model nobody can serve. `memory_model.cache_dtype_bytes`
holds the whole table; every rule below is vLLM's, verified against its source:

| Cache | Source | vLLM |
|-------|--------|------|
| weights | `quantization_config.quant_method`, then `torch_dtype` / `dtype` | on a quantized checkpoint the dtype fields are the *activation* dtype |
| KV cache | `quantization_config.kv_cache_scheme` / `kv_cache_quant_algo` → fp8 | `attention.py:281` promotes exactly this when the flag is `auto` |
| mamba conv state | `mamba_cache_dtype`, `auto` → weight dtype | `mamba_utils.py::_mamba_state_dtype` |
| mamba recurrent state | `mamba_ssm_dtype`, `auto` → **conv** dtype (not the weight dtype) | same, plus `models/config.py::Qwen3_5ForConditionalGenerationConfig` bridging the HF field |
| sparse-indexer side cache | fixed by the model | DeepSeek/GLM `torch.uint8`, M3 `resolve_indexer_kv_dtype("bf16")` — neither follows the KV dtype |

Note the weight row is the profiler's **variant folder name**, not vLLM's
`model_config.dtype`: vLLM calls DeepSeek-V3.2 bfloat16 and keeps fp8 in the
quant method, while the folder has to encode the quantization or two bundles
collide. That divergence is deliberate; the other four match vLLM exactly.

To simulate a different precision, **profile it** — the profiler's flags write
a separate bundle and the simulator reads the one the checkpoint names.

### CLI argument conventions
CLI flags follow vLLM naming where applicable:
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

### Tensor sizes, block weight and KV shape
`memory_model.calculate_sizes(model, layer_name, ...)` returns
`(input, weight, output)` bytes **per rank** for one canonical layer, and
**raises** on a name it does not know — so a catalog entry without a formula
here makes the model unsimulable, not merely mis-sized.

`get_weight()` walks the architecture yaml's blocks and the checkpoint's own
per-layer composition (`utils.get_architecture` / `utils.get_layer_stack`), one
built weight per distinct block shape. It used to sum a hardcoded
`layernorm + qkv_proj + o_proj + layernorm + mlp`, which cannot describe MLA
(no `qkv_proj`) or a hybrid stack. For PP it takes the **heaviest** contiguous
window of layers, since the first window is the light one on a stack whose
leading layers are dense.

`kv_bytes_per_token_per_layer()` is the one place that knows the KV shapes, and
they are not interchangeable:

| Shape | Per token per layer | TP |
|-------|---------------------|-----|
| GQA | `2 * kv_head * head_dim * kv_fp` (K and V) | sharded |
| MLA | `(kv_lora_rank + qk_rope_head_dim) * kv_fp`, one latent, no separate V | **replicated** (`num_kv_heads = 1`) |
| + sparse indexer | plus `index_head_dim + index_head_dim//128 * 4` bytes (fp8 keys + fp32 scales, uint8) | replicated |
| linear attention | 0 per token — the state is per **sequence** | n/a |

A per-sequence state is charged as blocks the request holds for its whole life
(`MemoryModel._state_blocks_per_request`), in a list separate from the token
blocks: the token list is positional, so a block backing no tokens must not
join it. 78.4 MB per sequence on Qwen3.8-27B, i.e. 75 blocks per request out of
37,173 on a 96 GB card — it bounds concurrency the way a KV cache bounds
context, and leaving it out lets the simulator admit requests vLLM could not.

Sizing DeepSeek-V3.2 as GQA read 1,748,992 bytes/token where MLA caches 78,324.

**Verify a new family's shapes against its published parameter count.** It is
the one number that catches a wrong shape anywhere in the stack, and it is
public. `.claude/check_param_count.py` sums `calculate_sizes`' weights over the
catalog and the resolved stack: DeepSeek-V3.2-Exp comes to 671.878B, and minus
the DSA indexer (0.852B) that is 671.026B — V3's published 671B, with the
difference being exactly what V3.2 adds.

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
- `dp_group`: DP group ID string (optional). Instances with the same string form one
  data-parallel group, wave-synchronized per iteration; for MoE they also share
  experts across the group. Works for dense models too (plain DP replicas)
- `npu_mem.mem_bw`: NPU memory bandwidth (also set as `local-mem-bw` in system.json)
- `npu_mem.mem_util`: fraction of `mem_size` usable for weights plus KV cache (optional,
  default from `--npu-memory-utilization`, itself `0.9`). KV capacity is
  `mem_size * mem_util - weight`. Sits inside `npu_mem` because its only job is to
  scale `mem_size`, and it follows that block's `mem_*` naming
- `cpu_mem.mem_bw`: CPU memory bandwidth (set as remote memory in memory_expansion.json)
- `link_bw`: inter-node bandwidth in GB/s (set in network.yml)
- `link_latency`: inter-node link latency in ns

Parallelism inference: users may provide partial info (e.g., `num_npus=4, tp_size=2`)
and `config_builder.py` infers the rest (`pp_size=2`). Validation ensures
`num_npus = tp_size * pp_size`, `pp_size <= num_hidden_layers` (stages are cut on
transformer-block boundaries, so a stage cannot be empty), and `ep_size` divides
`num_local_experts`. Members of a `dp_group` must agree on `tp_size`, `pp_size` and
`ep_size`. For a **MoE** model in a DP group `ep_size` is the total EP degree across
the group, so it must be divisible by `dp_group_size` and
`ep_size / dp_group_size <= tp_size`; for a **dense** model `ep_size` is 1 because
there are no experts to shard, so neither check applies.

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
For DP configurations the network topology is multi-dimensional, innermost dimension
first: `npus_count: [tp_size, dp_group_size]`, or `[tp_size, pp_size, dp_group_size]`
when `pp_size > 1`. This mirrors vLLM's rank layout
(`parallel_state.initialize_model_parallel`: `all_ranks.reshape(-1, dp, pp, pcp, tp)`);
the `pp_size` dimension is omitted when it is 1, so existing DP+TP configs keep their
2-D topology and their recorded baselines. Collectives are scoped to specific
dimensions via the `involved_dim` BoolList attribute on COMM_COLL_NODE protobuf nodes:
- ALLREDUCE (TP): the TP dim only — `[True, False]`, or `[True, False, False]` with PP
- EP: the DP dim, plus the TP dim when EP spans past one instance's GPUs —
  `[False, True]` / `[True, True]`, or `[False, False, True]` / `[True, False, True]`
  with PP. **Never** the PP dim: vLLM's EP group is
  `all_ranks.transpose(1, 2).reshape(-1, dp*pcp*tp)`, and that transpose pins the PP
  index, so experts are sharded across the DP x TP ranks of one pipeline stage.
  Marking PP involved would drag the other stages' NPUs into a collective they never
  join in vLLM

A DP round only completes if **every NPU of every member instance runs it**.
`Scheduler.add_done` enforces this by refusing to complete a batch until both
`start_npu` and the instance's last NPU appear in `batch.end`, and **nothing raises
when that cannot happen** — the batch simply never finishes and the group's collective
blocks forever. Any code path that creates or serves a batch the start NPU cannot
claim is therefore a silent deadlock; that was issue #65. `schedule()` lets the start
NPU fall through to `_schedule_existing` when the pipeline is full for exactly this
reason, and a DP batch is not servable to any NPU until the barrier has stamped its
`workload_name`.

The `involved_dim` is encoded in the trace `comm_type` field as `ALLTOALL:0,1` (parsed by
the Chakra converter's `_parse_comm_type`). ASTRA-Sim's `Workload::issue_comm()` reads this
and passes it to `generate_all_to_all()`, which skips dimensions where `involved_dim` is false.

The `system.json` collective implementations must have one entry per topology dimension
(e.g., `"all-to-all-implementation": ["ring", "ring"]` for 2D, three entries for 3D).
`config_builder.py` generates this automatically from the topology it emitted.

### Group-limited expert routing
DeepSeek-V3/V3.2 and GLM restrict a token's experts to `topk_group` of
`n_group` groups (`deepseek_v2.py` passes `num_expert_group=config.n_group`,
`topk_group=config.topk_group`, both defaulting to 1). `GateRouter` reads both
off the checkpoint. It matters because it changes how many EP ranks one token
reaches, and therefore the per-rank MoE token count and the ALLTOALL size:
DeepSeek-V3.2 at EP=8 sends a token to 45.4% of ranks against 65.6%
unrestricted. **Only DeepSeek-V3.2 actually restricts** — GLM-5 ships
`n_group: 1` and no other family declares the fields.

`_hit_probs` is exact and deterministic: a DP over which groups the token
selected, not sampling. It draws **without replacement**, matching
`torch.topk`; the older `1 - ((ep-1)/ep)**k` modelled independent draws and
read ~1% low even with no grouping. Don't "simplify" it back — the difference
is measurable, and the grouped and ungrouped cases must not have two different
answers to one question.

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
   `max(total_len) * hidden_size * fp` across the group. `dp_pending[dg][inst]` is a
   **FIFO**, not one slot: at `pp_size > 1` a member can have up to `pp_size` batches
   waiting, and a round pairs the members' *j*-th batches, mirroring vLLM, where DP
   rank A's *j*-th forward joins the same collective as rank B's *j*-th
2. **ASTRA-Sim ALLTOALL barrier**: all DP group instances' `.et` files are placed in a
   shared workload folder. The ALLTOALL collectives in both files have matching stream
   IDs, causing ASTRA-Sim to block until both NPUs reach the collective.

When one DP instance is idle (no requests), a dummy batch (1 decode token) is created
so it can participate in the sync. When one instance finishes all requests, it
continues generating dummy batches until all DP group members are done.

The dummy is gated on `len(inflight) < pp_size`, not `== 0`: vLLM requires every rank
of a DP group to run the same number of forwards, and with PP a rank has `pp_size`
microbatches in flight, so the gate is `schedule()`'s own pipeline-depth rule. Gating
on `== 0` lets the member holding a real request run `pp_size` batches ahead of a round
the idle members can never join. **Any** NPU of the instance may open a round (an empty
`dp_pending[dg][inst]` keeps it to one per member per round: nothing queued means the
member has not joined the round being assembled) — which NPU ASTRA-Sim asks about
is not ours to choose. Because of that the dummy is appended to
`schedulers[i].inflight` the way `_build_batch` registers a real batch, and the start
NPU must be able to join it (see the `involved_dim` section above).

A DP batch is the one place where the simulator cannot do what vLLM does and
schedule-then-dispatch in one step (`schedule()` then `execute_model()` in
`step_with_batch_queue`): its graph needs the group-wide padded `max_total_len`, so
dispatch waits for the barrier. The invariant that survives is that the dispatch still
lands **before the next `schedule()` for that NPU** — `dp_ready_workloads` is keyed by
the NPU that opened the round (`batch.fired[0]`) and is served ahead of `schedule()`.
Letting a new build take that poll instead is what hung `dp>1 x pp>1`: the NPU ran the
*second* microbatch's graph as its first iteration, `add_done` credited it to the first
by `id - 1`, and the other pipeline stage waited forever on a RECV that never came.

### Chakra graph converter
The Chakra converter (`astra-sim/extern/graph_frontend/chakra/src/converter/llm_converter.py`)
transforms text traces into protobuf `.et` files. It creates:
- `MEM_LOAD_NODE` for the first layer's input (from REMOTE/CPU memory)
- `COMP_NODE` for each computation layer
- `MEM_STORE_NODE` for the last layer's output (to REMOTE/CPU memory)
- `COMM_COLL_NODE` for ALLREDUCE/ALLTOALL (with optional `involved_dim` BoolList attribute)

The converter parses `comm_type` strings like `ALLTOALL:0,1` via `_parse_comm_type()`,
splitting into `comm_type="ALLTOALL"` and `involved_dim=[False, True]`.

The MEM_STORE node uses the **last layer's** `output_memory_loc` and
`output_memory_size`. This is why the sampler (not lm_head) must have
`output_loc=REMOTE:{node_id}`: what goes back to the host is the sampled token ids
(4 bytes per sequence), not the logits the sampler read.

Memory location types: `LOCAL` (NPU) = 1, `REMOTE` (CPU) = 2, `CXL` = 3, `STORAGE` = 4.
These must match the C++ enum in `astra-sim/astra-sim/system/AstraMemoryAPI.hh`.

### Docker environments
- **vLLM container** (used by `python -m profiler`, `python -m bench`, and
  `python -m workloads.generators`): `vllm/vllm-openai:v0.28.0` (or
  `v0.28.0-cu129` on a CUDA 12.9 host)
  - Launched via `scripts/docker-vllm.sh`. Set `VLLM_GPUS` to a docker
    device spec to keep it off GPUs someone else is using; the inner
    quotes are part of the value (`VLLM_GPUS='"device=2,3"'`), and
    without them docker reads the second field as a GPU count. Default
    is every GPU on the host
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

No unit-test suite. The simulator is deterministic, so validation is exact
equality against recorded results:

**Three things under `profiler/` are simulator inputs**, despite the path. The
trace generator reads each directly, so a change to any of them can move every
clock in `validate.sh`:

- **`profiler/models/*.yaml`** — the layer order. Merging two catalogs into one
  broke all 16 MoE scenarios exactly this way.
- **`profiler/core/stack.py`** — which block each decoder layer runs, resolved
  from the checkpoint's config.
- **`profiler/core/catalog_path.py`** — `model_type` → yaml resolution.

Both `.py` files are deliberately free of third-party imports so the simulator
container (no pydantic) can import them, and both exist as *one*
implementation because the two sides already drifted once. When deciding
whether a change can affect the simulator, the paths to check are
`serving/`, `configs/`, `bench/`, **`profiler/models/`**,
**`profiler/core/{stack,catalog_path}.py`** and `profiler/perf/`.

1. **`./serving/validate.sh`** — the whole check, ~8 min. Stage 1 compares every
   scenario against the `Total clocks (ns)` recorded in
   `serving/validate-baselines.txt`; stage 2 regenerates each `bench/examples`
   entry's `outputs/sim.csv` and `validation/summary.txt` and checks both md5s.
   Anything that moved is printed as a markdown table for the PR.
   `--clocks-only` skips stage 2, `--list` names the scenarios, `--update`
   rewrites the baselines, `--help` prints the rest.
   **Run it after every commit that touches `serving/`** — it is cheap enough,
   and it is how the 8-of-19 regression on the perf branch was caught.
2. For the *size* of an accuracy change, not just its presence:
   `./bench/examples/validate.sh` regenerates `validation/summary.txt` and the
   three plots. A changed `sim.csv` makes those stale, so regenerate and commit
   them in the same commit.
3. For profiler changes: edit `MODEL` / `HARDWARE` in `profiler/profile.sh`
   and run `./profiler/profile.sh` from the repo root inside the vLLM container.
4. For a catalog change (new or edited `profiler/models/*.yaml`), and after a
   vLLM upgrade: `python -m profiler coverage <model> --hardware <hw>` inside
   the vLLM container. It boots once, runs one forward per batch regime, and
   exits non-zero while any kernel is unbound. This is the only check that
   catches a catalog entry that names a real class and measures **nothing** —
   the profile tree holds only modules that launch a kernel of their own, and
   the module tree cannot tell you which those are. Every one of the four
   modern families had at least one such entry.

A scenario whose clock equals an existing one exercises flag parsing and
nothing else. Several knobs only bite once the KV cache is saturated, which is
what the `saturated_*` scenarios are for; `example_trace.jsonl` never gets
there, and its DP members always drain together, which is what the `*_uneven`
scenarios are for.

## Common Pitfalls

- **Don't reintroduce `sequence:`, or any second layer-order form.** There is
  one: `blocks:` + `shared:`. A uniform stack is the degenerate case, one entry
  per axis. `sequence:` was that flattened, and two forms meant two code paths —
  the simulator only implemented the flat one, so Qwen3.5 and MiniMax-M3 could
  not be simulated at all. It also encoded a false claim: with the axis in the
  key there is no per-layer question left, and the MLP was resolved once per
  *model*, modelling DeepSeek/GLM's first three dense layers as MoE
- **Don't resolve a per-layer property once per model.** Which block a layer
  runs comes from the checkpoint via `profiler/core/stack.py`, per layer. The
  tell for this class of bug is a name like `is_moe` on the context object
- **Don't reuse a built transformer block across layers without keying on the
  block shape.** `_block_copy_key` returns the layers' `LayerSpec`s, and the
  replay is what keeps trace generation O(1) in depth. Getting the key wrong is
  invisible on a uniform model: an earlier version emitted **one** block
  instead of `num_hidden_layers` whenever block copy was disabled, understating
  the clock 3.1x on 48 layers, and the recorded baseline enshrined it
- **Don't add a layer name to a catalog without a `calculate_sizes` formula.**
  `memory_model.calculate_sizes` **raises** on an unknown name, so the model
  becomes unsimulable — which is the state Qwen3.5, DeepSeek/GLM and MiniMax-M3
  are in today
- **Don't edit `astra-sim/`** unless the change targets simulator integration
  (e.g., `llm_converter.py`, `Workload.cc`, input configs). Chakra is *installed*
  into the container's site-packages by `scripts/compile.sh`, so editing
  `llm_converter.py` changes nothing until you reinstall it
  (`cd astra-sim/extern/graph_frontend/chakra && pip3 install .`)
- **Don't commit large files**: generated traces, `.et` files and scratch run
  output are gitignored (`outputs/*` with `!outputs/example_*.csv`,
  `bench/results/`). `astra-sim/inputs/runs/` is cleaned per run unless you
  pass `--keep-inputs` (or `--save-trace-text`, which implies it), either of
  which can leave gigabytes behind
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
- **Don't split pipeline stages by trace-line count.** `pp_stage_boundaries` in the
  trace header exists because a stage may only be cut on a transformer-block
  boundary: that is the one place where the upstream `output_size` and the
  downstream `input_size` are the same tensor (the hidden state). Inside a block
  they differ — `qkv_proj` emits Q+K+V while `rotary_emb` declares only Q+K — and
  ASTRA-Sim's analytical backend keys its send/recv tracker on
  `(tag, src, dst, chunk_size, chunk_id)`, so a mismatch silently deadlocks the
  receiving NPU instead of raising. That was issue #55: only the `pp_size` values
  whose line-count cuts happened to land between two size-agreeing layers ran
- **Don't gate a DP batch on which NPU is asking.** `sys == inst2npu_mapping[i]` does
  **not** mean "built this batch": any NPU of an instance may open a DP round, so the
  start NPU can arrive holding a batch it joined through `_schedule_existing`. Route on
  whether the batch is new (`len(batch.fired) == 1`) instead, or the round is
  registered twice and the graph regenerated. And never hand an NPU a DP batch whose
  `workload_name` is still `None` — that derives the solo `instance<id>_batch<id>`
  folder, which is never written for a DP batch, and ASTRA-Sim logs
  `[critical] workload file ... does not exist` and then **hangs instead of exiting**.
  Both were issue #65; `add_done` needs `start_npu in batch.end` to complete a batch,
  so any batch the start NPU cannot claim deadlocks silently
- **Don't give a DP group one slot per member anywhere.** `dp_pending[dg][inst]` and
  `dp_ready_workloads[npu]` are both FIFOs because at `pp_size > 1` a member has up to
  `pp_size` batches outstanding. Three separate hangs came from single slots: the second
  registration dropped the first batch from the barrier (it never got a `workload_name`,
  so the instance's other NPUs retried joining it forever), and the second round
  overwrote an unconsumed workload (the NPU then ran the *wrong* microbatch's graph). All
  three are invisible at `pp_size == 1`, where an instance holds one batch at a time
- **Don't let the start NPU's join depend on the pipeline being full.** `schedule()` tries
  `_schedule_existing` *before* the `len(inflight) >= pp_size` cap. A dummy opened by a
  non-start NPU leaves `pp_size - 1` slots free, and gating the join on a full pipeline
  sent the start NPU down the build path, where an idle member has nothing to build — so
  it passed forever and `add_done` never saw `start_npu in batch.end`
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
