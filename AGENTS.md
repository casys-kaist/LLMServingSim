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
│   │   ├── spec_decode.py      # Speculative-decoding acceptance model
│   │   ├── run_paths.py        # Per-run ASTRA-Sim input/output path layout
│   │   ├── logger.py           # Rich-based logger + stdio capture
│   │   └── utils.py            # Model config loading, formatting
│   ├── run.sh                  # One runnable example per feature (a menu, not a suite)
│   ├── validate.sh             # every scenario vs recorded clocks + bench/examples digests
│   └── validate-baselines.txt  # the recorded values; refresh with validate.sh --update
├── configs/
│   ├── cluster/                # Cluster topology configs (hardware, memory, instances)
│   ├── model/                  # Native-class-compatible HF model configs
│   └── pim/                    # PIM device configs (DRAMSim3 INI format)
├── workloads/                   # Request trace datasets (.jsonl)
│   └── generators/             # ShareGPT/etc → JSONL workload generators
├── profiler/                   # vLLM-based layerwise profiler (`python -m profiler`)
│   ├── __main__.py             # CLI dispatch (profile / slice / coverage)
│   ├── core/                   # internals
│   │   ├── runner.py           # Orchestration (spin_up → categories → spin_down)
│   │   ├── config.py           # Architecture / ProfileArgs / engine defaults
│   │   ├── engine.py           # vLLM lifecycle (tmpdir-based local config load)
│   │   ├── categories.py       # Dense / PerSequence / Attention / LinearAttention / MTP
│   │   ├── skew.py             # Heterogeneous-decode skew sweep
│   │   ├── skew_calibration.py # Offline reference-aligned fit + supported-N lookup
│   │   ├── skew_plan.py        # Dynamic heterogeneous distribution coverage
│   │   ├── writer.py           # CSV + meta.yaml writer, TP-stable replication
│   │   ├── bundle_metadata.py  # Locked variant-root metadata updates
│   │   ├── metadata_names.py   # Shared descriptive metadata names
│   │   ├── execution_metadata.py # Import/match observed serving graph policies
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
│   │   ├── kv_cache_metadata.py # qualified per-engine startup READY capacities
│   │   ├── execution_metadata.py # Read-only completed worker graph observations
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
├── scripts/                    # Shared environment, build and resource-safety entry points
│   ├── docker-vllm.sh          # vLLM container (profiler + bench)
│   ├── docker-sim.sh           # simulator container
│   ├── install-vllm.sh         # bare-metal vLLM install (uv venv)
│   ├── monitor_run.py          # bounded child-process execution + resource telemetry
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
The profiler uses lightweight PyTorch/Kineto capture via a worker extension class.
Explicit dynamic module scopes own native CUDA activities through CPU launch
correlations; CPU duration is never a latency input. No model-specific collector
branches, Python stacks, tensor shapes or memory snapshots are required. Architecture is
dispatched by the HF config's `model_type` field against YAML catalogs under
`profiler/models/<model_type>.yaml`, which bind canonical layer names (dense /
per-sequence / attention / linear-attention / moe) to vLLM class/kernel selectors
or one exact native CPU operator name. Operator CPU intervals establish launch
ownership only; their CUDA activities supply all measured latency.

**Catalog naming rule.** The file is named after the `model_type` it primarily
serves, spelled **verbatim** — including where vendors disagree: DeepSeek
writes V3.2 as `deepseek_v32`, Qwen writes 3.5 as `qwen3_5`, so the files are
`deepseek_v32.yaml` and `qwen3_5.yaml`. Copying upstream's spelling is what
keeps the rule mechanical; normalising it would mean the filename matches no
`model_type` and every lookup falls through to the directory scan. `model_type`
is read from the top level of the config handed to the profiler. Preserve the
shape that the checkpoint's config class consumes: Qwen's text tower is
flattened to `qwen3_5_text`, while MiniMax-M3 must retain its nested
`text_config`. Do not apply one family's flattening rule to another.
When several `model_type` values share a compatible implementation, one file
lists them under `model_types:` rather than duplicating or symlinking the catalog.
The recorded vLLM 0.28 DeepSeek-V3.2/GLM-5 decomposition shares `deepseek_v2`;
that is not a version-independent runtime claim. vLLM 0.31's native DeepSeek-V3.2
path is `vllm.models.deepseek_v32.nvidia.model`. Existing aliases and stored tables
do not certify acquisition on that path: require matching live coverage and
execution contracts before a new sweep. Two files claiming one `model_type` is
an error, not first-wins, including a matching filename alongside another file's
alias. Check all claims before choosing a catalog; a file's own alias is not
a second claimant.

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
opposite directions between vendors. The profiler automatically resolves the
smallest prefix that instantiates every block type required by each category —
4 for a Qwen3.8-27B category needing every attention type. Leave
`--num-hidden-layers` unset to use that resolution; an explicit lower count can
omit required block types.

**How finely to divide is mostly not a choice.** An entry exists iff the
profile tree holds a node carrying CUDA time that no other entry claims, and
the collector retains invoked modules and their descendant CUDA activity — so
vLLM's implementation sets the count, and the same operation needs seven
entries in one family and one in another. DeepSeek-V3.2's `Indexer` is an
`nn.Module` container (`self.wq_b = ReplicatedLinear`,
`self.wk_weights_proj = MergedColumnParallelLinear`, `self.k_norm = LayerNorm`,
plus `view`/`cat` in its `forward`), so it decomposes into 7; MiniMax-M3's
`MiniMaxM3IndexerTritonImpl` is a `forward` and nothing else, with its index
projections folded into `MinimaxM3QKVParallelLinearWithIndexer`, so it is 1.
That is also why `*_glue` exists only where the forward does eager tensor work
between module calls — Llama, Mixtral and Qwen3 blocks are pure module chains.

What *is* a choice is which level to bind, since a parent node's time already
covers its subtree. Entries may merge only when all three hold: **one scaling
axis** (categories cannot mix — `Indexer` cannot be one node because its
projections scale with tokens and its scoring with kv, and binding the parent
forces the 8,643-shot attention grid on work that `dense` covers in 152);
**nothing pinned** (`o_proj`/`down_proj` carry the TP all-reduce after them,
`attention`/`sparse_attention` have PIM swapped in front); **no `tp_stable`
mixing** (that flag buys one sweep at TP=1 plus replication). `moe` is the rule
in the other direction: one entry for the whole block, because everything in it
scales with tokens.

**Finer than the rule requires is allowed and usually right.** DeepSeek's 7
indexer entries could be 3. They are not merged because a per-entry number can
be checked against physics and a composite cannot — `lm_head` was caught at 9x
against `vocab * hidden * bytes / mem_bw`. Coarsen when trace width actually
costs, and coarsen the **trace** rather than the catalog: measurement fidelity
and graph size need not share a granularity.

**Write a catalog from a live profile dump, not from vLLM's source.** The
module tree and the profile tree differ both ways: `rotary_emb`, `q_norm`,
`k_norm` and `RMSNormGated` are real modules that never become profile nodes
(binding them silently measures nothing), while raw CUDA kernels that are not
modules *can* be bound by name (`vllm: _causal_conv1d_fwd_kernel`) and are the
only way to reach gated DeltaNet's conv and decode recurrence. Kernel identity
also changes with the batch regime — a GDN block runs one set for pure prefill,
another for pure decode, a third for a mixed batch — so a catalog written from
a single shot binds the wrong kernel for the others.

Kernel wildcards must not reclaim a bound module's descendants. DeepSeek/GLM
`indexer_glue` excludes `LayerNorm` and `SparseAttnIndexer`, whose copy/fill
kernels are already included in `indexer_k_norm` and `indexer`. A coverage
report with no gaps does not establish non-overlap: coverage counts a bound
parent once, while latency extraction continues into its children. Audit
nested bindings on the live tree, including across categories; a dense-only
slice still must not claim work owned by an attention entry. Refresh old
`indexer_glue` rows through dense acquisition after changing this guard.
Phi-MoE's `block_glue` likewise excludes `PhiMoE`, `LayerNorm` and
`PhiMoEAttention`; only the decoder's direct residual work belongs there.
Acquisition rejects shared CUDA activity across canonical bindings. Old Phi
glue measurements require a dense-category refresh, not a metadata restamp.

For `deepseek_v32` and `glm_moe_dsa`, normalize the native `layer_types` cache
label `deepseek_sparse_attention` to the catalog's full-attention block axis.
Sparse indexer selection remains a separate axis; this normalization changes
neither native execution nor the stored source config, and admits no new backend.

Every TP degree is profiled on a **single GPU**: the engine is always booted with
`tensor_parallel_size=1`, and per-rank shapes are emulated by dividing `SHARD_FIELDS`
by TP via `hf_overrides`. Collective timings are left to ASTRA-Sim.
Include separately declared `dense_intermediate_size` in this sharding step;
do not leave a dense branch at full width while reducing the routed branch.
Resolve the unsharded local configuration through installed vLLM first, after
user overrides and decoder-prefix materialization. `hooks/tp_shape.py` preserves
the native head width when reducing the head count would otherwise change it;
validate both the sharded candidate and any scalar-width pin by materializing
and reloading the raw config through vLLM, not by updating an already resolved
config object. A nested backbone can derive `head_dim` without declaring it in
JSON; write an admitted pin into that backbone and consume the forwarded key.
Unsupported special-width changes fail rather than guessing. Nested text
fields and user shape overrides feed the same sharding step. Strict dimension
divisibility remains; this does not add replicated-KV-head TP emulation.
Incorrectly shaped timings require fresh acquisition, including affected dense
projections, attention and skew; use a separate output root or explicit category
replacement. Source edits alone do not invalidate ordinary/skew resume identities.
Consume nested backbone-only overrides after materializing them in the local
config; do not reapply them to wrapper properties that may be read-only.
Keep independently declared root fields, including shared root/text token IDs,
forwarded. Validate the final native config, not only the materialized dict.
Recorded engine kwargs are not a complete resolved model configuration.
`probe_limits` snapshots the native text config without another RPC; persist it
as `engine_resolved.per_tp[tp].model_text_config` only for authoritative main
engines. Preserve other TP entries and MTP/short-stack refresh semantics. This
text snapshot does not certify the wrapper, backend or parameter layout, and
missing historical snapshots must not be reconstructed.
Ordinary and skew acquisition metadata record resolved geometry and preparation
semantics. Remeasure wrongly shaped rows, never restamp them as corrected.
TP-stable replication applies to every catalog, not only coordinated layouts.
Preserve copied rows and their source identities during local acquisition;
exclude them from measured-shot resume coverage and from local replacement.
Layout discovery probes remain subject to their own acquisition contract.

**The stack is shrunk per category, not per model.** Instrumentation, warmup
and forward work scale with executed operations. Runtime estimates must come
from a bounded acquisition on the intended model/backend/grid, not historical
Python-call profiler timings.

A category must not pay for an axis it does not measure. The profile tree
merges same-class siblings, so a second layer of a type already present adds
**no information**, only its op count on every shot. `Category.stack_axes`
declares the axes (`attention` / `linear_attention` → `(attn, sparse)`,
everything else → all), `stack.minimal_layer_count_for` answers per axis set,
and `run_full` boots one engine per distinct depth — deepest first, since that
engine's shapes are what `meta.yaml` and `--attention-max-kv` must describe.

Where it lands, from each checkpoint's own config:

| model | all axes | attention only |
|---|---|---|
| DeepSeek-V3.2, GLM-5 | 4 (`first_k_dense_replace 3`) | **1** — every layer has the same attention |
| Qwen3.8-27B | 4 | 4 — gated DeltaNet three times, then full attention |
| MiniMax-M3 | 4 | 4 — non-sparse three times, then sparse |
| Llama, Qwen3, Mixtral, PhiMoE | 1 | 1 |

Only DeepSeek and GLM-5 gain, and only on `attention` — which is the sweep
that dominates a run. **1,057 → 341 ms per shot, 3.1x measured.** Splitting a
category further does not help: cost is per forward and a forward runs the
whole stack, so measuring `dense`'s MLP entries in a separate 1-layer engine
would mean two sweeps rather than one, even though Qwen3.8's MLP axis is
uniform.

Two things to know.

Fewer layers means a larger `num_cache_tokens`, which the feasibility filters
read, so **more shots pass**. Coverage widens and the grid is not row-for-row
comparable with a deeper run's.

And **axes are a conservative proxy** for what a category needs, which is
really "every block its entries live in". The proxy never over-shrinks, so no
data is lost, but it does under-shrink: DeepSeek-V3.2's `dense` entries sit
only in `attn.full_attention` and `mlp.dense` — `moe` is its own category — so
layer 0 alone would serve, while the all-axes answer is 4 because the MLP turns
to MoE at layer 3. Qwen3.8's `linear_attention` is the same story. The slack is
deliberate: `attention` is 8,643 shots against `dense`'s 152,
`linear_attention`'s 78 and `per_sequence`'s 40, so the axis rule already
captures every minute that matters and an entry-to-block resolver would save
three or four.

The reason a category cannot simply be shrunk to one layer is not that its
entries are hard to separate — it is that **at one layer some of them do not
exist**. A 1-layer Qwen3.8 instantiates only the gated-DeltaNet block, so
`qkv_proj` and `o_proj` are never built, their rows are missing from
`dense.csv`, and the simulator charges those layers zero.

**Every field vLLM shards has to be in that list**, or one part of the model is
measured per-rank while the rest is measured at full size, and a mixed
measurement is worse than either. `linear_num_key_heads` /
`linear_num_value_heads` are there because
`MambaStateShapeCalculator.gated_delta_net_state_shape` divides both `conv_dim`
and `num_v_heads` by the TP world size; without them, Qwen3.8-27B at TP=2 had a
halved attention page against a full-size mamba page and the engine resolved
`block_size` 1568 instead of 784 — a rank that exists in no deployment. The
tell is the per-TP `block_size` the run logs: on a hybrid it should be the
*same* at every degree, since both pages scale together. The model's full `config.json` (read from
`configs/model/<org>/<name>.json`, or auto-fetched from the HF Hub on first run)
is written to a tmpdir at spin-up so vLLM never needs Hub access.
When reducing `num_hidden_layers`, retain the first N entries of an explicit
`layer_types` list, in flat and nested text configurations. Validate that N is a
positive integer within that list; retain the original model config unchanged.
If `hf_overrides` explicitly supplies `layer_types`, slice that override too:
vLLM applies it again after reading the materialized config.

Attribution: the original layerwise-profile methodology (worker-extension hooks
and single-GPU TP emulation via `hf_overrides`) is
adapted from [@waneon](https://github.com/waneon). The unified 4D attention
sweep, the heterogeneous-decode skew sweep in `profiler/core/skew.py`, and
the offline reference-aligned skew calibration in `profiler/core/skew_calibration.py` are
developed in this repo.

Acquisition writes the selected categories into a bundle. Historical whole-block
MoE tables may remain alongside new native component measurements:

```text
perf/<hw>/
├── hardware.yaml                 # separately acquired device/interconnect facts
└── <model>/<variant>/
    ├── meta.yaml                 # shared and per-TP acquisition records
    └── tp<N>/
        ├── dense.csv             # layer, tokens, time_us
        ├── per_sequence.csv      # layer, sequences, time_us
        ├── attention.csv         # layer, prefill_chunk, prefill_key, n_decode, kv_decode, decode_q_len, time_us
        ├── linear_attention.csv  # layer, prefill_tokens, n_decode, time_us (hybrid only)
        ├── mtp.csv               # layer, sequences, time_us (opt-in drafter pass)
        ├── moe.csv               # historical whole-block fallback; no new acquisition
        ├── skew.csv              # raw heterogeneous-decode measurements
        ├── skew_fit.csv          # derived per-bucket alpha table
        └── moe_components/       # per-deployment components.csv and optional raw checkpoints
```

**A refresh rewrites only what it measured.** `meta.yaml` describes more than
any one `slice` covers, so `persist_meta` takes two authority flags and carries
everything else over from the file:

- `engine_effective` / `engine_resolved` describe the **deepest main engine** —
  the one whose shapes the simulator will run, which is why `run_full` probes
  limits there. A `--profile-mtp` refresh boots a different model (one extra
  full-attention layer, a conv state widened by `num_speculative_tokens`), and
  a category slice may boot a shallower stack. MTP and shallower slices are
  not authoritative; an ordinary slice whose required depth equals the
  all-axis depth updates the selected TP's main-engine metadata.
  Letting the MTP pass write them put `block_size: 800` and half the KV cache
  into Qwen3.8-27B's bundle where the main engine resolves **784** and
  19,839,182 — and `_resolve_block_size` reads that block size whenever
  `--block-size` is omitted, which is the documented way to get the profiled
  value. Three of the four modern bundles had it.
- `attention_grid` may only be rewritten by a run that swept attention. It
  records **`decode_q_lens`** explicitly: row counts do not establish requested
  query-length coverage, and the once-only `q=1` pure-prefill rows do not imply
  that ordinary `q=1` decode was requested or measured.

An empty `limits` argument now **preserves** `engine_resolved` rather than
erasing it, which is how the MTP refresh could delete what the main engine
resolved.

`attention.csv`, `skew.csv` and `skew_fit.csv` are keyed by **`layer`** as well:
a sparse-attention model has two or three kernels in that category and they
share neither a latency curve nor an alpha (on MiniMax-M3 the same batch fits
0.24 / 0.74 / -0.01 for `attention` / `indexer` / `sparse_attention`). A bundle
profiled before those columns existed holds one kernel and it is `attention`.

`<variant>` retains checkpoint-derived naming and separates explicit activation/KV
overrides (e.g. `bf16`, `fp8-actfp16`, `fp8-kvfp8`) unless `--variant` is set.
Ambiguous precision or nested HF overrides require an intentional label; names
are not complete resolved-precision contracts. Times are in **microseconds**. Layers marked
`tp_stable: true` in the yaml (layernorms, sampler) are profiled once at TP=1.
The writer copies stable `dense` and `per_sequence` rows into other `tp<N>/`
folders; stable MTP rows stay at TP1 and are looked up there by the simulator.

**`tp_stable` is about the tensor a layer walks, not the weight it owns.** The
test is whether the *work* is TP-invariant, and a norm's weight can be
replicated while its input is sharded. Qwen3's `qk_norm` was declared stable on
the strength of `RMSNorm(head_dim)` -- head_dim does not depend on TP -- but
`qwen3.py` applies q_norm/k_norm to `q`/`k` of size `num_heads * head_dim`
where `self.num_heads = self.total_num_heads // tp_size`, i.e. this rank's
heads. So the work halves at TP=2 while the sibling `layernorm`, which
normalises the replicated hidden state, does not. Declaring it stable charged
the TP=1 figure at every degree: a TP=2 re-measure reads **0.52-0.63x** of it
(45.0 us against 87.0 at 2048 tokens on Qwen3-32B), worth +0.6 pp of a mixed
step.

The audit that settles it for a new catalog is one question per entry: what
does the kernel read? Every other `tp_stable` entry in the shipped catalogs
passes -- `layernorm` / `final_layernorm` / MTP norms walk the residual hidden
state (replicated); DeepSeek's `mla_a_layernorm` walks the q/kv latent behind a
**`ReplicatedLinear`**; its `indexer_k_norm` walks a `head_dim` slice of a
`MergedColumnParallelLinear` that passes **`disable_tp=True`**; and `sampler`
runs on logits that `LogitsProcessor._gather_logits` has already all-gathered
to full vocab. Qwen3.5's and MiniMax-M3's q/k norms are not bound at all (their
kernels are fused into `fusedMiniMaxM3QNormRopeKVInsertKernel` and friends), so
the question does not arise there.

The measurement that catches it needs a **control**: re-measure a genuinely
stable layer at the same degree in the same run. `layernorm` re-swept at TP=2
reads 1.00-1.03 while `qk_norm` reads 0.52 -- without the control, a 0.52 could
be the machine rather than the flag.

The profiler Docker targets **vLLM v0.31.0 / CUDA 13**
(`vllm/vllm-openai:v0.31.0`). Changing the environment pin does not certify
every catalog/backend or remeasure the historical bundles. Every symbol under
`profiler/core/hooks/` is a vLLM *internal* API and is version-specific. v0.28
restructured MoE substantially: the old `FusedMoE` module is gone, replaced by
`FusedMoEFactory` returning a `MoERunner` that owns a `router` (`BaseRouter`)
and a `RoutedExperts`.

Whole-block forced-routing acquisition has been removed. New native component
acquisition retains the original global expert count and top-k. Gate/routing
executes natively; expert-only stimuli vary local active experts and assignments
with balanced local histograms and uniform weights. These are independent
measurement boundaries, not trained-gate numerical equivalence. Historical
whole-block CSVs remain readable and must never be relabelled as components.

### Vocab-parallel target endpoints

Two all-reduces per dense decoder block are not the complete TP forward.
For the ordinary vLLM 0.28 target path, `VocabParallelEmbedding.forward`
reduces the local-vocabulary embedding output, and
`LogitsProcessor._get_logits` gathers vocabulary-sharded logits before
sampling. `_shared_tp_collective` emits these once, gated by the catalog
binding and shared prologue/head placement; it does not infer them for MTP
or alternative heads. The formulas and scope are documented in
`docs/docs/simulator/parallelism-mechanics.md`.

Vocabulary padding precedes TP division. The embedding and head weights
remain per-rank, logits use the configured head dtype, and the sampler reads
the gathered full vocabulary. Per-sequence tensor sizes use the same head
row count as their latency lookup, not all scheduled prompt tokens.
This endpoint contract does not settle idle-DP padding, speculative heads
or expert dispatch/combine; those require their own audits.

### Profiling the drafter (MTP)
A model that drafts with itself keeps its MTP module outside the ordinary
model, and it **cannot be loaded standalone**: the MTP config's `model_type`
(`deepseek_mtp` / `qwen3_5_mtp` / `minimax_m3_mtp`) is produced by
`SpeculativeConfig.hf_config_override` and HF Transformers does not know it.
`--profile-mtp` boots with `speculative_config` so vLLM builds the drafter
alongside the target. It takes **no draft count**: the engine is pinned to
`num_speculative_tokens=1`, so `mtp.csv` holds **one** drafter pass — the unit
the simulator multiplies by its own `--num-speculative-tokens`. Booting at N
recorded N passes and the simulator multiplied again, so the cost came out N²;
that is why the flag is not a number.

The kernels then arrive without new tooling, because the drafter runs inside
`sample_tokens()` (`propose_draft_token_ids` → `drafter.propose`) and the fire
path already calls `execute_model` then `sample_tokens(None)` inside one
native CUDA-activity context. **`profiler coverage --profile-mtp` is how the
`mtp:` sections were written** — it names the unbound nodes with their ancestor
paths, so the profile tree is the source, as for every other block. For the
opposite failure — an entry that binds *too much* — coverage checks overlapping
and structurally unrelated claims. Inspect the profile tree and ancestor paths
as well: unexecuted alternatives and structurally plausible but semantically
wrong bindings still require an audit.

The `mtp` category has **one** axis, the pass's token count: the loop's passes
are decode-shaped at `max_query_len = 1` (`llm_base_proposer.py`), where tokens
and sequences are the same number. Shot counts depend on sequence/token limits,
grid factors, target drafter graph anchors and cache feasibility. Use a bounded
sweep to estimate acquisition time for the chosen model and device.

Four things a catalog here must get right:

- **A guard on every entry.** The drafter's modules are the *same classes* as
  the target's (`RMSNorm`, `ColumnParallelLinear`, `Linear`,
  `DeepseekV2DecoderLayer` / `Qwen3_5DecoderLayer` /
  `MiniMaxM3DecoderLayer`), so an unguarded entry claims the target's nodes
  too. DeepSeek and M3 pin `within` to their predictor classes
  (`DeepSeekMultiTokenPredictorLayer`, `MiniMaxM3MultiTokenPredictorLayer`);
  Historical Qwen tables used `not_within:
  [Qwen3_5DecoderLayer, Qwen3_5ForCausalLM]` because the old Python-call tree
  omitted their kernel-less wrapper. The scoped collector retains an invoked
  `Qwen3_5MultiTokenPredictor` even without a direct kernel; methods bypassing
  `Module.__call__` still need explicit ownership. Recheck every MTP binding
  against the current runtime, rather than assuming historical top-level
  ancestry. Getting this wrong is
  invisible in the CSV: the curve stays smooth and monotone. Qwen's `mtp_norms`
  read **1287 us at one sequence** for two RMSNorms before the guard was right.
- **The decoder block is declared, not listed.** `mtp.prologue` / `mtp.head`
  hold only the wrapper; the block itself is the target's own, replayed, and
  `mtp.decoder_block` says *which* one. **`mtp.head` must name something**:
  the Chakra converter reads the trace's MEM_STORE node from the last entry's
  attributes, and a drafter whose block ends in MoE closes with an
  `EXPERT END` marker — which surfaced as `'Layer' object has no attribute
  'output_memory_loc'` from inside the converter on MiniMax-M3. Every family
  has one to name: the wrapper's norms are a **single merged profile node
  spanning both sides of the block** (Qwen's `pre_fc_norm_*` before `fc` plus
  `norm` after; M3's `enorm`/`hnorm` before `eh_proj` plus `final_layernorm`
  after), and they are same-class siblings so the profiler cannot separate
  them — charging the merged node once after the block is exactly as accurate
  as once before. It cannot be read off the checkpoint:
  vLLM's MTP modules force it per family (DeepSeek/GLM build at layer index
  `num_hidden_layers`, Qwen passes `layer_type="full_attention"`, M3 passes
  `force_sparse_attn=True, force_moe=True`), and the resolved stack has exactly
  `num_hidden_layers` entries so any index into it wraps to layer 0 — dense for
  DeepSeek/GLM, non-sparse for M3, linear attention for Qwen3.8. An axis left
  unset is inherited from the stack, and only when the stack agrees on it.
- **The block is the dominant term.** On Qwen3.8-27B at 4 layers, one drafter
  pass measures 597 us for its `Qwen3_5DecoderLayer` against ~136 us for the
  whole wrapper. Emitting only the wrapper read a pass at a fifth of its cost.
- **The families do not share names or TP behaviour.** All three combine
  projections are 2h→h and all dominate the wrapper, but DeepSeek/GLM use a
  plain `nn.Linear` (unsharded), Qwen a `ColumnParallelLinear` (TP-sharded), M3
  a `ReplicatedLinear` (replicated).

### Historical Python-call normalization trap (vLLM 0.28)
The following describes the older acquisition tree. The scoped collector builds
its own grouped tree but deliberately retains the catalog's normalization and
full parameter-signature identity. It does not reproduce the old tree's pruning.

Worth knowing well beyond MTP: it made **every latency in a 0.28-profiled
bundle 3x too large**, not just the top-level ones. Every profile node is
normally divided by **its parent's** invocation count to get a per-call figure, and the
top level has no parent node to read that from — so the whole subtree under it
inherits the error. Two things go wrong at once under 0.28:

- vLLM merges same-class siblings into one node (time summed, `invocations`
  counting them all) and reports it **once** under a parent that is itself a
  node — but once **per sibling module** at the top level, where the owning
  module launched no kernel and was flattened away. Qwen's three drafter norms
  arrive as three identical top-level entries, and `LogitsProcessor` arrives
  once per timed forward. `extract_samples` now drops repeats identical in
  (full module representation, time, invocations). Do not reduce the identity
  to a class name: differently shaped modules can have equal times and counts.
- The root's own invocation count is the **forward count**, so
  `extract_samples` takes `iterations` and uses it as the top level's
  `parent_invocations`. It used to hardcode 1.

Whole-block MoE is a distinct normalization contract. `_entry_dict` marks the
MoE category with `normalization: invocations`; `extract_samples` divides by
the matched MoE node's calls, not the merged decoder parent's calls. A hybrid
stack's parent includes dense layers that do not invoke MoE. Keep the parent
times occurrence denominator for other categories, including intentionally
merged projection pairs. Remeasure affected stored `moe.csv` grids rather
than applying a model-specific multiplier. Native DP+EP components have their
own per-forward attribution and are not changed by this fix.

Two controlled experiments, same model / hardware / flags, against the trusted
`vllm=0.19.0` bundle for Llama-3.1-8B on RTXPRO6000:

| | value | vs trusted |
|---|---|---|
| `lm_head@1`, before | 6416.67 us | 8.99x |
| `lm_head@1`, dedup only | 2139.95 us | 3.00x |
| `lm_head@1`, both fixes | **713.805 us** | **1.00x** (trusted: 714.006) |
| whole `dense.csv`, both fixes | 1368 rows | mean abs delta 1.5-6.5% per layer, i.e. noise |

The physics check that made it findable at all: `lm_head` reads the whole
output embedding, so `vocab * hidden * bytes / mem_bw` = 583 us is a hard floor
and 6417 us is suspiciously far above that floor. A lower bound alone does
not make a slower measurement impossible; the controlled 9x discrepancy
and upstream normalization explain the error. **A profiled curve gives no other signal** — it
stayed smooth and monotone the whole way, and `profiler coverage` passed,
because coverage reports only what is *un*bound.

**Every category** of all four `vllm=0.28.0` bundles was affected, at a factor
of exactly `measurement_iterations`, and every one was re-profiled. Five
categories confirm the factor independently — `dense` p50 2.9995 (2584 rows),
`moe` 2.996 / 2.999, `linear_attention` 3.01-3.12 across all six GDN kernels,
`per_sequence` by the controlled experiment above, and `attention` by the
KV-bandwidth check (28.5% of spec against 84-95% for every 0.19 bundle). The
±5% spread around 3.000 is the noise between two independent measurement runs,
not structure.

The four `vllm=0.19.0` bundles were never affected, which is what made them
usable as the reference: the fix was verified against them before anything was
rewritten. RTXPRO6000 acquisition versions now vary by model: Llama-3.1-8B,
Qwen3-32B and Qwen3-30B have refreshed TP1/TP2 data recording 0.31, while
other bundles and Qwen3-30B's historical whole-block fallback keep their
recorded provenance. `RTX4090/meta-llama/Llama-3.1-8B` stays 0.19.
**A re-profile can move recorded clocks.** Refresh affected
regression baselines and example validation together when adopting new data;
the suite also includes other architecture families and a DeepSeek diagnostic.
Bandwidth efficiency is the estimated memory-time bound divided by measured
time, not a claim that measured time is below the bound. An outlier requires a
controlled attribution check, not an automatic conclusion that it is a bug.
These normalization checks do not certify all stored rows against later
acquisition-protocol changes.

`num_mtp_modules` is capped to 1 in the config **file**, not via
`hf_overrides`: the drafter reads `speculative_config.draft_model_config.hf_config`,
built from disk. M3's 7 modules are ~103 GB otherwise. M3 additionally needs
`scripts/patches/vllm_m3_mtp_layer_name.py` to start at all.

### A dense model's sampler is invisible on vLLM 0.28
The collector's module scopes follow **module** calls, so a component that
is not an `nn.Module` cannot become a node no matter what it launches. vLLM 0.28
has two samplers and picks between them by model:

| runner | chosen when | sampler | a node? |
|---|---|---|---|
| `vllm.v1.worker.gpu_model_runner` (V1) | MoE, hybrid | `vllm.v1.sample.sampler.Sampler`, an `nn.Module` | yes |
| `vllm.v1.worker.gpu.model_runner` (V2) | **everything else** | `vllm.v1.worker.gpu.sample.sampler.Sampler`, a plain object | **no** |

The rule is `VllmConfig._is_default_v2_model_runner_model`:
`is_default_v2_architecture or not model_config.is_moe`. So **every dense model
profiled on 0.28 silently loses its `sampler` row**, and the simulator then
refuses the bundle with `Missing per-sequence profile for layer=sampler`. The
four 0.28 bundles that existed before this are all MoE, which is why it
surfaced only with the first dense refresh.

Two things make it hard to spot from the outside. The two runner classes share
the name `GPUModelRunner`, so `type(runner).__name__` cannot tell them apart —
only `__module__` can. The historical Python-call coverage passed at **100%**:
the sampler's work never entered that tree. Native-activity coverage now retains
it as an explicit gap unless the sampler facade gives it an owning module.

`profiler/core/hooks/sampler_shim.py` wraps a non-module sampler in an
`nn.Module` named `Sampler`, installed once per worker by `Extension.fire` and
`Extension.coverage`. The name is deliberate: a profile node carries its
module's class name and that name is what a catalog binds, so one
`sampler: {vllm: Sampler}` covers both runners and no catalog changes. The
alternative was a second spelling in all nine of them.

The term is not negligible — 21.6 us at one sequence to 107 us at 256 on
RTXPRO6000/Llama-3.1-8B, against `lm_head`'s 714 to 840. Two independent
checks that the shim measures exactly the missing work and nothing else:
CUDA-event timing around the sampler call (29.5 us at 1 sequence, 63.7 at 128,
90.3 at 256, an upper bound since it brackets the Python call too), and
coverage's own total, which rose by 22.3 / 24.1 / 29.4 us across the three
regimes once the shim was in.


### Skew acquisition and offline calibration

`skew_plan.py` generates workload-independent distributions from configured
bounds and resolved engine limits. It covers history families, request order,
token-frontier batches and equal/unequal multi-prefill splits. Decode query
lengths follow the attention sweep. Planning streams one expanded batch at a
time; feasibility uses page-aligned actual KV allocation and reserves the
sampler's final token. No benchmark inputs belong in the planner.

`skew_support.py` checks the base plan in the compiler's actual partition
coordinates using the attention table. It adds deterministic draws where an
observed N/partition lacks `MIN_ROWS` distinct cases. Selection never reads
skew timing targets or benchmark results. CPU preview and live acquisition
share this logic, including the retained raw data's prefill envelope. Extra
cases retain compact identities and regenerate their request arrays lazily.
Bounded candidate search may leave deficits: `support_completion` records
them separately from acquisition completion. Do not equate completing a
sweep with every lookup cell having direct support. Existing raw data remains
preserved unless `--force` is requested.

`hooks/skew_measurement.py` measures the actual heterogeneous batch through
vLLM 0.28. Three independent contexts, each containing three timed forwards
by default, produce a median of forward medians. CPU scopes establish call
ownership; correlated GPU annotations are not kernel work. Each measurement
context initializes its assigned dummy KV pages before warmup.
Warmups verify finite output and executed geometry.
Recurrent state groups are identified by vLLM KV-cache spec, not model names;
the paged attention groups supply the history/query geometry in hybrid stacks.
Backend metadata need not retain query starts: sparse indexers split and
discard them. During untimed warmup, verify the common metadata passed to the
actual builder and retain the exact returned object's identity. Use that
evidence only when direct backend geometry is unavailable; never skip an
unknown backend by name. Restore builder methods before timing and on failure.
Attention and ordinary categories use the same per-call CUDA activity union as
skew, but retain their category-specific averaging and axes.

`hooks/cuda_timing.py` forms the interval union within each raw module call,
separately per device, before the collector groups compatible invocations. Overlapping
streams on one device count once; different calls must not be pooled before
normalization. Resolve a native activity by correlation, name and exact device
interval, but use CPU launch scopes for ownership. Exclude annotations and
device gaps, reject missing/ambiguous activities, and preserve event coverage.
This is device-active time, not whole-step wall time. Catalog components may
overlap each other, so their summed unions need not equal a global union.
Coverage remains a kernel-work audit, not a union-latency percentage.

New ordinary CSVs contain coordinates and `time_us`, not provenance columns.
Record per-category acquisition identities, method descriptions and actual runtime
versions in variant-root `meta.yaml::per_tp[tp].measurements`; preserve earlier acquisitions when
extending compatible tables. Describe the PyTorch/Kineto collector, per-invocation
device-active union, normalization and dummy input preparation directly, without
numbered protocol branding. Resume compares semantic catalog, geometry, method
and execution contracts, not source-file hashes or exact library versions.
Those versions are provenance, not execution vetoes. Historical source hashes
remain observations; new runtime admission does not collect source-file hashes.
Actual API, backend, shape, dtype, layout, finite-output and coverage guards remain.
Missing or incompatible acquisition metadata requires a separate output root or
explicit remeasurement with `--force`; never restamp historical measurements.
TP-stable replication preserves acquisition provenance and rejects mixed methods.
When a replica-only destination begins local acquisition, promote its category
metadata to the local method while retaining each replica's source-TP identity.
Validate preserved rows against that source record during local resume; copying
TP1 rows never claims a local measurement or completed local coverage.
Fully skipped categories do not acquire a new measurement timestamp. Skew records
its per-forward method, contexts and runtime observations in root `per_tp[tp].skew`;
CPU `plan-skew` uses the same semantic projection without importing Torch/vLLM.
Do not read `skew.meta.yaml`; acquisition and calibration use root metadata only.
Native DP+EP component contracts and skew-fit invalidation remain separate.

Use the variant-root `meta.yaml` for all new profile metadata. Per-TP measurements,
category publication, context/callable records, skew acquisition and native MoE
records belong under `per_tp[tp]`; do not create metadata sidecars beside CSVs.
Already-central `engine_resolved.per_tp`, `attention_lookup.per_tp`,
`component_layouts.per_tp` and `skew_fit.per_tp` retain their existing locations.
The empty `.meta.lock` coordinates metadata writers; it is not a measurement
record. Preserve unrelated sections on updates. Narrow legacy reads do not
authorize restamping historical measurements or infer identity from directories.
Raw repetition checkpoints may support acquisition resume; failure-only error
records and diagnostic traces may support debugging. They are optional, not
always-on metadata/evidence files or simulator inputs, and are not committed
experiment artifacts.

Saved policy, method and format names use descriptive hyphen-separated labels,
not numbered suffixes. `metadata_names.py` holds shared names, not old-name aliases.
Obsolete symbolic contracts require fresh acquisition, not renaming stored records.
Legacy file-layout fallbacks still require current method contracts; unmarked
historical mainstream tables retain their existing reads. Real library/model
versions and numeric structural schema fields remain unchanged.

`hooks/history.py` prepares query request state and completes asynchronous output.
`hooks/dummy_cache.py` initializes assigned attention pages
through the installed vLLM `initialize_single_dummy_weight` with its default
uniform range and per-tensor seed. The backend defines the logical block axis;
kernel subdivisions map back to the request's manager block IDs. Leave
unassigned pages untouched. Reinitialize per context, outside warmup and timing,
so preceding queries cannot become the next shot's synthetic history.
FP8 initialization uses bounded slabs, each with the upstream per-tensor seed,
to limit its temporary FP16 conversion. Supported NVIDIA packed caches initialize
floating payloads with that same helper, then quantize and pack the native format:
`fp8_ds_mla` uses 512 FP8 values, four FP32 scales and 64 BF16 RoPE values in each
656-byte record; `DEEPSEEK_V32_INDEXER` uses separate value/scale planes within
each block (128 FP8 values and one FP32 scale per token), optionally shuffled in
16-by-16 tiles. Do not treat the index cache as interleaved 132-byte token records.
Main scales clamp `amax / 448` at `1e-4`; index scales clamp `amax` before dividing
by 448. Both round the scale upward to a power of two. Never initialize scale bytes
randomly, substitute unit scales or run a native forward to generate this history.

Dispatch by the verified cache spec/backend contract, not the model name. Check
dtype, head/state geometry, sparse role, explicit shuffle mode and byte strides;
unknown packed formats remain refusals. Floating `MLAAttentionSpec` indexer
caches use the typed resolver. A three-dimensional `[blocks, tokens, channels]`
view may omit the head axis only when the spec has one KV head and matching
floating dtype; restore that singleton axis for logical shape validation.
Require non-overlapping strides for this squeezed view: in ascending stride
order, each non-singleton axis must start beyond the preceding axes' extent.
Allow padded and permuted views that pass this check.
Keep existing four-dimensional typed views and strict opaque-byte checks.
A runtime without the sparse-role capability retains the original typed-cache
resolver. Preserve null/unassigned
pages and block padding, including when layer views share backing storage.
Bound floating-source chunks independently of total cache capacity; quantization
temporaries add a bounded multiple, so the source cap is not a peak-memory claim.
Initialization stays outside warmup/timing and in acquisition method metadata;
old rows must not be relabelled to bypass incompatible resume.
Recurrent states are zero-start, not generated history. None of these inputs
claims trained-workload KV statistics
or representative data-dependent sparse selection.

Token IDs use a local deterministic generator bounded by the live,
TP-overridden vocabulary. Drain each asynchronous result before reusing input
buffers or request state; fresh IDs must not become streaming extensions of the
previous shot. Metadata cleanup does not erase physical KV pages. Preparation
adds no simulator cost.

Query geometry and request phase are separate contracts. `PreparedQuery` carries
the shot's explicit prefill/decode boundary through dummy KV preparation.
For a history-bearing decode, the prompt ends before the query: V1's fresh
`CachedRequestState` is split before `InputBatch.add_request`, while V2 receives
the prompt separately from all token IDs. Preserve token values, computed
positions and physical pages, and leave sorting/backend thresholds to vLLM.
Restore instance hooks even on failure. Making the whole sequence a prompt
leaves decodes as short extends; a sparse indexer with a lower decode threshold
than the main attention backend can then classify an entire mixed batch as
prefill. Geometry-only checks cannot catch that. Zero-history category shots
remain fresh prompts. Record this request-phase preparation with the CUDA
device-active-union method, not as a numbered protocol suffix.

### Runtime contexts and coordinated component layouts

The worker extension installs capability-checked dummy-weight preparation before
native loading. Ordinary models call the original loader unchanged. For the
admitted native MXFP4/SITU backend, initialize checkpoint-format
expert bytes with the native quantizer in bounded row chunks; preserve native
backend selection and postload processing. Do not substitute arbitrary uint8
values or silently emulate unsupported TP/EP layouts. KDA's nonpersistent
decode conv/norm buffers copy their native parameter transforms after floating
initialization, rather than receiving independent random values. Preparation,
postload structure and measured-forward validity are separate facts. The main
engine persists the preparation receipt and method metadata describes the inputs.
Preparation alone does not certify a forward or serving support.

Architecture `native_profile` selects `NativeProfile` orchestration in
`callable_profile.py`, with native API/backend inspection delegated to an admitted
adapter in `hooks/runtime_admission.py`. Keep common category publication in
`context_tables.py`, strict loading in `context_lookup.py`, and contextual
bundle/sizing integration in `serving/core/component_sizes.py`; do not add
checkpoint-specific orchestration or loader copies. The hybrid/residual adapter
retains native capability, ABI, precision and execution-mode checks. Config-derived
geometry does not establish arbitrary model/backend or device support.

Kimi-K3's local eager adapter preserves the independently supplied nested config.
Shared stack helpers derive category prefixes and original-depth residual roles;
MoE covers the MLP axis and other categories retain their declared axes. Keep
model widths, expert count and top-k native, with distinct MLA and KDA dimensions.
A completed category requires every planned layer/coordinate, independently
rebuilt catalog coverage, compatible semantic/runtime contracts and an unchanged CSV.
Retain measured library versions as observations; do not infer absent versions
from the consuming environment or use an exact version as execution admission. Validate partial
CSVs before deduplication or preload can hide malformed/duplicate rows. A partial
shot is reacquired in full. Old publication schemas and changed execution semantics
require fresh acquisition, never relabeled measurements or inferred completion.
Preparation elapsed times remain engine telemetry but do not participate in
runtime identity. Keep actual layout checks and table-integrity hashes; source
file hashes in historical records do not pin execution. New admission records
retain actual backend/layout and version observations, not source-file digests.
Unscoped runner GPU work remains visible as coverage gaps; complete model
boundaries do not imply complete step-time accounting or a zero-gap CLI result.

KDA's projections and recurrent core are separate. Native MLA's q-LoRA phase
joins concurrently executing gate and attention branches; its per-call device
union must remain one atomic cost. This concurrent boundary is the exception
to splitting independent scaling axes: independently adding the branches
double-counts overlap. The gate application and output projection remain separate.

Schema-3 callable contexts preserve declared scalar arguments and tensor
metadata, never values or CPU duration. Contextual dense rows cannot enter
the token-only sink. The hybrid/residual planner uses original-depth source roles
through the unchanged native function, not the reduced engine's pooled calls.
Its isolated synthetic arguments, native ABI and CUDA call evidence are validated
before publication. Root `per_tp[tp]` retains category records under `categories`,
context contracts/receipts under `contexts.{ordinary,callables}`, and callable
options, plans and compact per-shot summaries under `callables.callables`.
Keep `callables/dense_context.csv` as timing data; do not write new raw CUDA
ledger/evidence JSON files. Historical sidecars remain read fallbacks. Lookup validates
them once, then interpolates tokens only within an identical structural context,
without cross-context fallback or extrapolation. Keep the fixed allocation
bound explicit; do not silently reduce context capacity to fit it.

`ContextualProfileBundle` requires complete ordinary and residual tables, local
TP1/PP1/EP1 eager text execution and explicit `duration_replay_only=True`.
Source-role residuals disable transformer-block copying. Logical byte triples
are not resident capacity or traffic. Distributed, graph-padded, speculative,
offloaded, prefix-cached, power and roofline execution remain unsupported;
do not present bounded profiling/trace readiness as full-depth GPU execution
or end-to-end accuracy. The hybrid/residual path does not enable skew acquisition.
Shared geometry and trace routing read expert top-k through `expert_top_k`;
preserve the existing expert-group selection semantics.

`capture_scope_contract.py` projects the whole architecture to ordinary, coverage
and skew RPCs, preserving exclusions across category slices and including them
in host/worker identities. Methods/operators use schema 1; explicit callable
declarations add schema-2 `callables`, including inactive alternatives, and
argument/layout declarations use schema 3. Common
`embed_input_ids`/`compute_logits` method scopes provide runtime ancestry for direct
calls, not independently bindable costs or arbitrary-method instrumentation.
Restore exact instance state; reject foreign replacement, async/generator methods
and ambiguous normalization. Exact native `operator:` scopes own CUDA activities,
never CPU duration; reject wildcards, launch APIs, collisions and ambiguous or
out-of-forward/thread scopes. Contexts add no descendant normalization units:
consume returned normalized components, not an augmented tree via an old extractor.

Each catalog entry chooses exactly one `vllm`, `operator` or typed `callable`
binding. A declared function resolves an exact attribute of an already loaded
module; a declared method resolves an exact object path under an observed runtime
owner, including subclasses. Do not import targets, dispatch by model family or
replace kernel implementations. Callable scopes remain transparent to descendant
normalization; their own timings are per-call CUDA unions, never CPU duration.
An explicit `normalization` chooses `parent` or `invocations`; omission preserves
ordinary parent and whole-MoE invocation defaults. Check that the selected unit
matches one trace component rather than using normalization as a fitted factor.

Unavailable targets remain visible in `callable_audit` and once-per-runner
warnings. They are not zero measurements or proof of selected-layout coverage.
Reject failing descriptors, unsupported async/generator or data-descriptor targets,
ambiguous ownership, kind collisions and overlapping canonical claims. Restore
exact prior attributes on exit; preserve a foreign replacement and invalidate the
capture rather than overwriting it. Both acquisition identities include the
full declarations and their capture semantics, not source-file hashes.
Preserve historical simulation inputs,
but never relabel them to resume an incompatible acquisition.

`CoverageReport.canonical_invocations` preserves actual per-measured-forward
module/operator/callable counts, positive-CUDA call counts and raw CUDA activity
counts separately. Normalization groups retain their exact divisors; never sum
them into a fabricated call count. Check expected calls against the source-reviewed
execution path, not the denominator or a coverage percentage. CPU-only calls do
not establish CUDA coverage, and missing entries must not become zero timings.
Warmup/preparation calls are excluded; this audit changes no latency formula.

`component_layouts.py` is a shared CPU-only contract for declared historical and
alternative decompositions over `dense`, `linear_attention` and `attention`. Select a
unique variant from positive prefill/decode/mixed CUDA coverage on every worker
and check every acquired shot. Never infer it from model/version names or absent
kernels. Small probes do not certify unmeasured shapes, backends or deployments.
Discovery uses fixed `runner._COVERAGE_REGIMES`, not limits-adapted probes; small
engine envelopes can reject discovery even when their requested grid is valid.
Validate all fixed probes on the host before the first worker RPC. Their minimum
envelope is 5 requests, `max_model_len >= 258` and an acquisition token buffer
of at least 68 (the logical token budget plus `max_num_seqs`). Refuse an undersized
envelope without resizing or skipping probes. `hooks.batch.validate_shot_limits`
also checks ordinary synthetic request count, query-token total and per-request
history plus query plus one output against actual engine buffers before page
allocation or GPU work; preserve this guard for direct worker calls.

`meta.yaml.component_layouts` (`component-layouts`, per TP) is the sole authority.
Publish `pending` before clearing, `writing` with identity, then `complete` only
after row validation. Changed shared identity requires explicit force across
coupled categories; changed representation/definition requires fresh output.
Resume whole valid shots, preserve unrelated TP-stable identities, and refuse
pending/mixed/stale/malformed data or missing complete-table regimes.
`writing` categories may resume before their first measured shot, including when
only valid TP1 replicas remain at TP2; validate replicas before exempting them from
local coverage. Empty/replica-only categories cannot become `complete`. Unmarked
historical tables keep their old sequence; alternative-only rows require metadata.
Qwen's packed core includes its norm, so refresh dense and linear_attention together.
Use one writer per variant root and immutable published bundles: atomic replacement
is recovery, not locking. See the output-bundle guide for the full publication rules.

Site-scoped alternatives name existing `blocks`/`shared` paths. Apply replacements
once; a `layer0` prefix may be consumed only at absolute target layer 0 by the
declared active prologue endpoint after its expected prefix matches. Never remove
an MTP or later-layer norm by analogy. `site-scope-calls` checks actual canonical
calls and timing divisors against the resolved prefix before publishing each shot.

`native-dsa-eager-tail` binds original source configuration and logical TP/EP
to live native runtime facts. Keep configured FP8/BF16 matrix widths separate
from BF16 activations;
do not shrink model width, emulate another TP or accept skipped/shared indexers.
The physical acquisition is local TP1/PP1/DP1/EP1, eager decode q=1, without MTP,
offload, sequence parallelism, dense-MHA prefill or quantized query input.
Retain packed cache geometry and the actual unwrapped sampler class/entrypoint
and entrypoint ABI. Source hashes and library versions are observations, not
admission locks. Sampler payload width is source-backed, not observed
tensor data; V1 returns int32, V2 int64. Do not force a runner to fit a byte formula.

New native DSA publication requires `dense`, `per_sequence` and `attention` with
matching runtime identity and per-category engine facts inside the existing per-TP
component-layout record. Native MoE components publish independently under their
deployment contract; new ordinary records must not require the removed `moe.csv`
acquisition. Historical records containing `moe` retain their original completion
checks. Never bless old rows by adding these fields. The combined tail includes indexer work and must use full-context
coordinates, not the historical sparse-attention cap. Historical representations
and MiniMax's independent native attention contract remain unchanged.

Native DSA traces require explicit `duration_replay_only=True`; default serving
must refuse the native 656+132-byte cache until capacity integration exists.
Logical component bytes are not traffic or allocator residency. Refuse distributed,
P/D, offload, PIM, speculative, graph-padded, prefix-cache, power and roofline modes.
Cache layout decisions and immutable substitutions by TP/site, sharing lazy caches
across TP views; recheck source configuration on native byte-contract cache hits,
including mutations of an already seen dictionary. Do not apply new restrictions
to historical unmarked bundles.

Ordinary category sweeps checkpoint the accumulated CSV between completed
shots, including early in a new run. `DedupSink.flush(clear=False)` retains
all rows, acquisition identities and duplicate counts; writing only the most
recent rows would erase earlier progress. Stage the measurement metadata until
the replacement CSV is serialized and fsynced. Publish
root `per_tp[tp].measurements.<category>.publication` with `state: pending` and the intended
`csv_sha256`, replace the CSV and fsync its directory, then mark the publication
`complete`. Apply the same ordering to later checkpoints and TP-stable replication.
Resume validates the CSV against either state's checksum: matching pending data
are reusable, while mismatches require fresh output or explicit `--force`.
Checking/preloading neither finalizes pending metadata nor stamps older metadata
that lacks a publication record. Keep these data-generation seals separate from
source/version provenance and category-coverage completion. A prior checkpoint
is resumable only with compatible metadata; never accept old CSV bytes under a
new forced-acquisition identity after an interrupted replacement. This
does not turn a partially acquired table into a completed simulation bundle:
finish the requested sweep before adopting it. Checkpoint I/O is outside the
CUDA timing context and does not add simulated CPU time.

Attention resume keys use `_attn_key` unchanged, as does `extract_points`.
The CSV writer formats latency only; coordinate floats keep their full
precision. Do not round only the planned key: weighted prefill coordinates
can have more than three decimal places and would miss their stored rows.
This identity alignment changes neither measurement protocol nor saved timings.

The common `hooks/activity_collector.py` captures actual runtime module
ancestry, including shared objects and repeated calls. Native OS-thread CPU
scopes and launch correlations establish ownership; never use GPU timestamp
containment. Duplicate, missing, cross-forward or ambiguous ownership fails
closed. Ordinary shots require the same positive active canonical set in every
forward, without requiring inactive catalog alternatives to execute. Reject
overlapping canonical claims before storing timings. Coverage keeps a unique
native kernel/copy/memset ledger and explicitly reports unscoped work; never
silently label it preparation or add it to an arbitrary layer. Preserve overlap
errors through the host CoverageReport and CLI exit status.

Untimed recurrent warmup checks must bind each query/history pair to its own
assigned state pages, including reused backend metadata. The audited ordinary
GDN contract uses cache mode `none`. Native vLLM 0.31 KDA has a separate,
native API/metadata contract: it consumes the complete non-speculative view, including
the decode prefix of a mixed batch, and does not populate GDN's prefill-only
views. Keep state-page, query/history, initial-state and token-conservation checks
for both. Reject unsupported APIs/schemas and active speculative, checkpoint,
RecoverSSM or optional FlashInfer partitions. This warmup guard does not certify
a model's catalog, acquisition grid or serving support. Changed acquisition
semantics require remeasurement, not metadata relabeling.
MTP shots require every requested canonical
entry, rather than publishing a partially observed drafter table.

`hooks/cache_pages.py` allocates all cache groups from one shared physical
pool, reserving null block 0. Group IDs must not overlap even when typed views
have different logical shapes. Check the sum of all groups' physical page
requirements against capacity. Initialize only assigned spans, respecting
backend block axes and kernel subdivisions; preserve null and unassigned pages.
Recurrent caches retain zero-start state. Unsupported state modes and packed
layouts fail explicitly. Record allocation semantics in ordinary and skew
method metadata; incompatible rows require remeasurement, not relabelling.

Raw skew CSV rows preserve ordered requests, query roles and actual times;
metadata records repetitions and method. Complete kernel sets and sufficient repetitions are required
for resume. Failed acquisition raises after checkpointing instead of silently
publishing partial coverage. Root `per_tp[tp].skew` records the actual per-TP plan,
resolved limits and completion; metadata must not fabricate this from defaults.
If all planned skew rows were saved before a completion-update interruption,
resume may seal that record without new timings or acquisition provenance.
`profiler plan-skew` previews the same plan and outstanding cases on CPU from
an existing bundle's resolved engine limits. Both `plan-skew` and
`profile --only-skew` accept an independent positive TP selection such as
`--tp 2`; they neither measure nor replicate TP-stable categories. Ordinary
category profiling still requires TP1.

`skew_calibration.py` recomputes both endpoints with serving's unchanged
query-weighted attention lookup. Exact kernel/query reference slices are
required. Bimodal historical raw measurements remain reconstructible; that
input conversion is not an old runtime estimator. General data requires full
geometry, and multi-query data an explicit prefill boundary.

The weighted median minimizes each cell's summed absolute relative
profile-latency error before clipping. N anchors require sufficient distinct
support within their kernel/query/prefill/lever partition. Sparse counts pool
to the nearest supported anchor using geometric midpoints. Prefill boundaries
scale with the measured token envelope; leverage boundaries are dimensionless.
Unsupported cells use their own kernel/query pooled fallback; missing
kernel/query data means zero correction. No cross-model, hardware, TP or
query borrowing is permitted.

Only the supported `runtime-skew-calibration` format can be enabled. Remove, rather than
revive, old inline alpha and legacy bucket lookup code. Disabled bundles need
no migration; the RTX4090 bundle remains untouched. The normal writer and
CPU-only `profiler refit-skew` use the same compiler. Identity, attention,
table, lookup-code and saturation fingerprints are checked before execution.
An empty rebuild is an error and must not enable a disabled fit. A root skew
record marked incomplete also blocks rebuilding until acquisition is resumed.
Repeat-level skew CSVs require complete root acquisition metadata; unmarked
historical tables retain their existing behavior.
`--skip-skew` skips new acquisition, not existing data; `--only-skew` requires
existing attention references before GPU work begins.

Validate every statistic through `bench validate` on committed examples,
without fitting their request latencies. Broader acquisition and a shared
rule do not guarantee improvements on every model. Generalization needs
unseen models, hardware and workloads; endpoint summaries are not sufficient
statistics of arbitrary distributions.

### CUDA graph timing and calibration boundaries

Ordinary profiles measure per-call CUDA activity unions, not eager engine
wall time. CUDA graph dispatch changes execution shapes and may change kernel
selection or overlap; graph padding alone is not a wall-time correction.
Do not subtract an isolated eager-versus-graph wall-time difference from
profiled GPU work or add CPU launch/preparation time to simulated GPU latency.

Use matched query/KV shapes, execution modes and measurement boundaries when
checking a residual. Tick-level throughput aggregates and end-to-end agreement
cannot identify a per-layer error or establish a universal per-step adjustment.
New native MoE components use eager-only acquisition, without isolated graph
capture. Serving graph padding is separate and still determines lookup rows.
Historical eager/graph component tables retain their measured mode selection.

Calibrate communication from primitive measurements using the backend's
actual cost model, not by fitting model benchmark errors. Keep raw samples,
rank count, units and calibration assumptions with the hardware bundle.
Full-depth graph timing and sharded layer profiling are different contracts:
sharding GPU work does not scale host costs or reproduce full-model cache state.


### hardware.yaml: the machine's own facts, measured
A cluster config mixes two kinds of statement:

```
tp_size, num_npus, mem_util, dp_group      what the user wants to simulate
link_bw, link_latency, npu_mem.mem_*       what the hardware actually is
```

The first stays under the user's control. For owned hardware, the second
should be characterised independently of model benchmarks. Fitting link
parameters against end-to-end accuracy can conceal unrelated model errors.

`python -m profiler hardware --hardware <hw> --npus 2` writes
`profiler/perf/<hw>/hardware.yaml`: one file per hardware folder, shared by
every model bundle under it. The hardware command defaults to INFO logging;
an explicit `--log-level` is optional and validated at argument parsing.
Three sections:

- **`spec`** — queried from the device. GPU name, SM count, bus width, clocks,
  PCIe generation and width, driver, power limit, and `memory_bw_gbps` derived
  from the reported clock and bus width rather than hardcoded.
- **`measured`** — the NCCL sweep: **all three collectives** the simulator
  emits (AllReduce, AllGather, ReduceScatter), timed inside CUDA graphs and
  separately in isolated calls, with raw samples retained alongside the fit.
- **`defaults`** — what a cluster config inherits, each entry carrying its own
  `source`: `measured`, `spec`, or `assumed`, and units. A run logs each
  inherited value with that provenance.

**Fit the model the backend executes.** `_ring_terms` in
`profiler/core/hardware.py` represents one local-dimension, single-chunk Ring
on a one-hop FullyConnected topology. With `N` ranks, AllReduce has `2(N-1)`
network phases; AllGather and ReduceScatter have `N-1`, not the same latency
multiplier. Raw sample bytes mean AllReduce input, AllGather local input, or
ReduceScatter local output. Convert them to Ring message size before counting
traffic; they equal charged traffic only at `N=2`.

Network BW uses **GiB/s**, as `NetworkFunction.cpp` multiplies it by `2^30`;
local-memory BW uses decimal **GB/s**. `PacketBundle.cc` separately charges
three local transfers per reduction step and `MemBus.cc` adds endpoint events.
Include these known costs in the fit, using the integer local-memory spec
that `config_builder.py` supplies to ASTRA; do not let them be absorbed into
link BW. The detailed formula is in
`docs/docs/profiler/adding-hardware.md#calibration-contract`.

**One pair is an approximation, not proof that NCCL shares a curve.** The
objective remains relative squared error over the primitive sweep, solved
as nonnegative least squares in latency and inverse BW without model-specific
coefficients or hardware-specific search bounds. The network term is continuous
for fitting; saved predictions use serialized parameters and the backend's
integer truncation. Record fit version, rank count, units, assumptions, timing
source and per-size/per-collective residuals. Invalid or unidentifiable inputs
must fail rather than emit link defaults. A zero-latency boundary is flagged.

The primitive sweep does **not** measure grouped multi-tensor or ragged MoE
communication. Summing their payload bytes into one primitive does not make
their execution equivalent. Other topology, splitting, rank count or memory
settings require validation against that contract. Existing `hardware.yaml`
files are not automatically rewritten by a code update; old fit residuals
cannot certify the corrected formula. Refit retained samples with their
measurement context or re-measure before validating new defaults.

**Optional per-operation links retain physical bytes.** `collective_links`
in the cluster config selects `link_bw` / `link_latency` for `all_reduce`,
`all_gather` and `reduce_scatter`, each scalar or one value per topology
dimension. Missing fields and operations use the common link. The builder
emits same-topology sidecar YAML files referenced by `collective_networks`.
The congestion-unaware analytical frontend supports these overrides for Ring
only; ns-3 and other collective algorithms are rejected. Preserve the original
collective on the stream/request: AllReduce's internal scatter/gather phases
must use its AllReduce curve, not standalone ReduceScatter/AllGather curves.
Do not change message bytes, rank groups or local-memory charges to emulate BW.

The hardware command keeps the joint fit and records `collective_fits` at its
fixed latency, with separate residuals. Each identifiable operation supplies
`defaults.collective_links.<operation>.link_bw`; an unavailable fit records its
reason and supplies no override. These effective primitive fits do not model
NCCL's grouped launch semantics or establish accuracy at other rank counts.
Existing hardware files are unchanged until explicitly refreshed.

Hardware operation defaults are inherited only when neither common link value
nor an explicit `collective_links` key is supplied, and every instance shares
one hardware label. An explicit map replaces the hardware map; `{}` disables
it. This precedence must remain idempotent across both config readers, so a
hypothetical explicit interconnect never silently gains the measured card's
operation curves.

The RTXPRO6000 Qwen3-32B TP2 and Qwen3-30B DP2/EP2 examples omit all three
link keys and consume the measured operation defaults automatically. Their
vLLM 0.31 references preserve the recorded custom-all-reduce setting: disabled
for Qwen3-32B TP2, not disabled for Llama TP1 or Qwen3-30B TP1/DP2/EP2.
Neither TP1 deployment exercises the TP custom-all-reduce path. Refresh
each example's recorded truth, simulator output, validation plots and
regression digests together when adopting a changed interconnect. Do not
pair refreshed hardware defaults with an older transport reference.

**Three inheritance rules** (`serving/core/hardware_defaults.py`, applied at
both config load sites):

1. **An explicit value always wins.** No exceptions and no warning — a config
   describing an 8-GPU NVLink node it does not own must be able to say so.
2. **A gap is filled from the bundle**, and the run logs the value and its
   provenance.
3. **A gap with nothing to fill it raises.** Not a default, not a warning. If
   the interconnect was never measured — one GPU on the machine, or hardware
   that is gone — no number is defensible.

`link_bw` / `link_latency` are cluster-level while `hardware` is per-instance,
so they are inherited only when every instance shares one label. A cluster
mixing two card types has a link that is neither one's intra-node measurement.

**Two GPUs are the floor, and the measurement is topology-specific.** A link has
two ends. `profiler hardware` still writes a useful file on a single-GPU machine
— the spec section needs only a device query — with `interconnect: null` and the
reason, and it **exits non-zero** so a script notices, the same shape as
`profiler coverage`. And `npus` is recorded because an all-reduce across two
PCIe-linked cards is not the physics of eight over NVLink; a config asking for
more is extrapolating.

**RTX4090 is the worked example of the empty case.** The card is gone, so its
`hardware.yaml` is hand-written with `measured: null`, carries no link defaults,
and its bench example keeps `link_bw` / `link_latency` in the config — where a
reader can see they are the author's choice rather than a measurement.

**`mem_bw` is deliberately not measured.** A profiled kernel latency already
contains the card's real memory behaviour; `mem_bw` applies to the
explicitly-modelled memory paths (`--prefix-storage` KV recall, PIM, remote
memory) and ASTRA's local reduction cost. It is carried from the spec,
marked `spec`, and `mem_latency` is
marked `assumed` because every committed config has carried 0 and nobody has
checked it. What will matter is **CPU** memory bandwidth, which those paths do
read.

**`mem_size` is CUDA's number, not the marketed one.** vLLM applies
`gpu_memory_utilization` to `init_snapshot.total_memory`, which is
`torch.cuda.mem_get_info()[1]` — 101,976,440,832 B = **94.97 GiB** on this
card, against a marketed "96 GB". That is not a 1000/1024 slip: 96 decimal
would be 89.41 GiB, so the marketing figure is GiB labelled GB, and the gap to
CUDA's number is ECC/firmware reserve (0.4%) plus driver context (635 MiB).
Since the simulator's `mem_util` mirrors vLLM's flag, its base has to be the
same one. It changes nothing on a run that does not saturate — Qwen3-32B peaks
at 36.8% KV with zero preemptions, and 96 vs 94.97 came out identical to four
decimals — and moves KV capacity by 1.1% on one that does.

### The attention grid's second axis describes the batch, not one sequence
It used to be `kv_prefill`, one prefill sequence's context, and the simulator
fed it `sum(batch.prefill_k_list)`. A step routinely carries several prefill
sequences -- one request finishing its prompt beside another just starting --
and summing their contexts produces a coordinate none of them has: 0, 1200 and
64 came out as 1264.

The axis is **`prefill_key`**: the query-weighted mean of each prefill's
effective causal key length --

    prefill_key = sum_i c_i * (k_i + c_i / 2) / sum_i c_i

`c_i` is what request *i* has scheduled this step and `k_i` what it already
holds; the half is causal masking, since the average query in a chunk sees
half of it. `profiler/core/attention_shape.py` owns this coordinate for both
profiling and serving. An empty prefill contributes zero. Cap each request's
effective key before weighting for a saturating kernel; this remains a window
approximation, not an exact discrete causal-work integral. Equal chunks retain
their old coordinate, but unequal chunks must not count as equally much work:
chunks 1792 and 256 with no history give 800, not the old sequence mean 512.

The following figures describe the earlier sequence-mean migration, not an
independent validation of query weighting. Matched against real vLLM runs -- table built from the shapes the
profiler sweeps, scored on multi-prefill steps -- that moves |err| p50 from
14.18% to 5.97% on DeepSeek-V3.2 and 12.96% to 7.93% on Llama-3.1-8B, and it
improves the single-prefill case too (2.15% -> 1.71%, 10.28% -> 9.88%). Adding
`n_prefill` as a fifth axis on top changes nothing measurable: once the key
coordinate is batch-aware the sequence count has nothing left to say. The grid
stays four axes wide.

**Sweeping it needs multi-sequence shots.** One sequence can only reach
`key >= tokens / 2`, so "many tokens, short keys" -- 4 requests of 500 tokens
with no context -- is unreachable, and **32-39% of a real run's multi-prefill
steps land there**, at the full token budget, which is exactly where a
saturated run's TTFT is set. `_compose_prefill` derives the batch that
realises a target coordinate: the fewest sequences that leave each a
non-negative context. About 27% of the grid's shots are multi-sequence.

**Old bundles are relabelled, not refused.** Every pre-change shot had exactly
one prefill sequence, so `prefill_key = kv_prefill + prefill_chunk / 2` is
exact and `_build_attention_table` applies it on load. What a relabel cannot
recover is the coverage above. `meta.yaml::attention_grid.axes` is the marker
that says which form a bundle holds.

`kv_prefill` still exists on `BatchCtx` and is still the summed context -- it
is a *length*, and tensor sizing and the PIM split need it as one.

### A saturating kernel is capped at lookup, per kernel
A sparse-attention query reads at most a fixed number of selected positions,
so its cost stops growing once a sequence's key window passes that bound:
DeepSeek-V3.2's 16-token prefill costs 968.8 us at no context, 742.4 at 512,
then 139.3 at 2048 and 138.3 at 8192 -- flat from exactly `index_topk`. The
same sweep on Llama rises throughout (8.5, 18.8, 51.2, 181.1), because a dense
query reads its whole causal window.

`stack.probe_key_saturation` reads `index_topk` for DeepSeek/GLM. For MiniMax-M3
it retains the committed tables' historical lookup rule, including the initial
and local block adjustment (17 x 128 = 2176). New native acquisitions instead
use `attention_lookup.py` and `meta.yaml.attention_lookup.per_tp`: the inspected
vLLM 0.31 Triton indexer/attend pair uses the configured selected-block count
with the reviewed 128-token block layout (2048 tokens for the shipped config)
for `sparse_attention` prefill lookup only. Initial/local blocks are part of that
total, not an addition. Exact selected classes, callable API, BF16
indexer/KV storage, selection-buffer layout and model/catalog parameters must
match; speculative acquisition and unreviewed backends/layouts are refused.
Native attention publication reads the current measurement metadata and recognizes
explicitly supported historical row protocols. Recognition does not waive
semantic acquisition compatibility or completion checks and never converts
historical measurements. Source-file hashes and version labels are not vetoes.
The catalog's `key_saturates` says which entries are subject to the bound. That
is a per-**kernel** property, not a per-model one: M3 is sparse only from its
fourth layer on, and a sparse model's `indexer` scores the whole KV to make
the selection and never saturates (52 -> 101 us from kv 2048 to 16384).

The cap is applied at lookup, never at sweep time. Both kernels share one
grid, so a capped column would collapse the indexer's large-key rows onto a
single cell. `BatchCtx` carries the mean both ways because the clip is per
sequence and must precede the mean -- `mean(min(x, cap))` is not
`min(mean(x), cap)`.

Native row protocols bind the per-TP semantic digest to the acquisition identity.
Pending records cannot be consumed; completed attention tables carry a checksum.
Missing contracts for marked rows, changed bytes or unknown schemas fail closed.
Never relabel historical rows, add a contract after acquisition, or change the
historical helper/fingerprint to migrate existing fits. Refresh in a separate
root, or explicitly remeasure the affected category. Serving, skew fitting,
support completion and `plan-skew` share this resolver and select each TP's cap
without mutating the shared cached database. Decode-history coordinates remain
uncapped. Partial non-attention refreshes preserve the authoritative contract.
Native skew publication also checks row identity and completion; a zero-case
plan creates no skew artifact, changes no existing artifact even with `--force`,
and contributes no new skew provenance or fit trigger to the caller.
TP1 prefix acquisition and boundary checks do not certify multi-TP deployment,
memory capacity, a full grid or end-to-end benchmark accuracy. See the
[public contract](docs/docs/profiler/output-bundle.md#native-attention-lookup-contract).

**The same flag governs the skew guard.** `alpha` is where a skewed batch
lands on the line between the two uniform attention-table lookups. That line
does not care which end is higher; the fitter and simulator accept either
sign for a kernel declared saturating. The former `t_max > t_mean` guard
was right for dense -- cost rises with kv, so a negative
gap is noise -- and wrong for a saturating kernel, where the gap is negative
for essentially every case, so a sparse skew sweep would have had almost every
row dropped as `nan`.

### Target graph coordinates and eager acquisition

`profiler/core/sampling.py` combines geometric points with target serving graph
coordinates through the shared `serving/core/cudagraph.py` resolver. Acquisition
always remains eager. Explicit `--cudagraph` JSON overrides the target policy;
otherwise prefer a scope-matching completed worker capture observation and derive
candidates only when none matches. Label observed captures, explicit options and
derived candidates separately. Acquisition-backend capability and a requested
grid do not establish full-model capture. Resolve from logical scheduler limits,
not padded acquisition buffer sizes; retain runner, query length, TP/compiler-SP
and hardware distinctions. Explicit `NONE` disables graph anchors.

Dense token grids include forward anchors; TP1 also unions the requested TP
degrees' anchors for TP-stable replication. Target logits and sampler execute
outside the backbone graph, so per-sequence sampling stays geometric. Attention
and linear attention add pure-decode request anchors only for FULL-capable modes
at the supported query length, and pure-prefill anchors only for `FULL`; mixed
shots retain geometric axes. Keep context, block-rounded KV and scheduler
feasibility checks. MTP adds drafter request anchors only: its table has no
complete first-pass query/token dimension. Skew keeps independent axes.

Root `per_tp[tp].sampling[category]` records the feasible plan, factors, target
coordinates, provenance and category limitations; it does not certify completed
measurements. Report excluded dense capture points. Native MoE records transformed
`expert_sampling.capture_tokens` and `sampling_graph` in each target contract;
do not place metadata in CSV columns or reinterpret eager timings as graph times.

### Attention-grid density and acquisition cost

`--grid-factor` defaults to 2 for ordinary geometric token/request coordinates.
Attention chunk/KV, skew and native MoE retain separate factor controls, all
defaulting to 2; only attention request counts retain `SQRT2 = 2.0 ** 0.5`.
The KV factor controls both prefill-key and decode-KV. The editable `profile.sh` leaves
optional settings unset and forwards only explicit nonempty values, so the
CLI owns its defaults. Keep every `profile` option reachable from the template,
including `GRID_FACTOR`, JSON-valued `CUDAGRAPH`, `DP_DEGREES` (`--dp`),
`EP_DEGREES` (`--ep`), `MOE_ROUNDS` and explicit logging; do not
silently enable a TP sweep or native DP acquisition. The separate multi-model
`profile-all.sh` remains an explicitly configured campaign, not a default-parity
wrapper. What picks sqrt(2)
over 1.5 -- which is *coarser*, not
finer, since 1.5 > sqrt(2) -- is that `_geometric_grid` accumulates in float
and rounds only for output, so the even powers land exactly:

```
sqrt(2)  0, 1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256
x2       0, 1, 2,    4,    8,     16,     32,     64,      128,      256   <- a strict subset
1.5      0, 1, 2, 3, 5, 8, 11, 17, 26, 38, 58, 86, 130, 195, 256          <- not nested
```

A doubling axis is therefore a **subset** of the sqrt(2) axis. Reuse still
requires compatible acquisition identities and matching complete shot keys;
changing another axis or the request composition can change those matches.
A factor of 1.5 is not nested in either grid. Compare actual coordinates
rather than promising that a density change reuses every previous row.

For a fixed range, interval count scales as `1 / log(factor)`. Changing a
factor from 2 to sqrt(2) roughly doubles intervals per axis, not 1.4x.
Attention combines four shape axes, so their costs multiply before rounding,
deduplication and feasibility filters. Enumerate `AttentionCategory.compose_shots`
with the requested settings and resolved limits for the actual shot count.
Do not infer whole-bundle duration from one axis factor or an old run's ETA;
skew, other TP degrees and native MoE acquisition are separate work.

**The accuracy case is specific to `n_decode`, and it does not extend to
chunk or kv.** That axis is smooth on a *pure decode* batch -- per-sequence
cost on Llama-3.1-8B is flat and slightly falling, 6.60 / 6.33 / 5.94 / 5.81
us at n = 16 / 32 / 64 / 128 -- which is what a doubling grid was chosen on,
and that measurement is right. It just does not describe a **mixed** batch,
where the same model rises 37% across the same interval (chunk 273, kv 2076,
one prefill sequence):

| n_decode | 16 | 32 | 64 | 72 | 80 | 88 | 96 | 104 | 112 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|
| pure us/seq | 6.60 | 6.33 | 5.94 | | | | | | | 5.81 |
| mixed us/seq | 9.18 | 9.28 | 8.99 | 9.33 | 9.45 | 9.47 | 10.39 | 11.03 | 11.55 | 12.31 |

The curve is flat to n=88 and then climbs, so the blend across a 64 -> 128
bracket over-charges the middle. Scored against those measured values, with a
profiled point at 91 standing in for what sqrt(2) samples:

| n | measured us | x2 blend | err | sqrt(2) blend | err |
|---|---|---|---|---|---|
| 72 | 671.8 | 700.4 | +4.3% | 670.0 | -0.3% |
| 80 | 756.0 | 825.4 | +9.2% | 764.7 | +1.2% |
| **88** | 833.4 | 950.5 | **+14.1%** | 859.4 | +3.1% |
| 96 | 997.4 | 1075.5 | +7.8% | 986.9 | -1.1% |
| 104 | 1147.1 | 1200.6 | +4.7% | 1134.1 | -1.1% |
| 112 | 1293.6 | 1325.6 | +2.5% | 1281.3 | -1.0% |

Worst |err| 14.1% -> 3.1%; attention is 54.8% of such a step, so the
step-level bias is 7.7% against 1.7%. And that is where a real run's error
was: matching Llama-3.1-8B's steps against a vLLM run one for one, the
simulator is exact at n_decode 0 and 32 (both grid points, 0.983-1.032) and
5-12% slow at 64-127, at every chunk size from 256 to 1792. A layer
decomposition puts the excess in `attention`. Everything else was ruled out
first -- skew is worth 0.4 points, `n_prefill` 2.0 points, and every other
layer is within a percent across the two shapes.

**Note what that table is and is not.** It is arithmetic on measured values,
so it bounds the interpolation error of the modelled quantity; it is not an
end-to-end result, and how much of Llama's +4.8% TTFT it recovers is settled
only by re-running the refreshed bundle against the truth. This is still a
different kind of evidence from the one that reverted `attention_chunk_factor`
-- that was an end-to-end A/B confounded by the bundle and the truth being on
different vLLM versions, while this is measured against the same engine in the
same run.

Chunk and KV axes default to doubling. The request-count evidence above does
not establish an accuracy benefit from densifying those other axes. Target
graph anchors add relevant pure-regime coordinates without changing their
geometric defaults or turning eager measurements into graph measurements.

**Skew acquisition density is separate from fit resolution.** Its geometric
factors retain their defaults in `ProfileArgs`; the new compiler creates N
anchors only where each partition has direct support. Increasing density can
add support or introduce new regimes, but neither necessarily improves the
fit. Prefill partitions scale with the measured token envelope; lever bins
remain dimensionless and fixed. Other measured shape axes still affect which
observations are pooled. Validate changed coverage through
the committed bench examples rather than assuming that more rows are better.

**A refresh has to restate the other axes.** Read each bundle's recorded
acquisition settings instead of assuming that historical profiles used today's
defaults. A category slice can change the grid when its factors or resolved
limits differ; a lower `--max-num-seqs` narrows support through feasibility
filters. Match the recorded chunk, KV and request factors, decode query lengths,
maximum KV, request/token limits and target graph policy when retaining the
same sampling envelope. A changed envelope is a new acquisition choice, not a
row-for-row refresh.

**A long sweep prints a heartbeat.** The rich progress bar needs a TTY.
Ordinary categories also write periodic atomic CSV checkpoints between
completed shots, independent of progress rendering. Low GPU utilization
alone does not establish a stall: CPU event processing contributes to
acquisition time but not simulated latency.
`_fire_one_category` logs count, rate and ETA every 1% of shots, and
`skew.sample_skew` logs at every 20-case atomic checkpoint. It reports the
actual acquisition rate, excluding reused cases, and retains completed cases
if a later measurement fails.

### Feasibility bounds shared by attention and skew
Both the uniform attention sweep and the skew sweep cap `n_reqs > max_num_seqs`
(strict `>`, not `>=`) so that `n = MSQ` **pure** cases (no prefill chunk) fit.
This uses vLLM V1's `input_batch` buffer exactly up to `MSQ`. Mixed cases at
`n = MSQ` need `MSQ + 1` requests and are still filtered. If a runtime workload
needs mixed-regime data at `n = X`, profile with `MAX_NUM_SEQS ≥ X + 1`.

The bounds that count **tokens** rather than requests have to see
`decode_q_len`, because a decode request submits `q` of them. They are inside
the `q` loop in `AttentionCategory.compose_shots` for that reason — see the
speculative-decoding section for what happens when they are not.

`--attention-max-kv` is resolved **once**, right after `probe_limits`, by
`engine.resolve_attention_max_kv`, and written back into the frozen
`ProfileArgs` with `dataclasses.replace`. Everything downstream — both grids
and the meta the run records — therefore reads one concrete number and never
`None`; `compose_shots` asserts on `None` rather than silently sweeping at the
wrong resolution.

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
| `qk_norm` | dense (TP-dependent; Qwen3 only) | `tokens = total_len` |
| `rotary_emb` | dense | `tokens = total_len` |
| `attention` | attention | `(prefill_chunk, prefill_key, n_decode, kv_decode)` |
| `o_proj` | dense + ALLREDUCE after (TP>1) | `tokens = total_len` |
| `gate_up_proj` | dense | `tokens = total_len` |
| `act_fn` | dense | `tokens = total_len` |
| `down_proj` | dense + ALLREDUCE after (TP>1) | `tokens = total_len` |
| `final_layernorm` | dense (tp_stable) | `tokens = total_len` |
| `lm_head` | per_sequence | `sequences = num_requests` |
| `sampler` | per_sequence (tp_stable) | `sequences = num_requests` |
| `moe` | moe (tp=1, one grid per EP degree; wrapped in the EP all-to-all) | `(ep_total, local_tokens, activated_experts)` |
| `mtp_*` | mtp (only with `--profile-mtp`; emitted N times per step, each pass wrapping a replay of `mtp.decoder_block`) | `tokens = total_len` of the pass |

The **`linear_attention`** category is keyed on `(prefill_tokens, n_decode)`,
and that key is what makes it regime-aware: a gated-DeltaNet block runs a
*different set of kernels* for a pure prefill, a pure decode and a mixed batch,
so a kernel simply has no rows for a regime it does not fire in, and the
simulator emits nothing for it there. The historical Qwen3.8-27B decomposition:

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

The selected packed layout replaces the GDN conv/scan/norm components with
`gdn_core`; projections and outer glue stay separate. Its size is packed qkvz+ba
input, gated-normalized value output, and owned conv/dt_bias/A_log/norm parameters;
persistent state capacity remains per sequence. Use model activation dtype (A_log
is float32), require positive TP-divisible heads, and reject unverified quantized
BA layouts. Sequence expansion is cached by TP and original sequence without
mutating the catalog or changing latency interpolation.

The `attention` category holds **more than one kernel** on a sparse model, and
they are not interchangeable — the lookup takes a layer name
(`attention_by_layer`) and so does the skew alpha. MiniMax-M3 profiles
`attention` (its non-sparse layers), `sparse_attention` and `indexer`;
DeepSeek/GLM profile `attention` (MLA) and `indexer`. There is deliberately no
pooled `tables["attention"]` shortcut: there was one, every lookup took it, and
a sparse layer got the dense kernel's latency (2.1x per layer on M3).

Names beyond this table are per-family and live in the catalogs
(`gdn_*`, `mla_*`, `indexer_*`, `sparse_*`, `mtp_*`, `*_glue`).
**Ordinary catalogs require `calculate_sizes` coverage in `memory_model.py`.**
Admitted native/contextual representations instead use their selected providers
in `component_sizes.py`; their logical trace bytes do not establish resident
capacity. Check every selected canonical name through its actual sizing path.
Catalog presence alone does not establish a valid shape or deployment.

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
  TP ALLREDUCE to `o_proj`/`down_proj` and declared row-output projections,
  caches selected component sequences per TP, swaps in PIM attention before the
  NPU attention kernel when offloading is enabled, and one-shot-warns when a
  layer is missing from the profile CSVs.
- `_emit_prologue()` / `_emit_pre_attn_layers()` / `_emit_post_attn_layers()` /
  `_emit_final_layers()` — thin wrappers over `_emit_sequence`.
- `_block_copy_key()` — the reuse key for a built block: `None` when it must be
  rebuilt (block mode, contextual profiles, or MoE with copying disabled),
  otherwise the layers' `LayerSpec`s and any site-specific resolved sequences.
  Build once per matching key and replay its rows. Deterministic BALANCED/RR
  routing is exact at a fixed batch shape; RAND copying reuses a draw and
  suppresses per-layer variation rather than automatically disabling reuse.
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
- `comm_type`: `NONE`, `ALLREDUCE`, `ALLGATHER`, `REDUCESCATTER`, or with
  dimension scoping `ALLREDUCE:1,0`, `ALLGATHER:0,1`, `ALLREDUCE:1,0,0` (the
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
linear over `(prefill_chunk, prefill_key, n_decode, kv_decode)`), and
`_lookup_moe` (2D over `(tokens, activated_experts)` within the table for
this run's `ep_total`, profiled at tp=1).
Lookups extrapolate **above** the top profiled value (time_us is linearly
extended); below the bottom they **clamp**. That asymmetry is only safe because
every axis sweeps a 0 or 1 point, so no runtime query can fall under it -- with
one exception, which is why `moe.csv` has an `ep` axis. See the MoE section.

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

Attention with skew correction: `_lookup_attention_with_skew` uses the
unchanged uniform mean/max lookups and resolves alpha via `_skew_alpha`.
Pure prefill, one decode and uniform decode bypass correction. For a
heterogeneous batch, both endpoints are needed to compute the lever even
when the selected alpha is zero. New calibration tables use a kernel/query/
prefill/lever partition and binary search over supported N anchors.
`_hydrate_skew_fit_tables` validates and loads the table once; no fitting,
measurement-neighbor interpolation or file access occurs per batch.
Enabled fits without a recognized calibration format are rejected with a
`profiler refit-skew` instruction. The current format is `runtime-skew-calibration`.
Disabled skew bundles, including RTX4090, bypass calibration unchanged.

Profile CSV path: `profiler/perf/<hardware>/<model>/<variant>/tp<N>/{dense,
per_sequence,attention,moe,skew,skew_fit}.csv` (resolved as
`../profiler/perf/...` from the `astra-sim/` working directory).

Variant resolution: `trace_generator.resolve_variant(model_config)` selects the
checkpoint-default folder and is a **pure function of the model config** —
it takes no dtype argument, because the simulator has no dtype
input. Weight dtype is `utils.config_weight_dtype` (`quantization_config`
first, then `torch_dtype` / `dtype`), KV dtype is `utils.config_kv_cache_dtype`
and appends a `-kv<short>` suffix when not `auto`. Runtime lookups verify the
resulting folder exists; a miss raises a clear `FileNotFoundError` pointing at
the missing variant. The profiler can still *write* other variants for the same
model (`--variant`, `--dtype`, `--kv-cache-dtype`), which is how a deliberate
second precision is measured and kept beside the first; the simulator just
never asks for one. Do not assume universal profiler/consumer name equality:
nested-only quantization, FP32 auto conversion and auto-promoted KV dtypes
retain compatibility-label limitations documented in the profiler running guide.

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
- Admission is **one step stale** under `--async-scheduling` (on by default, as
  vLLM's `scheduler_config.async_scheduling` is). vLLM's
  `max_concurrent_batches` is 2 at `pp_size` 1 with it on
  (`config/vllm.py`: "Async scheduling requires 2 concurrent batches to
  overlap"), and `EngineCore.step_with_batch_queue` composes batch k+1 right
  after submitting k, then blocks on k's future -- so the batch that executes
  next was built while the previous one was still on the GPU, and a request
  arriving after that cannot join it. `_arrival_cutoff` reproduces it by lagging
  phase B's arrival horizon to the completed batch's own `batch_time`. Working
  through vLLM's loop, a request arriving at `a` during batch A lands in C, not
  B, so `TTFT = (t_A_end - a) + dt_B + dt_C` on both sides.
  The lag is dropped when the instance was **not** busy right up to `current`
  (an idle engine's batch queue is empty and vLLM admits immediately), and
  `schedule()` retries phase B with the full horizon when the lagged pass
  admitted nothing. That retry is load-bearing, not cosmetic: the main loop
  advances the clock only for arrivals still pending in the *router*, so a
  request already sitting in `waiting` behind a stale cutoff would be invisible
  forever and the run would answer "pass" at a clock that never moves.
  At `pp_size > 1`, this additional lag applies only when the scheduler's
  `_uses_v2_model_runner` classification is true; the modeled V1 pipeline
  already supplies its in-flight window. The classification currently treats
  non-MoE, non-hybrid stacks as V2 and is separate from `cudagraph.runner`.
  Admission and TTFT effects depend on workload and batch composition.
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
  calibrate `mem_util` so each instance's block count matches its observed engine
  pool in the bench run's `meta.json` before comparing latency at all.** New runs
  record `kv_cache.capacity.per_engine`; require its complete status rather than
  interpreting a DP client aggregate as one pool. Historical single-engine
  references retain their recorded `kv_cache.num_gpu_blocks`. The RTX 4090
  example matches 2,588 blocks with `mem_util: 0.833919`. The refreshed Qwen3-30B
  example matches 38,256 blocks per DP rank; Llama and Qwen3-32B retain the
  recorded utilization without claiming a complete per-engine capacity match.
  Inspect preemption and request scheduling before attributing latency error
  to capacity; heartbeat occupancy percentages have different cache accounting
  from vLLM and do not establish equivalent pools.
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
- **The draft count is per *batch*, not per request.** `add_done` reads it from
  `Batch.spec_scheduled`, snapshotted at build time exactly as
  `Batch.scheduled_tokens` is. Left on the request it is rewritten by the next
  `schedule()` while the batch is still in flight — to 0, since the request is
  no longer caught up — so the rollback never runs and `num_computed_tokens`
  stays `N - 1` past `num_tokens_reached`, which no later step can schedule.
  Invisible at `pp_size == 1`, where an instance holds one batch at a time

The verification forward needs the **fifth attention axis**, `decode_q_len`
(query tokens per decode sequence, `1 + N`). It is opt-in in the profiler
(`--attention-decode-q-lens`, default `1`) because it multiplies the grid. An
unprofiled value falls back to the nearest with a one-shot warning rather than
interpolating: query length changes the kernel's tile shape, not just its size,
and unlike the other four axes there is no measurement saying a value between
two profiled ones lies between their costs.

Being opt-in meant it had **never been run**, and the grid's feasibility
filters sat outside the `decode_q_len` loop — so every one of them counted a
decode request as a single token. Right at `q = 1`, wrong above it: a shot at
`chunk = max_num_batched_tokens` whose `n_dec * q` ran past `max_num_seqs`
passed the filter and overflowed vLLM's own buffer, three and a half hours into
a sweep, as `operands could not be broadcast together with shapes (2304,)
(2432,) (2304,)` — 2304 being `MNBT + MSQ` and 2432 the real token count,
`2048 + 64*6`. The three token-based bounds (the combined sum, a decode's own
sequence length, and the KV block budget) are inside the loop now and take `q`.
At `q = 1` every one is identical to what it was, so committed grids still line
up shot for shot with a re-profile.

Two shapes of the grid are worth knowing when planning a run:

- **Query coverage.** Each requested query length adds decode-containing shots;
  pure prefill is emitted once at `q=1`, even if `1` is not requested. Separate
  sweeps can therefore overlap on prefill rows. Prefer a combined acquisition;
  independent runs need separate output roots, an explicit common KV bound and
  compatible identities before any merge. `decode_q_len` remains part of the row
  key. FULL-prefill anchors union the target configurations' capture sizes.
- **The reach.** `--attention-max-kv` defaults to the model's own context,
  `max_model_len - max(decode_q_len) - 1` — not `max_model_len`, which gets
  the top point filtered (a decode occupies `kv + q` positions and needs one
  more to be a decode) and stops the sweep a doubling short. Cost is close to
  linear in the number of kv values, since the KV-budget filter prunes the
  large-kv x large-n_decode corner: 8,643 shots at 16,384 against 14,653 at
  DeepSeek-V3.2's full 163,834.

**Reach matters more on a sparse model than a dense one.** The simulator
extrapolates linearly past the top profiled kv, which is right for decode
attention — a pure KV read fitting `a + b*(n_decode*kv_decode)` at R^2=1.0000 —
and wrong once selection is involved. Measured on DeepSeek-V3.2 at
`n_decode = 8`, `attention` is 342 us at `kv = 2048` and 350 us at 16,384, flat
from `index_topk` onward because past that it only reads the selected tokens,
while `indexer` goes 52 -> 101 us because it scores the whole KV to make the
selection. The unprofiled region is exactly where a long-context run's cost
lives.

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

Draft **time** is charged: `_emit_drafter` emits N passes after the target's
head (the drafter runs from `sample_tokens()`, after the target has sampled),
each one `mtp.prologue` → a replay of `mtp.decoder_block` → `mtp.head`. The
first pass reuses the target's own token layout and the rest are pure decode at
one query per sequence, which is what `llm_base_proposer.py` does — reusing the
target's shape for all N charged a prefill chunk N times over.

`_require_drafter_cost` still **refuses** a speculative run on a model with MTP
modules whose catalog is missing `mtp.prologue` or `mtp.decoder_block`, rather
than reporting a speedup no engine can deliver; the wrapper has to come from a
live profile dump like every other block. A model with no MTP modules drafts
externally (a second model, or n-gram); that is a serving choice rather than a
checkpoint property, so it warns instead of refusing.

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
| KV cache | `quantization_config.kv_cache_scheme` / `kv_cache_quant_algo` → fp8, otherwise `torch_dtype` / `dtype` (the *activation* dtype on a weight-only quantized checkpoint) | `attention.py:281` promotes a declared scheme when the flag is `auto`; otherwise `auto` is `model_config.dtype` (`platforms/interface.py`), so DeepSeek-R1's fp8 weights keep a bf16 cache. Sparse-MLA models (`index_topk`) keep the weight dtype: vLLM picks their format per backend (`fp8_ds_mla` on SM120) |
| mamba conv state | `mamba_cache_dtype`, `auto` → weight dtype | `mamba_utils.py::_mamba_state_dtype` |
| mamba recurrent state | `mamba_ssm_dtype`, `auto` → **conv** dtype (not the weight dtype) | same, plus `models/config.py::Qwen3_5ForConditionalGenerationConfig` bridging the HF field |
| sparse-indexer side cache | fixed by the model | DeepSeek/GLM `torch.uint8`, M3 `resolve_indexer_kv_dtype("bf16")` — neither follows the KV dtype |

Note the weight row is the profiler's **variant folder name**, not vLLM's
`model_config.dtype`: vLLM calls DeepSeek-V3.2 bfloat16 and keeps fp8 in the
quant method, while the folder has to encode the quantization or two bundles
collide. That divergence is deliberate; the other four match vLLM exactly.

Profiler `--dtype auto` uses the omitted-dtype naming path. An explicit BF16/FP16
matching a quantized checkpoint's declared activation dtype keeps the quantization
base; another activation dtype gets `-act<dtype>` (for example `fp8-actfp16`).
Explicit FP32 is not an auto match: vLLM may downcast an FP32 checkpoint.
HF precision-field or any nested `text_config` override requires `--variant`;
ambiguous dtype aliases or nested-only quantization with explicit `--dtype`
also require an intentional label. These names never change engine quantization.
Serving still selects only the checkpoint-default variant. Do not rename or
restamp old acquisitions to represent a new precision or measurement identity.

To simulate a different precision, **profile it** — the profiler's flags write
a separate bundle and the simulator reads the one the checkpoint names.

### CLI argument conventions
CLI flags follow vLLM naming where applicable:
- `--skip-prefill` — legacy first-token bookkeeping flag; it does not skip
  prompt computation or prepare cached KV and is not a decode-only generator
- `--request-routing-policy` (`LOAD`, `RR`, `RAND`, `CUSTOM`) — request routing across instances
- `--expert-routing-policy` (`BALANCED`, `RR`, `RAND`, `CUSTOM`) — expert token routing for MoE.
  `CUSTOM` reads the measured distinct-expert count from `--gate-stats` rather than
  deriving it from a uniform gate; see the gate-stats section
  (block-copy optimization is controlled separately via `--enable-block-copy`, default on)
- Reversible boolean flags use `argparse.BooleanOptionalAction` (e.g.,
  `--enable-prefix-caching` / `--no-enable-prefix-caching`). Opt-in `store_true`
  switches have no automatic negative form; document the actual parser.

Before release, compare each public entry point's registered arguments with
its directory README index and complete website reference, including negative
boolean forms, defaults, choices and subcommand restrictions. Check wrapper
forwarding without starting engines. Distinguish parser defaults from resolved
engine values; an unset profiler block size is chosen by vLLM, while serving
loads the profiled per-TP value and uses 16 only as a metadata fallback.

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

MiniMax-M3's canonical `indexer` is resolved before the separate DeepSeek
formula. Its index query and key-only cache use their independent BF16 dtype;
its output is int32 `[tokens, local index heads, topk blocks]`. The trace passes
exact prefill and decode cached histories through `kv_lens` for M3 `indexer`
and `sparse_attention` only: the indexer sums them whole, while sparse attention
sums `min(history, sparse_topk_blocks * sparse_block_size)` per request.
Do not reconstruct those histories from a floored decode mean or cap their
sum once. These are logical query plus preexisting-cache bytes, not kernel
DRAM transactions, workspace, capacity or causal-prefill read multiplicity.
The existing byte cap and historical lookup cap remain separate; this sizing
rule does not establish arbitrary runtime indexer dtypes or whole-model TP
correctness. Unknown indexer layouts and malformed M3 geometry are refused.

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
join it. It is counted in **pages**, not bytes — see the linear-attention
section above for why, and for the `none`/`align` page counts. 153.9 MB per
sequence on Qwen3.8-27B, i.e. 6 blocks per request with prefix caching on at
`block_size 784` — it bounds concurrency the way a KV cache bounds context, and
leaving it out lets the simulator admit requests vLLM could not.

Sizing DeepSeek-V3.2 as GQA read 1,748,992 bytes/token where MLA caches 78,324.

**Verify a new family's shapes against its published parameter count.** It is
the one number that catches a wrong shape anywhere in the stack, and it is
public. Summing `calculate_sizes`' weights over the catalog and the resolved
stack: DeepSeek-V3.2-Exp comes to 671.878B, and minus
the DSA indexer (0.852B) that is 671.026B — V3's published 671B, with the
difference being exactly what V3.2 adds.

### Model configs
Model configs live in `configs/model/{org}/{model}.json` and must retain the
fields and structure required by the checkpoint's native configuration class,
not just the fields read by simulator sizing. MiniMax-M3 and Kimi-K3 preserve
their nested text wrappers; the shipped Qwen text tower uses its native flat
text configuration. Do not flatten or prune a different family by analogy.

The simulator loads these via `get_config(model_name)` in `utils.py`.
`get_source_config(model_name)` retains the original wrapper for independent
native-profile validation. Preserve its cache boundary: the flattened simulator
config is a separate deep copy, not a mutation of the original source object.

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
- `link_bw`: network link bandwidth in GiB/s (set in network.yml)
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
(an all-to-all -- see the MoE section for how it is emitted). DP is achieved via
multiple instances with the same `dp_group`.

`config_builder.py` reads the cluster config and generates three ASTRA-Sim input files:
- `astra-sim/inputs/network/network.yml` — topology and bandwidth
- `astra-sim/inputs/system/system.json` — scheduling policy and memory bandwidth
- `astra-sim/inputs/memory/memory_expansion.json` — remote (CPU) memory config

### Working directory
Run `python -m serving` from the repository root. The entry point changes cwd
to `astra-sim/` internally, but relative cluster, dataset, output and gate-stat
paths are handled relative to the invocation root. Use repository-relative
dataset paths: that loader prepends `../` even to an absolute input.
An explicit relative `--inputs-root` instead resolves from `astra-sim/`;
use an absolute path when selecting a separate intermediate directory.

### Communication sizes for ASTRA-Sim
ASTRA-Sim's data-size convention depends on the collective: AllReduce and
ReduceScatter take the full pre-partition tensor; AllGather takes a local chunk.
Ring divides the former by its rank count, not the AllGather chunk.
- ALLREDUCE on `o_proj` and `down_proj`: pass full output tensor size at
  activation dtype. An FP8-weight checkpoint still reduces its bf16 output
- Dense `tp_output_reduction: all_reduce` entries reduce their full projection
  output at activation dtype, not the sharded input. For Qwen `gdn_out_proj`,
  that is effective token rows × hidden size × activation bytes.
  No reduction at TP1; the declaration applies only when the row is emitted.
  Historical and packed GDN layouts both use this declaration.
- MoE: **not** one size. Dispatch AllGathers take per-rank chunks, while the
  combine ReduceScatter takes a pre-scatter total. Native dispatch has three
  tensor collectives; the legacy path combines its dispatch payload. See the MoE section.

An MoE entry's `attention_sequence_parallel: true` requires `sequence_parallel: true`.
Serving refuses its TP>1, PP1, DP+EP-spanning deployment before trace emission:
attention-wide sequence shards need different gathers/scatters and local row counts,
not a wrapper-only restore. The guard is not certification of every other topology.

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

The `involved_dim` is encoded in the trace `comm_type` field as `ALLGATHER:0,1` (parsed by
the Chakra converter's `_parse_comm_type`). ASTRA-Sim's `Workload::issue_comm()` reads this
and passes it to `generate_all_to_all()`, which skips dimensions where `involved_dim` is false.

The `system.json` collective implementations must have one entry per topology dimension
(e.g., `"all-to-all-implementation": ["ring", "ring"]` for 2D, three entries for 3D).
`config_builder.py` generates this automatically from the topology it emitted.

### Group-limited expert routing
RR uses each token's ordinal in the full gathered/replicated batch, not a
constant zero or a counter restarted at each source partition. Unrestricted
RR chooses `(t + j) % E` for `j = 0..k-1`; grouped RR uses the same advancing
ordinal to select groups and experts. RR resets at each layer invocation,
so block copy is exact for it at a fixed batch shape. RAND draws and the
BALANCED/CUSTOM analytical paths are independent of this ordinal fix.

DeepSeek-V3/V3.2 and GLM restrict a token's experts to `topk_group` of
`n_group` groups (`deepseek_v2.py` passes `num_expert_group=config.n_group`,
`topk_group=config.topk_group`, both defaulting to 1). `GateRouter` reads both
off the checkpoint. It matters because it changes how many EP ranks one token
reaches, and therefore the per-rank MoE token count and the EP collective size:
DeepSeek-V3.2 at EP=8 sends a token to 45.4% of ranks against 66.2%
unrestricted (GLM-5's figure, same `E` and top-`k`). **Only DeepSeek-V3.2 actually restricts** — GLM-5 ships
`n_group: 1` and no other family declares the fields.

`_hit_probs` is exact and deterministic: a DP over which groups the token
selected, not sampling. It draws **without replacement**, matching
`torch.topk`; the older `1 - ((ep-1)/ep)**k` modelled independent draws and
read ~1% low even with no grouping. Don't "simplify" it back — the difference
is measurable, and the grouped and ungrouped cases must not have two different
answers to one question.

### Distinct expert counts and measured routing

Whole-block MoE lookup distinguishes gathered tokens, expert-token assignments
and distinct activated experts. With independent tokens choosing uniform top-k
experts, the expected distinct count per EP rank is:

```text
activated_per_rank = (E / ep) * (1 - ((E - k) / E) ** n)
```

At one token the global count is k and the expected per-rank count is k/ep;
the latter is not an integer placement for a particular token. Runtime counts
are rounded and bounded for lookup. BALANCED is an expectation, not a measured
routing histogram. Do not replace distinct counts with n*k/ep: repeated expert
selections collide, so assignments and distinct weights have different scaling.

`bench run --record-gate-stats --enforce-eager` records a trained gate's counts.
`bench/core/gate_stats.py` reduces them into `gate_stats.json`, consumed through
`--expert-routing-policy CUSTOM --gate-stats`. The curve describes the recorded
weights, inputs and batching, not a universal correction for that architecture.

- Interpolate between recorded batch sizes and clamp at the endpoints. The
  measured count is global; the balanced per-rank estimate divides it by EP.
- Check expert count, top-k and model basename before consuming a curve. The
  basename check is a typo guard, not full checkpoint identity verification.
  Missing, unreadable or mismatched curves warn and fall back to BALANCED.
- Aggregate only between explicit workload start/end markers. Concentrated
  routing with `distinct <= top_k` is valid and must not be classified as
  warmup. Reject unmarked, repeated-boundary and incomplete raw logs.
- The observer requires eager execution and synchronization. Mark its latency
  diagnostic and compare serving latency against a separate uninstrumented run.
  Recording does not change the default routing policy.
- Forced routing can change expert-load histograms and grouped-GEMM kernel
  selection. It is not a control that holds everything except the distinct
  count fixed, and end-to-end agreement cannot bound an individual MoE table's
  error without separating other costs.
- Historical whole-block measurements and native components have different
  contracts. Neither a measured count curve nor matching CSV columns establish
  arbitrary-histogram or deployment coverage.

### MoE dispatch/combine and TP restoration

The modeled transport is vLLM's `allgather_reducescatter` backend.
A matching native component table is preferred; otherwise the retained
whole-block `moe.csv` path is used with a warning for unverified DP coverage.
Other all-to-all backends need separate execution contracts.

Native DP+EP modular execution keeps local gate/routing, three ordered AllGathers
(hidden states, top-k weights, top-k IDs), gathered expert computation,
ReduceScatter and local finalization separate. Tensor widths follow the
recorded dtypes. Sequence-parallel wrappers dispatch over EP and restore the
TP output with AllGather; non-SP wrappers dispatch over DP and restore it
with AllReduce when TP is greater than one. Do not describe this path as a
single hidden-state/router-logit payload.

The legacy fallback approximates dispatch with one hidden-state/router-logit
AllGather and combine with ReduceScatter. At DP=1 there is no dispatch/combine:
replicated input reaches local experts, followed by a TP AllReduce if needed.
Sequence-parallel legacy traces also carry their trailing TP AllGather.
Chakra accepts multiple collective/size pairs on an expert marker and chains
them in order, preserving the tail through skipped expert ranks. Reinstall
the converter with the matching backend; the TP restore is not a missing feature.

For unequal DP counts, retain the full already-padded vector. SP wrappers
ceil-divide each member by TP before gathering; the total and minimum alone
cannot reconstruct that geometry. The Ring envelope uses the gathered row
count minus the smallest dispatch count. For a group of G ranks and tensor
width W bytes, AllGather uses ceil(remote_rows * W / (G - 1)) as its local
chunk; ReduceScatter uses the hidden-state chunk multiplied by G. Equal counts
recover the ordinary local chunk. This is an analytical worst-rank envelope,
not an exact model of grouped ragged NCCL execution or a measured guarantee
for every rank count.

Expert lookup uses all gathered/replicated rows and the appropriate rank's
active experts, not the number of unique tokens with a local expert from
`RoutingResult.local_tokens`. Local gate/finalization and gathered expert
work must keep separate row domains on the native path. Attention/request
geometry and head rows remain separate from model-forward padding.

### Native local and DP+EP component profiles

MoE acquisition defaults to native DP1 components; `--dp` selects other degrees.
Use `tp_degrees`, `dp_degrees` and `ep_degrees` for internal settings, with
`--tp`, `--dp`, `--ep` and corresponding uppercase shell variables. Optional
EP degrees filter the supported TP/DP targets; they never change expert sharding
or infer additional TP/DP degrees. Omitted EP keeps every target. Reject unmatched
EP degrees before any engine starts; a slice matches only `--tp-refresh`.
The filter affects only MoE components, not ordinary categories at the other TPs.
The removed whole-block path no longer shrinks global expert count/top-k.
The target is full EP=TP*DP with
DP>=1, independently of the one-GPU acquisition process. `MoeTarget` is shared
with serving: local gate/finalization rows and gathered expert rows differ,
and sequence-parallel wrappers require per-member ceil division before gathering.
Pass the full already-padded DP vector through both wave-completion paths; its
sum/minimum cannot reconstruct the sequence-parallel geometry.

The unquantized modular DP1 path is non-SP. Select its backend from the resolved target,
not the DP+EP selection: native priorities can differ. NoDPEP preparation must
preserve the input and return no scale or token metadata; with a NoOP reduction,
experts write directly to the caller's output alias. Observe no CUDA work in
prepare/finalize before recording `local_execution` and omit `finalize_copy`
entirely. Preserve the alias in every conditioned eager variant. DP1
traces contain gate/routing, experts and optional TP AllReduce, no DP collectives.
Full EP at TP>1 owns whole experts and is not plain tensor-sharded TP. Missing
DP1 native coverage falls back only when a historical whole-block table exists,
without consuming a mismatched EP component table. Match the legacy lookup's TP
choice (TP1 when present, otherwise its first available degree); a fresh bundle
without that table requires a complete matching component acquisition.
Cache missing deployment coverage for the lifetime of the simulation, just like
loaded native tables: bundles are immutable during a run. Do not reread the same
metadata on every layer. Corrupt contracts remain errors, never cached misses.
Initialize `native_moe_index_present` and `native_moe_components` on the persistent
perf DB before creating per-batch shallow copies. Otherwise the copies discard
new cache entries and rebuild metadata and interpolation support every batch.
Keep TP-specific attention caps on each view and native deployment keys separate;
cache reuse changes host work, not measured latency or routing.

Read `FusedMoEConfig.use_deferred_moe_finalize` as a strict boolean effective
capability. `defer_moe_finalize` is a request flag in 0.28 and a request method
in 0.31, not a stable execution-mode predicate. Reject an unknown/non-boolean
API; never invoke the request method or silently assume finalization is local.
Check actual native APIs, geometry, dtypes, backend and conditioning semantics;
source-file hashes and exact versions are not acquisition admission or grouping
keys. Do not restamp historical measurements after a semantic adapter change.

The separate live adapter admits actual APIs, standard token layouts and native
capabilities at TP1/DP1/EP1. Preserve processed weights/scales, native backend,
routing and activation parameters; do not rebuild an unquantized approximation.
Its `gate_routing` includes external routing, or projection only when routing is
inside the expert kernel. `experts` includes native preparation and finalization.
`wrapper` executes the native parent and accounts for shared experts, latent
transforms, normalization, scaling and combination outside the excluded gate/router
and routed-kernel scopes. Use launch ownership and conservation checks, not
subtraction of unrelated profile captures. Sum unique CUDA activity durations;
this is not an interval union or elapsed critical path across concurrent branches.
Compare every routing stimulus and each cloned bank exactly with the native method
outside timing and verify native state remains unchanged.
These guards compare parameter bindings/layouts, available version counters,
effective configuration and callable identities, not all device-memory contents.
Do not describe them as bytewise weight checks or arbitrary CUDA-write detection.

Keep gate dtype/width, model width and physical expert width explicit. Count unique
native weight/scale storage separately from all cloned bank storage; preserve
storage aliases, offsets and strides. Bank allocation uses the actual processed
bank footprint plus native workspace/free-memory reserve, not the unquantized
conversion-buffer heuristic. Missing or unsupported capability statements are
errors, including deferred, input-weighted, fused-shared-gate, biased/redundant,
LoRA, EPLB and padded-hidden execution. Monolithic routing requires native replay
evidence; a matching model name is not an admission.

The isolated MoE engine uses the MLP prefix and declared native-profile settings.
Its private `kv_cache_memory_bytes` override defaults to 1 GiB and accepts only a
positive integer. The internal caller may raise this resource cap, but never
reduce requested token/sequence/context limits to fit it. vLLM must validate native
capacity. Record resolved component-engine settings only in MoE observations,
never as main-engine cache facts or replacement target graph policy. Read nested
source geometry through `text_config`; local Kimi DP1/EP1 selectors are admitted,
distributed selectors remain refused. Preserve DSA/Kimi's duration-only consumers
and their separate capacity/serving restrictions.

Root `per_tp[tp].moe_components` selects deployment entries containing semantic
contracts, observations, coverage and CSV checksums. Paths locate tables; folder
names do not establish contract identity. No new index/contract/coverage JSON
sidecars are written. `moe_components` metadata is separately owned; a MoE refresh must not
restamp attention/skew or the main engine. Individual CUDA activities are counted
once per forward during eager measurements. No CPU duration is added.
No-op resume preserves original runtime observations. Record distinct later
`acquisition_contexts` only after validating new work; this list is not a count
of durable or completed measurement rows.
Expert measurements rotate positions and L2-sized weight banks, compare B/2B
under identical complete-cycle work, independently verify eager outputs and
require the recorded stability gate before publication. Those rules condition
the measurement; they do not prove full-model cache equivalence.
Publish only `coverage.conditioning_summary` with `checked_points`,
`failed_points` and `max_relative_change`; retain the tolerance in
`contract.expert_conditioning`. Individual control points and timings belong to
raw resume checkpoints, not published metadata. Preserve runtime observations,
geometry, dtypes, parallelism, backend, completion and the CSV integrity check.

New plans use `--moe-grid-factor` (default 2) for input rows N, active local
experts A and local assignments M, including feasible endpoints, envelope
corners and balanced-total anchors. At EP1 M=N*k collapses the assignment axis.
`expert_sampling` belongs in the root contract; CSV columns remain unchanged.
Target serving capture tokens G take precedence on token axes after
`target.token_domains((G,) * target.dp)` maps them to local and expert rows.
Use geometric points only in uncovered token ranges, retaining endpoints and
feasibility corners. Decode-only/NONE policies require the full eager range;
an out-of-capture DP peer can require small local eager rows on every member.
Include MRV1 compiler-SP rounding in physical bounds before native initialization.
Capture sizes never determine A or M. Keep transformed anchors, logical budget
and eager ranges in `expert_sampling`; source/target information belongs in
`sampling_graph`. Absent range metadata retains an existing acquisition plan;
a changed semantic plan requires a fresh output root.
New acquisition must not create CUDA graphs; eager timing and serving-padded
coordinates are distinct.

Serving automatically consumes matching tables for the supported modular
local/DP+EP or separately admitted live local path. Local components use token interpolation.
New experts use physical N/A/M interpolation: construct affine-rank-reduced
simplices at table load, then use nonnegative barycentric weights and bounded
query caches. Do not normalize by query-dependent feasible ranges: doing so
can reduce assignment work during interpolation. Check feasible geometry and
measured support; never extrapolate. Historical balanced-total tables retain
their original interpolation and graph/eager selection. The routing policy
supplies A/M; BALANCED and measured active-count CUSTOM use rounded expectations,
RR/RAND use selected pairs and native rank ownership. Within-rank imbalance is
not a fourth measured axis. Missing deployment/policy coverage warns and retains the
legacy EP table only when it exists; corruption and out-of-support queries are
errors. Shared, quantized and supported monolithic local execution require the
live contract, never guessed aliases. Deferred and tensor-sharded execution remain
unsupported. Keep historical acquisition
fingerprints as provenance; do not relabel old samples after changing a method.

DP+EP trace order is gate/routing, hidden/top-k-weights/top-k-IDs AllGathers, experts,
ReduceScatter, finalize-copy, then the wrapper's optional TP AllGather (SP) or
AllReduce (non-SP). Chakra must chain every marker collective and retain the
last dependency through ranks it skips. The ragged Ring envelope and independent
logical collective costs remain approximations of grouped NCCL execution.
See `docs/docs/profiler/native-moe-components.md` for flags and exact limits.

Shipped native data is the runtime subset: index, contract, coverage and CSV.
Its acquisition fingerprints remain historical provenance, not hashes to rewrite
when packaging or changing unrelated code. Original per-forward streams are
required for acquisition resume; use a separate output root when remeasuring a
published bundle. Treat the indexed target and contract bounds as authoritative;
the shipped Qwen3-30B TP1/DP2/EP2 data does not imply TP2 or another model is covered.

Hardware refreshes and historical examples have different ownership. The shipped
RTXPRO6000 operation fits retain the previously calibrated common latency and
refit bandwidth from NCCL primitives with the backend's actual formula. Record
that constraint and residuals; do not relabel it as an unconstrained automatic
fit. If retaining an older multi-GPU reference, explicitly retain its matching
common link and disable operation-map inheritance. The current Qwen examples
instead use refreshed NCCL references and inherit the shared operation defaults.
Refresh benchmark outputs, summaries and public plots together.

### The legacy MoE grid is per EP degree

Historical whole-block acquisition used a per-EP token grid. That acquisition
path has been removed; only its stored `moe.csv` lookup remains until remeasurement.
The following describes those historical measurements and their fallback, not the
native full-top-k component contract above. A reduced-top-k checkpoint is an
approximation and must not be presented as the native distributed kernel.
An EP rank does not run the model's MoE block; it runs a **slice** of it, and
the slice differs three ways at once: it permutes over `E/ep` local experts
rather than all E, a token contributes `k/ep` of its k expert assignments
rather than all k, and so the distinct experts it activates start at `k/ep`
rather than at k. Profiling at ep=1 and charging that per rank gets all three
wrong in the same direction.

The third is the one that bit. `activated_experts` is the only axis in the repo
whose minimum is a positive number the runtime can go under -- it floors at
`top_k`, because a token cannot activate fewer experts than it selects, and the
profiler's `ExpertRoute.forge` refuses to build one that does. The runtime goes
under it constantly: `_balanced_route_ep` gives a decode step
`activated = min(round(total_len * k / ep), E/ep)`, which is 1 at ep=8, and
**98.4% of MoE lookups in a GLM-5 EP=2 run asked for 4 against a floor of 8**.
Clamped, the MoE term reads 1.7x to 4.8x over at the decode operating point:

| model, one token on the rank | ep=1 (what was charged) | ep=8 (the real slice) |
|---|---|---|
| DeepSeek-V3.2 | 84.7 us | **28.1 us** |
| GLM-5 | 162.6 us | **33.7 us** |
| MiniMax-M3 | 139.5 us | **47.3 us** |
| Qwen3-30B-A3B | 47.8 us | **27.8 us** |

**Extrapolating instead of measuring would have over-corrected.** Fitting
`fixed + per_expert * a` through Qwen3-30B-A3B's own grid predicts 16.8 us at
`a=4` where the ep=2 slice measures 36.2 us -- 2.2x low, because the fixed
permute cost does not vanish with the expert count. The curve saturates from
ep=8 on (28.1 / 27.5 / 27.7 us on DeepSeek) for the same reason: once `k/ep`
floors at 1 only one expert plus that fixed cost is left.

These are historical reduced-expert measurements, not the current acquisition
plan or runtime estimate. `--moe-ep-degrees` and its config-slicing path were
removed. New component sweeps use `--tp`, `--dp`, optional `--ep` and unchanged global top-k;
unsupported native adapters fail rather than falling back to old acquisition.

The simulator picks the table for the instance's **total** EP degree and falls
back to the nearest profiled one with a one-shot warning. Not interpolated, for
the same reason `decode_q_len` is not: `E/ep` has to be a whole number of
experts and the permute width is a staircase in it. A bundle with no `ep`
column reads as ep=1 and prices exactly as it did before the axis existed.

### Reduced checkpoints and profile reuse

A reduced checkpoint is a diagnostic configuration, not validation of the
full-size trained model. The stored DeepSeek-V3.2-Exp-16L64E config reduces
layer and expert counts while retaining MLA, sparse indexing, grouped routing,
dense/MoE layer composition and MTP. Dummy loading still allocates its weights.

Reusing `dense`, `per_sequence`, `attention` or `mtp` rows requires matching
kernel shapes, dtypes, backend and acquisition geometry. Record inherited data
with `derived_from` metadata rather than representing it as fresh acquisition.
Reduced-stack profiling is an approximation of full-model execution and cache
state; model depth alone is not evidence that every execution property matches.

Expert count changes require a new `moe` table. Even at the same token,
top-k and active-expert coordinates, routing and permutation operate over a
different expert domain and can select different kernels. Native component
contracts also retain model/deployment identity and must not be relabelled.

### MoE expert blocks
Routing vectors use global EP ranks. Each DP member's trace uses
`global_rank = dp_rank * local_ep + local_rank`, where `dp_rank` is its
position in the DP group, never its global instance ID. Both wave-completion
paths must pass that position through ordinary and interleaved synthesis.
PP does not enter this index: each stage has its own EP group. Keep
`EXPERT {local_rank}` markers local, because the converter selects them
within an instance. Equal BALANCED vectors can hide an incorrect offset;
asymmetric routing is required to check rank selection.

Expert blocks use `EXPERT {i}` / `EXPERT END` markers for ASTRA-Sim. Each EP rank
gets a latency from the gathered/replicated token count and its own activated
experts (`key_0=tokens, key_1=activated_experts`, profiled at tp=1). Ranks execute
in parallel and sync at the dispatch/combine collectives. Expert-to-rank assignment uses even
partitioning: `expert_id * ep_size // num_experts`.

### DP+EP wave synchronization
For DP groups (instances with the same `dp_group`), wave synchronization is achieved
through two mechanisms:
1. **Python-side dp_pending barrier**: trace generation is deferred until all DP group
   members have scheduled their batches, and the EP collectives are sized from
   the round's gathered total. `dp_pending[dg][inst]` is a
   **FIFO**, not one slot: at `pp_size > 1` a member can have up to `pp_size` batches
   waiting, and a round pairs the members' *j*-th batches, mirroring vLLM, where DP
   rank A's *j*-th forward joins the same collective as rank B's *j*-th
2. **ASTRA-Sim collective barrier**: all DP group instances' `.et` files are placed in a
   shared workload folder. The EP collectives in both files have matching stream
   IDs, causing ASTRA-Sim to block until both NPUs reach the collective.

**Local graph padding precedes DP synchronization; it also applies at DP=1.**
`serving/core/cudagraph.py` resolves the target MRV1 capture grid and local
FULL/PIECEWISE/NONE mode by default, or MRV2 when selected explicitly or by a
matching serving observation. Both DP completion paths and the independent-instance
path share that resolver. Small prefills can use PIECEWISE graphs. The common
DP mode is the minimum of the local modes: non-NONE broadcasts the largest
locally padded size; NONE retains each local padded count. With an 8-token
capture point and maximum 256, `[6, 1529]` therefore becomes `[8, 1529]`.
Do not replace this with a threshold-only `max <= cap` rule.
MRV2 instead returns to raw local token counts when any DP rank selects NONE;
otherwise it redispatches using the common token maximum and decode eligibility.
Its FULL decode candidate sizes are distinct from PIECEWISE token sizes. A first
one-token prefill is not decode-eligible; a one-token extension over cached history
can be. Preserve historical MRV1 behavior when no target runner was recorded.

The per-instance `cudagraph` object describes deployment behavior independently
of the profiler's eager settings. Defaults derive the grid from scheduler
limits, speculative query length and hardware compute capability; explicit
mode/grid/cap, runner and compiler-SP overrides describe other targets. Use the effective
mode after attention-backend resolution. LoRA keys, cascade/encoder restrictions,
DBO, dynamic/variable speculative decoding and drafter graph dispatch are not inferred. Attention offloading is
outside this GPU graph contract and uses NONE.

Bench startup may record a read-only worker `execution` observation after graph
capture, separately from the frontend's resolved configuration. Unsupported or
incomplete observations remain unknown. Preserve complete capture lists even when
they exceed generic diagnostic list limits. Collect every DP engine's response;
the public load-balancing RPC may return only the first engine's result.
`profiler import-execution` can save
a matching observation under `meta.yaml.serving_execution`; require matching
model configuration, device, precision and deployment scope. Resolve local
declarations through the worker's native config class
before binding aliases/defaults to effective model fields; do not maintain an
offline model-specific config alias table. Keep only the matched declaration,
not a duplicate expanded config. A custom variant label needs explicit matching
acquisition precision. Keep one observation
per scope, not an execution history. Explicit instance `cudagraph` settings win;
otherwise serving uses a matching observation or derives settings with a warning
when saved observations do not match. Versions are provenance, not execution locks.
Acquisition metadata stays separate: importing graph settings neither changes
latency CSVs nor certifies eager measurements as graph measurements. Reduced-stack
diagnostics must not become full-model serving references.

Padding changes model-forward rows, not requests, query/KV lists, decode counts
or head rows. Keep those domains separate in trace lookup and completion
accounting. This is a shape rule, never a benchmark-fitted graph-time correction.
This does not mean vLLM always executes attention without padding. MRV1 sets
`pad_attn` for FULL graphs and passes padded token/request dimensions into
attention metadata; PIECEWISE keeps real dimensions there. Padded request
query boundaries repeat, sequence lengths are zero, and KV slots are invalid.
QKV/norm/RoPE, output projections and MLP/MoE still see padded forward tensors;
selected logits rows do not. Serving retains real attention geometry and does
not separately model FULL-graph empty-slot/backend scheduling overhead.

For non-speculative execution, `_build_batch_ctx` uses the number of actual
requests for per-sequence head lookup, tensor sizes and TP logits gathering.
vLLM selects `hidden_states[logits_indices]` outside the padded model forward;
an idle DP `_dummy_run` returns hidden states without logits or sampling.
`_emit_final_layers` therefore retains the final norm but skips per-sequence
head entries on idle ranks, including their power costs. Keep a zero-byte
REMOTE output on the final trace row so Chakra can emit its terminal store
without transferring nonexistent token IDs. Speculative execution retains its
existing head/drafter row contract: changing it independently can skip
DP-synchronized drafter collectives.
Batch-context construction participates in the skew lookup fingerprint. Refit
enabled bundles through `profiler refit-skew` when it changes; a head-only fix
can refresh the reference identity without changing any numerical CSV values.
Never silence the stale-fit guard by manually replacing a fingerprint.

Idle-head omission also requires communicator-scoped collective numbering.
When Chakra supplies `involved_dim` rather than an explicit communicator,
ASTRA-Sim allocates stream tags from separate dimension-mask counters, with
distinct tag residues for overlapping scopes. A TP-only logits gather must
not advance the next EP collective's sequence. Disjoint groups may reuse a
scope's tags because matching includes source and destination ranks. The
allocator lives on `Sys`, so its counters persist across batch graphs; it
fails on namespace exhaustion rather than wrapping. Explicit communicator
handling and all timing parameters remain unchanged. Rebuild ASTRA-Sim with
this frontend change; omitting the head against the old global counter can
deadlock a later EP wave when DP members have different head participation.

When one DP instance is idle (no requests), a dummy batch (one decode query,
`1 + num_speculative_tokens` tokens, matching `GPUWorker.execute_dummy_batch`) is created
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
- `COMM_COLL_NODE` for ALLREDUCE / ALLGATHER / REDUCESCATTER (with optional `involved_dim` BoolList attribute)

The converter parses `comm_type` strings like `ALLGATHER:0,1` via `_parse_comm_type()`,
splitting into `comm_type="ALLGATHER"` and `involved_dim=[False, True]`. It also accepts
`ALLTOALL`, which nothing emits -- see the MoE section.

The MEM_STORE node uses the **last layer's** `output_memory_loc` and
`output_memory_size`. This is why the sampler (not lm_head) must have
`output_loc=REMOTE:{node_id}`: what goes back to the host is the sampled token ids
(4 bytes per sequence), not the logits the sampler read.

Memory location types: `LOCAL` (NPU) = 1, `REMOTE` (CPU) = 2, `CXL` = 3, `STORAGE` = 4.
These must match the C++ enum in `astra-sim/astra-sim/system/AstraMemoryAPI.hh`.

### Docker environments

For long runs, `scripts/monitor_run.py` provides process-tree RSS, host available
memory, optional swap-growth and timeout guards. Choose explicit limits for the
host and a fresh log path; summed RSS can double-count shared pages. It stops
only its own command and descendants. Keep container memory limits as the hard
backstop. Optional GPU monitoring identifies one physical UUID but neither
reserves it nor establishes permission or exclusive access.

- **vLLM container** (used by GPU acquisition, `python -m bench run`, and
  `python -m workloads.generators`): `vllm/vllm-openai:v0.31.0` (CUDA 13;
  requires a compatible NVIDIA driver)
  - Launched via `scripts/docker-vllm.sh`. Set `VLLM_GPUS` to a docker
    device spec to keep it off GPUs someone else is using; the inner
    quotes are part of the value (`VLLM_GPUS='"device=2,3"'`), and
    without them docker reads the second field as a GPU count. Default
    is every GPU on the host
  - Mounts the **LLMServingSim repo root** as `/workspace`; container cwd
    is `/workspace`, so `python -m profiler …` etc. work directly
  - Installs `datasets`, `matplotlib`, `pandas` and Torch's declared
    `nvidia-nccl-cu13==2.29.7` dependency. This package pin does not force
    NCCL-only communication or change the benchmark's communication policy
  - Applies the MiniMax-M3 MTP namespace fix and inert-by-default observers.
    The SM120 sparse-MLA fix is upstream in 0.31.0; do not reapply its obsolete
    backport. `VLLM_STEP_SHAPE_LOG` is an optional CPU-entry cadence diagnostic,
    not a GPU step timer. `bench --record-gate-stats` enables the modular-router
    observer and requires eager execution; monolithic MoE paths can bypass it
  - `scripts/install-vllm.sh` targets Python 3.12 in the repository's `.venv`,
    uses uv's `--torch-backend cu130`, and applies the same dependency pin and
    patches. Check installed versions and live model/backend coverage separately
  - Export `HF_TOKEN` in the shell for gated-config auto-download; the launcher
    forwards it without embedding credentials in the script
- **Simulator container**: `astrasim/tutorial-micro2024` + Python deps
  - Launched via `scripts/docker-sim.sh`
  - Mounts the repo root at `/app/LLMServingSim`; ASTRA-Sim + Chakra are
    built inside via `scripts/compile.sh` on first use
  - `python -m bench validate` compares recorded files here on CPU, without
    booting vLLM or using a GPU

## README and docs split

The repo has two documentation surfaces with deliberate scope:

- **`README.md`** — minimal front door. Latest News / About / Getting Started /
  Publications / Citation only. Logo + link bar (Website / Documentation / Contribute /
  Contact / Changelog) point everything else out to the website. **Do not
  re-add detailed content (CLI flag tables, dataset schema, profiler
  walkthroughs, validation plots, etc.) to the README** — it lives on the
  website now. Keep Getting Started to runnable commands, the CPU/GPU boundary,
  runtime versions and documentation links. Measurement internals, model-specific
  limits and contribution policies belong in the guides.
- **`docs/`** — the public docs site (Docusaurus 3, deployed at
  `https://llmservingsim.ai`). All long-form content lives here. See
  `docs/AGENTS.md` for site-specific conventions.

When you add a new feature with user-visible behavior, document it on the
website (not the README).

Start directory READMEs with a short introduction to their purpose, inputs and
outputs, plus links to the main usage sections. Put detailed measurement rules,
model-specific limits and recovery contracts under topic headings after the
usage entry points; do not accumulate change descriptions in the introduction.
Directory READMEs use `## Layout` for source trees and `## Output schema` for
generated bundles. Use `text` fences, `├──` / `└──` / `│` connectors and four-column
nesting; keep explanations outside trees and label selected inventories as such.
Use language-tagged code fences, sentence-case section headings, and blank lines
after headings and before lists or code blocks. Do not rewrite dependency
READMEs for cosmetic consistency. Latest News links completed work to its PR;
in-progress entries are roadmap statements, not supported behavior or accuracy claims.

## Workload generation

ShareGPT free generation forwards `--vllm-repetition-penalty` to
`SamplingParams.repetition_penalty` with the parsed default of `1.1`;
`1.0` disables the penalty. Keep registered sampling options connected to
the generation call and check both defaults and explicit overrides. This path
requires `--use-vllm` without `--fix-len`; existing workload files are unchanged.
Do not promise an output length or EOS behavior from the penalty alone.

## Commit & Pull Request Guidelines

- Short imperative commit messages: `Fix incorrect evict_size accumulation`,
  `Add Qwen3 model support`
- Keep commits focused — one logical change per commit
- Record verified fixes for established bugs in separate commits. Commit only
  adopted changes, not unresolved experiments.
- **Do not commit intermediate work.** This includes development/session notes,
  intermediate measurements and generated results, experimental code, and
  temporary scripts or test code written to diagnose or verify a fix. Run
  validation locally, but keep these temporary files out of the staged diff.
  Do not delete retained experiments merely because they are excluded.
- **Every commit must include documentation updates**, including docs-only
  commits: the root README.md and affected directory READMEs, AGENTS.md,
  CHANGELOG.md, and relevant public pages under docs/. Review repository
  READMEs, including per-directory files, for affected or stale descriptions.
  Regenerate the site's changelog from CHANGELOG.md. Keep README updates
  concise and consistent with the README/docs split above.
- Public documentation describes final supported behavior, usage, verified
  changes and limitations only. Never publish session diaries, investigation
  checkpoints or intermediate experiment notes, including on contributor pages
  or in this file.
  Public documentation and reproduction instructions must not reference or
  depend on private archives or untracked intermediate experiment scripts/results.
- Keep one changelog section per release and one heading per change category.
  Consolidate related Unreleased entries into short, user-facing summaries;
  retain migration requirements, limitations and contributor credit. Link to
  guides for implementation details instead of accumulating per-commit stories.
  Preserve published release records when tidying pending changes.
- When consolidating release notes, compare commits and the net code diff since
  the previous release, including changed submodules. The old changelog is not
  an exhaustive inventory; do not advertise reverted experiments as features.
- Before committing, inspect the staged file list and diff for both exclusions
  and mandatory documentation coverage. Do not stage the whole dirty worktree.
- Do not add AI-tool references or AI attribution trailers to commit messages.
- Publish required submodule commits before their parents: Chakra, ASTRA-Sim,
  then the frontend. Check remote reachability and use
  `git push --recurse-submodules=check` for the parent; do not publish local backup
  refs or force-push over independent remote work. Rebuild the backend and
  reinstall Chakra after updating the recursive submodules.
- Include the exact command used for validation and note any output CSV path in PRs
- Describe which simulation mode is affected and the config/dataset used

## Comparing against a bench run: the metric definitions

Benchmark communication backend selection defaults to **vLLM's own choice,
not NCCL-only**. Do not force all-reduce environment values. Separately retain
the benchmark's `fuse_allreduce_rms=False` and `fuse_gemm_comms=False`, recording
both under `engine_kwargs.compilation_config`: changing backend selection does
not authorize enabling these two compiler fusions. Ordinary compute compilation
and CUDA graphs stay enabled; model-native fused operations are not governed
universally by these switches. `--disable-custom-all-reduce` is an explicit opt-in, off by default;
the shell template exposes it as `DISABLE_CUSTOM_ALL_REDUCE=1`. This flag alone
does not disable other optimized or model-native fused paths. Record the argument
in `engine_kwargs`, inherited communicator variables (including unset values)
in `hardware.all_reduce_environment`, and resolved parallel/compiler settings
in `resolved_config`. Configuration is not proof of actual collective dispatch.
Do not add source patches to force NCCL as the default. Historical NCCL-only
examples retain their recorded settings and are not relabelled. Simulator-side
NCCL calibration does not select the benchmark backend; keep every comparison's
network model, recorded backend and hardware environment matched.

Snapshot resolved engine settings and KV capacity before shutdown. Normal
serving runs write `engine_start.json` before submitting requests, exclusively;
use a fresh output directory. This startup snapshot is not a completed run.
`bench validate` rejects known partial, aborted, resolve-only and perturbed
diagnostic inputs while retaining legacy request/timeseries compatibility.
`--kv-cache-memory-bytes` fixes an explicit per-GPU budget, but comparisons must
still verify resolved block counts. Dummy weights do not establish equivalent
backend selection, routing or scheduling for a real checkpoint.

`bench/core/kv_cache_metadata.py` observes native READY payloads only during the
standalone synchronous constructor. Restore the callback before decoding or
request submission; add no worker RPC, request hook or simulated latency.
The reviewed vLLM 0.31 client sums connected engine block counts and retains the
last READY block size. Preserve those raw fields; their product is not generally
an aggregate token capacity. Match each instance to its directly observed DP
rank under complete `kv_cache.capacity.per_engine`, never an inferred division.
Unsupported callback/response APIs, replaced callbacks, client/topology mismatches
or incomplete observations remain explicit unknowns. Record the observed version,
callback signature and response fields; exact version equality is not required.
These startup scheduler-pool facts neither measure physical per-GPU allocation
nor track later elastic changes. Do not relabel historical benchmark metadata.

Use `bench/core/validate.py::_bench_latencies`, the implementation shared by
`bench validate` and committed summaries, rather than ad hoc metric formulas.

`requests.jsonl` mixes wall-clock `arrival_time` with monotonic lifecycle
timestamps. Align arrival to the engine clock before computing milliseconds:

```text
offset  = min over requests of (queued_ts - arrival_time)
arrival = arrival_time + offset
TTFT    = (first_token_ts - arrival) * 1000
TPOT    = (last_token_ts - first_token_ts) / (output_toks - 1) * 1000
latency = (last_token_ts - arrival) * 1000
```

The offset is an estimate: if pickup delay is nonnegative, its residual bias
is the minimum pickup delay in that run, not a guaranteed fixed tolerance.
When conversion is unavailable, the validator uses its documented queued-time
fallback. TPOT is defined only for outputs with more than one token.

Using queued time instead of arrival omits frontend-to-engine waiting from
TTFT and end-to-end latency. It does not affect token-to-token intervals.
Neither request latency nor `Total clocks (ns)` is simulator host wall time.

Report all fifteen TTFT, TPOT and latency statistics, including every fixed
repeat when repeated references are available. The simulator's deterministic
regression digests establish reproducibility, not benchmark accuracy. Engine
timing variation does not justify omitting a failing TTFT percentile or
replacing it with run span. Preserve recorded vLLM references when refreshing
simulator-side examples, and explain intentional changes in either direction.

## Testing & Validation

Published accuracy belongs to a fixed profile bundle, recorded benchmark and
execution contract. Reproducing the committed examples does not certify a
fresh-only acquisition from the latest profiler or every catalogued model.
Keep availability, acquisition provenance, deterministic regression and
end-to-end accuracy distinct in all public documentation. Historical coarse-grid
shot counts are examples, not current-default runtime estimates.

The example wrappers discover `bench/examples/*/*/config.json`, including
stored diagnostics beyond the headline benchmarks. Reproduction uses the
example's target graph policy and memory configuration. The RTXPRO6000 headline
configs inherit transport from the hardware bundle. Llama and both Qwen
bundles contain refreshed TP1 and TP2 acquisition data; headline validation
covers Llama TP1, Qwen3-32B TP2 and Qwen3-30B TP1/DP2/EP2 only. Preserve
adopted generated values and rows: profile and hardware metadata, measurement
tables, benchmark metadata, simulation outputs, validation files and logs.
The only publication exception is portable identifiers for environment-dependent
dataset and local model paths. Use repository-relative dataset paths and canonical
model identifiers where equivalent; temporary sharded-model identifiers must
retain their TP-shape semantics. Do not prune records or change measurements,
settings, hashes or validation results. Publish the runtime tables
described by the generated bundle metadata. Qwen3-30B's
instance memory fractions match the complete per-engine READY block counts,
not the aggregate client count or an end-to-end latency target. Reproduction uses the
recorded `kv_cache.block_size` unless `BLOCK_SIZE` explicitly overrides it.
If that metadata is absent, omit the flag so serving resolves its default;
do not force a dense-only page size onto sparse or hybrid examples.

Use focused local checks for changed contracts; temporary diagnosis and
fix-verification scripts or tests stay outside the published tree. The simulator
is deterministic, so its regression validation checks exact equality against
recorded results:

**Several parts of `profiler/` are simulator inputs**, despite the path.
Shared data and CPU-only helpers can change simulation clocks:

- **`profiler/models/*.yaml`** — the layer order. Merging two catalogs into one
  broke all 16 MoE scenarios exactly this way.
- **`profiler/core/stack.py`** — which block each decoder layer runs, resolved
  from the checkpoint's config.
- **`profiler/core/catalog_path.py`** — `model_type` → yaml resolution.
- **Attention shape, skew calibration and MoE contract helpers** — geometry,
  correction-table loading and deployment compatibility affect lookup.
- **`profiler/perf/`** — latency tables and hardware defaults.

These helpers must remain usable in the CPU simulator environment without
importing vLLM or GPU acquisition dependencies. Audit actual imports rather
than assuming only the stack and catalog resolvers can affect simulation.

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
   exits non-zero on either defect it can find. A **gap** is CUDA time no
   entry claims. An **over-match** is one entry claiming nodes in
   structurally unrelated places, so its number is the sum of two roles —
   which the coverage percentage cannot show, because over-matching leaves
   nothing unbound. It fires only when the matched nodes' ancestor chains
   share no common prefix, so an entry legitimately binding several kernels
   under one parent (M3's `sparse_attention`, any `*`-prefixed glue) is not
   flagged. Qwen3.5's `mtp_norms` read **1287 us for two RMSNorms** and its
   `embedding` read **2.1x** while passing coverage at 100%. Inspect dynamic
   invocations, native activities and unscoped gaps, not only the static module
   inventory. Coverage does not prove every declared alternative executed:
   an entry absent from every forward can leave no activity gap. Required-entry
   checks, including the MTP slice check, are separate. Shared-activity overlaps
   must never be suppressed by the target/drafter repeated-role exemption.

DP/TP/PP scenarios with uneven request streams can change round pairing when
a component cost changes. Total clocks may then move non-monotonically even
when that component gets faster. Preserve these scenarios as behavior checks;
inspect the resulting batches, request completion and token counts before
attributing a changed digest to a kernel-cost regression. An aggregate clock
alone does not identify its cause.

A scenario whose clock equals an existing one exercises flag parsing and
nothing else. Several knobs only bite once the KV cache is saturated, which is
what the `saturated_*` scenarios are for; `example_trace.jsonl` never gets
there, and its DP members always drain together, which is what the `*_uneven`
scenarios are for.

**A model family is an axis too.** Everything but the `hybrid_*`, `sparse_*`
and `spec_*` scenarios runs Llama-3.1-8B/70B, Qwen3-30B-A3B or Qwen3-32B, so
none of it reaches linear attention, sparse attention, MLA, a heterogeneous
stack or the drafter — a speculative `pp_size > 1` deadlock lived through a
green 58/58 for exactly that reason. Their cluster configs use the **smallest
parallelism in which each model fits** on the profiled hardware (DeepSeek-V3.2
is 625 GB at fp8, so PP=16 is a floor `MemoryModel` enforces, not a choice),
and PP rather than TP because only `tp1` is profiled for three of the four.
`--num-reqs` is deliberately small: what these check is which layers get
emitted, and the stacks are 60-78 deep.

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
- **Don't add a layer name without a formula on its selected sizing path.**
  Ordinary catalogs use `memory_model.calculate_sizes`; native/contextual
  profiles use the admitted providers in `component_sizes.py`. Check the
  selected layout, not every native name against the ordinary dispatcher.
  Logical bytes in a duration trace do not imply serving-capacity support
- **Don't edit `astra-sim/`** unless the change targets simulator integration
  (e.g., `llm_converter.py`, `Workload.cc`, input configs). Chakra is *installed*
  into the container's site-packages by `scripts/compile.sh`, so editing
  `llm_converter.py` changes nothing until you reinstall it
  (`cd astra-sim/extern/graph_frontend/chakra && pip3 install .`)
- **Don't commit large files**: generated traces, `.et` files and scratch run
  output are gitignored (`outputs/*` with `!outputs/example_*.csv`,
  `bench/results/`). `astra-sim/inputs/runs/` is cleaned per run, and each
  batch's workload folder already during the run once `pp_size + 1` newer ones
  of the same instance or DP group have been handed out, unless you pass
  `--keep-inputs` (or `--save-trace-text`, which implies it), either of which
  can leave gigabytes behind
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
- **Don't keep per-batch scheduler state on a `Request`.** If `add_done` reads
  it, it belongs in the `Batch`, snapshotted at build time like
  `scheduled_tokens` and `spec_scheduled`. At `pp_size > 1` the next
  `schedule()` runs while the batch is in flight and rewrites the request's
  fields for the step it is building, so `add_done` reads the wrong step's
  values. That deadlocked every speculative `pp_size > 1` run and is invisible
  at `pp_size == 1`
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
