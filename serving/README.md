# serving

LLMServingSim simulator core. Run as `python -m serving --cluster-config <...> [...]`.

## Layout

```
serving/                        Python package
├── __init__.py                 module map
├── __main__.py                 simulation entry point + main loop
├── core/                       internals (every .py module documented below)
│   ├── scheduler.py            vLLM-style continuous batching scheduler
│   ├── trace_generator.py      builds execution traces from profiled latencies
│   ├── memory_model.py         memory tracking, KV cache, tensor sizes
│   ├── graph_generator.py      Chakra protobuf graph generation
│   ├── controller.py           IPC with ASTRA-Sim subprocess
│   ├── router.py               request routing across instances
│   ├── gate_function.py        MoE expert token routing (incl. group-limited)
│   ├── spec_decode.py          speculative-decoding acceptance model
│   ├── config_builder.py       cluster config -> ASTRA-Sim input files
│   ├── power_model.py          power / energy estimation
│   ├── pim_model.py            PIM device model
│   ├── request.py              Request / Batch data classes
│   ├── block_pool.py           per-tier KV block pool + prefix-cache index
│   ├── kv_cache_manager.py     tiered KV cache manager (block hashing, allocation)
│   ├── run_paths.py            per-run ASTRA-Sim input/output path layout
│   ├── logger.py               Rich-based logger + stdio capture
│   └── utils.py                model config loading, formatting helpers
├── run.sh                      one runnable example per feature (a menu, not a suite)
├── validate.sh                 every scenario vs recorded clocks + bench/examples digests
└── validate-baselines.txt      the recorded values; refresh with validate.sh --update
```

## Everything you can set

Every flag is also a **per-instance** cluster-config field of the same name
with underscores, unless marked *run-wide*.
**[Reference → CLI flags](https://llmservingsim.ai/docs/reference/cli-flags)**
carries the semantics and the defaults; this is the index, so a flag missing
from one list is visible against the other.

| Group | Flags |
|-------|-------|
| **required** *(run-wide)* | `--cluster-config` |
| **workload** *(run-wide)* | `--dataset`, `--num-reqs`, `--skip-prefill` |
| **batching** | `--max-num-batched-tokens`, `--max-num-seqs`, `--enable-chunked-prefill`, `--long-prefill-token-threshold`, `--reserve-full-isl` |
| **memory** | `--block-size`, `--npu-memory-utilization` |
| **prefix caching** | `--enable-prefix-caching`, `--enable-prefix-sharing`, `--prefix-storage` |
| **routing** *(run-wide)* | `--request-routing-policy`, `--expert-routing-policy` |
| **speculative decoding** | `--num-speculative-tokens`, `--spec-acceptance-rate`, `--spec-acceptance-policy` |
| **offloading** | `--enable-attn-offloading`, `--enable-sub-batch-interleaving`, `--enable-local-offloading` |
| **trace / graph** *(run-wide)* | `--enable-block-copy`, `--save-trace-text`, `--keep-inputs` |
| **backend** *(run-wide)* | `--network-backend` |
| **output** *(run-wide)* | `--output`, `--run-id`, `--inputs-root`, `--log-interval`, `--log-level` |

Boolean flags use `argparse.BooleanOptionalAction`, so each has a
`--no-` form (`--no-enable-prefix-caching`).

**There is no dtype flag, and no `dtype` cluster-config field.** A modern
checkpoint carries five cache dtypes decided in four different places —
weights, KV cache, mamba conv state, mamba recurrent state, sparse-indexer
side cache — so all five are read from the model config.
`memory_model.cache_dtype_bytes()` is the single table. To simulate another
precision, profile it: the *profiler* still takes `--dtype` /
`--kv-cache-dtype` and writes a separate bundle, and the simulator reads the
one the checkpoint names.

`--block-size` is the other input that is really a derived value. vLLM
treats a block size as a floor and an alignment unit, raising it until one
attention page covers one mamba page, so the profiler records what the
engine settled on in `meta.yaml::engine_resolved.per_tp[tp]` and omitting
the flag reads that back. Qwen3.8-27B resolves to **784** from a requested
16. An explicit value that disagrees is allowed but warned about.

## Validating a change

There is no unit-test suite. The simulator is deterministic, so validation is
exact equality against recorded results:

```bash
./serving/validate.sh            # both stages, ~8 min
./serving/validate.sh --help     # options
```

Stage 1 compares every scenario against the `Total clocks (ns)` in
`validate-baselines.txt`. Stage 2 regenerates each `bench/examples` entry's
`outputs/sim.csv` and `validation/summary.txt` and checks their md5s. Anything
that moved is printed as a markdown table to paste into the PR — a difference
is not automatically a bug, but it always needs an explanation. See
[Validating your changes](https://llmservingsim.ai/docs/contributor/validating-changes).

The scenario list treats the **model family** as an axis of its own. Most of it
runs Llama-3.1-8B/70B, Qwen3-30B-A3B or Qwen3-32B, which between them reach no
linear attention, sparse attention, MLA, heterogeneous stack or drafter; the
`hybrid_*`, `sparse_*` and `spec_*` scenarios are what cover those, and they
exist because a speculative `pp_size > 1` deadlock survived a green run of
everything else.

## Architecture

The simulation loop in `serving/__main__.py` orchestrates these modules per iteration:

1. **Router** dispatches incoming requests to instances
2. **Scheduler** forms batches under memory and token budget constraints
3. **Trace generator** looks up profiled latencies and emits execution traces
4. **Graph generator** converts traces to Chakra protobuf graphs
5. **Controller** feeds graphs to ASTRA-Sim and reads back timing results
6. **Memory model** tracks KV cache allocation, eviction, and prefix cache hits

### Trace generation pipeline

The trace generator constructs per-iteration execution traces by walking the
``blocks:`` and ``shared:`` sections of the architecture yaml
(`profiler/models/<model_type>.yaml`). For a standard decoder-only model:

```
shared.prologue (embedding)
  → [attn.<type>.pre_attn  (layernorm → qkv_proj → [qk_norm] → rotary_emb → attention)
     → attn.<type>.post_attn (o_proj[ALLREDUCE] → layernorm)
     → mlp.dense (gate_up_proj → act_fn → down_proj[ALLREDUCE])
        or mlp.moe (moe[EP all-to-all])
    ] × N_layers
  → shared.head (final_layernorm → lm_head → sampler)
```

`blocks:` is keyed by **axis**, and which block a given layer runs comes from
the checkpoint's own config, not from the yaml — `layer_types` decides the
attention type, `first_k_dense_replace` / `decoder_sparse_step` /
`moe_layer_freq` the MLP, `sparse_attention_freq` / `index_topk_pattern`
whether a sparse-selection branch applies. `profiler/core/stack.py` owns those
rules and both the profiler and the simulator import it. A uniform stack is the
degenerate case: one entry per axis, one block built and replayed for every
layer.

Latencies come from the profiler's per-category CSVs under
`profiler/perf/<hardware>/<model>/<variant>/tp<N>/` — `dense.csv` (keyed on
`tokens`), `per_sequence.csv` (`sequences`), `attention.csv` (a 5D grid on
`prefill_chunk, kv_prefill, n_decode, kv_decode, decode_q_len`),
`linear_attention.csv` (`prefill_tokens, n_decode`, mamba/gated-DeltaNet only),
and `moe.csv` (`tokens, activated_experts`). `resolve_variant(model_config)`
names the `<variant>` folder as a **pure function of the checkpoint** — weight
dtype from `quantization_config.quant_method` or `torch_dtype`, plus a
`-kv<dtype>` suffix when the config declares a quantized KV cache. There is no
dtype flag on the simulator; the profiler's `--variant` / `--dtype` /
`--kv-cache-dtype` still write other bundles beside it, which the simulator
never asks for.

`meta.yaml` next to each variant records the engine flags the profiler swept
(notably `max_num_batched_tokens` and `max_num_seqs`); the simulator warns at
startup when the runtime values exceed them, signalling that lookups will
extrapolate.

### Head dimension

Some models (e.g., Qwen3) have `head_dim != hidden_size // num_attention_heads`. The
codebase always uses the explicit `head_dim` from model config:

```python
head_dim = config.get('head_dim', n_embd // n_head)
q_dim = n_head * head_dim        # NOT n_embd
kv_dim = kv_head * head_dim      # NOT n_embd // group
```

### Working directory

`serving/__main__.py` changes cwd to `astra-sim/` early in execution. All relative paths in the
simulator resolve from `astra-sim/`, not the repo root. Paths to `configs/`, `workloads/`,
`profiler/` are prefixed with `../` in code.

## Modules

All modules below live under `serving/core/`. Imports inside the
subpackage use relative form (`from .X import ...`); external callers
use `from serving.core.X import ...`.

### `request.py`
Defines the `Request` and `Batch` data classes. Tracks per-request state and latency
metrics (TTFT, TPOT, ITL).

### `scheduler.py`
Per-instance scheduler implementing vLLM-style continuous batching. Manages request queuing,
memory-constrained batch formation, KV cache block eviction and swapping to CPU, and prefix
cache lookup. Add custom scheduling policies here.

### `router.py`
Routes incoming requests across instances in real-time based on current system state.
Default policy `LOAD` uses vLLM-style weighted least-loaded scoring (`waiting * 4 + running`).
Requests are routed at their arrival time during the simulation loop, not upfront.
Handles request transfer in Prefill/Decode disaggregation mode.

### `gate_function.py`
Routes tokens to MoE experts according to configurable policies (Copy, Round Robin, Random,
Custom). `COPY` (default) enables block copy optimization. Provides EP-aware routing via
`route_ep()` with even expert-to-rank partitioning for per-rank latency lookup.

### `spec_decode.py`
The acceptance model behind `--num-speculative-tokens`. Which draft tokens the
target accepts is the one thing a simulator cannot compute — it needs both
models' distributions over real tokens — so acceptance is a **policy**, chosen
the way MoE expert routing is, with the default taken from what each model's
authors published (`configs/spec_decode.json`). The rate is `accepted / drafted`
and **marginal**, so `mean_accept_length = 1 + rate * N`; that identity
reproduces all nine published (rate, length) pairs to within 0.01 tokens.
`--spec-acceptance-policy` picks how the accepted count is drawn (`FIXED`,
`DECAY`, `CUSTOM`). A model with no published figure gets no default — the four
modern families range from 0.39 to 0.78, so there is nothing defensible to
guess.

### `memory_model.py`
Static sizing math plus a byte-level view over the block pools. Contains
`calculate_sizes(parallel=)` and `get_weight` for per-layer tensor size computation — the
`parallel` parameter is TP degree for dense layers and EP degree for MoE experts, and MoE
expert weights are sharded by `ep_size`. Modify these when adding a new model architecture.
Sizes the NPU KV cache the way vLLM does: `npu_mem.mem_size * npu_mem.mem_util - weight`, then
divided into blocks. `npu_used` / `cpu_used` are properties derived from the pools, so there
is exactly one ledger per tier.

### `block_pool.py`
One `BlockPool` per memory tier (NPU / CPU / CXL): a doubly linked free list in eviction
order, a `block_hash -> block` index, and a refcount per block. Port of vLLM v0.19.0's
`vllm/v1/core/block_pool.py`. `num_free_blocks` is exact, so an allocation either succeeds or
reports failure in the same call. Eviction is a silent side effect of allocation, and a freed
block goes to the queue *tail* so it is reused last — which is what lets a just-preempted
request find its blocks again.

### `kv_cache_manager.py`
`TieredKVCacheManager`: per-request NPU block tables, the tier lookup, and the transfer
accounting. Block hashes are chained once at the NPU block size
(`hash(parent_hash, block_tokens)`); a lower tier whose blocks are N times larger keys on
every Nth hash of the same chain, so all tiers share one key space and a single walk yields
both the NPU hit and the lower-tier hit. Recall from a lower tier is charged; the
write-through is reported for energy only, matching vLLM's `OffloadingConnector`, which
defers it to the next engine step on a dedicated stream.

### `trace_generator.py`
Core performance estimator. Loads the profiler's per-category CSVs under
`profiler/perf/<hardware>/<model>/<variant>/tp<N>/` plus the architecture
yaml (`profiler/models/<model_type>.yaml`) and walks the yaml's ``blocks:``
and ``shared:`` sections to emit each iteration's layers. Composable helpers:

- `resolve_variant()` / `_load_perf_db()` / `_load_architecture()` — turn
  `(hardware, model, dtype, kv_cache_dtype)` into a loaded DB with category
  tables, the block order, and the checkpoint's per-layer block resolution.
- `_lookup_dense()` / `_lookup_per_sequence()` / `_lookup_attention()` /
  `_lookup_moe()` — category-specific lookups with 1D linear interpolation
  (dense/per_sequence), 4D linear for attention (each of
  prefill_chunk / kv_prefill / n_decode / kv_decode bracketed by its two
  neighbouring profiled values and blended linearly), and 2D for MoE.
- `_lookup_attention_with_skew()` / `_skew_alpha()` — skew correction on
  the attention kernel: a lookup at the batch's mean decode kv, blended
  toward a second lookup at its max only when a non-zero bucket-specific
  alpha applies, resolved from `meta.yaml::skew_fit`. Bucket axes (`n`, `skew_rate`, `kv_big`, `kp`;
  `pc` used raw) are read from meta so the simulator automatically
  picks up whatever resolution the profiler ended up with. When meta
  predates the skew_fit block a pooled fallback constant is used —
  the simulator stays usable against older profile runs.
- `_hydrate_skew_fit_tables()` — on load, walks each TP's
  `bucket_table:` pointer and reads `tp<N>/skew_fit.csv` into the
  in-memory `alpha_by_bucket` map that `_skew_alpha` consults.
- `TraceCtx` / `BatchCtx` / `PowerAccumulator` — data classes for context passing
- `_emit_layer()` — single-layer emission that dispatches by catalog category
- `_emit_sequence()` — walks a list of canonical names from the yaml; attaches
  TP ALLREDUCE to `o_proj`/`down_proj` and swaps in PIM attention before
  the NPU attention kernel when offloading is enabled. Emits a one-shot warning
  when a sequence layer is missing from the profile CSVs.
- `_emit_prologue()` / `_emit_pre_attn_layers()` / `_emit_post_attn_layers()` /
  `_emit_final_layers()` — per-section wrappers over `_emit_sequence`.
- `_emit_drafter()` / `_emit_drafter_block()` / `_drafter_spec()` /
  `_drafter_loop_bctx()` — speculative decoding's draft passes, emitted after
  the target's head because that is where vLLM runs them (from
  `sample_tokens()`). N passes per step, each `mtp.prologue` → a replay of
  `mtp.decoder_block` → `mtp.head`. `_drafter_spec` resolves which block that
  replay is from the catalog (each family's MTP module forces it in vLLM's own
  source, and any index into the resolved stack wraps to layer 0); axes the
  catalog omits are inherited from the target's stack when it agrees on them.
  `_drafter_loop_bctx` reshapes passes 1..N-1 to one query per sequence, which
  is what `llm_base_proposer.py` pins inside its loop — pass 0 keeps the
  target's own token layout.
- `_spec_block_layers()` — `_block_layers()` for a bare `LayerSpec` rather than
  a layer index, which is what lets the drafter's declared block reuse the
  same block walk.
- `_synthesize_interleaved_trace()` — alternates two `BatchCtx` objects for
  sub-batch interleaving.

Handles tensor parallelism (ALLREDUCE placement), MoE expert routing with
`involved_dim` dimension scoping for DP+EP, PIM attention offloading, and
sub-batch interleaving. The `comm_type` field supports dimension scoping
(e.g., `ALLGATHER:0,1`) for multi-dimensional ASTRA-Sim topologies. To add a
new model architecture, add a `profiler/models/<model_type>.yaml` with a
matching `blocks:` / `shared:` rather than editing this file.

### `run_paths.py`
Resolves `--run-id` and the ASTRA-Sim input layout beneath it. An omitted run
id becomes a process-unique one, so two simulator invocations running at once
do not share intermediate files; an explicit one is validated as a safe path
component. `RunPaths` then carries the network / system / memory config paths
that `config_builder.py` writes and `controller.py` hands to ASTRA-Sim.

### `config_builder.py`
Parses the user-provided cluster config JSON from `configs/cluster/` and generates the
ASTRA-Sim input files under `astra-sim/inputs/runs/<run_id>/`: `network/network.yml`,
`memory/memory_expansion.json`, and `system/system.json`.
Per-iteration text traces are not produced at all by default -- the Chakra
converter takes the trace rows straight from the trace generator -- and the
generated run directory is removed after a successful simulation. Use
`--save-trace-text` to write the text for inspection (it implies `--keep-inputs`),
or `--keep-inputs` alone to preserve the Chakra workloads and input configs.
For DP groups, generates a multi-dimensional network topology, innermost dimension
first: `[tp_size, dp_group_size]`, or `[tp_size, pp_size, dp_group_size]` when
`pp_size > 1`, matching vLLM's `all_ranks.reshape(-1, dp, pp, pcp, tp)`. The
`system.json` collective implementations are sized to match the number of topology
dimensions. Computes `tp_dim`/`ep_dim` per instance for `involved_dim` scoping; EP is
scoped to the DP and TP dims and never PP.

### `power_model.py`
Estimates power and energy consumption per node, covering NPU, CPU, DRAM, interconnect, NIC,
and storage.

### `controller.py`
Manages the IPC protocol with the ASTRA-Sim subprocess. Writes workload graph paths to
ASTRA-Sim stdin and parses iteration timing from stdout.

### `graph_generator.py`
Invokes the Chakra converter to transform text-format execution traces into protobuf workload
graphs consumed by ASTRA-Sim.

### `pim_model.py`
Parses PIM device INI configuration files from `configs/pim/`. Derives bandwidth, latency, and
power parameters used by the trace generator for PIM-offloaded attention.

### `utils.py`
Helper functions for loading model configs, constructing workload paths, and formatting
terminal output.

### `logger.py`
Configures the LLMServingSim logger. Log level is set via `--log-level` on the
`python -m serving` CLI.
