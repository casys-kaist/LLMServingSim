---
title: Changelog
description: LLMServingSim release history
---

All notable changes to this project are documented in this file.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions.

## [Unreleased]

### Added
- Per-tier KV cache block pools (`serving/core/block_pool.py`) and a tiered
  manager over them (`serving/core/kv_cache_manager.py`), ported from vLLM
  v0.19.0's `block_pool.py` / `kv_cache_manager.py` and the block/queue
  primitives in `kv_cache_utils.py`. Each pool owns one tier's free list,
  prefix-cache index and refcounts, so `num_free_blocks` is exact and an
  allocation either succeeds or reports failure in the same call. Both modules
  carry a self-test that runs without Docker
  (`python3 -m serving.core.block_pool`,
  `python3 -m serving.core.kv_cache_manager`).
- Chained block hashes (`hash(parent_hash, block_tokens)`) shared across every
  tier: a lower tier whose blocks are N times larger keys on every Nth hash of
  the same chain, which is vLLM's
  `offloading/scheduler.py::_get_block_hashes`. One walk over a request's
  hashes now yields both the NPU hit and the lower-tier hit.
- `--npu-memory-utilization` (default `0.9`) with a per-instance
  `npu_mem.mem_util` cluster-config override, named to match its `mem_*`
  siblings and placed beside the `mem_size` it scales. Corresponds to vLLM's
  `--gpu-memory-utilization`, renamed because every other memory surface here
  uses NPU terminology. KV capacity is
  `npu_mem * utilization - model weight`, mirroring vLLM's
  `requested_memory - non_kv_cache_memory`. vLLM also subtracts the activation
  peak and CUDA context, which are not modelled, so the simulator's capacity is
  an upper bound at the same utilization.
- `--reserve-full-isl` (on by default) with a per-instance `reserve_full_isl`
  cluster-config override: admit a request only if its whole sequence fits, not
  merely its first chunk. Port of vLLM's `scheduler_reserve_full_isl`, which is
  also `True` there and is documented as preventing "over-admission and KV cache
  thrashing with chunked prefill".
- `meta.json` from `python -m bench run` now records vLLM's **resolved**
  configuration, not just the ten engine kwargs we asked for: `kv_cache`
  (`num_gpu_blocks`, `block_size`, `num_kv_tokens`, `gpu_memory_utilization`),
  `hardware` (accelerator name, total memory, compute capability, CUDA and torch
  versions), and `resolved_config` (the whole `VllmConfig`, one key per
  sub-config, ~296 leaf fields on v0.19.0). `num_gpu_blocks` is the number a
  simulator has to match, and the only place the activation peak and CUDA
  context vLLM subtracts from its budget become visible. Built by walking the
  config's own field list, so a vLLM upgrade that adds a knob appears without a
  code change; values JSON cannot hold become a short type tag, keeping the file
  at ~12 KB. All three are optional -- read them with `meta.get(...)`, since
  older runs lack them.
- A `KV Cache Initialization` section in the startup output, listing each
  instance's derived block/token capacity and the utilization it came from. The
  fraction alone does not tell you where memory pressure will land; the block
  count does.
- `Request.num_tokens_reached` (prompt + generated), the independent sequence
  length that mirrors vLLM's `len(_all_token_ids)`. It cannot be derived from
  `num_computed_tokens`, which preemption resets to 0.
- Public Docusaurus 3 documentation site at
  [llmservingsim.ai](https://llmservingsim.ai), built from `docs/` and
  deployed via GitHub Actions Pages. Replaces the old `docs/index.html`
  placeholder and shifts long-form content (CLI flag tables, dataset
  schema, profiler walkthroughs, validation plots, etc.) off the README.
  The repo's `README.md` is now a minimal front door (About / Getting
  Started / Publications / Citation) that links to the website. The
  `README and docs split` policy is documented in `AGENTS.md` /
  `CLAUDE.md`.
- Local search on the docs site via
  `@easyops-cn/docusaurus-search-local`. Indexes all `/docs/*` and
  top-level page routes (Contact, Changelog) at build time. Access via
  the navbar input or Ctrl/Cmd-K once the production build runs (dev
  mode does not generate the index — `pnpm build && pnpm serve` to test
  locally).
- Module helper `full_cluster_kv_bytes_per_token(model, fp, kv_cache_dtype)`
  in `serving/core/memory_model.py`. Computes full-cluster KV bytes per
  token directly from a HuggingFace-style config, avoiding the per-rank
  floor-division roundoff in `MemoryModel.get_kv(1) * num_npus`. Used by
  `__main__.py` to size shared prefix pools at startup, before any
  `MemoryModel` exists.

### Changed
- The attention grid is interpolated on a **linear** scale instead of in log
  space (`_axis_bracket`). The profiler sweeps every axis geometrically, which
  made log space look like the matching choice, but grid spacing decides where
  the kernel is sampled while the blend decides how two samples are combined —
  and the kernel is linear in each axis. Profiled decode attention fits
  `time_us = a + b * (n_decode * kv_decode)` with R^2 = 1.0000, at an implied
  rate within a few percent of the card's memory bandwidth, i.e. a pure
  KV-cache read. Blending a per-axis-linear function in log space is
  convex-biased upward by up to +6.0% per axis on a doubling grid. Validated
  model-free: predicting each profiled grid point from its two neighbours and
  comparing against what the GPU reported puts log space at +11.6% to +14.4%
  mean error against +2.3% to +3.7% for linear, with linear ahead on all four
  axes, across every bundle in `profiler/perf/`. End-to-end against the three
  recorded vLLM runs in `bench/examples/`, TPOT improves on all 15 of 15
  metrics and end-to-end latency on 13 of 15.
- With no skew profile at all, the simulator now applies **no** skew
  correction (`_ATTN_SKEW_ALPHA_FALLBACK` 0.093 -> 0, i.e. `t_mean`) instead of
  a borrowed constant. Bundles that carry a real `skew_fit` still resolve alpha
  per bucket and are unaffected. The two blend endpoints are far apart
  (`t_max / t_mean` median ~1.5, p90 ~3.7), so each 0.1 of alpha is several
  percent of attention time and alpha has to be known to a couple of
  hundredths to be worth applying. Alpha is also not bounded to `[0, 1]`: over
  ~13k raw shots where `t_mean`, `t_max` and `t_skew` are each measured
  directly on the GPU, it runs well outside that interval and is negative in
  roughly one shot in six, because a skewed batch is sometimes genuinely
  faster than a uniform batch at the same mean. `t_mean` is looked up at the
  **arithmetic** mean `kv_decode_mean`, not a median: attention cost tracks the
  total KV read, and a uniform batch at the arithmetic mean reproduces it
  exactly.
- `Scheduler` now follows vLLM V1's `schedule()`: a persistent `self.running`
  set is served first, preempting only from its own tail, then `self.waiting` is
  admitted while budget and sequence slots remain. Admission never preempts, and
  it is skipped entirely on any step that preempted. `schedule_base` and
  `schedule_with_prefix` collapse into one `schedule()` — the pool handles
  `enable_caching=False` the way vLLM does, so two paths had no reason to exist.
  `scheduler.py` drops from ~1300 to ~510 lines, `memory_model.py` from 885 to
  ~545.
- Preemption is vLLM verbatim, including `num_computed_tokens = 0`. That is not
  re-prefill: `free_blocks` keeps the blocks' hashes, so on re-admission the
  still-resident prefix is found, a lower tier returns what was written down,
  and only the remainder is recomputed. Recovery comes from the tier hierarchy
  rather than from a special "preserve the decode state" path.
- The three prefix-cache modes now map onto three real vLLM configurations:
  `--no-enable-prefix-caching` behaves like vLLM with prefix caching off, where
  a resumed request recomputes its whole sequence; `--enable-prefix-caching` is
  default vLLM; adding `--prefix-storage CPU/CXL` is vLLM with LMCache or
  `OffloadingConnector` attached. The previous middle case billed a KV transfer
  against a tier that held nothing.
- `num_computed_tokens` is advanced when the batch is formed, as in vLLM's
  `_update_after_schedule`, with `Batch.scheduled_tokens` as the snapshot
  `add_done` works from. Advancing at completion let `pp_size > 1` schedule the
  same tokens twice.
- Prefill and decode are no longer distinct scheduler states. A request catches
  up to `num_tokens_reached`, and the trace classifies by scheduled token count
  (>1 = prefill chunk, ==1 = decode) — the classification the attention profile
  axes expect, and the only one that survives a resumed request.
- A host offload tier now uses 256-token chunks (LMCache's default) uniformly.
  The non-shared second tier and the shared CXL pool previously used page size
  1, which matched at token granularity and over-reported hits relative to any
  real offload tier.

### Removed
- `serving/core/radix_tree.py` (675 lines), the SGLang-derived prefix-cache
  radix tree, **replaced by `block_pool.py` + `kv_cache_manager.py`** (see
  Added). It had served as both the prefix index and the allocator, and as an
  allocator it was inexact: `evictable_size_` counted every unlocked token while
  `evict()` could only drop unlocked *leaves*, and charging happened later via
  tree events rather than in the call that could fail. Prefix caching keeps all
  of its user-visible behaviour -- `--enable-prefix-caching`,
  `--enable-prefix-sharing`, `--prefix-storage` are unchanged. The SGLang
  attribution stays in `CONTRIBUTORS.md` as history.
- `--prioritize-prefill` and the per-instance `prioritize_prefill` key, along
  with `Scheduler._merge_by_arrival_id` (its only caller). vLLM v0.19.0 has no
  equivalent: `vllm/core/` — the V0 scheduler whose `_schedule_default` served
  prefills first — no longer exists, and `SchedulerPolicy` is `fcfs` or
  `priority`, which is request priority rather than prefill-vs-decode.
- `Request.is_prefill()`, `evict`, `npu_last_node`, `cpu_last_node`,
  `storage_last_node`, `_prefix_locked`; and from `MemoryModel`:
  `avail_size`, `evictable_size`, `get_block_kv`, `get_evict_kv`,
  `lock_prefix`, `unlock_prefix`, `cache_unfinished_req`, `cache_finished_req`,
  `evict_prefix_cache`, `prefix_match`, `apply_kv_cache_events` and the two
  `_*_cache_hashtolen` maps.
- `Scheduler.get_first_arrival_time`, which read an attribute that was never
  assigned and had no callers (`Router.get_first_arrival_time` is the live one).

- Trace-level PP modeling write-up overhauled in
  `docs/docs/simulator/parallelism-mechanics.md` — explicitly describes
  the Chakra layer split + `COMM_SEND` / `COMM_RECV` between stages,
  with a stage-split figure. Replaces the previous
  "scheduling-only / lower bound" framing which underdescribed what
  the simulator actually models.
- `--expert-routing-policy` default documented as `BALANCED`
  everywhere (expert-parallel example, troubleshooting,
  trace-generation, `AGENTS.md`) — the earlier docs referenced a
  non-existent `COPY` default. `CUSTOM` listed under both request- and
  expert-routing options; `--enable-block-copy` decoupled from routing
  policy in the docs.
- `LOAD` request-routing scoring (`waiting * 4 + running`) documented
  in the multi-instance example. Policy lists reformatted into bullets
  across affected pages.
- `MemoryModel.get_weight` now divides the transformer-block weight by
  `pp_size` (heaviest-rank conservative bound:
  `embedding + n_layer//pp × per_block + final_layernorm + lm_head`).
  Required adding a `pp_size` parameter to `MemoryModel.__init__`
  (threaded through from `Scheduler`). PP=1 behavior unchanged — fix
  only affects future PP > 1 runs (no current cluster config exercises
  PP > 1).
- `MemoryModel.apply_kv_cache_events` now drains the second-tier event
  queue for CXL prefix storage and CPU + prefix-sharing modes (in
  addition to the previously-handled CPU non-sharing case). The CPU
  non-sharing branch keeps bridging events into `cpu_used`; the other
  paths just drain the queue (no accounting impact — pool memory usage
  is already tracked via `total_size * kv_size` in `total_memory_usage`).
  Prevents unbounded growth of the event queue over the simulation
  lifetime.

### Fixed
- **Model-architecture YAML docs matched the code again**
  ([#52](https://github.com/casys-kaist/LLMServingSim/issues/52)). The
  `adding-model-architecture` page documented `cls:`, `category:`,
  `tp_collective:` and `ep_collective:` — none of which exist. `LayerEntry`
  is `extra="forbid"`, so the documented example was rejected with 13
  validation errors; nobody could follow the page. The real schema is
  `vllm:` for the class name, profile kind as the *catalog block* a layer
  sits in, and `within:`/`tp_stable:` as the only other fields. `within:`
  was missing entirely, which is what the report called out: it is the
  ancestor-class filter that lets one `RMSNorm` entry serve the input,
  post-attention and final norms. Also corrected: `attention` is listed
  explicitly in `sequence.pre_attn` rather than implicit, `lm_head` maps to
  `LogitsProcessor` not `ParallelLMHead`, Llama 3 uses
  `Llama3RotaryEmbedding`, and the MoE catalog names the sparse block
  (`Qwen3MoeSparseMoeBlock`) not `FusedMoE`. The page's example is now
  checked against the pydantic models.
- Doc/code alignment sweep: `_lookup_attention_with_skew` was described as
  always doing "two 4D lookups" in five places (AGENTS.md, `serving/README.md`,
  its own docstring, and two docs pages) when the second lookup is
  conditional — it returns `t_mean` after one lookup for `n_decode <= 1`, for
  a batch whose decode kv lengths are equal, or for `alpha == 0`, which is now
  the default without a skew profile. Also fixed the `Workload.cc` path
  (`system/` -> `workload/`) and the PIM config paths, which pointed at
  directories where `configs/pim/*.ini` are files.
- `outputs/*` (except the committed `outputs/example_*.csv`) and
  `astra-sim/inputs/runs/` are now gitignored. `AGENTS.md` claimed output CSVs
  and generated traces already were; they were not, so scratch from every run
  accumulated in `git status`, and `--no-cleanup-inputs` could leave gigabytes
  of ASTRA-Sim inputs behind.
- Documentation corrections: the attention lookup was described as
  "nearest-neighbour on `(prefill_chunk, n_decode)`" when it has always
  bracketed and interpolated both axes; the trace file was described as
  tab-separated when `utils.py::_FMT` emits fixed-width space-padded columns;
  and the `SKIP_SKEW=1` fallback alpha was documented as "roughly 0.3 across
  observed hardware", a figure no bundle in the repo reproduces.
- P/D disaggregation shipped 3x too much KV. `convert_prefill` took the
  per-layer SEND/RECV size from the `*v_proj` layer's `output_size`, i.e. the
  whole QKV activation, so it transferred Q as well — a factor of
  `(q_dim + 2*kv_dim) / (2*kv_dim)`: 3x for Llama-3.1-8B, 1.5x for MHA, more at
  wider GQA ratios — and it ignored `kv_cache_dtype`, making
  `--kv-cache-dtype fp8` 6x high. The frontend now puts the per-layer, per-rank
  K+V bytes in the trace's `comm_size` column and the converter reads that. The
  count also includes a request's prefix-cache hit on its first step: the decode
  side needs that KV even though the prefill side read it from cache.
- Preemption freed nothing. The loop guarded on `gen_req[-1].is_prefill()` over a
  list built from non-prefill requests, so the condition never fired: requests
  were marked evicted and the batch shrank, but not a byte was released. This is
  why a 24 GiB configuration could crash in `apply_kv_cache_events -> allocate`.
- `kv_cache_pct` was 0.0 in every bench `timeseries.csv`. `SchedulerStats` has no
  `gpu_cache_usage` field; it is `kv_cache_usage`, and a `getattr` default hid
  the mismatch. The attribute is now read directly so a future rename fails
  loudly.
- Block hashes were unhashed at the prefix level: `hash(tuple(page_tokens))`
  meant two different prefixes ending in the same 16 tokens collided (53
  duplicates observed on a 300-request ShareGPT replay). Hashes are now chained
  through the parent, as in vLLM.
- `TTFT` could be overwritten when a request resumed after preemption, because
  `set_ttft` ran again on the recomputed prefill. It is now recorded exactly
  once, gated on `is_init`.
- `Scheduler` defined `schedule_with_prefix` twice; the first definition (228
  lines) was shadowed and never ran.
- Chunked prefill double-counted prefix-cache hits. In
  `schedule_with_prefix`, `chunk_size = original_input - num_computed_tokens`
  already excludes prefix-cached tokens (because `num_computed_tokens`
  is bumped to `prefix_cache_hit` on the first `prefix_match`). The
  scheduler then accumulated `hit_len += prefix_hit` on top of that,
  and `_build_batch_ctx` (trace_generator.py) subtracted the prefix
  hit a second time — collapsing `total_len` to 1 for any prefill
  chunk with prefix caching on. Dense-layer latency and TP collective
  sizing were both being looked up at 1 token instead of `chunk_size`.
  Fix: drop the second subtraction; sub-batch interleaving and the
  `Batch.hit_len` field were removed as part of the cleanup.
- `_make_sub_batch` (sub-batch interleaving) was not chunked-prefill
  aware: it used `req.is_init` (later chunks have `is_init=False` and
  would be misclassified as decode), `req.input` (full prompt length
  instead of this step's chunk), and `prefill_k_list=0` (ignoring KV
  already produced by prior chunks). It also failed to reset
  `prefill_q_list` / `prefill_k_list` / `decode_k_list` between the
  two sub-batches, leaking batch1 state into batch2. Now reads
  `batch.scheduled_tokens` (set by the scheduler), keys off
  `req.is_prefill()`, and uses `req.num_computed_tokens` for KV
  already in cache.
- `MemoryModel.evict_prefix_cache` over-evicted the second-tier
  (CPU/CXL) cache by `num_npus`× because `space_needed` was computed
  with the per-rank `self._bytes_per_token` while each second-tier
  token represents full-cluster bytes (`per-rank × num_npus`). Now
  uses the cache's own `kv_size` for the per-token bytes (per-rank for
  NPU, full-cluster for second-tier). TP=1 unaffected; TP>1 prefix
  hit rates were collapsing as the storage tier was over-evicted on
  every spill.
- `MemoryModel.evict_prefix_cache` early-return guard required *both*
  `not enable_prefix_caching` AND `bytes <= 0`. Changed to `or` — the
  intent is to return early if either condition holds.
- NPU→CPU offload alloc/free in `scheduler.py` used per-rank bytes
  while prefix-cache events tracked full-cluster bytes
  (`get_kv(tlen) * num_npus`). At TP>1 `cpu_used` drifted between
  the two paths. Offload paths now scale by `num_npus` to match the
  existing CPU accounting convention so `cpu_used` is consistently
  full-cluster bytes per instance.
- `MemoryModel.storage_cache_evicted_req` called
  `npu_prefix_cache.inc_lock_ref(new_last_node)` where
  `new_last_node` belongs to the **second-tier** prefix tree.
  Walking up parents from a foreign-tree node never reaches
  `npu_prefix_cache.root_node` and ultimately dereferences `None`,
  crashing the simulator when evicting from NPU to CPU/CXL storage
  with prefix caching on. Now uses the correct tree (PR #25).
- `MemoryModel.avail_size` returned `RadixCache.avail_size() *
  self._bytes_per_token`, but `RadixCache.avail_size()` already
  returns bytes (`capacity - total_memory_usage()`). The extra
  multiplication produced a meaninglessly large value, making
  scheduler decisions based on it (e.g.
  `avail_size + evictable_size`) under-conservative even at TP=1.
  Now passes the byte value through unchanged (PR #25).
- Hardcoded `131072` bytes-per-token (Llama-3.1-8B bf16-specific)
  in five sites in `serving/__main__.py` (prefix-pool creation +
  CPU/CXL usage display) replaced with model-aware values: pools
  now build via `full_cluster_kv_bytes_per_token` at startup, and
  display lines use each `RadixCache`'s own `kv_size`. Fixes
  utilization readout for non-Llama-3.1-8B models (Qwen3 family,
  etc.).
- Tuple-unpacking crash in the CXL + prefix-sharing display path:
  `for i, cxl_id, cxl_pool in enumerate(prefix_pools):` would
  raise `ValueError: not enough values to unpack` because
  `enumerate()` yields 2-tuples. Replaced with proper 2-element
  unpacking.
- Refreshed validation baselines + website plots after the
  chunked-prefill + prefix-cache fix. Means / P99s now slightly
  over-predict vLLM instead of slightly under-predicting (the
  prior under-prediction came from dense layers being looked up
  at 1 token whenever a prefill chunk had any prefix-cache hit).
  All three bundled configurations still land within ~2.5% on
  TTFT / TPOT / latency means.

### Security
- Bump `fast-uri` to ≥3.1.2 (CVE-2026-6321 path traversal via
  percent-encoded dot segments + CVE-2026-6322 host confusion via
  percent-encoded authority delimiters, both rated High). Pinned in
  `pnpm.overrides` since the package ships as a transitive
  Docusaurus dependency.
- Bump `@babel/plugin-transform-modules-systemjs` to ≥7.29.4
  (GHSA-fv7c-fp4j-7gwp, CVE-2026-44728, High). Arbitrary code
  generation when compiling malicious input; affects 7.12.0–7.29.3.
  We shipped 7.29.0 via `@docusaurus/preset-classic`. Pinned in
  `pnpm.overrides`.
- Bump `serialize-javascript` to ≥7.0.5 (Dependabot, XSS via
  deferred function / regexp serialization). Pulled in transitively
  by `copy-webpack-plugin` and `css-minimizer-webpack-plugin` in
  Docusaurus 3.10.
- Bump `uuid` to ≥14.0.0 (Dependabot, missing buffer bounds check
  in v3/v5/v6 when `buf` is provided). Replaces both transitive
  8.3.2 (via `sockjs`) and 11.1.1.

## [v1.1.0] - 2026-04-26

### Added
- New vLLM-based layerwise profiler (`profiler/`) replacing the old `llm_profile/`
  module. Uses vLLM's built-in `layerwise_profile()` via a worker extension class to
  capture per-layer CUDA kernel timings from real vLLM execution paths. Architecture
  is dispatched by the HF config's `model_type` against YAML catalogs under
  `profiler/models/`, and each run emits a per-category CSV bundle
  (`dense.csv`, `per_sequence.csv`, `attention.csv`, and `moe.csv` for MoE) under
  `perf/<hw>/<model>/<variant>/tp<N>/`, with latencies in microseconds.
  The base layerwise-profile methodology — driving a real vLLM engine via a worker
  extension class and emulating TP=N on a single GPU by sharding `hf_overrides` — is
  adapted from [@waneon](https://github.com/waneon).
- Unified 4D attention profiling (`attention.csv`) replacing the earlier
  prefill/decode-separated scheme with a single table over
  `prefill_chunk × kv_prefill × n_decode × kv_decode` that matches what
  vLLM's chunked-prefill scheduler actually produces each step.
  Geometric axes with `ATTENTION_CHUNK_FACTOR` / `ATTENTION_KV_FACTOR`
  (default 2.0 = doubling) tune density against profile time
- Skew profiling + 5-axis alpha fit for heterogeneous-decode attention
  (`profiler/core/skew.py`, `fit_alpha.py`). The sweep fires bimodal
  decode batches and measures `(t_mean, t_max, t_skew)` per case; `fit_alpha`
  then groups rows by a 5-axis key `pc | n_label | skew_rate_label |
  kv_big_label | kp_label` and runs weighted least-squares per cell.
  At query time the simulator blends two uniform-attention lookups via the
  fitted alpha to recover the FlashAttention tile-padding / SM-imbalance
  penalty the uniform grid can't see (`serving/core/trace_generator.py`
  `_lookup_attention_with_skew` / `_skew_alpha`). Axis ablation on the
  widened ~13k-sample dataset picked the 5-axis scheme over the earlier
  3-axis fit (test p50/p90 ≈ 2.7% / 14.8% vs 3.5% / 16.4% on TP=1)
- Data-derived bucket axes for the skew fit. `n` and `kp` buckets are one
  per unique profiled value (+ `kp=0` sentinel + overflow); `kv_big` uses
  log-4x bins adapted to the observed max; `skew_rate` is a fixed
  normalised [0, 1] scheme; `pc` is keyed raw. Derived axes are written
  to `meta.yaml::skew_fit.bucket_axes` and the simulator reads them from
  there, so widening `MAX_NUM_SEQS` or `ATTENTION_MAX_KV` lights up finer
  resolution without any simulator code change
- Per-axis skew density knobs: `SKEW_N_FACTOR` / `SKEW_PC_FACTOR` /
  `SKEW_KP_FACTOR` / `SKEW_KVS_FACTOR` (CLI: `--skew-*-factor`, default
  2.0 = doubling). Crank higher to coarsen a given axis and cut profile
  time; effective values land in `meta.yaml::skew_profile.factors`
- Per-TP `skew_fit.csv` file spills the full per-bucket alpha table out
  of `meta.yaml` so the latter stays readable (~100 lines vs ~3100 lines
  for Qwen3-32B at 2 TPs). `meta.yaml::skew_fit.per_tp[tp].bucket_table`
  points at `tp<N>/skew_fit.csv`; the simulator hydrates it back into
  `alpha_by_bucket` on `_load_perf_db()`
- Compact `attention_grid` / `skew_profile` grid specs in `meta.yaml`
  (e.g. `"0, 16-2048 x2"` instead of the full value list)
- RTXPRO6000 (NVIDIA RTX PRO 6000 Blackwell) hardware support: 96 GB, 1597 GB/s,
  600W TDP
- DP+EP (Data Parallel + Expert Parallel) support with ASTRA-Sim ALLTOALL synchronization
  via `involved_dim` dimension scoping. Instances with the same `dp_group` share a single
  ASTRA-Sim process; the 2D topology `[tp_size, dp_group_size]` enables per-dimension
  collective routing (ALLREDUCE on TP dim, ALLTOALL on DP dim)
- Wave synchronization for DP groups: Python-side `dp_pending` barrier ensures all instances
  schedule before trace generation. ALLTOALL `comm_size` synchronized to `max(total_len)`
  across the group. Dummy batches keep idle instances participating in ALLTOALL sync
- `single_node_moe_dp_ep_instance.json` cluster config for MoE with DP+EP
  (2 instances, TP=1, EP=2, same DP group)
- Agentic session support for closed-loop workloads (e.g., SWE-bench). The new JSONL
  format uses `sub_requests` arrays with `tool_duration_ns` to model dependency chains
  where each LLM call waits for the previous one to complete plus tool execution time.
  The router dynamically releases sub-requests as their predecessors finish, enabling
  accurate simulation of multi-step agentic workflows
- `--num-reqs` CLI argument (replaces `--num-req`), default changed from 100 to 0
  (load all entries from dataset). For agentic datasets, counts sessions not sub-requests
- Example SWE-bench agentic dataset (`workloads/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl`)
- Qwen3-32B and Qwen3-30B-A3B-Instruct-2507 model configs with explicit `head_dim`
  support for models where `head_dim != hidden_size // num_attention_heads`
- FP8 KV cache simulation support (`--kv-cache-dtype fp8`): selects `profile_fp8.csv`
  for compute latency lookup and halves KV cache memory usage in the memory model
- FP8 KV cache profiling support (`kv_cache_dtype: "fp8"` in receipts, outputs
  `profile_fp8.csv`)
- Chunked prefill support (enabled by default, matching vLLM v1) with
  `--long-prefill-token-threshold` for per-request token cap per step
  (chunked prefill core by [@HyunsuYEE](https://github.com/HyunsuYEE))
- Chunked prefill compatible with prefix caching (RadixAttention)
- Prefix cache lock tracking (`_prefix_locked`) to prevent incorrect eviction during
  multi-chunk prefill
- Non-Docker vLLM installer (`scripts/install-vllm.sh`) using `uv` with
  precompiled vLLM 0.19.0 wheels ([@junwha](https://github.com/junwha))
- End-to-end vLLM benchmark + simulator validation suite (`bench/`,
  invoked as `python -m bench {run,validate}`). `bench run` replays a
  workload through a real vLLM `AsyncLLM` engine with `output_toks`
  pinned via `SamplingParams(min_tokens=N, max_tokens=N, ignore_eos=True)`
  so results are bit-for-bit comparable to the simulator's view of the
  same dataset. A custom `vllm.v1.metrics.loggers.StatLoggerBase` writes
  per-tick scheduler / iteration stats; `RequestStateStats` from
  `vllm.v1.metrics.stats` lands in `requests.jsonl`. `bench validate`
  loads a finished run plus the simulator's `sim.csv` / `sim.log` and
  emits throughput, running/waiting, and TTFT/TPOT/latency-CDF plots
  plus a numeric diff% summary
- Workload generators (`workloads/generators/`, invoked as
  `python -m workloads.generators sharegpt …`). Multi-turn ShareGPT
  parser with running context accumulation; default source
  `shibing624/sharegpt_gpt4`. Runs in tokenizer-only mode by default
  (output IDs from the assistant turn) or with `--use-vllm` to drive an
  offline batched `vllm.LLM` for free-generated outputs at maximum
  throughput. Optional `--fix-len` (random fixed-length tokens) and
  `--pulse` (bursty arrivals) modes
- Per-model invocation templates under `workloads/examples/`
  (`gen-llama-3.1-8b.sh`, `gen-qwen3-30b-a3b.sh`, `gen-qwen3-32b.sh`)
- Module READMEs for `bench/`, `scripts/` (top-level wrappers for the
  vLLM and simulator container launchers, the bare-metal vLLM installer,
  and the ASTRA-Sim build)
- Rich-backed logger shared between simulator, profiler, and bench
  (`serving/core/logger.py`, `profiler/core/logger.py`,
  `bench/core/logger.py`).
  Keeps the original `[HH:MM:SS.mmm] [Component] [node=X,inst=Y] LEVEL msg`
  line shape via a custom ``_RichSimHandler`` (public API unchanged —
  ``configure_logger`` / ``get_logger`` / the ``ComponentLoggerAdapter``
  still work for every existing call site) and adds:
  - ``.success()`` (green ✓ at INFO) and ``.summary()`` (verbatim,
    no prefix) on the adapter, plus module-level ``print_banner()`` /
    ``print_input_config()`` / ``print_markup()`` / ``print_rule()``
    and ``stage(title)`` / ``progress(label, total)`` context managers
    mirroring the profiler's helpers.
  - Rich theme + ``soft_wrap=True`` so colour renders in interactive
    terminals, long lines stay on one logical row, and redirected
    files (``> out.log``, ``nohup`` …) get clean plain-text logs
    with no stray ANSI escape bytes. ``FORCE_COLOR=1`` still forces
    colour when an IDE terminal doesn't self-identify as a TTY.
  - Banner / logo / input-config / simulation-results blocks in
    `serving/__main__.py` migrated to the new helpers (with `bench/__main__.py`
    using the same banner / stage / progress conventions); heartbeat status tree
    (``├─`` / ``└─``) now builds each line as a string and emits
    via Rich markup for consistent colouring.
  - ``RadixCache.format_prefix_info()``,
    ``Scheduler.print_result()``, and
    ``PowerModel.print_power_summary()`` rewritten around the new
    helpers. ``serving/utils.py`` loses its ANSI colour
    wrappers (``cyan`` / ``bold`` / ``ANSI_*`` / …) and the logo /
    input-config renderers now live in ``logger.py``
- READMEs for `configs/model/`, `configs/pim/`, `workloads/`, `serving/`
- `.gitignore` entries for AI agent cache files (`.claude/`, `.cursor/`, `.copilot/`,
  `.codex/`, `.aider*`, `.continue/`)

### Fixed
- Skew sweep feasibility filter used strict `n_reqs >= max_num_seqs` and
  dropped every `n = MSQ` case (including the pure-decode corner the
  attention sweep was already allowing). Relaxed to `>` to match
  attention and unlock pure `n = MSQ` shots. Mixed-regime `n = MSQ`
  (requires MSQ+1 requests) still filtered; profile with `MAX_NUM_SEQS`
  one above runtime MSQ to cover that corner too
- Missing `prefix_match` call on non-chunked prefill path: prefix cache hits were not
  detected for full prefill requests, preventing prefix caching benefits when chunked
  prefill was disabled ([@junwha](https://github.com/junwha))
- Typo in timer reference in legacy Mixtral profiler model
  ([@junwha](https://github.com/junwha))
- Prompt throughput now includes prefix cache hit tokens. Previously only actually
  computed prefill tokens were counted, making throughput appear lower than vLLM's
  reported prompt throughput when prefix caching was active
- Prefix cache `is_init` never cleared for full prefix cache hits, causing
  `total_requested_tokens` to inflate on every decode step and `lock_ref` leaks
- Prefix cache `lock_prefix` not called for full prefix hits, causing memory leaks
  at simulation end
- MoE expert latency aggregated both EP ranks onto one GPU (2x overestimate);
  now each GPU uses only its own rank's tokens and activated experts
- MoE weight calculation in `memory_model.py` now uses `ep_size` (not `tp_size`)
  for expert weight sharding
- Status print timing: only prints on start NPU to avoid transient "0 running" states
- `system.json` collective implementations now match topology dimensions (2 entries
  for 2D topologies) — previously 1 entry caused ASTRA-Sim to create only 1 dimension
- DP group termination: instances wait for all DP members to finish before marking done
- `argparse` `allow_abbrev=False` to prevent silent prefix matching of wrong arguments
- Add missing `return parser.parse_args()` in legacy profiler layers/main.py
  (reported and fixed by [@junwha](https://github.com/junwha), [@gleb-kun](https://github.com/gleb-kun))

### Changed
- `--fp` flag replaced with `--dtype` (vLLM-style: `float16`, `bfloat16`, `float32`,
  `int8`)
- `--gen` flag replaced with `--skip-prefill` for clarity
- `--request-routing-policy` default changed from `RR` to `LOAD` (vLLM-style weighted
  least-loaded). Requests are now routed in real-time based on current system state
  instead of upfront assignment
- `--expert-routing-policy` `FAST` renamed to `COPY` for clarity (enables block copy)
- Cluster config: `npu_num`/`npu_group` replaced with `tp_size`/`pp_size`/`ep_size`/`dp_group`.
  Partial configs supported (e.g., `num_npus=4, tp_size=2` infers `pp_size=2`).
  TP and EP share the same GPU set; DP via multiple instances with same `dp_group`
- MoE modeling: per-EP-rank latency lookup (`key_0=local_tokens, key_1=activated_experts`),
  even expert-to-rank partitioning, ASTRA-Sim ALLTOALL with `involved_dim` for cross-DP sync
- MoE `calculate_sizes`: uses `moe_intermediate_size` (per-expert FFN dim) separate from
  `intermediate_size` (dense FFN dim)
- `calculate_sizes` parameter renamed: `tp` → `parallel` (generic for TP or EP)
- Trace `comm_type` now supports dimension scoping: `ALLREDUCE:1,0`, `ALLTOALL:0,1`
- Network topology for DP groups: `npus_count: [tp_size, dp_group_size]` with per-dimension
  collective implementations in `system.json`
- Removed analytical ALLTOALL workaround functions (`_inflate_comm_size`,
  `_ring_alltoall_time_ns`, `_bw_gb_to_bpns`) — replaced by native ASTRA-Sim ALLTOALL
- `link_bw`/`link_latency` removed from `TraceCtx` and `generate_trace` (no longer needed
  for analytical fallback)
- Latency lookup extrapolates beyond profiled range instead of clamping for improved
  accuracy on large batch sizes
- Profiler rewritten from PyTorch Profiler + scikit-learn predictor to direct vLLM
  `layerwise_profile()` approach. Architecture yamls live in `profiler/models/`
  keyed on the HF config's `model_type`; CLI flags match vLLM (`--dtype`,
  `--kv-cache-dtype`, `--max-num-batched-tokens`, `--max-num-seqs`, `--tp`,
  `--variant`). Docker pinned to vLLM v0.19.0 (`vllm/vllm-openai:v0.19.0` or
  `v0.19.0-cu130` for CUDA 13.x)
- Old profiler preserved under `profiler/v0/` for reference
- Layer names unified between profiler and simulator: `qkv_projection`, `o_projection`,
  `ffn1`, `ffn2`, `attention`, `layernorm` (old names removed)
- `memory_model.py` updated to use explicit `head_dim` and `q_dim`/`kv_dim` for correct
  tensor size computation on models like Qwen3
- `trace_generator.py` rewritten with composable helpers (`TraceCtx`, `BatchCtx`,
  `_emit_layer`, `_emit_pre_attn_layers`, `_emit_post_attn_layers`) and unified profile
  CSV lookup with 2D bilinear interpolation
- Sampler output location changed to `REMOTE` (was on `lm_head`) to match Chakra
  converter's MEM_STORE node placement
- Removed `--enable-attn-prediction` flag (scikit-learn predictor replaced by direct
  profiled latency lookup)
- Cluster configs updated to RTXPRO6000 hardware specs
- `AGENTS.md` expanded with full repo structure, simulation flow, trace format
  documentation, and additional pitfalls
- `--max-batch` renamed to `--max-num-seqs` (default: 128, matching vLLM);
  now limits total running requests across inflight batches
- `--enable-chunked-prefill` now enabled by default (matching vLLM v1);
  use `--no-enable-chunked-prefill` to disable
- `--enable-prefix-caching` now enabled by default (matching vLLM v1);
  use `--no-enable-prefix-caching` to disable
- Scheduler rewritten to use vLLM-style token-budget-based allocation for both
  chunked and non-chunked prefill paths (`schedule_base`, `schedule_with_prefix`)
- KV cache block allocation uses vLLM-style cumulative ceiling division
- Radix tree `cache_unfinished_req` now uses `num_computed_tokens` instead of
  `req.input`, enabling correct incremental caching across chunks
- Prefix cache memory accounting changed to free-before-allocate order
- Hash-to-length map in `memory_model.py` changed from `{hash: tlen}` to
  `{hash: [tlen, refcount]}` to handle duplicate block hashes
- All `Request` attributes now properly initialized in `__init__`; removed
  `getattr` fallbacks throughout scheduler and radix tree
- Directory restructuring:
  - `cluster_config/` → `configs/cluster/`
  - `model_config/` → `configs/model/`
  - `pim_config/` → `configs/pim/`
  - `dataset/` → `workloads/` (the directory holds ShareGPT-style
    request workloads consumed by the simulator and bench)
  - `output/` → `outputs/`
  - `script/` → `scripts/`
  - `llm_profile/` → `profiler/legacy_profiler/` (later moved to `profiler/v0/`)
- Top-level package layout finalized as Python-style sibling modules:
  - `inference_serving/` → `serving/` with internals under `serving/core/`
    (every `.py` previously at the package root now lives one directory
    deeper); entrypoint `main.py` becomes `serving/__main__.py` and is
    invoked as `python -m serving …`.
  - `llm_profiler/` → `profiler/` (collapses the duplicated
    `llm_profiler/profiler/` package layer) with internals under
    `profiler/core/` and `profiler/core/hooks/`.
  - `bench/` added with the same shape (`bench/core/`).
  - `workloads/` ships the ShareGPT generator under
    `workloads/generators/sharegpt.py` (invoked as
    `python -m workloads.generators sharegpt …`) with per-model
    invocation templates under `workloads/examples/`. The package
    deliberately avoids the name `datasets/` so the HuggingFace
    `datasets` library imports cleanly.
  - Module-specific shell scripts live at the module home (e.g.
    `profiler/profile.sh`, `bench/bench.sh`, `serving/run.sh`); only
    cross-cutting environment / build helpers stay in `scripts/`
    (`docker-vllm.sh`, `docker-sim.sh`, `install-vllm.sh`, `compile.sh`).
- Evaluation configs moved from `config/` to `configs/` subdirectories within each
  figure folder
- `run.sh` updated with reorganized examples and commented out unavailable MoE config

### Removed
- `internal/` directory (debug docs and scheduler tests moved or removed)
- `scripts/` batch experiment scripts (superseded by `run.sh` examples)
- `evaluation/` directory (preserved on `ispass26-artifact` branch)
- `--enable-attn-prediction` flag and scikit-learn attention predictor
- `--fp` flag (replaced by `--dtype`)
- `--gen` flag (replaced by `--skip-prefill`)
- `--expert-routing-policy FAST` (renamed to `COPY`)
- `serving/attn_utils.py` (stale scikit-learn attention feature helper)
- `npu_num`/`npu_group` config fields (replaced by `tp_size`/`pp_size`/`ep_size`)
- `--num-req` flag (replaced by `--num-reqs`)
- Analytical ALLTOALL workaround functions (`_inflate_comm_size`, `_ring_alltoall_time_ns`)
- `evaluation/` directory (preserved on `ispass26-artifact` branch)

---

## [v1.0.0] - 2026-02-25

### Added
- Multi-instance simulation with configurable request routing policies (Round Robin, Random, Custom)
- Prefill/Decode (P/D) disaggregation support across instances
- Mixture of Experts (MoE) support with expert parallelism, expert offloading, and configurable
  routing policies (Round Robin, Random, Fast, Custom)
- Prefix caching using RadixAttention (based on SGLang), with support for second-tier prefix cache
  pooling across CPU and CXL memory (`--enable-prefix-caching`, `--enable-prefix-sharing`)
- Sub-batch interleaving to overlap prefill and decode phases within an iteration
  (`--enable-sub-batch-interleaving`)
- Attention latency predictor using scikit-learn for real-time per-request estimation
  (`--enable-attn-prediction`)
- Power and energy modeling per node covering NPU, CPU, DRAM, interconnect, NIC, and storage
- CXL memory expansion support with configurable bandwidth and latency
- Enhanced PIM (Processing-In-Memory) model with per-device INI configuration (`configs/pim/`)
- Cluster-level configuration system (`configs/cluster/*.json`) that consolidates all hardware,
  topology, and placement parameters into a single file
- Per-layer weight, KV cache, and expert placement rules in cluster config
- Additional latency metrics: ITL (Inter-Token Latency) and p99 for TTFT, TPOT, ITL
- Hardware performance profiles for TPU-v6e-1
- Batch experiment scripts for systematic evaluation (`scripts/`)
- Artifact evaluation scripts and reference results (`evaluation/`)
- `llm_profile` integrated as a local module with support for MoE models and power profiling

### Changed
- All hardware and topology parameters are now specified via `cluster_config` JSON files;
  per-invocation hardware arguments (`--model_name`, `--hardware`, `--npu_num`, etc.) are removed
- Command-line argument style changed from underscore to hyphen (e.g., `--cluster-config`,
  `--num-req`, `--block-size`)
- Dataset format changed from `.tsv` to `.jsonl`
- Build process consolidated into `./compile.sh` and `./docker.sh`
- Performance model directory relocated from `perf_model/` to `llm_profile/perf_models/`
- `serving/` modules renamed for clarity:
  - `control.py` → `controller.py`
  - `generate_graph.py` → `graph_generator.py`
  - `generate_trace.py` → `trace_generator.py`
  - `config_generator.py` → `config_builder.py`
  - `pim.py` → `pim_model.py`
- Fix incorrect `evict_size` accumulation

### Removed
- `trace_test/` directory (superseded by `evaluation/` scripts)
- Direct per-invocation hardware arguments (`--model_name`, `--hardware`, `--npu_num`,
  `--npu_group`, `--npu_mem`, `--remote_bw`, `--link_bw`)

---

## [v0.2.1] - 2025-07-18

### Added
- `llm_profile` module with PyTorch Profiler for GPU layer and attention latency measurement
- Llama-3.1-8B-Instruct model support (replaces GPT-3 6.7B as the default model)
- Hugging Face model configuration support for easy addition of new models

### Changed
- Function names standardized to snake_case (e.g., `createNetworkConfig` → `create_network_config`,
  `calculateSizes` → `calculate_sizes`)
- Model configuration files updated to Llama-3.1-8B-Instruct format

### Fixed
- Collective operation stall caused by unresolved dependencies in the ASTRA-Sim workload graph
- Network dimension calculation for full pipeline parallelism (`npus_per_dim` formula corrected)

---

## [v0.2.0] - 2025-06-04

### Changed
- ASTRA-Sim submodule updated to latest version (branch `v0.2.0`)
- Chakra updated to latest version
- Network configuration format changed from JSON to YAML
- `local_bw` and `remote_bw` parameters replaced with `link_latency`
- Conda environment dependencies updated and simplified

---

## [v0.1.0] - 2025-01-03

### Added
- GPU performance model based on TensorRT-LLM profiling (replaces NPU simulator)
- Auto config generator for network and memory configurations
- New parameters: `--hardware`, `--local_bw`, `--remote_bw`, `--link_bw`, `--fp`
- Additional metrics: `queuing_delay`, TTFT, TPOT
- Verbose logging option for detailed execution output

### Changed
- ASTRA-Sim submodule branch updated from `artifact` to `v0.1.0`
- Output format changed from TSV to CSV

### Removed
- Polymath and codelets_src submodules (NPU simulator components replaced by performance model)

---

## [artifact] - 2024-06-23

### Added
- Initial project release as IISWC 2024 artifact: "LLMServingSim: A HW/SW Co-Simulation Infrastructure for LLM Inference Serving at Scale"
- NPU simulator-based co-simulation infrastructure (ASTRA-Sim + Polymath + codelets_src)
- Evaluation scripts and benchmark results
- Conda environment configuration (`environment.yml`)
