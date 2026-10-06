# Changelog

All notable changes to this project are documented in this file.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions.

## [v1.2.0] - 2026-10-06

Version 1.2.0 expands serving simulation, profiling and validation across dense,
MoE and sparse/hybrid models. Major changes include reworked scheduling and
tiered KV caching, TP/PP/DP/EP execution and communication corrections,
model-aware attention/skew and deployment-aware MoE profiling, and
speculative-decoding support.

Trace reuse and streamlined ASTRA-Sim/Chakra integration reduce simulation
overhead. Hardware profiling, RTX 4090 support, refreshed benchmark examples
and documentation accompany the vLLM 0.28.0 compatibility upgrade
([#76](https://github.com/casys-kaist/LLMServingSim/pull/76)).

Accuracy claims apply to the recorded bundles and workloads on the
[validation page](https://llmservingsim.ai/docs/validation), not every supported
architecture or newly acquired profile.

Existing checkouts must update the pinned submodules and rebuild ASTRA-Sim and
Chakra; see the [update instructions](https://llmservingsim.ai/docs/contributor/pr-workflow#publishing-submodule-changes).

### Added

- Architecture catalogs, tensor sizing and execution paths for DeepSeek-V3.2,
  GLM-5, MiniMax-M3 and Qwen3.5/3.6/3.8 families, including sparse and hybrid
  linear attention. See the [sparse](https://llmservingsim.ai/docs/examples/model-families/sparse-attention)
  and [hybrid](https://llmservingsim.ai/docs/examples/model-families/hybrid-linear-attention) guides
  for requirements and validation limits.
- Modern-family profile bundles and a reduced DeepSeek-V3.2 diagnostic checkpoint
  and benchmark; these do not establish full-scale DeepSeek accuracy.
- Speculative decoding with configurable acceptance policies, multi-query
  attention profiles and MTP drafter time and memory accounting.
  [Usage and profile requirements](https://llmservingsim.ai/docs/examples/advanced/speculative-decoding).
- Deployment-matched DP+EP MoE component profiles with explicit TP/DP geometry,
  eager and CUDA graph measurements, resumable acquisition and coverage checks.
  Missing deployment coverage warns and falls back to the legacy EP table;
  invalid execution contracts and out-of-range component queries are rejected.
  [Supported contracts](https://llmservingsim.ai/docs/profiler/native-moe-components).
- Dynamic heterogeneous skew acquisition from token, request and KV bounds,
  with per-forward timings, resumable progress and CPU-only `plan-skew` and
  `refit-skew` commands.
- Standalone `profiler hardware` acquisition and shared `hardware.yaml` defaults
  for device specifications and measured NCCL links, with assumptions identified
  separately. Explicit cluster settings take precedence.
- Per-instance runtime overrides and capacity-normalized least-load routing,
  plus per-dimension link settings with matching collective dimensions
  ([#33](https://github.com/casys-kaist/LLMServingSim/pull/33),
  [#37](https://github.com/casys-kaist/LLMServingSim/pull/37),
  [#38](https://github.com/casys-kaist/LLMServingSim/pull/38)).
- Run-isolated backend inputs through `--run-id` and `--inputs-root`, output-path
  substitution and automatic cleanup of transient inputs; use `--save-trace-text`
  or `--keep-inputs` to retain readable traces or backend replay artifacts
  ([#43](https://github.com/casys-kaist/LLMServingSim/pull/43),
  [#51](https://github.com/casys-kaist/LLMServingSim/pull/51)).
- Optional per-operation analytical Ring links for AllReduce, AllGather and
  ReduceScatter, measured by hardware profiling. Cluster overrides take precedence
  over hardware defaults; unspecified operations retain the common link.
  [Configuration](https://llmservingsim.ai/docs/reference/cluster-config#collective-specific-links).
- Tiered KV block pools with chained prefix hashes, memory-utilization controls,
  full-sequence admission checks and startup capacity reporting.
- Optional MoE gate-statistics recording and `CUSTOM` routing from measured
  distinct-expert curves. Diagnostic gate recordings are not latency references.
- RTX 4090 Llama-3.1-8B TP1 profiles and a recorded benchmark example, contributed
  by [@Arifuzzamanjoy](https://github.com/Arifuzzamanjoy)
  ([#59](https://github.com/casys-kaist/LLMServingSim/pull/59)).
  RTX 4090 TP2 configs remain templates until that degree is profiled.
- Profiler coverage checks for missing or over-matched catalog bindings, and
  `serving/validate.sh` regression checks against recorded clocks and examples.
- Benchmark configuration-only checks, eager/dummy execution controls, explicit
  KV capacity and tokenizer-free startup, plus opt-in step-shape and expert
  diagnostics. Preemption and computed-versus-cached prompt work are recorded.
- An opt-in resource watchdog for child processes, memory, swap, timeouts and
  selected GPU telemetry; GPU selection through `VLLM_GPUS` in the vLLM launcher.
- Public documentation with local search, model-family guides and rendered-site
  checks. Release notes summarize final changes and link to detailed guides.
- A [contributors page](https://github.com/casys-kaist/LLMServingSim/blob/main/CONTRIBUTORS.md)
  crediting community code, documentation and issue analysis.

### Changed

- Upgrade profiling and benchmarking from vLLM 0.19.0 to 0.28.0, including V1/V2
  runner and MoE-router integration. Benchmark recording defaults to NCCL-only
  collectives and preserves effective engine, compilation and placement settings.
- Measure ordinary and skew latency using per-call CUDA activity unions.
  Initialize assigned KV pages with vLLM's deterministic dummy initializer outside
  timing; CPU preparation and gaps are not added to simulated GPU work.
- Derive profiling coverage and category-specific layer counts from the model and
  resolved engine limits; the default attention KV reach follows model context.
  Checkpoint ordinary sweeps and preserve unswept categories, TP/EP coverage,
  query-length axes and authoritative main-engine geometry during partial refreshes.
- Densify default token and sequence grids, including whole-block MoE, and use
  configurable square-root-of-two attention axes. Skew density remains separately
  configurable; denser acquisition increases profiling time.
- Align `profile.sh` with CLI defaults and expose TP/DP and skew-only options.
  `profile-all.sh` accepts per-model overrides and reports failed jobs without
  abandoning the remaining campaign.
- Share per-layer stack and catalog resolution between profiler and simulator,
  using one catalog per model family rather than per checkpoint shape.
  Account for MLA, sparse-indexer caches, linear-attention state and the heaviest
  pipeline stage instead of assuming a uniform dense model.
- Use linear attention-grid interpolation and query-weighted prefill coordinates.
  Skew uses per-kernel, reference-aligned offline tables with bounded runtime
  lookup; absent skew data applies no correction.
  [Calibration and migration](https://llmservingsim.ai/docs/profiler/skew-alpha-fit).
- Bound sparse-attention lookup by each kernel's checkpoint-defined key reach,
  while indexer work still scales with the full history. Apply caps before
  aggregating requests; the longest sequence is not always the slowest reference.
- Key whole-block MoE profiles by EP degree, gathered token count and distinct
  active experts. Select an exact EP table when available, otherwise warn and
  use the nearest measured degree rather than interpolating across EP.
- Align continuous batching, preemption and prefix-cache recovery with vLLM.
  Hybrid prefix-cache chunks respect state-page boundaries; CUDA graph padding
  and DP synchronization remain separate from real attention and head geometry.
- Model asynchronous-scheduling admission lookahead, with per-instance control
  and idle recovery, instead of admitting every arrival at batch completion.
- Generalize the PIM latency estimate using head count, head dimension and GQA
  ratio. This remains a baseline-scaled estimate, not independent measurements
  for every architecture ([#45](https://github.com/casys-kaist/LLMServingSim/pull/45)).
- Reduce simulation overhead with in-process Chakra conversion, reusable graphs,
  lazy profile loading and fewer idle backend handshakes
  ([#67](https://github.com/casys-kaist/LLMServingSim/pull/67)).
- Organize benchmark examples by hardware/model with recorded configs and KV
  block sizes. Refresh RTXPRO6000 Llama and NCCL-only Qwen references, native
  Qwen3-30B component tables, hardware links and validation artifacts; keep the
  RTX 4090 vLLM 0.19.0 reference identified separately.

### Fixed

- Prevent pipeline deadlocks by splitting traces at transformer-block boundaries,
  and prevent duplicate request completion with multiple in-flight batches
  ([#55](https://github.com/casys-kaist/LLMServingSim/issues/55),
  [#62](https://github.com/casys-kaist/LLMServingSim/issues/62)).
  Reported by [@hu-op1](https://github.com/hu-op1) and
  [@hsule](https://github.com/hsule). Reject PP stages exceeding the layer count
  and unsupported PP with sub-batching; regenerate old PP traces with stage boundaries.
- Fix dense DP validation and DP+TP/PP synchronization when members drain
  unevenly, and isolate collectives between independent non-DP instances.
  Snapshot speculative state per batch to prevent rollback deadlocks.
  DP synchronization reports: [@hsule](https://github.com/hsule),
  [#65](https://github.com/casys-kaist/LLMServingSim/issues/65).
- Correct prefix-hit accounting, preemption, cross-tier allocation/eviction and
  sub-batch geometry. Preserve TTFT across recomputation and correct P/D handoff
  accounting; transfer per-rank K+V bytes rather than the full QKV activation.
  Cache fixes include contributions from @horser1, @shermanjlim and @Veilwalker
  ([#49](https://github.com/casys-kaist/LLMServingSim/pull/49)).
- Include TP embedding reduction and logits gathering with correct vocabulary
  padding and tensor sizes. Idle DP forwards omit the head; terminal host stores
  contain output token IDs rather than logits.
- Preserve ordered MoE tensor collectives and dependencies through Chakra and
  dimension-scoped backend numbering. Correct global EP-rank lookup,
  round-robin token assignment, group-limited distinct-expert routing and
  small-batch rounding that could incorrectly charge zero expert work.
- Fit collective links using ASTRA-Sim's Ring phases, rank count, GiB/s units
  and separate local reduction costs; record fit residuals and assumptions.
  Primitive calibration does not model exact grouped NCCL execution.
- Correct vLLM 0.28 timing normalization, top-level deduplication and CUDA
  ownership, including whole-block MoE invocation counts. Capture the V2
  sampler through the layerwise profiler without adding GPU computation.
- Measure Qwen3 query/key normalization per TP degree instead of copying TP1
  timings, and shard linear-attention state consistently during TP emulation.
  Preserve resolved KV block sizes separately for each TP degree.
- Retain native top-k work in forced-routing MoE profiles and avoid
  double-counting sparse-indexer kernels. Re-profile older forced-routing or
  hybrid MoE tables and affected sparse-indexer glue rows.
- Preserve profiled decode request boundaries, fractional resume coordinates,
  multi-query feasibility and skew batch identity. Reject incompatible or
  corrupt acquisitions instead of silently resuming them.
- Correct per-kernel attention/skew lookup, regime-dependent linear attention,
  MTP attribution, quantization/variant resolution and architecture-specific
  tensor sizes. Disabling block-copy optimization now retains every layer.
- Restore MiniMax-M3 MTP startup and SM120 sparse-MLA prefill through targeted
  vLLM patches; correct container dependencies and Chakra protobuf compatibility.
- Repair the first-node dependency in PIM graphs, absolute cluster-config paths
  and runtime argument shadowing.
- Anchor benchmark TTFT and end-to-end latency at request arrival using an
  estimated epoch-to-monotonic clock offset; retain the queued-time fallback
  when conversion is unavailable. TPOT still uses token timestamps.
- Correct benchmark KV utilization and throughput reporting, preserve engine
  metadata before shutdown, and reject incomplete or perturbed validation inputs.
- Correct architecture-schema examples
  ([#52](https://github.com/casys-kaist/LLMServingSim/issues/52)), README layouts,
  setup instructions and documentation rendering. Complete CLI indexes and
  align option defaults and subcommand scope. Align MTP costs, shipped
  sparse profiles, accelerator-acquisition scope and metric definitions with
  current code; keep public guides focused on supported behavior and limits.

### Removed

- Legacy skew fitting and lookup on the enabled simulator path; use
  `profiler refit-skew` for compatible raw data. Incompatible acquisition
  protocols require remeasurement, not a CSV format conversion.
- Simulator `--dtype` / `--kv-cache-dtype` flags; dtypes now come from the model
  config. Profiler and benchmark
  precision options remain available.
- `--prioritize-prefill` and its instance override; requests follow the unified
  continuous-batching scheduler.
- The catalog `sequence:` format; use `blocks:` and `shared:`. Replace custom
  imports of the old radix-tree cache with the tiered block-pool APIs.

### Security

- Update documentation dependencies: `fast-uri` to at least 3.1.2
  (CVE-2026-6321 / CVE-2026-6322), `@babel/plugin-transform-modules-systemjs`
  to at least 7.29.4 (CVE-2026-44728), `serialize-javascript` to at least 7.0.5
  and `uuid` to at least 14.0.0 for their security fixes.

## [v1.1.0] - 2026-04-26

### Added
- New vLLM-based layerwise profiler (`profiler/`) replacing `llm_profile/`. Drives
  vLLM's built-in `layerwise_profile()` through a worker extension to capture per-layer
  CUDA kernel timings from real execution paths, dispatching on the HF config's
  `model_type` against YAML catalogs in `profiler/models/`. Each run emits a per-category
  CSV bundle under `perf/<hw>/<model>/<variant>/tp<N>/`, latencies in microseconds. The
  base methodology — a worker extension plus TP=N emulation on one GPU via `hf_overrides`
  — is adapted from [@waneon](https://github.com/waneon)
- Unified 4D attention profiling (`attention.csv`) replacing the earlier
  prefill/decode-separated scheme with a single table over
  `prefill_chunk × kv_prefill × n_decode × kv_decode` that matches what
  vLLM's chunked-prefill scheduler actually produces each step.
  Geometric axes with `ATTENTION_CHUNK_FACTOR` / `ATTENTION_KV_FACTOR`
  (default 2.0 = doubling) tune density against profile time
- Skew profiling + 5-axis alpha fit for heterogeneous-decode attention
  (`profiler/core/skew.py`, `fit_alpha.py`). The sweep fires bimodal decode batches,
  measures `(t_mean, t_max, t_skew)` per case and fits a per-bucket alpha by weighted
  least squares; at query time the simulator blends two uniform lookups through it to
  recover the FlashAttention tile-padding / SM-imbalance penalty the uniform grid cannot
  see. Axis ablation on ~13k samples picked 5 axes over the earlier 3 (test p50/p90
  ≈ 2.7% / 14.8% vs 3.5% / 16.4% at TP=1)
- Data-derived bucket axes for the skew fit: one bucket per unique profiled value for
  `n` and `kp` (plus sentinel and overflow), log-4x bins for `kv_big`, a fixed
  normalised scheme for `skew_rate`, raw `pc`. Written to
  `meta.yaml::skew_fit.bucket_axes` and read from there, so widening the sweep lights up
  finer resolution with no simulator code change
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
- End-to-end vLLM benchmark + simulator validation suite (`bench/`, invoked as
  `python -m bench {run,validate}`). `bench run` replays a workload through a real
  `AsyncLLM` with `output_toks` pinned via
  `SamplingParams(min_tokens=N, max_tokens=N, ignore_eos=True)`, so it is directly
  comparable to the simulator's view of the same dataset, and records per-tick
  scheduler stats plus `RequestStateStats`.
  `bench validate` diffs a finished run against `sim.csv` / `sim.log` and emits
  throughput, running/waiting and TTFT/TPOT/latency-CDF plots with a numeric summary
- Workload generators (`workloads/generators/`, invoked as
  `python -m workloads.generators sharegpt …`). Multi-turn ShareGPT parser with running
  context accumulation, default source `shibing624/sharegpt_gpt4`. Tokenizer-only by
  default, or `--use-vllm` to drive an offline batched `vllm.LLM` for free-generated
  outputs; optional `--fix-len` and `--pulse` (bursty arrival) modes
- Per-model invocation templates under `workloads/examples/`
  (`gen-llama-3.1-8b.sh`, `gen-qwen3-30b-a3b.sh`, `gen-qwen3-32b.sh`)
- Module READMEs for `bench/`, `scripts/` (top-level wrappers for the
  vLLM and simulator container launchers, the bare-metal vLLM installer,
  and the ASTRA-Sim build)
- Rich-backed logger shared between simulator, profiler and bench
  (`serving/core/logger.py` and siblings). Keeps the original
  `[HH:MM:SS.mmm] [Component] [node=X,inst=Y] LEVEL msg` shape and public API, adding
  `.success()` / `.summary()`, banner / input-config / rule printers and
  `stage()` / `progress()` context managers. Colour renders in interactive terminals
  while redirected output stays clean plain text (`FORCE_COLOR=1` forces it). Banners,
  the heartbeat status tree, `format_prefix_info()`, `print_result()` and
  `print_power_summary()` move onto the helpers; `serving/utils.py` loses its ANSI
  colour wrappers
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
  `inference_serving/` → `serving/` (internals under `serving/core/`, entrypoint
  `serving/__main__.py`, invoked as `python -m serving …`); `llm_profiler/` →
  `profiler/` (collapsing the duplicated package layer, internals under
  `profiler/core/`); `bench/` added with the same shape; `workloads/` ships the ShareGPT
  generator under `workloads/generators/`, deliberately not named `datasets/` so the
  HuggingFace library imports cleanly. Module-specific shell scripts live at the module
  home (`profiler/profile.sh`, `bench/bench.sh`, `serving/run.sh`); only cross-cutting
  environment / build helpers stay in `scripts/`
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
