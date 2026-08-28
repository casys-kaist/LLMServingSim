# Changelog

All notable changes to this project are documented in this file.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions.

## [Unreleased]

### Added
- `docs/scripts/check-rendered.mjs` — scans the built site for source syntax that
  survived into visible text (unparsed admonitions, bold, links, headings, table rows,
  doubled list markers, visible HTML comments, JSX brace leaks), plus a structural
  check on every mermaid diagram — unknown type, empty, unbalanced `subgraph`/`end` —
  read from the source, since theme-mermaid leaves nothing in the static HTML. Runs as
  a `postbuild` hook and as an explicit CI step, and exits non-zero.
  `pnpm check-rendered`.
- `serving/validate.sh` — 58 scenarios checked against recorded total clocks, plus md5
  checks on each `bench/examples` entry's `outputs/sim.csv` and
  `validation/summary.txt`, and a markdown report of anything that moved, ready to paste
  into a PR. `--help` for options, `--clocks-only` to skip the slow stage, `--update` to
  refresh `serving/validate-baselines.txt`. `serving/run.sh` is now one example per
  feature; coverage lives in the new script.
- `configs/cluster/single_node_dp_pp_instance.json` — dense `dp=2 x pp=2`, the DP+PP
  shape with no MoE collective.
- RTX 4090 profile bundle for `meta-llama/Llama-3.1-8B` (bf16, TP=1) with skew sweep,
  plus `configs/cluster/rtx4090_{single,tp2,multi}_instance.json` — the `tp2` one is a
  template, since only `tp1` is profiled for that card. Every TTFT / TPOT / latency
  metric lands within 1% of a real vLLM run on the same card. Contributed by
  [@Arifuzzamanjoy](https://github.com/Arifuzzamanjoy)
  ([#59](https://github.com/casys-kaist/LLMServingSim/pull/59))
- `bench/examples/` is keyed by `<hardware>/<model>` and each example carries its own
  `config.json`, replacing the parallel `configs/` tree. `run.sh` / `validate.sh`
  discover examples from the layout, so adding one needs no script edit
- Per-tier KV cache block pools (`serving/core/block_pool.py`) and a tiered manager
  (`serving/core/kv_cache_manager.py`), ported from vLLM v0.19.0. Each pool owns one
  tier's free list, prefix index and refcounts. Both carry a Docker-free self-test
- Chained block hashes (`hash(parent_hash, block_tokens)`) shared across tiers, as in
  vLLM's `offloading/scheduler.py`: one walk yields both the NPU and lower-tier hit
- `--npu-memory-utilization` (default `0.9`) with a per-instance `npu_mem.mem_util`
  override. KV capacity is `npu_mem * utilization - weight`, mirroring vLLM's
  `--gpu-memory-utilization`
- `--reserve-full-isl` (on by default, per-instance `reserve_full_isl`): admit a request
  only if its whole sequence fits, not merely its first chunk. Port of vLLM's
  `scheduler_reserve_full_isl`
- `python -m bench run` records vLLM's **resolved** configuration in `meta.json`:
  `kv_cache` (`num_gpu_blocks` is the number a simulator must match), `hardware` and
  `resolved_config`. All three are optional -- read them with `meta.get(...)`
- A `KV Cache Initialization` block in the startup output, listing each instance's
  derived block/token capacity and the utilization it came from
- `Request.num_tokens_reached` (prompt + generated), mirroring vLLM's
  `len(_all_token_ids)`. It cannot be derived from `num_computed_tokens`, which
  preemption resets to 0
- Public Docusaurus 3 docs site at [llmservingsim.ai](https://llmservingsim.ai), built
  from `docs/` and deployed via GitHub Actions Pages. Long-form content moves off the
  README, now a minimal front door; the split is documented in `AGENTS.md`
- Local docs search via `@easyops-cn/docusaurus-search-local` (Ctrl/Cmd-K). The index is
  built only in production -- use `pnpm build && pnpm serve` to test it
- `full_cluster_kv_bytes_per_token()` in `memory_model.py`, computing full-cluster KV
  bytes per token straight from an HF config. Avoids the per-rank roundoff in
  `get_kv(1) * num_npus` and works before any `MemoryModel` exists
- **Support for heterogeneous (hybrid) layer stacks in the profiler**, and the
  first architecture that needs it: `profiler/models/qwen3_5_text.yaml`
  (Qwen3.5 / Qwen3.8 text tower — gated DeltaNet interleaved with full
  attention, 3:1).
  - Architecture yamls may now declare `blocks:` + `shared:` instead of the
    flat `sequence:`; the two are mutually exclusive and validated as such, so
    the five existing yamls are untouched. `blocks` is keyed by **axis**
    (`attn.<layer_types value>`, `mlp.dense|moe`) rather than by block name,
    because a layer's identity in a modern stack is a tuple — Qwen3.8 varies
    the attention type, GLM and DeepSeek the MLP, MiniMax-M3 both — and naming
    every combination explodes. The per-layer values come from the
    checkpoint's own config, not from the yaml
  - `Architecture.layer_occurrences()` maxes across block types, which is
    correct because a layer belongs to one block type or is shared by all of
    them with the same count
  - New `catalog.linear_attention` group, and `catalog.attention` no longer has
    to hold exactly one entry: a sparse-attention model runs an indexer kernel
    beside the attention kernel on the same axes, so both belong in that group
- **`linear_attention` profile category** → `tp<N>/linear_attention.csv`, keyed
  `(layer, prefill_tokens, n_decode)`.
  - Two axes, not four: a gated-DeltaNet layer keeps a fixed-size conv state
    and a fixed-size recurrent state per sequence, neither a function of
    position, so **cost is independent of kv length and no skew correction
    applies** — measured, a 64x spread in kv length moves it 1.1% and a skewed
    batch is indistinguishable from a uniform one
  - Two axes, not one, even though the prefill and decode kernels *are*
    additive — because **which** kernel runs depends on the mix. A pure-decode
    batch runs a recurrent kernel; add a prefill chunk and vLLM switches to a
    fused-gating one, 4.5% apart at the same decode count. A pair of 1-D
    tables cannot represent a kernel-identity switch
  - Carries a `layer` column, unlike `attention.csv`, because the block runs
    several non-interchangeable kernels on the same axes
  - Decode requests in a linear-attention shot carry history (256 tokens) even
    though the cost does not depend on it. vLLM's *classification* does:
    `split_decodes_and_prefills` assumes a decodes-first batch and returns
    "no decode requests" outright when the first request's query length exceeds
    the threshold. A pure batch of 1-token requests takes an earlier fast path
    and needs no history, but a **mixed** batch does not — with zero-history
    decodes the whole batch was classified as prefill and the mixed-regime
    kernel never ran, leaving exactly the rows nothing else can supply empty
    (the mixed regime runs a different kernel from the pure one)
  - The prefill axis is sampled **chunk-aware**: inside the first chunk, at
    every chunk boundary, and at the token just past each boundary. Cost is a
    staircase, not a line — one token past a 64-boundary costs 13.5% more than
    the boundary itself and the interval to the next is nearly flat, so a plain
    geometric grid lands only on boundaries and interpolating between them
    underestimates most of what a chunked-prefill scheduler produces. Chunk
    length resolves from the model config's `chunk_size`, else vLLM's
    `FLA_CHUNK_SIZE`
- Profiler CLI knobs, for parity with the simulator and for hybrid stacks:
  `--block-size`, `--gpu-memory-utilization` (the simulator spells it
  `--npu-memory-utilization`), `--max-model-len`, `--num-hidden-layers`,
  `--linear-attn-chunk`, and `--hf-override KEY=VALUE`.
  `ProfileArgs.hf_overrides` had existed with no CLI to set it, so it was
  always `None`; values are JSON-parsed with a string fallback and dotted keys
  nest. `--num-hidden-layers` is the one hybrid profiling cannot do without:
  at the hardcoded 1, a hybrid catalog can only ever see one of its block types

### Fixed
- **`torch_dtype` vs `dtype`.** HuggingFace renamed the field; Qwen3.8's config
  carries only `dtype`, and both `ProfileArgs.effective_variant` and
  `trace_generator.resolve_variant` read only `torch_dtype`. The profiler wrote
  to a `default/` variant folder while the simulator looked for `bf16/` — a
  silent divergence that either raises `FileNotFoundError` or picks a folder
  that isn't the one measured. Both sides now read `torch_dtype` first, then
  `dtype`; every committed model config has only the former, so no variant
  folder changes and no baseline moves
- **DP groups no longer hang with `tp > 1` or `pp > 1`**
  ([#65](https://github.com/casys-kaist/LLMServingSim/issues/65)). A DP group only makes
  progress if every NPU of every member runs the same round, and `add_done` enforces
  that silently, so any path that creates or serves a batch the start NPU cannot claim
  deadlocks with no error. Six defects broke it, each invisible at `tp=pp=1` where an
  instance owns one NPU -- unregistered dummy batches, the solo workload path taken for
  a shared folder, `pp_size` missing from the topology, a `len(inflight) == 0` dummy
  gate, a servable batch with no `workload_name` yet, and `schedule()` refusing to let
  the start NPU join. `ep_size` was irrelevant. The topology is now `[tp, pp, dp]`
  innermost-first, matching vLLM, with `pp` dropped when it is 1. Reported by
  [@hsule](https://github.com/hsule)
- **Every admonition on the docs site rendered as raw text.** `:::caution Title` is
  Docusaurus v2 syntax; v3 needs `:::caution[Title]`, and the bare form is not
  recognised as a directive, so the block became a literal paragraph. 14 occurrences
  across 13 pages, with no build warning. All bracketed, and
  `docs/scripts/check-rendered.mjs` now fails the build on the whole class
- `configs/cluster/rtx4090_tp2_instance.json` was documented as runnable in
  `configs/cluster/README.md` ("not validated", i.e. runs without ground truth) and in
  the examples table. It raises `FileNotFoundError`: only `tp1` is profiled for RTX4090.
  Both now say it is a template until you profile the card with `TP_DEGREES=2`
- **`dp > 1` with `pp > 1` still hung once the members stopped draining together**
  ([#65](https://github.com/casys-kaist/LLMServingSim/issues/65), follow-up). Three
  single-slot assumptions, each correct at `pp_size == 1` and wrong once an instance
  holds `pp_size` batches: `schedule()` only let the start NPU join a formed batch when
  the pipeline was **full**; `dp_ready_workloads` held one workload per instance, so an
  NPU could run the wrong microbatch's graph; and `dp_pending[dg][inst]` held one batch,
  silently dropping a member's first from the barrier. Both maps are now FIFOs, and the
  join is tried before the depth cap. Needs members that drain at different times, which
  `example_trace.jsonl` cannot reach, so the four `*_uneven` scenarios in
  `serving/validate.sh` cover it. Reported by [@hsule](https://github.com/hsule)
- Data parallelism over a **dense** model was rejected at startup with
  `ep_size (1) not divisible by dp_group_size (2)`. `ep_size` is 1 there because a dense
  model has no
  experts to shard, so the EP divisibility checks now apply only to MoE.
  `configs/cluster/single_node_dp_instance.json` is the repro
- **`pp > 1` could complete the same request twice**
  ([#62](https://github.com/casys-kaist/LLMServingSim/issues/62)). A request is
  legitimately in more than one in-flight batch, and `add_done` ran the whole completion
  path per batch: `KeyError` in `cache_blocks` with prefix caching on, a duplicated CSV
  row with it off. `add_done` now skips requests already `FINISHED`, the same guard as
  vLLM V1's `update_from_output`. Reported by [@hsule](https://github.com/hsule)
- **Pipeline parallelism no longer deadlocks at most `pp_size` values**
  ([#55](https://github.com/casys-kaist/LLMServingSim/issues/55)). The Chakra converter
  split stages by *trace-line* count, so a boundary could land inside a transformer
  block, where the sending layer's `output_size` and the receiving layer's `input_size`
  are different tensors. ASTRA-Sim keys its send/recv tracker on `chunk_size`, so a
  mismatch never resolves and the receiver waits forever with no error.
  `trace_generator.py` now stamps `pp_stage_boundaries` into the trace header the way
  vLLM's `get_pp_indices` partitions blocks, and the converter consumes them. Reported
  by [@hu-op1](https://github.com/hu-op1), root cause narrowed down by
  [@hsule](https://github.com/hsule)
- `--enable-sub-batch-interleaving` is rejected with `pp_size > 1` instead of emitting a
  wrong graph: an interleaved trace leaves both sub-batches mid-block at every group
  edge. `pp_size` above `num_hidden_layers` is rejected too
- The end-of-iteration `MEM_STORE` node was sized from the sampler's `input_size` -- the
  full logits tensor -- billing a multi-megabyte write-back every iteration. vLLM V1
  ships only token ids, which is the sampler's `output_size`. Affects every simulation
- `--log-interval` above 1 second reported every windowed throughput as `0.0` and then
  crashed the summary with `ZeroDivisionError`, from floor division in the scale factor
- Chakra's `pyproject.toml` pinned `protobuf==6.*` while its checked-in `et_def_pb2.py`
  needs the 7.35.1 runtime, so a fresh `scripts/compile.sh` downgraded protobuf and left
  the converter raising `VersionError` on import. Hit any first-time setup
- `scripts/docker-sim.sh` never installed `rich`, which both loggers import -- it came in
  only transitively via `transformers`. `rich` is now declared, and the five packages
  nothing in this container imports (`transformers`, `datasets`, `msgspec`,
  `scikit-learn`, `xgboost`) are gone; `workloads/generators` asks for `transformers` by
  name rather than surfacing a bare `ModuleNotFoundError`
- **Model-architecture YAML docs match the code again**
  ([#52](https://github.com/casys-kaist/LLMServingSim/issues/52)). The page documented
  `cls:`, `category:`, `tp_collective:` and `ep_collective:`, none of which exist --
  `LayerEntry` is `extra="forbid"`, so the example was rejected with 13 validation
  errors. The real schema is `vllm:`, profile kind as the catalog block a layer sits in,
  and `within:` / `tp_stable:`; `within:` was missing entirely. Class names for
  `lm_head`, rotary embedding and the MoE block are corrected too, and the example is
  now checked against the pydantic models
- Doc/code alignment sweep: `_lookup_attention_with_skew` was described as always doing
  "two 4D lookups" in five places when the second is conditional. Also fixed the
  `Workload.cc` path and the PIM config paths
- `outputs/*` (except the committed `outputs/example_*.csv`) and
  `astra-sim/inputs/runs/` are now gitignored. `AGENTS.md` claimed they already were, so
  scratch from every run accumulated in `git status`
- Documentation corrections: the attention lookup was described as
  "nearest-neighbour" when it has always bracketed and interpolated; the trace as
  tab-separated when `_FMT` emits fixed-width space-padded columns; and the `SKIP_SKEW=1`
  fallback alpha as "roughly 0.3", a figure no bundle reproduces
- P/D disaggregation shipped 3x too much KV. `convert_prefill` sized the per-layer
  SEND/RECV from the whole QKV activation, so it shipped Q as well (3x for
  Llama-3.1-8B), and ignored `kv_cache_dtype`, making `--kv-cache-dtype fp8` 6x high.
  The frontend now puts per-layer, per-rank K+V bytes in the trace's `comm_size` column
- Preemption freed nothing. The loop guarded on `gen_req[-1].is_prefill()` over a list
  built from non-prefill requests, so requests were marked evicted and the batch shrank
  without a byte being released -- why a 24 GiB config could crash in `allocate`
- `kv_cache_pct` was 0.0 in every bench `timeseries.csv`: `SchedulerStats` has no
  `gpu_cache_usage` field (it is `kv_cache_usage`) and a `getattr` default hid the
  mismatch. Read directly now, so a future rename fails loudly
- Block hashes were unhashed at the prefix level, so two prefixes ending in the same 16
  tokens collided (53 duplicates on a 300-request ShareGPT replay). Now chained through
  the parent, as in vLLM
- `TTFT` could be overwritten when a request resumed after preemption, because
  `set_ttft` ran again on the recomputed prefill. Now recorded once, gated on `is_init`
- `Scheduler` defined `schedule_with_prefix` twice; the first (228 lines) never ran
- Chunked prefill double-counted prefix-cache hits, collapsing `total_len` to 1 for any
  prefill chunk with a hit -- so dense-layer latency and TP collective sizing were looked
  up at 1 token. `chunk_size` already excludes cached tokens; `Batch.hit_len` went with
  the redundant subtraction
- `_make_sub_batch` was not chunked-prefill aware: it used `req.is_init`, `req.input`
  (the full prompt, not this step's chunk) and `prefill_k_list=0`, and leaked batch1
  state into batch2. It now reads `batch.scheduled_tokens` and `req.num_computed_tokens`
- `MemoryModel.evict_prefix_cache` over-evicted the second-tier cache by `num_npus`x,
  sizing `space_needed` from per-rank bytes while a second-tier token is full-cluster.
  Now uses the cache's own `kv_size`. TP=1 unaffected; TP>1 hit rates were collapsing
- `MemoryModel.evict_prefix_cache`'s early-return guard required *both*
  `not enable_prefix_caching` AND `bytes <= 0`; changed to `or`
- NPU->CPU offload alloc/free in `scheduler.py` used per-rank bytes while prefix-cache
  events tracked full-cluster ones, so `cpu_used` drifted at TP>1. Now scales by
  `num_npus`
- `MemoryModel.storage_cache_evicted_req` passed a **second-tier** node to
  `npu_prefix_cache.inc_lock_ref()`; walking its parents never reaches the NPU tree's
  root and dereferences `None`, crashing on eviction to CPU/CXL with prefix caching on
  (PR #25)
- `MemoryModel.avail_size` multiplied `RadixCache.avail_size()` -- already bytes -- by
  `self._bytes_per_token`, making scheduler decisions under-conservative even at TP=1
  (PR #25)
- Hardcoded `131072` bytes-per-token (Llama-3.1-8B bf16) in five sites in
  `serving/__main__.py` replaced with model-aware values, fixing the utilization readout
  for the Qwen3 family and other models
- Tuple-unpacking crash in the CXL + prefix-sharing display path:
  `for i, cxl_id, cxl_pool in enumerate(prefix_pools)` raised `ValueError`
- Refreshed validation baselines and website plots after the chunked-prefill +
  prefix-cache fix. Means / P99s now slightly over-predict vLLM instead of
  under-predicting, still within ~2.5% on TTFT / TPOT / latency means

### Changed
- Shot-feasibility filters read the **live** KV block size
  (`RuntimeLimits.block_size`) instead of a hardcoded 16. What
  `HOST_ENGINE_DEFAULTS` requests is not always what the engine uses: on a
  hybrid stack vLLM enlarges the attention block until an attention page costs
  at least as many bytes as a mamba state page, then pads the mamba page to
  match, so one uniform pool covers both — measured at **784 tokens** on
  Qwen3.8-27B against the 16 we asked for. A filter off by 49x either emits
  shots the cache cannot hold or silently drops ones it can. At 16 the
  arithmetic is the old constant exactly, so nothing already profiled moves
  (`.claude/check_block_size.py`: `dense` and `moe` fire one request and so
  can never reach the filter; `per_sequence` and `attention` fire n and
  tighten by 8 and 263 shots at 784)
- Per-layer timings are normalized by **parent invocations x how many times the
  block sequence emits the layer**, not by the profiled node's `invocations`.
  vLLM merges every same-class sibling under one parent into a single node,
  summing CUDA time and counting *calls*, so `invocations` is only the number
  of trace nodes when those siblings are interchangeable. In Qwen3.5/3.8's
  gated-DeltaNet block they are not: `in_proj_qkvz` (5120 -> 16384) and
  `in_proj_ba` (5120 -> 96) are both `MergedColumnParallelLinear` children, and
  dividing by `invocations` returned the mean of a large GEMM and a tiny one.
  There is no discriminator to recover — one node is all vLLM emits — so the
  catalog models the pair as one layer and the sum is what one trace node
  costs. The two formulas agree whenever
  `invocations == parent_invocations x occurrences`, which holds for every
  homogeneous model, so no committed profile bundle and no
  `serving/validate.sh` baseline moves; the new behaviour appears only where
  the old one was wrong. Also unblocks profiling a hybrid stack with
  `num_hidden_layers > 1`, which is the only way to reach both block types
- **vLLM pinned to v0.28.0** (was v0.19.0), in `scripts/docker-vllm.sh`,
  `scripts/install-vllm.sh` and the docs. The tag semantics inverted along the
  way: `v0.28.0` is the CUDA 13.x build and `v0.28.0-cu129` is the fallback,
  where `v0.19.0` was CUDA 12.x with a `-cu130` variant. Four vLLM-internal
  APIs the profiler binds to moved, all under `profiler/core/hooks/`:
  - `FusedMoE` no longer exists. Models now call `FusedMoEFactory`, which
    returns a `MoERunner` owning a `router` and a `RoutedExperts`. `moe_hook.py`
    forges expert routing by swapping `_compute_routing` on the live **router
    instance** instead of monkey-patching `FusedMoE.forward_native` and then
    `select_experts` and then `_compute_routing` on the class — one patch where
    there were three, and no `layer_name` guard needed. `_select_experts` still
    runs, so EPLB mapping and index-dtype conversion happen as in production
  - v0.28 ships **two** GPU model runners and picks between them per config —
    Llama-3.1-8B boots on V2, the Qwen3.8-27B hybrid on V1 — and V2 keeps no
    persistent `input_batch`. `batch.py` now reads KV-cache-group block sizes
    from `kv_cache_config.kv_cache_groups[i].kv_cache_spec.block_size`, which
    both runners derive their tables from, and which is the KV-manager block
    size directly (no `block_size * blocks_per_kv_block` arithmetic)
  - `NewRequestData.prefill_token_ids` is declared `= None` but asserted
    non-None by the V2 runner. Filled in for synthetic requests, matching
    vLLM's own scheduler, which passes `req._all_token_ids`
  - `SchedulerOutput` is built from `make_empty()` and then overridden, rather
    than positionally. v0.28 alone added eight fields; naming them all is a
    breakage per release for no benefit
- `probe_limits` reads KV capacity from `cache_config.kv_cache_size_tokens`
  rather than `num_gpu_blocks * block_size`. vLLM added the former in v0.28
  precisely because the latter "can be wrong for hybrid models where requests
  occupy multiple KV cache groups" — measured at **1.72x too high** on
  Qwen3.8-27B, where vLLM unifies the page size to 784 tokens so an attention
  page and a mamba state page cost the same bytes. Unchanged for non-hybrid
  models, so no profiled latency moves
- `scripts/docker-vllm.sh` takes `VLLM_GPUS` to narrow which GPUs the container
  claims on a shared machine (`VLLM_GPUS='"device=2,3"'` — the inner quotes are
  part of the value; without them Docker reads the second field as a GPU count)
- `pandas` is named explicitly in both install scripts. `profiler/core/skew.py`
  and `fit_alpha.py` import it directly but had only ever been getting it
  transitively through `datasets`, and the v0.28.0 image doesn't ship it
- The simulator is roughly **11x faster** with byte-identical results. The four
  `bench/examples/` workloads go 16m 40s to 1m 26s in total (per-example 5.2x to 24.6x)
  and the 19 `serving/run.sh` scenarios 18.95 min to 1.94 min, with every
  `Total clocks (ns)` unchanged and all four `sim.csv` files byte-identical:
  - The Chakra converter runs **in-process** instead of a subprocess per batch (~56 ms
    each, ~52 ms of it interpreter startup), which had been 73-85% of wall-clock
  - The converter takes the trace **as field tuples**, not text. The text file is now
    written only for `--save-trace-text`
  - **Converted graphs are reused** on an identical trace -- 4,405 of 8,810 batches are
    dummies on the swe-bench MoE DP+EP example, only 22 of them distinct
  - **ASTRA-Sim stops re-asking an idle NPU** until its answer could change: handshakes
    drop from 337,786 to 2,462 on a 10-request 8-NPU run
  - The analytical frontends no longer print a per-tick `Checking NPU ...` line, which
    the frontend had to drain off the pipe (78-99.6% of lines read)
  - Profile tables are built in plain Python and only for the TP degrees a run touches
    (TP=1 startup 1435 ms to 50 ms); the architecture config is cached
- **`--cleanup-inputs` is replaced by `--save-trace-text` and `--keep-inputs`**, both
  defaulting off -- same observable default, no double negative. It was doing two jobs:
  writing each batch's trace for inspection, and preserving the `.et` workloads and
  generated configs for a manual ASTRA-Sim replay. `--no-cleanup-inputs` callers need
  one of the two instead
- Graph metadata stores the trace's **path**, not its text. Nothing consumed it and it
  was 70% of every `.et`: 117,112 to 24,403 bytes per file on the swe-bench MoE DP+EP
  example, ~720 MB to ~180 MB of I/O per run
- All four canonical examples were regenerated on current `main`, since the previous
  summaries predated the block pool rework, the pipeline-stage fix and the interpolation
  change. TPOT means now land within 1.7% and latency means within 2.2%; TTFT means span
  +1.3% to -13.6% (the MoE run, 30 ms absolute on the smallest values in the set). Docs
  quoting "within 1.5%" or "-2.6% to -5.8%" are updated
- **`npu_mem.mem_util` must be calibrated against the run you compare against.** The
  simulator models neither vLLM's activation peak nor its CUDA context, so `0.9` buys
  more KV cache here than in vLLM -- and KV capacity drives preemption. It only bites
  when a run **saturates** the cache: read `kv_cache.num_gpu_blocks` from the bench
  run's `meta.json` and pick the `mem_util` whose startup line reports the same count.
  On the RTX 4090 example that is `0.833919`, worth -20.7% TTFT / +12.9% TPOT versus
  +0.6% / +0.2%. The 96 GB RTXPRO6000 examples peak at 58-97% and are unaffected
- The attention grid is interpolated **linearly**, not in log space (`_axis_bracket`).
  Grid spacing decides where the kernel is sampled; the blend decides how samples
  combine -- and the kernel is linear in each axis (decode attention fits
  `time_us = a + b * (n_decode * kv_decode)` at R^2 = 1.0000). Leave-one-out over the
  measured grid puts log space at +11.6-14.4% mean error against +2.3-3.7% for linear,
  on all four axes across every bundle in `profiler/perf/`
- With no skew profile the simulator applies **no** skew correction
  (`_ATTN_SKEW_ALPHA_FALLBACK` 0.093 -> 0, i.e. `t_mean`) rather than a borrowed
  constant; bundles with a real `skew_fit` are unaffected. The blend endpoints are far
  apart (`t_max / t_mean` median ~1.5), so alpha has to be known to a couple of
  hundredths to be worth applying. It is not bounded to `[0, 1]` either
- `Scheduler` follows vLLM V1's `schedule()`: `self.running` first, preempting only from
  its own tail, then `self.waiting` while budget and slots remain. Admission never
  preempts and is skipped on any step that preempted. `schedule_base` and
  `schedule_with_prefix` collapse into one `schedule()`; `scheduler.py` drops ~1300 to
  ~510 lines, `memory_model.py` 885 to ~545
- Preemption is vLLM verbatim, including `num_computed_tokens = 0`. That is not
  re-prefill -- the blocks keep their hashes, so recovery comes from the tier hierarchy
  rather than a "preserve the decode state" path
- The three prefix-cache modes map onto three real vLLM configurations:
  `--no-enable-prefix-caching` is prefix caching off, `--enable-prefix-caching` is
  default vLLM, and `--prefix-storage CPU/CXL` is vLLM with LMCache or
  `OffloadingConnector`. The previous middle case billed a transfer against an empty tier
- `num_computed_tokens` advances when the batch is formed, as in vLLM's
  `_update_after_schedule`, with `Batch.scheduled_tokens` as `add_done`'s snapshot.
  Advancing at completion let `pp_size > 1` schedule the same tokens twice
- Prefill and decode are no longer distinct scheduler states. A request catches up to
  `num_tokens_reached`, and the trace classifies by scheduled token count (>1 = prefill
  chunk, ==1 = decode) -- the only classification that survives a resumed request
- A host offload tier uses 256-token chunks (LMCache's default) uniformly. Page size 1
  matched at token granularity and over-reported hits against any real offload tier
- `MemoryModel.get_weight` divides the transformer-block weight by `pp_size`
  (heaviest-rank bound), adding a `pp_size` parameter to `__init__`. PP=1 unchanged
- `MemoryModel.apply_kv_cache_events` also drains the second-tier queue for CXL prefix
  storage and CPU + prefix-sharing, preventing unbounded growth. No accounting impact
- Documentation: the PP write-up in `simulator/parallelism-mechanics.md` now describes
  the Chakra layer split and inter-stage `COMM_SEND` / `COMM_RECV`, replacing a
  "scheduling-only" framing; `--expert-routing-policy` is documented as defaulting to
  `BALANCED` (not the non-existent `COPY`), with `--enable-block-copy` decoupled from
  it; and `LOAD` scoring (`waiting * 4 + running`) is documented

### Removed
- `bench/bench-rtx4090.sh` -- a copy of `bench/bench.sh` differing only in defaults that
  are already environment overrides there. Now an example in that script's header
- `host_metadata.txt` and `scripts/capture-host-metadata.sh`. The script wrote three
  `nvidia-smi` fields against ten hand-written ones in the file, and the information is
  already in the profiler's `meta.yaml` and a bench run's `meta.json`
- `bench/results/` and `outputs/rtx4090_llama/` artifacts committed past `.gitignore`.
  Committed bench artifacts belong under `bench/examples/<hardware>/<model>/`
- `serving/core/radix_tree.py` (675 lines), the SGLang-derived prefix-cache radix tree,
  **replaced by `block_pool.py` + `kv_cache_manager.py`** (see Added). It served as both
  index and allocator, and as an allocator it was inexact. Every user-visible
  prefix-caching flag is unchanged; the SGLang attribution stays in `CONTRIBUTORS.md`
- `--prioritize-prefill`, the per-instance `prioritize_prefill` key, and
  `Scheduler._merge_by_arrival_id` (its only caller). vLLM v0.19.0 has no equivalent:
  `SchedulerPolicy` is `fcfs` or `priority`, i.e. request priority
- `Request.is_prefill()`, `evict`, `npu_last_node`, `cpu_last_node`,
  `storage_last_node`, `_prefix_locked`; and from `MemoryModel`: `avail_size`,
  `evictable_size`, `get_block_kv`, `get_evict_kv`, `lock_prefix`, `unlock_prefix`,
  `cache_unfinished_req`, `cache_finished_req`, `evict_prefix_cache`, `prefix_match`,
  `apply_kv_cache_events` and the two `_*_cache_hashtolen` maps
- `Scheduler.get_first_arrival_time`, which read a never-assigned attribute and had no
  callers (`Router.get_first_arrival_time` is the live one)

### Security
- Bump `fast-uri` to ≥3.1.2 (CVE-2026-6321 path traversal, CVE-2026-6322 host confusion,
  both High). Pinned in `pnpm.overrides` as a transitive Docusaurus dependency
- Bump `@babel/plugin-transform-modules-systemjs` to ≥7.29.4 (GHSA-fv7c-fp4j-7gwp,
  CVE-2026-44728, High): arbitrary code generation on malicious input in 7.12.0-7.29.3.
  We shipped 7.29.0 via `@docusaurus/preset-classic`. Pinned in `pnpm.overrides`
- Bump `serialize-javascript` to ≥7.0.5 (XSS via deferred function / regexp
  serialization), pulled in by webpack plugins in Docusaurus 3.10
- Bump `uuid` to ≥14.0.0 (missing buffer bounds check in v3/v5/v6 when `buf` is given),
  replacing both the transitive 8.3.2 via `sockjs` and 11.1.1

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
