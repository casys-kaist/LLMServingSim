---
title: Changelog
description: LLMServingSim release history
---

All notable changes to this project are documented in this file.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions.

## [Unreleased]

### Added
- **`scripts/patches/`, applied by `docker-vllm.sh` at container start.** Two
  one-line fixes to the installed vLLM, both idempotent and both no-ops on a
  vLLM that already carries them.
  - `vllm_m3_mtp_layer_name.py` gives MiniMax-M3's MTP module its own
    layer-name prefix (`model` -> `model.mtp`). Without it M3 cannot start with
    speculative decoding at all: `static_forward_context` is keyed by layer
    name and shared between target and drafter, and M3 is the one MTP family
    that neither offsets its layer index (DeepSeek/GLM) nor separates its
    prefix (Qwen3.5), so it collides on `model.layers.0.self_attn.attn`.
    Parameter names are unaffected -- the prefix feeds the name registry while
    parameter names come from the module tree -- verified on a live engine, so
    the patch is safe for a real weight load and not only for
    `load_format=dummy`. **No upstream fix to backport**: the file is
    byte-identical between 0.28.0 and vLLM main as of 2026-09-01 and no issue
    reports it.
  - `vllm_sm120_sparse_mla.py` backports vLLM [PR #51395](https://github.com/vllm-project/vllm/pull/51395)
  onto 0.28.0: without it DeepSeek-V3.2 and GLM-5 crash partway through a
  profile run on any Blackwell card, because `FlashInferMLASparseSM120Impl` --
  the only sparse-MLA backend accepting compute capability 12 -- does not
  override `supports_dense_mha_prefill`, so `mla_attention.py` reads a
  `masked_mha_available` it never set. Idempotent and a no-op on a vLLM that
  already has the fix. `python -m profiler coverage` does **not** catch this:
  it fires three fixed shots, and the failure needs the dense-MHA prefill
  path, first reached at shot 89 of 152. `docker-vllm.sh` also pins
  `nvidia-nccl-cu13`, since installing `datasets` pulls a newer one than the
  image's torch declares and pip installs it over the reported conflict.
- **Three example pages for the features this cycle added**, under a new
  "Model families" section plus one in Advanced: hybrid linear attention
  (gated DeltaNet), sparse attention (DSA and block-sparse MSA), and
  speculative decoding. Each carries the setup, the per-family requirements
  that fail loudly if missed (MiniMax-M3 needs `--block-size 128`; its config
  must stay nested), and the caveats -- including, plainly, that no bundle is
  shipped for any of these models and no end-to-end validation has happened.
- **`profile.sh` and `profile-all.sh` now reach every profiler flag.**
  `profile.sh` gained `ATTENTION_DECODE_Q_LENS` (the speculative-decoding
  axis, which existed on the CLI but not in the editable template),
  `OUT_ROOT` and `MODEL_CONFIG_ROOT`.
- **A complete "everything you can set" index in `profiler/README.md` and
  `serving/README.md`**, checked against the live argparse surface: 22 of 30
  profiler flags and 27 of 31 simulator flags had never been mentioned in
  either file. The semantics stay on the website (which was already complete
  bar one row) so there is one description to keep correct, not two.
- **The KV block size is read back from the profile bundle, not defaulted to
  16.** vLLM treats a block size as a *floor and an alignment unit* rather than
  the answer: `platforms/interface.py` takes
  `alignment = max(min(backend.get_supported_kernel_block_sizes()),
  cache_config.block_size)`, derives the smallest multiple of it whose
  attention page covers one mamba page, and raises the block size to that --
  never lowering it. So the resolved value depends on what was asked for:
  Qwen3.8-27B gives **784** from `--block-size 16`
  (`16 * cdiv(3,207,168, 16 * 4,096)`) and **832** from 64. The profiler now
  records what the engine settled on in `meta.yaml::engine_resolved`
  (`block_size`, `max_model_len`, `num_cache_tokens`) and the simulator reads
  it back, so lookups match the block size the latencies were measured at. An
  explicit value that disagrees is still allowed -- studying a hypothetical
  block size is a legitimate thing to simulate -- but it is warned about. The
  four bundles shipped in this repo predate the field and still fall back to 16
  until they are re-profiled.
- **Speculative decoding.** `--num-speculative-tokens N`,
  `--spec-acceptance-rate`, `--spec-acceptance-policy {FIXED,DECAY,CUSTOM}`,
  all overridable per instance. Scheduling follows vLLM's own framing --
  `num_tokens_with_spec = num_tokens + spec_tokens`, a request catches up to
  it, and rejection rolls back with `num_computed_tokens -= num_rejected`.
  - **Defaults come from each model's own published measurement**
    (`configs/spec_decode.json`, one entry per model with its source):
    DeepSeek-V3.2 accept length 2.55 at N=4 and GLM-5 2.76 at N=4 (GLM-5
    technical report Table 2), MiniMax-M3 ~3.0 at 67% with N=3 (vLLM's day-0
    serving guide), Qwen3.8-27B 4.89 at 77.9% with N=5 (vLLM's measurement on
    Qwen3.5-27B, same family and size). A model with no published figure gets
    **no** default — the four range from 0.39 to 0.78, so there is nothing
    defensible to guess
  - The rate is `accepted / drafted` and **marginal**, so
    `mean_accept_length = 1 + rate * N` — an identity that reproduces all nine
    published (rate, length) pairs to within 0.01 tokens. Deliberately not
    Leviathan's conditional per-position alpha (ICML 2023): passing a published
    rate to that capped-geometric formula under-predicts the published accept
    length by 25-30%, because real acceptance is front-loaded rather than
    i.i.d. Qwen's published decline from 95% at p1 to 60% at p5 averages 0.775
    read as marginals against a published 0.779, and 0.621 read as conditionals
- **A fifth attention axis, `decode_q_len`** — query tokens per decode
  sequence, 1 normally and `1 + N` for a speculative verification step. That
  shape is expressible by none of the other four: *n* sequences each submitting
  *k+1* queries against **their own** KV is neither one prefill chunk of
  `n*(k+1)` tokens nor `n*(k+1)` single-token decodes, because the k+1 queries
  of one sequence share that sequence's KV read. Opt-in in the profiler
  (`--attention-decode-q-lens`, default `1`) since it multiplies the grid; a
  bundle without the column reads as `q=1` and is unchanged

- **Per-sequence layer state is charged against pool capacity.** A
  linear-attention layer caches nothing per token but holds a fixed conv +
  recurrent state for as long as the sequence lives, so it bounds *concurrency*
  the way a KV cache bounds context. It is counted in **pages**, because that
  is what vLLM allocates: it sizes the attention block so one attention page
  covers one mamba page (`attn_block_size = alignment * cdiv(mamba_page_size,
  alignment * attn_page_size_1_token)`) and then pads the mamba page up to it,
  so a layer's whole state occupies exactly one page. On Qwen3.8-27B the mamba
  page is 3,207,168 bytes against an attention page of 3,211,264 at
  `block_size 784` — which is where 784 comes from.
  `MambaSpec.max_memory_usage_bytes` then gives `1 + N` pages per mamba layer
  with prefix caching off and `2 + N` with it on (`mamba_cache_mode` `none` vs
  `align`): one page for the state being written this step, one for the last
  checkpoint at a block boundary that a later prefix hit resumes from. On
  Qwen3.8-27B that is 48 layers x 2 pages = **6 pool blocks per request** with
  caching on, 3 with it off. `N` is `--num-speculative-tokens`, which also
  widens the conv state itself (`conv_kernel_size - 1 + N`).
  - The state blocks live in a list **separate** from the token blocks
    (`req_to_state_blocks`), because the token list is positional — block `i`
    backs tokens `[i*block_size, (i+1)*block_size)` — and a state block backs
    no tokens, so nothing may hash or index it
  - Released on preemption as well as completion: vLLM rebuilds a preempted
    sequence's recurrent state from the recomputed prefix rather than keeping it
- `utils.config_weight_dtype()` — the weight dtype a checkpoint declares, in
  the profiler's order. Shared by `--dtype`'s default and
  `resolve_variant`, which must agree or the simulator reads a variant folder
  the profiler never wrote

- **Tensor sizes and per-sequence state for gated DeltaNet** (Qwen3.5 / 3.6 /
  3.8) — the last of the four modern families. Twelve new entries
  (`gdn_in_proj`, `gdn_conv_prefill`, `gdn_conv_decode`, `gdn_post_conv`,
  `gdn_norm`, `gdn_out_proj`, `gdn_prefill`, `gdn_decode`, `gdn_decode_mixed`,
  `gdn_glue`, `qk_norm_rope`, `attn_glue`), shapes from vLLM's
  `layers/mamba/gdn/`. **Every layer name in all five catalogs now has a size
  formula** — the count was 28 missing across three families.
  - Verified the same way as the rest: Qwen3.8-27B comes to **26.9B**
    parameters
- `memory_model.state_bytes_per_sequence()` /
  `full_cluster_state_bytes_per_sequence()` — the **fourth** KV shape, which is
  no KV at all. Gated DeltaNet holds a rolling conv state and a recurrent state
  whose sizes do not depend on sequence length, which is why its *cost* does
  not either — but they are held for as long as the sequence lives. On
  Qwen3.8-27B that is 3.21 MB per sequence per layer, and 48 of its 64 layers
  are linear attention: **153.9 MB per concurrent sequence**. Its KV per token
  is 65,536 bytes, from the 16 full-attention layers only. The two states do
  not share a dtype: the conv state follows `mamba_cache_dtype` and the
  recurrent state `mamba_ssm_dtype`, which `auto` resolves to the **conv**
  dtype rather than the weight dtype (`MambaStateDtypeCalculator`). Qwen3.8
  declares `float32` there, and the recurrent state is 98% of the total

- **Tensor sizes, block weight and KV shape for MiniMax-M3's block-sparse
  attention.** Six new entries (`qk_norm_rope`, `sparse_qkv_proj`,
  `sparse_qk_norm_rope`, `sparse_o_proj`, `indexer`, `sparse_attention`), read
  off vLLM's `models/minimax_m3/` plugin.
  - `sparse_qkv_proj` is one column-parallel GEMM emitting
    `[q | k | v | index_q | index_k]`. `index_q` carries the KV head count and
    shares `head_dim`, so it shards like K and V; `index_k` is a single head,
    **replicated** to every rank
  - `sparse_attention` reads at most `sparse_topk_blocks * sparse_block_size`
    keys however long the sequence is — but every token is still *stored*, so
    that bound applies to the read, not to KV capacity
- `utils.num_experts()` / `utils.is_moe()` — one helper for a fact the families
  spell three ways (`num_local_experts`, `num_experts`, `n_routed_experts`)
- **DeepSeek-V3.2 now runs end to end.** Verified against a trace dump: 61 MLA
  + indexer layers, 3 dense MLPs (`first_k_dense_replace`), 58 `EXPERT` blocks,
  and every row's byte counts matching the formulas by hand

- **Tensor sizes, block weight and KV shape for MLA + DeepSeek sparse
  attention** (DeepSeek-V3.2, GLM-5) — the first of the modern families the
  simulator can size at all. Ten new `calculate_sizes` entries
  (`mla_qkv_a_proj`, `mla_a_layernorm`, `mla_b_proj`, `indexer_wq_b`,
  `indexer_wk_proj`, `indexer_k_norm`, `indexer_rope_emb`,
  `indexer_q_rope_quant`, `indexer`, `indexer_glue`), every shape read off
  vLLM's `deepseek_v2.py` rather than inferred.
  - Checked against the published parameter count, which is the one number that
    catches a wrong shape anywhere in the stack: summing the formulas over the
    checkpoint's own layer composition gives **671.878B** for
    DeepSeek-V3.2-Exp, and subtracting the DSA indexer (0.852B) leaves
    **671.026B** — DeepSeek-V3's published 671B. The gap is exactly what V3.2
    adds over V3
  - `mla_qkv_a_proj` and both indexer projections are **replicated**, not
    TP-sharded (`disable_tp=True` / `ReplicatedLinear`); `indexer_k_norm` is a
    bare `nn.LayerNorm` so it carries a bias as well as a weight
- `memory_model.kv_bytes_per_token_per_layer()` — one place that knows the
  three KV shapes: GQA (K+V, sharded), MLA (one replicated latent of
  `kv_lora_rank + qk_rope_head_dim`, `num_kv_heads = 1`, so **not** divided by
  TP), and the sparse indexer's own second cache (fp8 keys plus one fp32 scale
  per 128 elements, 132 bytes/token/layer). `get_kv()` and
  `full_cluster_kv_bytes_per_token()` both go through it
- `utils.get_architecture()` / `utils.get_layer_stack()` — the architecture
  yaml and the checkpoint's per-layer block list, loaded and cached once.
  `memory_model` needs them to weigh a block and `trace_generator` needs them
  to emit one; `utils` is the module both already import, so there is one
  loader rather than two to keep in step

- **One catalog form: `blocks:` + `shared:`.** `sequence:` is gone, and with it
  the second code path. `blocks:` is keyed by axis
  (`attn.<layer_types value>`, `sparse_attn.<same>` as an overlay,
  `mlp.dense|moe`); `shared:` holds `prologue` and `head`. A uniform stack is
  the degenerate case — one entry per axis — and all seven bundled catalogs are
  migrated.
  - `sequence:` was this form **flattened**: its `pre_attn` / `post_attn` were
    the single implicit `attn.full_attention` block and its `mlp_dense` /
    `mlp_moe` were the `mlp` axis with the value baked into the key name. Two
    forms meant two code paths, and `trace_generator` only ever implemented the
    flat one — it rejected `blocks:` outright, so Qwen3.5/3.8 and MiniMax-M3
    could not be simulated at all
  - Baking the axis into the key also removed the per-layer question, and the
    MLP was resolved once per **model** (`is_moe = gate is not None`). See
    *Fixed*
- **Per-layer block resolution in the simulator.** `trace_generator` now reads
  `profiler/core/stack.py` — the same module the profiler uses to decide how
  many layers to instantiate — so which block a layer emits comes from the
  checkpoint's own config (`layer_types`, `first_k_dense_replace`,
  `decoder_sparse_step`, `moe_layer_freq`, `sparse_attention_freq`,
  `index_topk_pattern`). One implementation, because the vendors' rules
  disagree on the off-by-one in opposite directions and a second copy would be
  a second chance to get them backwards
  - Blocks are built once per distinct block **shape** and replayed for every
    layer that shares it, so a uniform model still builds one block and replays
    it N times while a heterogeneous one gets the right block per layer
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
- **`python -m profiler coverage <model>`** — boots the model once, runs one
  forward per batch regime (prefill-only / decode-only / mixed), and reports how
  much of the measured CUDA time the architecture catalog binds. Writes nothing,
  exits non-zero while any kernel is unbound, and reuses the profiler's own
  matcher, so a clean report means a real run binds the same things.
  - Exists because a catalog entry can name a real vLLM class and measure
    **nothing**. vLLM's profile tree holds only modules that launch a kernel of
    their own, and modern models fuse q-norm / rope / KV-write into one kernel
    with no module, or write attention as bare Triton kernels launched straight
    from the block. The module tree — the natural thing to read a catalog off —
    still shows the classes, and the symptom is a layer that looks free
  - Found something in every one of the four modern families. Details in
    *Fixed* below
- `profiler/models/minimax_m3_vl.yaml` — MiniMax-M3 (block-level sparse
  attention over GQA, MoE). The third sparse-attention shape in the repo and
  deliberately distinct from DeepSeek/GLM's token-level top-k: M3 selects the
  top `sparse_topk_blocks` blocks of `sparse_block_size` tokens, which is the
  granularity a KV block pool already speaks. Heterogeneous on two axes at once
  (layers 0-2 non-sparse + dense MLP, 3+ sparse + MoE), both resolved from the
  checkpoint. Needs `--block-size 128`; the platform default of 16 fails with
  "No common block size for 16"
  - New `blocks.sparse_attn` overlay in the architecture schema, used when a
    layer's `sparse` flag is set. M3's block difference is the sparse flag, not
    the attention type — every layer is full attention — which `blocks.attn`,
    keyed by attention type, could not express
- Catalog `vllm:` entries accept a trailing **`*`** for prefix matching, and a
  **list** now means "every one of these", with the matches **summed**. Both are
  forced by real models: a fused kernel reports its template arguments inline
  (`fusedMiniMaxM3QNormRopeKVInsertKernel<c10::BFloat16, ...`), so an exact
  binding silently stops matching on the fp8 variant; and M3's sparse attention
  is three Triton kernels with no module to aggregate them, whose sum is what
  one trace node costs. Summing is a no-op for every catalog whose entries each
  match one node

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
- **The simulator had no `linear_attention` category**, so every gated-DeltaNet
  recurrence — the defining computation of the architecture — was skipped out
  of every trace with a one-line warning. `_layer_category` iterated four
  categories and a fifth reads as "not in the catalog". Neither the coverage
  check (profiler-side) nor the parameter count (weights only, and these layers
  carry almost none) could catch it; running Qwen3.8 end to end and reading the
  trace did
- **Regime-dependent kernels were filed under `dense` / `per_sequence`**, which
  emit on every batch. A gated-DeltaNet block runs a *different set* of kernels
  for a pure prefill, a pure decode and a mixed batch, so this charged a decode
  conv on a pure prefill (1077 ns x 48 layers) and a prefill conv on a pure
  decode. `gdn_conv_prefill`, `gdn_post_conv` and `gdn_conv_decode` moved to
  `linear_attention`, whose `(prefill_tokens, n_decode)` key makes the
  profile's own absence of rows for a regime mean "does not fire here".
  Which kernel belongs where was **measured** with
  `python -m profiler coverage`, not inferred; DeepSeek and MiniMax-M3 have no
  regime-dependent layers
- **`--dtype`'s default ignored `quantization_config`**, so a quantized
  checkpoint defaulted to its *activation* dtype. DeepSeek-V3.2-Exp
  (`quant_method: fp8`, `torch_dtype: bfloat16`) defaulted to `bfloat16` and
  looked for a `bf16/` bundle the profiler had written to `fp8/`, while
  `resolve_variant` — reading the same config — would have said `fp8`. Both go
  through `utils.config_weight_dtype()` now, and DeepSeek runs with no `--dtype`
  at all
- **`qkv_proj` was a third too small on Qwen3-Next / Qwen3.5.** Those fuse an
  output gate into the Q half (`total_num_heads * (1 + attn_output_gate)`), so
  Q is double width and the gate has no parameters of its own. Defaulted the
  way vLLM's class does, but only for checkpoints on that code path — the ones
  with a linear-attention stack — so plain Qwen3 and MiniMax-M3 are unaffected
- **KV was charged for gated-DeltaNet layers**, which cache nothing per token.
  On Qwen3.8-27B that read 4096 bytes/token for all 64 layers where only 16 of
  them have a KV cache at all
- **The expert count was read four different ways, and every one of them
  missed `n_routed_experts`.** DeepSeek and GLM therefore read as *dense* in
  `config_builder`'s `is_moe` and expert-divisibility check, in
  `trace_generator`'s gate construction (leaving `ctx.gate` `None`, so the MoE
  block could not be emitted at all) and in its ALLTOALL sizing. All four go
  through `utils.num_experts()` now
- **`get_config()` returned the wrapper for a nested checkpoint**, so
  `config['hidden_size']` raised on MiniMax-M3 and every helper reaching for a
  dimension answered for a model nobody asked about. It flattens through
  `stack.text_config` at load; a flat config is unchanged
- **`attention` sized MLA as grouped-query attention**, with `head_dim` derived
  as `hidden_size / num_attention_heads` (56 on DeepSeek-V3.2, a number the
  model does not use anywhere). MLA reads Q at `qk_head_dim` (192), the cache
  as one replicated latent per token (576), and writes `v_head_dim` (128) —
  three different widths
- **MiniMax-M3's dense MLP was sized 4x narrow.** M3 inverts the usual
  convention: its `intermediate_size` (3072) is the *per-expert* width and the
  dense layers' is `dense_intermediate_size` (12288)
- **`get_weight()` summed a hardcoded block** —
  `layernorm + qkv_proj + o_proj + layernorm + mlp` — which is right for
  families whose blocks look like Llama's and silently wrong for the rest. It
  now walks the architecture yaml and the checkpoint's own per-layer
  composition. MLA has no `qkv_proj` at all, so DeepSeek and GLM could not have
  been weighed by the old path even in principle. Llama's weight is unchanged;
  Qwen3's grows by 512 bytes per block, the `qk_norm` parameters the hardcoded
  list omitted, which is below one KV block on every bundled scenario (checked)
  so no baseline moves
- **KV cache was sized as GQA for every model.** On DeepSeek-V3.2 that read
  1,748,992 bytes per token where MLA actually caches **78,324** — a 22x
  overstatement, and the whole point of MLA
- **The expert-count fallback missed `n_routed_experts`**, so a DeepSeek or
  GLM MoE block weighed one expert instead of 256. Same omission in
  `MemoryModel.is_moe`
- `moe` weight now counts the **shared** expert(s) (`n_shared_experts`), which
  run on every token beside the routed ones. No effect on families without the
  field
- `o_proj` and `rotary_emb` follow MLA's dims on an MLA checkpoint: o_proj
  reads `n_head * v_head_dim` (V is a different width from Q — 128x128 on
  DeepSeek-V3.2, 64x256 on GLM-5) and rope rotates only `qk_rope_head_dim`
- **PP weight took the first `n_layer // pp` layers**, not the heaviest. On a
  heterogeneous stack the first window is the light one — DeepSeek's leading
  three layers are the dense-MLP ones — so the "conservative upper bound" the
  docstring promises was not one
- **`--no-enable-block-copy` emitted one transformer block instead of
  `num_hidden_layers`.** Measured on Qwen3-30B-A3B (48 layers): 19 trace lines
  where the same run with block copy enabled emits 583, and a total clock
  3.1x low. The loop was
  `iter_count, copy_count = (num_layers, 1) if block_mode_on else (1, num_layers)`,
  so with `block_mode_on` false it ran **once** and the `can_copy=False` branch
  emitted that single block — `copy_count` was computed and unreachable there.
  The flag means "build each block separately"; it meant "emit one block".
  Block copy is now what it is documented to be, an optimization that does not
  change results: `no_block_copy` and `moe` are the same config differing only
  by the flag, and they now produce identical traces and identical clocks.
  **`serving/validate-baselines.txt` moves for `no_block_copy` only**
  (1037528119 → 1945309759, matching `moe`); the other 57 are unchanged
- **MoE-ness was decided once per model and applied to every layer.**
  `is_moe = gate is not None` gated the whole MLP choice, so a hybrid stack's
  dense layers were modelled as MoE layers. Invisible on Qwen3-30B-A3B, whose
  `mlp_only_layers` is empty, and wrong for DeepSeek-V3.2 and GLM-5, whose
  first `first_k_dense_replace` (3) layers run a dense MLP. Resolved per layer
  now. `mlp_only_layers` is no longer ignored
- **Every attention-category layer was served the same kernel's latency.** The
  attention CSV grew a `layer` column and `_build_attention_tables_by_layer`
  split the tables per kernel, but the lookup kept taking a pooled
  `tables["attention"]` alias and no call site passed a layer name. So on
  MiniMax-M3 the 57 sparse layers got the non-sparse FlashAttention number for
  **both** their `indexer` and `sparse_attention` nodes — 32.1 us against a
  measured 15.0 at a 4-decode/kv-256 batch, **2.1x per sparse layer** — and on
  DeepSeek-V3.2 / GLM-5 the indexer got `MLAAttention`'s. The alias is gone;
  there is one way in and it needs a layer name, and an unknown one raises
  instead of silently resolving. No effect on `llama` / `qwen3`, which have a
  single kernel in the category (58/58 baselines unchanged).
- **The skew sweep measured one kernel and applied its alpha to all of them.**
  `skew.py::_measure` filtered on the literal name `"attention"`. On MiniMax-M3
  that is an alpha fitted on 3 of 60 layers, applied to the other 57; on
  DeepSeek/GLM it drops the indexer, which scans the whole KV and is the most
  skew-sensitive part of the model. `skew.csv` and `skew_fit.csv` now carry a
  `layer` column and the fit runs per kernel, with a per-kernel
  `alpha_default_by_layer` fallback. Measured on M3, same batch: **0.24 for
  `attention`, 0.74 for `indexer`, -0.01 for `sparse_attention`** — the signs
  disagree, so this was never a rounding error. Bundles profiled before the
  column read as `attention`, and the simulator falls back to unprefixed keys
  for `attention` only: any other kernel gets no correction rather than a
  borrowed one, matching the `SKIP_SKEW=1` behaviour and for the same reason.
- Docs claimed `alpha < 0` and `alpha > 1` are "clipped at fit time" and that
  the fit "ignores" them. Nothing is clipped anywhere — out-of-range alphas are
  real, the weighted-LS fit down-weights noisy rows by `(t_max - t_mean)²`
  instead of discarding them, and only `nan` rows are dropped. The sibling
  `output-bundle` page said "**Not clamped**" on the same column
- **Catalog gaps in all four modern families**, each a layer bound to a class
  that launches no kernel of its own, so it measured nothing and the layer read
  as free. Found with the new `coverage` subcommand; none is visible in vLLM's
  source or module tree. Coverage is now 100% of measured CUDA time in all three
  regimes for MiniMax-M3, Qwen3.5/3.8, DeepSeek-V3.2 and GLM-5.
  - MiniMax-M3: q-norm + rope + KV-write are one fused kernel
    (`fusedMiniMaxM3QNormRopeKVInsertKernel`), and the **whole sparse attention
    kernel** was unbound — `MiniMaxM3SparseAttention` has no `Attention` node,
    launching `_gqa_sparse_fwd_kernel` / `_gqa_sparse_decode_kernel` +
    `_merge_topk_attn_out_kernel` by regime
  - Qwen3.5/3.8: `_fused_qk_rmsnorm_rope_gate_kernel` (q-norm, k-norm, rope and
    the output gate, fused) had no entry at all, and the eager reshape/copy work
    inside the gated-DeltaNet block was unattributed — **12.6% of that block's
    other kernels put together**, which is exactly the comparison this
    architecture exists to inform
  - DeepSeek-V3.2 / GLM-5: both rope modules (the class differs by checkpoint —
    `DeepseekScalingRotaryEmbedding` on DeepSeek, plain `RotaryEmbedding` on
    GLM-5), the fused indexer q-rope/quant kernel, and the indexer's own glue.
    This also answers the question the catalog had left open: a rope module
    shared across layers **is** reported under each caller, so it comes out
    per-layer and `within` tells the two apart
- **Catalog resolution is one implementation, shared by profiler and
  simulator** (`profiler/core/catalog_path.py`). The `model_types:` alias
  lookup had been added to the profiler's resolver only; the simulator still
  matched on filename alone, so merging `qwen3_moe.yaml` into `qwen3.yaml`
  broke all 16 MoE scenarios in `serving/validate.sh` with
  `FileNotFoundError: ... qwen3_moe.yaml`. The module is free of third-party
  imports so the simulator container, which has no pydantic, can import it;
  the simulator puts the repo root on `sys.path` explicitly because
  `sys.path[0]` is `''`, which re-resolves against the current directory, and
  `serving/__main__.py` chdirs into `astra-sim/` first.
  `profiler/models/*.yaml` is a **simulator input** despite its path, and both
  `AGENTS.md` and the contributor docs now say so.
- **`attention.csv` carries a `layer` column.** Relaxing the "exactly one
  attention entry" rule was not enough: `AttentionCategory.extract_points`
  averaged every sample into a single point, which was correct while that rule
  held but silently merged two different kernels once it didn't. Measured on
  DeepSeek-V3.2-Exp, `MLAAttention` and `SparseAttnIndexer` collapsed into one
  value per key -- 211 keys, 211 rows, no duplicates. The averaging is also no
  longer needed for its original purpose, since the timing extractor
  normalizes by parent invocations.
  - The simulator splits the attention table **by layer**
    (`attention_by_layer`), and `_layer_available` asks for the requested
    kernel instead of answering "is there attention data at all". A bundle
    profiled before the column existed has one kernel in it, so every row
    belongs to `attention` and the table comes out identical -- that invariant
    is what keeps the committed bundles valid
- **`block_size` is no longer forced to 16** in `HOST_ENGINE_DEFAULTS`. vLLM's
  platform layer picks a value the model's attention backend can use, and
  pinning one can leave it with none: DeepSeek-V3.2's sparse MLA fails backend
  selection outright at 16 (`TRITON_MLA: [sparse not supported],
  FLASHINFER_MLA_SPARSE_SM120: [block_size not supported]`) where letting vLLM
  choose gives 64. The comment beside the default already said it "does not
  change kernel time", so there was nothing to gain by pinning it.
  `--block-size` still overrides, and `probe_limits` reports what the engine
  settled on
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
- **`profile-all.sh` could not profile MiniMax-M3 at all.** It applied one
  global flag set to every model in its list, and M3 *requires*
  `--block-size 128` -- its sparse selection works in 128-token blocks and the
  platform default of 16 fails outright with "No common block size for 16" --
  while passing 128 globally would describe a paging regime the other models
  are not simulated under. Entries are now `"<model>|<extra flags>"` with the
  extras appended last, the model list covers all seven families, and a
  failing model no longer aborts the rest of an overnight sweep (it is
  reported and the exit code is non-zero).
  - TP is per job for the same reason it is not uniform: expert weights are
    ~98% of a big MoE model and they shard by **EP**, so a max-DP layout runs
    every instance at `tp_size 1` and only `tp1/` is ever read. Sweeping TP
    there costs hours and buys nothing. The dense models keep `--tp 1,2`.
- **The docs said the MoE block was wrapped in `ALLTOALL`; the trace has said
  `ALLGATHER` + `REDUCESCATTER` since v1.1.0.** Both are true, which is why the
  drift survived: an MoE dispatch/combine *is* an all-to-all, but it has
  several implementations and vLLM picks one with `--all2all-backend`. Its
  default is `allgather_reducescatter` -- "all2all based on allgather and
  reducescatter" in vLLM's own option list -- and the simulator emits that
  pair, because ASTRA-Sim costs the collective it is handed. The pages say so
  now, and carry the two sizes, which are *not* the same: AllGather's
  `data_size` is the per-rank chunk
  (`total_len / ep_total * (hidden + num_experts) * fp`, router logits
  included), ReduceScatter's the pre-scatter total (`total_len * hidden * fp`).
  "Pass the full activation tensor size" described neither.
  - The trace-format page's MoE example was **invented**: a
    `moe_expert_local_3_rank0` layer name that is emitted as `expert`, an
    `EXPERT END` per rank where there is one for the whole block, and
    `ALLTOALL` in a `comm_type` column that is always `NONE` there. Replaced
    with real lines.
  - `examples/parallelism/dp-ep-moe.mdx` claimed the heartbeat carries an
    `alltoall` field and a `batch=4+4` notation. Neither exists in any log the
    repo ships. Replaced with what the heartbeat actually shows, plus a pointer
    to `--save-trace-text` for the message size.
- **Prefill chunks are block-aligned on a hybrid with prefix caching**, which
  is vLLM's `Scheduler._mamba_block_aligned_split`. A mamba state slot holds
  the state after exactly `(p + 1) * block_size` tokens and state is written at
  chunk ends, so a chunk floors to a block boundary (the prompt's last chunk
  exempted), a mid-block chunk stops at the next boundary, and no chunk runs
  past the last cacheable position. With `block_size 784` and a 2048 budget a
  chunk is 784 or 1568, never 2048. A zero-token result is vLLM's "insufficient
  budget for a block-aligned chunk", not the scheduler deadlock the
  `num_new <= 0` guard catches. Differential-tested against vLLM's rule over
  14,310 (block size, budget, threshold, prompt, position, chunk)
  combinations: 0 mismatches.
- **The drafter's KV cache is charged.** An MTP module wraps a real decoder
  layer, so it publishes a KV cache spec and vLLM allocates for it: +1.6%
  bytes/token on DeepSeek-V3.2's one module, +11.7% on MiniMax-M3's seven,
  +6.2% on Qwen3.8-27B. Its attention is full attention whatever the target's
  layers are, which is explicit in Qwen3.5's MTP
  (`layer_type="full_attention"`), so a hybrid's drafter carries no recurrent
  state.
- **Group-limited expert routing** (DeepSeek-V3/V3.2's `n_group` /
  `topk_group`, read off the checkpoint the way `deepseek_v2.py` reads them).
  A token's k experts are drawn only from `topk_group` of `n_group` groups, so
  it reaches fewer EP ranks than an unrestricted gate would -- on
  DeepSeek-V3.2 at EP=8 (`n_group: 8, topk_group: 4`) the per-rank token count
  drops 31%, from 0.662 to 0.454 of the batch -- 0.662 being GLM-5's
  figure at the same `E` and top-`k` -- and the ALLTOALL shrinks with
  it. GLM-5 ships `n_group: 1`, the unrestricted case spelled out, and no
  other family declares the fields at all.
  - `GateRouter._hit_probs` computes P(token reaches rank r) **exactly**, by a
    DP over which groups the token selected rather than by sampling, so the
    simulator stays deterministic. Verified against a 400k-trial Monte Carlo
    on seven (E, k, n_group, topk_group, ep) points: agreement within 0.0011,
    which is the Monte Carlo's own noise.
  - The same expression fixes the **ungrouped** case, which had used
    `1 - ((ep-1)/ep)**k` -- k *independent* draws, where `torch.topk` selects
    k **distinct** experts. Modelling replacement read ~1% low (Qwen3-30B at
    EP=8: 0.6564 against the exact 0.6674). Six MoE scenarios with EP>1 move
    by at most 0.0071% as a result; none with EP=1 moves, since every token
    reaches the only rank either way.
- **Every dtype is read from the model config; none is an input any more.**
  A model carries five cache dtypes -- weights, KV cache, mamba conv state,
  mamba recurrent state, sparse-indexer side cache -- and they are decided in
  four different places, so a flag per dtype is both unusable and unfaithful:
  the checkpoint already says what it is, and overriding it describes a model
  nobody can serve. `memory_model.cache_dtype_bytes()` is the single table.
  Four of the five rules are vLLM's verbatim, checked against its source:
  - **KV cache** -- `quantization_config.kv_cache_scheme` (compressed-tensors)
    or `kv_cache_quant_algo` (ModelOpt) promotes the cache to fp8, which is
    exactly what `attention.py:281` does when the flag is `auto`, its default.
    vLLM's own source states the direction: *"kv cache dtype should be
    specified in the FP8 checkpoint config and become the 'auto' behavior"*
  - **mamba recurrent state** falls back to the **conv** dtype, not the model
    dtype (`mamba_utils.py::_mamba_state_dtype`), and picks up the HF field via
    `models/config.py::Qwen3_5ForConditionalGenerationConfig`. Qwen3.8-27B
    declares `mamba_ssm_dtype: float32`, so its recurrent state is 4 bytes
    against the conv state's 2 -- and it is the large half, 786,432 elements
    per layer against 30,720. Sizing it at the model dtype understated state
    memory by nearly half: 78.4 -> 153.9 MB per sequence, 75 -> 147 state
    blocks per request
  - **sparse-indexer side cache** is fixed by the model and follows neither:
    DeepSeek/GLM build `DeepseekV32IndexerCache` with `dtype=torch.uint8`, and
    MiniMax-M3's indexer asks for `resolve_indexer_kv_dtype("bf16")`. M3's had
    been following the main KV dtype, which understated an fp8-KV run's cache
    by 10% (68,736 against 76,032 bytes/token over 60 layers)
  - the **weight** row is deliberately *not* vLLM's `model_config.dtype`: it is
    the profiler's variant folder name, which has to encode quantization or a
    bf16 and an fp8 bundle collide. vLLM calls DeepSeek-V3.2 bfloat16 and keeps
    fp8 in the quant method
  `resolve_variant(model_config)` is now a pure function of the config and
  takes no dtype argument. To simulate another precision, profile it: the
  profiler's `--variant` / `--dtype` / `--kv-cache-dtype` still write a
  separate bundle, and the simulator reads the one the checkpoint names
- **The profiler works out how many layers to instantiate**, instead of always
  shrinking to one. `profiler/core/stack.py` resolves the per-layer block
  composition from the checkpoint's config and shrinks to the smallest
  **prefix** that instantiates every distinct block, logging what it chose.
  One layer is right only when every block is identical; on a hybrid it means
  the catalog sees whichever block type comes first and every other layer type
  reads as free. Qwen3.8-27B resolves to 4, every config predating hybrid
  support still resolves to 1, and `--num-hidden-layers` overrides — warning
  when it goes below what the stack needs.
  - The count is computed over the **tuple** of (attention type, MLP type),
    not per axis. A checkpoint interleaving attention 3:1 *and* switching MLP
    at layer 3 has three distinct blocks, the last first appearing at layer 4,
    so the answer is 5 — axis-by-axis reasoning says 4 and is wrong
  - Only rules read out of vLLM's own source are implemented (`layer_types`,
    `full_attention_interval`, `first_k_dense_replace`, `decoder_sparse_step`
    with `mlp_only_layers`, and a list-valued `moe_layer_freq`). The
    conventions genuinely disagree — DeepSeek tests
    `layer_idx % moe_layer_freq`, Qwen3-MoE tests
    `(layer_idx + 1) % decoder_sparse_step` — so an unrecognised layout falls
    back to "uniform" and says so rather than guessing
- `LayerEntry.vllm` also accepts a **list of alternatives**, for a family that
  swaps the class by checkpoint rather than by structure. `llama.yaml` now
  binds `rotary_emb: [Llama3RotaryEmbedding, RotaryEmbedding]`, which is what
  its own header said it could not do: Llama 3 uses the first for its extended
  rope scaling where Llama 1/2 and Mistral use the second, and the single
  binding measured nothing there on half the family. No change for Llama-3.1,
  whose only rotary node is the Llama-3 class
- **One architecture catalog per model family, not per checkpoint shape.**
  `profiler/models/qwen3_moe.yaml` is merged into `qwen3.yaml`, which now
  serves both `qwen3` and `qwen3_moe`. vLLM implements the two in separate
  modules whose classes differ only by a `Moe` infix
  (`Qwen3DecoderLayer` vs `Qwen3MoeDecoderLayer`, `Qwen3Attention` vs
  `Qwen3MoeAttention`) and are otherwise identical layer for layer, so the two
  files were 90% duplicated and a fix to one silently missed the other.
  - `LayerEntry.within` accepts a **list of alternatives**; matching takes the
    deepest one actually present in the ancestor chain, so naming a class this
    checkpoint doesn't have costs nothing
  - Both `mlp_dense` and `mlp_moe` are populated in the merged `sequence:`.
    Nothing new is needed to pick between them: the simulator already emits
    whichever matches the checkpoint's `is_moe`, read from the model config
  - A `catalog.moe` entry now means "this family has MoE checkpoints", not
    "this checkpoint is MoE", so the expert sweep is **skipped** for a member
    that declares no experts instead of raising. A config that *does* mention
    MoE but from which `num_experts` / `top_k` cannot both be read still
    raises — that is an unknown field name, not a dense model
- `profiler/models/qwen3_5_text.yaml` renamed to **`qwen3_5.yaml`**: it serves
  Qwen3.5, **Qwen3.6** and Qwen3.8, dense and MoE, and the old name suggested
  one text tower of one generation. Those three are one architecture —
  Qwen3.6-27B has the same 64 layers at hidden 5120 as Qwen3.5-27B,
  Qwen3.6-35B-A3B matches Qwen3.5-35B-A3B, and all of them report
  `model_type: qwen3_5` / `qwen3_5_moe` and run vLLM's `qwen3_5.py`. The
  generation number is a checkpoint refresh, not a new structure
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
- **Speculative decoding on a model with MTP modules now refuses** until its
  architecture catalog has an `mtp:` block to price the drafter from. vLLM runs
  the drafter N times per step (once, then `num_speculative_tokens - 1` more),
  each a decode-shaped forward over a norm pair, an `eh_proj`, a full decoder
  layer, `lm_head` and the sampler. Charging zero for that reports a speedup no
  engine can deliver, so it is refused the way `calculate_sizes` refuses a
  layer name it has no formula for. The catalog block has to be written from a
  live profile dump, so it lands with the profiling work. A model with **no**
  MTP modules drafts externally and is warned about rather than refused: the
  drafter is a serving choice there, not a checkpoint property.
- **`--dtype` and `--kv-cache-dtype` on `python -m serving`**, along with the
  cluster-config `dtype` / `kv_cache_dtype` per-instance overrides. Both are
  read from the model config now (see Changed). `python -m bench` keeps its own
  flags: those drive real vLLM, which does have them. No committed scenario or
  bench example changes -- vLLM recorded `bfloat16` / `auto` for all four
  examples, which is what the configs derive
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
