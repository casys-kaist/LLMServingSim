# profiler

vLLM-based layerwise profiler for LLMServingSim. Drives a real vLLM
engine with synthetic batches and records per-layer CUDA kernel
latency. Output CSVs feed the simulator's trace generator.

## Directory layout

```
profiler/                     Python package — `python -m profiler ...`
  __init__.py                 package marker + _typeshed shim for vLLM
  __main__.py                 CLI entry (profile / slice / coverage subcommands)
  core/                       internals
    runner.py                 Orchestration loop (run_full / run_slice / run_coverage)
    config.py                 Architecture + ProfileArgs + engine defaults
    engine.py                 vLLM lifecycle (spin_up, probe_limits, spin_down)
    categories.py             Dense / PerSequence / Attention / LinearAttention / Expert
    skew.py                   Heterogeneous-decode skew sweep (skew.csv writer)
    fit_alpha.py              per-kernel 5-axis alpha fit, data-derived bucket axes
    writer.py                 CSV + meta.yaml writer (incl. skew_fit.csv spill)
    stack.py                  per-layer block composition from the HF config  *
    catalog_path.py           model_type -> yaml resolution                   *
    logger.py                 Rich-based logging & progress
    hooks/                    vLLM-internal-API touchpoints
      extension.py            worker extension class (fire / coverage)
      batch.py                synthetic SchedulerOutput builder
      timings.py              layerwise_profile tree parser + coverage accounting
      moe_hook.py             MoERunner forced-routing patch
  models/                     architecture catalogs (one YAML per model family)
    llama.yaml
    qwen3.yaml                  qwen3 + qwen3_moe
    qwen3_5.yaml                Qwen3.5 / 3.6 / 3.8, dense + MoE (hybrid GDN)
    deepseek_v32.yaml           DeepSeek-V3.2 + GLM-5 (MLA + token-level DSA)
    minimax_m3_vl.yaml          MiniMax-M3 (block-level sparse attention)
    mixtral.yaml
    phimoe.yaml
  power/                      nvidia-smi / IPMI power-logging helpers
  perf/                       output root (one folder per hw/model/variant)
  profile.sh                  editable user-run script — edit MODEL/HARDWARE/… then run
  profile-all.sh              helper template: sweep several MODELs × TP degrees

  * imported by the **simulator** too, and kept free of third-party imports for
    it. One implementation each, because these two already drifted once and it
    broke every MoE scenario. A change to either moves simulator results.

scripts/                      shared environment / build entry points (top-level)
  docker-vllm.sh              launches the vLLM container (mounts repo root)
  install-vllm.sh             local (non-Docker) uv venv setup
```

## Everything you can set

`python -m profiler` has three subcommands, and every flag below is a
`profile.sh` variable of the same name in caps unless noted.
**[Profiler → Running](https://llmservingsim.ai/docs/profiler/running)**
carries the semantics; this is the index, so a flag missing from one list is
visible against the other.

```
python -m profiler profile   <model> --hardware <hw> [flags]   full sweep
python -m profiler slice     <model> --hardware <hw> --tp-refresh N --group G
python -m profiler coverage  <model> --hardware <hw>            catalog check
```

| Group | Flags |
|-------|-------|
| **required** | `--hardware` |
| **sharding** | `--tp` (must include 1) |
| **precision / naming** | `--dtype`, `--kv-cache-dtype`, `--variant` |
| **engine limits** | `--max-num-batched-tokens`, `--max-num-seqs`, `--block-size`, `--gpu-memory-utilization`, `--max-model-len` |
| **model shape** | `--num-hidden-layers`, `--hf-override KEY=VALUE` (repeatable) |
| **drafter (MTP)** | `--profile-mtp` (a flag — the engine boots at N=1 so the CSV holds one pass) |
| **attention grid** | `--attention-max-kv`, `--attention-chunk-factor`, `--attention-kv-factor`, `--attention-decode-q-lens` |
| **linear attention** | `--linear-attn-chunk` |
| **measurement** | `--measurement-iterations` |
| **skew** | `--skip-skew`, `--only-skew`, `--skew-n-factor`, `--skew-pc-factor`, `--skew-kp-factor`, `--skew-kvs-factor` |
| **resume** | `--force` (default is resume) |
| **paths** | `--out-root`, `--model-config-root` (no `profile.sh` variable) |
| **verbosity** | `--log-level`, `--silent`, `--verbose` (`VERBOSITY`) |
| **slice only** | `--tp-refresh`, `--group {dense,per_sequence,attention,linear_attention,moe,mtp}`. `--tp-refresh N` needs `N` to be in `--tp` too |

The five that decide how long a run takes, in rough order of effect:

1. `--measurement-iterations` (default 3) — a straight multiplier. 1 is 3x
   faster and 15-25% noisier per shot. It is also the **top level's**
   invocation count: every profile node divides by its parent's, and the top
   level has no parent node, so `extract_samples` is handed this value.
   `embedding`, `lm_head`, `sampler` and Qwen3.5's whole drafter bind there.
2. `--attention-chunk-factor` / `--attention-kv-factor` (2.0) — coarsen the
   two biggest axes geometrically.
2b. `--attention-max-kv` — defaults to the model's own context
   (`max_model_len - max(decode_q_len) - 1`, not `max_model_len` itself: a
   decode occupies `kv + q` positions and needs one more to be a decode, so
   the context length verbatim gets the top point filtered). Capping it at
   16384 nearly halves the sweep. Do that only for a **dense** model: the
   simulator extrapolates past the top profiled kv, and decode attention is a
   pure KV read that is linear in it. On a sparse model the two kernels
   diverge there -- DeepSeek-V3.2's `attention` is flat from `index_topk`
   (2048) onward while its `indexer` keeps growing with the whole KV, so the
   unprofiled region is exactly where the cost lives.
3. `--attention-decode-q-lens` (`1`) — each extra value **doubles** the
   attention sweep. Only needed for speculative decoding.
4. `--skip-skew` — drops the whole second sweep.
5. `--num-hidden-layers` — normally auto-resolved and best left alone, but it
   is why a hybrid costs ~4x a uniform stack: every shot's forward runs all
   the layers the stack needs to expose each block type.

The four that must match what you intend to **simulate**, or the profile
describes a different machine: `--block-size`,
`--gpu-memory-utilization`, `--max-num-batched-tokens`, `--max-num-seqs`.
The last two additionally bound the sweep's axes, and
`--gpu-memory-utilization` sets the KV block count that every
shot-feasibility filter is measured against — so all four change *which*
shots exist, not just the numbers in them.

## Quick start

### 1. Launch the Docker container

```bash
./scripts/docker-vllm.sh
```

The official vLLM image (`vllm/vllm-openai:v0.19.0`, or `:v0.19.0-cu130`
for CUDA 13.x GPUs — edit `scripts/docker-vllm.sh`) already includes every
dependency the profiler needs: vllm, pydantic, pyyaml, rich,
huggingface_hub. No extra pip installs.

The container mounts the **LLMServingSim repo root** as `/workspace`
and starts there. Set your HuggingFace token in
`scripts/docker-vllm.sh` (`-e HF_TOKEN=…`) so gated configs (Llama
etc.) can be fetched automatically on first run.

### 2. Edit `profiler/profile.sh` for your run

The script is a template — open it, change `MODEL` and `HARDWARE`, and
optionally tweak the rest. Every knob below maps to a CLI flag on
`python -m profiler profile`; shell variables left unset stay at the
profiler's built-in defaults.

#### Required

```bash
MODEL="meta-llama/Llama-3.1-8B"     # HF-style <org>/<name>. A raw HF config.json
                                    # must live at configs/model/<MODEL>.json
                                    # (auto-downloaded on first run).
HARDWARE="RTXPRO6000"               # Free-form label → folder name under perf/.
```

#### Sweep shape

```bash
TP_DEGREES="1,2,4"                  # must include 1; profiled one TP at a time on one GPU
MAX_NUM_BATCHED_TOKENS=2048         # vLLM's --max-num-batched-tokens (advisory: the
                                    # profiler internally bumps by +MSQ for shot-bypass
                                    # headroom and subtracts back when recording meta)
MAX_NUM_SEQS=256                    # vLLM's --max-num-seqs. Profile with MSQ > runtime MSQ
                                    # (e.g. profile 256 for a runtime targeting 128) so the
                                    # n = runtime_MSQ mixed corner is feasible.
```

#### Attention grid

```bash
ATTENTION_MAX_KV=""                 # kv_prefill / kv_decode bound; empty = the model's own context
ATTENTION_CHUNK_FACTOR=2.0          # geometric factor for prefill_chunk axis (doubling)
ATTENTION_KV_FACTOR=2.0             # geometric factor for kv axes (doubling)
```

Smaller factors densify that axis; larger factors coarsen it.

#### Measurement averaging

```bash
MEASUREMENT_ITERATIONS=3            # timed forwards per shot, averaged. A single sample
                                    # swings 15–25% on large GEMMs due to DVFS / clock
                                    # jitter; N=3 cuts that to ~5% at ~3× profile time.
```

#### Skew sweep

After the uniform attention grid the profiler runs a heterogeneous-decode
sweep that drives the simulator's FlashAttention-varlen skew correction
(`skew.csv` + `skew_fit.csv`). Four per-axis geometric factors and two
mode switches control it:

```bash
SKIP_SKEW=1                         # skip the sweep entirely — simulator then
                                    # applies no skew correction (alpha = 0).
ONLY_SKEW=1                         # run ONLY the skew step (dense / per_seq /
                                    # attention / moe untouched). Useful when the
                                    # uniform sweep is already done and you just want
                                    # to refresh skew.csv or change the factors.

SKEW_N_FACTOR=2.0                   # n (total decodes) axis — 2.0 = doubling.
SKEW_PC_FACTOR=2.0                  # pc (prefill chunk) axis.
SKEW_KP_FACTOR=2.0                  # kp (prefill history length) axis.
SKEW_KVS_FACTOR=2.0                 # kvs (small-decode kv) axis.
```

Crank any factor above 2.0 to coarsen that axis and cut profile time
(skew fires 3 shots per case, so coarsening compounds). Drop below 2.0
for denser sampling in axes where accuracy matters. The effective
values land in `meta.yaml::skew_profile.factors`.

#### Resume vs force

```bash
FORCE=1                             # wipe every CSV for this variant and re-profile
                                    # from scratch.
```

Default is **resume**: existing CSVs are preloaded row by row, and only
shots whose identity key isn't already present get fired. This lets you
extend an earlier sweep after changing feasibility (e.g. raising
`MAX_NUM_SEQS` from 128 to 256 so mixed `n=128` corners become feasible)
in minutes instead of hours. Resume applies to every category plus
skew; `FORCE=1` nukes them all.

#### Output naming

```bash
VARIANT="my_experiment"             # override the auto-derived <variant> folder name.
```

When omitted, `<variant>` is composed from the effective DTYPE + KV
dtype — `bf16`, `bf16-kvfp8`, `fp8-kvfp8`, etc. — so you never collide
when profiling multiple precisions. Set this explicitly only for named
runs (quantization schemes, experiments).

#### Dtype

```bash
DTYPE="bfloat16"                    # bfloat16 / float16 / float32 / fp8. Inferred
                                    # from the model's torch_dtype when unset.
KV_CACHE_DTYPE="fp8"                # auto / fp8 / fp16 / bf16 — defaults to "auto"
                                    # (inherits DTYPE). `fp8` produces a `-kvfp8`
                                    # suffix on the variant folder and halves KV
                                    # cache memory in the simulator.
```

#### Verbosity

```bash
VERBOSITY="--silent"                # warnings only
VERBOSITY="--verbose"               # DEBUG + vLLM stdout
```

### 3. Run

```bash
./profiler/profile.sh
```

The profiler:

1. Reads `configs/model/<MODEL>.json` (a raw HF `config.json`). If the
   file is absent and `MODEL` is an HF-style id, the config is
   downloaded from the hub and cached at that path automatically.
2. Picks the matching architecture yaml under `models/` by
   `model_type` (the config's field must equal the yaml filename).
   Fails with a clear error and an "available architectures" list if
   nothing matches.
3. Writes the model config to a temp directory and spins vLLM up
   against it — no HF round-trip is needed after the first fetch.
4. Sweeps dense / per-sequence / attention (and MoE if applicable)
   shot grids, writing CSVs under `perf/<HW>/<MODEL>/<variant>/tp<N>/`.

`<variant>` is auto-named from the weight + KV dtype (`bf16`,
`bf16-kvfp8`, `fp8-kvfp8`, …) so different precisions land in
different folders without collisions. Override via `VARIANT=<name>`
only for named runs (quantization schemes, experiments).

### 4. Use in simulation

The simulator's `trace_generator.py` reads from
`profiler/perf/<hardware>/<model>/<variant>/tp<N>/*.csv`
automatically when the cluster config names a matching hardware and
the CLI selects a matching model.

### Sweeping several models: `profiler/profile-all.sh`

Helper template that wraps `python -m profiler profile` in a loop over
a few canned models. Current list: `Qwen/Qwen3-32B`,
`Qwen/Qwen3-30B-A3B-Instruct-2507`, `meta-llama/Llama-3.1-8B` — each
profiled at TP=1 and TP=2 on the same hardware. Useful for bringing
up a fresh GPU target in one shot.

```bash
./profiler/profile-all.sh
```

All knobs are environment variables (no argparse). Defaults match
`profiler/profile.sh`; override inline when you need something else:

```bash
HARDWARE=H100 \
TP_DEGREES=1,2,4 \
ATTENTION_CHUNK_FACTOR=1.5 \
./profiler/profile-all.sh
```

Recognised variables:
`HARDWARE`, `TP_DEGREES`, `MAX_NUM_BATCHED_TOKENS`, `MAX_NUM_SEQS`,
`ATTENTION_MAX_KV`, `ATTENTION_CHUNK_FACTOR`, `ATTENTION_KV_FACTOR`,
`SKEW_N_FACTOR`, `SKEW_PC_FACTOR`, `SKEW_KP_FACTOR`, `SKEW_KVS_FACTOR`,
`SKIP_SKEW`, `ONLY_SKEW`, `MEASUREMENT_ITERATIONS`, `DTYPE`,
`KV_CACHE_DTYPE`, `VARIANT`, `VERBOSITY`.

To change the model list, edit the `MODELS=( ... )` array at the top
of the script. This file is meant to be copied or tweaked in-place,
not treated as a stable CLI.

## Output schema

Each `perf/<hw>/<model>/<variant>/` directory contains one `meta.yaml`
(profiler / vLLM version, GPU, timestamps, effective engine kwargs,
compact sweep specs, skew fit summary) and one `tp<N>/` subfolder per
profiled TP degree:

```
tp<N>/
  dense.csv              layer, tokens, time_us
  per_sequence.csv       layer, sequences, time_us
  attention.csv          layer, prefill_chunk, kv_prefill, n_decode,
                         kv_decode, decode_q_len, time_us
  linear_attention.csv   layer, prefill_tokens, n_decode, time_us
                                                     (mamba / gated-DeltaNet only)
  moe.csv                tokens, activated_experts, time_us          (MoE only)
  mtp.csv                layer, sequences, time_us      (one drafter pass;
                                             only with --profile-mtp)
  skew.csv               raw heterogeneous-decode shots (layer, regime, n, nb,
                         ratio, skew, pc, kp, kvs, kv_big, kv_mean, t_mean_us,
                         t_max_us, t_skew_us, alpha)                  (skew-enabled runs)
  skew_fit.csv           fitted per-bucket alpha table (pc, n_label,
                         skew_rate_label, kv_big_label, kp_label,
                         alpha, n_samples)                            (skew-enabled runs)
```

Times are in microseconds.

**Every category is keyed by `layer` except `moe`.** That is not cosmetic:
the `attention` category holds **more than one kernel** on a sparse model and
they share neither a latency curve nor a skew alpha. MiniMax-M3 profiles
`attention` (its non-sparse layers), `sparse_attention` and `indexer`;
DeepSeek-V3.2 / GLM-5 profile `attention` (MLA) and `indexer`. Reading a
pooled table would give a sparse layer the dense kernel's latency — 2.1x per
layer on M3.

Attention is a single **5D** table covering pure-prefill, pure-decode and
mixed kernel shapes (what vLLM's chunked-prefill scheduler actually produces
each step). The axes grow geometrically — `prefill_chunk` and the kv axes by
`ATTENTION_CHUNK_FACTOR` and `ATTENTION_KV_FACTOR` (both default 2.0),
`n_decode` always on doubling. `decode_q_len` is the fifth axis and defaults
to just `[1]`, because it only matters for speculative decoding and each extra
value multiplies the whole sweep; see `ATTENTION_DECODE_Q_LENS`.

`linear_attention.csv` has only two axes because there is no kv axis at all:
a gated-DeltaNet state is fixed-size per sequence, so cost is independent of
sequence position (measured: 1.1% over a 64x kv spread) and no skew
correction applies. It has *two* rather than one because **which kernel runs
depends on the batch mix** — a pure decode runs a recurrent kernel, add a
prefill chunk and vLLM switches to a fused-gating one. A kernel that does not
fire in a regime simply has **no rows** for it, and the simulator reads that
absence as "does not run here" rather than interpolating across the gap.
Measured on Qwen3.8-27B:

| kernel | prefill | decode | mixed |
|---|---|---|---|
| `gdn_conv_prefill`, `gdn_post_conv`, `gdn_prefill` | yes | — | yes |
| `gdn_conv_decode`, `gdn_decode` | — | yes | — |
| `gdn_decode_mixed` | — | — | yes |
| `gdn_in_proj`, `gdn_out_proj`, `gdn_norm`, `gdn_glue` | \* | \* | \* |

\* the always-on four live in `dense`, not here — `dense` has no notion of
regime, so a regime-dependent kernel placed there would be charged on every
batch.

`meta.yaml` contains the resolved engine state plus three groups of sweep
metadata:

- `engine_resolved.per_tp[tp]` — what the engine *settled on* at each TP
  degree, as opposed to what was asked for: `block_size`, `max_model_len`,
  `num_cache_tokens`. The block size is the one that matters, because vLLM
  treats `--block-size` as a floor and an alignment unit and raises it until
  one attention page covers one mamba page. Qwen3.8-27B resolves to **784**
  from a requested 16. The simulator reads back the entry for its instance's
  `tp_size`, so lookups match the block size the latencies were measured at.

  Keyed by TP because the resolved size is a per-rank fact — both pages scale
  with the shard. A `slice` refresh of one TP **merges** into what is already
  there rather than replacing it. A bundle written before the split carries a
  flat `block_size`, which is still read as a fallback.

- `attention_grid` — the 4D attention sweep's caps (`max_kv`),
  geometric factors (`chunk_factor`, `kv_factor`), and compact spec
  strings for the `chunks` / `n_decode` / `kv` axes.
- `skew_profile` — per-axis factors (`n`, `pc`, `kp`, `kvs`) and
  compact grid specs for the skew sweep. `factors` appears above
  `grid` so you can see the density knobs before the values they
  produced.
- `skew_fit` — the fit summary per TP (`method`, `n_samples`,
  `alpha_default`, `rel_err_p50/p90/p99`, `signed_mean`,
  `bucket_table` pointer) plus the shared `bucket_axes` block. The
  full per-bucket alpha mapping lives in each TP's `skew_fit.csv`.

## Skew profiling & alpha fit

FlashAttention's varlen kernel pays a tile-padding + SM-imbalance
penalty when a decode batch's kv lengths aren't uniform. The uniform
attention grid can't see that — every shot there has all decodes at
the same kv — so we run a second, narrower sweep on purpose-built
bimodal batches:

```
t_mean   — all decodes uniform at the batch's mean kv
t_max    — all decodes uniform at the batch's max kv
t_skew   — the actual skewed batch [nb × kv_big, (n-nb) × kvs]
```

From these three we get a normalised alpha per case:

```
alpha = (t_skew - t_mean) / (t_max - t_mean)
```

Nothing clamps this ratio, and the measured data does leave `[0, 1]`:
across the bundles in `perf/`, 14-20% of rows are negative (the
endpoint gap sitting inside measurement noise) and 2-5% exceed 1 (a
skewed mix genuinely costing more than uniform-max, since tile padding
and SM imbalance are not bounded by it). p50 is 0.07-0.13, p90 is
0.46-0.96. Rows where `t_max <= t_mean` are recorded as `nan` and
dropped by the fit.

which the simulator then applies at query time:

```
t_predicted = t_mean_lookup(batch.mean_kv) +
              alpha(batch.shape) × (t_max_lookup(batch.max_kv) − t_mean_lookup(batch.mean_kv))
```

### Sweep structure

Two tiers make up `skew.csv`:

- **Tier 1** — factorial over `(n, ratio, pc, kp, kvs)` at a single
  representative skew factor (`_SKEW_REP = 4.0`). Gives the bulk of the
  rows and covers every (pc, n_bin, kv_big_bin, kp_bin, skew_rate_bin)
  cell the fit discriminates on.
- **Tier 2** — skew-axis sweep at a handful of anchor pivots with
  `skew ∈ {1.5, 2.0, 4.0, 8.0, 16.0}`. The only source of rows with
  `skew ≠ 4.0`; covers how alpha saturates as the outlier decode
  stretches.

(A former Tier 3 for the kvs axis was removed once T1 grew dense
enough along kvs.)

### Density knobs

All five axes are user-controllable via per-axis geometric factors
(defaults 2.0 = doubling):

| Variable | Axis | Effect |
|---|---|---|
| `SKEW_N_FACTOR` | `n` (total decodes) | coarsen to fire fewer batch sizes |
| `SKEW_PC_FACTOR` | `pc` (prefill chunk) | coarsen to skip prefill-chunk scales |
| `SKEW_KP_FACTOR` | `kp` (prefill history) | coarsen long-context anchors |
| `SKEW_KVS_FACTOR` | `kvs` (small-decode kv) | coarsen the kv sweep |

Higher values → fewer points → faster sweep. Lower → denser grid →
more accurate alpha near the fine structure. The effective values
hit `meta.yaml::skew_profile.factors` so you can tell later which
density produced which CSV.

### 5-axis alpha fit

`fit_alpha.py` runs right after profiling and groups rows by a
5-tuple bucket key:

```
pc | n_label | skew_rate_label | kv_big_label | kp_label
```

Each cell gets a weighted-LS alpha. Axis ablation on the widened
~13k-sample dataset selected this 5-axis scheme (test p50 / p90 /
p99 ≈ 2.7 / 14.8 / 44.1 % on TP=1 vs 3.5 / 16.4 / 39.9 for the
previous 3-axis fit).

**Bucket axes are data-driven.** `n` and `kp` bins are derived one
per unique profiled value (with a sentinel-bin for `kp=0` and an
overflow bin for runtime values beyond the sweep); `kv_big` uses a
log-4x doubling scheme adapted to the observed max; `skew_rate` is a
normalised [0, 1] metric with fixed bin edges; `pc` is used raw (not
bucketed) so every profiled grid point becomes its own alpha column.
This means widening the sweep (raising `MAX_NUM_SEQS` above 128 or
`ATTENTION_MAX_KV`, whose default is now the model's own context) lights up proper resolution on the
affected axis without any code change — the fitter writes the axes it
used into `meta.yaml::skew_fit.bucket_axes` and the simulator reads
them from there.

### Disabling / re-fitting

- Set `SKIP_SKEW=1` to skip the sweep entirely (uniform attention
  grid only). The simulator's fallback constant is **0**, so it then
  applies no skew correction at all and returns `t_mean` -- it does
  not borrow an alpha from another GPU.
- Set `ONLY_SKEW=1` to skip every other category and refresh just
  `skew.csv` + `skew_fit.csv` — useful after widening the grid or
  tweaking factors.

## Architecture yamls

`models/<model_type>.yaml` describes one vLLM model family's class
structure — embedding, layernorm, qkv_proj, attention, etc. The file name is
the `model_type` it primarily serves; a file may serve several by listing them
under `model_types:`, which is how one catalog covers a family whose dense and
MoE variants report different `model_type` values (`qwen3` and `qwen3_moe`
both resolve to `qwen3.yaml`). Catalog entries bind a canonical name to a vLLM
class, with an optional `within:` parent — a single name, or a list of
alternatives when the class differs by checkpoint shape — to disambiguate
duplicate class names:

```yaml
catalog:
  dense:
    qkv_proj:
      vllm: QKVParallelLinear
    layernorm:
      vllm: RMSNorm
      within: LlamaDecoderLayer    # disambiguates from final_layernorm
      tp_stable: true
    …
  per_sequence:
    lm_head:
      vllm: LogitsProcessor
    sampler:
      vllm: Sampler
      tp_stable: true
  attention:
    attention:
      vllm: Attention
  moe:                             # present only for MoE families
    moe:
      vllm: Qwen3MoeSparseMoeBlock
```

`tp_stable: true` marks layers whose kernel cost doesn't change with
TP (layernorms, sampler). They're profiled once at TP=1 and replicated
to other tp folders by the writer.

A `vllm:` name may also be a **raw CUDA kernel name** (matching strips a
trailing `(...)`, and kernel nodes have no parentheses), a **list** of names —
"every one of these the checkpoint has", summed when several match — or carry a
trailing `*` to match by **prefix**, which is the only workable binding for a
fused kernel whose reported name inlines its dtypes. `not_within:` excludes an
ancestor, for when one class plays two roles `within:` cannot separate.

Beside the catalog, a yaml declares the layer order as `blocks:` (keyed by
axis: `attn.<layer_types value>`, `sparse_attn.<same>` as an overlay,
`mlp.dense|moe`) plus `shared:` (`prologue`, `head`). Which block a given layer
runs comes from the checkpoint's own config, resolved by `core/stack.py` — the
same module the simulator reads, so the two cannot disagree. A uniform stack is
the degenerate case: one entry per axis.

### Checking a catalog: `python -m profiler coverage`

A catalog entry can name a real vLLM class and measure **nothing**: the profile
tree holds only modules that launch a kernel of their own, and modern models
fuse q-norm/rope/KV-write into one kernel with no module, or write attention as
bare Triton kernels launched straight from the block. The module tree still
shows the classes, so the mistake is invisible in the source.

```bash
python -m profiler coverage MiniMaxAI/MiniMax-M3 --hardware <hw>
```

boots once, runs one forward per batch regime (prefill-only / decode-only /
mixed — which kernels fire depends on the mix) and reports how much of the
measured CUDA time the catalog binds, exiting non-zero while any is unbound.
Run it whenever you write or edit a catalog, and after a vLLM upgrade.

### The opposite failure: an entry that binds too much

Coverage reports what is *un*bound, so it is blind to an entry that also claims
the **target's** nodes. That happens whenever the guard is missing or wrong —
the drafter's modules are the same classes as the target's, and so is
DeepSeek's shared expert against a dense layer's `mlp`. Qwen3.8-27B's
`mtp_norms` recorded **1287 µs at one sequence** for two RMSNorms with a
perfectly smooth monotone curve. Dump the tree and read the ancestor chains:

```bash
python3 .claude/dump_mtp_tree.py Qwen/Qwen3.8-27B 4 RMSNorm    # drafter
python3 .claude/dump_m3_tree.py 4                              # per regime
```

### Sanity-check a fresh bundle against a bandwidth bound

Nothing in a CSV reveals a constant-factor error. What does is physics:
`lm_head` reads the whole output embedding, so

```
vocab * hidden * dtype_bytes / mem_bw
```

is a hard floor. Llama-3.1-8B on an RTX PRO 6000: 128256 × 4096 × 2 B ÷
1.8 TB/s = 583 µs, against which a measured 714 µs is 82% efficiency — and
every bundle in `profiler/perf/` lands at 80-83%, so an outlier is a bug.
Decode attention is a pure KV read and admits the same check. Also compare
against an existing bundle for the same model on other hardware, scaled by
memory bandwidth.

## Adding a new model

1. **Drop its HF `config.json`** at `configs/model/<org>/<name>.json`.
   (Or let the profiler auto-download on first run if `HF_TOKEN` is
   set in the container.)
2. **If the model's `model_type` is already supported** (llama / qwen3 /
   qwen3_moe / qwen3_5 / qwen3_5_moe / deepseek_v32 / glm_moe_dsa /
   minimax_m3_vl / mixtral / phimoe), you're done — edit `MODEL=` in
   `profiler/profile.sh` and run. A hybrid stack needs no extra setting: the
   layer count is resolved from the checkpoint's config and logged
   (`NUM_HIDDEN_LAYERS` still overrides it).
3. **If it's a new architecture family** (e.g., `gemma2`):
   * Create `models/<model_type>.yaml` mapping the new family's vLLM
     classes to canonical names.
   * Read the model's source under
     `../vllm/vllm/model_executor/models/<name>.py` for orientation, but
     **write the catalog from a live profile dump, not from the source** — the
     module tree and the profile tree differ both ways.
   * Run `python -m profiler coverage <model>` until it binds every kernel,
     *then* profile.
   * If the family varies anything per layer, teach `core/stack.py` the rule —
     read it out of vLLM's source rather than guessing, since the vendors'
     conventions genuinely disagree.

## Custom model shapes

To profile hypothetical shapes (e.g., "Llama-300B":
16384 hidden × 128 heads × 80 layers), just drop a custom config into
`configs/model/custom/my-model.json` with the desired dimensions and
a recognized `model_type`:

```json
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 16384,
  "intermediate_size": 53248,
  "num_attention_heads": 128,
  "num_hidden_layers": 80,
  "num_key_value_heads": 16,
  "vocab_size": 128256,
  "max_position_embeddings": 32768,
  "rms_norm_eps": 1e-05,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "hidden_act": "silu"
}
```

Set `MODEL="custom/my-model"` in `profile.sh` and run. The profiler
writes this exact config into a temp dir for vLLM, so no HF repo has
to exist for the shape you want to measure.

## Verbosity

```
(default)                    INFO — TP limits, stage timings, progress.
--silent                     WARNING — warnings only.
--verbose                    DEBUG + vLLM stdout/stderr.
--log-level {DEBUG,INFO,…}   explicit override.
```

Set via `VERBOSITY="--silent"` / `"--verbose"` in `profiler/profile.sh`,
or pass `--log-level X` to `python -m profiler profile` directly.

## Slice-refresh (partial re-profile)

After the first full sweep, iterate on one category (e.g., tune the
attention grid) without redoing everything:

```bash
python -m profiler slice meta-llama/Llama-3.1-8B \
    --hardware RTXPRO6000 --tp-refresh 1 --group attention
```

Overwrites only that `tp1/attention.csv` and refreshes `meta.yaml`.

`--tp-refresh N` requires `N` to be one of `--tp`'s degrees, which defaults to
`1` — so refreshing a `tp2/` folder needs both:

```bash
python -m profiler slice Qwen/Qwen3.8-27B --hardware RTXPRO6000     --tp 1,2 --tp-refresh 2 --group mtp --profile-mtp
```

Otherwise it exits with `tp=2 is not in the session's tp_degrees ([1])`.
