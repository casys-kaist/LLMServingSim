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
    skew_calibration.py       runtime-reference fit + supported-N bucket lookup
    skew_plan.py              dynamic distribution coverage and feasibility
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
| **MoE expert parallelism** | `--moe-ep-degrees` (default `1`; needed whenever the deployment runs EP > 1) |
| **drafter (MTP)** | `--profile-mtp` (a flag — the engine boots at N=1 so the CSV holds one pass) |
| **attention grid** | `--attention-max-kv`, `--attention-chunk-factor`, `--attention-kv-factor`, `--attention-decode-q-lens` |
| **linear attention** | `--linear-attn-chunk` |
| **measurement** | `--measurement-iterations` |
| **skew** | `--skip-skew`, `--only-skew`, `--skew-n-factor`, `--skew-pc-factor`, `--skew-kp-factor`, `--skew-kvs-factor`, `--skew-samples-per-cell`, `--skew-rounds`, `--skew-seed` |
| **resume** | `--force` (default is resume) |
| **paths** | `--out-root`, `--model-config-root` (no `profile.sh` variable) |
| **verbosity** | `--log-level`, `--silent`, `--verbose` (`VERBOSITY`) |
| **slice only** | `--tp-refresh`, `--group {dense,per_sequence,attention,linear_attention,moe,mtp}`. `--tp-refresh N` needs `N` to be in `--tp` too |

The five that decide how long a run takes, in rough order of effect:

1. `--measurement-iterations` (default 3) — a straight multiplier, and the
   biggest single knob: the profiler costs **372 us per forward against 16 us
   outside it** on a 4-layer DeepSeek-V3.2, and that is linear in the forwards
   (0/1/3/6 inside one session → 6.9/372/1,125/2,310 ms). Session
   setup/teardown is only 6.9 ms, so 1 is ~3x faster and 15-25% noisier per
   shot. It is also the **top level's** invocation count: every profile node
   divides by its parent's, and the top level has no parent node, so
   `extract_samples` is handed this value. `embedding`, `lm_head`, `sampler`
   and Qwen3.5's whole drafter bind there. Repeated top-level summaries are
   deduplicated by full module representation, time and invocation count;
   class-name equality alone must not discard differently shaped modules.
1b. **Layer count, per category** — not a flag; resolved from the checkpoint.
   The same per-forward cost means op count sets the wall clock, and the
   profile tree merges same-class siblings, so a second layer of a type
   already present adds no information. Each category is shrunk to the axes it
   measures (`Category.stack_axes`), and `run_full` boots one engine per
   distinct depth. DeepSeek-V3.2 and GLM-5 need 4 layers for their MLP axis and
   **1** for attention, which is the expensive sweep: 1,057 → 341 ms per shot,
   3.1x measured. Qwen3.8-27B and MiniMax-M3 vary on the attention axis too, so
   they gain nothing. Fewer layers also means a larger `num_cache_tokens`, so
   more shots clear the feasibility filters -- wider coverage, and a grid not
   row-for-row comparable with a deeper run's.
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
   attention sweep. Only needed for speculative decoding. The halves are
   disjoint (`q > 1` never yields a pure-prefill shot, and `decode_q_len` is
   part of the row key), so one model's sweep can run `q=1` on one GPU and
   `q=N` on another and the CSVs concatenate — pin `--attention-max-kv` on
   both, or each half derives its own cap from its own `max(q)` and they
   sweep different kv sets.
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
ATTENTION_MAX_KV=""                 # key-history bound; empty = the model's own context
ATTENTION_CHUNK_FACTOR=2.0          # geometric factor for prefill_chunk axis (doubling)
ATTENTION_KV_FACTOR=2.0             # geometric factor for kv axes (doubling)
```

Smaller factors densify that axis; larger factors coarsen it.

**If the run dies with CUDA OOM, this is the first knob.** Left empty,
`ATTENTION_MAX_KV` is the model's own context window, so a long-context
checkpoint sweeps out to 131k or beyond and the largest decode shots ask the
engine for `n_decode * kv_decode` tokens of KV in one go. Setting it bounds the
biggest allocation the sweep ever makes, and cuts runtime with it -- 8,643
shots at 16,384 against 14,653 at DeepSeek-V3.2's full 163,834. The cost is
long-context coverage: the simulator extrapolates past the last profiled kv,
which is safe on a dense kernel and not on a sparse one (see *Skew profiling*
and the `--attention-max-kv` section of the docs site).

The next knobs down, in order, are `MAX_MODEL_LEN` (caps the engine's context
and hence the KV cache it reserves -- it also caps this one, since the grid
runs to `min(the two)`), `MAX_NUM_SEQS` (caps `n_decode` per shot) and
`MAX_NUM_BATCHED_TOKENS` (caps tokens per shot, i.e. the activation peak).
None of them is free: every shot-feasibility filter is measured against the KV
cache the engine resolved, so these change *which shots the sweep contains*,
not only whether it survives. `GPU_MEMORY_UTILIZATION` cuts both ways -- lower
leaves more room for activations but shrinks the KV cache, filtering out more
of the large-kv shots.

#### Engine limits and model shape

The defaults suit a dense model that fits one card. The modern families need
some of these -- MiniMax-M3 fails outright at the platform's default block size
and asks for 10.5 GiB of KV before the drafter is built:

```bash
BLOCK_SIZE=16                       # vLLM's --block-size. A model whose attention works
                                    # in wider units raises it or refuses to boot:
                                    # MiniMax-M3 needs 128 ("No common block size for 16").
                                    # A hybrid derives its own from the mamba page and
                                    # ignores this -- see "Block size is derived" below.
GPU_MEMORY_UTILIZATION=0.9          # vLLM's --gpu-memory-utilization. Lower it when the
                                    # engine cannot fit both the model and its KV cache
                                    # (MiniMax-M3: 0.75).
MAX_MODEL_LEN=                      # vLLM's --max-model-len; empty = the checkpoint's own.
                                    # Lower it when the declared context reserves more KV
                                    # than the card has (MiniMax-M3 declares 1,048,576).
NUM_HIDDEN_LAYERS=                  # empty = resolved from the checkpoint: the smallest
                                    # prefix that instantiates every block type (4 on a
                                    # hybrid, 1 on a uniform stack). Set it only to
                                    # override that.
HF_OVERRIDES=()                     # array of KEY=VALUE, applied on top of the config on
                                    # disk: HF_OVERRIDES=(index_topk=1024).
LINEAR_ATTN_CHUNK=                  # gated-DeltaNet chunk length; empty = the config's
                                    # chunk_size, else vLLM's FLA_CHUNK_SIZE.
PROFILE_MTP=                        # any non-empty value profiles the drafter. A flag, not
                                    # a count: the engine boots at num_speculative_tokens=1
                                    # so mtp.csv holds ONE pass, which is the unit the
                                    # simulator multiplies by its own N.
```

#### Where output lands

```bash
OUT_ROOT=                           # empty = profiler/perf. Point it elsewhere to build a
                                    # bundle without touching the committed tree -- which is
                                    # how one model's attention sweep is split across two
                                    # GPUs and the halves concatenated.
MODEL_CONFIG_ROOT=                  # empty = configs/model. A different tree of HF configs,
                                    # for profiling hypothetical shapes.
```

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
SKIP_SKEW=1                         # skip acquisition; retain existing data.
                                    # A fresh bundle then has alpha = 0.
ONLY_SKEW=1                         # run ONLY the skew step (dense / per_seq /
                                    # attention / moe untouched). Useful when the
                                    # uniform sweep is already done and you just want
                                    # to refresh skew.csv or change the factors.

SKEW_N_FACTOR=2.0                   # n (total decodes) axis — 2.0 = doubling.
SKEW_PC_FACTOR=2.0                  # pc (prefill chunk) axis.
SKEW_KP_FACTOR=2.0                  # kp (prefill history length) axis.
SKEW_KVS_FACTOR=2.0                 # decode-history envelope axis.
SKEW_SAMPLES_PER_CELL=32            # distribution draws per operating cell.
SKEW_ROUNDS=3                       # independent timed contexts.
SKEW_SEED=0                         # deterministic sampling seed.
```

Crank any factor above 2.0 to coarsen that axis and cut profile time
(each case has three independent contexts by default). Drop below 2.0
for denser sampling. Actual per-TP plans are saved in `skew.meta.yaml`
and copied into `meta.yaml::skew_profile.per_tp`.


#### Measuring the machine: `profiler hardware`

Separate from any model sweep, because it characterises the hardware rather
than a model:

```bash
python -m profiler hardware --hardware RTXPRO6000 --npus 2
```

writes `profiler/perf/RTXPRO6000/hardware.yaml` — **one file per hardware
folder**, shared by every model bundle under it. It carries the card's spec
(queried from the device), an NCCL all-reduce sweep, and a `defaults` block
that cluster configs inherit `link_bw` / `link_latency` /
`npu_mem.mem_size|mem_bw|mem_latency` from when they omit them. Each default
records its `source` — `measured`, `spec`, or `assumed` — and a simulation logs
it, so a run says whether its link numbers came from a benchmark or from
nobody.

That distinction is why the command exists: the committed examples carried
`link_latency: 20000` as a fitted value for four months, NCCL puts it at
16,100 ns, and the fitted number over-charged a decode-sized all-reduce by
10.4% while being free to absorb whatever else was mis-modelled.

**Two GPUs are the floor** — a link has two ends. On a single-GPU machine the
command still writes the spec section, records `interconnect: null` with the
reason, and **exits non-zero** so a script notices. A cluster config on that
hardware then has to name `link_bw` and `link_latency` itself; the simulator
raises rather than substituting a number nobody measured.

`--npus` is recorded, because an all-reduce across two PCIe-linked cards is not
the physics of eight over NVLink and a config asking for more is
extrapolating.

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

Skew feasibility uses the resolved block size for the actual heterogeneous
shot; uniform reference batches are looked up offline, not measured. Skew checkpoints use atomic file
replacement and preserve existing permissions. Corrupt CSVs fail explicitly
instead of being silently replaced; `FORCE=1` still requests a fresh sweep.

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
  attention.csv          layer, prefill_chunk, prefill_key, n_decode,
                         kv_decode, decode_q_len, time_us
  linear_attention.csv   layer, prefill_tokens, n_decode, time_us
                                                     (mamba / gated-DeltaNet only)
  moe.csv                ep, tokens, activated_experts, time_us      (MoE only)
  mtp.csv                layer, sequences, time_us      (one drafter pass;
                                             only with --profile-mtp)
  skew.csv               ordered requests, query roles, per-forward repetitions,
                         protocol, family and measured t_skew_us
  skew.meta.yaml         actual per-TP acquisition plan and completion status
  skew_fit.csv           versioned supported-N calibration cells
                         (layer, decode_q_len, pc_label, lev_label, n_anchor,
                          alpha, direct_rows, pooled_rows)            (skew-enabled runs)
```

Times are in microseconds.

`prefill_key` is query-weighted across prefill chunks; profiling and serving
share its implementation. See the [attention schema](https://llmservingsim.ai/docs/profiler/output-bundle#attentioncsv)
for the formula and legacy single-prefill compatibility.

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
`ATTENTION_CHUNK_FACTOR` and `ATTENTION_KV_FACTOR` (both default sqrt(2)),
`n_decode` uses a sqrt(2) factor. `decode_q_len` is the fifth axis and defaults
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
- `skew_profile` — actual per-TP dynamic plans, resolved capacity and completion.
- `skew_fit` — versioned identity, reference fingerprints, adaptive prefill
  partitions, supported N anchors, per-kernel/query defaults and table checksum.

## Skew profiling & calibration

The default sweep covers bimodal, outlier, trimodal, ramp, lognormal, Pareto,
near-uniform and uniform-spread histories; ordered, reversed, interleaved and
shuffled requests; and equal/unequal multi-prefill splits. Axes follow the
user's sequence/token/context bounds and resolved engine capacity. Query
lengths follow `--attention-decode-q-lens`; each requires its own attention
reference slice. A geometric operating cell gets 32 distribution draws by
default, not workload-derived samples.

Only the actual heterogeneous batch is measured. Three independent contexts
of three timed forwards produce a median of forward medians. Per-forward
times, geometry and query roles are preserved. Failed cases stop with a
checkpoint; incomplete kernel sets and insufficient repetitions are retried.

The writer computes endpoints through the unchanged serving attention lookup,
then fits weighted-median cells at supported N anchors. Prefill boundaries
scale with the measured token envelope; leverage is dimensionless. Runtime
picks one cell, with no fitting or measurement search. Old skew fits are
rejected; disabled bundles, including RTX4090, need no migration.

Preview the acquisition using an existing bundle's engine limits, or rebuild
its calibration without a GPU:

```bash
python -m profiler plan-skew meta-llama/Llama-3.1-8B --hardware RTXPRO6000 --tp 1
python -m profiler refit-skew meta-llama/Llama-3.1-8B --hardware RTXPRO6000 --tp 1
```

`--skip-skew` skips acquisition, not existing calibration. `--only-skew`
requires an existing attention table. New acquisitions record completion
and resolved limits in `tp<N>/skew.meta.yaml`. Refit rejects unfinished
acquisitions and absent raw data
and stale references are rejected at simulator startup. See the
[skew guide](../docs/docs/profiler/skew-alpha-fit.md) for contracts and limits.

Validate every reported `bench/examples` statistic. Reusing broader measured
data can improve one model and worsen another; neither sampling families nor
the mean/max summaries establish generalization to unseen distributions.

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
perfectly smooth monotone curve. Dump the profile tree and read the ancestor
chains: boot at the depth in question -- with `--profile-mtp` for a drafter --
and print every node's class with its ancestor path, once per batch regime,
filtered to the class under suspicion.

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
