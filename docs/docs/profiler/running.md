---
sidebar_position: 2
title: Running
---

# Running the profiler

The profiler is invoked through `profiler/profile.sh`: an editable
template. You change the variables at the top to whatever you want
to profile, then run it.

> Looking for adding a brand-new hardware target (GPU or non-GPU)?
> See **[Adding new hardware](./adding-hardware)**. This page covers
> the day-to-day "I have a config, I want to profile it" flow.

## Quick start

From inside the vLLM Docker container at `/workspace`:

```bash
# Edit the variables at the top of profiler/profile.sh, then:
./profiler/profile.sh
```

The script auto-resolves the model architecture from the HF
`config.json`'s `model_type` field, you don't specify it on the
command line. The matching architecture YAML must exist under
`profiler/models/<model_type>.yaml`. See
**[Adding a model architecture](./adding-model-architecture)** if it
doesn't.

## What `profile.sh` does, in order

1. Reads `configs/model/<MODEL>.json` (a raw HF `config.json`). If
   absent and `MODEL` is an HF id, downloads from the hub and caches
   there.
2. Picks the matching architecture YAML by `model_type`.
3. Writes the model config to a tmpdir; spins vLLM up against that.
4. Sweeps **dense / per_sequence / attention / moe** shot grids,
   writing CSVs under `profiler/perf/<HW>/<MODEL>/<variant>/tp<N>/`.
5. (If `SKIP_SKEW=0`, the default) Runs the heterogeneous-decode
   skew sweep and fits per-bucket alphas to `skew_fit.csv`.
6. Writes `meta.yaml` summarizing the run.

For each TP degree in `TP_DEGREES`, the simulator emulates that TP
on a single GPU by dividing the model's per-rank shapes via
`hf_overrides`. **You only need one GPU** to profile any TP degree.

## Required variables

| Variable | Meaning |
| --- | --- |
| `MODEL` | HF-style `<org>/<name>`. Must have a config at `configs/model/<MODEL>.json` (auto-downloaded on first run) |
| `HARDWARE` | Free-form label that becomes the folder name under `profiler/perf/`. Pick something meaningful (e.g., `RTXPRO6000`, `H100`, `MI300X`) |

## Sweep shape

| Variable | Default | Meaning |
| --- | --- | --- |
| `TP_DEGREES` | `1,2` in `profile.sh` (`--tp` defaults to `1`) | Comma-separated TP degrees. **Must include `1`** (TP-stable layers are profiled once at TP=1 and replicated to other TP folders) |
| `MAX_NUM_BATCHED_TOKENS` | `2048` | Profiler internally bumps this by `+MSQ` for shot-bypass headroom; subtracted back when recording meta |
| `MAX_NUM_SEQS` | `256` | Profile with `MSQ > runtime MSQ` so mixed-regime cases at `n = runtime_MSQ` stay feasible |

## Attention grid

:::note[The `decode_q_len` axis is opt-in]
`--attention-decode-q-lens` (default `1`) sweeps how many query tokens each
decode sequence submits. Ordinary decoding is 1; a **speculative-decoding**
verification step is `1 + num_speculative_tokens`, which none of the other four
axes can express — *n* sequences each submitting *k+1* queries against their own
KV is neither one prefill chunk of `n*(k+1)` tokens nor that many single-token
decodes, because the k+1 queries of one sequence share that sequence's KV read.

It multiplies the whole grid, so pass only the `1 + N` values you intend to
simulate; the published N for the four modern families are 3, 4 and 5. The
simulator falls back to the nearest profiled value with a warning rather than
interpolating, because query length changes the kernel's tile shape rather than
just its size.
:::


The 4D attention sweep covers `(prefill_chunk, kv_prefill, n_decode,
kv_decode)`. Three knobs control its shape:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ATTENTION_MAX_KV` | `16384` | Upper bound for `kv_prefill` and `kv_decode` axes |
| `ATTENTION_CHUNK_FACTOR` | `2.0` | Geometric factor for `prefill_chunk` axis (doubling) |
| `ATTENTION_KV_FACTOR` | `2.0` | Geometric factor for `kv` axes (doubling) |

Smaller factors densify the axis (more shots, slower); larger factors
coarsen it (fewer shots, faster).

## Measurement averaging

```bash
MEASUREMENT_ITERATIONS=3
```

Number of timed forwards per shot, averaged. A single sample swings
15–25% on large GEMMs due to DVFS / clock jitter. `N=3` cuts that to
~5% at ~3× profile time. Bump to 5 if you need very tight numbers.

## Skew sweep

After the uniform attention grid, the profiler runs a
heterogeneous-decode sweep that drives the simulator's
FlashAttention-varlen skew correction:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SKIP_SKEW` | unset | Set to `1` to skip the skew sweep entirely. The simulator then applies no skew correction (`alpha = 0`) |
| `ONLY_SKEW` | unset | Set to `1` to run **only** the skew step, leaving dense / per_seq / attention / moe untouched. Useful for refreshing `skew.csv` |
| `SKEW_N_FACTOR` | `2.0` | `n` (total decodes) axis density. Higher = fewer shots |
| `SKEW_PC_FACTOR` | `2.0` | `pc` (prefill chunk) axis |
| `SKEW_KP_FACTOR` | `2.0` | `kp` (prefill history length) axis |
| `SKEW_KVS_FACTOR` | `2.0` | `kvs` (small-decode kv) axis |

The skew sweep fires three shots per case (`t_mean`, `t_max`,
`t_skew`), so coarsening with `>2.0` factors cuts profile time
substantially. See **[Skew & alpha fit](./skew-alpha-fit)** for the
methodology.

## Resume vs force

| Variable | Default | Meaning |
| --- | --- | --- |
| `FORCE` | unset | Set to `1` to wipe every CSV for this variant and re-profile from scratch |

Default is **resume**: existing CSVs are preloaded row by row, and
only shots whose identity key isn't already present get fired. This
lets you extend an earlier sweep after changing feasibility (e.g.,
raising `MAX_NUM_SEQS` from 128 to 256) in **minutes** instead of
hours. Resume applies to every category plus skew; `FORCE=1` nukes
them all.

## Output naming

| Variable | Default | Meaning |
| --- | --- | --- |
| `VARIANT` | auto-derived | Override the variant folder name |

When omitted, `<variant>` is auto-composed from `DTYPE` + `KV_CACHE_DTYPE`:

- `bfloat16` → `bf16`
- `bfloat16` + `fp8` KV → `bf16-kvfp8`
- `fp8` + `fp8` KV → `fp8-kvfp8`

You almost never need to override this. Set explicitly only for
named experimental runs (quantization schemes, ablations).

## Dtype

| Variable | Default | Meaning |
| --- | --- | --- |
| `DTYPE` | `bfloat16` | Model weight dtype: `bfloat16` / `float16` / `float32` / `fp8`. Inferred from `torch_dtype` when unset |
| `KV_CACHE_DTYPE` | `auto` | KV cache dtype: `auto` (inherits `DTYPE`) / `fp8` / etc. `fp8` halves KV memory in the simulator |

## Verbosity

```bash
VERBOSITY="--silent"        # warnings only
VERBOSITY="--verbose"       # DEBUG + vLLM stdout
VERBOSITY=""                # default (INFO)
```

## Calling `python -m profiler` directly

`profile.sh` is a convenience wrapper; every variable in it maps to a
flag. Call the module yourself when you want to script a sweep, or for
the `slice` and `coverage` subcommands, which `profile.sh` does not expose
at all.

```bash
python -m profiler profile  <model> --hardware <hw> [options]
python -m profiler slice    <model> --hardware <hw> --tp-refresh N --group G [options]
python -m profiler coverage <model> --hardware <hw> [options]
```

`<model>` is an HF-style `<org>/<name>` resolving to
`configs/model/<org>/<name>.json`, or an explicit path ending in
`.json`. HF-style ids are auto-downloaded from the Hub on first use
(honouring `HF_TOKEN`); explicit paths are never fetched, so a missing
file is an error.

### Flags shared by both subcommands

| Flag | Default | `profile.sh` variable |
| --- | --- | --- |
| `--hardware` | **required** | `HARDWARE` |
| `--tp` | `1` | `TP_DEGREES` |
| `--variant` | auto-derived from dtypes | `VARIANT` |
| `--dtype` | vLLM default (model's `torch_dtype`) | `DTYPE` |
| `--kv-cache-dtype` | `auto` | `KV_CACHE_DTYPE` |
| `--max-num-batched-tokens` | `2048` | `MAX_NUM_BATCHED_TOKENS` |
| `--max-num-seqs` | `256` | `MAX_NUM_SEQS` |
| `--block-size` | `16` | `BLOCK_SIZE` |
| `--gpu-memory-utilization` | `0.9` | `GPU_MEMORY_UTILIZATION` |
| `--max-model-len` | from the model config | `MAX_MODEL_LEN` |
| `--num-hidden-layers` | `1` | `NUM_HIDDEN_LAYERS` |
| `--hf-override KEY=VALUE` | none | `HF_OVERRIDES` (array) |
| `--linear-attn-chunk` | config `chunk_size`, else vLLM's `FLA_CHUNK_SIZE` | `LINEAR_ATTN_CHUNK` |
| `--attention-max-kv` | `16384` | `ATTENTION_MAX_KV` |
| `--attention-decode-q-lens` | `1` | `ATTENTION_DECODE_Q_LENS` |
| `--attention-chunk-factor` | `2.0` | `ATTENTION_CHUNK_FACTOR` |
| `--attention-kv-factor` | `2.0` | `ATTENTION_KV_FACTOR` |
| `--measurement-iterations` | `3` | `MEASUREMENT_ITERATIONS` |
| `--skip-skew` | off | `SKIP_SKEW=1` |
| `--only-skew` | off | `ONLY_SKEW=1` |
| `--skew-n-factor` | `2.0` | `SKEW_N_FACTOR` |
| `--skew-pc-factor` | `2.0` | `SKEW_PC_FACTOR` |
| `--skew-kp-factor` | `2.0` | `SKEW_KP_FACTOR` |
| `--skew-kvs-factor` | `2.0` | `SKEW_KVS_FACTOR` |
| `--force` | off (resume) | `FORCE=1` |
| `--out-root` | `profiler/perf` | `OUT_ROOT` |
| `--model-config-root` | `configs/model` | `MODEL_CONFIG_ROOT` |
| `--log-level` | `INFO` | `VERBOSITY` |
| `--silent` | — | `VERBOSITY="--silent"` |
| `--verbose` | — | `VERBOSITY="--verbose"` |

Use `--out-root` to write a bundle somewhere other than
`profiler/perf/`, and `--model-config-root` to point at a different
tree of HF configs — useful for profiling hypothetical shapes without
adding them to the repo.

`--log-level`, `--silent`, and `--verbose` are mutually exclusive.
`--silent` is `WARNING`, `--verbose` is `DEBUG` **plus** vLLM's own
stdout, and `--log-level` overrides either explicitly.

`--tp` must include `1`: TP-stable layers (layernorms, sampler) are
profiled once at TP=1 and replicated into the other `tp<N>/` folders by
the writer, so a sweep without TP=1 has nothing to replicate from.

### `slice`: refresh one (tp, category) pair

After a full sweep, iterate on a single category without redoing the
rest:

```bash
python -m profiler slice meta-llama/Llama-3.1-8B \
    --hardware RTXPRO6000 --tp-refresh 1 --group attention
```

| Flag | Required | Description |
| --- | --- | --- |
| `--tp-refresh` | ✓ | The single TP degree to refresh. Must be a member of `--tp` |
| `--group` | ✓ | One of `dense`, `per_sequence`, `attention`, `linear_attention`, `moe` |

It boots one engine at that TP, fires only that category's grid, and
rewrites `tp<N>/<group>.csv` plus `meta.yaml`. Errors out if the
architecture YAML has no entries in `catalog.<group>` — asking for
`moe` on a dense model, for instance.

Note `slice` handles only the uniform categories. The skew sweep is not a
`--group` value; refresh it with
`python -m profiler profile ... --only-skew` instead.

### `coverage`: does the catalog bind every kernel?

```bash
python -m profiler coverage MiniMaxAI/MiniMax-M3 --hardware RTXPRO6000
```

Boots one engine at TP=1, runs one forward per batch regime
(prefill-only / decode-only / mixed) and reports how much of the measured CUDA
time `profiler/models/<model_type>.yaml` accounts for. Writes nothing, and
exits non-zero while any kernel is left unbound.

```
Coverage check: minimax_m3_vl (18 catalog entries, 3 regimes)
prefill    3997.0 us total,   3997.0 us bound (100.0%), 0 unbound node(s)
decode     4213.3 us total,   4213.3 us bound (100.0%), 0 unbound node(s)
mixed      4454.3 us total,   4454.3 us bound (100.0%), 0 unbound node(s)
Catalog binds every measured kernel, in all 3 regimes.
```

This exists because a catalog entry can name a real vLLM class and still
measure nothing: vLLM's profile tree only contains modules that launch a kernel
of their own, and modern models fuse q-norm, rope and the KV write into a
single kernel with no module wrapper, or write attention as bare Triton kernels
launched straight from the block. The module tree still shows the classes, so
the mistake is invisible in the source — and the symptom is a layer that looks
free rather than an error. TP=1 only, because coverage is about which nodes
exist and TP changes tensor shapes, not the module graph.

Run it whenever you write or edit a catalog, and after a vLLM upgrade. Details
and how to act on a gap: [Adding a model
architecture](./adding-model-architecture#3-check-what-the-catalog-binds).

## Multi-model batch sweep: `profile-all.sh`

For bringing up a fresh GPU target across multiple models in one
shot:

```bash
./profiler/profile-all.sh
```

This wraps `python -m profiler profile` in a loop over a canned
list of models (currently `Qwen/Qwen3-32B`,
`Qwen/Qwen3-30B-A3B-Instruct-2507`, `meta-llama/Llama-3.1-8B`) at
TP=1 and TP=2. All knobs from `profile.sh` are recognized as
environment variables:

```bash
HARDWARE=H100 \
TP_DEGREES=1,2,4 \
ATTENTION_CHUNK_FACTOR=1.5 \
./profiler/profile-all.sh
```

To change the model list, edit the `MODELS=( ... )` array at the top
of the script. This file is meant to be copied or tweaked in-place,
not treated as a stable CLI.

## Expected runtime

Rough numbers for a single model + single TP on RTXPRO6000-class
hardware (`MAX_NUM_BATCHED_TOKENS=2048`, `MAX_NUM_SEQS=256`, default
factors):

| Step | Time |
| --- | --- |
| `dense` | seconds |
| `per_sequence` | seconds |
| `attention` (uniform 4D grid) | 5–15 minutes |
| `moe` (MoE only) | 10–30 minutes |
| `skew` sweep | 10–25 minutes |
| `skew_fit` (post-process) | seconds |

A full multi-TP, multi-model sweep with `profile-all.sh` typically
runs **1–4 hours**. Use `SKIP_SKEW=1` for a much faster pass when
you don't need varlen-skew correction.

The Rich-based logger renders per-step progress bars; redirect
stdout with `--silent` for a quieter run.

## Output

Profile data lands at:

```
profiler/perf/<HARDWARE>/<MODEL>/<variant>/
├── meta.yaml
└── tp<N>/
    ├── dense.csv
    ├── per_sequence.csv
    ├── attention.csv
    ├── moe.csv         (MoE models only)
    ├── skew.csv         (skew-enabled runs)
    └── skew_fit.csv     (skew-enabled runs)
```

Schema reference: **[Output bundle](./output-bundle)**.

## Tips

1. **Always start with `SKIP_SKEW=1`** when bringing up a new
   `(hardware, model)` combo, get the uniform grid done first,
   then add skew once you know the rest works.
2. **`profile.sh` is intended for in-place editing.** Don't try to
   parameterize it via flags; copy it for scenarios that diverge
   substantially.
3. **Profile resumption is granular**: if a single shot crashes,
   you can fix the issue and re-run; the previously-completed shots
   stay cached.
4. **Coarsen the attention grid first**. The 4D attention sweep is
   the longest step. Bump `ATTENTION_CHUNK_FACTOR` to `4.0` if you
   only need rough numbers, then re-run with `2.0` later for
   precision.
5. **Don't profile across CUDA driver versions.** Driver upgrades
   change kernel timings by a few percent; either re-profile after
   driver change or accept the drift.

## What's next

- **[Output bundle](./output-bundle)**: schema for the CSVs you
  just produced.
- **[Skew & alpha fit](./skew-alpha-fit)**: what the skew sweep is
  doing under the hood.
