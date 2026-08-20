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
   writing CSVs under `perf/<HW>/<MODEL>/<variant>/tp<N>/`.
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
| `HARDWARE` | Free-form label that becomes the folder name under `perf/`. Pick something meaningful (e.g., `RTXPRO6000`, `H100`, `MI300X`) |

## Sweep shape

| Variable | Default | Meaning |
| --- | --- | --- |
| `TP_DEGREES` | `1,2,4` | Comma-separated TP degrees. **Must include `1`** (TP-stable layers are profiled once at TP=1 and replicated to other TP folders) |
| `MAX_NUM_BATCHED_TOKENS` | `2048` | Profiler internally bumps this by `+MSQ` for shot-bypass headroom; subtracted back when recording meta |
| `MAX_NUM_SEQS` | `256` | Profile with `MSQ > runtime MSQ` so mixed-regime cases at `n = runtime_MSQ` stay feasible |

## Attention grid

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
