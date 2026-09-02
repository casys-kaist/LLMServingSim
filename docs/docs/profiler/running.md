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

:::tip[One model's sweep can run on two GPUs]
`q > 1` never yields a pure-prefill shot, and `decode_q_len` is part of the row
key the CSV is written under. So a `q=1` sweep and a `q=N` sweep are disjoint,
and together they are exactly the combined grid — 14,653 + 14,083 = 28,736 for
DeepSeek-V3.2 at its full context. Run one half per GPU and concatenate:

```bash
CUDA_VISIBLE_DEVICES=0 python -m profiler slice <model> --hardware <hw> \
    --tp-refresh 1 --group attention --force \
    --attention-max-kv 163834 --attention-decode-q-lens 1

CUDA_VISIBLE_DEVICES=1 python -m profiler slice <model> --hardware <hw> \
    --tp-refresh 1 --group attention --force \
    --attention-max-kv 163834 --attention-decode-q-lens 5 \
    --out-root /tmp/half_q5
```

**Pin `--attention-max-kv` on both halves.** Left to derive, the resolver reads
each run's own `max(decode_q_lens)`, so the `q=1` half would take 163,838 and
the `q=5` half 163,834 — different kv sets, and the halves stop lining up. Use
the value a combined run would resolve to: `max_model_len - max(q) - 1`.
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
| `--profile-mtp` | off | `PROFILE_MTP=1` |
| `--linear-attn-chunk` | config `chunk_size`, else vLLM's `FLA_CHUNK_SIZE` | `LINEAR_ATTN_CHUNK` |
| `--attention-max-kv` | the model's own context | `ATTENTION_MAX_KV` |
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

### `--measurement-iterations` — averaging out clock jitter

Each shot runs one discarded warm-up forward, then N timed forwards inside a
single `layerwise_profile` context. A single sample can swing 15-25% on a large
GEMM from DVFS and boost-clock jitter, so the default is 3 and the per-call
figure comes from dividing by the invocation count.

The division is **per parent**: every node divides by its parent node's
invocation count. The top level has no parent node, so `extract_samples` is
given `iterations` as its count — and vLLM 0.28 additionally reports a
top-level node once per forward (and once per sibling module) where 0.19 merged
them into one wrapper, so identical sibling entries are deduped first. Both
matter because `embedding`, `lm_head`, `sampler` and Qwen3.5's whole drafter
bind at the top level; without them those read `iterations x repeats` too high.

:::tip[Sanity-check a fresh bundle against a bandwidth bound]
Nothing in a profiled CSV reveals a constant-factor error — the curve stays
smooth and monotone in the sweep axis, and `coverage` passes, because coverage
reports only what is *un*bound. What does reveal it is physics. `lm_head` reads
the whole output embedding, so

```
vocab * hidden * dtype_bytes / mem_bw
```

is a hard floor: 128256 × 4096 × 2 B ÷ 1.8 TB/s = 583 µs for Llama-3.1-8B on
an RTX PRO 6000, against which a measured 714 µs is 82% efficiency and 6417 µs
is impossible. Spot-check `lm_head`, `embedding` and one decode-attention row
this way, and against an existing bundle for the same model on other hardware
scaled by memory bandwidth.
:::

### `--attention-max-kv` — how far out the KV axes reach

Unset, this is **the model's own context window**, resolved against the live
engine as `max_model_len - max(decode_q_len) - 1`. It is not `max_model_len`
itself: a decode request occupies `kv + q` positions and needs one more to be a
decode rather than the whole window, so passing the context length verbatim
gets the top point filtered and the sweep stops one doubling short — on
DeepSeek-V3.2 that is 131,072 where you asked for 163,840.

Lowering it trades coverage for time. Cost is close to linear in the number of
kv values, because the KV-budget filter prunes the large-kv × large-n_decode
corner:

| `--attention-max-kv` | shots (q=1) | with a second `decode_q_len` |
| --- | --- | --- |
| 16,384 | 8,643 (≈4 h) | 16,926 (≈8 h) |
| 131,072 | 13,685 (≈6.3 h) | 26,830 (≈12.4 h) |
| 163,834 (DeepSeek's full context) | 14,653 (≈6.8 h) | 28,736 (≈13.3 h) |

:::caution[On a sparse model the top of the range is not optional]
The simulator extrapolates linearly past the highest profiled kv. That is safe
for a dense kernel, which is linear in kv — decode attention is a pure KV read
and fits `a + b·(n_decode·kv_decode)` at R²=1.0000. It is **not** safe for a
sparse one, where two kernels diverge. Measured on DeepSeek-V3.2 at
`n_decode = 8`:

| `kv_decode` | `attention` (MLA) | `indexer` |
| --- | --- | --- |
| 1,024 | 838 µs | 27 µs |
| 2,048 | 342 µs | 52 µs |
| 8,192 | 350 µs | 72 µs |
| 16,384 | 350 µs | 101 µs |

`attention` flattens at `index_topk: 2048` — past that it only reads the
selected tokens. `indexer` keeps growing, because it scores the whole KV to
make the selection. Extrapolating a flat curve is harmless; extrapolating the
indexer ten-fold past its last measured point is the term that decides a
long-context run.
:::

### `--profile-mtp` — profiling the drafter

A model that drafts with itself keeps its MTP module out of the ordinary
model: vLLM only builds it when the engine boots with a
`speculative_config`, and the MTP config's `model_type`
(`deepseek_mtp` / `qwen3_5_mtp` / `minimax_m3_mtp`) is produced by
`SpeculativeConfig.hf_config_override` and is unknown to HF
Transformers, so it cannot be loaded on its own. `--profile-mtp` boots
with speculative decoding so the drafter exists.

**It takes no draft count.** The engine is pinned to
`num_speculative_tokens=1`, so what lands in `mtp.csv` is **one**
drafter pass — the unit the simulator multiplies by its own
`--num-speculative-tokens`. Booting at N would record N passes in a
single shot and the simulator would multiply again, so the cost came
out N².

Its kernels then arrive for free. The drafter runs inside
`sample_tokens()` (`propose_draft_token_ids` → `drafter.propose`), and
the fire path already calls `execute_model` then `sample_tokens(None)`
inside the same `layerwise_profile` context.

**Run `coverage` with it first.** The drafter's kernels report as
unbound until the catalog binds them, with their ancestor paths — which
is how the `mtp:` sections in `profiler/models/` were written:

```bash
python -m profiler coverage <model> --hardware <hw> --profile-mtp
```

Coverage catches an entry that binds *nothing*. It cannot catch the
opposite — an entry that binds the **target's** layers as well, because
the drafter's modules are the same classes as the target's and
over-matching leaves nothing unbound. That one needs the tree:

```bash
python3 .claude/dump_mtp_tree.py Qwen/Qwen3.8-27B 4 RMSNorm
```

Read the ancestor chains before trusting an `mtp.csv`. Qwen3.8-27B's
`mtp_norms` recorded **1287 µs at one sequence** for two RMSNorms while
its guard was wrong, and the curve stayed smooth and monotone the whole
way.

The sweep itself is cheap. The `mtp` category has **one** axis (the
pass's token count) rather than attention's four: 40 shots in under a
minute. Refresh just that category with
`slice --group mtp --profile-mtp`.

Two per-model requirements, both of which the profiler handles or
reports:

- `num_mtp_modules` is capped to 1 in the config **file**. The drafter
  reads `speculative_config.draft_model_config.hf_config`, built from
  the config on disk, so an `hf_overrides` entry never reaches it.
  MiniMax-M3 declares 7 modules, each a full MoE decoder layer at
  ~14.8 GB — ~103 GB, which does not fit one card. They are identical
  and the simulator multiplies by the declared count.
- MiniMax-M3 also needs `--max-model-len` lowered (its declared
  1,048,576 asks for 10.5 GiB of KV before the drafter is built) and
  benefits from a lower `--gpu-memory-utilization`.

:::caution[MiniMax-M3 needs a patch to start at all]
`scripts/patches/vllm_m3_mtp_layer_name.py`, applied by
`docker-vllm.sh`. Without it M3 fails with `Duplicate layer name:
model.layers.0.self_attn.attn` — it is the one MTP family that neither
offsets its layer index nor separates its prefix, so its drafter
collides with the target in vLLM's layer-name registry.
:::

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
| `--group` | ✓ | One of `dense`, `per_sequence`, `attention`, `linear_attention`, `moe`, `mtp` |

It boots one engine at that TP, fires only that category's grid, and
rewrites `tp<N>/<group>.csv` plus `meta.yaml`. Errors out if the
architecture YAML has no entries in `catalog.<group>` — asking for
`moe` on a dense model, for instance.

`--tp` defaults to `1`, and `--tp-refresh` has to name one of its degrees, so
refreshing a `tp2/` folder needs both — otherwise it exits with `tp=2 is not
in the session's tp_degrees ([1])`:

```bash
python -m profiler slice Qwen/Qwen3.8-27B --hardware RTXPRO6000 \
    --tp 1,2 --tp-refresh 2 --group mtp --profile-mtp
```

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

**Measured**, one TP degree per row, on an RTX PRO 6000 Blackwell with
`MAX_NUM_BATCHED_TOKENS=2048`, `MAX_NUM_SEQS=256`,
`MEASUREMENT_ITERATIONS=3` and default grid factors:

| Model | dense | per_seq | attention | linear_attn / moe | total |
| --- | --- | --- | --- | --- | --- |
| Qwen3.8-27B (hybrid) | 3:05 | 0:39 | **3:55:15** | 3:14 | ~4 h |
| DeepSeek-V3.2-Exp (MLA + DSA) | 3:08 | 0:59 | **2:59:27** | 1:02 | ~3 h |
| GLM-5 (MLA + DSA) | 2:38 | 0:51 | **2:51:39** | 0:58 | ~3 h |
| MiniMax-M3 (block-sparse) | 2:01 | 0:45 | **2:06:37** | 0:37 | ~2 h |

:::caution[`attention` dominates, and it is hours, not minutes]
Every other category finishes in minutes. The attention sweep is
8,643 shots at these limits — measured, not estimated — and at
`MEASUREMENT_ITERATIONS=3` that is ~26,000 timed forwards.

The bottleneck is **not the GPU**. `VLLM::EngineCore` sits at ~100% of
one core for the whole sweep while GPU utilisation samples in the
low tens of percent: the cost is `layerwise_profile`'s single-threaded
attribution of every CUDA kernel to a node in the profile tree.

That is why a hybrid or sparse model costs more than a dense one even
at the same shot count. Two multipliers stack:

- **Layers instantiated.** A uniform stack shrinks to 1; a hybrid needs
  the smallest prefix reaching every block type — 4 for Qwen3.8-27B,
  and 4 for the three MoE families (3 dense MLP + 1 MoE).
- **Nodes attributed per forward.** Catalog entries: Llama 12, Qwen3
  14, MiniMax-M3 18, DeepSeek/GLM 22, Qwen3.5/3.8 **24**.
:::

To cut it, in order of effect: `MEASUREMENT_ITERATIONS=1` (a straight
3x, at 15–25% per-shot noise), then
`ATTENTION_CHUNK_FACTOR` / `ATTENTION_KV_FACTOR` above 2.0 to coarsen
the two biggest axes. `SKIP_SKEW=1` removes the skew sweep entirely —
every number above was measured with it off.

Adding TP degrees multiplies by roughly one more pass each: Qwen3.8-27B
at `--tp 1,2` spent a further **207 minutes** on the TP=2 pass, with
TP=1 resumed from its existing CSVs in seconds.

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
