---
sidebar_position: 2
title: Running
---

# Running the profiler

The profiler is invoked through `profiler/profile.sh`: an editable
template. You change the variables at the top to whatever you want
to profile, then run it. Optional variables are unset by default, so an
unedited script inherits `python -m profiler profile` defaults. Inspect that
command's `--help` for current values; commented assignments are examples.
Nonempty boolean settings enable their flags (even `0`); unset or empty
disables them. The separate `profile-all.sh` is an explicit multi-model
campaign with its own overrides, not a CLI-default wrapper.

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
command line. Resolution tries `profiler/models/<model_type>.yaml`, then
the catalogs' declared `model_types:` aliases. See
**[Adding a model architecture](./adding-model-architecture)** if it
doesn't.

### Characterise the machine first (once per hardware)

Before profiling any model on a new machine, measure what the hardware
actually is:

```bash
python -m profiler hardware --hardware RTXPRO6000 --npus 2
```

This writes `profiler/perf/RTXPRO6000/hardware.yaml` — **one file per hardware
folder**, shared by every model bundle under it. Cluster configs then inherit
`link_bw`, `link_latency` and `npu_mem.*` from it instead of carrying values
somebody guessed. Run it once; it has nothing to do with any particular model.

It needs **two GPUs**, because a link has two ends. On a single-GPU machine it
still records the card's spec, writes `interconnect: null` with the reason, and
**exits non-zero** so a script notices — and a cluster config on that hardware
then has to name `link_bw` / `link_latency` itself rather than inheriting a
number nobody measured.

See **[Cluster config → Hardware facts are inherited](../reference/cluster-config#hardware-facts-are-inherited)**.

## What `profile.sh` does, in order

1. Reads `configs/model/<MODEL>.json` (a raw HF `config.json`). If
   absent and `MODEL` is an HF id, downloads from the hub and caches
   there.
2. Picks the matching architecture YAML by `model_type`.
3. Writes the model config to a tmpdir; spins vLLM up against that.
4. Sweeps **dense / per_sequence / attention / moe** shot grids,
   writing CSVs under `profiler/perf/<HW>/<MODEL>/<variant>/tp<N>/`.
5. (Unless `SKIP_SKEW` is set) Runs the heterogeneous-decode
   skew sweep and fits per-bucket alphas to `skew_fit.csv`.
6. Writes `meta.yaml` summarizing the run.

For each TP degree in `TP_DEGREES`, the profiler emulates that TP
on a single GPU by dividing the model's per-rank shapes via
`hf_overrides`. **You only need one GPU** to profile any TP degree.

## Required variables

| Variable | Meaning |
| --- | --- |
| `MODEL` | HF-style `<org>/<name>`. Must have a config at `configs/model/<MODEL>.json` (auto-downloaded on first run) |
| `HARDWARE` | Free-form label that becomes the folder name under `profiler/perf/`. Pick something meaningful (e.g., `RTXPRO6000`, `H100`, `MI300X`) |

## Sweep shape

The following assignments illustrate optional sweep overrides. Leave them
unset to inherit the CLI defaults.

| Variable | Template example | Meaning |
| --- | --- | --- |
| `TP_DEGREES` | `1,2` | Positive TP degrees. Ordinary category profiling **must include `1`** for TP-stable replication; `plan-skew` and `profile --only-skew` can select a degree independently |
| `MAX_NUM_BATCHED_TOKENS` | `2048` | Profiler internally bumps this by `+MSQ` for shot-bypass headroom; subtracted back when recording meta |
| `MAX_NUM_SEQS` | `256` | Profile with `MSQ > runtime MSQ` so mixed-regime cases at `n = runtime_MSQ` stay feasible |

## MoE parallelism

For supported unquantized DP+EP models, explicit CLI `--dp` selects
[native component profiling](./native-moe-components), with EP=TP*DP and
preserved global expert IDs/top-k. It accepts `profile` or `slice --group moe`;
`--moe-rounds` controls repeated contexts. Target ranks are emulated on one
physical GPU, and communication is measured separately. In `profile.sh`, set
`DP_DEGREES` and optionally `MOE_ROUNDS`. Supported native targets require
DP>=2; omit `MOE_EP_DEGREES` because EP is derived from TP and DP. Do not
combine native acquisition with `ONLY_SKEW`, `PROFILE_MTP` or `FORCE`; use a
fresh `OUT_ROOT` to remeasure an immutable native contract.

Without `--dp`, `--moe-ep-degrees` (`MOE_EP_DEGREES`) selects retained whole-block `moe.csv`
profiles. That path shrinks the expert/top-k checkpoint shape and approximates
rank-local work; it is not the native DP+EP execution contract. Do not combine
the two acquisition modes or infer DP/backend coverage from the legacy EP axis.
See [Output bundle](./output-bundle) for compatibility and fallback behavior.

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

:::tip[Keep query-length coverage and metadata together]
`q > 1` omits pure-prefill shots, and `decode_q_len` is part of the row key.
Each additional value adds a decode-containing grid; it does not double all
previous grids. Prefer a combined acquisition:

```bash
python -m profiler slice <model> --hardware <hw> \
    --tp-refresh 1 --group attention \
    --attention-max-kv 16384 --attention-decode-q-lens 1,5 \
    --out-root outputs/profile_q1_q5
```

Independent GPU acquisitions need separate output roots and the same explicit
KV bound, since the implicit bound depends on each run's maximum query length.
Disjoint row keys alone do not make arbitrary CSV concatenation a complete
bundle: acquisition identity, engine contracts and recorded query-length
coverage must also agree. Do not have two writers update one bundle.
:::


The attention sweep covers `(prefill_chunk, prefill_key, n_decode,
kv_decode, decode_q_len)`. `prefill_key` is the query-weighted causal key
coordinate described in [the output schema](./output-bundle#attentioncsv).
The ordinary decode slice has `decode_q_len = 1`.

| Variable | Template example | Meaning |
| --- | --- | --- |
| `ATTENTION_MAX_KV` | model-derived | Bounds the histories used to construct prefill and decode shots |
| `ATTENTION_CHUNK_FACTOR` | `2.0` | Geometric factor for `prefill_chunk` axis (doubling) |
| `ATTENTION_KV_FACTOR` | `2.0` | Geometric factor for `kv` axes (doubling) |
| `ATTENTION_N_FACTOR` | `1.4142135623730951` | Geometric factor for decode request count |

Smaller factors densify the axis (more shots, slower); larger factors
coarsen it (fewer shots, faster). The KV factor controls both prefill-key and
decode-KV. Costs multiply across the axes; see [runtime planning](#expected-runtime).

## Measurement averaging

```bash
MEASUREMENT_ITERATIONS=3
```

Timed forwards per ordinary shot, averaged per invocation. More repetitions
increase acquisition cost and can reduce variability, but do not guarantee
an error bound. Fixed preparation and analysis costs also affect runtime.

## Skew sweep

After the uniform attention grid, the profiler runs a
heterogeneous-decode sweep that drives the simulator's
FlashAttention-varlen skew correction:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SKIP_SKEW` | unset | Skip new skew measurements. Existing stored calibration is retained; a fresh bundle without skew data uses zero correction |
| `ONLY_SKEW` | unset | Set to `1` to run **only** the skew step, leaving dense / per_seq / attention / moe untouched. Useful for refreshing `skew.csv` |
| `SKEW_N_FACTOR` | `2.0` | `n` (total decodes) axis density. Higher = fewer shots |
| `SKEW_PC_FACTOR` | `2.0` | `pc` (prefill chunk) axis |
| `SKEW_KP_FACTOR` | `2.0` | `kp` (prefill history length) axis |
| `SKEW_KVS_FACTOR` | `2.0` | `kvs` (small-decode kv) axis |

Each heterogeneous case uses three independent contexts with three timed
forwards each by default. Only the actual batch is measured; references come
from `attention.csv`. Density factors above 2.0 reduce the number of cases. See **[Skew & alpha fit](./skew-alpha-fit)** for the
methodology.



## Preview or rebuild skew calibration without a GPU

`python -m profiler plan-skew MODEL --hardware HARDWARE --tp 1` streams the
same acquisition plan using saved engine limits, reporting family/query
coverage, automatic reference-selected support additions, unresolved lookup
cell deficits and remaining cases. The live run rechecks actual capacity. This
preview cannot determine whether a GPU is available. Both this preview and
`profile --only-skew` accept `--tp 2` without a TP1 pass. Existing attention
references and, for the preview, saved engine limits must cover that degree.


```bash
python -m profiler refit-skew meta-llama/Llama-3.1-8B \
    --hardware RTXPRO6000 --variant bf16 --tp 1
```

This uses local model configuration and existing `skew.csv`/`attention.csv`;
it does not start vLLM or acquire GPU measurements. Omit `--tp` for all
measured degrees, or use a comma-separated list. `--out` selects an alternate
profile root. Only derived fit tables and skew-fit metadata are changed.
The normal profiler's metadata writer invokes this same compiler automatically.

The new table is versioned and tied to its attention references. Rebuild after
changing reference data or lookup semantics; serving rejects stale fits.
Enabled unversioned fits are rejected until rebuilt. Disabled bundles remain unchanged.
See [Skew & alpha fit](./skew-alpha-fit) for migration and validation details.

## Resume vs force

| Variable | Default | Meaning |
| --- | --- | --- |
| `FORCE` | unset | Set to `1` to replace the existing acquisitions for the categories and TP degrees selected by this run |

Default is **resume** within the same acquisition method: existing CSVs are preloaded row by row, and
only shots whose identity key isn't already present get fired. This
lets you extend an earlier sweep after changing feasibility (e.g.,
raising `MAX_NUM_SEQS` from 128 to 256) in **minutes** instead of
hours. Resume applies to ordinary categories and skew. Historical files without
the current identity or files measured with a different method require a
separate output root or explicit `--force`; they are not silently mixed with
new timings. See [acquisition identity](./output-bundle#cuda-activity-and-acquisition-identity).
Fully skipped ordinary categories retain their previous measurement timestamp.

Ordinary sweeps checkpoint accumulated rows atomically between completed
shots, including near the start of a new acquisition. A worker failure or
interruption leaves the last checkpoint intact; resume skips its completed
coordinates and remeasures work since that checkpoint. Checkpoint writes retain
earlier rows and acquisition identities rather than replacing them with just
the latest samples. Use a separate output root for a new bundle and finish the
requested sweep before using its table in simulation: a partial checkpoint is
progress, not a completed profile.

Attention resume uses the full-precision query-weighted `prefill_key` saved
in the CSV, not a rounded display coordinate. Multi-prefill shots can produce
fractional keys; compatible rows at those coordinates are skipped just like
integer-key rows. This does not alter stored timings or their acquisition identity.

Skew resume additionally requires a matching measurement implementation
fingerprint, resolved block size, complete kernel set and sufficient per-forward
repetitions. A change to timing attribution requires remeasurement even when
the requested geometry is unchanged. Retained historical rows are not proof
that the current acquisition protocol has completed. Current skew acquisition
uses the same per-call CUDA interval union as the ordinary attention reference
sweep; old kernel-sum rows are not relabelled as interval-union measurements.

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

Use `LOG_LEVEL` or one `VERBOSITY` shortcut, not both:

```bash
LOG_LEVEL="ERROR"          # explicit level; alternatively choose one shortcut below
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
#   G in {dense, per_sequence, attention, linear_attention, moe, mtp}
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
| `--num-hidden-layers` | category/checkpoint-derived minimal stack | `NUM_HIDDEN_LAYERS` |
| `--hf-override KEY=VALUE` | none | `HF_OVERRIDES` (array) |
| `--moe-ep-degrees` | `1` | `MOE_EP_DEGREES` |
| `--dp` | unset (native acquisition opt-in) | `DP_DEGREES` |
| `--moe-rounds` | see `profile --help` | `MOE_ROUNDS` |
| `--profile-mtp` | off | `PROFILE_MTP=1` |
| `--linear-attn-chunk` | config `chunk_size`, else vLLM's `FLA_CHUNK_SIZE` | `LINEAR_ATTN_CHUNK` |
| `--attention-max-kv` | the model's own context | `ATTENTION_MAX_KV` |
| `--attention-decode-q-lens` | `1` | `ATTENTION_DECODE_Q_LENS` |
| `--attention-chunk-factor` | see `profile --help` | `ATTENTION_CHUNK_FACTOR` |
| `--attention-kv-factor` | see `profile --help` | `ATTENTION_KV_FACTOR` |
| `--attention-n-factor` | see `profile --help` | `ATTENTION_N_FACTOR` |
| `--measurement-iterations` | `3` | `MEASUREMENT_ITERATIONS` |
| `--skip-skew` | off | `SKIP_SKEW=1` |
| `--only-skew` | off | `ONLY_SKEW=1` |
| `--skew-n-factor` | `2.0` | `SKEW_N_FACTOR` |
| `--skew-pc-factor` | `2.0` | `SKEW_PC_FACTOR` |
| `--skew-kp-factor` | `2.0` | `SKEW_KP_FACTOR` |
| `--skew-kvs-factor` | `2.0` | `SKEW_KVS_FACTOR` |
| `--skew-samples-per-cell` | `32` | `SKEW_SAMPLES_PER_CELL` |
| `--skew-rounds` | `3` | `SKEW_ROUNDS` |
| `--skew-seed` | `0` | `SKEW_SEED` |
| `--force` | off (resume) | `FORCE=1` |
| `--out-root` | `profiler/perf` | `OUT_ROOT` |
| `--model-config-root` | `configs/model` | `MODEL_CONFIG_ROOT` |
| `--log-level` | `INFO` | `LOG_LEVEL` |
| `--silent` | — | `VERBOSITY="--silent"` |
| `--verbose` | — | `VERBOSITY="--verbose"` |

### `--measurement-iterations` — averaging out clock jitter

Shots with history initialize their assigned attention pages by calling the
installed vLLM `initialize_single_dummy_weight`.
For vLLM 0.28 its defaults are a uniform distribution in `[-0.001, 0.001)` and
seed `1234`, with a local generator per tensor; low-precision rounding can reach
an endpoint. Initialization repeats before each context, outside warmup and CUDA
timing, without modifying unassigned pages. FP8 views are initialized in bounded
slabs, each using that same per-tensor seed, to bound the temporary conversion.
Recurrent caches start from zero; packed payload/scale layouts without a supported
typed view are rejected. This is synthetic state, not the trained model's KV
distribution or a guarantee of representative sparse-indexer selection.

Asynchronous outputs are completed before input buffers and request IDs are
reused. Neither preparation nor CPU time contributes to stored query latency.

For the measured query, decode requests have a completed prompt and separate
query/output tokens; prefill requests retain their unfinished prompt. This
preserves vLLM's native ordering and each backend's own phase selection. Query
length alone is insufficient: representing every request as a fresh prompt can
send a mixed sparse-indexer batch down the all-prefill path. V1 and V2 request
registration are handled separately without changing token values or KV pages.

Each shot runs one discarded warm-up forward, then N timed forwards inside a
single `layerwise_profile` context. A single sample can swing 15-25% on a large
GEMM from DVFS and boost-clock jitter, so the default is 3 and the per-call
figure comes from dividing by the invocation count.

For whole-block MoE shots, warmup and timed calls use the same forced expert
distribution. Native routing kernels still run; only their selected IDs and
weights are replaced. A backend that bypasses the routing hook is rejected.
See the [whole-block table contract](./output-bundle#moecsv-legacy-whole-block-moe-profiles)
for refresh requirements and the distinction from native DP+EP components.

Instrumentation is also a major part of acquisition cost, separately from
dummy KV preparation. Measured on a 4-layer DeepSeek-V3.2, one
shot at the default: 0.1 ms to assemble, 49 ms for the three forwards, 2.1 ms
to convert the tree — and **1,125 ms inside the `layerwise_profile` context**.
Session setup/teardown is only 6.9 ms of that; the rest is per-op instrumentation
at **372 ms per forward against 16 ms outside**, linear in the forwards:

| forwards in one session | ms |
| --- | --- |
| 0 | 6.9 |
| 1 | 372 |
| 3 | 1,125 |
| 6 | 2,310 |

In this measurement, one forward takes about a third of the three-forward
profile context; that does not imply a threefold full-sweep speedup or an error
bound, since preparation, warmup and processing also contribute. Turning off
the profiler's `with_stack` does **not** help in this case, despite being
14x in a raw `key_averages()` path: `layerwise_profile` builds its tree from
`experimental_event_tree()`, which does not pay for it.

:::tip[The layer count is the same axis, and it is free]
Op count sets the wall clock, and the profile tree merges same-class siblings —
so a second layer of a type already present adds **no information**, only its
op count on every shot. Each category is therefore shrunk to the axes it
measures, and the profiler boots one engine per distinct depth. DeepSeek-V3.2
and GLM-5 need 4 layers for their MLP axis (`first_k_dense_replace 3`) and
**1** for attention, since every one of their layers has the same attention:
1,057 → 341 ms per shot, **3.1x**, with no loss of accuracy. Qwen3.8-27B and
MiniMax-M3 vary on the attention axis too, so they gain nothing here.

Fewer layers means a larger `num_cache_tokens`, which the feasibility filters
read, so more shots pass — wider coverage, and a grid that is not row-for-row
comparable with a deeper run's.

Why a category cannot just be shrunk to one layer: not because its entries are
hard to tell apart, but because **at one layer some of them do not exist**. A
1-layer Qwen3.8 instantiates only the gated-DeltaNet block, so `qkv_proj` and
`o_proj` are never built, their rows go missing from `dense.csv`, and the
simulator charges those layers zero. The axes are also a conservative proxy for
what a category needs — DeepSeek's `dense` would in truth serve at one layer,
since `moe` is its own category — and the slack is left in on purpose:
the multidimensional attention sweep is much larger than the dense sweep.
:::

The division is **per parent**: every node divides by its parent node's
invocation count. The top level has no parent node, so `extract_samples` is
given `iterations` as its count — and vLLM 0.28 additionally reports a
top-level node once per forward (and once per sibling module) where 0.19 merged
them into one wrapper, so identical sibling entries are deduped first. Identity
includes the full module representation, time and invocation count: a shared
class name and equal timing do not make differently shaped modules duplicates. Both
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

estimates a lower bound for a bandwidth-limited head when its weights must be
read from device memory. For a full Llama-3.1-8B head, 128256 × 4096 × 2 B ÷
1.8 TB/s is 583 µs; 714 µs corresponds to about 82% bandwidth efficiency.
A 6417 µs measurement is suspicious, not mathematically impossible: a lower
bound cannot prove a slow measurement wrong. Check invocation normalization,
TP-local tensor sizes, cache residency and a controlled measurement before
concluding that attribution is faulty. Embedding reads selected rows, not the
whole vocabulary matrix, so it needs a different traffic estimate.
:::

### `--attention-max-kv` — how far out the KV axes reach

Unset, this is **the model's own context window**, resolved against the live
engine as `max_model_len - max(decode_q_len) - 1`. It is not `max_model_len`
itself: a decode request occupies `kv + q` positions and needs one more to be a
decode rather than the whole window, so passing the context length verbatim
gets the top point filtered and the sweep stops one doubling short — on
DeepSeek-V3.2 that is 131,072 where you asked for 163,840.

Lowering it trades coverage for time. The KV-budget filter prunes the
large-KV × large-decode-count corner, so counts depend on all axis factors,
query lengths, stack depth and the engine's resolved capacity. Estimate runtime
from the actual acquisition plan and measured stage progress, not a fixed
hours-per-model table; see [Expected runtime](#expected-runtime).

:::caution[Cover the context range you intend to simulate]
Dense attention can be approximately linear in KV traffic over a measured
regime, but extrapolation is not guaranteed across kernel or occupancy changes.
For sparse models, selected attention can saturate at the selection cap while
the indexer still scores the full history. The lookup's per-kernel saturation
rules do not establish the indexer's cost outside its measured support.
Profile the intended range and validate any extrapolation separately.
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
over-matching leaves nothing unbound. That one needs the profile tree
itself: boot with `--profile-mtp` at the depth in question and print every
node's class with its ancestor path, filtered to the class you suspect
(`RMSNorm`, say).

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
stdout; use `--log-level` instead for an explicit level.

Ordinary category profiling requires `--tp` to include `1`: TP-stable layers
(layernorms, sampler) are measured once at TP1 and replicated into other
`tp<N>/` folders. Skew-only acquisition and `plan-skew` do not replicate these
categories, so they can select another positive degree alone.

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

It measures only that TP/category and updates `tp<N>/<group>.csv` plus the
relevant `meta.yaml` fields. Compatible rows resume by default; `--force`
replaces the selected category. A whole-block MoE slice can boot a separate
engine for each requested EP degree. It errors out if the
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

This wraps `python -m profiler profile` in a loop over the `JOBS` array.
Unlike the single-model template, it has explicit global and per-job overrides;
inspect its command construction for accepted environment variables. Per-job
flags take precedence over global settings. For example:

```bash
HARDWARE=H100 \
ATTENTION_CHUNK_FACTOR=1.5 \
./profiler/profile-all.sh
```

To change models or their TP/DP settings, edit the `JOBS=( ... )` array at the top
of the script. This file is meant to be copied or tweaked in-place,
not treated as a stable CLI.

## Expected runtime

Estimate acquisition time from the requested plan and its observed rate, not
from the model name or an older run's defaults. Attention combines prefill-token,
prefill-key, decode-request and decode-KV axes, with optional decode query lengths.
The number of geometric intervals over a fixed range scales as
`log(max / start) / log(factor)`. For example, changing an axis factor from 2
to the square root of 2 roughly doubles its intervals. Refining multiple axes
multiplies their counts; it is not a single 1.4x increase for the whole sweep.
Rounding, degenerate-axis deduplication and live feasibility limits determine
the final count emitted by `AttentionCategory.compose_shots`.

Acquisition also includes request/KV preparation, warmup, timed forwards,
CPU profile-tree processing and checkpoint writes. Reducing timed repetitions
does not proportionally remove those fixed costs and changes the measurement
protocol. Low GPU utilization can coexist with active CPU profile processing;
check the worker and saved progress before diagnosing a stall. None of that
CPU preparation or processing time is added to simulated GPU latency.

The per-shot cost depends on the instantiated category stack and batch shape.
A partial stage's rate can change as it reaches longer histories or more
requests. Its ETA excludes later categories, skew rounds, native MoE acquisition
and additional TP configurations. Report those stages separately when estimating
complete-bundle time. Existing CSVs or checkpoints are not evidence that a
requested full sweep has finished.

Choose bounds, factors and repetitions before acquisition, record them with
the bundle, and preserve compatible acquisition identities when resuming.
Coarser grids reduce sampling resolution and may change interpolation error;
denser grids also need end-to-end validation and do not guarantee better accuracy. Do not
silently narrow an agreed measurement plan to meet an earlier runtime estimate.

The Rich-based logger renders per-step progress bars; redirect
stdout with `--silent` for a quieter run.

Those bars need a TTY. Ordinary categories also checkpoint accumulated CSV
rows atomically between completed shots, independently of the display. Long
sweeps log a plain heartbeat,
carrying the count, the rate and an ETA:

```
TP=1  skew: 2560/13476 cases (19.0%), 3.41 case/s, eta 53.4 min
```

Both the per-category sweeps and the skew sweep print it.

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
