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

## Reduced model configurations

The profiler reduces decoder depth separately for each category. When a model
provides an explicit `layer_types` list, a depth of N retains its first N entries
in their original order, including inside a nested `text_config`. A requested
depth that is non-integer, non-positive or longer than that list is rejected.
The source model config is not modified. Explicit `hf_overrides.layer_types`
values are truncated consistently because vLLM reapplies them after config load.
This preserves the selected prefix; it does not certify catalog coverage or
end-to-end accuracy for a new model family.

For TP emulation, installed vLLM resolves the local unsharded configuration
after user overrides. The profiler then shrinks declared TP dimensions while
preserving the original attention head width. Otherwise a derived width can
grow when the head count shrinks, as in Mixtral configurations without an
explicit `head_dim`. Nested text fields follow the same resolution path.
Unsupported specialized width changes and non-divisible dimensions fail before
engine construction; this does not add replicated-KV-head emulation.
Old affected rows must be measured again in a separate output root or with an
explicit category refresh. Changing their metadata is not a correction.

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
4. Sweeps the catalog's **dense / per_sequence / attention / linear_attention /
   moe** shot grids (and the separate MTP pass when requested),
   writing CSVs under `profiler/perf/<HW>/<MODEL>/<variant>/tp<N>/`.
5. (Unless `SKIP_SKEW` is set) Runs the heterogeneous-decode
   skew sweep and fits per-bucket alphas to `skew_fit.csv`.
6. Writes `meta.yaml` summarizing the run.

For each TP degree in `TP_DEGREES`, the profiler emulates that TP
on a single GPU by dividing the model's per-rank shapes via
`hf_overrides`. **You only need one GPU** for supported TP degrees.

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

Each extra query length adds decode-containing shots, so pass only the `1 + N`
values you intend to simulate; the published N for the four modern families
are 3, 4 and 5. The
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

For coupled component layouts, acquire every affected category before simulation.
Qwen's packed GDN core includes the formerly separate norm, so new `dense` and
`linear_attention` data must be acquired together. Compatible interrupted runs
resume whole validated shots; `pending`/`writing` categories remain unusable.
An interrupted `writing` category can resume before its first measured shot,
including a TP2 file holding only validated TP1 replicas; this is not completion.
Changed shared identity plus `--force` invalidates all coupled categories, even
when only one is selected; same-identity refresh preserves valid siblings.
Changed representation/definition requires fresh output. Historical tables remain
readable but need coordinated explicit refresh or new output for new acquisition.
Use one writer per variant root, including across TP/category slices, and do not
modify a bundle while simulation reads it. See the
[layout contract](./output-bundle#runtime-selected-component-layouts).

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

Skew resume additionally requires matching source, effective catalog, full contexts
and installed runtime versions, resolved block size, complete kernel set and
sufficient per-forward repetitions. A change to timing attribution requires remeasurement even when
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
| `DTYPE` | unset | Omit the engine dtype override and let vLLM resolve the checkpoint configuration. Quantization comes from the checkpoint; the profile variant uses its quantization method or `torch_dtype` / `dtype` |
| `KV_CACHE_DTYPE` | unset (`auto` in vLLM) | Optional KV dtype override, such as `fp8`. Match the effective cache dtype to the simulator's model config; this flag does not edit that config |

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
other subcommands, which `profile.sh` does not expose.

```bash
python -m profiler profile  <model> --hardware <hw> [options]
python -m profiler slice    <model> --hardware <hw> --tp-refresh N --group G [options]
#   G in {dense, per_sequence, attention, linear_attention, moe, mtp}
python -m profiler coverage <model> --hardware <hw> [options]
python -m profiler plan-skew <model> --hardware <hw> [options]
python -m profiler refit-skew <model> --hardware <hw> [options]
python -m profiler hardware --hardware <hw> [options]
```

`<model>` is an HF-style `<org>/<name>` resolving to
`configs/model/<org>/<name>.json`, or an explicit path ending in
`.json`. HF-style ids are auto-downloaded from the Hub on first use
(honouring `HF_TOKEN`); explicit paths are never fetched, so a missing
file is an error.

### Common model-command flags

`profile`, `slice`, `coverage` and `plan-skew` register the flags below.
Registration does not mean every command runs every sweep: `coverage` only
checks catalog coverage, and `plan-skew` uses saved `meta.yaml` engine limits
and existing attention references without starting vLLM. Engine-limit flags
do not replace those saved limits in a plan preview.

`slice` additionally requires `--tp-refresh N` and `--group G`; include `N`
in `--tp`. `--dp` is supported only for `profile` or `slice --group moe`.
`--profile-mtp` with `slice` requires `--group mtp`. `--only-skew` selects
the skew-only path of `profile`; it is not a slice category.

Defaults below are effective acquisition defaults where the parser leaves
an override unset. `profile.sh` omits unset or empty optional values, so the
CLI remains their source of truth. Boolean wrapper variables enable their
flags for any non-empty value, including `0`; unset them to disable.

| Flag | Default | `profile.sh` variable |
| --- | --- | --- |
| `--hardware` | **required** | `HARDWARE` |
| `--tp` | `1` | `TP_DEGREES` |
| `--variant` | auto-derived from dtypes | `VARIANT` |
| `--dtype` | unset; vLLM resolves the checkpoint dtype | `DTYPE` |
| `--kv-cache-dtype` | `auto` | `KV_CACHE_DTYPE` |
| `--max-num-batched-tokens` | `2048` | `MAX_NUM_BATCHED_TOKENS` |
| `--max-num-seqs` | `256` | `MAX_NUM_SEQS` |
| `--block-size` | unset; vLLM resolves the backend's block size | `BLOCK_SIZE` |
| `--gpu-memory-utilization` | `0.9` | `GPU_MEMORY_UTILIZATION` |
| `--max-model-len` | from the model config | `MAX_MODEL_LEN` |
| `--num-hidden-layers` | category/checkpoint-derived minimal stack | `NUM_HIDDEN_LAYERS` |
| `--hf-override KEY=VALUE` | none | `HF_OVERRIDES` (array) |
| `--moe-ep-degrees` | `1` | `MOE_EP_DEGREES` |
| `--dp` | unset (native acquisition opt-in) | `DP_DEGREES` |
| `--moe-rounds` | `3` | `MOE_ROUNDS` |
| `--profile-mtp` | off | `PROFILE_MTP=1` |
| `--linear-attn-chunk` | config `chunk_size`, else vLLM's `FLA_CHUNK_SIZE` | `LINEAR_ATTN_CHUNK` |
| `--attention-max-kv` | resolved context minus `max(decode_q_len) + 1` | `ATTENTION_MAX_KV` |
| `--attention-decode-q-lens` | `1` | `ATTENTION_DECODE_Q_LENS` |
| `--attention-chunk-factor` | square root of 2 | `ATTENTION_CHUNK_FACTOR` |
| `--attention-kv-factor` | square root of 2 | `ATTENTION_KV_FACTOR` |
| `--attention-n-factor` | square root of 2 | `ATTENTION_N_FACTOR` |
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

The default `--out-root` and `--model-config-root` paths are resolved from
the repository location; explicit relative paths use the working directory.
`--block-size` is a request to vLLM, not a guarantee of its resolved value;
use the recorded per-TP `engine_resolved` metadata when configuring simulation.

### Separate offline and hardware commands

These commands do **not** accept the common model-command flags:

| Command | Accepted options | Defaults and scope |
| --- | --- | --- |
| `refit-skew <model>` | `--hardware`, `--variant`, `--tp`, `--out`, `--log-level` | Hardware is required; variant comes from the local checkpoint config; TP defaults to all measured degrees. Rebuilds calibration from stored CSVs, without a GPU |
| `hardware` | `--hardware`, `--npus`, `--out`, `--log-level` | Hardware is required; `--npus` defaults to `2`. Acquires device/interconnect facts, without a model argument |

Both use `--out` (not `--out-root`), defaulting to `profiler/perf` relative
to the working directory, and `--log-level INFO`. Log levels are `DEBUG`,
`INFO`, `WARNING`, `ERROR`. See [skew calibration](./skew-alpha-fit) and
[adding hardware](./adding-hardware) for their input contracts.

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
single lightweight PyTorch/Kineto context. The default is 3; the per-call
figure uses the catalog's invocation normalization. Repetition reduces sampling
noise but does not guarantee an error bound across models and hardware.

For whole-block MoE shots, warmup and timed calls use the same forced expert
distribution. Native routing kernels still run; only their selected IDs and
weights are replaced. A backend that bypasses the routing hook is rejected.
See the [whole-block table contract](./output-bundle#moecsv-legacy-whole-block-moe-profiles)
for refresh requirements and the distinction from native DP+EP components.

Collection overhead is separate from KV preparation and device execution.
The collector records CPU ownership scopes and native CUDA activities, with
Python stacks, tensor shapes, memory snapshots and FLOP estimation disabled.
It does not export or rebuild vLLM's Python-call tree. Module calls use indexed
CPU-scope ownership; CPU duration never becomes a stored GPU latency.

Measure throughput on a bounded subset of the intended model, backend and grid
before estimating a complete sweep. First-use compilation, warmup, KV allocation,
CSV checkpoints and the number of executed layers also contribute; a collector
speedup is not a promise of the same full-sweep speedup.

:::tip[The layer count is the same axis, and it is free]
Op count sets the wall clock, and the profile tree merges same-class siblings —
so a second layer of a type already present adds **no information**, only its
op count on every shot. Each category is therefore shrunk to the axes it
measures, and the profiler boots one engine per distinct depth. DeepSeek-V3.2
and GLM-5 need 4 layers for their MLP axis (`first_k_dense_replace 3`) and
**1** for attention, since every one of their layers has the same attention:
fewer executed layers per shot, while preserving the requested architectural axis. Qwen3.8-27B and
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

Ordinary bindings default to **parent calls × catalog occurrences**; whole-block
MoE defaults to its own invocation count. A catalog's explicit `normalization`
can select either rule. Top-level parent-normalized bindings use the timed-forward
count as their parent count. Declared function/method scopes retain these units
without adding descendant parent counts; see the
[callable contract](./adding-model-architecture#declared-callable-boundaries).
The collector retains individual calls before
grouping compatible full parameter-signature paths. Legacy summary deduplication
remains a compatibility rule: class-name equality and equal timings alone must
not collapse differently shaped modules.

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
inside the same native-activity capture context.

**Run `coverage` with it first.** The drafter's kernels report as
unbound until the catalog binds them, with their ancestor paths — which
is how the `mtp:` sections in `profiler/models/` were written:

```bash
python -m profiler coverage <model> --hardware <hw> --profile-mtp
```

Coverage checks unclaimed activities and overlapping or structurally unrelated
claims, not whether every catalog alternative executed. An entry absent from all
forwards may leave no activity gap; MTP acquisition separately requires every
requested MTP entry. Target and drafter roles still need a dynamic-ancestry audit:
boot with `--profile-mtp` and inspect the call paths for the class in question
(`RMSNorm`, say). A coverage percentage alone does not prove the decomposition.

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

For [coupled layouts](./output-bundle#runtime-selected-component-layouts), a slice
can leave sibling categories pending after an acquisition change. It does not
certify those siblings or make the bundle usable before they finish. Use the
same acquisition settings and output root for the remaining category refreshes;
do not run them concurrently.

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
measure nothing: a static module may be bypassed at runtime, and modern models
fuse q-norm, rope and the KV write into a
single kernel with no module wrapper, or write attention as bare Triton kernels
launched straight from the block. The module tree still shows the classes, so
the mistake is invisible in the source — and the symptom is a layer that looks
free rather than a valid measurement. This coverage command uses physical TP=1;
it does not certify another TP degree's fused path or collective placement.

Declared callable boundaries include a `callable_audit` in raw collector/coverage
results: target resolution, unavailable reasons and observed invocation counts.
The runner also warns once for each unavailable-target diagnostic, including when
a timing-only RPC would otherwise discard the audit. An inactive alternative need
not execute, but missing required positive CUDA measurements cannot be replaced by
zero. Review ownership, overlap and the selected component set, not only the
percentage of bound work. A callable declaration by itself is not model/backend
certification; see [declared callable boundaries](./adding-model-architecture#declared-callable-boundaries).

The host coverage report preserves `canonical_invocations`: per-measured-forward
`scope_calls` by module/operator/callable kind, `positive_cuda_scope_calls`,
separate `raw_cuda_activities`, missing canonical entries and the exact
`normalization_groups` divisors. Inspect these against the expected execution
sequence. A CPU-only call is observed but cannot satisfy a positive-CUDA check;
several kernels from one call are not several module calls. Do not infer calls
from a timing divisor or sum group divisors into a total. Warmup and preparation
are excluded. This is an audit, not an automatic proof of model correctness or
a change to measured latencies.

Layout discovery requires positive prefill/decode/mixed CUDA coverage and per-shot
agreement. Its [fixed probes](./output-bundle#runtime-selected-component-layouts)
must fit the engine envelope; they are not adapted to smaller configured limits.
Small probes do not certify unmeasured shapes or deployments; see the
[hybrid trace limits](../examples/model-families/hybrid-linear-attention#packed-core-traces).

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

## Resource safety

Run a long acquisition through `scripts/monitor_run.py` when you need explicit
process-tree memory and time limits. It monitors only its child command and
descendants; it does not reserve GPUs or enforce a container memory limit.
Select limits appropriate to the host and obtain exclusive device access first.

```bash
python3 scripts/monitor_run.py --output outputs/profile-resources.csv \
  --max-rss-gib 48 --min-available-gib 128 --timeout 86400 \
  -- python3 -m profiler profile meta-llama/Llama-3.1-8B --hardware <hw>
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--output` | required | Fresh resource-log CSV path; a JSON completion summary is written alongside it |
| `--max-rss-gib` | `48` | Stop when summed child-process RSS exceeds this limit; shared pages may be counted more than once |
| `--min-available-gib` | `128` | Stop when host-wide available memory falls below this floor |
| `--interval` | `2` | Sampling interval in seconds |
| `--timeout` | `1800` | Command wall-time limit in seconds; increase explicitly for long sweeps |
| `--max-swap-growth-gib` | unset | Optional limit on host swap growth since startup |
| `--gpu-uuid` | unset | Optional telemetry for exactly one physical GPU UUID, not a device-index list |
| `--max-gpu-temp-c` | unset | Temperature stop threshold; required when `--gpu-uuid` is supplied |

Place the child command after `--`. Supply the GPU UUID and temperature limit
together; CPU-only commands need neither.
Telemetry detects a breached limit after sampling, so keep container limits
as the hard backstop. See `python3 scripts/monitor_run.py --help` for syntax.

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
