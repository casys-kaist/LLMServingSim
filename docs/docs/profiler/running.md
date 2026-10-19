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
values are truncated consistently, including when forwarded to vLLM after config load.
This preserves the selected prefix; it does not certify catalog coverage or
end-to-end accuracy for a new model family.

DeepSeek-V3.2 and GLM-5 native configs may populate `layer_types` with the cache
label `deepseek_sparse_attention`. Stack resolution maps that label to the
catalog's full-attention block axis, retaining sparse indexing as a separate
axis. It does not turn sparse attention into dense execution or alter the source
configuration; actual runtime/backend admission is still required.

For a nested `text_config`, declared backbone fields and decoder depth are
written into the temporary config before native loading. Backbone-only keys
are then removed from the forwarded `hf_overrides`: assigning them again to
the wrapper can fail on read-only properties or create conflicting values.
Keys independently declared at the root stay forwarded, including token IDs
shared by the root and text config, so both retain the requested value.
Unrelated root overrides and flat-model behavior are unchanged. Inspect the
source/materialized config and resolved native fields, not just the flat
engine-argument dictionary. Authoritative main-engine text fields are saved per
TP; see [engine metadata](./output-bundle#engine_resolved).

For TP emulation, installed vLLM resolves the local unsharded configuration
after user overrides. The profiler then shrinks declared TP dimensions while
preserving the original attention head width. Otherwise a derived width can
grow when the head count shrinks, as in Llama or Mixtral configurations without
an explicit `head_dim`. Both the sharded candidate and any scalar-width pin are
checked by materializing and reloading the raw configuration through vLLM.
Updating an already constructed config is insufficient: it can retain a derived
default that a new config file would recompute. The preserved width comes from
the user's effective unsharded configuration, not a model-specific constant.
Nested text fields follow the same resolution path; a required scalar `head_dim`
pin is written inside the backbone even when its raw JSON omits the field, and
is not forwarded again as an unrelated wrapper property.
Separately declared `dense_intermediate_size` is reduced by TP as well, so
hybrid dense/MoE stacks do not retain a full-width dense branch during emulation.
Unsupported specialized width changes and non-divisible dimensions fail before
engine construction; this does not add replicated-KV-head emulation.
Old affected rows must be measured again in a separate output root or with an
explicit category refresh, including head-width-dependent projections, attention
and raw skew used for calibration. Source edits alone do not invalidate existing
ordinary/skew resume identities. Changing metadata is not a correction.

## Quick start

From inside the vLLM Docker container at `/workspace`:

```bash
# Edit the variables at the top of profiler/profile.sh, then:
./profiler/profile.sh
```

The script auto-resolves the model architecture from the HF
`config.json`'s `model_type` field, you don't specify it on the
command line. Resolution checks `profiler/models/<model_type>.yaml` and
the catalogs' declared `model_types:` aliases. Multiple files claiming the same
type are rejected even when one filename matches. See
**[Adding a model architecture](./adding-model-architecture)** if it
doesn't.

### Native Kimi profiling

Catalog `native_profile: hybrid-residual` selects Kimi-K3's adapter through
`profile`, category `slice` and `coverage` commands, under a narrower
[native execution contract](./output-bundle#native-kimi-acquisition). For a small
acquisition on a device supporting the admitted native backend, inside the
vLLM 0.31 container:

```bash
python -m profiler slice moonshotai/Kimi-K3 --hardware RTXPRO6000 \
  --tp 1 --dtype bfloat16 --max-model-len 1024 \
  --max-num-batched-tokens 16 --max-num-seqs 2 \
  --attention-max-kv 128 --gpu-memory-utilization 0.68 \
  --measurement-iterations 2 --group dense --tp-refresh 1 --out-root /runs/kimi-profile
```

The small token range is an acquisition example, not a recommended accuracy grid
or a completed GPU-validation claim. The intended scope is bounded TP1 acquisition
and saved-duration traces; it does not establish completed full-grid profiling
or an end-to-end benchmark.
Reserve sufficient device memory for original-width prefix engines and bound host
memory as described in the resource-safety guide. Do not shrink the model geometry
to make a failed native acquisition appear supported. Engine context limits must
also accommodate the native MLA chunk workspace; an extremely short model length
can be incompatible with its hybrid block alignment even for a small token shot.

The original configuration determines category prefix depths and all original-layer
AttnRes roles; geometry is not fixed to one checkpoint's dimensions. The dense
pass includes their isolated native contexts. Ordinary categories and residual
contexts resume together, with partial shots remeasured and malformed/duplicate
partial CSVs or incompatible semantic identities refused. Old model-specific
publications and changed execution semantics require fresh acquisition, not a
metadata conversion. Source hashes and exact runtime versions are provenance;
actual API/backend, shape, dtype and layout checks still apply. Configuration-derived planning is not evidence
of arbitrary device support. Full profiles require `--skip-skew` and a separately
completed [live local MoE acquisition](./native-moe-components#live-local-boundaries)
using the admitted native expert backend. Explicit `--dp 1 --ep 1` is allowed;
ordinary category slices do not certify MoE completion. TP>1, DP/EP greater than one, MTP, shape overrides and
non-BF16 activation/KV paths are not admitted by this contract. Use `coverage`
with the same model/device and sufficient engine limits to inspect actual CUDA
ownership; it does not prove end-to-end serving accuracy.

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
4. Sweeps the catalog's **dense / per_sequence / attention / linear_attention**
   shot grids (and the separate MTP pass when requested),
   writing CSVs under `profiler/perf/<HW>/<MODEL>/<variant>/tp<N>/`.
5. (Unless `SKIP_SKEW` is set) Runs the heterogeneous-decode
   skew sweep and fits per-bucket alphas to `skew_fit.csv`.
6. Acquires native MoE components for MoE models, at DP1 by default.
7. Writes `meta.yaml` summarizing the run.

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

For supported native execution contracts, MoE acquisition defaults to
[native components](./native-moe-components) at DP1, preserving global expert
IDs and top-k. `--dp` selects other target degrees for `profile` or
`slice --group moe`; EP is TP*DP. Optional `--ep` filters those full-EP targets;
omission keeps all, and unmatched degrees fail before engine startup. It does not
infer TP/DP values or enable tensor-sharded experts. A MoE slice matches only
`--tp-refresh`; other categories retain their ordinary TP sweep.
Each rank is emulated on one physical GPU, with communication modeled separately.
In either shell template, `DP_DEGREES="1,2"` selects local and DP+EP targets,
`EP_DEGREES` forwards `--ep`, and `MOE_ROUNDS` controls repeated contexts.
For a mixed dense/MoE `profile-all.sh` job list, put explicit `--dp`/`--ep`
in each MoE job's extra arguments, not global variables that also reach dense jobs.
Modular DP1 omits verified no-op prepare/finalize and DP collectives;
TP>1 still means full EP, not tensor-sharded experts.
The live local adapter requires TP1/DP1/EP1 and retains native quantized or shared
expert execution only after capability and output-reference checks. Its extra
wrapper component includes surrounding shared/latent work. These rows sum unique
CUDA activity durations; they do not reconstruct concurrent elapsed latency.

New component measurements are eager only, without isolated CUDA graph capture.
Expert sampling covers tokens N, active local experts A and local assignments M.
`--moe-grid-factor` (`MOE_GRID_FACTOR`, default `2`) controls geometric spacing
on routing-work axes and uncovered token ranges, with mandatory feasible geometry
corners. Target serving graph points replace interior geometric token points in
fully covered regimes after DP/SP transformation. Eager-only regimes, out-of-cap
rows and DP cancellation retain geometric coverage; A and M keep their own axes.
Sampling and conditioning checks do not by
themselves establish full-model accuracy.

Omitted DP does not affect dense models, coverage, ordinary category slices or
skew-only runs. A full `--profile-mtp` run acquires target components separately
from its drafter pass. Explicit `--dp` or `--ep` cannot be combined with
`--only-skew` or `--profile-mtp`. Native contracts are immutable: use a fresh
`OUT_ROOT` to remeasure instead of `FORCE`.

Whole-block acquisition and `--moe-ep-degrees` have been removed. Historical
`moe.csv` files remain readable until supported components are remeasured.
Unsupported native APIs, layouts and deployments fail explicitly;
no removed whole-block acquisition is substituted. See
[Output bundle](./output-bundle) for compatibility and fallback behavior.
The component engine has a separate internal KV-memory cap, defaulting to 1 GiB.
It keeps the requested context/token/sequence limits and lets vLLM validate native
capacity, failing if the cap is insufficient. This does not change ordinary
profiling cache allocations or target serving graph coordinates; the
[component guide](./native-moe-components) explains its recorded observations.

## Sampling policy

Acquisition remains eager. The sampling plan combines geometric coordinates
with target serving graph coordinates where a category uses graph-padded work.
Matching completed worker capture observations in the bundle take precedence
over derived defaults. Explicit `--cudagraph` options override that selection.
Without a matching observation, the shared serving resolver derives candidates
from logical scheduler limits, query length, target TP/compiler SP, runner and
available compute-capability/backend facts. An acquisition backend's capability
is not evidence that a full serving model completed those captures.

`--grid-factor` (`GRID_FACTOR`, default `2`) controls ordinary geometric
token/request spacing. Attention, skew and native MoE keep their separate factor
options; their defaults are also `2`, except `--attention-n-factor`, which
retains sqrt(2). Graph anchors supplement these grids and do not replace
feasibility checks, KV history axes or expert-work axes.

| Category | Graph-coordinate coverage |
| --- | --- |
| Dense | Forward-token anchors; TP1 measurements for stable entries include anchors from requested TP degrees |
| Per sequence | Geometric actual output rows; target logits and sampler run outside the backbone graph |
| Attention and linear attention | Pure-decode request anchors for FULL-capable modes at the selected query length; pure-prefill anchors only for `FULL`; mixed axes remain geometric |
| Native MoE | Capture-first local/expert token axes after DP/SP mapping, geometric eager fallback; active experts and assignments remain independent |
| MTP | Drafter request anchors only; the table does not represent the complete first-pass query/token layout |
| Skew | Existing independent geometric acquisition axes |

`--cudagraph` accepts a JSON object with `mode`, `capture_sizes`,
`max_capture_size`, `enable_sp` and `runner`. Modes are `NONE`, `PIECEWISE`,
`FULL`, `FULL_DECODE_ONLY` and `FULL_AND_PIECEWISE`; runners are `v1` and `v2`.
Use the effective target mode after attention-backend capability resolution.
An explicit `NONE` requests geometric sampling without graph anchors.

```bash
python -m profiler profile <model> --hardware <hw> \
    --grid-factor 2 \
    --cudagraph '{"mode":"FULL_AND_PIECEWISE","runner":"v2","capture_sizes":[1,2,4,8,16,32,64]}'
```

These options describe target sampling only; they never enable acquisition
graphs or certify graph-specific latency. Both wrappers forward `GRID_FACTOR`
and JSON-valued `CUDAGRAPH` only when explicitly nonempty. Sampling metadata
records observed or derived provenance, resolved coordinates and feasible
plan size separately from measurement completion. See the
[output metadata](./output-bundle#sampling-plans).

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
`decode_q_len` is part of the row key. Each additional value adds a
decode-containing grid; pure prefill is emitted once with `q=1`, including
when the requested decode query lengths omit `1`. Prefer a combined acquisition:

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

The new table records its calibration format and attention references. Rebuild after
changing reference data or lookup semantics; serving rejects stale fits.
Enabled fits without a recognized calibration format are rejected until rebuilt.
The current descriptive name is `runtime-skew-calibration`.
Disabled fits need no migration for simulation. Explicitly running `refit-skew`
with valid raw measurements rebuilds and enables the calibration, even if it was
previously disabled. With no raw measurements, the command fails without changing
the metadata.
See [Skew & alpha fit](./skew-alpha-fit) for migration and validation details.

## Resume vs force

| Variable | Default | Meaning |
| --- | --- | --- |
| `FORCE` | unset | Set to `1` to replace the existing acquisitions for the categories and TP degrees selected by this run |

Native MoE acquisition rejects `--force`, including an ordinary full `profile`
of a MoE checkpoint, which automatically selects native components. Use a fresh
output root for a complete remeasurement, or select an ordinary category with
`slice --group ... --force`. A skew-only refresh can use `--only-skew --force`.

Default is **resume** within the same acquisition method: existing CSVs are preloaded row by row, and
only shots whose identity key isn't already present get fired. This
lets you extend an earlier sweep after changing feasibility (e.g.,
raising `MAX_NUM_SEQS` from 128 to 256) in **minutes** instead of
hours. Resume applies to ordinary categories and skew. Historical files without
the current identity or files measured with a different method require a
separate output root or explicit `--force`; they are not silently mixed with
new timings. See [acquisition identity](./output-bundle#cuda-activity-and-acquisition-identity).
Fully skipped ordinary categories retain their previous measurement timestamp.
New acquisition metadata is kept in the variant-root `meta.yaml`, with per-TP
records under `per_tp[tp]`, rather than metadata files beside the tables.
Supported historical sidecars other than `skew.meta.yaml` remain readable;
skew acquisition and calibration use root metadata only. Resume does not restamp
observations as a new measurement or derive compatibility from a folder name.

TP-stable rows copied from TP1 retain their source acquisition when a higher-TP
category starts or resumes, including ordinary catalogs without component
layouts. These rows do not count as locally measured shots. A replica-only CSV
therefore still measures every planned TP-local shot; `--force` preserves the
copies while replacing local measurements.

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
shots, including near the start of a new acquisition. Resume skips completed
coordinates in a compatible checkpoint and remeasures work since that checkpoint.
An interrupted CSV/metadata replacement must also pass the
[publication checksum check](./output-bundle#cuda-activity-and-acquisition-identity):
matching pending data can resume without rewriting the record; mismatched data
require fresh output or explicit `--force`. In particular, old rows cannot be
reused under new forced-acquisition metadata. Checkpoint writes retain
earlier rows and acquisition identities rather than replacing them with just
the latest samples. Use a separate output root for a new bundle and finish the
requested sweep before using its table in simulation: a partial checkpoint is
progress, not a completed profile.

Attention resume uses the full-precision query-weighted `prefill_key` saved
in the CSV, not a rounded display coordinate. Multi-prefill shots can produce
fractional keys; compatible rows at those coordinates are skipped just like
integer-key rows. This does not alter stored timings or their acquisition identity.

Skew resume additionally requires compatible method semantics, effective catalog,
full contexts, resolved block size, complete kernel set and sufficient per-forward
repetitions. Runtime versions are recorded observations, not exact-match vetoes.
A change to timing attribution requires remeasurement even when
the requested geometry is unchanged. Retained historical rows are not proof
that the current acquisition protocol has completed. Current skew acquisition
uses the same per-call CUDA interval union as the ordinary attention reference
sweep; old kernel-sum rows are not relabelled as interval-union measurements.

## Output naming

| Variable | Default | Meaning |
| --- | --- | --- |
| `VARIANT` | auto-derived | Override the variant folder name |

When `VARIANT` is omitted, the name retains checkpoint precision and separates
explicit activation/KV overrides. Common examples are:

- BF16 checkpoint with unset `DTYPE` or `DTYPE=auto` → `bf16`
- FP8 checkpoint declaring BF16 activations with `DTYPE=bfloat16` → `fp8`
- That FP8 checkpoint with `DTYPE=float16` → `fp8-actfp16`
- `KV_CACHE_DTYPE=fp8` additionally appends `-kvfp8`

Use an explicit `VARIANT` for HF precision-field or any nested `text_config`
override, ambiguous configurations or intentionally distinct acquisitions.
Labels do not enable quantization or select an alternate bundle in the simulator.
See [precision and variant names](#precision-and-variant-names) for the rules
and limitations; old data is not renamed.

## Dtype

| Variable | Default | Meaning |
| --- | --- | --- |
| `DTYPE` | unset | Let vLLM resolve the floating dtype (`auto` is equivalent). Explicit values do not remove checkpoint quantization; a different activation dtype gets a distinct suffix |
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
python -m profiler import-execution <model> --hardware <hw> --bench-dir DIR [options]
python -m profiler hardware --hardware <hw> [options]
```

`<model>` is an HF-style `<org>/<name>` resolving to
`configs/model/<org>/<name>.json`, or an explicit path ending in
`.json`. HF-style ids are auto-downloaded from the Hub on first use
(honouring `HF_TOKEN`); explicit paths are never fetched, so a missing
file is an error.

### Precision and variant names

`--dtype` selects vLLM's floating dtype; it does not enable or remove checkpoint
quantization. Omitting it or using `--dtype auto` keeps the existing checkpoint
variant name. For an FP8 checkpoint declaring BF16 activations:

- `--dtype bfloat16` keeps `fp8`.
- `--dtype float16` uses `fp8-actfp16`, preserving its different activation timings.
- `--kv-cache-dtype fp8` independently appends `-kvfp8`.

Explicit FP32 gets an activation suffix on quantized checkpoints, even when the
config declares FP32, because vLLM's automatic dtype can downcast it. Missing
activation metadata is not assumed to match an explicit dtype. `fp8` is a
quantization method, not a supported vLLM 0.31 `--dtype` value.

Use an explicit `--variant` for HF overrides of `dtype`, `torch_dtype`,
`quantization_config` or any `text_config` (including nested shape overrides).
That label is also required for conflicting dtype aliases or nested-only
quantization with explicit `--dtype`; naming does not guess config-class behavior.
Other top-level shape overrides retain their existing behavior.

The simulator still selects the checkpoint-default variant automatically;
alternate `-act...` measurements are not automatically selected. This naming
change does not migrate existing files. Acquisition identities include profiler
configuration code: acquire fresh data after an identity changes, rather than
renaming or restamping old rows.

Auto names are compatibility labels, not a complete resolved-precision record.
For an unquantized FP32 checkpoint, the legacy default name remains `fp32` even
when vLLM automatically downcasts; use a separate `--variant` for explicit FP32
measurements. Nested-only quantization and automatically promoted KV dtypes also
need a check of the recorded configuration and consumer-selected path. Do not
assume every config-specific default resolves to a matching profiler/simulator
folder solely from its label.

### Common model-command flags

`profile`, `slice`, `coverage` and `plan-skew` register the flags below.
Registration does not mean every command runs every sweep: `coverage` only
checks catalog coverage, and `plan-skew` uses saved `meta.yaml` engine limits
and existing attention references without starting vLLM. Engine-limit flags
do not replace those saved limits in a plan preview.

`slice` additionally requires `--tp-refresh N` and `--group G`; include `N`
in `--tp`. `--dp` and `--ep` are supported only for `profile` or `slice --group moe`.
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
| `--variant` | checkpoint label plus explicit activation/KV suffixes; see naming limits | `VARIANT` |
| `--dtype` | unset; vLLM resolves the checkpoint dtype | `DTYPE` |
| `--kv-cache-dtype` | `auto` | `KV_CACHE_DTYPE` |
| `--max-num-batched-tokens` | `2048` | `MAX_NUM_BATCHED_TOKENS` |
| `--max-num-seqs` | `256` | `MAX_NUM_SEQS` |
| `--block-size` | unset; vLLM resolves the backend's block size | `BLOCK_SIZE` |
| `--gpu-memory-utilization` | `0.9` | `GPU_MEMORY_UTILIZATION` |
| `--max-model-len` | from the model config | `MAX_MODEL_LEN` |
| `--num-hidden-layers` | category/checkpoint-derived minimal stack | `NUM_HIDDEN_LAYERS` |
| `--hf-override KEY=VALUE` | none | `HF_OVERRIDES` (array) |
| `--dp` | automatic DP1 for MoE acquisition | `DP_DEGREES` |
| `--ep` | all full-EP targets (TP*DP); optional comma-separated filter | `EP_DEGREES` |
| `--moe-rounds` | `3` | `MOE_ROUNDS` |
| `--moe-grid-factor` | `2.0` | `MOE_GRID_FACTOR` |
| `--grid-factor` | `2.0` | `GRID_FACTOR` |
| `--cudagraph` | matching observed policy, else derived candidates | `CUDAGRAPH` (JSON object) |
| `--profile-mtp` | off | `PROFILE_MTP=1` |
| `--linear-attn-chunk` | config `chunk_size`, else vLLM's `FLA_CHUNK_SIZE` | `LINEAR_ATTN_CHUNK` |
| `--attention-max-kv` | resolved context minus `max(decode_q_len) + 1` | `ATTENTION_MAX_KV` |
| `--attention-decode-q-lens` | `1` | `ATTENTION_DECODE_Q_LENS` |
| `--attention-chunk-factor` | `2.0` | `ATTENTION_CHUNK_FACTOR` |
| `--attention-kv-factor` | `2.0` | `ATTENTION_KV_FACTOR` |
| `--attention-n-factor` | square root of 2 | `ATTENTION_N_FACTOR` |
| `--measurement-iterations` | `3` | `MEASUREMENT_ITERATIONS` |
| `--skip-skew` | off | `SKIP_SKEW=1` |
| `--only-skew` | off | `ONLY_SKEW=1` |
| `--skew-n-factor` | `2.0` | `SKEW_N_FACTOR` |
| `--skew-pc-factor` | `2.0` | `SKEW_PC_FACTOR` |
| `--skew-kp-factor` | `2.0` | `SKEW_KP_FACTOR` |
| `--skew-kvs-factor` | `2.0` | `SKEW_KVS_FACTOR` |
| `--skew-samples-per-cell` | `32`; minimum `8` | `SKEW_SAMPLES_PER_CELL` |
| `--skew-rounds` | `3`; positive integer | `SKEW_ROUNDS` |
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
| `import-execution <model>` | `--hardware`, `--bench-dir`, `--variant`, `--out`, `--log-level` | Hardware and benchmark directory are required. Requires an HF-style ID with an existing local config and profile bundle. Saves a compatible completed worker graph observation without GPU work or timing-table changes |
| `refit-skew <model>` | `--hardware`, `--variant`, `--tp`, `--out`, `--log-level` | Hardware is required; variant comes from the local checkpoint config; TP defaults to all measured degrees. Rebuilds calibration from stored CSVs, without a GPU |
| `hardware` | `--hardware`, `--npus`, `--out`, `--log-level` | Hardware is required; `--npus` defaults to `2`. Acquires device/interconnect facts, without a model argument |

These commands use `--out` (not `--out-root`), defaulting to `profiler/perf` relative
to the working directory, and `--log-level INFO`. Log levels are `DEBUG`,
`INFO`, `WARNING`, `ERROR`. See [skew calibration](./skew-alpha-fit) and
[adding hardware](./adding-hardware), and
[serving observations](./output-bundle#serving_execution) for their input contracts.

### `--measurement-iterations` — averaging out clock jitter

Shots with history initialize their assigned attention pages by calling the
installed vLLM `initialize_single_dummy_weight`.
In vLLM 0.28 and 0.31 its defaults are a uniform distribution in `[-0.001, 0.001)` and
seed `1234`, with a local generator per tensor; low-precision rounding can reach
an endpoint. Initialization repeats before each context, outside warmup and CUDA
timing, without modifying unassigned pages. FP8 views are initialized in bounded
slabs, each using that same per-tensor seed, to bound the temporary conversion.
Floating MLA indexer caches follow this typed path, including three-dimensional
views that omit a singleton head axis when their dtype and shape match the spec.
These views must pass a non-overlap stride check; padding and axis permutations
are allowed when logical elements occupy separate storage.
Supported packed MLA/indexer layouts initialize floating payloads through the
same helper and then write native-format scales and bytes; the initializer is
never applied indiscriminately to integer storage. This preparation uses bounded
source chunks, preserves unassigned pages and block padding, and adds no measured
latency. See [packed attention history](./output-bundle#packed-attention-history)
for exact layouts and exclusions. Recurrent caches start from zero; unknown packed
layouts remain rejected. This is synthetic state, not the trained model's KV
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

Native MoE components use their separate eager acquisition and conditioning
protocol. Expert work is measured at feasible token/active-expert/assignment
coordinates; no isolated CUDA graph is captured. See the
[native component contract](./native-moe-components).

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
since native MoE components are measured separately — and the slack is left in on purpose:
the multidimensional attention sweep is much larger than the dense sweep.
:::

Ordinary bindings default to **parent calls × catalog occurrences**; historical
whole-block MoE measurements used the block's own invocation count. A catalog's explicit `normalization`
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

For new MiniMax-M3 attention measurements, the inspected native backend also
owns the [per-TP lookup contract](./output-bundle.md#native-attention-lookup-contract).
The reviewed vLLM 0.31 Triton BF16 path uses the configured selected-block count
with 128-token blocks; initial/local blocks are included. The shipped
configuration therefore gives a 2048 sparse-prefill cap.
Record the full raw context range for the indexer and diagnostics; do not clip
the acquisition grid to this cap. Historical unmarked profiles retain 2176.
Use a fresh variant root or explicitly remeasure incompatible attention rows;
changing metadata cannot convert them. Native contract checks are separate
from context coverage, capacity admission and end-to-end accuracy validation.

Native DeepSeek-V3.2's combined `dsa_attention_tail` includes indexing work and
keeps the full-context axis; do not cap its grid at `index_topk`. The
historical representation requires TP1, eager BF16 activations with the original
FP8 weight configuration, and no MTP. New ordinary publication completes dense,
head and attention categories together; native MoE components publish independently
under their local deployment contract. Historical records that include whole-block
MoE retain their original checks. A short attention-only prefix cannot certify
the head or native MoE acquisition. See [native DSA acquisition](./output-bundle.md#native-dsa-acquisition)
for live mode checks and the explicit duration-only consumer boundary.

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

The `mtp` category has one request-count axis for a single drafter pass.
Its shot count follows the sequence/token limits, geometric factor, target
drafter graph anchors and cache feasibility. Measure a bounded sweep to estimate
runtime for the intended model and device. Refresh just that category with
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

Ordinary slices measure only that TP/category and update `tp<N>/<group>.csv`
plus the relevant `meta.yaml` fields. Compatible rows resume by default;
`--force` replaces the selected ordinary category. A MoE slice instead writes
native deployment component tables and their root metadata, defaults to DP1,
and requires a fresh output root for remeasurement. It errors when the checkpoint
or native adapter does not support the requested acquisition; a dense checkpoint
cannot supply a MoE slice.

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
Coverage and component-layout discovery require `max_num_seqs >= 5`,
`max_model_len >= 258` and an acquisition query-token buffer of at least `68`
(the logical token budget plus `max_num_seqs`). Every fixed probe is checked
before the first worker call; an undersized configuration is refused without
resizing or skipping probes. Ordinary synthetic shots also check request count,
query-token total and per-request history plus query plus one output before
page allocation or GPU execution.
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
    ├── moe_components/ (native MoE deployment tables)
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
