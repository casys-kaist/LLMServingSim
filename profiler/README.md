# profiler

## Overview

vLLM-based layerwise profiler for LLMServingSim. Drives a real vLLM
engine with synthetic batches and records per-layer CUDA latency.
Output CSVs and bundle metadata feed the simulator's trace generator.

Start with [Quick start](#quick-start) to run a sweep, or use
[Everything you can set](#everything-you-can-set) as the CLI index.
[Measurement behavior](#measurement-behavior), [Execution scope](#execution-scope)
and [Output schema](#output-schema) explain how timings are collected and consumed.

## Layout

Selected entry points and shared helpers; this is not an exhaustive file list.

```text
profiler/                          Python package: python -m profiler
├── __init__.py                    package marker + _typeshed shim for vLLM
├── __main__.py                    model, coverage, skew and hardware CLI dispatch
├── core/                          orchestration and shared profiling contracts
│   ├── runner.py                  full sweeps, slices and catalog coverage
│   ├── config.py                  architecture, ProfileArgs and engine defaults
│   ├── engine.py                  vLLM lifecycle and resolved engine limits
│   ├── categories.py              per-category shot grids and measurement
│   ├── sampling.py                target graph anchors and geometric sampling
│   ├── attention_shape.py         shared attention geometry and feasibility
│   ├── measurement.py             acquisition identity and timing protocols
│   ├── bundle_metadata.py         locked updates to the variant-root metadata document
│   ├── metadata_names.py          shared descriptive metadata names
│   ├── execution_metadata.py      import/match observed serving graph policies
│   ├── component_layouts.py       runtime alternatives and coupled acquisition state
│   ├── context_tables.py          category/structural-context publication and receipts
│   ├── context_lookup.py          strict category loading and exact-context interpolation
│   ├── callable_profile.py        catalog-selected native and isolated-call orchestration
│   ├── hardware.py                device facts and NCCL link calibration
│   ├── moe_profile.py             native local/DP+EP component acquisition
│   ├── moe_deployment.py          TP/DP/EP ownership and token domains
│   ├── moe_geometry.py            global expert placement and routing geometry
│   ├── moe_conditioning.py        matched-work expert measurement controls
│   ├── skew.py                    heterogeneous-decode acquisition
│   ├── skew_calibration.py        reference-aligned fit and supported-N lookup
│   ├── skew_plan.py               dynamic distribution coverage and feasibility
│   ├── skew_support.py            reference-only completion of lookup support
│   ├── writer.py                  CSV and metadata persistence
│   ├── attention_lookup.py        per-TP native attention semantics and table seals
│   ├── stack.py                   shared checkpoint-derived layer composition
│   ├── catalog_path.py            shared model-type and catalog resolution
│   ├── logger.py                  Rich-based logging and progress
│   └── hooks/                     vLLM-internal API integrations (selected files)
│       ├── extension.py           worker extension and measurement dispatch
│       ├── batch.py               synthetic SchedulerOutput construction
│       ├── history.py             query state and asynchronous output completion
│       ├── dummy_cache.py         assigned-page dummy KV initialization
│       ├── timings.py             profile-tree extraction and coverage
│       ├── cuda_timing.py         per-call CUDA interval accounting
│       ├── activity_collector.py  scoped native CUDA collection and coverage
│       ├── runner_capture.py      common warmup validation and capture
│       ├── capture_scope_contract.py full-architecture ownership contexts
│       ├── method_scopes.py       direct-method ancestry without extra execution units
│       ├── operator_scopes.py     exact native operator launch ownership
│       ├── callable_scopes.py     declared function/method ownership and restoration
│       ├── recurrent_metadata.py  recurrent query/history and state-page checks
│       ├── attention_lookup.py    inspect selected native sparse backend and storage
│       ├── runtime_admission.py   explicit native runtime/configuration admission
│       ├── tp_shape.py            native config and TP head-width preservation
│       ├── cache_pages.py         shared physical KV page allocation
│       ├── moe_live_components.py native local gate/expert/wrapper boundaries
│       ├── moe_live_region.py     processed native expert state and bank cloning
│       └── moe_hook.py            native MoE runner discovery
├── models/                        architecture catalogs and model-type aliases
│   ├── llama.yaml
│   ├── qwen3.yaml                 Qwen3 dense and MoE
│   ├── qwen3_5.yaml               Qwen3.5-family hybrid attention
│   ├── deepseek_v32.yaml          DeepSeek-V3.2 and GLM-5
│   ├── minimax_m3_vl.yaml         MiniMax-M3
│   ├── mixtral.yaml
│   └── phimoe.yaml
├── power/                         nvidia-smi and IPMI power-logging helpers
├── perf/                          hardware/model/variant profile bundles
├── v0/                            archived profiler, not current simulator inputs
├── profile.sh                     editable single-model template
└── profile-all.sh                 explicit multi-model campaign template
```

The simulator also imports the shared stack, catalog, attention, skew and MoE
contracts; these paths must remain usable without vLLM or GPU dependencies.
Changes to those helpers can affect simulator results. Environment launchers
and build entry points live separately under [scripts/](../scripts/README.md).

## Everything you can set

Use `python -m profiler --help` for the full subcommand list. The model
profiling flags below are available through `profile.sh`: unset optional
variables inherit CLI defaults rather than maintaining a second default set.
Most names are the flag in capitals with underscores; TP/DP/EP use `TP_DEGREES` /
`DP_DEGREES` / `EP_DEGREES`, repeated overrides use `HF_OVERRIDES`, and logging uses
`LOG_LEVEL` or `VERBOSITY`. Hardware characterization runs separately as
`python -m profiler hardware --hardware <hw> --npus 2`; it defaults to INFO
logging and accepts an explicit `--log-level` override.
Hardware characterization saves a shared Ring fit and optional per-collective
bandwidth fits at its fixed latency, using primitive NCCL measurements rather
than model benchmarks. See the [calibration contract](https://llmservingsim.ai/docs/profiler/adding-hardware#calibration-contract)
for units, residuals and the limits of grouped or ragged communication.
**[Profiler → Running](https://llmservingsim.ai/docs/profiler/running)**
carries the semantics; this is the index, so a flag missing from one list is
visible against the other.

```text
python -m profiler profile   <model> --hardware <hw> [flags]   full sweep
python -m profiler slice     <model> --hardware <hw> --tp-refresh N --group G
python -m profiler coverage  <model> --hardware <hw>            catalog check
python -m profiler plan-skew <model> --hardware <hw> [flags]    saved-limit preview
python -m profiler refit-skew <model> --hardware <hw> [flags]    offline calibration
python -m profiler hardware  --hardware <hw> [flags]            device/link measurement
python -m profiler import-execution <model> --hardware <hw> --bench-dir DIR
```

`import-execution` is CPU-only and accepts `--variant`, `--out` (default
`profiler/perf`) and `--log-level`. It imports a compatible completed worker
observation from a new benchmark's `meta.json` into an existing bundle. It does
not run profiling, replace CSVs or certify eager timings as graph timings.

| Group | Flags |
|-------|-------|
| **required** | `--hardware` |
| **sharding** | `--tp` (must include 1 except for `plan-skew` or `profile --only-skew`) |
| **precision / naming** | `--dtype`, `--kv-cache-dtype`, `--variant` |
| **engine limits** | `--max-num-batched-tokens`, `--max-num-seqs`, `--block-size`, `--gpu-memory-utilization`, `--max-model-len` |
| **model shape** | `--num-hidden-layers`, `--hf-override KEY=VALUE` (repeatable) |
| **MoE parallelism** | Native DP1 components by default; `--dp` selects target degrees with EP=TP*DP; optional `--ep` filters those targets; `--moe-rounds` controls repetitions and `--moe-grid-factor` (default `2`) spaces all three expert axes |
| **drafter (MTP)** | `--profile-mtp` (a flag — the engine boots at N=1 so the CSV holds one pass) |
| **sampling policy** | `--grid-factor` (default `2`), `--cudagraph` (target serving JSON; acquisition stays eager) |
| **attention grid** | `--attention-max-kv`, `--attention-chunk-factor`, `--attention-kv-factor`, `--attention-n-factor`, `--attention-decode-q-lens` |
| **linear attention** | `--linear-attn-chunk` |
| **measurement** | `--measurement-iterations` |
| **skew** | `--skip-skew`, `--only-skew`, `--skew-n-factor`, `--skew-pc-factor`, `--skew-kp-factor`, `--skew-kvs-factor`, `--skew-samples-per-cell`, `--skew-rounds`, `--skew-seed` |
| **resume** | `--force` (default is resume) |
| **paths** | `--out-root`, `--model-config-root` (`OUT_ROOT`, `MODEL_CONFIG_ROOT`) |
| **verbosity** | `--log-level` (`LOG_LEVEL`), `--silent`, `--verbose` (`VERBOSITY`; choose one logging setting) |
| **slice only** | `--tp-refresh`, `--group {dense,per_sequence,attention,linear_attention,moe,mtp}`. `--tp-refresh N` needs `N` to be in `--tp` too |

Variant names retain checkpoint quantization: an FP8/BF16 checkpoint stays
`fp8` with `--dtype bfloat16`, while `--dtype float16` uses `fp8-actfp16`.
Omitted dtype and `auto` share the default name. HF precision or any nested
`text_config` override needs an explicit `--variant`; the simulator still
selects only its checkpoint-default variant. See
[precision and naming](https://llmservingsim.ai/docs/profiler/running#precision-and-variant-names).
Changed acquisition identities require fresh measurements, not renamed rows.

The table covers `profile`, `slice`, `coverage` and `plan-skew`; not every
accepted option affects every command. `plan-skew` reads saved engine limits
and does not boot an engine. `--dp` and `--ep` are valid only for `profile` or
`slice --group moe`; `--profile-mtp` with `slice` requires `--group mtp`.

`refit-skew` instead accepts `--hardware`, `--variant`, `--tp`, `--out` and
`--log-level`; omitting `--tp` refits every measured degree. `hardware` accepts
`--hardware`, `--npus` (default `2`), `--out` and `--log-level`.
Both use `--out` (default `profiler/perf`, relative to the working directory),
not `--out-root`. Common-command path defaults are rooted at the repository.
Use `--help` on any subcommand for its accepted options.

Native MoE components use one physical GPU to emulate each deployment rank,
preserving global expert IDs and top-k. MoE acquisition defaults to DP1;
`DP_DEGREES="1,2"` adds local and DP+EP targets in the shell templates.
`EP_DEGREES` (`--ep`) optionally filters their TP*DP values; omission keeps all.
It does not infer TP/DP settings or enable tensor-sharded experts. Every explicit
EP degree must match a target, using only `--tp-refresh` for a MoE slice.
Other profiling categories keep the session's TP sweep unchanged.
Contracts are immutable and resumable: use a new output root to remeasure,
rather than `--force`. Expert forward counts can exceed
`--measurement-iterations` to cover matched weight cycles. `MOE_ROUNDS` and
`MOE_GRID_FACTOR` control repetitions and grid spacing. Dense models, coverage,
ordinary category slices and skew-only acquisition do not acquire components.
Modular DP1 measures gate/routing and experts, verifying identity preparation and
aliased no-op finalization instead of adding a copy. The separately admitted
[live local path](../docs/docs/profiler/native-moe-components.md#live-local-boundaries)
retains native expert prepare/finalize and adds measured wrapper work.
TP>1 means full EP, not tensor-sharded
experts. The native backend is selected independently for each target.
The deferred-finalization guard reads vLLM's effective boolean property, not
its request flag or method; unknown capability APIs are rejected. See the
[native MoE guide](https://llmservingsim.ai/docs/profiler/native-moe-components)
for supported backends, CUDA attribution, quality checks and runtime fallback.
The shipped RTXPRO6000 Qwen3-30B component index covers TP1/DP2/EP2. Its
contract states measured bounds; other deployments still require their own data.
Only the runtime subset is bundled, so use a separate output root for acquisition.

The main contributors to acquisition time, in rough order of effect:

1. `--measurement-iterations` — controls timed forwards, not preparation,
   warmup, first-use compilation or processing. Estimate throughput with the
   current collector on a bounded subset of the intended configuration.
   A repetition count does not guarantee a measurement error bound. Ordinary
   bindings default to parent calls times catalog occurrences; historical
   whole-block MoE measurements used the block's own invocation count. A catalog's explicit `normalization`
   selects either rule. Top-level parent-normalized bindings use the timed-forward
   count, retaining full parameter signatures when grouping compatible calls.

2. **Layer count, per category** — not a flag; resolved from the checkpoint.
   The same per-forward cost means op count sets the wall clock, and the
   profile tree merges same-class siblings, so a second layer of a type
   already present adds no information. Each category is shrunk to the axes it
   measures (`Category.stack_axes`), and `run_full` boots one engine per
   distinct depth. DeepSeek-V3.2 and GLM-5 need 4 layers for their MLP axis and
   **1** for attention. Qwen3.8-27B and MiniMax-M3 vary on the attention axis too, so
   they gain nothing. Fewer layers also means a larger `num_cache_tokens`, so
   more shots clear the feasibility filters -- wider coverage, and a grid not
   row-for-row comparable with a deeper run's.

3. `--attention-chunk-factor` / `--attention-kv-factor` — adjust geometric
   spacing, defaulting to doubling. Attention request counts retain sqrt(2).
   Target graph anchors remain in applicable pure-regime shots.

4. `--attention-max-kv` — defaults to the model's own context
   (`max_model_len - max(decode_q_len) - 1`, not `max_model_len` itself: a
   decode occupies `kv + q` positions and needs one more to be a decode, so
   the context length verbatim gets the top point filtered). A lower cap reduces
   coverage and acquisition cost, with the reduction depending on the other
   axes and live KV capacity. Dense kernels may scale approximately linearly
   over a measured regime, but extrapolation beyond it is not guaranteed.
   Sparse attention and its indexer also have different KV scaling.

5. `--attention-decode-q-lens` (`1`) — each extra value adds a separate
   decode-containing grid, not another doubling of the accumulated sweep.
   Pure prefill is emitted once at `q=1`, even if only larger decode query
   lengths are requested; each decode query length has its own feasibility. Prefer one
   invocation with all required values so row identity and metadata remain
   consistent; use separate output roots for independent acquisitions.

6. `--skip-skew` — drops the whole second sweep.

7. `--num-hidden-layers` — normally auto-resolved and best left alone, but it
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

The launcher uses `vllm/vllm-openai:v0.31.0` (CUDA 13), installs the shared
`datasets`, `matplotlib` and `pandas` dependencies plus the pinned NCCL package,
and applies the repository's vLLM patches. No manual installation is needed
inside that container. Existing bundles retain their recorded acquisition
versions; the environment upgrade does not certify every catalog/backend.

The container mounts the **LLMServingSim repo root** as `/workspace`
and starts there. Export `HF_TOKEN` in the host shell before launching if
gated configs are needed; the launcher forwards it. Do not put credentials
in the script. On a shared host, set `VLLM_GPUS` to only the allocated devices;
see the [launcher instructions](../scripts/README.md).

### 2. Edit `profiler/profile.sh` for your run

The script is a template — open it, change `MODEL` and `HARDWARE`, and
optionally tweak the rest. Every knob below maps to a CLI flag on
`python -m profiler profile`; shell variables left unset stay at the
profiler's built-in defaults.
The assignments below are opt-in examples, not active template defaults.
Nonempty boolean variables enable their flags, including the string `0`;
leave them unset or empty to disable them.

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

#### Sampling policy

Ordinary token/request grids use `--grid-factor` (`GRID_FACTOR`, default `2`).
Attention, skew and native MoE retain their own factor options; only attention
request counts default to sqrt(2). Eager measurements include target serving
graph coordinates where the category uses them. Matching completed worker
captures are preferred unless `--cudagraph` (`CUDAGRAPH`) explicitly overrides
the target. Otherwise candidates are derived from logical scheduler limits,
query length, TP/compiler SP, runner and available hardware/backend facts.
The recorded source distinguishes observations from assumptions.

`CUDAGRAPH` is a JSON object with `mode`, `capture_sizes`, `max_capture_size`,
`enable_sp` and `runner` keys. It configures target sampling, not acquisition
engine graph execution. Both shell wrappers forward it and `GRID_FACTOR` only
when explicitly nonempty. For example:

```bash
GRID_FACTOR=2.0 CUDAGRAPH='{"mode":"FULL_AND_PIECEWISE","runner":"v2"}' ./profiler/profile.sh
```

Dense token axes include forward graph anchors alongside geometric points.
Linear attention preserves chunk-boundary points and adds applicable pure-regime
anchors. Target logits and sampler rows remain geometric
because they execute outside the backbone graph. Attention adds pure-decode
request anchors only for FULL-capable modes at the selected query length and
pure-prefill token anchors only for `FULL`; mixed shots keep their geometric
axes. MTP adds drafter request anchors, but its table does not represent the
complete first-pass query/token layout. Feasibility filters still apply to every
shot. See the [sampling guide](../docs/docs/profiler/running.md#sampling-policy).

#### Attention grid

```bash
ATTENTION_MAX_KV=""                 # key-history bound; empty = the model's own context
ATTENTION_CHUNK_FACTOR=2.0          # geometric factor for prefill_chunk axis (doubling)
ATTENTION_KV_FACTOR=2.0             # geometric factor for kv axes (doubling)
ATTENTION_N_FACTOR=1.4142135623730951  # geometric factor for decode request count
```

These are opt-in overrides; the unedited template leaves them unset and
inherits `python -m profiler profile` defaults. Check its `--help` before a run.
Smaller factors densify axes, and their costs multiply across the attention
grid; the KV factor controls both prefill-key and decode-KV axes. See the
[cost-planning guide](../docs/docs/profiler/running.md#expected-runtime).

**If the run dies with CUDA OOM, this is the first knob.** Left empty,
`ATTENTION_MAX_KV` is the model's own context window, so a long-context
checkpoint sweeps out to 131k or beyond and the largest decode shots ask the
engine for `n_decode * kv_decode` tokens of KV in one go. Setting it bounds the
biggest allocation the sweep ever makes, and can reduce its shot count. The cost is
long-context coverage: the simulator extrapolates past the last profiled kv,
whose accuracy must be checked even for a dense kernel. Sparse indexers may
continue scaling after the selected-attention cost saturates; see the
`--attention-max-kv` section of the profiling guide.

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
MEASUREMENT_ITERATIONS=3            # timed forwards per shot, averaged per invocation;
                                    # more repeats do not guarantee an error bound.
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
for denser sampling. Actual per-TP acquisition plans are saved in the
variant-root `meta.yaml::per_tp[tp].skew` record.


#### Measuring the machine: `profiler hardware`

Separate from any model sweep, because it characterises the hardware rather
than a model:

```bash
python -m profiler hardware --hardware RTXPRO6000 --npus 2
```

writes `profiler/perf/RTXPRO6000/hardware.yaml` — **one file per hardware
folder**, shared by every model bundle under it. It carries the card's spec
(queried from the device), NCCL AllReduce/AllGather/ReduceScatter sweeps, and a `defaults` block
that cluster configs inherit `link_bw` / `link_latency` /
`npu_mem.mem_size|mem_bw|mem_latency` from when they omit them. Each default
records its `source` — `measured`, `spec`, or `assumed` — and a simulation logs
it, so a run says whether its link numbers came from a benchmark or from
nobody.

The shared BW/latency fit uses the analytical backend's single-chunk,
one-hop Ring costs at the measured rank count, including local reductions.
Network bandwidth is **GiB/s**; local memory bandwidth is decimal **GB/s**.
Per-size and per-collective residuals describe where one pair cannot match
NCCL. Additional `collective_fits` derive individual bandwidths at the shared
latency and populate optional `defaults.collective_links` entries. The shipped
RTXPRO6000 bundle supplies these defaults to both Qwen TP2 and DP2/EP2 examples
when their cluster configs omit link overrides. Explicit
common-link settings suppress automatic inheritance of those operation curves;
an explicit operation map replaces them. This measures individual collectives,
not grouped MoE dispatches, and
does not fit coefficients against model benchmarks. See the
[hardware guide](../docs/docs/profiler/adding-hardware.md#calibration-contract)
for the formulas, assumptions and saved metadata. Updating the code alone
does not rewrite existing hardware measurements or defaults.

The refreshed RTXPRO6000 Llama-3.1-8B, Qwen3-32B and Qwen3-30B bundles
contain vLLM 0.31.0 acquisition data at TP1 and TP2, including the TP1 sources
for stable entries. Qwen3-30B also includes native TP1/DP1/EP1 and TP1/DP2/EP2
component data. Generated `meta.yaml` preserves emitted values except portable
identifiers for environment-dependent paths; temporary sharded-model identifiers
retain their TP-shape semantics. Per-TP records describe the measurements and
matching runtime tables without pruning acquisition records or changing settings.
The retained Qwen3-30B whole-block `moe.csv` remains a historical fallback.
Eager CUDA activity measurements are separate from target serving graph settings.
Headline accuracy covers Llama TP1, Qwen3-32B TP2 and Qwen3-30B TP1/DP2/EP2;
the other acquisition scopes require separate end-to-end validation. See the
[recorded comparisons](../docs/docs/validation.md).

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
FORCE=1                             # replace acquisitions for the selected categories
                                    # and TP degrees; use a new output root to preserve them.
```

Default is **resume** within the same acquisition method: existing CSVs are preloaded row by row, and only
shots whose identity key isn't already present get fired. This lets you
extend an earlier sweep after changing feasibility (e.g. raising
`MAX_NUM_SEQS` from 128 to 256 so mixed `n=128` corners become feasible)
in minutes instead of hours. Resume applies to every category plus
skew. Historical or incompatible acquisition identities require a separate
output root or explicit `--force`, which replaces the categories and TP degrees
selected by the run. A fully skipped ordinary category keeps its prior
measurement timestamp. Updating profiler code alone never updates stored times.

Native MoE contracts reject `--force`. A full MoE `profile --force` is therefore
rejected because it acquires native components. Use a new output root for a full
remeasurement, an ordinary category `slice --force` for that category alone, or
`profile --only-skew --force` for skew alone.

Skew feasibility uses the resolved block size for the actual heterogeneous
shot; uniform reference batches are looked up offline, not measured. Skew checkpoints use atomic file
replacement and preserve existing permissions. Corrupt CSVs fail explicitly
instead of being silently replaced; `FORCE=1` still requests a fresh sweep.

#### Output naming

```bash
VARIANT="my_experiment"             # override the auto-derived <variant> folder name.
```

When omitted, `<variant>` keeps the established checkpoint label and adds
suffixes for explicit activation/KV overrides, such as `fp8-actfp16-kvfp8`.
Use an explicit `VARIANT` for HF precision or any nested `text_config` override,
ambiguous configurations and deliberately separate acquisitions. A label alone
does not prove the resolved engine precision; see the
[naming limits](https://llmservingsim.ai/docs/profiler/running#precision-and-variant-names).

#### Dtype

```bash
DTYPE="bfloat16"                    # explicit engine dtype override; normally unset
KV_CACHE_DTYPE="fp8"                # explicit KV override; normally vLLM's "auto"
```

Without `DTYPE` (or with `DTYPE=auto`), vLLM resolves the checkpoint dtype. The default variant
uses its quantization method or `torch_dtype` / `dtype`; setting an engine dtype
alone does not quantize or dequantize a checkpoint. Matching BF16/FP16 activation
overrides retain the quantization name; different activations add `-act...`.
An explicit FP8 KV override adds a
`-kvfp8` suffix. Match it to the simulator's model config: profiler flags do
not edit that config or the simulator's cache dtype.

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
   `model_type`, checking both filenames and declared `model_types:` aliases.
   A matching filename cannot hide another catalog's claim to the same type.
   Fails with a clear error and an "available architectures" list if
   nothing matches.
3. Writes the model config to a temp directory and spins vLLM up
   against it — no HF round-trip is needed after the first fetch.
4. Sweeps dense / per-sequence / attention (and MoE if applicable)
   shot grids, writing CSVs under `perf/<HW>/<MODEL>/<variant>/tp<N>/`.

`<variant>` follows the checkpoint/activation/KV naming rules above. Explicit
activation alternatives can use names such as `fp8-actfp16`; ambiguous HF
overrides require `VARIANT=<name>`. The simulator does not automatically select
an alternate acquisition merely because the folder exists.

### 4. Use in simulation

The simulator's `trace_generator.py` reads from
`profiler/perf/<hardware>/<model>/<variant>/tp<N>/*.csv`
automatically when the cluster config names a matching hardware and
the CLI selects a matching model.

### Sweeping several models: `profiler/profile-all.sh`

Helper template that wraps `python -m profiler profile` in a loop over
the `JOBS` array. Each entry specifies a model and explicit per-job flags.
Inspect that array and the global overrides before bringing up a fresh GPU.

```bash
./profiler/profile-all.sh
```

Global knobs accept environment variables (no argparse). Unlike `profile.sh`,
this is an explicitly configured multi-model campaign, with its own attention
bounds/factors and per-job TP settings; it does not promise CLI-default parity.
Per-job flags override globals. For example:

```bash
HARDWARE=H100 \
ATTENTION_CHUNK_FACTOR=1.5 \
./profiler/profile-all.sh
```

For accepted globals, inspect the command construction in `profile-all.sh`;
arbitrary per-job CLI options go in the entry's extra flags.

To change the model list or deployment degrees, edit the `JOBS=( ... )` array at the top
of the script. This file is meant to be copied or tweaked in-place,
not treated as a stable CLI.

## Measurement behavior

### Acquisition bounds

Synthetic shots are checked against actual engine request, query-token and
context limits before page allocation or GPU execution. Each request must fit
its history, query and one output token within the context bound.

Coverage and component-layout discovery use fixed probes requiring at least
5 requests, context length 258 and an acquisition token buffer of 68 (logical
token budget plus `max_num_seqs`). All probes are validated before the first
worker call. Smaller settings are refused; probes are never resized or skipped
to fit the requested grid. See the [coverage guide](../docs/docs/profiler/running.md).

### CUDA timing and ownership

Ordinary and skew acquisition use one lightweight PyTorch/Kineto collector:
explicit module scopes retain CUDA kernels, copies and memsets without recording
Python stacks, tensor shapes or memory snapshots. Geometry and finite-output
checks run outside timing. New native MoE components are measured eagerly,
without isolated CUDA graph capture. Historical eager/graph tables retain their
recorded modes.

Ordinary layerwise categories and skew use per-call CUDA interval unions: streams
on the same device share an interval union, while distinct module calls remain
separate before averaging. CPU time and gaps are excluded.

Common direct-method and exact native operator contexts preserve runtime
ownership across category slices without adding CPU latency or descendant
normalization counts. Catalogs may also declare exact loaded-function or live-object
method scopes and select parent or invocation normalization. These scopes time
correlated CUDA activity, not Python execution; unavailable targets are reported,
never replaced with zero timings. See the
[callable contract](../docs/docs/profiler/adding-model-architecture.md#declared-callable-boundaries).

Coverage also reports actual calls per measured forward separately from
positive-CUDA calls, raw CUDA activity counts and timing-normalization divisors.

See the [acquisition contract](../docs/docs/profiler/output-bundle.md#cuda-activity-and-acquisition-identity)
for the supported metadata paths and remeasurement requirements.

### Synthetic inputs

Synthetic KV preparation reserves disjoint pages from vLLM's shared physical
pool, preserves the null block and initializes only assigned pages outside timing.

History-bearing shots initialize assigned attention pages with vLLM's
`initialize_single_dummy_weight`, using its deterministic uniform defaults.
Initialization is repeated outside warmup and timing;
asynchronous outputs are drained before request/input buffers are reused.
Recurrent caches use zero-start state, not synthetic historical recurrence.

Supported NVIDIA packed MLA/indexer caches use floating dummy payloads from the
same vLLM initializer, packed with the native format's deterministic scales and
layout. Only assigned pages are written, in bounded source chunks outside timing;
unknown packed formats still fail explicitly. Floating MLA indexer caches use
the typed path, including three-dimensional views that omit a singleton head
axis; dtype and logical shape must match the cache spec, and strides must pass
the non-overlap check. Padded and permuted views are allowed. See
[packed attention history](../docs/docs/profiler/output-bundle.md#packed-attention-history).

Dummy values do not claim to reproduce trained KV content.

Recurrent warmup checks distinguish GDN's prefill-only views from native KDA's
complete non-speculative view without relaxing state-page or token checks.

Dummy loading prepares supported native MXFP4 expert bytes and KDA derived
buffers before vLLM's unchanged postload processing. Ordinary models retain
their native initialization. Preparation receipts are not forward-validation
certificates; see [dummy weights](../docs/docs/profiler/output-bundle.md#dummy-model-weight-preparation).

Measured decode requests also preserve their prompt/output boundary. V1 registers
query tokens as output state; V2 receives a separate prompt length and full token
list. vLLM still chooses request order and attention kernels. A shape-correct
batch alone does not prove the intended prefill/decode execution path.

### Model shape and TP emulation

Category-specific decoder reductions retain the model's explicit layer-type prefix,
including nested text configurations.

Backbone-only overrides are consumed after writing the nested config, avoiding
duplicate assignments to wrapper properties. Root-only and shared root/text
overrides stay forwarded; flat model configuration is unchanged.

TP emulation resolves user overrides before sharding and preserves vLLM's
unsharded attention head width. Sharded candidates and any width pin are checked
after writing and reloading the raw config, so constructor-derived defaults
cannot silently change the measured shape. Nested backbones receive a required
scalar `head_dim` pin even when that field is absent from their raw JSON.
Separately declared `dense_intermediate_size` is sharded alongside the other TP
dimensions. Old incorrectly shaped projections, attention and skew need fresh
measurement in a separate output root or explicit category replacement, not
metadata relabelling; a source edit alone does not invalidate their resume keys.

### MoE measurement

New MoE measurements separate gate/routing, experts and applicable finalization.
They run eagerly and never capture an isolated expert CUDA graph. Expert sampling
covers physical token rows N, active local experts A and local assignments M,
with geometric spacing controlled by `--moe-grid-factor` (default `2`) and
mandatory feasible geometry corners. Token axes prefer serving capture sizes
after target DP/SP transformation; geometric points cover eager regimes and
out-of-capture ranges instead of being added everywhere. DP cancellation can
require small local eager points even when only another rank exceeds the cap.
Compiler-SP padding is included in the physical acquisition bound. A and M
remain independent geometric axes, since capture sizes do not determine routing.
Conditioning and repeated measurements remain required; sampling
alone does not establish an interpolation error bound or full-model accuracy.

The simulator constructs each deployment table's Delaunay support on first use,
retains it across batch views for that immutable bundle, and uses
cached barycentric interpolation in physical N/A/M coordinates, with at most four
measured points per query. This is table interpolation, not a fitted latency
predictor. Historical balanced-assignment component tables retain their lookup.

Whole-block acquisition, forced routing and `--moe-ep-degrees` are removed.
Historical `moe.csv` files remain readable until supported components are
remeasured. The live TP1/DP1/EP1 adapter preserves admitted native quantized
weights/backends and measures gate/routing, experts and surrounding wrapper work,
including shared experts and latent transforms. It sums unique CUDA activity
durations, not an overlap-aware elapsed latency. Exact native-reference checks,
storage-preserving bank clones and capability admission remain mandatory;
unsupported paths cannot fall back to the removed acquisition method. See the
[native component contract](../docs/docs/profiler/native-moe-components.md).

The isolated MoE engine uses the MLP prefix and a privately configurable KV-memory
cap (1 GiB by default), preserving the requested context/token/sequence limits.
vLLM must validate native cache capacity; the cap is not a universal feasibility
guarantee. Component-engine observations do not replace main-engine metadata.

## Execution scope

### Serving graph observations

Acquisition settings and target serving settings are separate. Ordinary
acquisition remains eager; importing a serving graph policy does not change it.
After a benchmark has recorded an observed worker execution policy, import it:

```bash
python -m profiler import-execution meta-llama/Llama-3.1-8B \
  --hardware RTXPRO6000 --bench-dir bench/results/<run_id>
```

The variant-root `meta.yaml.serving_execution` retains one compact observation
per deployment scope. The model ID/configuration, device and precision must
match the bundle. Serving uses a record only when its parallelism, scheduler,
context, cache and speculative settings match; explicit instance `cudagraph`
settings win. Missing or incompatible observations retain derived defaults.
Reduced-stack diagnostics cannot be imported as full-model references, and
unknown startup observations are not usable policies. No additional metadata
sidecar or source/version lock is created.

The bundled acquisition hooks target NVIDIA CUDA. Another accelerator requires
an audited acquisition adapter, not just a different image or hardware label;
see [adding hardware](../docs/docs/profiler/adding-hardware.md).

The tracked `perf/` bundles and their metadata define available model, precision
and parallelism coverage. Native MoE additionally requires a matching TP/DP/EP
component contract; a legacy EP table alone does not establish that coverage.

### Catalog boundaries and component layouts

Dense catalog entries can declare `tp_output_reduction: all_reduce`, including
Qwen's GDN output projection. This describes simulator communication, not a new
profiling measurement; see the [catalog fields](../docs/docs/profiler/adding-model-architecture.md#catalog-entry-fields).

Runtime-selected component layouts couple affected tables:
Qwen's fused GDN core needs coordinated `dense` and `linear_attention` acquisition.
Per-TP metadata rejects incomplete or incompatible publication; unmarked historical
tables keep their original decomposition. Use one writer per variant root and keep
published bundles immutable. See the
[layout contract](https://llmservingsim.ai/docs/profiler/output-bundle#runtime-selected-component-layouts).

### Kimi-K3

Catalog `native_profile` selects a runtime-admitted adapter through the normal
`profile`, `slice` and `coverage` entry points. Shared category completion and
callable publication validate full catalog coverage, runtime identities and
partial CSVs before resume. Kimi-K3's hybrid/residual adapter derives prefix
depths and original-stack contexts from configuration, retaining joined MLA
timing and separate KDA components. Ordinary categories, residual contexts and
the separate deployment-matched native MoE acquisition must complete before
opt-in duration trace replay. Full profiles require `--skip-skew`; explicit local
`--dp 1 --ep 1` is admitted, while distributed degrees remain unsupported.
The shared machinery does not admit another backend. See
[native Kimi acquisition](../docs/docs/profiler/output-bundle.md#native-kimi-acquisition)
for supported category slices, memory bounds and resume rules.

### DeepSeek-V3.2

Native DeepSeek-V3.2 uses declared sequence sites and a guarded first-layer norm
fusion. New TP1 eager acquisitions require complete `dense`, `per_sequence` and
`attention` tables sharing a runtime identity, plus independently published native
MoE components; source
geometry, backend modes and the actual native sampler route are checked before
acquisition. Historical records that include a whole-block `moe` category retain
their original completion checks; new components do not certify old pending rows.
Its combined attention/indexer tail retains full-context coordinates.
See [native DSA acquisition](../docs/docs/profiler/output-bundle.md#native-dsa-acquisition)
for supported modes and the separate duration-only trace boundary.

### MiniMax-M3

Native MiniMax-M3 attention acquisition also records a runtime-checked per-TP
lookup contract: Triton BF16 uses the configured selected-block count with the
reviewed 128-token block layout. Initial/local blocks are included in that count;
the shipped configuration gives a 2048 sparse-prefill cap.
Unmarked historical tables retain 2176; raw coordinates
and timings are never relabeled. See the
[native attention contract](../docs/docs/profiler/output-bundle.md#native-attention-lookup-contract).

## Checkpoints and resume

New CSVs contain coordinates and timings;
variant-root `meta.yaml::per_tp[tp].measurements` records per-category
method, dummy-input preparation, execution facts and runtime versions. Semantic
worker/host mismatches, incompatible resumes and mixed-method TP-stable replication
are rejected. Source hashes and exact library versions are provenance, not vetoes;
actual API/backend, shape, dtype, layout and finite-output guards remain.

Starting local acquisition beside TP-stable replicas preserves their source-TP
provenance; local resume never relabels copied timings as locally measured rows.
This applies to ordinary catalogs as well as coupled component layouts. A
replica-only CSV does not skip any unmeasured TP-local shots.
Native DP+EP components retain their separate measurement contract.

Ordinary categories save periodic atomic checkpoints between completed shots.
Resume preserves earlier rows and their acquisition identities; publish a
bundle only after the requested sweeps have completed.

Metadata seals each new CSV publication with its intended data checksum. Resume
rejects an interrupted replacement whose CSV and metadata disagree; a matching
pending publication is reusable without rewriting it. These are data-integrity
checks, not source/version locks or changes to measurement timing.

Attention resume keys retain the CSV's full-precision query-weighted prefill
coordinate; fractional coordinates do not trigger redundant acquisition.

## Output schema

Each `perf/<hw>/<model>/<variant>/` directory contains one `meta.yaml`
(profiler / vLLM version, GPU, timestamps, effective engine kwargs,
compact sweep specs, skew fit summary) and one `tp<N>/` subfolder per
profiled TP degree:

```text
tp<N>/
├── dense.csv              layer, tokens, time_us
├── per_sequence.csv       layer, sequences, time_us
├── attention.csv          layer, prefill_chunk, prefill_key, n_decode,
│                          kv_decode, decode_q_len, time_us
├── linear_attention.csv   layer, prefill_tokens, n_decode, time_us (hybrid only)
├── moe.csv                historical whole-block fallback only
├── mtp.csv                layer, sequences, time_us (one drafter pass; opt-in)
├── skew.csv               ordered geometry and per-forward timings
├── skew_fit.csv           layer/query/prefill/leverage/N calibration buckets
└── moe_components/        deployment-specific component CSVs and optional raw checkpoints
```

Times are in microseconds. Files depend on the model and selected categories;
this overview lists key coordinates, not every CSV column. Acquisition identities
and runtime observations are metadata, not new timing columns. See the [output schema](../docs/docs/profiler/output-bundle.md)
and [native MoE contract](../docs/docs/profiler/native-moe-components.md)
for complete fields and component-bundle contents.

All new profile metadata shares the variant-root `meta.yaml`. Its `per_tp[tp]`
records hold measurements, category publications, contexts/callables, skew and
native MoE entries. Existing root-only engine, lookup, layout and calibration
sections stay in place. New acquisitions do not create standalone metadata JSON
or YAML files beside their tables; historical sidecars have narrow read fallbacks,
except `skew.meta.yaml`, which is no longer loaded. MoE publication retains
settings and a compact conditioning summary, not individual control-point timings;
quality checks still run and raw controls remain in acquisition checkpoints.
Raw repetition checkpoints may remain for acquisition resume. The empty
`.meta.lock` is concurrency control, not another metadata record.
Saved method, policy and format labels use descriptive hyphen-separated names.
Obsolete symbolic contracts require fresh acquisition, not metadata relabeling.
Legacy file-layout reads still require current method contracts; actual
library/model versions and numeric structural schemas are retained.

`prefill_key` is query-weighted across prefill chunks; profiling and serving
share its implementation. See the [attention schema](https://llmservingsim.ai/docs/profiler/output-bundle#attentioncsv)
for the formula and legacy single-prefill compatibility.

**Ordinary categories are keyed by `layer`; historical `moe.csv` is not.** That is not cosmetic:
the `attention` category holds **more than one kernel** on a sparse model and
they share neither a latency curve nor a skew alpha. MiniMax-M3 profiles
`attention` (its non-sparse layers), `sparse_attention` and `indexer`;
historical DeepSeek-V3.2 / GLM-5 tables profile `attention` (MLA) and `indexer`.
The admitted native DeepSeek representation instead measures their combined
`dsa_attention_tail`; it must not also charge the historical pair. Reading a
pooled table would give a sparse layer the dense kernel's latency — 2.1x per
layer on M3.

Attention is a single **5D** table covering pure-prefill, pure-decode and
mixed kernel shapes (what vLLM's chunked-prefill scheduler actually produces
each step). The axes grow geometrically — `prefill_chunk` and the kv axes by
`ATTENTION_CHUNK_FACTOR` and `ATTENTION_KV_FACTOR` (both default `2`),
`n_decode` uses a sqrt(2) factor. `decode_q_len` is the fifth axis and defaults
to just `[1]`, because it only matters for speculative decoding and each extra
value adds its decode-containing slice (not another pure-prefill sweep);
see `ATTENTION_DECODE_Q_LENS`.
Applicable pure-prefill and pure-decode slices additionally include the target
graph anchors described under [sampling policy](#sampling-policy).

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

`meta.yaml` records resolved engine state and sweep metadata, including:

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

  New authoritative engine records also retain `model_text_config`, a snapshot
  of the native resolved text config, including consumed nested overrides and
  reduced depth. It survives deletion of the temporary model directory. MTP and
  short-stack refreshes preserve the main engine's entry; older records without
  a snapshot remain valid but do not gain invented fields. This is not a complete
  wrapper, backend or parameter-layout record.

- `attention_grid` — the 4D attention sweep's caps (`max_kv`),
  geometric factors (`chunk_factor`, `kv_factor`), and compact spec
  strings for the `chunks` / `n_decode` / `kv` axes, plus the separate
  `decode_q_lens` table selections. Only an attention sweep rewrites this grid.
- `per_tp[tp].sampling` — per-category feasible shot counts, geometric factors,
  target graph anchors and observed/derived source. These are acquisition plans,
  not completion certificates; MTP records its request-axis limitation.
- `skew_profile` — actual per-TP dynamic plans, resolved capacity and completion.
- `attention_lookup.per_tp[tp]` — native attention semantics, acquisition identity
  and completed table checksum. Only matching new measurements establish this
  contract; other category refreshes preserve it. Empty skew plans publish no
  new skew provenance and preserve existing files, including with `--force`.
- `component_layouts.per_tp[tp]` — selected representation and coupled category
  states. Native DSA additionally retains the shared runtime identity and each
  category's live engine evidence; table presence alone cannot certify completion.
- `skew_fit` — recognized calibration format, reference fingerprints, adaptive prefill
  partitions, supported N anchors, per-kernel/query defaults and table checksum.

## Skew profiling and calibration

### Dynamic coverage

The default sweep covers bimodal, outlier, trimodal, ramp, lognormal, Pareto,
near-uniform and uniform-spread histories; ordered, reversed, interleaved and
shuffled requests; and equal/unequal multi-prefill splits. Axes follow the
user's sequence/token/context bounds and resolved engine capacity. Query
lengths follow `--attention-decode-q-lens`; each requires its own attention
reference slice. A geometric operating cell gets 32 distribution draws by
default, not workload-derived samples. The profiler then checks those draws
against the compiler's per-kernel/query/prefill/leverage support floor and
selects extra batches for undersampled N anchors. Selection reads attention
references, never skew latencies or benchmark results. Candidate search is
bounded; any remaining deficits are recorded rather than hidden by a
successful acquisition status. Expanded batches are streamed, not retained.

### Measurement and validation

Skew warmups verify the backend's query/history geometry. When a backend
discards query boundaries, verification follows its common-metadata builder
and exact returned object; no backend is exempted from the check.
Timed skew kernels are assigned by native launch correlation and the launching
module's CPU scope, with OS-thread mapping across profiler event namespaces.
This changes ownership only, not GPU durations or CPU overhead. Missing or
ambiguous launch evidence rejects the measurement. Skew resume checks semantic
method, effective catalog and full contexts; runtime versions are observations,
not exact-version execution locks. Old rows are not relabelled as new acquisitions.

Only the actual heterogeneous batch is measured. Three independent contexts
of three timed forwards produce a median of forward medians. Per-forward
times, geometry and query roles are preserved. Failed cases stop with a
checkpoint; incomplete kernel sets and insufficient repetitions are retried.

### Offline calibration and lookup

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

The lookup fingerprint includes batch-context construction. A change there
can require a CPU refit even when attention references and fitted CSV values
remain unchanged; do not bypass the stale-fit check by editing hashes.

`--skip-skew` skips acquisition, not existing calibration. `--only-skew`
requires an existing attention table. Existing raw rows are retained unless
`--force` is explicit; incompatible acquisition methods are rejected rather
than merged into a new sweep. `plan-skew` includes support additions and unresolved
deficits in its preview. New acquisitions record completion
and resolved limits in root `meta.yaml::per_tp[tp].skew`. If all rows were saved
before the completion update was interrupted, resume seals the record without
new measurements or provenance. Refit rejects unfinished
acquisitions and absent raw data
and stale references are rejected at simulator startup. See the
[skew guide](../docs/docs/profiler/skew-alpha-fit.md) for contracts and limits.

Validate every reported `bench/examples` statistic. Reusing broader measured
data can improve one model and worsen another; neither sampling families nor
the mean/max summaries establish generalization to unseen distributions.

## Architecture catalogs

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
    # Additional bindings omitted.
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
TP (layernorms, sampler). They're profiled once at TP=1. The writer copies stable
`dense` and `per_sequence` rows to other TP folders; stable MTP rows remain at
TP1 and the simulator reads them there.

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

A catalog entry can name a real vLLM class and measure **nothing**: an existing
module can be bypassed in the actual forward, and modern models
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

Kernel wildcards must also exclude already-owned subtrees. DeepSeek/GLM
`indexer_glue` excludes `LayerNorm` and `SparseAttnIndexer`: their copy, fill and
normalization kernels are already included in `indexer_k_norm` or `indexer`.
Coverage can report no gaps while latency extraction still double-counts a
parent binding and its bound child. Audit nested matches as well. Existing
`indexer_glue` rows need a dense-category refresh; changing the catalog does
not repair stored CSV timings.
Phi-MoE's `block_glue` excludes its MoE, normalization and attention child
subtrees, retaining only direct decoder residual work. Refresh old Phi dense
rows as well. These exclusions also apply to category slices. The collector
rejects overlapping canonical CUDA claims within a capture before saving timings.

Coverage checks unbound work and overlapping or structurally unrelated claims.
With MTP enabled, it suppresses expected non-MTP matches on the drafter's
replayed block, but still rejects MTP overmatches and shared CUDA ownership.
An unexecuted catalog alternative may leave no activity gap, and structurally
plausible bindings still need a semantic audit. The drafter's modules share
classes with the target, as DeepSeek's shared expert does with a dense layer's
`mlp`. Qwen3.8-27B's historical
`mtp_norms` recorded **1287 µs at one sequence** for two RMSNorms with a
perfectly smooth monotone curve. Dump the profile tree and read the ancestor
chains: boot at the depth in question -- with `--profile-mtp` for a drafter --
and print every node's class with its ancestor path, once per batch regime,
filtered to the class under suspicion.

### Sanity-check a fresh bundle against a bandwidth bound

Nothing in a CSV reveals a constant-factor error. What does is physics:
`lm_head` reads the whole output embedding, so

```text
vocab * hidden * dtype_bytes / mem_bw
```

estimates a memory-time lower bound when the head weights must be read from
device memory. Llama-3.1-8B on an RTX PRO 6000: 128256 × 4096 × 2 B ÷
1.8 TB/s = 583 µs, against which 714 µs is about 82% bandwidth efficiency.
Use TP-local sizes and account for cache residency; a slower outlier alone
does not prove a bug. Check attribution and invocation counts with controlled
measurements. Attention also requires its own traffic estimate rather than
assuming every kernel, query length and backend is purely bandwidth-bound.

## Adding a new model

1. **Drop its HF `config.json`** at `configs/model/<org>/<name>.json`.
   (Or let the profiler auto-download on first run if `HF_TOKEN` is
   set in the container.)
2. **If the model's `model_type` resolves to an existing catalog**, check live
   coverage, cache layout and the applicable [execution scope](#execution-scope)
   before acquisition. A catalog or alias alone does not establish support for
   another backend, deployment or runtime version. Edit `MODEL` and `HARDWARE`
   in `profiler/profile.sh` and choose settings within the admitted scope.
   Category depth is resolved from the checkpoint and logged; leave
   `NUM_HIDDEN_LAYERS` unset unless the acquisition contract permits an override.
3. **If it's a new architecture family** (e.g., `gemma2`):
   * Create `models/<model_type>.yaml` mapping the new family's vLLM
     classes to canonical names.
   * Read the model's source under
     `../vllm/vllm/model_executor/models/<name>.py` for orientation, but
     **write the catalog from a live profile dump, not from the source** — the
     module tree and the profile tree differ both ways.
   * Run `python -m profiler coverage <model> --hardware <hw>` until it binds every kernel,
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

```text
(default)                    INFO — TP limits, stage timings, progress.
--silent                     WARNING — warnings only.
--verbose                    DEBUG + vLLM stdout/stderr.
--log-level {DEBUG,INFO,…}   explicit override.
```

Set `LOG_LEVEL="ERROR"` or `VERBOSITY="--silent"` / `"--verbose"` in
`profiler/profile.sh`, not both, or pass the corresponding CLI option directly.

## Slice-refresh (partial re-profile)

After the first full sweep, iterate on one category (e.g., tune the
attention grid) without redoing everything:

```bash
python -m profiler slice meta-llama/Llama-3.1-8B \
    --hardware RTXPRO6000 --tp-refresh 1 --group attention
```

Resumes compatible rows in `tp1/attention.csv` and refreshes the relevant
`meta.yaml` fields. Add `--force` to replace that category's measurements;
use a separate output root when old measurements must be preserved.

`--tp-refresh N` requires `N` to be one of `--tp`'s degrees, which defaults to
`1` — so refreshing a `tp2/` folder needs both:

```bash
python -m profiler slice Qwen/Qwen3.8-27B --hardware RTXPRO6000 \
    --tp 1,2 --tp-refresh 2 --group mtp --profile-mtp
```

Otherwise it exits with `tp=2 is not in the session's tp_degrees ([1])`.
