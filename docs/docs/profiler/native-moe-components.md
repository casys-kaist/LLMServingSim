---
sidebar_position: 5
title: Native MoE components
---

# Native MoE components

DP+EP does not apply one token count to an entire MoE block. Gate and routing
operate on local rows, experts operate on gathered rows, and finalization returns
to local rows. Native component profiling measures these GPU regions separately
while leaving communication to the network model.
DP1 uses the same component acquisition and lookup pipeline, but only measures
local gate/routing and experts. Its admitted native boundaries perform no CUDA
work; it has no DP dispatch/combine.

The shipped RTXPRO6000 BF16 Qwen3-30B bundle includes a TP1/DP2/EP2 component
entry in its root metadata. Its contracts define the actual bounds and backend; the same model name
at another TP or DP degree does not imply coverage. Published bundles contain
the runtime files, not the original per-forward acquisition streams. Keep the
original acquisition directory to resume it, or choose a separate `--out-root`
for a new measurement.

The adapter uses version-sensitive vLLM internal APIs. Deferred finalization
is checked through `use_deferred_moe_finalize`, the effective boolean property
in 0.28 and 0.31. The similarly named `defer_moe_finalize` is a request flag
in 0.28 and a request method in 0.31; its truthiness is not an execution mode.
Unknown capability APIs and actually deferred execution are rejected.
A vLLM upgrade or a different expert backend requires auditing the whole
execution contract, not merely reusing the same CSV column names. Compatibility
depends on native APIs, geometry, backend and conditioning semantics, not source
hashes, exact library versions or directory names. A semantic method change does
not relabel historical measurements.

## Acquire a deployment-matched table

For a supported unquantized model, refresh only the MoE category:

```bash
python -m profiler slice Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --hardware RTXPRO6000 --tp 1 --tp-refresh 1 --group moe --dp 2 \
  --max-num-batched-tokens 2048 --max-num-seqs 128 \
  --moe-rounds 3 --measurement-iterations 3
```

`--dp` accepts comma-separated target degrees. EP is resolved as `TP * DP`.
Use `--dp 1` for a local target or `--dp 1,2` to acquire both independently.
DP1 with TP1 is a single-GPU MoE block. At TP greater than one, this option
means whole experts distributed over EP=TP, not tensor-sharded experts.
Do not reuse one geometry's measurements for the other, or combine `--dp`
with the legacy `--moe-ep-degrees` sweep. To refresh TP2,
use `--tp 1,2 --tp-refresh 2`; the existing CLI still requires TP1 in its
session list, but a slice measures only the selected degree.

The profiler uses **one physical GPU**, not TP times DP processes. It retains
the checkpoint's global expert count and top-k, resolves each target rank's
expert map through vLLM, and builds local expert weights in the native layout.
Expert weights are synthetic; this measures the selected implementation and
tensor geometry, not a trained router's distribution. Collective execution
and CPU time are excluded. Target and acquisition parallelism are recorded
separately, so a TP2/DP2 table is not evidence of a four-GPU communication run.

Omitting `--dp` preserves the legacy whole-block acquisition. The explicit-DP
path also works in `profile`, replacing only its MoE category. A MoE slice
does not restamp the main engine, attention grid or skew metadata.

## Shapes and measurement protocol

`MoeTarget` is shared by acquisition and serving. Given already-padded DP
counts `n_i`, a non-sequence-parallel rank uses `n_i` local rows and
`sum(n_i)` expert rows. A sequence-parallel rank uses `ceil(n_i / TP)` local
rows and `TP * sum(ceil(n_i / TP))` expert rows. The catalog declares whether
the audited model wrapper uses sequence parallelism; TP alone does not decide it.
This adapter's DP1 path is non-sequence-parallel: local and expert row counts
are equal. The native backend is selected from each target's resolved configuration;
it may differ between DP1 and DP+EP even on the same model and GPU.

For DP1, native preparation must preserve the input without scales or dispatch
metadata. Experts write to the caller's output buffer; the admitted native
NoOP reduction then needs no copy. Acquisition verifies alias identity and
absence of CUDA activity in prepare/finalize before recording these boundaries
as `local_execution` in the root contract. No zero-time or synthetic-copy row
is added. A backend requiring additional work is rejected rather than silently
omitted or replaced by another backend.

The token grid grows geometrically to the configured local budget and its
derived gathered maximum. Expert coordinates retain global top-k and contain
only feasible local active-expert counts at the balanced local assignment total.
The grid is derived from configuration and model geometry, not benchmark requests.

Each component is measured in eager and CUDA graph modes. Launch-correlated
CUDA activities are attributed to individual forwards without adding CPU
durations or launch gaps. `--moe-rounds` controls independent measurement
contexts; `--measurement-iterations` is the minimum forwards per context.
Expert measurements may use more forwards to complete the required weight cycles.
The stored time is the median of the per-context forward medians, in microseconds.

Expert measurements rotate local expert positions and independent weight banks.
Bank count is derived from local expert weight bytes and GPU L2 capacity, with
a minimum of two banks. The `B` and `2B` arms use identical warmup and timed
forward counts covering complete cycles. Every expert point must pass the
recorded 3% pooled-median stability check before the higher-bank measurement is
published. Graph variants use independent allocator pools and are checked
against independent eager output copies, including after all captures.

These are explicit measurement-conditioning rules, not a guarantee of exact
full-model cache behavior. Local gate/finalization regions remain isolated-warm.
Kernel selection and arbitrary trained routing distributions can still differ.

## Files, resume and lookup

The variant-root `meta.yaml::per_tp[tp].moe_components` record indexes target
tables. New output uses readable deployment directories rather than contract hashes:

```text
<variant>/
├── meta.yaml
└── tp<N>/
    └── moe_components/
        └── dp2_ep2_rank0_sp0/
            ├── components.csv
            └── samples.jsonl    # optional raw acquisition checkpoint
```

`components.csv` stores `component,mode,tokens,activated_experts,local_assignments,time_us`.
DP+EP stores `gate_routing`, `experts` and `finalize_copy`; DP1 stores only
`gate_routing` and `experts`. The contract declares the expected component set.
The root record has `schema: moe-components` and `entries`, each containing
`target`, a TP-relative table `path`, `contract` and `coverage`. The contract
describes geometry, target TP/DP/EP/rank, expert ownership, dtypes, native backend
and measurement conditioning, with acquisition observations kept as provenance.
Coverage records completion, completed rounds and the table checksum. Its compact
`conditioning_summary` contains `checked_points`, `failed_points` and
`max_relative_change`; the tolerance stays in `contract.expert_conditioning`.
Every control check still runs, but individual control coordinates and timings
are not published in metadata.
`samples.jsonl` preserves individual forwards and control measurements for resume
and auditing; runtime lookup needs only root metadata and the component CSV.
Acquisition failures may additionally retain `failures.jsonl` and a diagnostic
Chrome trace for the failed graph-attribution check. These are optional debugging
artifacts emitted on failure, not normal-path metadata or simulator inputs; do
not commit diagnostic artifacts as part of a published profile bundle.
New acquisitions do not write standalone index, contract or coverage JSON files.
Historical standalone layouts remain readable when their method contracts are
current; file-layout fallback does not translate obsolete symbolic names.
Moving metadata does not alter timing-table bytes or turn them
into fresh measurements. Paths locate tables, not execution identity.

Re-running the same contract resumes completed rounds. An incomplete trailing raw
record is archived before repair; malformed complete records are rejected. A
target has an exclusive `writer.lock` and immutable semantic conditions. `--force`
is rejected: use another output root for a fresh acquisition. Run only one
acquisition writer per target; distinct targets serialize their shared root
metadata updates through `.meta.lock`. Changed version/provenance observations
alone do not change resume identity. Raw samples are needed to resume acquisition,
but not to consume a published metadata-and-CSV bundle.
No-op resume preserves the original runtime observations. Distinct later
observations enter the root entry's `acquisition_contexts` only after validated
new work; this is a context list, not a count of durable or completed rows.
Incomplete or quality-failing acquisitions retain raw evidence but do not publish
a new runtime entry. A historical standalone index is consulted only when the root
MoE owner is absent; a malformed or incomplete root entry cannot fall back to it.

Serving automatically selects a matching index entry. Local components use
one-dimensional interpolation; experts use bounded interpolation within the
measured feasible token/active-expert surface. Local assignment totals are checked,
not treated as an independently measured arbitrary-routing axis. Lookups are cached
with bounded capacity. There is no fitted neural predictor or per-workload multiplier.

Missing deployment coverage or an unsupported routing policy warns and retains
the legacy EP-table path, whose DP/backend accuracy is unverified. Corrupt contracts,
checksum mismatches and queries outside an installed table's supported geometry
are errors rather than silently extrapolated measurements.
Deployment selection caches both loaded tables and absent coverage for the run,
avoiding repeated metadata reads on every MoE layer. Corrupt contracts are not
cached as missing coverage. Restart the simulation after changing a bundle.

## Runtime communication and limits

The supported DP+EP modular path emits gate/routing, three logical AllGathers (hidden
states, top-k weights, top-k IDs), experts, ReduceScatter and local finalization
in dependency order. Tensor widths come from recorded dtypes. For TP greater than
one, the audited wrapper also restores TP output with AllGather for sequence
parallelism or AllReduce otherwise. Native components and collectives are charged
separately; collective time is not included in the component CSV.
DP1 emits gate/routing and experts, followed by TP AllReduce only when TP>1.
There are no DP AllGathers, ReduceScatter or finalize-copy in that trace.

The analytical path retains a worst-rank Ring envelope for unequal DP counts.
Logical tensors and dependency order do not reproduce NCCL's grouped launch,
channel selection or arbitrary ragged task schedule. Optional
[collective links](../reference/cluster-config#collective-specific-links) change
effective communication parameters without changing these tensor shapes.

The current adapter requires positive DP, full EP, linear expert placement,
and unquantized native modular experts with standard activation layout. Shared
experts, monolithic/quantized backends, deferred finalization, input-weighted
routing, padded hidden dimensions, biased or redundant experts, LoRA, EPLB and tensor-sharded experts need
their own adapters. BALANCED and the supported CUSTOM active-count curve use
the balanced assignment surface; arbitrary rank histograms are not covered.
No coverage of other models, hardware or execution paths should be inferred
solely from a matching file format.
