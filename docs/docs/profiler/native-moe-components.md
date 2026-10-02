---
sidebar_position: 5
title: Native MoE components
---

# Native MoE components

DP+EP does not apply one token count to an entire MoE block. Gate and routing
operate on local rows, experts operate on gathered rows, and finalization returns
to local rows. Native component profiling measures these GPU regions separately
while leaving communication to the network model.

The adapter follows vLLM 0.28 internal APIs. A vLLM upgrade or a different
expert backend requires auditing that execution contract, not merely reusing
the same CSV column names.

## Acquire a deployment-matched table

For a supported unquantized model, refresh only the MoE category:

```bash
python -m profiler slice Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --hardware RTXPRO6000 --tp 1 --tp-refresh 1 --group moe --dp 2 \
  --max-num-batched-tokens 2048 --max-num-seqs 128 \
  --moe-rounds 3 --measurement-iterations 3
```

`--dp` accepts comma-separated target degrees. EP is resolved as `TP * DP`;
do not combine it with the legacy `--moe-ep-degrees` sweep. To refresh TP2,
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

Within each `tp<N>/`, `moe_components.json` indexes target-specific folders:

```text
moe_components.json
moe_components/<contract-id>/
  contract.json
  coverage.json
  components.csv
  samples.jsonl
```

`components.csv` stores `component,mode,tokens,activated_experts,local_assignments,time_us`.
The three components are `gate_routing`, `experts` and `finalize_copy`.
`contract.json` records model/hardware identity, target TP/DP/EP/rank, expert
ownership, dtypes, native backend, source fingerprints and measurement protocol.
`coverage.json` records completed rounds, stability checks and the table checksum.
`samples.jsonl` preserves individual forwards and control measurements for resume
and auditing; runtime lookup needs only the index, contract, coverage and CSV.

Re-running the same contract resumes completed rounds. An incomplete trailing raw
record is archived before repair; malformed complete records are rejected. A
contract has an exclusive writer lock and is immutable. `--force` is rejected:
use another output root for a fresh acquisition. Run only one acquisition writer
per output bundle; the per-contract lock is not a multi-writer index transaction.
Incomplete or quality-failing acquisitions retain raw evidence but do not publish
a new runtime index entry.

Serving automatically selects a matching index entry. Local components use
one-dimensional interpolation; experts use bounded interpolation within the
measured feasible token/active-expert surface. Local assignment totals are checked,
not treated as an independently measured arbitrary-routing axis. Lookups are cached
with bounded capacity. There is no fitted neural predictor or per-workload multiplier.

Missing deployment coverage or an unsupported routing policy warns and retains
the legacy EP-table path, whose DP/backend accuracy is unverified. Corrupt contracts,
checksum mismatches and queries outside an installed table's supported geometry
are errors rather than silently extrapolated measurements.

## Runtime communication and limits

The supported modular path emits gate/routing, three logical AllGathers (hidden
states, top-k weights, top-k IDs), experts, ReduceScatter and local finalization
in dependency order. Tensor widths come from recorded dtypes. For TP greater than
one, the audited wrapper also restores TP output with AllGather for sequence
parallelism or AllReduce otherwise. Native components and collectives are charged
separately; collective time is not included in the component CSV.

The analytical path retains a worst-rank Ring envelope for unequal DP counts.
Logical tensors and dependency order do not reproduce NCCL's grouped launch,
channel selection or arbitrary ragged task schedule. Optional
[collective links](../reference/cluster-config#collective-specific-links) change
effective communication parameters without changing these tensor shapes.

The current adapter requires DP at least two, full EP, linear expert placement,
and unquantized native modular experts with standard activation layout. Shared
experts, monolithic/quantized backends, deferred finalization, input-weighted
routing, padded hidden dimensions, LoRA, EPLB and tensor-sharded experts need
their own adapters. BALANCED and the supported CUSTOM active-count curve use
the balanced assignment surface; arbitrary rank histograms are not covered.
No coverage of other models, hardware or execution paths should be inferred
solely from a matching file format.
