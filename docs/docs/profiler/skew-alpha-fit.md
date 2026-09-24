---
sidebar_position: 5
title: Skew & Alpha Fit
---

# Skew & alpha fit

Skew calibration corrects the uniform attention table for heterogeneous decode
histories. It does not replace `attention.csv`, change attention interpolation,
or fit benchmark request latencies. The default profiler compiles a small
lookup table offline; simulation only selects an already fitted bucket.

Enabled fits must use `runtime-skew-calibration-v1`. Legacy lookup and inline
coefficients are no longer supported; rebuild old enabled bundles. Disabled
bundles, including RTX4090, are unchanged. Rebuilding is not an accuracy guarantee. The broader shipped Llama TP1 and
Qwen3-32B TP2 raw data are retained measurements, not a completed run of the
new acquisition defaults. Their metadata distinguishes historical coverage
from the new protocol; planned cases lacking verified repetitions are
remeasured on the next sweep.

## Measurements and lookup references

The sweep measures the **actual heterogeneous batch**, not its uniform
controls. Each case uses three independent contexts with three timed forwards
by default. Its target is the median of the per-context forward medians.
Every forward time is retained, per attention-category kernel. Historical
measured controls remain diagnostic; they are not relabelled lookup estimates.

The default fit instead reconstructs the requests and computes both references
through the **same attention lookup used by serving**:

```text
m = attention_lookup(batch with uniform mean decode history)
M = attention_lookup(batch with uniform maximum decode history)
g = M - m
a = (measured_skew_time - m) / g
predicted_time = m + fitted_alpha * g
```

This matters because measured controls and interpolated table references need
not agree. A coefficient fitted against one pair cannot be assumed to
compensate the other. Prefill coordinates use the shared query-weighted helper,
including catalog-specific prefill key saturation. Decode references retain
the runtime lookup's history semantics.

Both references must have an exact attention `layer` and `decode_q_len`
slice. Offline fitting refuses to substitute another query length.

### Raw shape contract

New `skew.csv` rows carry `requests_json`: an ordered list of
`[query_tokens, computed_history_tokens]` pairs, plus `decode_q_len`.
At query length one, requests are classified with the same shared helper as
serving; one-query prompt tails count as decode for this lookup.
Multi-query rows additionally require `n_prefill`, an explicit role boundary.
Both acquisition and compilation support the configured
`--attention-decode-q-lens`, requiring matching attention reference slices.

Legacy bimodal rows are reconstructed from their original shape columns.
General distributions must include the complete request list; mean/max
summaries cannot recover their geometry. One observation per kernel, query
length and ordered case is required. Summarize repeated measurements before
fitting. Resume deduplicates by this geometry, not by absent bimodal fields.

## Sweep structure (`skew.csv`)

Eight families cover bimodal, few-outlier, trimodal, ramp, lognormal, Pareto,
near-uniform and uniform-spread histories. Draws span ordered, reversed,
interleaved and shuffled requests and equal/unequal multi-prefill splits.
Mixed batches include the scheduled-token frontier.

Geometric axes follow configured sequence/token/context bounds and resolved
engine capacity, not model names or benchmark distributions. The four
`--skew-*-factor` flags control density; `--skew-samples-per-cell` defaults to
32 distribution draws per operating cell. `--skew-seed` defaults to zero and
`--skew-rounds` to three. See [Running the profiler](./running).

The operating grid is not the lookup grid. The profiler checks its base
draws in the compiler's kernel/query/prefill/leverage partitions and selects
additional batches when an observed N anchor has fewer than `MIN_ROWS`
distinct cases. It uses only request geometry and `attention.csv` references,
not measured skew times or benchmark results. The support floor itself and
runtime picking are unchanged.

Candidate search has a finite bound (`skew_support.SEARCH_MULTIPLIER` times
the configured draws). Rare cells can remain below the floor; the plan's
`support_completion.remaining_deficits` lists them, and the existing
unsupported-cell lookup policy still applies. `completed: true` means all
selected batches were acquired, **not** that every cell has direct support.
Support completion targets cells observed in the base plan; it is not a claim
of exhaustive coverage of every reachable distribution.
The CPU preview includes these additions and deficits. Selected identities
are compact; their request arrays are regenerated one batch at a time.

Existing raw measurements are retained unless `--force` is explicit. The
support check accounts for their usable heterogeneous prefill envelope so its partitions match
the eventual merged fit, but counts support on the new plan alone; legacy
rows cannot hide gaps in the current measurement protocol.

Only the actual shot consumes KV capacity: its allocation is page-aligned,
includes all scheduled queries, and respects the context boundary plus one
sampler token. The collector verifies executed geometry and finite warmup
output, restores unambiguous CPU module containment, and excludes GPU user
annotations from kernel-duration sums.

The plan streams batches rather than allocating the full Cartesian product.
Raw checkpoints are replaced atomically. Successful cases survive failures;
incomplete kernel sets, missing repetitions or a changed protocol need
measurement. A failed run raises and keeps `skew.meta.yaml::completed: false`,
rather than reporting success with missing cases. CPU refitting also rejects
an acquisition marked incomplete; resume it before rebuilding. Acquisition status and actual
resolved bounds are recorded per TP, not reconstructed from CLI defaults.

Preview coverage and remaining cases using saved engine limits:

```bash
python -m profiler plan-skew meta-llama/Llama-3.1-8B --hardware RTXPRO6000 --tp 1
```

This CPU-only preview cannot establish current GPU availability or capacity.
The live sweep always resolves its limits again.

## The fit (`skew_fit.csv`)

Fits are separated by hardware, model, variant, TP, attention kernel and
decode query length. Within each kernel/query slice:

1. Partition by prefill-token count `pc` and relative endpoint gap
   `lev = (M - m) / m`.
2. A measured decode count `n` becomes an anchor only when it has at least
   `MIN_ROWS` distinct cases **in that partition**.
3. Pool observations at unsupported counts into the nearest supported anchor
   on a logarithmic scale. Supported anchors are not merged with one another.
4. Fit a weighted median in each resulting cell, then clip the fitted value.

Sparse new `n` values therefore do not insert unsupported boundaries into all
other partitions. Runtime bucket selection uses the same geometric midpoints;
an exact tie selects the smaller anchor. This is picking, not interpolation
between neighboring alpha values.

The `n` axis is support-adaptive. Prefill cutoffs scale at one-eighth and
one-half of the measured maximum prefill-token count, with a separate tiny
prefill bucket; duplicate cutoffs collapse on small envelopes. Leverage
cutoffs are dimensionless. All boundaries are persisted with the fit. Defaults and support thresholds live in
`profiler/core/skew_calibration.py`; they are common to all hardware and models,
not workload-specific coefficients.

### Why a weighted median?

For one measured case with positive latency `y`:

```text
abs(predicted_time - y) / y
  = abs(g) / y * abs(fitted_alpha - a)
```

Therefore a weighted median of per-case `a`, with weight `abs(g) / y`,
minimizes the sum of absolute relative **profile-latency** errors in that
cell. It is not a neural predictor, a confidence interval, or a guarantee
about end-to-end benchmark errors. The clip is regularization, not a physical
bound. Raw measured values remain unchanged.

Dense kernels omit nonpositive reference gaps. Catalog-declared
key-saturating kernels also accept negative gaps; zero gaps cannot define a
coefficient.

### Fallbacks

An unsupported partition or a decode count outside the measured range uses
the same kernel/query slice's pooled least-squares fallback. That fallback is
not clipped. No fitted kernel/query slice means zero correction: there is no
borrowing across models, hardware, TP, kernels or query lengths.

Pure prefill, a single decode and uniform decode retain the existing bypass.
That does not establish that their underlying attention lookup is exact.

### Storage and stale-data checks

`skew_fit.csv` stores:

```text
layer,decode_q_len,pc_label,lev_label,n_anchor,alpha,direct_rows,pooled_rows
```

`meta.yaml::skew_fit.per_tp[tp]` stores the schema, identity, axes, support
threshold, clipping range, per-kernel/query fallback and measured range, plus
the table path and checksums. The reference records the source attention CSV,
raw skew CSV, lookup implementation and key-saturation semantics.

At load time the simulator validates the attention data, lookup implementation,
fitted table, profile identity and saturation contract. A mismatch raises and
requests a rebuild; it does not silently use an obsolete fit. Raw skew data is
needed for rebuilding, not per-batch execution.

The simulator hydrates cells once. Runtime lookup uses a partition lookup and
binary search over its `n` anchors; there is no fit, neighbor search over
measurements, or file access in the batch loop.

## Skip / refresh modes

`--skip-skew` skips **acquisition**, not previously stored calibration.
On a fresh bundle with no skew data, the simulator applies zero correction.
Existing measurements are retained and recompiled when profile metadata is
written. `--only-skew` refreshes skew measurements without remeasuring ordinary
categories; calibration also requires an existing `attention.csv`.

Rebuild existing measurements on CPU without starting vLLM:

```bash
python -m profiler refit-skew meta-llama/Llama-3.1-8B \
    --hardware RTXPRO6000 --variant bf16 --tp 1
```

The command requires the local model config, `meta.yaml`, `attention.csv`
and `skew.csv`. A bundle with no raw skew measurements is an error: its
metadata stays unchanged and a disabled calibration is not enabled.
Omit `--tp` to rebuild every measured TP; use `--out` for an
alternate profile root. It writes only derived fit tables and the skew-fit
metadata, preserving other categories and metadata fields. Keep a copy before
migrating a published bundle. If interrupted between table and metadata
publication, rerun the command to restore a matching pair.

## Validation and limitations

Use the committed `bench/examples` workloads and the standard validator;
inspect TTFT, TPOT and latency at every reported statistic, not only their
average. Keep those request-level observations out of coefficient fitting.
Development examples are regression checks, not independent generalization
evidence. A shared rule can improve one model while worsening another.

Mean/max and lever do not uniquely represent a request distribution.
Sampling families, relative partitions, the support floor and fallback remain
empirical choices. Wider distributions and unseen hardware/model combinations
need separate validation. No model-specific branch is used to choose a
favorable calibration.

See [Output bundle](./output-bundle#skew_fitcsv-skew-enabled-runs) for the file
contract and [Trace generation](/docs/simulator/trace-generation#heterogeneous-decode-skew-correction)
for serving-side behavior.
