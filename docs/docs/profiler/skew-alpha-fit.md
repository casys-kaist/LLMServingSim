---
sidebar_position: 5
title: Skew & Alpha Fit
---

# Skew & alpha fit

Skew calibration corrects the uniform attention table for heterogeneous decode
histories. It does not replace `attention.csv`, change attention interpolation,
or fit benchmark request latencies. The default profiler compiles a small
lookup table offline; simulation only selects an already fitted bucket.

Existing unversioned bundles retain their legacy lookup until explicitly
rebuilt. New fits use `runtime-skew-calibration-v1`. Rebuilding a bundle changes
simulation results; it is not an accuracy guarantee.

## Measurements and lookup references

The built-in skew sweep still measures bimodal decode batches, with uniform
mean and maximum controls at the same prefill geometry. Each attention-category
kernel has its own row and measured `t_skew_us`. Control timings and their raw
`alpha` remain diagnostic measurements and are never overwritten by estimates.

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
The compiler accepts such rows but the built-in skew sweep currently emits
query length one only.

Legacy bimodal rows are reconstructed from their original shape columns.
General distributions must include the complete request list; mean/max
summaries cannot recover their geometry. One observation per kernel, query
length and ordered case is required. Summarize repeated measurements before
fitting. Resume deduplicates by this geometry, not by absent bimodal fields.

## Sweep structure (`skew.csv`)

The existing two-tier sweep is retained:

- Tier 1 varies decode count, big-decode fraction, prefill chunk, prefill
  history and small-decode history at a representative skew ratio.
- Tier 2 varies the skew ratio at selected anchor shapes.

The geometric grid follows configured sequence, token and context bounds.
Use `--skew-n-factor`, `--skew-pc-factor`, `--skew-kp-factor` and
`--skew-kvs-factor` to change sampling density. See
[Running the profiler](./running). More points do not automatically give a
better fit, and bimodal coverage is not coverage of all request distributions.

Every case must fit the actual resolved KV block size, sequence and token
limits, and cache capacity. Capacity checks include the **largest uniform
control**, newly scheduled tokens and the sampler's context-boundary reserve.
Checkpoints use temporary files, fsync and atomic replacement. Corrupt prior
CSV files raise rather than being silently replaced.

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

Only the `n` axis is support-adaptive. Prefill and lever bins remain fixed and
are persisted with the fit. Defaults and support thresholds live in
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
and `skew.csv`. Omit `--tp` to rebuild every measured TP; use `--out` for an
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

Mean/max and lever do not uniquely represent a request distribution. Fixed
prefill/lever bins, bimodal acquisition, the support floor and fallback remain
empirical choices. Wider distributions and unseen hardware/model combinations
need separate validation. No model-specific branch is used to choose a
favorable calibration.

See [Output bundle](./output-bundle#skew_fitcsv-skew-enabled-runs) for the file
contract and [Trace generation](/docs/simulator/trace-generation#heterogeneous-decode-skew-correction)
for serving-side behavior.
