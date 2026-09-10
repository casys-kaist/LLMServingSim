"""Fit a per-bucket alpha from ``skew.csv``.

The skew case measures three latencies at the same operating point:

    t_mean   all decodes uniform at the mean kv
    t_max    all decodes uniform at the max kv
    t_skew   the actual skewed batch (nb at kv_big, n-nb at kv_small)

The simulator uses alpha at lookup time via

    t_predicted = t_mean + alpha * (t_max - t_mean)

**What alpha is.** Split a batch's cost into its prefill and decode parts.
All three shots carry the *same* prefill by construction, so writing
``D(x)`` for the decode cost with every decode at ``kv = x``::

    t_mean = T_pf + D(mean)   t_max = T_pf + D(max)   t_skew = T_pf + D(list)

    gap   = t_max - t_mean           = D(max)  - D(mean)      <- T_pf cancels
    alpha = (t_skew - t_mean) / gap  = [D(list) - D(mean)] / [D(max) - D(mean)]

So alpha is a purely decode-side ratio: the prefill's additive cost cancels
twice. That is what makes it the right quantity to fit -- the same penalty
expressed relative to ``t_mean`` keeps ``T_pf`` in its denominator and so
varies 2.9x with ``kp`` where alpha varies 1.2x.

And note what alpha measures. If the kernel were exactly linear in kv,
``D(x) = a + b*n*x``, then ``D(list) = a + b*sum(kv) = D(mean)`` -- the real
list and the uniform-at-the-mean batch read the same total KV -- and alpha
would be **exactly 0**. Alpha is therefore the *departure from kv-linearity*:
tile padding, CTA-wave quantisation, SM imbalance. It is a second-order term,
which is why measured values run 0.0006-0.03 on real batches and why it is
noise-sensitive.

---------------------------------------------------------------------
The three axes
---------------------------------------------------------------------
``n | pc | lev``, where ``lev = (t_max - t_mean) / t_mean``. Each names one
component of that departure:

    n     its size. One CTA per (query tile, head) means the CTA count scales
          with n, and the quantisation loss depends on how n falls against the
          device's SM count.
    pc    its coupling to the prefill. Alpha cancels T_pf's *additive* part,
          but a model like Llama runs prefill and decode in one varlen call,
          so the decodes' CTAs share waves with the chunk's and ``D(.)``
          itself depends on pc. The interaction does not cancel.
    lev   its normalisation. The departure grows sub-linearly with the spread,
          so alpha falls roughly as 1/lev -- measured 0.20 / 0.066 / 0.087 /
          0.065 / 0.016 across the five lev bins on Llama-3.1-8B.

**This was chosen by running every subset of six candidate axes end to end**,
64 of them x 3 committed bench examples, scored on all 15 metrics
(TTFT/TPOT/latency x mean/p50/p90/p95/p99). Summed mean |err| over the three
examples:

    n | pc | lev                    4.35   <- chosen
    n | pc | disp                   4.43
    n | pc                          4.55
    n | pc | kp                     4.82
    pc(raw) | n | skew_rate
        | kv_big | kp               7.51   <- what shipped before
    lev alone                       8.5
    srate | kp                     63.9    <- worst of the 64

On the one example whose bundle reproduces exactly when re-measured
(Llama-3.1-8B) that is 15 of 15 metrics inside +-1.1%, against +1.6..+4.9%
before.

Three axes were dropped, and the reasons match the algebra above:

    kv_big, disp   already inside ``gap`` / ``lev``; adding them moves the
                   score 4.55 -> 4.43, inside what one example can resolve.
    kp             acts only through the additive prefill term alpha cancels.
                   Real batches make the point anyway: over all 470 mixed
                   batches of the two dense examples, kp's median is **0** and
                   only 3 of them exceed 2048.
    skew_rate      noise. Every subset containing it but not ``n, pc`` lands in
                   the bottom half, and ``srate | kp`` is last of the 64.

**Bins are fixed, not derived from the sweep -- except that ``n`` keeps one
bucket per profiled batch size, split at the geometric midpoints so a runtime
``n`` reads the nearest profiled size on a log scale.** Fixed bins split the
data more slowly than derived ones (at 30 cells only 4 of 20 measured real
batches hit one and the rest fell back to the pooled constant regardless), but
``n`` cannot be coarsened: alpha differs **2.1-2.5x** between two adjacent
profiled sizes, measured at 42 coordinates with (nb, skew, kvs) held identical,
and a run at ``max_num_seqs`` never reaches the larger one. A bucket spanning
128 and 256 charged real batches the average of their own regime and one they
cannot enter -- and its median flipped between the two populations as rows
arrived (0.0437 at 116 rows, 0.0285 at 127). Scored against 1,084 batches
measured on the live engine, the lever-weighted alpha residual is **1.77%** of
the two dense examples' spans pooled that way against **0.21-0.26%** with one
bucket per profiled size.

And ``pc`` used to be keyed by its **raw** value, which is why every mixed
batch missed: a runtime chunk is ``min(remaining prompt, budget left after the
decodes)`` and lands on a profiled grid point only by coincidence -- 208 of 208
lookups on the Qwen3-32B workload fell through to the pooled alpha.

Within a cell the fitted constant is the **median** of the per-row alphas, not
the weighted-LS optimum ``sum(dtm*dts) / sum(dtm^2)``. WLS is the right
objective when the noise on ``dts`` is homoscedastic -- the charged error is
``(a - a*) * dtm``, so weighting the residual by ``dtm^2`` is optimal -- but
``dts`` is a difference of two nearly-equal measurements, so its noise scales
with ``t_mean``, which grows with the batch and correlates with ``dtm``. Under
that noise model ``dtm^2`` over-trusts the few largest-gap rows, which are
exactly where one noisy shot dominates. Measured both ways through the whole
pipeline, median wins on the sweep's own rows (Llama ``rel_err_p50``
0.0588 -> 0.0308) *and* end to end. That A/B swapped only the estimator, at
the ``n`` bins of the time; the shipped fit reads 0.0259.

A cell with fewer than ``_MIN_BUCKET_ROWS`` rows is not written; those
batches read the layer's pooled ``alpha_default``, measured on the same
kernel. Cells are clipped to ``_ALPHA_CLIP``.

The axes are written to ``meta.yaml::skew_fit.bucket_axes`` with an explicit
``axes`` list, and the simulator reads them from there. A bundle fitted before
this change carries no ``axes`` key; its cells are then unreachable and every
batch reads that bundle's pooled ``alpha_default``, which is what mixed
batches already got.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd


# ---------------------------------------------------------------------------
# Bucket axes
# ---------------------------------------------------------------------------
# ``pc`` and ``lev`` carry fixed edges rather than one bucket per profiled
# value. At that granularity the data splits faster than it adds information --
# which is what the retired ``kp`` axis did, and at 30 cells only 4 of 20
# measured real batches hit a cell at all -- and fixed edges make two bundles'
# tables comparable.
#
# ``n`` is the exception: ``_derive_n_axis`` gives it one bucket per profiled
# batch size, because those follow whatever ``max_num_seqs`` the sweep ran at
# and alpha differs 2.1-2.5x between two adjacent ones. The constants below are
# only the fallback for a CSV with no usable ``n`` column.
_N_BINS_FALLBACK = (0, 2, 4, 8, 16, 32, 64, 128, 1_000_000_000)
_N_LABELS_FALLBACK = ("n<=2", "n<=4", "n<=8", "n<=16", "n<=32", "n<=64",
                      "n<=128", "n>128")
_PC_BINS = (-1, 1, 256, 1024, 1_000_000_000)
_PC_LABELS = ("pc0", "pcS", "pcM", "pcL")
_LEV_BINS = (0.0, 0.25, 0.75, 1.5, 3.0, 1_000_000_000.0)
_LEV_LABELS = ("lev0", "lev1", "lev2", "lev3", "lev4")

# A cell needs real support. Below this it falls back to the layer's pooled
# alpha, which is measured on the same kernel.
_MIN_BUCKET_ROWS = 20

# Cells outside this range are noise: the sweep's own alpha p10-p90 is
# -0.04..0.82, and an alpha below -1/lever drives the blended lookup negative.
_ALPHA_CLIP = (-0.2, 1.0)


# ---------------------------------------------------------------------------
# Helpers for deriving / applying bucket axes
# ---------------------------------------------------------------------------


def _short_kv(v: int) -> str:
    """Readable short form: 1024 → '1k', 65536 → '64k', 300 → '300'."""
    if v >= 1024 and v % 1024 == 0:
        return f"{v // 1024}k"
    return str(int(v))


def _bucket_label(bins: tuple, labels: tuple, val) -> str:
    """Generic ``(bins[i], bins[i+1]]`` lookup. Inclusive on the right
    so the label matches its intuitive reading (``n<=8`` includes 8).
    """
    for i in range(len(labels)):
        if val <= bins[i + 1]:
            return labels[i]
    return labels[-1]


def _derive_n_axis(df: pd.DataFrame) -> tuple[tuple, tuple]:
    """One bucket per profiled ``n`` value, split at the geometric midpoints.

    Grid ``[2, 4, 8, 16, 32, 64, 128, 256]``
      -> bins   ``(0, 3, 6, 11, 23, 45, 90, 181, 362, 1e9)``
      -> labels ``('n=2', ..., 'n=256', 'n>256')``

    The edges sit at ``sqrt(v_i * v_{i+1})``, so a runtime ``n`` reads the
    profiled batch size **nearest to it on a log scale** -- the same rule
    ``decode_q_len`` and the MoE ``ep`` degree already use, and the right one
    for a grid the sweep spaces geometrically. Putting the edges *at* the
    profiled values instead makes every ``n`` in ``(64, 128]`` read n=128's
    alpha, so n=65 is priced by a batch twice its size.

    Derived rather than fixed so the buckets follow whatever ``max_num_seqs``
    the sweep ran at: a run at 256 gets a bucket for 256, a run at 512 gets one
    for 512, and neither pools two profiled sizes. That matters more here than
    on any other axis -- alpha differs **2.1-2.5x** between two adjacent
    profiled sizes (42 coordinates with ``nb``/``skew``/``kvs`` held
    identical), and a run never schedules past ``max_num_seqs``, so a bucket
    spanning 128
    and 256 charges real batches the average of their own regime and one they
    cannot enter. On Llama-3.1-8B that cell held 38 rows at n=128 (median
    0.0589) and 37 at n=256 (0.0219), and its pooled median came out 0.0258 --
    **0.44x** the 0.0539 measured on 224 of the run's own n=128 batches.

    The overflow bin lets the simulator fall back cleanly for a runtime batch
    larger than anything the profiler fired.
    """
    vals = (sorted({int(v) for v in df["n"].dropna().unique()})
            if "n" in df else [])
    if not vals:
        return _N_BINS_FALLBACK, _N_LABELS_FALLBACK
    edges = [int(round((vals[i] * vals[i + 1]) ** 0.5))
             for i in range(len(vals) - 1)]
    # A midpoint that collides with its lower neighbour (adjacent integers on
    # the grid) would empty that bucket; nudge it up so every profiled value
    # keeps a bucket of its own.
    for i in range(len(edges)):
        floor = vals[i] if i == 0 else max(vals[i], edges[i - 1] + 1)
        edges[i] = max(edges[i], floor)
    bins = (0,) + tuple(edges) + (int(round(vals[-1] * 1.415)),
                                  1_000_000_000)
    labels = tuple(f"n={v}" for v in vals) + (f"n>{vals[-1]}",)
    return bins, labels


def _derive_bucket_axes(df: pd.DataFrame) -> dict[str, Any]:
    """The bucket axes. ``pc`` and ``lev`` are fixed; ``n`` follows the sweep.

    ``axes`` is what tells a reader -- and the simulator -- which form a
    bundle holds. A bundle fitted before this change carries no ``axes`` key
    and its cells are unreachable by the new key builder, so every batch falls
    back to that bundle's pooled ``alpha_default``: the behaviour mixed batches
    already had, since ``pc`` used to be keyed by its raw value and a runtime
    chunk lands on a profiled grid point only by coincidence.
    """
    n_bins, n_labels = _derive_n_axis(df)
    return {
        "axes": ["n", "pc", "lev"],
        "n_bins": list(n_bins),
        "n_labels": list(n_labels),
        "pc_bins": list(_PC_BINS),
        "pc_labels": list(_PC_LABELS),
        "lev_bins": list(_LEV_BINS),
        "lev_labels": list(_LEV_LABELS),
    }


def _bucket_key(axes: Mapping[str, Any], n, pc, lev,
                layer: str | None = None) -> str:
    """``[{layer}|]{n_label}|{pc_label}|{lev_label}``.

    ``lev`` is the endpoint gap in units of the batch's own cost,
    ``(t_max - t_mean) / t_mean``. It is the axis alpha varies most strongly
    along -- 0.20 at lev<0.25 down to 0.016 above 3.0 on Llama-3.1-8B -- and
    it costs nothing to compute at either end: the fit has both timings in the
    row, and the simulator has both lookups in hand before it needs the alpha.
    """
    n_label = _bucket_label(tuple(axes["n_bins"]), tuple(axes["n_labels"]),
                            int(n))
    pc_label = _bucket_label(tuple(axes["pc_bins"]), tuple(axes["pc_labels"]),
                             int(pc))
    lev_label = _bucket_label(tuple(axes["lev_bins"]),
                              tuple(axes["lev_labels"]), float(lev))
    key = f"{n_label}|{pc_label}|{lev_label}"
    return key if layer is None else f"{layer}|{key}"


# The axes with no CSV to derive ``n`` from. ``lookup_alpha`` needs them for a
# fit block that predates ``bucket_axes``, and the scratch scoring tools build
# their keys through the same path.
def default_bucket_axes() -> dict[str, Any]:
    """The axes, for a caller with no profile data to hand.

    ``n`` falls back to ``_N_BINS_FALLBACK``, so a key built this way only
    lines up with a bundle whose sweep used those batch sizes.
    """
    return _derive_bucket_axes(pd.DataFrame())


def _fit_constant_wls(dtm: pd.Series, dts: pd.Series) -> float:
    """Closed-form weighted-LS scalar alpha. Returns 0 for degenerate
    signal (every dtm is zero).
    """
    num = float((dtm * dts).sum())
    den = float((dtm ** 2).sum())
    return num / den if den > 0 else 0.0


def fit_alpha(skew_csv: Path) -> dict[str, Any]:
    """Read one TP's ``skew.csv`` and return the per-bucket alpha fit.

    ``pc`` and ``lev`` carry fixed edges; ``n`` is derived from this CSV's own
    values (see ``_derive_bucket_axes``), so the fit follows the sweep when it
    widens to a larger ``max_num_seqs``.

    Returns a dict with:
        method: "per_bucket_median_3axis_n_pc_lev"
        n_samples: total rows used
        alpha_default: pooled WLS constant over every row (legacy scalar)
        alpha_default_by_layer: {layer -> pooled WLS constant for that kernel}
        bucket_axes: bin edges + labels derived from this TP's data
        alpha_by_bucket: {bucket_key -> alpha}, keys prefixed by layer
        n_by_bucket: {bucket_key -> samples_in_bucket}
        rel_err_p50/p90/p99: self-evaluation on the per-bucket prediction
        signed_mean: mean signed error (positive = over-predict)

    The fit is **per attention kernel**. Pooling them would average alphas
    that describe different work: MiniMax-M3's sparse layers stop tracking kv
    length past their block budget, so their endpoint gap collapses, while its
    indexer scans the whole KV. The fallback ``alpha_default`` is per layer for
    the same reason -- an unfitted sparse bucket must not inherit the
    non-sparse kernel's alpha. ``alpha_default`` stays as a pooled scalar so a
    meta.yaml written by this version still reads on an older simulator.
    """
    if not skew_csv.exists():
        return {"enabled": False, "reason": "skew.csv missing"}
    df = pd.read_csv(skew_csv).dropna(subset=["alpha"])
    if len(df) == 0:
        return {"enabled": False, "reason": "no valid rows"}

    dtm = df["t_max_us"] - df["t_mean_us"]
    dts = df["t_skew_us"] - df["t_mean_us"]

    alpha_default = _fit_constant_wls(dtm, dts)

    axes = _derive_bucket_axes(df)

    # Bucket fit on (n-bin, pc-bin, lev-bin). ``lev`` comes from the row's own
    # two timings, so nothing new has to be swept for it.
    lev_col = (df["t_max_us"] - df["t_mean_us"]) / df["t_mean_us"]

    # A bundle profiled before the layer column existed holds one kernel, and
    # it is the one the catalog calls ``attention``. ``fillna`` before
    # ``astype(str)``, or a blank cell becomes the string "nan" and gets its
    # own alpha table.
    layers = (df["layer"].fillna("attention").replace("", "attention").astype(str)
              if "layer" in df.columns
              else pd.Series(["attention"] * len(df), index=df.index))
    keys = [
        _bucket_key(axes, r.n, r.pc, lev, layer)
        for r, lev, layer in zip(df.itertuples(index=False), lev_col, layers)
    ]
    df = df.assign(_bk=keys, _layer=layers.values)

    alpha_default_by_layer = {
        str(layer): round(
            _fit_constant_wls(
                grp["t_max_us"] - grp["t_mean_us"],
                grp["t_skew_us"] - grp["t_mean_us"],
            ),
            4,
        )
        for layer, grp in df.groupby("_layer", sort=True)
    }

    alpha_by_bucket: dict[str, float] = {}
    n_by_bucket: dict[str, int] = {}
    for bk, grp in df.groupby("_bk", sort=True):
        if len(grp) < _MIN_BUCKET_ROWS:
            continue
        dtm_g = grp["t_max_us"] - grp["t_mean_us"]
        dts_g = grp["t_skew_us"] - grp["t_mean_us"]
        # The **median** of the per-row alphas, not the weighted-LS optimum.
        # WLS is the right objective when the noise on ``dts`` is
        # homoscedastic: the charged error is ``(a - a*) * dtm``, so weighting
        # the residual by ``dtm^2`` is optimal. But ``dts`` is the difference
        # of two nearly-equal measurements, so its noise scales with
        # ``t_mean`` -- which grows with the batch and correlates with
        # ``dtm`` -- and under that noise model ``dtm^2`` over-trusts the few
        # largest-gap rows, which are exactly where one noisy shot dominates.
        # Measured both ways through the whole pipeline, median wins on the
        # sweep's own rows (Llama rel_err_p50 0.0588 -> 0.0308) *and* end to
        # end (0.87% -> 0.42% mean |err| over 15 metrics; 4.89 -> 4.66 summed
        # over the three committed examples). Both figures are the estimator
        # A/B at the ``n`` bins of the time -- the shipped fit's own self-eval
        # is 0.0259.
        a = float((dts_g / dtm_g).median())
        alpha_by_bucket[bk] = round(
            max(_ALPHA_CLIP[0], min(_ALPHA_CLIP[1], a)), 4)
        n_by_bucket[bk] = int(len(grp))

    # Self-eval. A row whose own bucket did not survive the support floor is
    # scored against the layer's pooled alpha, the value the simulator will
    # read for it.
    predicted = df["_bk"].map(alpha_by_bucket)
    predicted = predicted.fillna(
        df["_layer"].map(alpha_default_by_layer)).fillna(alpha_default)
    pred_t = df["t_mean_us"] + predicted * (df["t_max_us"] - df["t_mean_us"])
    abs_err = ((pred_t - df["t_skew_us"]).abs() / df["t_skew_us"]).dropna()
    signed = ((pred_t - df["t_skew_us"]) / df["t_skew_us"]).dropna()

    out: dict[str, Any] = {
        "method": "per_bucket_median_3axis_n_pc_lev",
        "n_samples": int(len(df)),
        "alpha_default": round(alpha_default, 4),
        "alpha_default_by_layer": alpha_default_by_layer,
        "bucket_axes": axes,
        "alpha_by_bucket": alpha_by_bucket,
        "n_by_bucket": n_by_bucket,
    }
    if len(abs_err):
        out["rel_err_p50"] = round(float(abs_err.quantile(0.50)), 4)
        out["rel_err_p90"] = round(float(abs_err.quantile(0.90)), 4)
        out["rel_err_p99"] = round(float(abs_err.quantile(0.99)), 4)
        out["signed_mean"] = round(float(signed.mean()), 4)

    return out


def _tp_dirs_with_skew(variant_root: Path) -> list[int]:
    """Every TP degree under ``variant_root`` that actually holds skew data."""
    found: list[int] = []
    for d in variant_root.glob("tp*"):
        if not d.is_dir() or not (d / "skew.csv").exists():
            continue
        try:
            found.append(int(d.name[2:]))
        except ValueError:
            continue
    return sorted(found)


def fit_alpha_per_tp(
    variant_root: Path, tp_degrees: list[int] | None = None
) -> dict[str, Any]:
    """Walk every ``tp{N}/skew.csv`` under ``variant_root`` and fit.

    The block this feeds describes the **whole variant**, and the writer
    replaces it wholesale, so it has to cover every TP the bundle holds --
    not just the one this run happened to profile. Iterating the caller's
    ``tp_degrees`` instead silently deleted the others: a ``--tp 1`` refresh
    dropped ``skew_fit.per_tp[2]`` from meta.yaml while ``tp2/skew_fit.csv``
    sat on disk, and ``_skew_alpha`` then found no entry for tp=2 and fell
    back to ``_ATTN_SKEW_ALPHA_FALLBACK`` -- i.e. turned skew correction off
    for every tp=2 run. ``tp_degrees`` is kept only as a hint, unioned in, so
    a caller naming a TP whose CSV is missing still gets the same silent skip
    as before.

    Returns a meta-friendly dict with a ``per_tp`` map. TPs whose skew.csv is
    absent or empty are silently skipped. Each TP's ``bucket_axes`` is derived
    from its own data; the writer then dedups and promotes them to the block
    top-level when they match across TPs (which they usually do, since all TPs
    share the same profile grid).
    """
    wanted = sorted(set(_tp_dirs_with_skew(variant_root))
                    | {int(t) for t in (tp_degrees or ())})
    per_tp: dict[int, dict[str, Any]] = {}
    for tp in wanted:
        fit = fit_alpha(variant_root / f"tp{tp}" / "skew.csv")
        if fit.get("enabled") is False:
            continue
        per_tp[int(tp)] = fit
    if not per_tp:
        return {"enabled": False}
    return {"enabled": True, "per_tp": per_tp}


# Lookup helper so the simulator (or anyone else) can reuse the
# bucket mapping without re-implementing _bucket_key.
def lookup_alpha(
    fit_block: dict[str, Any],
    tp: int,
    pc: int,
    n: int,
    lev: float,
    layer: str = "attention",
) -> float:
    """Resolve alpha for a specific batch from a ``skew_fit`` block.

    ``lev`` is ``(t_max - t_mean) / t_mean``, so the caller must have both
    endpoint lookups in hand -- which it does, since the blend needs them
    anyway. A batch with no skew (``kv_max == kv_min``) should short-circuit
    before calling this: its lever is 0 and there is nothing to correct.

    Uses ``bucket_axes`` from the fit_block when present (preferred, since the
    profiler stores the axes it actually used); falls back to the module-level
    defaults otherwise.
    """
    axes = fit_block.get("bucket_axes") or default_bucket_axes()
    per_tp = fit_block.get("per_tp", {})
    entry = per_tp.get(tp) or per_tp.get(int(tp))
    if not entry:
        return 0.0
    # Per-TP axes override when present (pre-promotion fit blocks).
    axes = entry.get("bucket_axes", axes)
    alphas = entry.get("alpha_by_bucket", {})
    key = _bucket_key(axes, n, pc, lev, layer)
    if key in alphas:
        return float(alphas[key])
    # A bundle profiled before the layer prefix existed holds unprefixed keys,
    # all of them the ``attention`` kernel. Reading them for any other layer
    # would hand a sparse kernel the dense one's alpha, which is the bug this
    # prefix exists to prevent -- so only ``attention`` falls back.
    if layer == "attention":
        bare = _bucket_key(axes, n, pc, lev)
        if bare in alphas:
            return float(alphas[bare])
    per_layer = entry.get("alpha_default_by_layer") or {}
    if layer in per_layer:
        return float(per_layer[layer])
    if layer == "attention":
        return float(entry.get("alpha_default", 0.0))
    # No data for this kernel: no correction, matching the SKIP_SKEW default.
    return 0.0
