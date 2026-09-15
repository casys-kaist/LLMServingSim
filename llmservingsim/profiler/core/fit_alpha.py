"""Fit a per-bucket alpha from ``skew.csv``.

The skew case measures three latencies at the same operating point:

    t_mean   all decodes uniform at the mean kv
    t_max    all decodes uniform at the max kv
    t_skew   the actual skewed batch (nb at kv_big, n-nb at kv_small)

The simulator uses alpha at lookup time via

    t_predicted = t_mean + alpha * (t_max - t_mean)

Empirically (on the widened Qwen3-32B / RTXPRO6000 sweep with ~13k
samples per TP) per-batch alpha has a real 5-axis structure:
``pc`` (prefill chunk), ``n`` (total decodes), ``skew_rate``
(normalised heavy-fraction in the batch), ``kv_big`` (the per-batch
max decode kv), and ``kp`` (prefill history length). The axes capture,
respectively, the compute-shape regime (pc, n), how the skew is
distributed (skew_rate), how much the outlier decode stretches the
kernel's tile padding / SM-imbalance behaviour (kv_big), and the
interaction between pre-loaded prefill KV and the decode-skew term
(kp).

Axis ablation (5-fold CV on the widened data, test p50 / p90 / p99 /
mean-signed-error):

    TP=1:
        α = 0                        4.1 / 17.9 / 35.7 / -0.062   (old)
        global constant              3.9 / 19.2 / 34.4 / -0.008
        per-(pc, n_bin)              3.5 / 16.4 / 39.9 / +0.013
        per-(pc, n_bin, kv_big_bin)  2.9 / 16.1 / 49.2 / +0.005   (4-axis)
        per-(pc, n_bin, kv_big_bin,
             skew_rate_bin, kp_bin)  2.7 / 14.8 / 44.1 / +0.003   ← chosen (5-axis)

Within a bucket the fitted constant is the weighted-LS optimum:

    alpha_hat(bin) = argmin_a sum_{i in bin} (a*dtm_i - dts_i)^2
                   = sum(dtm*dts) / sum(dtm^2)

which naturally down-weights noise-dominated points (small dtm) and
up-weights signal-bearing ones.

A pooled constant (``alpha_default``) is the fallback for buckets
outside the fitted table.

---------------------------------------------------------------------
Data-driven bucket axes
---------------------------------------------------------------------
The bin edges for ``n``, ``kv_big``, and ``kp`` are derived from the
skew.csv actually present on disk, not hard-coded. This means:

    * ``n`` gets one bucket per unique profiled value (plus an
      overflow ``n>{max}`` for runtime batches beyond the sweep).
    * ``kp`` gets one bucket per unique profiled value, starting from
      the ``kp=0`` sentinel.
    * ``kv_big`` uses a log-4x doubling scheme extended to the
      observed maximum, since ``kv_big = kvs * skew`` varies
      continuously and one-bin-per-value would fragment too finely.

``skew_rate`` remains fixed (it's a normalised [0, 1] metric — its
bucket boundaries are a judgment call about how to slice the
distribution, not a range-coverage question). ``pc`` is not bucketed
at all: every profiled grid point becomes its own alpha column.

The derived axes are written to ``meta.yaml::skew_fit.bucket_axes``
and the simulator reads them from there, so widening the profile
sweep (e.g. ``max_num_seqs`` to 512 or ``attention_max_kv`` to 65536)
lights up proper resolution on the affected axis without any code
change in either component.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd


# skew_rate bucketisation — fixed because the axis is normalised to
# [0, 1] regardless of profile sweep width. ``skew_rate`` is defined
# as
#
#     skew_rate = (kv_mean - kv_min) / (kv_max - kv_min)
#
# For the profiler's bimodal cases this equals ``nb / n`` exactly
# (verified to 0 error). For runtime continuous distributions it's a
# smooth "effective heavy-fraction" metric. Small skew_rate means few
# sequences near the max (heavy tail); large means most sequences are
# near the max. The five bins below are a judgment call about how to
# slice that distribution for alpha-fitting granularity.
_SKEW_RATE_BINS = (-0.01, 0.05, 0.15, 0.40, 0.70, 1.01)
_SKEW_RATE_LABELS = ("sr<=5%", "sr<=15%", "sr<=40%", "sr<=70%", "sr>70%")

# ---------------------------------------------------------------------------
# Default bucket axes — used as a fallback when no data is available
# (e.g. ``lookup_alpha`` called without a populated fit block) and for
# documentation of the "typical" bucket shape the simulator used to
# hard-code. ``fit_alpha`` overwrites these at fit time with values
# derived from skew.csv so the shipped bucket_axes always match the
# profiled data.
# ---------------------------------------------------------------------------
_DEFAULT_N_BINS = (0, 2, 4, 8, 16, 32, 64, 128, 1_000_000)
_DEFAULT_N_LABELS = (
    "n<=2", "n<=4", "n<=8", "n<=16", "n<=32", "n<=64", "n<=128", "n>128",
)
_DEFAULT_KV_BIG_BINS = (0, 1024, 4096, 16384, 1_000_000_000)
_DEFAULT_KV_BIG_LABELS = ("kvB<=1k", "kvB<=4k", "kvB<=16k", "kvB>16k")
_DEFAULT_KP_BINS = (-1, 0, 2048, 1_000_000_000)
_DEFAULT_KP_LABELS = ("kp=0", "kp<=2k", "kp>2k")


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
    """One bucket per profiled ``n`` value + an overflow bin.

    Example: profile grid ``[2, 4, 8, 16, 32, 64, 128, 256]``
      → bins   ``(0, 2, 4, 8, 16, 32, 64, 128, 256, 1e9)``
      → labels ``('n<=2', 'n<=4', ..., 'n<=256', 'n>256')``

    The overflow bin lets the simulator fall back cleanly for runtime
    batches larger than anything the profiler fired (an edge case —
    usually ``max_num_seqs`` is the same in both).
    """
    vals = sorted(int(v) for v in df["n"].dropna().unique())
    if not vals:
        return _DEFAULT_N_BINS, _DEFAULT_N_LABELS
    bins = (0,) + tuple(vals) + (1_000_000,)
    labels = tuple(f"n<={v}" for v in vals) + (f"n>{vals[-1]}",)
    return bins, labels


def _derive_kp_axis(df: pd.DataFrame) -> tuple[tuple, tuple]:
    """``kp=0`` sentinel + one bucket per profiled non-zero kp value
    + an overflow.

    Example: profile grid ``[0, 512, 1024, 2048, 4096, 8192]``
      → bins   ``(-1, 0, 512, 1024, 2048, 4096, 8192, 1e9)``
      → labels ``('kp=0', 'kp<=512', 'kp<=1k', ..., 'kp<=8k', 'kp>8k')``
    """
    vals = sorted(int(v) for v in df["kp"].dropna().unique() if v > 0)
    if not vals:
        return _DEFAULT_KP_BINS, _DEFAULT_KP_LABELS
    bins = (-1, 0) + tuple(vals) + (1_000_000_000,)
    labels = ("kp=0",) + tuple(f"kp<={_short_kv(v)}" for v in vals) + (
        f"kp>{_short_kv(vals[-1])}",
    )
    return bins, labels


def _derive_kv_big_axis(df: pd.DataFrame) -> tuple[tuple, tuple]:
    """Log-4x doublings from 1024 up to the observed max.

    ``kv_big = kvs * skew`` takes many distinct values (roughly
    cartesian-product over the kvs and skew grids), so a per-value
    scheme would fragment into dozens of cells with insufficient
    samples each. Log-4x (1k, 4k, 16k, 64k, …) keeps the cell count
    manageable while adapting the ceiling to whatever the profile
    actually covered.
    """
    if "kv_big" not in df.columns or df["kv_big"].dropna().empty:
        return _DEFAULT_KV_BIG_BINS, _DEFAULT_KV_BIG_LABELS
    kv_max = int(df["kv_big"].dropna().max())
    edges: list[int] = []
    v = 1024
    while v <= kv_max:
        edges.append(v)
        v *= 4
    if not edges:
        edges = [1024]
    # Ensure the topmost edge is at least the observed max so every
    # sample fits in a non-overflow bin (the overflow bin then only
    # catches runtime values beyond the sweep).
    if edges[-1] < kv_max:
        edges.append(kv_max)
    bins = (0,) + tuple(edges) + (1_000_000_000,)
    labels = tuple(f"kvB<={_short_kv(e)}" for e in edges) + (
        f"kvB>{_short_kv(edges[-1])}",
    )
    return bins, labels


def _derive_bucket_axes(df: pd.DataFrame) -> dict[str, Any]:
    """Pick bin edges + labels for every axis from the skew.csv data.

    Returned dict is flat-YAML-friendly and mirrors the structure the
    simulator reads from ``meta.yaml::skew_fit.bucket_axes``.
    """
    n_bins, n_labels = _derive_n_axis(df)
    kv_bins, kv_labels = _derive_kv_big_axis(df)
    kp_bins, kp_labels = _derive_kp_axis(df)
    return {
        "pc": "raw pc value (profiled grid point)",
        "n_bins": list(n_bins),
        "n_labels": list(n_labels),
        "skew_rate_bins": list(_SKEW_RATE_BINS),
        "skew_rate_labels": list(_SKEW_RATE_LABELS),
        "kv_big_bins": list(kv_bins),
        "kv_big_labels": list(kv_labels),
        "kp_bins": list(kp_bins),
        "kp_labels": list(kp_labels),
    }


def _bucket_key(axes: Mapping[str, Any], pc, n, skew_rate, kv_big, kp) -> str:
    """Stringified 5-axis key used in the per-TP skew_fit CSV. Format is
    ``pc={pc}|{n_label}|{skew_rate_label}|{kv_big_label}|{kp_label}``.

    ``axes`` is any mapping that carries the ``*_bins`` / ``*_labels``
    tuples. At fit time these come from ``_derive_bucket_axes``; at
    simulator lookup time they come from
    ``meta.yaml::skew_fit.bucket_axes``; both places agree on the
    label strings so the CSV rows resolve cleanly either way.
    """
    n_label = _bucket_label(
        tuple(axes["n_bins"]), tuple(axes["n_labels"]), int(n),
    )
    sr = max(0.0, min(1.0, float(skew_rate)))
    sr_label = _bucket_label(
        tuple(axes["skew_rate_bins"]), tuple(axes["skew_rate_labels"]), sr,
    )
    kvb_label = _bucket_label(
        tuple(axes["kv_big_bins"]), tuple(axes["kv_big_labels"]), int(kv_big),
    )
    kp_label = _bucket_label(
        tuple(axes["kp_bins"]), tuple(axes["kp_labels"]), int(kp),
    )
    return f"pc={int(pc)}|{n_label}|{sr_label}|{kvb_label}|{kp_label}"


# Backward-compatible alias — some external callers (or older tests)
# may still import the default-axes version.
def default_bucket_axes() -> dict[str, Any]:
    """Hard-coded axes used when no profile data is available."""
    return {
        "pc": "raw pc value (profiled grid point)",
        "n_bins": list(_DEFAULT_N_BINS),
        "n_labels": list(_DEFAULT_N_LABELS),
        "skew_rate_bins": list(_SKEW_RATE_BINS),
        "skew_rate_labels": list(_SKEW_RATE_LABELS),
        "kv_big_bins": list(_DEFAULT_KV_BIG_BINS),
        "kv_big_labels": list(_DEFAULT_KV_BIG_LABELS),
        "kp_bins": list(_DEFAULT_KP_BINS),
        "kp_labels": list(_DEFAULT_KP_LABELS),
    }


def _fit_constant_wls(dtm: pd.Series, dts: pd.Series) -> float:
    """Closed-form weighted-LS scalar alpha. Returns 0 for degenerate
    signal (every dtm is zero).
    """
    num = float((dtm * dts).sum())
    den = float((dtm ** 2).sum())
    return num / den if den > 0 else 0.0


def fit_alpha(skew_csv: Path) -> dict[str, Any]:
    """Read one TP's ``skew.csv`` and return the per-bucket alpha fit.

    Bucket axes are derived from the data (see ``_derive_bucket_axes``)
    so the fit automatically adapts when the profile sweep widens on
    the ``n``, ``kv_big``, or ``kp`` axis.

    Returns a dict with:
        method: "per_bucket_wls_5axis"
        n_samples: total rows used
        alpha_default: pooled WLS constant (used for bins with no data)
        bucket_axes: bin edges + labels derived from this TP's data
        alpha_by_bucket: {bucket_key -> alpha}
        n_by_bucket: {bucket_key -> samples_in_bucket}
        rel_err_p50/p90/p99: self-evaluation on the per-bucket prediction
        signed_mean: mean signed error (positive = over-predict)
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

    # Bucket fit on (pc, n-bin, skew_rate-bin, kv_big-bin, kp-bin).
    # skew_rate = (kv_mean - kv_min) / (kv_max - kv_min), exactly
    # ``nb / n`` for bimodal profiler cases.
    kv_min_col = df["kvs"] if "kvs" in df.columns else df["kv_mean"]
    gap = (df["kv_big"] - kv_min_col).clip(lower=1)
    skew_rate_col = (df["kv_mean"] - kv_min_col) / gap

    keys = [
        _bucket_key(axes, r.pc, r.n, sr, r.kv_big, r.kp)
        for r, sr in zip(df.itertuples(index=False), skew_rate_col)
    ]
    df = df.assign(_bk=keys)

    alpha_by_bucket: dict[str, float] = {}
    n_by_bucket: dict[str, int] = {}
    for bk, grp in df.groupby("_bk", sort=True):
        a = _fit_constant_wls(
            grp["t_max_us"] - grp["t_mean_us"],
            grp["t_skew_us"] - grp["t_mean_us"],
        )
        alpha_by_bucket[bk] = round(a, 4)
        n_by_bucket[bk] = int(len(grp))

    # Self-eval.
    predicted = df["_bk"].map(alpha_by_bucket)
    pred_t = df["t_mean_us"] + predicted * (df["t_max_us"] - df["t_mean_us"])
    abs_err = ((pred_t - df["t_skew_us"]).abs() / df["t_skew_us"]).dropna()
    signed = ((pred_t - df["t_skew_us"]) / df["t_skew_us"]).dropna()

    out: dict[str, Any] = {
        "method": "per_bucket_wls_5axis",
        "n_samples": int(len(df)),
        "alpha_default": round(alpha_default, 4),
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


def fit_alpha_per_tp(
    variant_root: Path, tp_degrees: list[int]
) -> dict[str, Any]:
    """Walk every ``tp{N}/skew.csv`` under ``variant_root`` and fit.

    Returns a meta-friendly dict with a ``per_tp`` map. TPs whose
    skew.csv is absent or empty are silently skipped. Each TP's
    ``bucket_axes`` is derived from its own data; the writer then
    dedups and promotes them to the block top-level when they match
    across TPs (which they usually do, since all TPs share the same
    profile grid).
    """
    per_tp: dict[int, dict[str, Any]] = {}
    for tp in tp_degrees:
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
    skew_rate: float,
    kv_big: int,
    kp: int,
) -> float:
    """Resolve alpha for a specific batch from a ``skew_fit`` block.

    Uses ``bucket_axes`` from the fit_block when present (preferred,
    since the profiler stores the axes it actually used); falls back
    to the module-level defaults otherwise. Runtime batches with zero
    skew (``kv_max == kv_min``) should short-circuit before calling
    this — a ``skew_rate`` of NaN / inf will still produce a valid key
    (snapped to the edge bin) but no skew correction should be applied
    in that case.
    """
    axes = fit_block.get("bucket_axes") or default_bucket_axes()
    per_tp = fit_block.get("per_tp", {})
    entry = per_tp.get(tp) or per_tp.get(int(tp))
    if not entry:
        return 0.0
    # Per-TP axes override when present (pre-promotion fit blocks).
    axes = entry.get("bucket_axes", axes)
    key = _bucket_key(axes, pc, n, skew_rate, kv_big, kp)
    alphas = entry.get("alpha_by_bucket", {})
    if key in alphas:
        return float(alphas[key])
    return float(entry.get("alpha_default", 0.0))
