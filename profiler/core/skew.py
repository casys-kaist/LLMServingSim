"""Skew profiling — measures how FlashAttention kernel cost shifts when
a decode batch has non-uniform kv distributions.

Unlike the uniform attention grid (all decodes at the same kv), each
skew case measures three latencies at the same operating point:

    t_mean   — all decodes uniform at the batch's mean kv
    t_max    — all decodes uniform at the batch's max kv
    t_skew   — the actual skewed batch [nb × kv_big, (n-nb) × kvs]

From these we can compute the normalized interpolation factor

    alpha = (t_skew - t_mean) / (t_max - t_mean)

which the simulator uses at query time:

    t_predicted = t_mean_lookup(batch.mean_kv) + alpha(batch.shape) ×
                  (t_max_lookup(batch.max_kv) - t_mean_lookup(batch.mean_kv))

Output: ``<variant>/tp<N>/skew.csv`` with per-case columns
``regime, n, nb, ratio, skew, pc, kp, kvs, kv_big, kv_mean,
t_mean_us, t_max_us, t_skew_us, alpha``.

Downstream pipeline:

  * ``fit_alpha`` reads each TP's skew.csv, derives bucket axes from
    the observed (n, kv_big, kp) coverage (so widening the sweep
    automatically lights up more resolution), and emits a 5-axis
    weighted-LS fit.
  * ``writer.persist_meta`` spills the fitted (bucket → alpha) table
    to ``<variant>/tp<N>/skew_fit.csv`` and records only a per-TP
    summary + the derived ``bucket_axes`` under
    ``meta.yaml::skew_fit``.
  * At query time the simulator reads ``bucket_axes`` from meta.yaml
    and reconstructs the same bucket key for whatever runtime batch
    it's evaluating.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from profiler.core import logger as log
from profiler.core.config import Architecture, ProfileArgs
from profiler.core.hooks.batch import Shot
from profiler.core.hooks.timings import TimingSample


# ---------------------------------------------------------------------------
# Grid design
# ---------------------------------------------------------------------------
#
# The grid axes are generated from the CLI-provided caps + per-axis
# geometric factors so that the skew sweep scales with each model /
# hardware target and can be coarsened for quick runs:
#
#   n axis      — geometric from 2 up to ``max_num_seqs``
#                 (``--skew-n-factor``, default 2.0 = doubling)
#   ratio axis  — fixed (unitless shape characterization)
#   pc axis     — {0} + geometric from 16 up to
#                 ``max_num_batched_tokens``. Starting at 16 gives
#                 dense sampling in the small-chunk regime where skew
#                 effects are strongest. (``--skew-pc-factor``)
#   kp axis     — {0} + geometric from 512 up to
#                 ``attention_max_kv // 2`` so prefill + existing-KV
#                 interactions get swept across short (512) to long
#                 histories. (``--skew-kp-factor``)
#   kvs axis    — geometric in [128, attention_max_kv]
#                 (``--skew-kvs-factor``)
#   skew axis   — fixed {2, 4, 8, 16} (physical saturation-curve fit)
#
# Bumping any factor above 2.0 coarsens that axis (faster profile,
# wider interpolation gaps); dropping below 2.0 densifies it.
#
# Tier 1 is a factorial of the above at a single representative skew
# (4.0, in the saturated regime). Tier 2 adds a sweep along the skew
# axis at a few anchor points — the only source of skew ≠ 4.0 data
# in the dataset. (A former Tier 3 for the kvs axis was removed once
# T1 grew dense enough along kvs to make T3 fully redundant.)

_RATIO_VALS = (0.0625, 0.125, 0.25, 0.5, 0.75, 0.9)
# Absolute nb counts that should be probed for every n, regardless of
# what the fractional ``_RATIO_VALS`` produce. Needed because
# ratio-based sampling at large n skips the "a few heavy outliers"
# regime entirely — e.g. n=128 with ratio=0.0625 already gives nb=8,
# never touching nb=1..4 where alpha peaks physically.
_NB_ABSOLUTE = (1, 2, 3, 4)
_SKEW_REP   = 4.0
# kp axis grows from ``_KP_START`` up to ``AMK // 2`` so short
# histories (512, 1024) get representation alongside the longer ones.
_KP_START = 512

# Tier 2 skew sweep — physical values, fixed.
_T2_SKEW_MIXED = [1.5, 2.0, 4.0, 8.0, 16.0]
_T2_SKEW_PURE  = [2.0, 4.0, 8.0, 16.0]


def _doubling(start: int, max_val: int, factor: float = 2.0) -> list[int]:
    """Geometric grid from ``start`` up to ``max_val`` inclusive.

    ``factor=2.0`` (the default) gives the classic doubling sequence.
    Higher factors coarsen the grid (fewer samples, faster profile);
    lower factors densify it. Adjacent values that round to the same
    integer are deduplicated (can happen for factors close to 1.0 at
    small scales). ``max_val`` is always appended when it isn't
    already on the grid so the top of the sweep is never missed.
    """
    if factor <= 1.0:
        raise ValueError(f"factor must be > 1.0; got {factor}")
    if max_val < start:
        return [max_val] if max_val > 0 else []
    vals: list[int] = []
    v: float = float(start)
    while True:
        iv = int(round(v))
        if iv > max_val:
            break
        if not vals or iv != vals[-1]:
            vals.append(iv)
        v *= factor
    if vals and vals[-1] != max_val:
        vals.append(max_val)
    return vals


def _build_grid(args: ProfileArgs, limits) -> dict:
    """Generate dynamic axis values from user-set envelopes and the
    per-axis geometric factors on ``args``.
    """
    AMK = args.attention_max_kv
    MNBT = args.max_num_batched_tokens or limits.max_num_batched_tokens
    MSQ = args.max_num_seqs or limits.max_num_seqs

    n_vals = _doubling(
        2, min(MSQ, limits.max_num_seqs), factor=args.skew_n_factor,
    )
    if not n_vals:
        n_vals = [1]

    # pc: start from 16 to cover the small-chunk regime where
    # decode-tile imbalance is the dominant term.
    pc_vals = [0] + _doubling(16, MNBT, factor=args.skew_pc_factor)
    pc_vals = sorted(set(pc_vals))

    # kp: 0 plus geometric from 512 up to AMK // 2 so both short (512)
    # and long histories get representation. E.g. AMK=16384, factor=2
    # → [0, 512, 1024, 2048, 4096, 8192].
    kp_vals = [0]
    kp_cap = max(_KP_START, AMK // 2)
    kp_vals.extend(_doubling(_KP_START, kp_cap, factor=args.skew_kp_factor))
    kp_vals = sorted(set(kp_vals))

    # kvs: geometric across the full AMK range.
    kvs_vals = _doubling(128, max(512, AMK), factor=args.skew_kvs_factor)

    return {
        "n": tuple(n_vals),
        "ratio": _RATIO_VALS,
        "pc": tuple(pc_vals),
        "kp": tuple(kp_vals),
        "kvs": tuple(kvs_vals),
    }


def _tier2_pivots(grid: dict) -> list[tuple]:
    """Skew-sweep anchors derived from the current grid.

    Pick pivots that span both regimes (pure pc=0, mixed pc>0) and
    both extreme and balanced ratios. Uses first-kvs as the anchor
    so the skew axis sweep is comparable across tier 2 entries.
    """
    if not grid["n"] or not grid["pc"] or not grid["kvs"]:
        return []
    pcmid = grid["pc"][min(1, len(grid["pc"]) - 1)]
    kvsmid = grid["kvs"][0]
    # Two mid-sized n's if available
    n_samples = []
    for i in (1, 2):
        if i < len(grid["n"]):
            n_samples.append(grid["n"][i])
    if not n_samples:
        n_samples = [grid["n"][0]]
    pivots: list[tuple] = []
    for n in n_samples[:2]:
        pivots.append((n, 0.125, pcmid, 0, kvsmid, _T2_SKEW_MIXED))
    if len(n_samples) >= 1:
        pivots.append((n_samples[0], 0.5, pcmid, 0, kvsmid, _T2_SKEW_MIXED[:3]))
    # Pure-regime pivot (pc=0)
    if 0 in grid["pc"]:
        pivots.append((n_samples[0], 0.125, 0, 0, kvsmid, _T2_SKEW_PURE))
    return pivots



# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------

@dataclass
class SkewCase:
    n: int
    ratio: float
    skew: float
    pc: int
    kp: int
    kvs: int
    # Derived:
    nb: int = 0
    kv_big: int = 0
    kv_mean: int = 0

    def __post_init__(self):
        self.nb = max(1, min(self.n - 1, int(round(self.ratio * self.n))))
        self.kv_big = int(round(self.kvs * self.skew))
        total_kv = self.nb * self.kv_big + (self.n - self.nb) * self.kvs
        self.kv_mean = total_kv // self.n


def _feasible(case: SkewCase, limits, args: ProfileArgs) -> bool:
    """Feasibility filter — mirrors the attention category's checks.

    Since the shot bypasses the vLLM scheduler, MNBT is advisory:
    ``pc + n`` is allowed up to ``MNBT + MSQ`` so pc=MNBT can still
    pair with the full n-axis range. The hard caps are the max_seqs
    buffer (strict ``n_seqs < MSQ``), the per-request position bound
    (``pos + 1 > MML``), and the KV-cache block budget.
    """
    AMK = args.attention_max_kv
    MNBT = args.max_num_batched_tokens or limits.max_num_batched_tokens
    MSQ = limits.max_num_seqs
    if case.pc > MNBT + 1: return False
    if case.kvs > AMK + 1: return False
    if case.kv_big > AMK + 1: return False
    if case.kp > AMK + 1: return False
    # Sum bound, mirrored from AttentionCategory: the shot bypasses the
    # vLLM scheduler via assemble_scheduler_output, so MNBT is advisory.
    # Allow pc + n up to MNBT + MSQ so pc=MNBT can still pair with the
    # full n-axis range. The strict n_seqs < MSQ check further down is
    # the hard cap that actually protects vLLM's input_batch buffer.
    if case.pc + case.n > MNBT + MSQ: return False

    MML = limits.max_model_len
    if case.kv_big + 1 > MML: return False
    if case.pc > 0 and case.pc + case.kp + 1 > MML: return False
    # n_seqs vs max_num_seqs — mirrored from AttentionCategory so the
    # skew sweep can fire the same (n = MSQ) corner the uniform grid
    # reaches. vLLM V1's ``input_batch`` pre-allocation sizes to MSQ
    # sequences, so MSQ itself fits; MSQ+1 overflows. Pure-regime
    # n=MSQ therefore fires; mixed-regime n=MSQ (which would need
    # MSQ+1 requests with the prefill case) is still filtered.
    n_seqs = case.n + (1 if case.pc > 0 else 0)
    if n_seqs > limits.max_num_seqs: return False

    BS = 16
    def aligned(t): return ((t + BS - 1) // BS) * BS
    big_blk = aligned(case.kv_big) * case.nb
    small_blk = aligned(case.kvs) * (case.n - case.nb)
    pfx_blk = aligned(case.pc + case.kp) if case.pc > 0 else 0
    if big_blk + small_blk + pfx_blk > limits.num_cache_tokens:
        return False
    return True


def _build_cases(args: ProfileArgs, limits) -> list[SkewCase]:
    """Compose cases across three tiers. All axis values derive from
    ``args`` + ``limits`` via ``_build_grid``.
    """
    grid = _build_grid(args, limits)
    cases: list[SkewCase] = []

    # Tier 1 — factorial at representative skew.
    # Per-n, the effective ratio list is the fractional ``_RATIO_VALS``
    # plus ``nb_abs / n`` for each nb_abs in _NB_ABSOLUTE (dedup'd).
    # This guarantees the "few heavy outliers" regime (nb=1..4) is
    # measured for every batch size, not just for small n where the
    # fractional ratios happen to collapse to nb≈1.
    for n in grid["n"]:
        ratios_for_n = set(grid["ratio"])
        for nb_abs in _NB_ABSOLUTE:
            if 1 <= nb_abs < n:
                ratios_for_n.add(nb_abs / n)
        for r in sorted(ratios_for_n):
            for pc in grid["pc"]:
                for kp in grid["kp"]:
                    if pc == 0 and kp != 0:
                        continue
                    for kvs in grid["kvs"]:
                        c = SkewCase(n=n, ratio=r, skew=_SKEW_REP,
                                     pc=pc, kp=kp, kvs=kvs)
                        if _feasible(c, limits, args):
                            cases.append(c)

    # Tier 2 — skew-axis sweep at a few pivots. This is the only
    # source of skew ≠ _SKEW_REP samples in the dataset; keep even
    # though T1 covers the skew=4 baseline.
    for (n, r, pc, kp, kvs, skews) in _tier2_pivots(grid):
        for sk in skews:
            c = SkewCase(n=n, ratio=r, skew=sk, pc=pc, kp=kp, kvs=kvs)
            if _feasible(c, limits, args):
                cases.append(c)

    # Dedup (T1 + T2 can overlap at skew=4)
    seen = set()
    uniq: list[SkewCase] = []
    for c in cases:
        key = (c.n, c.nb, round(c.skew, 3), c.pc, c.kp, c.kvs)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _measure(llm, reqs, slice_, iters: int) -> float:
    shot = Shot(requests=[tuple(r) for r in reqs])
    raw = llm.collective_rpc(
        "fire", args=(shot.as_dict(), slice_, "attention", iters))
    timings = [TimingSample(layer=d["layer"],
                            microseconds=float(d["microseconds"]))
               for d in raw[0]]
    return sum(t.microseconds for t in timings if t.layer == "attention")


def _measure_case(llm, case: SkewCase, slice_, iters: int) -> dict:
    base = [(case.pc, case.kp)] if case.pc > 0 else []
    uni_mean_reqs = base + [(1, case.kv_mean)] * case.n
    uni_max_reqs  = base + [(1, case.kv_big)]  * case.n
    skew_reqs     = base + [(1, case.kv_big)] * case.nb + \
                    [(1, case.kvs)] * (case.n - case.nb)

    t_mean = _measure(llm, uni_mean_reqs, slice_, iters)
    t_max  = _measure(llm, uni_max_reqs,  slice_, iters)
    t_skew = _measure(llm, skew_reqs,     slice_, iters)

    alpha = ((t_skew - t_mean) / (t_max - t_mean)
             if t_max > t_mean else float("nan"))

    return {
        "regime": "pure" if case.pc == 0 else "mixed",
        "n": case.n, "nb": case.nb,
        "ratio": round(case.nb / case.n, 4),
        "skew": case.skew, "pc": case.pc, "kp": case.kp, "kvs": case.kvs,
        "kv_big": case.kv_big, "kv_mean": case.kv_mean,
        "t_mean_us": round(t_mean, 3),
        "t_max_us":  round(t_max, 3),
        "t_skew_us": round(t_skew, 3),
        "alpha": round(alpha, 4) if alpha == alpha else None,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _case_key(case: SkewCase) -> tuple:
    """Identity key for delta/resume matching against an existing CSV."""
    return (case.n, case.nb, round(case.skew, 3), case.pc, case.kp, case.kvs)


def _existing_keys(csv_path: Path) -> set[tuple]:
    """Keys already present in ``skew.csv`` (empty set if missing)."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return set()
    keys: set[tuple] = set()
    for _, r in df.iterrows():
        keys.add((
            int(r["n"]), int(r["nb"]), round(float(r["skew"]), 3),
            int(r["pc"]), int(r["kp"]), int(r["kvs"]),
        ))
    return keys


def _flush_rows(csv_path: Path, new_rows: list[dict]) -> pd.DataFrame:
    """Merge ``new_rows`` with any existing CSV and rewrite atomically.

    Returns the combined DataFrame. De-duplicates on the case key so
    a re-run of the same case overwrites rather than duplicates.
    """
    frames: list[pd.DataFrame] = []
    if csv_path.exists():
        try:
            frames.append(pd.read_csv(csv_path))
        except Exception:
            pass
    if new_rows:
        frames.append(pd.DataFrame(new_rows))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Keep the latest measurement for any repeated key.
    df = df.drop_duplicates(
        subset=["n", "nb", "skew", "pc", "kp", "kvs"], keep="last"
    ).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    return df


def sample_skew(
    llm,
    arch: Architecture,
    args: ProfileArgs,
    limits,
    tp: int,
    tp_root: Path,
) -> Path:
    """Fire feasible skew cases, write/update ``tp_root/skew.csv``.

    Resume behaviour: if ``skew.csv`` already exists, its rows are
    preserved and only cases whose keys are absent from it are fired.
    New rows are appended. This lets the user re-run after a
    feasibility change (e.g. added pc=2048 cases) without losing the
    hours of prior measurement.
    """
    # Serialize attention catalog slice for the RPC.
    slice_ = {
        name: {"vllm": e.vllm, "within": e.within, "tp_stable": e.tp_stable}
        for name, e in arch.catalog.attention.items()
    }
    iters = args.measurement_iterations
    grid = _build_grid(args, limits)
    from profiler.core.writer import _geometric_spec
    log.info(
        "skew grid: n=%s ratio=%s pc=%s kp=%s kvs=%s skew=(T1=%.1f, T2 swept)",
        _geometric_spec(grid["n"]), list(grid["ratio"]),
        _geometric_spec(grid["pc"]), _geometric_spec(grid["kp"]),
        _geometric_spec(grid["kvs"]), _SKEW_REP,
    )
    all_cases = _build_cases(args, limits)
    if not all_cases:
        log.warning("skew: no feasible cases at tp=%d; skipping", tp)
        return tp_root / "skew.csv"

    out = tp_root / "skew.csv"
    tp_root.mkdir(parents=True, exist_ok=True)

    # Resume vs force: the default preserves prior rows and fires only
    # new keys (matches the main loop's resume policy). ``--force``
    # wipes the CSV and re-fires every case.
    if args.force and out.exists():
        log.info("skew force: removing existing %s before re-firing", out)
        out.unlink()

    prior_keys = _existing_keys(out)
    cases = [c for c in all_cases if _case_key(c) not in prior_keys]
    if prior_keys:
        log.info(
            "skew resume: %d prior rows on disk, %d new cases to fire "
            "(skipped %d already-measured)",
            len(prior_keys), len(cases), len(all_cases) - len(cases),
        )
    if not cases:
        log.info("skew: %s already has all %d cases; nothing to do",
                 out, len(all_cases))
        return out

    label = f"TP={tp}  skew"
    rows: list[dict] = []
    with log.progress(label, total=len(cases)) as bar:
        for i, case in enumerate(cases):
            try:
                row = _measure_case(llm, case, slice_, iters)
            except Exception as e:
                log.warning(
                    "skew shot failed (n=%d nb=%d pc=%d kp=%d): %s",
                    case.n, case.nb, case.pc, case.kp, e,
                )
                bar.advance(1)
                continue
            rows.append(row)
            bar.advance(1)
            # Save incrementally every 20 rows so a crash doesn't lose data
            if (i + 1) % 20 == 0:
                _flush_rows(out, rows)

    if rows:
        df = _flush_rows(out, rows)
        # Per-regime alpha summary
        for regime, sub in df.groupby("regime"):
            alphas = sub["alpha"].dropna()
            log.info(
                "%s: n=%d alpha mean=%.3f min=%.3f max=%.3f",
                regime, len(sub),
                float(alphas.mean()) if len(alphas) else 0.0,
                float(alphas.min()) if len(alphas) else 0.0,
                float(alphas.max()) if len(alphas) else 0.0,
            )
    else:
        log.warning("skew: no valid rows measured at tp=%d", tp)

    log.success("skew → %s", out)
    return out
