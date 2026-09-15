"""Statistical tests behind ``bench validate``'s summary table.

The diff% table answers "how far apart are these two numbers"; it cannot
answer either question a validation run actually asks:

  * do sim and vLLM draw from the same distribution?  ->  two-sample
    Kolmogorov-Smirnov (``ks_2samp``), which compares the whole CDF and
    not just the five percentiles the table prints.
  * is the sim close *enough* to call it a match?  ->  TOST equivalence
    (``tost``) against a +/-margin band around the vLLM mean.  A plain
    difference test is the wrong tool here: with a few hundred requests
    it rejects on a 0.5% gap, so "significantly different" stops meaning
    "wrong".  TOST inverts it -- a small p means the sim is provably
    *inside* the band.

Both are stdlib-only (``statistics.NormalDist``).  scipy is not a
dependency of this repo, and at bench sample sizes (hundreds of
requests) the asymptotic forms below are what scipy would compute
anyway.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Sequence

ALPHA = 0.05


def ks_2samp(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Two-sample KS test: max CDF gap and its asymptotic p-value.

    p is the probability of a gap this large if both samples came from
    the same distribution, so a *small* p means they differ.
    """
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")

    xa, xb = sorted(a), sorted(b)
    na, nb = len(xa), len(xb)
    i = j = 0
    d = 0.0
    while i < na and j < nb:
        x = min(xa[i], xb[j])
        while i < na and xa[i] <= x:
            i += 1
        while j < nb and xb[j] <= x:
            j += 1
        d = max(d, abs(i / na - j / nb))

    en = math.sqrt(na * nb / (na + nb))
    return d, _ks_pvalue((en + 0.12 + 0.11 / en) * d)


def _ks_pvalue(lam: float) -> float:
    """Kolmogorov tail: Q(lam) = 2 * sum (-1)^(k-1) exp(-2 k^2 lam^2)."""
    if lam <= 0:
        return 1.0
    total = 0.0
    for k in range(1, 101):
        term = math.exp(-2.0 * k * k * lam * lam)
        total += (-1.0) ** (k - 1) * term
        if term < 1e-12:
            break
    return min(1.0, max(0.0, 2.0 * total))


def tost(a: Sequence[float], b: Sequence[float],
         margin: float) -> tuple[float, float]:
    """Equivalence test of mean(b) against mean(a) within +/-``margin``.

    ``margin`` is a fraction of mean(a) (0.10 = +/-10%).  Returns the
    relative mean difference (b vs a, as a fraction) and the TOST
    p-value: the larger of the two one-sided p-values, normal
    approximation with a Welch standard error.  p < ALPHA means the sim
    mean is inside the band -- equivalent, not merely "not proven
    different".
    """
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")

    ma, mb = _mean(a), _mean(b)
    rel = (mb - ma) / ma if ma else float("nan")
    delta = abs(ma) * margin
    diff = mb - ma
    se = math.sqrt(_var(a) / len(a) + _var(b) / len(b))
    if se == 0.0:
        return rel, 0.0 if abs(diff) < delta else 1.0

    z = NormalDist()
    p_lower = 1.0 - z.cdf((diff + delta) / se)   # H0: diff <= -delta
    p_upper = z.cdf((diff - delta) / se)         # H0: diff >= +delta
    return rel, max(p_lower, p_upper)


def summary_lines(
    metrics: Sequence[tuple[str, Sequence[float], Sequence[float]]],
    margin: float,
) -> list[str]:
    """Format one row per metric for the validation summary."""
    header = (f"{'Metric':<25}{'n(vLLM/Sim)':>14}{'KS D':>8}{'KS p':>9}"
              f"{'Mean diff%':>12}{'TOST p':>9}{'Equivalent':>12}")
    lines = [header, "-" * len(header)]
    for name, va, sa in metrics:
        d, p_ks = ks_2samp(va, sa)
        rel, p_tost = tost(va, sa, margin)
        equiv = "yes" if p_tost < ALPHA else "no"
        lines.append(
            f"{name:<25}{f'{len(va)}/{len(sa)}':>14}{d:>8.3f}{p_ks:>9.3f}"
            f"{rel * 100.0:>+11.1f}%{p_tost:>9.3f}{equiv:>12}"
        )
    lines.append("")
    lines.append(f"KS: p < {ALPHA} means the distributions differ.")
    lines.append(f"TOST: p < {ALPHA} means the sim mean is inside "
                 f"+/-{margin * 100.0:g}% of the vLLM mean.")
    return lines


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _var(xs: Sequence[float]) -> float:
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


if __name__ == "__main__":
    import random

    random.seed(0)
    base = [random.gauss(100.0, 10.0) for _ in range(500)]

    # Same distribution: no CDF gap worth reporting, means equivalent.
    same = [random.gauss(100.0, 10.0) for _ in range(500)]
    d, p = ks_2samp(base, same)
    assert d < 0.1 and p > ALPHA, (d, p)
    rel, p_t = tost(base, same, 0.10)
    assert abs(rel) < 0.05 and p_t < ALPHA, (rel, p_t)

    # Identical samples: zero gap, p = 1, trivially equivalent.
    d, p = ks_2samp(base, base)
    assert d == 0.0 and p == 1.0, (d, p)
    assert tost(base, base, 0.10)[1] < ALPHA

    # Shifted by 30%: distributions differ, means not within 10%.
    shifted = [x * 1.3 for x in base]
    d, p = ks_2samp(base, shifted)
    assert d > 0.5 and p < ALPHA, (d, p)
    rel, p_t = tost(base, shifted, 0.10)
    assert abs(rel - 0.3) < 0.01 and p_t > ALPHA, (rel, p_t)

    # A 3% shift is still equivalent within 10% but not within 1% --
    # the case a plain difference test gets backwards.
    small = [x * 1.03 for x in base]
    assert tost(base, small, 0.10)[1] < ALPHA
    assert tost(base, small, 0.01)[1] > ALPHA

    # Degenerate inputs stay nan rather than raising.
    assert all(math.isnan(v) for v in ks_2samp([1.0], []))
    assert all(math.isnan(v) for v in tost([1.0], [], 0.1))

    print("\n".join(summary_lines(
        [("TTFT", base, same), ("TPOT", base, shifted)], 0.10)))
    print("stats self-check OK")
