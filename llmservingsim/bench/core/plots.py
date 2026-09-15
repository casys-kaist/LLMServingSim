"""Plot helpers for ``bench validate``.

Three matplotlib plots, all written as PNGs into the validation output
directory; plus a plain-text summary table:

    <prefix>_throughput.png     prompt + gen throughput (sim vs vLLM)
    <prefix>_requests.png       running / waiting requests (sim vs vLLM)
    <prefix>_latency.png        TTFT / TPOT / latency CDFs (sim vs vLLM)
    <prefix>_summary.txt        TTFT / TPOT / latency at mean/median/p90/p95/p99
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

# matplotlib is imported lazily inside each helper so the module is
# importable in environments without matplotlib (e.g. CLI --help).


def plot_throughput(output_dir: Path, prefix: str,
                    bench_t: Sequence[float], bench_prompt: Sequence[float],
                    bench_gen: Sequence[float],
                    sim_t: Sequence[float], sim_prompt: Sequence[float],
                    sim_gen: Sequence[float],
                    title: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(bench_t, bench_prompt, label="vLLM", color="C0")
    axes[0].plot(sim_t, sim_prompt, label="Sim", color="C1", linestyle="--")
    axes[0].set_ylabel("Prompt tokens/s")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(bench_t, bench_gen, label="vLLM", color="C0")
    axes[1].plot(sim_t, sim_gen, label="Sim", color="C1", linestyle="--")
    axes[1].set_ylabel("Generation tokens/s")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"Throughput — {title}")
    fig.tight_layout()
    out = _save_fig(fig, output_dir, prefix, "throughput")
    plt.close(fig)
    return out


def plot_requests(output_dir: Path, prefix: str,
                  bench_t: Sequence[float], bench_running: Sequence[int],
                  bench_waiting: Sequence[int],
                  sim_t: Sequence[float], sim_running: Sequence[int],
                  sim_waiting: Sequence[int],
                  title: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(bench_t, bench_running, label="vLLM", color="C0")
    axes[0].plot(sim_t, sim_running, label="Sim", color="C1", linestyle="--")
    axes[0].set_ylabel("Running")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(bench_t, bench_waiting, label="vLLM", color="C0")
    axes[1].plot(sim_t, sim_waiting, label="Sim", color="C1", linestyle="--")
    axes[1].set_ylabel("Waiting")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"Running / Waiting — {title}")
    fig.tight_layout()
    out = _save_fig(fig, output_dir, prefix, "requests")
    plt.close(fig)
    return out


def plot_latency_cdfs(output_dir: Path, prefix: str,
                      bench_ttft: Sequence[float], sim_ttft: Sequence[float],
                      bench_tpot: Sequence[float], sim_tpot: Sequence[float],
                      bench_latency: Sequence[float], sim_latency: Sequence[float],
                      title: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, (label, va, sa) in zip(
        axes,
        [("TTFT (ms)", bench_ttft, sim_ttft),
         ("TPOT (ms)", bench_tpot, sim_tpot),
         ("Latency (ms)", bench_latency, sim_latency)],
    ):
        for series, name, color, style in [
            (va, "vLLM", "C0", "-"),
            (sa, "Sim", "C1", "--"),
        ]:
            xs, ys = _cdf(series)
            ax.plot(xs, ys, label=name, color=color, linestyle=style)
        ax.set_xlabel(label)
        ax.set_ylabel("CDF")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"Latency CDFs — {title}")
    fig.tight_layout()
    out = _save_fig(fig, output_dir, prefix, "latency")
    plt.close(fig)
    return out


def write_summary(output_dir: Path, prefix: str,
                  bench_ttft: Sequence[float], sim_ttft: Sequence[float],
                  bench_tpot: Sequence[float], sim_tpot: Sequence[float],
                  bench_latency: Sequence[float], sim_latency: Sequence[float]
                  ) -> Path:
    """Write a plain-text TTFT/TPOT/Latency table with sim-vs-vLLM diff%."""
    lines: list[str] = []
    header = f"{'Metric':<25}{'vLLM':>12}{'Sim':>12}{'Diff%':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for name, va, sa in [("TTFT", bench_ttft, sim_ttft),
                         ("TPOT", bench_tpot, sim_tpot),
                         ("Latency", bench_latency, sim_latency)]:
        for stat_label, fn in _STATS:
            v = fn(va) if va else float("nan")
            s = fn(sa) if sa else float("nan")
            diff = (s - v) / v * 100.0 if v else float("nan")
            lines.append(
                f"{name + ' ' + stat_label:<25}"
                f"{v:>12.1f}{s:>12.1f}{diff:>+9.1f}%"
            )
        lines.append("")

    out = output_dir / _name(prefix, "summary.txt")
    out.write_text("\n".join(lines))
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cdf(series: Sequence[float]) -> tuple[list[float], list[float]]:
    if not series:
        return [], []
    xs = sorted(series)
    n = len(xs)
    ys = [(i + 1) / n for i in range(n)]
    return xs, ys


def _percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    idx = (q / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


_STATS = [
    ("Mean", _mean),
    ("Median", lambda xs: _percentile(xs, 50)),
    ("P90", lambda xs: _percentile(xs, 90)),
    ("P95", lambda xs: _percentile(xs, 95)),
    ("P99", lambda xs: _percentile(xs, 99)),
]


def _name(prefix: str, fname: str) -> str:
    return f"{prefix}_{fname}" if prefix else fname


def _save_fig(fig, output_dir: Path, prefix: str, stem: str) -> Path:
    out = output_dir / _name(prefix, f"{stem}.png")
    fig.savefig(out, dpi=150)
    return out
