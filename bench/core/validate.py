"""Compare a bench run against simulator output.

Reads ``<bench_dir>/{meta.json,requests.jsonl,timeseries.csv}`` and the
simulator's ``sim.csv`` / ``sim.log`` for the matching workload, derives
TTFT / TPOT / end-to-end latency on both sides, and writes plots + a
text summary into ``<bench_dir>/<output-subdir>/``.

Latency definitions on both sides (kept consistent so diff% is meaningful):

    TTFT   = first_token_ts - arrival_time      # incl. queueing
    TPOT   = (last_token_ts - first_token_ts) / max(1, output_toks - 1)
    e2e    = last_token_ts - arrival_time

The simulator's ``sim.csv`` already exposes ``arrival``, ``end_time``, and
the per-token ITL list directly; bench's ``requests.jsonl`` carries raw
vLLM ``RequestStateStats`` timestamps from which we compute the same.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from bench.core import logger as log
from bench.core import plots


def register_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bench-dir", required=True, dest="bench_dir",
                   help="Path to a finished bench run "
                        "(bench/results/<run_id>/).")
    p.add_argument("--sim-csv", required=True, dest="sim_csv",
                   help="Simulator per-request CSV output.")
    p.add_argument("--sim-log", required=True, dest="sim_log",
                   help="Simulator log (parsed for per-tick running/waiting).")
    p.add_argument("--output-subdir", default="validation",
                   dest="output_subdir",
                   help="Subdirectory under bench-dir to write plots/summary "
                        "into (default: validation).")
    p.add_argument("--prefix", default="",
                   help="Filename prefix for plots / summary.")
    p.add_argument("--title", default="vLLM vs LLMServingSim",
                   help="Plot title suffix.")
    p.add_argument("--log-level", default="INFO",
                   dest="log_level",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logger verbosity (default: INFO).")


def run(args: argparse.Namespace) -> int:
    log.configure(args.log_level)
    log.print_banner(
        "LLMServingSim Validate",
        f"{args.bench_dir} vs {args.sim_csv}",
    )

    bench_dir = Path(args.bench_dir)
    output_dir = bench_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Bench side.
    # ------------------------------------------------------------------
    with log.stage("Loading bench results"):
        bench_reqs = _load_bench_requests(bench_dir / "requests.jsonl")
        bench_ts = _load_bench_timeseries(bench_dir / "timeseries.csv")
        bench_ttft, bench_tpot, bench_lat = _bench_latencies(bench_reqs)
    log.info("bench: %d requests, %d timeseries rows",
             len(bench_reqs), len(bench_ts))

    # ------------------------------------------------------------------
    # Simulator side.
    # ------------------------------------------------------------------
    with log.stage("Loading simulator results"):
        sim_reqs = _load_sim_csv(Path(args.sim_csv))
        sim_ts = _load_sim_log(Path(args.sim_log))
        sim_ttft, sim_tpot, sim_lat = _sim_latencies(sim_reqs)
    log.info("sim:   %d requests, %d timeseries rows",
             len(sim_reqs), len(sim_ts))

    # ------------------------------------------------------------------
    # Plot + summary.
    # ------------------------------------------------------------------
    with log.stage("Rendering plots + summary"):
        plots.plot_throughput(
            output_dir, args.prefix,
            bench_t=[r["t"] for r in bench_ts],
            bench_prompt=[r["prompt_throughput"] for r in bench_ts],
            bench_gen=[r["gen_throughput"] for r in bench_ts],
            sim_t=[r["t"] for r in sim_ts],
            sim_prompt=[r["prompt_throughput"] for r in sim_ts],
            sim_gen=[r["gen_throughput"] for r in sim_ts],
            title=args.title,
        )
        plots.plot_requests(
            output_dir, args.prefix,
            bench_t=[r["t"] for r in bench_ts],
            bench_running=[r["running"] for r in bench_ts],
            bench_waiting=[r["waiting"] for r in bench_ts],
            sim_t=[r["t"] for r in sim_ts],
            sim_running=[r["running"] for r in sim_ts],
            sim_waiting=[r["waiting"] for r in sim_ts],
            title=args.title,
        )
        plots.plot_latency_cdfs(
            output_dir, args.prefix,
            bench_ttft, sim_ttft, bench_tpot, sim_tpot, bench_lat, sim_lat,
            title=args.title,
        )
        summary_path = plots.write_summary(
            output_dir, args.prefix,
            bench_ttft, sim_ttft, bench_tpot, sim_tpot, bench_lat, sim_lat,
        )
    log.success("Wrote plots + summary -> %s", output_dir)
    log.info("Summary: %s", summary_path)
    return 0


# ---------------------------------------------------------------------------
# Bench loaders
# ---------------------------------------------------------------------------

def _load_bench_requests(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_bench_timeseries(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            out.append({
                "t": float(row["t"]),
                "prompt_throughput": float(row["prompt_throughput"]),
                "gen_throughput": float(row["gen_throughput"]),
                "running": int(float(row["running"])),
                "waiting": int(float(row["waiting"])),
                "kv_cache_pct": float(row.get("kv_cache_pct", 0.0)),
            })
    return out


def _bench_latencies(reqs: list[dict]) -> tuple[list[float], list[float], list[float]]:
    """Compute TTFT / TPOT / e2e in milliseconds from RequestStateStats.

    vLLM records a mixed clock domain in ``requests.jsonl``:
    ``arrival_time`` is wall-clock epoch seconds, while
    ``queued_ts``/``scheduled_ts``/``first_token_ts``/``last_token_ts``
    are monotonic engine timestamps. Convert ``arrival_time`` into that
    domain and anchor there -- **not** on ``queued_ts``, which is stamped a
    whole in-flight step later. See ``_bench_arrival_ts``.
    """
    offset = bench_epoch_to_monotonic_offset(reqs)
    ttft, tpot, lat = [], [], []
    for r in reqs:
        arr, _ = _bench_arrival_ts(r, offset)
        first = r.get("first_token_ts")
        last = r.get("last_token_ts")
        if arr is None or first is None or last is None:
            continue
        out_toks = max(1, int(r.get("output_toks", 1)))
        ttft.append((first - arr) * 1000.0)
        if out_toks > 1:
            tpot.append((last - first) / (out_toks - 1) * 1000.0)
        lat.append((last - arr) * 1000.0)
    return ttft, tpot, lat


def bench_epoch_to_monotonic_offset(reqs: list[dict]) -> float | None:
    """Estimate the epoch -> monotonic offset from a run's own requests.

    ``queued_ts = arrival_time + offset + pickup``, where ``pickup >= 0`` is
    the wait until the engine core next polls its input queue, so the minimum
    of ``queued_ts - arrival_time`` over the run bounds the offset from above
    by ``min(pickup)``. With a few hundred requests some arrival lands close
    to a loop boundary, so the residual bias is a couple of milliseconds --
    against the tens of milliseconds the anchor choice is worth. Returns None
    when the two fields are not in different domains (nothing to convert).
    """
    deltas = [r["queued_ts"] - r["arrival_time"] for r in reqs
              if r.get("queued_ts") is not None
              and r.get("arrival_time") is not None]
    if not deltas:
        return None
    return min(deltas)


def _bench_arrival_ts(req: dict, offset: float | None = None) -> tuple[float | None, bool]:
    """Pick the arrival timestamp in the same clock domain as first/last.

    **Anchor at ``arrival_time``, not ``queued_ts``.** They differ by the wait
    until the engine core next polls its input queue: ``QUEUED`` is recorded
    inside ``Scheduler.add_request``, which runs at a loop boundary, so a
    request that arrives mid-step is registered only when the in-flight step
    ends. Measured on RTXPRO6000/Qwen3-30B-A3B, ``queued_ts - arrival_time``
    is p50 **21.7 ms** / p90 61.6 ms -- the same distribution as the
    simulator's own wait-for-the-in-flight-batch term (p50 22.5, p90 62.4),
    because it is the same physical wait.

    Anchoring at ``queued_ts`` therefore drops that wait from vLLM's TTFT
    while the simulator, which starts from the workload's arrival time, keeps
    it -- and the comparison charges the simulator for a term it measured
    correctly. It is worth 18% of a 124 ms TTFT and 0.07% of a 32 s
    end-to-end latency, which is exactly why TTFT looked wrong on this example
    while TPOT and latency sat at 0.5%. With the anchor fixed, that example's
    per-request |TTFT error| median goes from 16.8% to 9.7% and its median
    bias from +13.5% to -2.5%.

    ``arrival_time`` is epoch seconds (set at frontend entry) and the
    lifecycle timestamps are monotonic, so ``offset`` converts between them;
    see ``bench_epoch_to_monotonic_offset``. Without one, fall back to
    ``queued_ts`` as before rather than subtract across clock domains.
    """
    arr = req.get("arrival_time")
    queued = req.get("queued_ts")
    first = req.get("first_token_ts")
    last = req.get("last_token_ts")

    if queued is None:
        return arr, False
    if arr is None:
        return queued, True

    if _same_time_domain(arr, first) and _same_time_domain(arr, last):
        return arr, False
    if offset is not None:
        shifted = arr + offset
        if _same_time_domain(shifted, first) or _same_time_domain(shifted, last):
            return shifted, False
    if _same_time_domain(queued, first) or _same_time_domain(queued, last):
        return queued, True
    return arr, False


def _same_time_domain(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    # Requests finish in seconds, not days. If two timestamps differ by
    # more than ~28 hours, they're almost certainly from different clocks
    # (e.g. epoch wall-clock vs process monotonic time).
    return abs(a - b) < 100_000


# ---------------------------------------------------------------------------
# Simulator loaders
# ---------------------------------------------------------------------------

def _load_sim_csv(path: Path) -> list[dict]:
    """Parse sim.csv. Times are in nanoseconds."""
    out: list[dict] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            out.append({
                "input": int(row["input"]),
                "output": int(row["output"]),
                "arrival_ns": float(row["arrival"]),
                "end_ns": float(row["end_time"]),
                "latency_ns": float(row["latency"]),
                "queuing_delay_ns": float(row["queuing_delay"]),
                "ttft_ns": float(row["TTFT"]),
                "tpot_ns": float(row["TPOT"]),
            })
    return out


def _sim_latencies(rows: list[dict]) -> tuple[list[float], list[float], list[float]]:
    """Convert sim ns columns to ms, matching bench definitions.

    sim.csv's TTFT already includes queueing (arrival -> first token), and
    TPOT is the per-token decode time. ``latency_ns`` is end-to-end.
    """
    ttft = [r["ttft_ns"] / 1e6 for r in rows]
    tpot = [r["tpot_ns"] / 1e6 for r in rows]
    lat = [r["latency_ns"] / 1e6 for r in rows]
    return ttft, tpot, lat


# Sim log lines look like::
#
#   [12.0s] Avg prompt throughput: 1234.0 tokens/s, Avg generation throughput: 5678.0 tokens/s
#           ├─Running Instance[0]: 32 reqs, Waiting: 0 reqs, Total # 1 NPUs, ...
#           ├─Running Instance[1]: 30 reqs, Waiting: 1 reqs, Total # 1 NPUs, ...
#
# We accumulate running/waiting across instances at each tick and emit one
# row per timestamp.
_TS_RE = re.compile(r"^\[(\d+\.?\d*)s\]")
_TPUT_RE = re.compile(
    r"Avg prompt throughput:\s*(\d+\.?\d*).*generation throughput:\s*(\d+\.?\d*)"
)
_INST_RE = re.compile(
    r"Running Instance\[(\d+)\]:\s*(\d+) reqs, Waiting:\s*(\d+) reqs"
)


def _load_sim_log(path: Path) -> list[dict]:
    rows: list[dict] = []
    cur: dict | None = None

    def _flush(c: dict | None) -> None:
        if c is not None and "t" in c:
            rows.append({
                "t": c["t"],
                "prompt_throughput": c.get("prompt_throughput", 0.0),
                "gen_throughput": c.get("gen_throughput", 0.0),
                "running": c.get("running", 0),
                "waiting": c.get("waiting", 0),
            })

    with path.open() as f:
        for line in f:
            m_ts = _TS_RE.match(line)
            if m_ts:
                _flush(cur)
                cur = {"t": float(m_ts.group(1)), "running": 0, "waiting": 0}
                m_t = _TPUT_RE.search(line)
                if m_t:
                    cur["prompt_throughput"] = float(m_t.group(1))
                    cur["gen_throughput"] = float(m_t.group(2))
                continue
            m_i = _INST_RE.search(line)
            if m_i and cur is not None:
                cur["running"] += int(m_i.group(2))
                cur["waiting"] += int(m_i.group(3))
    _flush(cur)
    return rows
