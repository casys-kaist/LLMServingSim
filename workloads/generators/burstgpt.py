"""BurstGPT weekly minute trace -> LLMServingSim JSONL.

Converts a ``weekly_minute_trace.csv`` (曜日, 時, 分, frequency) into flat
request arrivals. Token lengths are sampled from a reference ShareGPT JSONL
trace so prompt/decode sizes match an existing workload distribution.

Output rows contain ``input_toks``, ``output_toks``, and ``arrival_time_ns``.
``input_tok_ids`` / ``output_tok_ids`` are omitted by default because a full
week can exceed 100k requests; pass ``--include-tok-ids`` for smaller runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


_DOW = {"月": 0, "火": 1, "水": 2, "木": 3, "金": 4, "土": 5, "日": 6}


def register_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--trace-csv",
        required=True,
        help="Path to weekly_minute_trace.csv (曜日, 時, 分, frequency).",
    )
    p.add_argument(
        "--token-reference",
        required=True,
        help="Reference flat JSONL (e.g. sharegpt-llama-3.1-8b-300-sps10.jsonl) "
        "used to sample input/output token lengths.",
    )
    p.add_argument("--output", required=True, help="Output JSONL path.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed.")
    p.add_argument(
        "--first-arrival-sec",
        type=int,
        default=0,
        dest="first_arrival_sec",
        help="Offset added to the first request's arrival (seconds).",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiply every minute's frequency by this factor (e.g. 0.1).",
    )
    p.add_argument(
        "--count-mode",
        choices=("round", "poisson", "floor"),
        default="round",
        help="How to turn fractional requests/min into an integer count.",
    )
    p.add_argument(
        "--intra-minute",
        choices=("uniform", "start"),
        default="uniform",
        help="Placement of arrivals inside each minute slot.",
    )
    p.add_argument(
        "--max-reqs",
        type=int,
        default=0,
        dest="max_reqs",
        help="Stop after this many requests (0 = no cap).",
    )
    p.add_argument(
        "--include-tok-ids",
        action="store_true",
        dest="include_tok_ids",
        help="Copy input_tok_ids/output_tok_ids from the sampled reference row.",
    )


def _minute_offset(row: dict[str, str]) -> int:
    day = _DOW[row["曜日"].strip()]
    hour = int(row["時"])
    minute = int(row["分"])
    return day * 24 * 60 + hour * 60 + minute


def _poisson(lam: float, rng: random.Random) -> int:
    if lam <= 0:
        return 0
    if lam > 30:
        # Normal approximation for large lambda.
        n = int(round(rng.gauss(lam, math.sqrt(lam))))
        return max(n, 0)
    l = math.exp(-lam)
    k = 0
    p = 1.0
    while p > l:
        k += 1
        p *= rng.random()
    return k - 1


def _requests_for_minute(freq: float, count_mode: str, rng: random.Random) -> int:
    if freq <= 0:
        return 0
    if count_mode == "round":
        return int(round(freq))
    if count_mode == "floor":
        return int(freq)
    return _poisson(freq, rng)


def _load_reference(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No reference requests found in {path}")
    return rows


def run(args: argparse.Namespace) -> int:
    trace_path = Path(args.trace_csv)
    ref_path = Path(args.token_reference)
    out_path = Path(args.output)
    rng = random.Random(args.seed)

    reference = _load_reference(ref_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    total_freq = 0.0
    with trace_path.open(encoding="utf-8", newline="") as f_in, out_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        required = {"曜日", "時", "分", "frequency"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Expected columns {sorted(required)}, got {reader.fieldnames}"
            )

        for row in reader:
            freq = float(row["frequency"]) * args.scale
            total_freq += freq
            n_reqs = _requests_for_minute(freq, args.count_mode, rng)
            if n_reqs <= 0:
                continue

            minute_base_sec = (
                args.first_arrival_sec + _minute_offset(row) * 60
            )
            for i in range(n_reqs):
                if args.max_reqs and written >= args.max_reqs:
                    break

                if args.intra_minute == "start":
                    offset_sec = 0.0
                else:
                    offset_sec = rng.random() * 60.0

                ref = reference[rng.randrange(len(reference))]
                req = {
                    "input_toks": int(ref["input_toks"]),
                    "output_toks": int(ref["output_toks"]),
                    "arrival_time_ns": int((minute_base_sec + offset_sec) * 1e9),
                }
                if args.include_tok_ids:
                    if "input_tok_ids" in ref:
                        req["input_tok_ids"] = ref["input_tok_ids"]
                    if "output_tok_ids" in ref:
                        req["output_tok_ids"] = ref["output_tok_ids"]

                f_out.write(json.dumps(req, ensure_ascii=False) + "\n")
                written += 1

            if args.max_reqs and written >= args.max_reqs:
                break

    duration_sec = args.first_arrival_sec + 7 * 24 * 60 * 60
    avg_sps = written / duration_sec if duration_sec else 0.0
    print(
        f"Wrote {written:,} requests to {out_path} "
        f"(integrated freq≈{total_freq:,.1f}/week, avg≈{avg_sps:.3f} req/s)."
    )
    return 0
