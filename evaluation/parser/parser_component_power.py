#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
LINE_RE = re.compile(r"([A-Za-z ]+energy consumption) \(J\):\s+([\d.]+)")

COMPONENT_MAP = {
    "Base Node energy consumption": "Others",
    "NPU energy consumption": "GPU",
    "CPU energy consumption": "CPU",
    "Memory energy consumption": "Memory",
    "Link energy consumption": "Link",
    "NIC energy consumption": "NIC",
    "Storage energy consumption": "Storage",
}

ORDER = ["Others", "GPU", "CPU", "Memory", "Link", "NIC", "Storage"]


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_components(log_path: str) -> dict[str, float]:
    values = {}
    with open(log_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = strip_ansi(raw_line.rstrip("\n")).replace("├─ ", "").replace("└─ ", "")
            match = LINE_RE.search(line)
            if not match:
                continue

            source_name = match.group(1).strip()
            if source_name in COMPONENT_MAP:
                values[COMPONENT_MAP[source_name]] = float(match.group(2))

    missing = [name for name in ORDER if name not in values]
    if missing:
        raise ValueError(f"Missing component values in {log_path}: {', '.join(missing)}")

    return values


def default_output_path(first_log_path: str) -> str:
    return os.path.join(os.path.dirname(first_log_path), "component_power.tsv")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse per-component energy values from one or more logs into a TSV."
    )
    parser.add_argument("logs", nargs="*", help="Log paths when using the flexible interface.")
    parser.add_argument("-o", "--output", dest="output_path", help="Output TSV path.")
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Column labels corresponding to each log. Count must match the number of logs.",
    )

    args = parser.parse_args(argv)

    # Backward-compatible mode: LOG1 LOG2 [OUTPUT]
    if not args.logs and not args.output_path and args.labels is None:
        parser.error("No input logs provided.")

    if args.labels is None and len(args.logs) in (2, 3):
        candidate_logs = args.logs[:2]
        if all(os.path.exists(path) for path in candidate_logs):
            if len(args.logs) == 3 and not args.output_path:
                args.output_path = args.logs[2]
            args.logs = candidate_logs
            args.labels = ["TP 1", "TP 2"]

    if not args.logs:
        parser.error("At least one log path is required.")

    if args.labels is None:
        parser.error("--labels is required unless using the legacy two-log interface.")

    if len(args.labels) != len(args.logs):
        parser.error("The number of --labels values must match the number of logs.")

    if args.output_path is None:
        args.output_path = default_output_path(args.logs[0])

    return args


def main() -> int:
    args = parse_args(sys.argv[1:])
    parsed_columns = [parse_components(log_path) for log_path in args.logs]

    with open(args.output_path, "w", encoding="utf-8") as handle:
        handle.write("component\t" + "\t".join(args.labels) + "\n")
        for component in ORDER:
            values = [str(column[component]) for column in parsed_columns]
            handle.write(f"{component}\t" + "\t".join(values) + "\n")

    print(f"Saved {len(ORDER)} rows to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
