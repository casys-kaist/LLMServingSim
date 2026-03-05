#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys


TOTAL_ENERGY_RE = re.compile(r"Total energy consumption \(kJ\):\s+([\d.]+)")
GENERATED_TOKENS_RE = re.compile(r"Total generated tokens:\s+(\d+)")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse total energy per generated token from one or more logs."
    )
    parser.add_argument("--labels", nargs="+", required=True, help="Column labels for each log.")
    parser.add_argument("-o", "--output", dest="output_path", required=True, help="Output TSV path.")
    parser.add_argument("logs", nargs="+", help="Log paths.")
    args = parser.parse_args(argv)

    if len(args.labels) != len(args.logs):
        parser.error("The number of --labels values must match the number of logs.")

    return args


def parse_energy_per_token(log_path: str) -> float:
    total_energy_kj = None
    generated_tokens = None

    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if total_energy_kj is None:
                energy_match = TOTAL_ENERGY_RE.search(line)
                if energy_match:
                    total_energy_kj = float(energy_match.group(1))
                    continue

            if generated_tokens is None:
                generated_match = GENERATED_TOKENS_RE.search(line)
                if generated_match:
                    generated_tokens = int(generated_match.group(1))

    if total_energy_kj is None:
        raise ValueError(f"Could not find total energy consumption in {log_path}")
    if generated_tokens in (None, 0):
        raise ValueError(f"Could not find total generated tokens in {log_path}")

    return total_energy_kj * 1000.0 / generated_tokens


def main() -> int:
    args = parse_args(sys.argv[1:])
    values = [parse_energy_per_token(path) for path in args.logs]

    with open(args.output_path, "w", encoding="utf-8") as handle:
        handle.write("\t".join(args.labels) + "\n")
        handle.write("\t".join(str(value) for value in values) + "\n")

    print(f"Saved 1 row to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
