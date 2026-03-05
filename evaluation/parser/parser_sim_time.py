#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


SIM_TIME_PATTERN = re.compile(
    r"Total simulation time:\s*(\d+)h\s+(\d+)m\s+([0-9]+(?:\.[0-9]+)?)s"
)


def parse_simulation_time(log_path: Path) -> float:
    for line in log_path.read_text().splitlines():
        match = SIM_TIME_PATTERN.search(line)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            seconds = float(match.group(3))
            return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Could not find total simulation time in {log_path}")


def default_output_path(log_path: Path) -> Path:
    stem = log_path.stem
    if stem.endswith("_output"):
        stem = stem[: -len("_output")]
    return log_path.parent / f"{stem}_sim_time.tsv"


def write_output(output_path: Path, simulation_time_s: float) -> None:
    output_path.write_text(f"simulation_time_s\n{simulation_time_s:g}\n")


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(f"Usage: {sys.argv[0]} LOG_FILE [OUTPUT_TSV]", file=sys.stderr)
        return 1

    log_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) == 3 else default_output_path(log_path)

    simulation_time_s = parse_simulation_time(log_path)
    write_output(output_path, simulation_time_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
