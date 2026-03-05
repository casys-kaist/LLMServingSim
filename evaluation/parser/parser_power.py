#!/usr/bin/env python3
import ast
import os
import sys

PREFIX = "Power per 1.0 sec (W):"


def parse_power(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if PREFIX not in line:
                continue

            _, values = line.split(PREFIX, 1)
            return ast.literal_eval(values.strip())

    raise ValueError(f"Could not find '{PREFIX}' in {log_path}")


def main():
    if len(sys.argv) not in (2, 3):
        print(f"Usage: {sys.argv[0]} LOG_FILE [OUTPUT_TSV]", file=sys.stderr)
        sys.exit(1)

    log_path = sys.argv[1]
    power_values = parse_power(log_path)

    if len(sys.argv) == 3:
        output_path = sys.argv[2]
    else:
        base_name = os.path.splitext(os.path.basename(log_path))[0]
        if base_name.endswith("_output"):
            base_name = base_name[:-len("_output")]
        output_path = os.path.join(
            os.path.dirname(log_path),
            f"{base_name}_power.tsv",
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("time_s\tpower_w\n")
        for idx, value in enumerate(power_values):
            f.write(f"{idx}\t{value}\n")

    print(f"Saved {len(power_values)} rows to {output_path}")


if __name__ == "__main__":
    main()
