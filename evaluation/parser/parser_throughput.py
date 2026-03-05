#!/usr/bin/env python3
import ast
import os
import sys

PREFIX = "Throughput per 1.0 sec:"


def parse_throughput(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if PREFIX not in line:
                continue

            _, values = line.split(PREFIX, 1)
            pairs = ast.literal_eval(values.strip())
            return [0.0] + [pair[1] for pair in pairs]

    raise ValueError(f"Could not find '{PREFIX}' in {log_path}")


def main():
    if len(sys.argv) not in (2, 3):
        print(f"Usage: {sys.argv[0]} LOG_FILE [OUTPUT_TSV]", file=sys.stderr)
        sys.exit(1)

    log_path = sys.argv[1]
    throughput_values = parse_throughput(log_path)

    if len(sys.argv) == 3:
        output_path = sys.argv[2]
    else:
        base_name = os.path.splitext(os.path.basename(log_path))[0]
        if base_name.endswith("_output"):
            base_name = base_name[:-len("_output")]
        output_path = os.path.join(
            os.path.dirname(log_path),
            f"{base_name}_throughput.tsv",
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("time_s\tthroughput\n")
        for idx, value in enumerate(throughput_values):
            f.write(f"{idx}\t{value}\n")

    print(f"Saved {len(throughput_values)} rows to {output_path}")


if __name__ == "__main__":
    main()
