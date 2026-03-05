#!/usr/bin/env python3
import os
import re
import sys

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
THROUGHPUT_RE = re.compile(r"Average generation throughput \(tok/s\):\s+([\d.]+)")
INSTANCE_RE = re.compile(r"Instance \[(\d+)\]")
TTFT_RE = re.compile(r"Mean TTFT \(ms\):\s+([\d.]+)")
TPOT_RE = re.compile(r"Mean TPOT \(ms\):\s+([\d.]+)")
ITL_RE = re.compile(r"Mean ITL \(ms\):\s+([\d.]+)")


def strip_ansi(text):
    return ANSI_RE.sub("", text)


def get_experiment_name(log_path):
    base_name = os.path.splitext(os.path.basename(log_path))[0]
    if base_name.endswith("_output"):
        base_name = base_name[:-len("_output")]
    return base_name


def average(values):
    valid = [value for value in values if value is not None]
    if not valid:
        return ""
    return sum(valid) / len(valid)


def parse_log(log_path):
    throughput = None
    current_instance = None
    instance_metrics = {}

    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = strip_ansi(raw_line.rstrip("\n"))

            throughput_match = THROUGHPUT_RE.search(line)
            if throughput_match:
                throughput = float(throughput_match.group(1))
                continue

            instance_match = INSTANCE_RE.search(line)
            if instance_match:
                current_instance = int(instance_match.group(1))
                instance_metrics.setdefault(
                    current_instance, {"ttft": None, "tpot": None, "itl": None}
                )
                continue

            if current_instance is None:
                continue

            if "No TTFT data available" in line:
                instance_metrics[current_instance]["ttft"] = None
                continue

            if "No TPOT data available" in line:
                instance_metrics[current_instance]["tpot"] = None
                continue

            if "No ITL data available" in line:
                instance_metrics[current_instance]["itl"] = None
                continue

            ttft_match = TTFT_RE.search(line)
            if ttft_match:
                instance_metrics[current_instance]["ttft"] = float(ttft_match.group(1))
                continue

            tpot_match = TPOT_RE.search(line)
            if tpot_match:
                instance_metrics[current_instance]["tpot"] = float(tpot_match.group(1))
                continue

            itl_match = ITL_RE.search(line)
            if itl_match:
                instance_metrics[current_instance]["itl"] = float(itl_match.group(1))

    if throughput is None:
        raise ValueError(f"Could not find average generation throughput in {log_path}")

    return (
        throughput,
        average([metrics["ttft"] for metrics in instance_metrics.values()]),
        average([metrics["tpot"] for metrics in instance_metrics.values()]),
        average([metrics["itl"] for metrics in instance_metrics.values()]),
    )


def main():
    if len(sys.argv) not in (2, 3):
        print(f"Usage: {sys.argv[0]} LOG_FILE [OUTPUT_TSV]", file=sys.stderr)
        sys.exit(1)

    log_path = sys.argv[1]
    if len(sys.argv) == 3:
        output_path = sys.argv[2]
    else:
        base_name = os.path.splitext(os.path.basename(log_path))[0]
        if base_name.endswith("_output"):
            base_name = base_name[:-len("_output")]
        output_path = os.path.join(os.path.dirname(log_path), f"{base_name}_latency.tsv")

    throughput, ttft, tpot, itl = parse_log(log_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("throughput_tok_s\tmean_ttft_ms\tmean_tpot_ms\tmean_itl_ms\n")
        f.write(f"{throughput}\t{ttft}\t{tpot}\t{itl}\n")

    print(f"Saved 1 row to {output_path}")


if __name__ == "__main__":
    main()
