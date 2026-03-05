#!/usr/bin/env python3
import os
import re
import sys

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TIME_RE = re.compile(r"\[(\d+(?:\.\d+)?)s\]")
INSTANCE_RE = re.compile(
    r"Instance\[(\d+)\].*?Each NPU Memory Usage\s+([\d.]+)\s+MB\s+\(([\d.]+)\s+% Used\)"
)
CPU_RE = re.compile(
    r"Node\[(\d+)\]:\s+Total CPU Memory Usage\s+([\d.]+)\s+MB,\s+([\d.]+)\s+% Used"
)
HIT_RE = re.compile(r"Prefix Cache Hit ratio\s+([\d.]+)\s+%")


def strip_ansi(text):
    return ANSI_RE.sub("", text)


def default_output_path(log_path):
    base_name = os.path.splitext(os.path.basename(log_path))[0]
    if base_name.endswith("_output"):
        base_name = base_name[:-len("_output")]
    return os.path.join(os.path.dirname(log_path), f"{base_name}_memory.tsv")


def parse_memory(log_path):
    rows = {}
    current_time = None
    instance_ids = set()
    node_ids = set()

    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = strip_ansi(raw_line.rstrip("\n"))

            time_match = TIME_RE.search(line)
            if time_match:
                current_time = float(time_match.group(1))
                rows.setdefault(current_time, {"instances": {}, "nodes": {}})
                continue

            if current_time is None:
                continue

            inst_match = INSTANCE_RE.search(line)
            if inst_match:
                instance_id = int(inst_match.group(1))
                instance_ids.add(instance_id)
                rows[current_time]["instances"][instance_id] = {
                    "gpu_mem_mb": float(inst_match.group(2)),
                    "prefix_hit_pct": 0.0,
                }
                hit_match = HIT_RE.search(line)
                if hit_match:
                    rows[current_time]["instances"][instance_id]["prefix_hit_pct"] = float(
                        hit_match.group(1)
                    )
                continue

            cpu_match = CPU_RE.search(line)
            if cpu_match:
                node_id = int(cpu_match.group(1))
                node_ids.add(node_id)
                rows[current_time]["nodes"][node_id] = {
                    "shared_cpu_mem_mb": float(cpu_match.group(2)),
                    "shared_cpu_prefix_hit_pct": 0.0,
                }
                hit_match = HIT_RE.search(line)
                if hit_match:
                    rows[current_time]["nodes"][node_id]["shared_cpu_prefix_hit_pct"] = float(
                        hit_match.group(1)
                    )

    return rows, sorted(instance_ids), sorted(node_ids)


def split_output_path(output_path, suffix):
    root, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".tsv"
    return f"{root}{suffix}{ext}"


def iter_times(rows, drop_first=False):
    times = sorted(rows)
    if drop_first and times:
        times = times[1:]
    return list(enumerate(times))


def write_instance_tsv(output_path, rows, instance_id, drop_first=False, shift_memory_only=False):
    header = ["time_s", "gpu_mem_mb", "prefix_hit_pct"]
    times = sorted(rows)

    if drop_first and times:
        times = times[1:]

    if shift_memory_only and len(times) >= 2:
        paired_times = list(zip(times[:-1], times[1:]))
    else:
        paired_times = [(time_s, time_s) for time_s in times]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for idx, (prefix_time_s, memory_time_s) in enumerate(paired_times):
            row = [str(idx)]
            prefix_instance = rows[prefix_time_s]["instances"].get(instance_id)
            memory_instance = rows[memory_time_s]["instances"].get(instance_id)
            if prefix_instance is None and memory_instance is None:
                row.extend(["", ""])
            else:
                row.extend(
                    [
                        "" if memory_instance is None else str(memory_instance["gpu_mem_mb"]),
                        "" if prefix_instance is None else str(prefix_instance["prefix_hit_pct"]),
                    ]
                )

            f.write("\t".join(row) + "\n")


def write_shared_cpu_tsv(output_path, rows, node_ids, drop_first=False):
    header = [
        "time_s",
        "shared_cpu_mem_mb",
        "shared_cpu_prefix_hit_pct",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for idx, time_s in iter_times(rows, drop_first=drop_first):
            row = [str(idx)]
            node = rows[time_s]["nodes"].get(node_ids[0]) if node_ids else None
            if node is None:
                row.extend(["", ""])
            else:
                row.extend(
                    [
                        str(node["shared_cpu_mem_mb"]),
                        str(node["shared_cpu_prefix_hit_pct"]),
                    ]
                )
            f.write("\t".join(row) + "\n")


def write_multi_instance_tsvs(output_path, rows, instance_ids, node_ids):
    output_paths = []
    for instance_id in instance_ids:
        inst_output_path = split_output_path(output_path, f"_inst{instance_id}")
        write_instance_tsv(inst_output_path, rows, instance_id, drop_first=True)
        output_paths.append(inst_output_path)

    if node_ids:
        shared_output_path = split_output_path(output_path, "_shared_cpu")
        write_shared_cpu_tsv(shared_output_path, rows, node_ids, drop_first=True)
        output_paths.append(shared_output_path)

    return output_paths


def write_outputs(output_path, rows, instance_ids, node_ids):
    if len(instance_ids) <= 1:
        instance_id = instance_ids[0] if instance_ids else 0
        write_instance_tsv(output_path, rows, instance_id, shift_memory_only=True)
        return [output_path]

    return write_multi_instance_tsvs(output_path, rows, instance_ids, node_ids)


def main():
    if len(sys.argv) not in (2, 3):
        print(f"Usage: {sys.argv[0]} LOG_FILE [OUTPUT_TSV]", file=sys.stderr)
        sys.exit(1)

    log_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) == 3 else default_output_path(log_path)

    rows, instance_ids, node_ids = parse_memory(log_path)
    output_paths = write_outputs(output_path, rows, instance_ids, node_ids)

    for path in output_paths:
        print(f"Saved {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
