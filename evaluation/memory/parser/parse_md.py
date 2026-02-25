#!/usr/bin/env python3
import re
import sys

# 사용법:
#   python parse_two_instance_log.py input.log > output.tsv

time_re = re.compile(r'^\[(\d+(?:\.\d+)?)s\]')
inst_re = re.compile(
    r'Instance\[(\d+)\].*?Each NPU Memory Usage ([\d.]+) MB .*?Prefix Cache Hit ratio ([\d.]+) %'
)
cpu_re = re.compile(
    r'Node\[0\]:\s+Total CPU Memory Usage ([\d.]+) MB.*?Prefix Cache Hit ratio ([\d.]+) %'
)

def parse_lines(lines):
    # time -> dict with keys: inst0_mem, inst0_hit, inst1_mem, inst1_hit, cpu_mem, cpu_hit
    data = {}
    current_time = None

    for line in lines:
        line = line.rstrip("\n")

        # 시간 라인
        m_time = time_re.match(line)
        if m_time:
            current_time = float(m_time.group(1))
            if current_time not in data:
                data[current_time] = {
                    "inst0_mem": None,
                    "inst0_hit": None,
                    "inst1_mem": None,
                    "inst1_hit": None,
                    "cpu_mem":  None,
                    "cpu_hit": None,
                }
            continue

        if current_time is None:
            # 아직 time을 못 본 상태면 스킵
            continue

        # Instance 라인
        m_inst = inst_re.search(line)
        if m_inst:
            idx = int(m_inst.group(1))
            mem = float(m_inst.group(2))
            hit = float(m_inst.group(3))

            if idx == 0:
                data[current_time]["inst0_mem"] = mem
                data[current_time]["inst0_hit"] = hit
            elif idx == 1:
                data[current_time]["inst1_mem"] = mem
                data[current_time]["inst1_hit"] = hit
            continue

        # CPU 라인
        m_cpu = cpu_re.search(line)
        if m_cpu:
            cpu_mem = float(m_cpu.group(1))
            hit = float(m_cpu.group(2))
            data[current_time]["cpu_mem"] = cpu_mem
            data[current_time]["cpu_hit"] = hit
            continue

    return data


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} LOG_FILE", file=sys.stderr)
        sys.exit(1)

    log_path = sys.argv[1]
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = parse_lines(lines)

    # TSV 헤더
    print("time_s\tinst0_mem_mb\tinst0_hit_pct\tinst1_mem_mb\tinst1_hit_pct\tcpu_mem_mb\tcpu_hit_pct")

    for t in sorted(data.keys()):
        row = data[t]

        def fmt(x):
            return "" if x is None else f"{x}"

        print(
            f"{t}\t"
            f"{fmt(row['inst0_mem'])}\t{fmt(row['inst0_hit'])}\t"
            f"{fmt(row['inst1_mem'])}\t{fmt(row['inst1_hit'])}\t"
            f"{fmt(row['cpu_mem'])}\t{fmt(row['cpu_hit'])}"
        )


if __name__ == "__main__":
    main()
