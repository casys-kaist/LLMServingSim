import re

LOG_PATH = "SD_pulse_output.txt"
OUT_PATH = "parsed_llmservingsim.tsv"

# [1.0s]
time_re = re.compile(r"\[(\d+\.?\d*)s\]")
# Each NPU Memory Usage 15316.51 MB (37.394 % Used)
mem_re = re.compile(r"Each NPU Memory Usage\s+([\d\.]+)\s+MB\s+\(([\d\.]+)\s+% Used\)")
# Prefix Cache Hit ratio 25.69 %
hit_re = re.compile(r"Prefix Cache Hit ratio\s+([\d\.]+)\s+%")

rows = []

with open(LOG_PATH, "r", encoding="utf-8") as f:
    current_time = None
    for line in f:
        # 시간 파싱
        m_time = time_re.search(line)
        if m_time:
            current_time = float(m_time.group(1))
            continue

        # NPU 메모리 & hit ratio가 있는 줄
        if "Each NPU Memory Usage" in line:
            if current_time is None:
                continue  # 방어 코드

            m_mem = mem_re.search(line)
            if not m_mem:
                continue

            mem_mb = float(m_mem.group(1))
            mem_pct = float(m_mem.group(2))

            m_hit = hit_re.search(line)
            if m_hit:
                hit_pct = float(m_hit.group(1))
            else:
                # 1.0s처럼 Prefix Cache Hit ratio가 안 찍힌 경우 0으로 처리
                hit_pct = 0.0

            rows.append((current_time, mem_mb, mem_pct, hit_pct))

# TSV로 저장
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("time_s\tnpu_mem_mb\tnpu_mem_used_pct\tprefix_hit_ratio_pct\n")
    for t, mem_mb, mem_pct, hit_pct in rows:
        f.write(f"{t}\t{mem_mb}\t{mem_pct}\t{hit_pct}\n")

print(f"Saved {len(rows)} rows to {OUT_PATH}")