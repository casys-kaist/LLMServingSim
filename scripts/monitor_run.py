"""Run one command with a process-tree RSS watchdog and a CSV resource log.

Example: python3 scripts/monitor_run.py --output outputs/audit/resources.csv \
    --max-rss-gib 48 --min-available-gib 128 -- python3 -m serving ...

RSS sums the process tree, conservatively counting shared pages more than
once. The available-memory floor is host-wide. Container memory limits remain
the hard limit; this watchdog provides an earlier, observable stop.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


GPU_FIELDS = ('uuid', 'temperature.gpu', 'clocks.current.sm', 'clocks.current.memory',
              'utilization.gpu', 'power.draw', 'memory.used')
GPU_COLUMNS = ('gpu_uuid', 'gpu_temperature_c', 'gpu_sm_mhz', 'gpu_memory_mhz',
               'gpu_utilization_pct', 'gpu_power_w', 'gpu_memory_mib')


def gpu_snapshot(uuid):
    """Read only the explicitly selected device; temperature must be available."""
    result = subprocess.run(['nvidia-smi', '--id=' + uuid,
        '--query-gpu=' + ','.join(GPU_FIELDS), '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=5)
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != len(GPU_FIELDS) or rows[0][0].strip() != uuid:
        raise ValueError('GPU telemetry did not identify exactly the requested UUID')
    values = [uuid]
    for cell in rows[0][1:]:
        try:
            value = float(cell)
        except ValueError:
            value = None
        values.append(value if value is not None and math.isfinite(value) else None)
    if values[1] is None:
        raise ValueError('GPU temperature unavailable; refusing an unmonitored run')
    return values


def snapshot(root_pid, tracked=None):
    tracked = {} if tracked is None else tracked
    records = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            status = (directory / "status").read_text().splitlines()
            rss = next((int(s.split()[1]) for s in status
                        if s.startswith("VmRSS:")), 0)
            records[int(directory.name)] = (int(stat[1]), int(stat[2]),
                                             stat[19], rss)
        except (OSError, ValueError, IndexError):
            continue
    members = {pid for pid, (_, group, birth, _) in records.items()
               if pid == root_pid or group == root_pid or tracked.get(pid) == birth}
    while True:
        children = {pid for pid, (parent, _, _, _) in records.items()
                    if parent in members}
        added = children - members
        if not added:
            break
        members.update(added)
    for pid in members:
        tracked[pid] = records[pid][2]
    rss_kib = sum(records[pid][3] for pid in members)
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        mem[key] = int(value.split()[0])
    return (len(members), rss_kib / 2 ** 20,
            mem["MemAvailable"] / 2 ** 20,
            (mem["SwapTotal"] - mem["SwapFree"]) / 2 ** 20)


def stop_group(pgid, tracked):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        # timeout and multiprocessing children can create their own groups.
        # Verify start times to avoid signalling an unrelated reused PID.
        for pid, birth in list(tracked.items()):
            try:
                stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                if stat[19] == birth:
                    os.kill(pid, sig)
            except (ProcessLookupError, FileNotFoundError):
                pass
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        if sig == signal.SIGTERM:
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rss-gib", type=float, default=48)
    parser.add_argument("--min-available-gib", type=float, default=128)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--gpu-uuid", help="Monitor exactly this physical GPU UUID")
    parser.add_argument("--max-gpu-temp-c", type=float,
                        help="Stop at this temperature; required with --gpu-uuid")
    parser.add_argument("--max-swap-growth-gib", type=float,
                        help="Stop if host swap use grows by more than this amount")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    limits = (args.interval, args.timeout, args.max_rss_gib, args.min_available_gib)
    if (not command or not all(math.isfinite(x) for x in limits)
            or args.interval <= 0 or args.timeout <= 0
            or args.max_rss_gib <= 0 or args.min_available_gib < 0):
        parser.error("a command, positive limits, and non-negative available-memory floor are required")
    if bool(args.gpu_uuid) != (args.max_gpu_temp_c is not None):
        parser.error('--gpu-uuid and --max-gpu-temp-c must be supplied together')
    if args.gpu_uuid and (not args.gpu_uuid.startswith('GPU-')
            or not math.isfinite(args.max_gpu_temp_c) or args.max_gpu_temp_c <= 0):
        parser.error('use a physical GPU UUID and a finite positive temperature limit')
    if args.max_swap_growth_gib is not None and (not math.isfinite(args.max_swap_growth_gib)
            or args.max_swap_growth_gib < 0):
        parser.error('--max-swap-growth-gib must be finite and nonnegative')
    _, _, available, initial_swap = snapshot(-1)
    if available < args.min_available_gib:
        parser.error(f"only {available:.1f} GiB available; command not started")
    if args.gpu_uuid:
        try:
            initial_gpu = gpu_snapshot(args.gpu_uuid)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            parser.error(f'GPU preflight failed; command not started: {exc}')
        if initial_gpu[1] >= args.max_gpu_temp_c:
            parser.error(f'GPU is already at {initial_gpu[1]} C; command not started')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Fail on an unwritable log before starting any work.
    summary_path = args.output.with_suffix('.json')
    if summary_path.exists():
        parser.error('resource summary exists; use a new output path')
    stream = args.output.open("x", newline="")
    started = time.monotonic()
    try:
        child = subprocess.Popen(command, start_new_session=True)
    except BaseException:
        stream.close()
        raise
    peak = 0.0
    reason = ""
    tracked = {}
    min_available = available
    max_temperature = None
    max_swap = initial_swap
    interrupted = None

    def terminate(signum, _frame):
        nonlocal interrupted
        interrupted = signum
        raise SystemExit(128 + signum)

    previous_term = signal.signal(signal.SIGTERM, terminate)
    try:
        with stream:
            writer = csv.writer(stream)
            writer.writerow(["elapsed_s", "processes", "rss_gib",
                             "available_gib", "swap_used_gib", "monotonic_s", "epoch_s"]
                            + (list(GPU_COLUMNS) if args.gpu_uuid else []))
            while True:
                count, rss, available, swap = snapshot(child.pid, tracked)
                elapsed = time.monotonic() - started
                peak = max(peak, rss)
                min_available = min(min_available, available)
                max_swap = max(max_swap, swap)
                gpu = []
                if args.gpu_uuid:
                    try:
                        gpu = gpu_snapshot(args.gpu_uuid)
                        max_temperature = max(max_temperature or gpu[1], gpu[1])
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        reason = f'GPU telemetry failed: {exc}'
                writer.writerow([round(elapsed, 3), count, round(rss, 4),
                                 round(available, 4), round(swap, 4),
                                 time.monotonic(), time.time()] + gpu)
                stream.flush()
                if child.poll() is not None:
                    break
                if reason:
                    pass
                elif gpu and gpu[1] >= args.max_gpu_temp_c:
                    reason = f'GPU temperature {gpu[1]} C reached {args.max_gpu_temp_c} C'
                elif rss > args.max_rss_gib:
                    reason = f"RSS {rss:.1f} GiB exceeds {args.max_rss_gib}"
                elif available < args.min_available_gib:
                    reason = f"available RAM fell to {available:.1f} GiB"
                elif elapsed > args.timeout:
                    reason = f"timeout after {elapsed:.1f} s"
                elif (args.max_swap_growth_gib is not None
                      and swap - initial_swap > args.max_swap_growth_gib):
                    reason = f'host swap grew by {swap - initial_swap:.3f} GiB'
                if reason:
                    print(f"monitor: stopping command: {reason}", flush=True)
                    stop_group(child.pid, tracked)
                    break
                time.sleep(args.interval)
    finally:
        # Include grandchildren even when the command itself already exited.
        stop_group(child.pid, tracked)
        child.wait()
        signal.signal(signal.SIGTERM, previous_term)
        summary = dict(status='stopped' if reason or interrupted else
                       ('completed' if child.returncode == 0 else 'failed'),
                       reason=reason or (f'signal {interrupted}' if interrupted else None),
                       command=command, returncode=child.returncode,
                       limits={k: v for k, v in vars(args).items() if k not in ('command', 'output')},
                       peak_rss_gib=peak, min_available_gib=min_available,
                       initial_swap_gib=initial_swap, max_swap_gib=max_swap,
                       max_gpu_temperature_c=max_temperature,
                       elapsed_s=time.monotonic() - started,
                       source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        with summary_path.open('x') as summary_stream:
            json.dump(summary, summary_stream, indent=2)
            summary_stream.write('\n')
        print(f"monitor: peak RSS {peak:.2f} GiB; log {args.output}", flush=True)
    return 124 if reason else child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
