"""Optionally log scheduled shapes and CPU execute_model entry intervals.

This inserts one line per `execute_model` call into
`$VLLM_STEP_SHAPE_LOG.<pid>` -- one file per process, because every rank of a
TP group and every engine of a DP group calls `execute_model`:

    {"t": .., "num_tokens": .., "n_reqs": .., "n_prefill": ..,
     "prefill_tokens": ..}

Consecutive timestamps measure CPU-entry cadence, not GPU execution duration or
per-step completion. CPU scheduling, asynchronous execution and pipeline depth
can change the interval. This observer adds Python/file-I/O overhead and must
not be used as a replacement for the profiler's CUDA-activity collector.

Off unless `VLLM_STEP_SHAPE_LOG` is set, so the patch is inert in an image.

Both registered runner sources are patched: `v1/worker/gpu_model_runner.py`
and `v1/worker/gpu/model_runner.py`. vLLM 0.31.0 can select either at runtime.
Python execute_model entry still runs when model work uses CUDA graph replay.

Shape comes from the scheduler output rather than the batch:
`num_scheduled_tokens` maps request id -> scheduled tokens. The legacy
`n_prefill`/`prefill_tokens` labels use the q > 1 heuristic, not request-phase
metadata: one-token prefills and speculative multi-token decodes are not
classified correctly by that heuristic. Interpret those fields accordingly.

Run:  python3 scripts/patches/vllm_step_shape_log.py     (idempotent)
"""

from __future__ import annotations

import sys
from pathlib import Path

MARK = "# --- llmservingsim step-shape log ---"

BLOCK = '''        # --- llmservingsim step-shape log ---
        import os as _lss_os
        if _lss_os.environ.get("VLLM_STEP_SHAPE_LOG"):
            try:
                import json as _lss_json, time as _lss_time
                _lss_ns = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
                _lss_rec = {
                    "t": _lss_time.perf_counter(),
                    "num_tokens": sum(_lss_ns.values()),
                    "n_reqs": len(_lss_ns),
                    "n_prefill": sum(1 for _v in _lss_ns.values() if _v > 1),
                    "prefill_tokens": sum(_v for _v in _lss_ns.values() if _v > 1),
                }
                # One file per process. Every rank of a TP group and every
                # engine of a DP group calls execute_model, so a shared path
                # interleaves their lines and destroys each process's
                # CPU-entry cadence.
                _lss_p = "%s.%d" % (
                    _lss_os.environ["VLLM_STEP_SHAPE_LOG"], _lss_os.getpid())
                with open(_lss_p, "a") as _lss_f:
                    _lss_f.write(_lss_json.dumps(_lss_rec) + chr(10))
            except Exception:
                pass
'''


def patch_file(path):
    src = path.read_text()
    if MARK in src:
        return "already patched"
    lines = src.split("\n")
    for i, line in enumerate(lines):
        if line.strip() != "def execute_model(":
            continue
        # Walk to the end of the signature and insert straight after it. A
        # docstring below becomes a bare string statement, which is harmless.
        j = i
        while j < len(lines) and not lines[j].rstrip().endswith(":"):
            j += 1
        lines.insert(j + 1, BLOCK)
        path.write_text("\n".join(lines))
        return "patched at line %d" % (i + 1)
    return "no execute_model found"


def main():
    import vllm

    root = Path(vllm.__file__).parent
    targets = [
        root / "v1" / "worker" / "gpu_model_runner.py",       # V1
        root / "v1" / "worker" / "gpu" / "model_runner.py",   # V2
    ]
    rc = 0
    for t in targets:
        if not t.exists():
            print("[step-shape-log] missing %s" % t)
            rc = 1
            continue
        print("[step-shape-log] %s: %s" % (t.name, patch_file(t)))
    return rc


if __name__ == "__main__":
    sys.exit(main())
