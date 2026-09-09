"""Log every engine step's shape and its interval, for validating the cost model.

`bench` records per-request latencies and a 1-second timeseries. Neither can
answer the question a cost-model residual needs answering: **which step shapes
is the simulator wrong about?** A tick's `running / gen_throughput` averages
over ~16 steps of mixed kinds, and a tick that carries any prefill says nothing
about how much, or in which of its steps.

This inserts one line per `execute_model` call into
`$VLLM_STEP_SHAPE_LOG.<pid>` -- one file per process, because every rank of a
TP group and every engine of a DP group calls `execute_model`:

    {"t": .., "num_tokens": .., "n_reqs": .., "n_prefill": ..,
     "prefill_tokens": ..}

The **interval between consecutive lines** is the step time, and that is the
right measure precisely because it needs no synchronisation: a saturated
engine's loop is GPU-bound, so the rate at which `execute_model` is entered is
the rate the GPU finishes steps. Timing the call itself would catch only the
CPU-side launch -- it returns long before the GPU is done -- and adding a sync
would drain the pipeline, which inflates a step by 3-9x (Llama-3.1-8B: 14.5% of
a step shows as idle when synced per step against 4.4% when the steps run
back-to-back).

Off unless `VLLM_STEP_SHAPE_LOG` is set, so the patch is inert in an image.

Both runners are patched. vLLM 0.28 chooses between them with
`VllmConfig._is_default_v2_model_runner_model` --
`is_default_v2_architecture or not model_config.is_moe` -- so MoE and hybrid
models take `v1/worker/gpu_model_runner.py` and everything else
`v1/worker/gpu/model_runner.py`. The two share the class name `GPUModelRunner`
and only `__module__` separates them, so covering one records nothing for half
the models. That is the trap the cudagraph hook fell into, where a V2-only
patch measured a saving of -1 us on an MoE model and looked like an answer.

Shape comes from the scheduler output rather than the batch:
`num_scheduled_tokens` maps request id -> tokens this step, so a request with
more than one is a prefill chunk and one with exactly one is a decode -- the
same classification `trace_generator` makes, which is what makes the two sides
comparable row by row.

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
                # interleaves their lines and the interval between two
                # consecutive ones stops being any step's duration -- which
                # read as prefill steps costing 7-9x what they do.
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
        root / "v1" / "worker" / "gpu_model_runner.py",       # V1: MoE, hybrid
        root / "v1" / "worker" / "gpu" / "model_runner.py",   # V2: the rest
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
