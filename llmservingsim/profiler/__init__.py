"""Layerwise profiler for LLMServingSim.

Drives a real vLLM engine with synthetic batches to capture per-layer
CUDA kernel latency. The output (one folder per hardware/model/variant,
subfolders per TP degree) is consumed directly by LLMServingSim's
trace_generator at simulation time.

Module map:
    __main__.py                 CLI dispatch (profile / slice)
    core/                       profiler internals
        runner.py               full-run orchestration
        config.py               Architecture loader + ProfileArgs + engine defaults
        engine.py               vLLM engine lifecycle
        categories.py           profile categories (dense / per_seq / attn / moe)
        skew.py / fit_alpha.py  heterogeneous-decode skew sweep + fit
        writer.py               CSV output + meta.yaml
        logger.py               rich-based logging & progress UI
        hooks/                  vLLM-internal-API touchpoints
            extension.py        vLLM worker extension class
            batch.py            synthetic SchedulerOutput assembly
            timings.py          layerwise_profile tree extraction
            moe_hook.py         FusedMoE forced-routing patcher
    models/                     architecture yamls (one per HF model_type)
    power/                      nvidia-smi / IPMI power-logging shell helpers
    perf/                       profile output root (per hardware/model/variant)
    profile.sh                  editable user-run script
    profile-all.sh              multi-model sweep helper
"""

# ---------------------------------------------------------------------------
# _typeshed shim — runs FIRST, before anything else imports.
# ---------------------------------------------------------------------------
#
# vLLM has an import path that references ``_typeshed.DataclassInstance``
# at runtime. ``_typeshed`` is a typing-only stub module and isn't
# available as a real module, so the import fails with "No module named
# '_typeshed'". The fix is a tiny shim: register a fake ``_typeshed``
# module containing the bare attribute vLLM happens to touch.
#
# This runs in every process that imports ``profiler`` — the host
# (before spin_up() constructs vllm.LLM) and every vLLM worker
# process (before the Extension class is loaded via
# worker_extension_cls). Placing it in ``profiler/__init__.py``
# covers both paths automatically.

import sys as _sys
import types as _types

if "_typeshed" not in _sys.modules:
    _shim = _types.ModuleType("_typeshed")
    _shim.DataclassInstance = object  # type: ignore[attr-defined]
    _sys.modules["_typeshed"] = _shim
    del _shim

del _sys, _types


__version__ = "1.0.0"
