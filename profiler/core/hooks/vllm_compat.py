"""Workarounds for vLLM bugs that block the profiler.

Everything here is a shim for a defect in the *installed* vLLM, not for an
intentional API change — those belong in the module that uses the API. Each
entry names the upstream commit that introduced the problem so it can be
deleted the moment a release fixes it.
"""

from __future__ import annotations

import sys
import types


def _install_typeshed_stub() -> bool:
    """Make ``import _typeshed`` succeed, if it doesn't already.

    ``vllm/profiler/utils.py`` does an unguarded
    ``from _typeshed import DataclassInstance`` at module scope, introduced by
    vLLM commit ``fb946a7f89`` ("Make ``mypy`` opt-out instead of opt-in",
    #33205). ``_typeshed`` is a *stub-only* module — it ships with type
    checkers and never exists at runtime — so on vLLM v0.28.0
    ``vllm.profiler.layerwise_profile`` raises ``ModuleNotFoundError`` on
    import. That module is the profiler's single most important dependency:
    without it there are no per-layer CUDA timings and nothing else in this
    package has anything to do.

    ``DataclassInstance`` is referenced only by two annotations in that file
    (``type[DataclassInstance]`` and ``list[DataclassInstance]``). The file
    has no ``from __future__ import annotations``, so Python evaluates them
    when the functions are defined, but nothing ever reads them afterwards —
    a placeholder class satisfies them exactly.

    Returns True if a stub was installed, False if the name already resolved
    (a real ``_typeshed`` on the path, or a stub installed by an earlier
    call). Installing is process-local and idempotent.
    """
    if "_typeshed" in sys.modules:
        return False
    try:
        import _typeshed  # noqa: F401  - probing whether it is importable
    except ModuleNotFoundError:
        pass
    else:
        return False

    stub = types.ModuleType("_typeshed")
    stub.__doc__ = (
        "Placeholder installed by profiler.core.hooks.vllm_compat so that "
        "vllm/profiler/utils.py can be imported. Not the real typeshed."
    )

    class DataclassInstance:  # noqa: D401 - annotation placeholder only
        """Stand-in for typeshed's ``DataclassInstance`` protocol."""

    stub.DataclassInstance = DataclassInstance
    sys.modules["_typeshed"] = stub
    return True


def import_layerwise_profile():
    """Return vLLM's ``layerwise_profile`` context manager.

    Applies :func:`_install_typeshed_stub` first so the import works on vLLM
    releases carrying the ``_typeshed`` defect. On a fixed release the stub is
    never installed and this is a plain import.
    """
    _install_typeshed_stub()
    from vllm.profiler.layerwise_profile import layerwise_profile

    return layerwise_profile
