"""CUDA-graph dispatch hook.

The profiler measures every latency with ``enforce_eager=True`` --
``layerwise_profile`` builds its tree from per-module CUDA events and
torch.compile fuses those boundaries away, so it has no choice. Production runs
the compiled + cudagraph path, and the difference is a real term the simulator
would otherwise be missing entirely.

Measuring it needs the *same* batch run both ways. Booting twice does not work
well: the two modes then sit in different engines, and boot-to-boot drift
(compile artifacts, memory layout, clock state) came out at 1.4 percentage
points against an effect of 2-5%, while the within-run standard error was
0.1%. So the same quantity measured twice disagreed by more than it is worth.

vLLM has the knob that fixes this. ``dispatch_cg_and_sync_dp`` takes
``need_eager`` (``v1/worker/gpu/dp_utils.py:98``) and returns a descriptor
pinned to ``CUDAGraphMode.NONE`` when it is set, so graph replay can be turned
off for a *single forward* in an otherwise normal engine. Alternating forwards
in one boot makes the ratio a paired measurement and the drift cancels.

What this isolates is graph replay against no graph, with compiled kernels on
both sides. Booting with ``enforce_eager=True`` additionally disables
torch.compile, so the simulator's full term is this times the compile effect,
which needs its own (much less precise) second boot.

The control that says the hook works: on a batch above
``max_cudagraph_capture_size`` vLLM was already dispatching NONE, so the toggle
must be a no-op there. Measured on RTXPRO6000/Llama-3.1-8B it reads 0.9994 at
0.6 sigma, i.e. exactly 1.

Every symbol here is a vLLM internal API and is version-specific.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


class _Toggle:
    """Module-level switch the patched dispatchers read.

    Not a closure variable: the patch is installed once per worker and then
    flipped per forward, and the flipping happens from a different call.

    ``consulted`` counts how often a patched dispatcher actually ran. A step
    sweep whose toggle was never consulted measured nothing, and that is an
    error rather than a saving of zero -- which is exactly how the first
    version of this hook read ~0 us on every MoE model while looking healthy.
    """

    force_none = False
    consulted = 0


@contextmanager
def graph_dispatch_patched() -> Iterator[type[_Toggle]]:
    """Patch **both** runners' dispatch so ``_Toggle`` can force NONE.

    vLLM 0.28 has two model runners and they reach the cudagraph decision by
    different objects, so one patch covers only half the models:

        vllm.v1.worker.gpu.model_runner   (V2)  dispatch_cg_and_sync_dp(...)
                                                -> CudaGraphManager.dispatch
        vllm.v1.worker.gpu_model_runner   (V1)  CudagraphDispatcher.dispatch

    and which one a model gets is ``VllmConfig._is_default_v2_model_runner_model``:
    ``is_default_v2_architecture or not model_config.is_moe``. So **every MoE
    and hybrid model takes V1**, where the V2-only patch is inert -- the V1
    runner does not call ``dispatch_cg_and_sync_dp`` even once. Patching only
    V2 made Qwen3-30B-A3B measure a saving of -1 us against Llama's 587,
    indistinguishable from "cudagraphs buy nothing here".

    Both are patched, and both inject vLLM's *own* force-eager argument rather
    than fabricating a return value -- ``need_eager=True`` on V2,
    ``valid_modes={NONE}`` on V1, which is what vLLM's internal ``force_eager``
    passes. The unused one is harmless.

    Every symbol here is a vLLM internal API and is version-specific.
    """
    import vllm.v1.cudagraph_dispatcher as CGD
    import vllm.v1.worker.gpu.model_runner as MR2
    from vllm.config.compilation import CUDAGraphMode

    orig_v2 = MR2.dispatch_cg_and_sync_dp
    orig_v1 = CGD.CudagraphDispatcher.dispatch

    def patched_v2(*args, **kwargs):
        _Toggle.consulted += 1
        if _Toggle.force_none:
            kwargs["need_eager"] = True
        return orig_v2(*args, **kwargs)

    def patched_v1(self, *args, **kwargs):
        _Toggle.consulted += 1
        if _Toggle.force_none:
            kwargs["valid_modes"] = {CUDAGraphMode.NONE}
        return orig_v1(self, *args, **kwargs)

    MR2.dispatch_cg_and_sync_dp = patched_v2
    CGD.CudagraphDispatcher.dispatch = patched_v1
    try:
        yield _Toggle
    finally:
        MR2.dispatch_cg_and_sync_dp = orig_v2
        CGD.CudagraphDispatcher.dispatch = orig_v1
        _Toggle.force_none = False


def capture_sizes(model_runner) -> list[int]:
    """The token counts vLLM captured a graph for, ascending.

    This is the grid the step sweep should enumerate. vLLM pads a batch up to
    the next captured size and replays *that* graph, so a measurement taken
    anywhere else describes no batch the engine actually runs -- and the
    simulator's lookup is a round-up into this list, which is what makes it
    charge the padding the engine really pays.
    """
    cc = model_runner.vllm_config.compilation_config
    return sorted(int(s) for s in (cc.cudagraph_capture_sizes or []))
