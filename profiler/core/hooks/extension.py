"""vLLM worker extension.

Registered via ``worker_extension_cls="profiler.core.hooks.extension.Extension"``
when constructing the ``vllm.LLM``. vLLM instantiates one Extension per
TP-rank worker process and exposes its methods through
``llm.collective_rpc(method_name, args=...)``.

The sole public method here is ``fire()``: it takes a serialized Shot
plus a catalog slice (the subset of the layer map relevant to the
category being profiled), runs the synthetic batch through
``model_runner.execute_model`` under ``layerwise_profile``, and
returns per-layer CUDA timings.

Measurement protocol per shot:
    1 warmup forward (discarded) — amortises JIT / paged-buffer setup
    N timed forwards inside ``layerwise_profile`` — the hook aggregates
        ``cuda_time_us`` across invocations; ``extract_samples``
        divides by ``invocations`` to return the per-call mean.

N defaults to ``ProfileArgs.measurement_iterations`` (3). A single
timed sample can swing 15-25%% on large GEMMs due to DVFS / boost
jitter; averaging cuts that noise floor dramatically.
"""

from __future__ import annotations

from typing import Any

from profiler.core.hooks.batch import Shot, assemble_scheduler_output
from profiler.core.hooks.moe_hook import (
    ExpertRoute,
    force_moe_routing,
    single_moe_runner,
)
from profiler.core.hooks.timings import extract_samples


class Extension:
    """Worker-side profiling entry point.

    vLLM instantiates this class inside each TP worker process and
    injects ``self.model_runner`` via attribute assignment before any
    ``collective_rpc`` call.
    """

    def fire(
        self,
        shot_dict: dict[str, Any],
        slice_: dict[str, dict[str, Any]],
        kind: str,
        iterations: int = 3,
    ) -> list[dict[str, Any]]:
        """Run one profiling shot and return per-layer timings.

        Args:
            shot_dict: Serialized ``Shot``; rehydrated inside the worker.
            slice_: Serialized catalog slice
                ``{canonical_name: {"vllm": cls, "within": parent, ...}}``
                scoped to the category we're profiling (so timings for
                unrelated layers aren't returned).
            kind: One of ``"dense"``, ``"per_sequence"``, ``"attention"``,
                ``"moe"``. Used to decide whether to forge MoE routing.
            iterations: Number of timed forward passes (averaged via
                the hook's invocation count). Default 3.

        Returns:
            List of ``TimingSample`` as plain dicts (pickled back to host).
        """
        shot = Shot.hydrate(shot_dict)
        iterations = max(1, int(iterations))

        def _fresh_batch():
            # Rebuild the synthetic SchedulerOutput on every forward so
            # prior-iteration KV writes / request state don't bleed into
            # the next measurement.
            batch, _ = assemble_scheduler_output(shot, self.model_runner)
            return batch

        # -- warm-up run, result discarded -----------------------------
        # The first forward pays for JIT compilation, CUDA context
        # setup, paged-attention buffer allocation. We also call
        # sample_tokens to exercise the sampler path (if execute_model
        # returns None it means the scheduler consumed everything and
        # sample_tokens finalizes the step).
        warmup_out = self.model_runner.execute_model(_fresh_batch())
        if warmup_out is None:
            self.model_runner.sample_tokens(None)

        # -- optional MoE routing forge --------------------------------
        route: ExpertRoute | None = None
        if kind == "moe":
            if shot.experts is None or "activated" not in shot.experts:
                raise ValueError(
                    "moe shot missing experts.activated payload"
                )
            moe_runner = single_moe_runner(self.model_runner)
            num_tokens = sum(new for new, _ in shot.requests)
            route = ExpertRoute.forge(
                moe_runner,
                num_tokens=num_tokens,
                activated_experts=int(shot.experts["activated"]),
            )

        # -- measured runs (N iterations, averaged) -------------------
        # Local import so that profiler/__init__.py doesn't require
        # vllm.profiler to be importable at package-import time.
        #
        # vLLM's layerwise_profile hook accumulates ``cuda_time_us``
        # and ``invocations`` across every forward inside its context.
        # ``extract_samples`` divides one by the other, so running
        # execute_model N times here yields the per-call mean — the
        # cheap statistical fix for DVFS / boost-clock jitter that
        # single-sample measurements don't mitigate.
        from profiler.core.hooks.vllm_compat import import_layerwise_profile

        layerwise_profile = import_layerwise_profile()

        with force_moe_routing(route):
            with layerwise_profile() as hook:
                for _ in range(iterations):
                    measured_out = self.model_runner.execute_model(_fresh_batch())
                    if measured_out is None:
                        self.model_runner.sample_tokens(None)

        stats = hook.results.convert_stats_to_dict()
        summary = stats["summary_stats"]

        samples = extract_samples(summary, slice_)
        return [s.as_dict() for s in samples]
