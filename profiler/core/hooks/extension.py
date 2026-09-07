"""vLLM worker extension.

Registered via ``worker_extension_cls="profiler.core.hooks.extension.Extension"``
when constructing the ``vllm.LLM``. vLLM instantiates one Extension per
TP-rank worker process and exposes its methods through
``llm.collective_rpc(method_name, args=...)``.

The main public method here is ``fire()``: it takes a serialized Shot
plus a catalog slice (the subset of the layer map relevant to the
category being profiled), runs the synthetic batch through
``model_runner.execute_model`` under ``layerwise_profile``, and
returns per-layer CUDA timings.

``coverage()`` runs the same forward and the same matching rules but reports
what the catalog *failed* to bind instead of what it bound -- see
``timings.CoverageReport``.

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
from profiler.core.hooks.cudagraph_hook import (
    capture_sizes,
    graph_dispatch_patched,
)
from profiler.core.hooks.moe_hook import (
    ExpertRoute,
    force_moe_routing,
    single_moe_runner,
)
from profiler.core.hooks.sampler_shim import wrap_sampler_for_profiling
from profiler.core.hooks.timings import attribute_tree, extract_samples


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

        # vLLM 0.28's V2 model runner -- which every dense model takes --
        # holds a sampler that is not an nn.Module, so it never becomes a
        # profile node. Give it a module scope before anything fires.
        wrap_sampler_for_profiling(self.model_runner)

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
        from vllm.profiler.layerwise_profile import layerwise_profile

        with force_moe_routing(route):
            with layerwise_profile() as hook:
                for _ in range(iterations):
                    measured_out = self.model_runner.execute_model(_fresh_batch())
                    if measured_out is None:
                        self.model_runner.sample_tokens(None)

        stats = hook.results.convert_stats_to_dict()
        summary = stats["summary_stats"]

        samples = extract_samples(summary, slice_, iterations=iterations)
        return [s.as_dict() for s in samples]

    def capture_sizes(self) -> list[int]:
        """The token counts this engine captured a graph for, ascending.

        The step sweep enumerates exactly these, because vLLM pads a batch up
        to the next one and replays *that* graph -- a measurement taken
        anywhere else describes no batch the engine runs.
        """
        return capture_sizes(self.model_runner)

    def step_time_paired(
        self,
        shot_dict: dict[str, Any],
        iterations: int = 60,
        warmups: int = 5,
    ) -> dict[str, list[float]]:
        """Time one batch with and without graph replay, alternating.

        Returns ``{"graph": [...], "none": [...]}`` in microseconds, one entry
        per pair, measured with CUDA events around ``execute_model`` plus
        ``sample_tokens`` -- the whole step, with no profiler in the way.

        Alternating inside one engine is the point: the two modes otherwise
        need two boots, and boot-to-boot drift (1.4pp measured) swamps the
        effect (2-5%). Paired, the standard error is 0.1-0.2%.

        Both sides run *compiled* kernels; only graph replay differs. The
        compile half of the term needs its own boot at
        ``enforce_eager=True``.

        Batch assembly is deliberately outside the timed region, and rebuilt
        per forward so prior-iteration KV writes do not bleed into the next
        measurement -- the same rule ``fire`` follows.
        """
        import torch

        shot = Shot.hydrate(shot_dict)
        iterations = max(1, int(iterations))
        wrap_sampler_for_profiling(self.model_runner)

        def _fresh_batch():
            batch, _ = assemble_scheduler_output(shot, self.model_runner)
            return batch

        def _one() -> float:
            batch = _fresh_batch()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            out = self.model_runner.execute_model(batch)
            if out is None:
                self.model_runner.sample_tokens(None)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) * 1000.0     # ms -> us

        graph: list[float] = []
        none: list[float] = []
        with graph_dispatch_patched() as toggle:
            toggle.consulted = 0
            # Warm both paths: the no-graph path compiles its own kernels on
            # first use, and charging that to the first pair would bias it.
            for forced in (False, True):
                toggle.force_none = forced
                for _ in range(max(1, int(warmups))):
                    _one()
            for _ in range(iterations):
                toggle.force_none = False
                graph.append(_one())
                toggle.force_none = True
                none.append(_one())
            consulted = toggle.consulted
        if consulted == 0:
            # Neither runner's dispatch ran, so nothing was toggled and both
            # columns are the same execution. A saving of zero measured this
            # way is indistinguishable from a real one -- which is how the
            # V2-only patch read -1 us on every MoE model. Fail instead.
            raise RuntimeError(
                "step_time_paired: no cudagraph dispatch was intercepted, so "
                "the two columns are the same execution and the saving is "
                "meaningless. The runner is "
                f"{type(self.model_runner).__module__}; the hook patches both "
                "vllm.v1.worker.gpu.model_runner.dispatch_cg_and_sync_dp (V2) "
                "and vllm.v1.cudagraph_dispatcher.CudagraphDispatcher.dispatch "
                "(V1), so a vLLM version that reaches the decision by a third "
                "route needs a third patch."
            )
        return {"graph": graph, "none": none, "dispatches": consulted}

    def coverage(
        self,
        shot_dict: dict[str, Any],
        slice_: dict[str, dict[str, Any]],
        iterations: int = 1,
    ) -> dict[str, Any]:
        """Run one shot and report which of its CUDA time the catalog binds.

        Same measurement protocol as :meth:`fire` (one warmup, then timed
        forwards) and the same matching rules, but ``slice_`` here is the
        **whole** catalog rather than one category's: coverage is a property
        of the catalog as a whole, and a layer bound in the wrong category
        still binds.

        MoE routing is deliberately not forged. Which experts fire changes the
        cost of the ``moe`` block, not whether anything binds it, and forging
        would need an ``experts`` payload that has nothing to do with the
        question being asked.
        """
        shot = Shot.hydrate(shot_dict)
        iterations = max(1, int(iterations))

        # vLLM 0.28's V2 model runner -- which every dense model takes --
        # holds a sampler that is not an nn.Module, so it never becomes a
        # profile node. Give it a module scope before anything fires.
        wrap_sampler_for_profiling(self.model_runner)

        def _fresh_batch():
            batch, _ = assemble_scheduler_output(shot, self.model_runner)
            return batch

        warmup_out = self.model_runner.execute_model(_fresh_batch())
        if warmup_out is None:
            self.model_runner.sample_tokens(None)

        from vllm.profiler.layerwise_profile import layerwise_profile

        with layerwise_profile() as hook:
            for _ in range(iterations):
                measured_out = self.model_runner.execute_model(_fresh_batch())
                if measured_out is None:
                    self.model_runner.sample_tokens(None)

        summary = hook.results.convert_stats_to_dict()["summary_stats"]
        return attribute_tree(summary, slice_).as_dict()
