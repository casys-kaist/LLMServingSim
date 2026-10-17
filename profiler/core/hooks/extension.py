"""Worker-side RPCs backed by lightweight native CUDA-activity capture.

The common collector records explicit module scopes and native CUDA activities
without Python stacks, tensor shapes or memory snapshots. Warmup and validation
remain outside timing; the catalog controls matching and normalization.
"""

from __future__ import annotations

from typing import Any

from profiler.core.hooks.batch import Shot
from profiler.core.hooks.history import complete_forward, prepare_history
from profiler.core.hooks.moe_hook import (
    ExpertRoute,
    force_moe_routing,
    single_moe_runner,
)
from profiler.core.hooks.sampler_shim import wrap_sampler_for_profiling
from profiler.core.hooks.runner_capture import capture, verified_warmup
from profiler.core.hooks.activity_collector import coverage_report


class Extension:
    """Worker-side profiling entry point.

    vLLM instantiates this class inside each TP worker process and
    injects ``self.model_runner`` via attribute assignment before any
    ``collective_rpc`` call.
    """

    def moe_component_initialize(self, target, local_budget):
        from .moe_components import ComponentMeasurement
        self._moe_components = None
        self._moe_components = ComponentMeasurement(self.model_runner, target, local_budget)
        return self._moe_components.contract

    def moe_component_measure(self, point, iterations, failure_dir=None):
        try:
            return self._moe_components.measure(point, iterations, failure_dir)
        except Exception as exc:
            import traceback
            raise RuntimeError(f"Native MoE point {point}:\n{traceback.format_exc()}") from exc

    def moe_component_release(self):
        self._moe_components = None

    def skew_initialize(self):
        from .skew_measurement import initialize
        return initialize(self.model_runner)

    def attention_lookup_contract(self, profile_tp):
        from .attention_lookup import runtime_semantic
        return runtime_semantic(self.model_runner, profile_tp)

    def skew_measure(self, shot_dict, catalog, iterations=3, scope_contexts=None,
                     attention_lookup_semantic=None):
        from .capture_scope_contract import normalize_contexts
        normalize_contexts(scope_contexts, catalog)
        if type(iterations) is not int or iterations < 1:
            raise ValueError('A positive timed-forward count is required')
        from .skew_measurement import measure
        from .attention_lookup import verify
        attention_lookup_sha256 = verify(self.model_runner, attention_lookup_semantic)
        return measure(self.model_runner, shot_dict, catalog, iterations, scope_contexts=scope_contexts,
                       attention_lookup_sha256=attention_lookup_sha256)

    def fire(
        self,
        shot_dict: dict[str, Any],
        slice_: dict[str, dict[str, Any]],
        kind: str,
        iterations: int = 3,
        scope_contexts: dict[str, Any] | None = None,
        component_layout_sha256: str | None = None,
        attention_lookup_semantic: dict[str, Any] | None = None,
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
        if type(iterations) is not int or iterations < 1:
            raise ValueError('A positive timed-forward count is required')
        # Reject malformed contracts before sampler mutation, cache setup,
        # warmup or any measured GPU work.
        from profiler.core.measurement import layerwise_measurement
        attention_lookup_sha256 = None
        if kind == 'attention':
            from .attention_lookup import verify
            attention_lookup_sha256 = verify(self.model_runner, attention_lookup_semantic)
        elif attention_lookup_semantic is not None:
            raise ValueError('Attention lookup identity on another category')
        acquisition = layerwise_measurement(slice_, iterations, scope_contexts=scope_contexts,
                                           component_layout_sha256=component_layout_sha256,
                                           attention_lookup_sha256=attention_lookup_sha256)

        # vLLM 0.28's V2 model runner -- which every dense model takes --
        # holds a sampler that is not an nn.Module, so it never becomes a
        # profile node. Give it a module scope before anything fires.
        wrap_sampler_for_profiling(self.model_runner)

        # Dummy KV initialization prepares the requested operating point;
        # it is not part of the measured query latency.
        _fresh_batch = prepare_history(self.model_runner, shot)

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

        # -- warm-up run, result discarded -----------------------------
        # Warm the same expert distribution as the measured forwards. A
        # natural gate can activate a different set and leave shape-dependent
        # expert setup in the first timed call. Use separate contexts so both
        # warmup and timed execution must actually consume forced routing.
        with force_moe_routing(route):
            verified_warmup(self.model_runner, shot,
                            lambda: complete_forward(self.model_runner, _fresh_batch()))

        # -- measured runs (N iterations, averaged) -------------------
        with force_moe_routing(route):
            measured = capture(self.model_runner,
                               lambda: complete_forward(self.model_runner, _fresh_batch()),
                               slice_, iterations, scope_contexts=scope_contexts)
        if kind == 'mtp':
            missing = set(slice_) - set(measured['layer_us'])
            if missing:
                raise ValueError('MTP acquisition is incomplete; inspect callable coverage before '
                                 f'saving timings: {sorted(missing)}')
        return [dict(layer=name, microseconds=value, **acquisition)
                for name, value in measured["layer_us"].items()]

    def coverage(
        self,
        shot_dict: dict[str, Any],
        slice_: dict[str, dict[str, Any]],
        iterations: int = 1,
        scope_contexts: dict[str, Any] | None = None,
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
        if type(iterations) is not int or iterations < 1:
            raise ValueError('A positive timed-forward count is required')
        from .capture_scope_contract import normalize_contexts
        normalize_contexts(scope_contexts, slice_)

        # vLLM 0.28's V2 model runner -- which every dense model takes --
        # holds a sampler that is not an nn.Module, so it never becomes a
        # profile node. Give it a module scope before anything fires.
        wrap_sampler_for_profiling(self.model_runner)

        _fresh_batch = prepare_history(self.model_runner, shot)

        verified_warmup(self.model_runner, shot,
                        lambda: complete_forward(self.model_runner, _fresh_batch()))

        measured = capture(self.model_runner,
                           lambda: complete_forward(self.model_runner, _fresh_batch()),
                           slice_, iterations, diagnostic=True, scope_contexts=scope_contexts)
        report = coverage_report(measured)
        return report
