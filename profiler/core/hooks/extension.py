"""Worker-side RPCs backed by lightweight native CUDA-activity capture.

The common collector records explicit module scopes and native CUDA activities
without Python stacks, tensor shapes or memory snapshots. Warmup and validation
remain outside timing; the catalog controls matching and normalization.
"""

from __future__ import annotations

from typing import Any

from profiler.core.hooks.batch import Shot
from profiler.core.hooks.history import complete_forward, prepare_history
from profiler.core.hooks.sampler_shim import wrap_sampler_for_profiling
from profiler.core.hooks.runner_capture import capture, verified_warmup
from profiler.core.hooks.activity_collector import coverage_report
from profiler.core.hooks.dummy_weights import install as _install_dummy_weights

# vLLM resolves the worker extension before constructing/loading the model.
_install_dummy_weights()


class Extension:
    """Worker-side profiling entry point.

    vLLM instantiates this class inside each TP worker process and
    injects ``self.model_runner`` via attribute assignment before any
    ``collective_rpc`` call.
    """

    def dummy_weight_preparation(self):
        from .dummy_weights import preparation_facts
        return preparation_facts(self.model_runner.get_model())

    def sampling_runtime(self):
        """Read acquisition device/backend facts without creating CUDA graphs."""
        from .sampling_runtime import snapshot
        return snapshot(self)

    def native_profile_admission(self, architecture, descriptor):
        from profiler.core.callable_profile import profile_for
        profile = profile_for(architecture)
        if profile is None:
            raise ValueError('Native runtime admission requires a declared profile')
        return profile.adapter.inspect_runtime(self.model_runner, descriptor)

    def component_runtime_admission(self, descriptor, shot_dict=None):
        """Read and validate an explicitly declared native execution contract."""
        from .runtime_admission import validate_descriptor, collect_runtime_facts, admit_runtime
        validate_descriptor(descriptor)
        return admit_runtime(descriptor, collect_runtime_facts(self.model_runner), shot=shot_dict)

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
                     attention_lookup_semantic=None, component_layout_contract=None):
        from .capture_scope_contract import normalize_contexts
        normalize_contexts(scope_contexts, catalog)
        if type(iterations) is not int or iterations < 1:
            raise ValueError('A positive timed-forward count is required')
        from .skew_measurement import measure
        from .attention_lookup import verify
        attention_lookup_sha256 = verify(self.model_runner, attention_lookup_semantic)
        return measure(self.model_runner, shot_dict, catalog, iterations, scope_contexts=scope_contexts,
                       attention_lookup_sha256=attention_lookup_sha256,
                       component_layout_contract=component_layout_contract)

    def fire(
        self,
        shot_dict: dict[str, Any],
        slice_: dict[str, dict[str, Any]],
        kind: str,
        iterations: int = 3,
        scope_contexts: dict[str, Any] | None = None,
        component_layout_sha256: str | None = None,
        attention_lookup_semantic: dict[str, Any] | None = None,
        component_layout_contract: dict[str, Any] | None = None,
        native_profile_contract: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run one profiling shot and return per-layer timings.

        Args:
            shot_dict: Serialized ``Shot``; rehydrated inside the worker.
            slice_: Serialized catalog slice
                ``{canonical_name: {"vllm": cls, "within": parent, ...}}``
                scoped to the category we're profiling (so timings for
                unrelated layers aren't returned).
            kind: Ordinary category name; MoE uses component RPCs.
            iterations: Number of timed forward passes (averaged via
                the hook's invocation count). Default 3.

        Returns:
            List of ``TimingSample`` as plain dicts (pickled back to host).
        """
        if kind == "moe":
            raise ValueError("Whole-block MoE acquisition was removed; use native components")
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
        from profiler.core import component_layouts as layouts
        invocation_contract = layouts.prepare_invocation_contract(
            component_layout_contract, component_layout_sha256, slice_,
            self.model_runner.vllm_config if component_layout_contract is not None else None,
            scope_contexts=scope_contexts)
        runtime_sha256 = None
        if component_layout_contract is not None and component_layout_contract.get('runtime') is not None:
            runtime = component_layout_contract['runtime']
            admitted = self.component_runtime_admission(runtime['descriptor'], shot_dict)
            runtime_sha256 = admitted['runtime_sha256']
            if runtime_sha256 != runtime['runtime_sha256']:
                raise ValueError('Live component runtime changed before this shot')
        native_profile_sha256 = None
        if native_profile_contract is not None:
            from profiler.core.callable_profile import profile_for
            from .callable_workload import canonical
            from profiler.core.measurement import native_scope_token, semantic_identity
            profile = profile_for(native_profile_contract['architecture'])
            if profile is None:
                raise ValueError('Native acquisition contract has no declared profile')
            profile.validate_scope(native_profile_contract)
            native_profile_sha256 = native_scope_token(native_profile_contract)
            cached = getattr(self, '_native_profile_scope', None)
            if cached != native_profile_sha256:
                observed = self.native_profile_admission(
                    native_profile_contract['architecture'], native_profile_contract['descriptor'])
                if canonical(semantic_identity(observed)) != canonical(semantic_identity(native_profile_contract['runtime'])):
                    raise ValueError('Actual native runtime differs before acquisition')
                self._native_profile_scope = native_profile_sha256
        acquisition = layerwise_measurement(slice_, iterations, scope_contexts=scope_contexts,
            component_layout_sha256=component_layout_sha256, runtime_sha256=runtime_sha256,
            attention_lookup_sha256=attention_lookup_sha256, native_profile_sha256=native_profile_sha256)

        # vLLM 0.28's V2 model runner -- which every dense model takes --
        # holds a sampler that is not an nn.Module, so it never becomes a
        # profile node. Give it a module scope before anything fires.
        wrap_sampler_for_profiling(self.model_runner)

        # Dummy KV initialization prepares the requested operating point;
        # it is not part of the measured query latency.
        _fresh_batch = prepare_history(self.model_runner, shot)

        warmup = verified_warmup(self.model_runner, shot,
                        lambda: complete_forward(self.model_runner, _fresh_batch()))
        measured = capture(self.model_runner,
                           lambda: complete_forward(self.model_runner, _fresh_batch()),
                           slice_, iterations, scope_contexts=scope_contexts)
        if native_profile_contract is not None:
            profile.validate_capture(native_profile_contract, measured, slice_, iterations, warmup)
        if invocation_contract is not None:
            architecture, choices, expected = invocation_contract
            from profiler.core.categories import _split_linear_attention_shot
            prefill, decode = _split_linear_attention_shot(shot)
            regime = "mixed" if prefill and decode else "prefill" if prefill else "decode"
            layouts.validate_invocation_report(architecture, choices, coverage_report(measured),
                                               expected, iterations, slice_, regime)
        if kind == 'mtp':
            missing = set(slice_) - set(measured['layer_us'])
            if missing:
                raise ValueError('MTP acquisition is incomplete; inspect callable coverage before '
                                 f'saving timings: {sorted(missing)}')
        return [dict(layer=name, microseconds=value, **acquisition)
                for name, value in measured["layer_us"].items()]

    def fire_callable(self, workload, catalog, scope_contexts):
        """Run a declared native callable with independently planned contexts."""
        from .callable_workload import execute
        return execute(workload, catalog, scope_contexts)

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

        Coverage executes the model's natural routing.
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
