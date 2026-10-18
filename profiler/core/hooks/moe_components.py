"""Worker-local native component acquisition, with explicit token domains."""
import torch

from .moe_expert_region import NativeExpertRegion
from .moe_hook import single_moe_runner
from .skew_measurement import MARKER
from .eager_measurement import extract_eager_forwards
from ..moe_deployment import MoeTarget
from ..moe_geometry import ExpertPlacement
from ..metadata_names import (MOE_WEIGHT_POLICY, MOE_CONDITIONING_POLICY,
                              EAGER_ATTRIBUTION)


def _deferred_finalize_enabled(config):
    """Read vLLM's effective capability, never its request flag or method.

    Both supported APIs expose ``use_deferred_moe_finalize`` as a boolean
    property. In 0.28 ``defer_moe_finalize`` is a request flag; in 0.31 it is
    a request method, whose truthiness does not describe the running path.
    Unknown APIs are unsafe for this component boundary and fail closed.
    """
    try:
        enabled = config.use_deferred_moe_finalize
    except AttributeError as exc:
        raise NotImplementedError(
            "Native MoE components require use_deferred_moe_finalize"
        ) from exc
    if type(enabled) is not bool:
        raise TypeError("use_deferred_moe_finalize must be a boolean property")
    return enabled


class ProfileCall(torch.nn.Module):
    def __init__(self, call):
        super().__init__()
        self.call = call

    def forward(self):
        return self.call()


class ComponentMeasurement:
    def __new__(cls, model_runner, target, local_budget):
        runner = single_moe_runner(model_runner)
        if (model_runner.vllm_config.quant_config is not None
                or getattr(runner, '_shared_experts', None) is not None
                or getattr(runner, 'routed_output_transform', None) is not None
                or getattr(runner, 'routed_input_transform', None) is not None
                or runner.moe_config.skip_final_all_reduce):
            from .moe_live_components import LiveComponentMeasurement
            try:
                return LiveComponentMeasurement(model_runner, target, local_budget)
            except Exception as exc:
                import traceback
                raise RuntimeError('Native live MoE initialization failed:\n'+traceback.format_exc()) from exc
        return super().__new__(cls)

    def __init__(self, model_runner, target, local_budget):
        from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceNoOP

        self.target = MoeTarget(**target)
        self.runner = single_moe_runner(model_runner)
        runner = self.runner
        if model_runner.vllm_config.quant_config is not None:
            raise NotImplementedError("Quantized MoE requires a quantization-aware component adapter")
        cfg = runner.moe_config
        if (getattr(runner, '_shared_experts', None) is not None
                or _deferred_finalize_enabled(cfg) or cfg.skip_final_all_reduce
                or cfg.hidden_dim != cfg.hidden_dim_unpadded
                or getattr(runner.routed_experts, 'apply_router_weight_on_input', False)):
            raise NotImplementedError("Shared, deferred, padded-hidden or input-weighted MoE needs a separate component contract")
        self.gate = getattr(runner, 'gate', None)
        if self.gate is None:
            parents = [m for m in model_runner.get_model().modules()
                       if runner in m._modules.values() and isinstance(getattr(m, 'gate', None), torch.nn.Module)]
            if len(parents) != 1:
                raise NotImplementedError("External gate binding is absent or ambiguous")
            self.gate = parents[0].gate
        if tuple(self.gate.weight.shape) != (cfg.num_experts, cfg.hidden_dim):
            raise ValueError("The gate must retain the full global expert width")
        self.cfg, metadata = self.target.resolve(cfg, local_budget)
        self.place = ExpertPlacement(self.cfg.num_experts, self.cfg.experts_per_token,
                                     self.target.ep, self.target.rank)
        self.device = self.gate.weight.device
        gen = torch.Generator(device=self.device).manual_seed(0)
        local_e, inter, h = self.cfg.num_local_experts, self.cfg.intermediate_size_per_partition, self.cfg.hidden_dim
        w13 = torch.randn((local_e, self.cfg.w13_num_shards*inter, h), device=self.device,
                          dtype=self.cfg.in_dtype, generator=gen)*.01
        w2 = torch.randn((local_e, h, inter), device=self.device,
                         dtype=self.cfg.in_dtype, generator=gen)*.01
        self.expert = NativeExpertRegion(self.cfg, w13, w2)
        self.checkpoint_weights = (w13,w2)
        self.expert_banks = [self.expert]
        self.reduction = self.expert.experts.finalize_weight_and_reduce_impl()
        if not isinstance(self.reduction, TopKWeightAndReduceNoOP):
            raise NotImplementedError("Post-expert weight reduction needs a separately timed pre-combine region")
        if self.target.dp == 1:
            from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import MoEPrepareAndFinalizeNoDPEPModular
            self.finalize = MoEPrepareAndFinalizeNoDPEPModular()
        else:
            self.finalize = naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPModular(self.target.sequence_parallel)
        probe_rows = min(2, self.cfg.max_num_tokens)
        hidden = torch.randn((probe_rows, h), device=self.device, dtype=self.cfg.in_dtype, generator=gen)
        with torch.inference_mode():
            weights, ids = self._gate(hidden)
        self.indices_dtype = ids.dtype
        if tuple(ids.shape) != (probe_rows, self.cfg.experts_per_token) or weights.dtype != torch.float32:
            raise ValueError("Unexpected native router output contract")
        props = torch.cuda.get_device_properties(self.device)
        self.contract = dict(metadata, schema='moe-components', expert_region=self.expert.contract,
            components=list(self.target.components), modes=['eager'],
            topk_ids_dtype=str(ids.dtype), topk_weights_dtype=str(weights.dtype),
            hidden_dim=h, intermediate_size=inter, input_dtype=str(self.cfg.in_dtype),
            gate_weight_bytes=self.gate.weight.numel()*self.gate.weight.element_size(),
            expert_weight_bytes=w13.numel()*w13.element_size()+w2.numel()*w2.element_size(),
            activation=str(self.cfg.activation),
            measurement_context=MOE_WEIGHT_POLICY,
            expert_conditioning=dict(schema=MOE_CONDITIONING_POLICY,
                l2_bytes=int(props.L2_cache_size),capacity_multiple=2,
                minimum_banks=2,
                quality_relative_tolerance=.03,representative='higher_bank_control',
                local_components='isolated_warm',warmup_complete_cycles=3),
            routing_stimulus='balanced_local_histogram_with_full_global_topk',
            eager_attribution=EAGER_ATTRIBUTION,
            expert_output_reference='independent_eager_clone',
            torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
            synthetic_expert_weights=True,
            gpu=dict(name=props.name, total_memory=props.total_memory,
                     capability=list(torch.cuda.get_device_capability(self.device))),
            excluded=['collectives', 'cpu_time', 'launch_gaps', 'shared_experts'])
        if self.target.dp == 1:
            self._verify_local_boundary(hidden, weights, ids)
            self.contract['local_execution'] = dict(
                prepare='identity', finalize='output_alias_noop', expert_output_alias=True)

    @torch.inference_mode()
    def _verify_local_boundary(self, hidden, weights, ids):
        """Omit DP1 boundaries only after observing their native no-op contract."""
        from torch.autograd import DeviceType
        from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG

        output = torch.empty_like(hidden)
        call = self.expert.bind(hidden, ids, weights, output_alias=output)
        if call() is not output:
            raise NotImplementedError('Native DP1 expert output does not preserve its output alias')
        reference = output.clone()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as hook:
            prepared = self.finalize.prepare(hidden, weights, ids, self.cfg.num_experts,
                self.expert.mapping, False, FUSED_MOE_UNQUANTIZED_CONFIG)
            self.finalize.finalize(output, output, weights, ids, False, self.reduction)
        if (len(prepared) != 5 or prepared[0] is not hidden
                or any(value is not None for value in prepared[1:])):
            raise NotImplementedError('Native DP1 preparation is not an unquantized identity')
        if any(event.device_type() == DeviceType.CUDA and not event.is_user_annotation()
               for event in hook.profiler.kineto_results.events()):
            raise NotImplementedError('Native DP1 prepare/finalize performs unprofiled CUDA work')
        torch.testing.assert_close(output, reference, atol=0, rtol=0)

    def _gate(self, hidden):
        logits, _ = self.gate(hidden)
        return self.runner.router.select_experts(hidden_states=hidden, router_logits=logits,
                                                  topk_indices_dtype=self.finalize.topk_indices_dtype())

    @torch.inference_mode()
    def measure(self, point, iterations, failure_dir=None):
        if point['mode'] != 'eager':
            raise ValueError('Native MoE component acquisition requires eager mode')

        from torch.profiler import record_function
        from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep

        n = point['tokens']
        if point['component'] not in self.target.components:
            raise ValueError('Component is not part of the native target execution')
        if type(iterations) is not int or iterations < 1:
            raise ValueError("Measurement needs positive iterations")
        if point['component'] == 'experts':
            from .moe_conditioned import measure_experts
            return measure_experts(self,point,iterations,failure_dir)
        generator = torch.Generator(device=self.device).manual_seed(n)
        hidden = torch.randn((n, self.cfg.hidden_dim), device=self.device,
                             dtype=self.cfg.in_dtype, generator=generator)
        details = {}
        original_group = naive_dp_ep.get_ep_group
        if point['component'] == 'gate_routing':
            call = lambda: self._gate(hidden)
        elif point['component'] == 'finalize_copy':
            output = torch.empty_like(hidden)
            gathered_n = n * (self.target.ep if self.target.sequence_parallel else self.target.dp)
            fused = torch.empty((gathered_n, self.cfg.hidden_dim), device=self.device, dtype=hidden.dtype)
            ids = torch.zeros((gathered_n, self.cfg.experts_per_token), device=self.device, dtype=self.indices_dtype)
            weights = torch.full(ids.shape, 1/self.cfg.experts_per_token, device=self.device, dtype=torch.float32)
            target = self.target
            class CompletedCombine:
                def combine(self, value, is_sequence_parallel=False):
                    if value is not fused or is_sequence_parallel != target.sequence_parallel:
                        raise ValueError("Unexpected native combine contract")
                    return hidden
            completed = CompletedCombine()
            def call():
                self.finalize.finalize(output, fused, weights, ids, False, self.reduction)
                return output
        else:
            raise ValueError("Unknown MoE component")
        try:
            if point['component'] == 'finalize_copy':
                naive_dp_ep.get_ep_group = lambda: completed
            wrapped = ProfileCall(call)
            for _ in range(3):
                result = wrapped()
            outputs = result if isinstance(result, tuple) else (result,)
            if not all(torch.isfinite(value).all().item() for value in outputs):
                raise ValueError("Non-finite native component output")
            if point['component'] == 'finalize_copy':
                torch.testing.assert_close(result, hidden, atol=0, rtol=0)
            torch.cuda.synchronize()
            context = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                       torch.profiler.ProfilerActivity.CUDA])
            with context as hook:
                for index in range(iterations):
                    with record_function(MARKER+str(index)):
                        wrapped()
            torch.cuda.synchronize()
            raw = extract_eager_forwards(hook.profiler.kineto_results, iterations)
            return dict(raw, verified=True, geometry=details)
        finally:
            naive_dp_ep.get_ep_group = original_group
