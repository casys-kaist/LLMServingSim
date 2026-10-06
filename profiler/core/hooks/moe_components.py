"""Worker-local native component acquisition, with explicit token domains."""
import hashlib
import inspect
from pathlib import Path

import torch

from .moe_expert_region import NativeExpertRegion
from .moe_hook import single_moe_runner
from .skew_measurement import MARKER
from .graph_measurement import extract_graph_forwards
from .eager_measurement import extract_eager_forwards
from ..moe_deployment import MoeTarget
from ..moe_geometry import ExpertPlacement


class ProfileCall(torch.nn.Module):
    def __init__(self, call):
        super().__init__()
        self.call = call

    def forward(self):
        return self.call()


class ComponentMeasurement:
    def __init__(self, model_runner, target, local_budget):
        from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import TopKWeightAndReduceNoOP
        from ..moe_conditioning import expert_conditioning
        from .moe_conditioned import measure_experts

        self.target = MoeTarget(**target)
        if self.target.dp == 1:
            raise NotImplementedError("The native component contract currently covers DP+EP; DP1 needs its own prepare/finalize adapter")
        self.runner = single_moe_runner(model_runner)
        runner = self.runner
        if model_runner.vllm_config.quant_config is not None:
            raise NotImplementedError("Quantized MoE requires a quantization-aware component adapter")
        cfg = runner.moe_config
        if (getattr(runner, '_shared_experts', None) is not None
                or cfg.defer_moe_finalize or cfg.skip_final_all_reduce
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
        self.finalize = naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPModular(self.target.sequence_parallel)
        hidden = torch.randn((2, h), device=self.device, dtype=self.cfg.in_dtype, generator=gen)
        with torch.inference_mode():
            weights, ids = self._gate(hidden)
        self.indices_dtype = ids.dtype
        if tuple(ids.shape) != (2, self.cfg.experts_per_token) or weights.dtype != torch.float32:
            raise ValueError("Unexpected native router output contract")
        sources = (type(self), type(self.gate), type(runner.router), type(self.finalize), type(self.reduction))
        props = torch.cuda.get_device_properties(self.device)
        self.contract = dict(metadata, schema='moe-components-v1', expert_region=self.expert.contract,
            components=['gate_routing', 'experts', 'finalize_copy'], modes=['eager', 'graph'],
            topk_ids_dtype=str(ids.dtype), topk_weights_dtype=str(weights.dtype),
            hidden_dim=h, intermediate_size=inter, input_dtype=str(self.cfg.in_dtype),
            gate_weight_bytes=self.gate.weight.numel()*self.gate.weight.element_size(),
            expert_weight_bytes=w13.numel()*w13.element_size()+w2.numel()*w2.element_size(),
            activation=str(self.cfg.activation),
            measurement_context='native_rotated_expert_weights_v1',
            expert_conditioning=dict(schema='rotated-weight-cycles-v2',
                l2_bytes=int(props.L2_cache_size),capacity_multiple=2,
                minimum_banks=2,
                quality_relative_tolerance=.03,representative='higher_bank_control',
                local_components='isolated_warm',warmup_complete_cycles=3),
            routing_stimulus='balanced_local_histogram_with_full_global_topk',
            graph_attribution='native_graph_launch_correlation_v1',
            graph_memory_pool='independent_per_variant',
            eager_attribution='native_eager_launch_correlation_v1',
            graph_output_reference='independent_pre_capture_clone',
            torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
            synthetic_expert_weights=True,
            source_sha256={f'{c.__module__}.{c.__name__}': hashlib.sha256(
                Path(inspect.getfile(c)).read_bytes()).hexdigest() for c in (*sources,
                    extract_graph_forwards, extract_eager_forwards, MoeTarget, ExpertPlacement,
                    expert_conditioning,measure_experts)},
            gpu=dict(name=props.name, uuid=str(props.uuid), total_memory=props.total_memory,
                     capability=list(torch.cuda.get_device_capability(self.device))),
            excluded=['collectives', 'cpu_time', 'launch_gaps', 'shared_experts'])

    def _gate(self, hidden):
        logits, _ = self.gate(hidden)
        return self.runner.router.select_experts(hidden_states=hidden, router_logits=logits,
                                                  topk_indices_dtype=self.finalize.topk_indices_dtype())

    @torch.inference_mode()
    def measure(self, point, iterations, failure_dir=None):
        from torch.profiler import record_function
        from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep

        n = point['tokens']
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
            if point['mode'] == 'graph':
                # Native expert outputs may alias a shared workspace. Keep
                # independent values before capture can overwrite that storage.
                expected_values = tuple(value.clone() for value in outputs)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured_result = call()
                def replay():
                    graph.replay()
                    return captured_result
                wrapped = ProfileCall(replay)
                captured = wrapped()
                torch.cuda.synchronize()
                captured_values = captured if isinstance(captured, tuple) else (captured,)
                if len(expected_values) != len(captured_values):
                    raise ValueError('Captured component output structure changed')
                for expected, actual in zip(expected_values,captured_values):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            elif point['mode'] != 'eager':
                raise ValueError("Unknown measurement execution mode")
            context = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                       torch.profiler.ProfilerActivity.CUDA])
            with context as hook:
                for index in range(iterations):
                    with record_function(MARKER+str(index)):
                        wrapped()
            torch.cuda.synchronize()
            if point['mode'] == 'graph':
                try:
                    raw = extract_graph_forwards(hook.profiler.kineto_results, iterations)
                except Exception as exc:
                    if failure_dir is not None:
                        import tempfile
                        Path(failure_dir).mkdir(parents=True, exist_ok=True)
                        diagnostic = Path(tempfile.mkdtemp(prefix='graph-', dir=failure_dir))/'trace.json'
                        hook.export_chrome_trace(str(diagnostic))
                        raise ValueError(f'{exc}; original CUDA trace: {diagnostic}') from exc
                    raise
                return dict(raw, verified=True, geometry=details)
            raw = extract_eager_forwards(hook.profiler.kineto_results, iterations)
            return dict(raw, verified=True, geometry=details)
        finally:
            naive_dp_ep.get_ep_group = original_group
