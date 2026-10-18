"""Local native MoE components with quantization and wrapper work intact."""
from contextlib import ExitStack, contextmanager

import torch

from .eager_measurement import extract_eager_forwards
from .moe_hook import single_moe_runner
from .moe_live_region import LiveExpertRegion, tensor_bytes, tensor_storages
from .skew_measurement import MARKER
from ..metadata_names import MOE_WEIGHT_POLICY, MOE_CONDITIONING_POLICY, EAGER_ATTRIBUTION
from ..moe_deployment import MoeTarget
from ..moe_geometry import ExpertPlacement


@contextmanager
def _scope_method(owner, name, label):
    original = getattr(owner, name)
    had_local = name in vars(owner)
    def scoped(*args, **kwargs):
        with torch.profiler.record_function(label):
            return original(*args, **kwargs)
    setattr(owner, name, scoped)
    try:
        yield
    finally:
        if had_local:
            setattr(owner, name, original)
        else:
            delattr(owner, name)


def _wrapper_measurements(kineto, iterations):
    """Partition actual native CUDA launches; no CPU durations or subtraction."""
    from torch.autograd import DeviceType
    events = list(kineto.events())
    regions = [e for e in events if e.device_type() == DeviceType.CPU
               and e.is_user_annotation() and e.name().startswith('moe_partition:')]
    excluded = set()
    for launch in events:
        if (launch.device_type() != DeviceType.CPU
                or launch.activity_type() not in ('cuda_runtime', 'cuda_driver')):
            continue
        scopes = [r for r in regions if r.device_resource_id() == launch.device_resource_id()
                  and r.start_ns() <= launch.start_ns() and launch.end_ns() <= r.end_ns()]
        if scopes:
            # Shared-expert execution nested in a quant method belongs to the
            # wrapper. The narrowest dynamic scope wins, independently of stream.
            scope = min(scopes, key=lambda e: e.duration_ns())
            if scope.name() == 'moe_partition:excluded':
                excluded.add(launch.correlation_id())
    selected = [e for e in events if not (e.device_type() == DeviceType.CUDA
                and not e.is_user_annotation() and e.correlation_id() in excluded)]
    class Partition:
        def events(self):
            return selected
    # Validate the complete full forward first, then its disjoint remainder.
    full = extract_eager_forwards(kineto, iterations)
    result = extract_eager_forwards(Partition(), iterations)
    result['wrapper_partition'] = dict(
        complete_forward_us=full['per_forward_us'],
        excluded_us=[a-b for a,b in zip(full['per_forward_us'],result['per_forward_us'])],
        ownership='deepest_native_cpu_launch_scope',
        overlapping_streams='sum_of_unique_cuda_activity_durations')
    return result


class LiveComponentMeasurement:
    def __init__(self, model_runner, target, local_budget):
        from .moe_components import _deferred_finalize_enabled
        self.model_runner = model_runner
        self.target = MoeTarget(**target)
        if (self.target.tp, self.target.dp, self.target.ep, self.target.rank) != (1,1,1,0):
            raise NotImplementedError('Live quantized/shared MoE currently requires TP1/DP1/EP1')
        self.runner = single_moe_runner(model_runner)
        runner = self.runner
        cfg = runner.moe_config
        # Native DS/GLM set skip_final_all_reduce for their following fused
        # layer norm. At TP1/EP1 neither branch runs a collective (vLLM
        # MoERunner._maybe_reduce_final_output); retain the flag as identity.
        unsupported = dict(deferred=_deferred_finalize_enabled(cfg),has_bias=cfg.has_bias,
            lora=cfg.is_lora_enabled,eplb=cfg.moe_parallel_config.enable_eplb,
            redundant_experts=cfg.num_experts != cfg.num_logical_experts,
            input_weighted=runner.routed_experts.apply_router_weight_on_input,
            fused_shared_gate=runner._fse_fuse_gate)
        if any(unsupported.values()):
            raise NotImplementedError(f'Unsupported native MoE capabilities: {unsupported}')
        if cfg.hidden_dim != cfg.hidden_dim_unpadded:
            raise NotImplementedError('Padded hidden experts need an explicit output shape contract')
        self.cfg, metadata = self.target.resolve(cfg, local_budget)
        # Kernel shape/config stays native. The requested acquisition bound
        # must fit what the live kernel initialized; do not mutate it afterward.
        if self.cfg.max_num_tokens > cfg.max_num_tokens:
            raise ValueError('Requested token budget exceeds the native initialized expert budget')
        parents = [m for m in model_runner.get_model().modules()
                   if runner in m._modules.values() and isinstance(getattr(m,'gate',None),torch.nn.Module)]
        if len(parents) != 1:
            raise NotImplementedError('Native MoE wrapper with external gate is absent or ambiguous')
        self.parent, = parents
        self.gate = self.parent.gate
        self.gate_hidden_dim = self.gate.weight.shape[1]
        if self.gate.weight.ndim != 2 or self.gate.weight.shape[0] != cfg.num_experts:
            raise ValueError('Native gate must preserve the complete expert width')
        self.device = self.gate.weight.device
        self.place = ExpertPlacement(cfg.num_experts,cfg.experts_per_token,1,0)
        with torch.inference_mode():
            gate_probe = self.gate(torch.zeros((1,self.gate_hidden_dim),
                                               device=self.device,dtype=cfg.in_dtype))
        gate_probe = gate_probe[0] if isinstance(gate_probe,tuple) else gate_probe
        if gate_probe.shape != (1,cfg.num_experts) or not torch.isfinite(gate_probe).all().item():
            raise ValueError('Native gate output violates its complete expert width/dtype contract')
        self.expert = LiveExpertRegion(runner,gate_probe.dtype)
        self.expert_banks = [self.expert]
        self.expert_output_alias = False
        self.indices_dtype = runner.routed_experts.quant_method.topk_indices_dtype or torch.int32
        self.expert.prime(self.cfg.max_num_tokens,self.indices_dtype)
        properties = torch.cuda.get_device_properties(self.device)
        all_weights = list(self.parent.parameters()) + list(self.parent.buffers())
        owned_storages = tensor_storages(all_weights)
        gate_storages = tensor_storages(list(self.gate.parameters())+list(self.gate.buffers()))
        gate_bytes = sum(gate_storages.values())
        if set(gate_storages) & set(self.expert.weight_storages):
            raise ValueError('Gate and routed expert weights share physical storage')
        wrapper_bytes = sum(value for key,value in owned_storages.items()
            if key not in gate_storages and key not in self.expert.weight_storages)
        self.contract = dict(metadata, schema='moe-components',expert_region=self.expert.contract,
            components=['gate_routing','experts','wrapper'], modes=['eager'],
            hidden_dim=self.gate_hidden_dim, expert_hidden_dim=cfg.hidden_dim,
            intermediate_size=cfg.intermediate_size_per_partition_unpadded,
            expert_intermediate_size=cfg.intermediate_size_per_partition,
            gate_hidden_dim=self.gate_hidden_dim,gate_weight_dtype=str(self.gate.weight.dtype),
            gate_weight_bytes=gate_bytes,expert_weight_bytes=self.expert.weight_bytes,
            expert_bank_bytes=self.expert.bank_bytes,
            wrapper_weight_bytes=wrapper_bytes,
            input_dtype=str(cfg.in_dtype),activation=str(cfg.activation),
            topk_ids_dtype=str(self.indices_dtype),topk_weights_dtype='torch.float32',
            gate_routing_boundary='projection_only' if self.expert.monolithic else 'projection_and_routing',
            wrapper_boundary='native_parent_excluding_gate_router_and_routed_kernel',
            wrapper_overlap='sum_of_unique_cuda_activity_durations',
            local_execution=dict(prepare='included_in_experts',finalize='included_in_experts'),
            skip_final_all_reduce=cfg.skip_final_all_reduce,
            native_capabilities=unsupported,
            measurement_context=MOE_WEIGHT_POLICY,
            expert_conditioning=dict(schema=MOE_CONDITIONING_POLICY,
                l2_bytes=int(properties.L2_cache_size),capacity_multiple=2,minimum_banks=2,
                quality_relative_tolerance=.03,representative='higher_bank_control',
                local_components='isolated_warm',warmup_complete_cycles=3),
            routing_stimulus='balanced_local_histogram_with_full_global_topk',
            eager_attribution=EAGER_ATTRIBUTION,expert_output_reference='independent_eager_clone',
            native_method_reference='exact_output_at_each_routing_stimulus',
            synthetic_expert_weights=False,weight_source='live_dummy_loaded_native',
            torch_version=str(torch.__version__),cuda_version=torch.version.cuda,
            gpu=dict(name=properties.name,total_memory=properties.total_memory,
                     capability=list(torch.cuda.get_device_capability(self.device))),
            excluded=['collectives','cpu_time','launch_gaps'])

    def _gate(self, hidden, routed_hidden):
        value = self.gate(hidden)
        logits = value[0] if isinstance(value,tuple) else value
        if self.expert.monolithic:
            return logits
        return self.runner.router.select_experts(hidden_states=routed_hidden,router_logits=logits,
                                                 topk_indices_dtype=self.indices_dtype)

    def _native_state(self):
        """Weights, configuration and callable bindings; caches may grow lazily."""
        weights = tuple((name,t.data_ptr(),tuple(t.shape),tuple(t.stride()),str(t.dtype),
                         None if torch.is_inference(t) else t._version)
                        for name,t in self.parent.named_parameters())
        method = self.runner.routed_experts.quant_method
        bindings = (self.gate.forward,self.runner.router.select_experts,
                    method.apply,method.apply_monolithic)
        if self.runner._shared_experts is not None:
            bindings += (self.runner._shared_experts.forward,)
        return weights,repr(self.runner.moe_config),bindings

    @contextmanager
    def _wrapper_scopes(self):
        with ExitStack() as stack:
            stack.enter_context(_scope_method(self.gate,'forward','moe_partition:excluded'))
            stack.enter_context(_scope_method(self.runner.router,'select_experts','moe_partition:excluded'))
            method = self.runner.routed_experts.quant_method
            stack.enter_context(_scope_method(method,
                'apply_monolithic' if self.expert.monolithic else 'apply','moe_partition:excluded'))
            shared = self.runner._shared_experts
            if shared is not None:
                stack.enter_context(_scope_method(shared,'forward','moe_partition:wrapper'))
            yield

    @torch.inference_mode()
    def measure(self, point, iterations, failure_dir=None):
        if (point['mode'] != 'eager' or type(iterations) is not int or iterations < 1
                or point['component'] not in self.contract['components']):
            raise ValueError('Invalid live native component measurement')
        original = self._native_state()
        if point['component'] == 'experts':
            from .moe_conditioned import measure_experts
            result = measure_experts(self,point,iterations,failure_dir)
            if self._native_state() != original:
                raise ValueError('Expert measurement changed native weights, config or callable bindings')
            result['geometry']['native_state_unchanged'] = True
            return result
        from vllm.forward_context import set_forward_context
        n = point['tokens']
        if not 1 <= n <= self.contract['local_token_budget']:
            raise ValueError('Local native component token count outside contract')
        gen = torch.Generator(device=self.device).manual_seed(n)
        hidden = torch.randn((n,self.gate_hidden_dim),device=self.device,dtype=self.cfg.in_dtype,generator=gen)
        routed_hidden = hidden if hidden.shape[1] == self.cfg.hidden_dim else torch.randn(
            (n,self.cfg.hidden_dim),device=self.device,dtype=self.cfg.in_dtype,generator=gen)
        call = (lambda value:self._gate(value,routed_hidden)) if point['component']=='gate_routing' else self.parent
        with set_forward_context(None,self.model_runner.vllm_config,num_tokens=n):
            for _ in range(3):
                result = call(hidden)
            values = result if isinstance(result,tuple) else (result,)
            if not all(torch.isfinite(v).all().item() for v in values):
                raise ValueError('Nonfinite live native component output')
            torch.cuda.synchronize()
            with self._wrapper_scopes(), torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as hook:
                for index in range(iterations):
                    with torch.profiler.record_function(MARKER+str(index)):
                        call(hidden)
            torch.cuda.synchronize()
        extract = _wrapper_measurements if point['component']=='wrapper' else extract_eager_forwards
        if self._native_state() != original:
            raise ValueError('Local measurement changed native weights, config or callable bindings')
        return dict(extract(hook.profiler.kineto_results,iterations),verified=True,
                    geometry=dict(native_state_unchanged=True))
