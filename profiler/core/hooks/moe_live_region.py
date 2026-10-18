"""Retain a live native expert kernel, including quantized runtime layouts.

This adapter admits local TP1 only: it never reselects a backend or converts
weights. Native preparation and finalization stay inside the timed region.
"""
import copy
from dataclasses import asdict, fields, is_dataclass
from enum import Enum

import torch


def _layout(value):
    """Stable numerical policy and tensor layouts, without random dummy values."""
    if value is None or type(value) in (str,int,float,bool):
        return value
    if isinstance(value,torch.Tensor):
        return dict(shape=list(value.shape),stride=list(value.stride()),
                    dtype=str(value.dtype),storage_offset=value.storage_offset())
    if isinstance(value,(torch.dtype,torch.device,Enum)):
        return str(value)
    if isinstance(value,(list,tuple)):
        return [_layout(v) for v in value]
    if isinstance(value,dict):
        return {str(k):_layout(v) for k,v in value.items()}
    if is_dataclass(value):
        return {f.name:_layout(getattr(value,f.name)) for f in fields(value)}
    if hasattr(value,'__dict__') and not callable(value):
        return dict(type=f'{type(value).__module__}.{type(value).__name__}',
                    fields=_layout(vars(value)))
    raise NotImplementedError(f'Unrepresented native numerical policy {type(value)}')


def _tensors(value, seen=None):
    """Walk kernel state without following class objects or callable closures."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item, seen)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item, seen)
    elif not isinstance(value, type) and not callable(value) and hasattr(value, '__dict__'):
        yield from _tensors(vars(value), seen)


def tensor_storages(tensors):
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
    return storages


def tensor_bytes(tensors):
    return sum(tensor_storages(tensors).values())


def _clone_tensors(tensors):
    """Clone each physical storage once, retaining native aliasing and strides."""
    storages, memo = {}, {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device),storage.data_ptr())
        if key not in storages:
            raw = torch.empty(0,dtype=torch.uint8,device=tensor.device)
            raw.set_(storage,0,(storage.nbytes(),),(1,))
            storages[key] = raw.clone().untyped_storage()
        view = torch.empty(0,dtype=tensor.dtype,device=tensor.device)
        memo[id(tensor)] = view.set_(storages[key],tensor.storage_offset(),
                                    tensor.size(),tensor.stride())
    return memo


class LiveExpertRegion:
    def __init__(self, runner, router_logits_dtype):
        import vllm
        from vllm.model_executor.layers.fused_moe import modular_kernel as mk

        self.config = runner.moe_config
        layer = runner.routed_experts
        method = layer.quant_method
        self.native_method, self.native_layer = method, layer
        self.kernel = method.moe_kernel
        if self.kernel is None:
            raise NotImplementedError('Native expert method has no initialized kernel')
        self.experts = self.kernel.fused_experts
        self.monolithic = method.is_monolithic
        self.router_logits_dtype = router_logits_dtype
        if self.experts.activation_format() != mk.FusedMoEActivationFormat.Standard:
            raise NotImplementedError('Live native experts require standard token layout')
        self.w13, self.w2 = layer.w13_weight, layer.w2_weight
        self.mapping = layer.expert_map
        if (self.config.activation != layer.activation
                or self.config.num_experts != layer.global_num_experts
                or self.mapping is not None):
            raise ValueError('Native expert activation, global width or local mapping differs from its config')
        self.routing = dict(num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor)
        if self.monolithic and not self.experts.supports_routing_replay_capture():
            raise NotImplementedError('Fused routing needs native routing replay to verify N/A/M')
        cfg = self.config
        self.contract = dict(schema='native-live-expert-region',
            vllm_version=str(vllm.__version__),
            backend=f'{type(self.experts).__module__}.{type(self.experts).__name__}',
            expert_class=f'{type(self.experts).__module__}.{type(self.experts).__name__}',
            quant_method=f'{type(method).__module__}.{type(method).__name__}',
            expert_parallel_config=asdict(cfg.moe_parallel_config),
            global_experts=cfg.num_experts, global_top_k=cfg.experts_per_token,
            local_experts=cfg.num_local_experts, expert_map=None,
            hidden_dim=cfg.hidden_dim,
            intermediate_size=cfg.intermediate_size_per_partition,
            activation=str(cfg.activation), input_dtype=str(cfg.in_dtype),
            weight_dtype=str(self.w13.dtype), workspace='native_shared_manager',
            routing='fused' if self.monolithic else 'external',
            router_logits_dtype=str(router_logits_dtype),
            routing_parameters=_layout(self.routing),routing_method=str(cfg.routing_method),
            quantization=_layout(self.experts.quant_config),
            weight_tensors=dict(w13=_layout(self.w13),w2=_layout(self.w2)),
            activation_parameters={name:_layout(getattr(cfg,name)) for name in (
                'activation_situ_beta','activation_situ_linear_beta',
                'swiglu_limit','swiglu_alpha','swiglu_beta')
                if hasattr(cfg,name)},
            weight_layout='live_processed_native',
            excluded=['gate_projection', 'collectives', 'shared_experts', 'wrapper', 'cpu_time'])
        self.weight_storages = tensor_storages(_tensors(
            (self.w13,self.w2,self.experts.quant_config)))
        self.weight_bytes = sum(self.weight_storages.values())
        self.bank_bytes = tensor_bytes(_tensors((self.w13,self.w2,self.kernel)))

    @torch.inference_mode()
    def prime(self, maximum, indices_dtype):
        """Materialize native bank-local scratch at the acquisition upper bound."""
        cfg = self.config
        hidden = torch.zeros((maximum,cfg.hidden_dim),device=self.w13.device,dtype=cfg.in_dtype)
        ids = torch.arange(maximum*cfg.experts_per_token,device=self.w13.device)
        ids = (ids % cfg.num_experts).reshape(maximum,cfg.experts_per_token).to(indices_dtype)
        weights = torch.full(ids.shape,1/cfg.experts_per_token,device=self.w13.device,dtype=torch.float32)
        result = self.bind(hidden,ids,weights)()
        if result.shape != hidden.shape or not torch.isfinite(result).all().item():
            raise ValueError('Native expert upper-bound preparation has invalid output')
        self.bank_bytes = tensor_bytes(_tensors((self.w13,self.w2,self.kernel)))

    def clone(self):
        # Preserve shared tensor references within each bank, including the
        # scales cached on the expert object. Never rebuild/reselect a backend.
        state = (self.w13, self.w2, self.kernel)
        memo = _clone_tensors(_tensors(state))
        bank = copy.copy(self)
        bank.w13, bank.w2, bank.kernel = copy.deepcopy(state, memo)
        bank.experts = bank.kernel.fused_experts
        if tensor_bytes(_tensors((bank.w13, bank.w2, bank.kernel))) != self.bank_bytes:
            raise ValueError('Cloned native expert storage changed its physical footprint')
        return bank

    def bind(self, hidden, ids, weights, *, output_alias=None):
        if output_alias is not None:
            raise ValueError('The live kernel owns its output allocation')
        cfg = self.config
        if (hidden.ndim != 2 or hidden.shape[1] != cfg.hidden_dim
                or not 1 <= hidden.shape[0] <= cfg.max_num_tokens
                or hidden.dtype != cfg.in_dtype
                or ids.shape != (hidden.shape[0], cfg.experts_per_token)
                or weights.shape != ids.shape or weights.dtype != torch.float32
                or ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError('Live native expert input shape or dtype mismatch')
        if any(t.device != self.w13.device for t in (hidden, ids, weights)):
            raise ValueError('Live native expert inputs must share the weight device')
        if ids.min().item() < 0 or ids.max().item() >= cfg.num_experts:
            raise ValueError('Global expert ID outside native expert bank')
        ordered = ids.sort(dim=1).values
        if (ordered[:, 1:] == ordered[:, :-1]).any().item():
            raise ValueError('Duplicate expert in a token route')
        if not torch.isfinite(hidden).all().item() or not torch.isfinite(weights).all().item():
            raise ValueError('Nonfinite live native expert stimulus')
        common = dict(activation=cfg.activation, global_num_experts=cfg.num_experts,
                      expert_map=self.mapping, apply_router_weight_on_input=False)
        if not self.monolithic:
            def compute():
                return self.kernel.apply(hidden,self.w13,self.w2,weights,ids,**common)
            reference = self.native_method.apply(self.native_layer,hidden,weights,ids,
                                                  shared_experts=None,shared_experts_input=None).clone()
            result = compute()
            if not torch.isfinite(reference).all().item():
                raise ValueError('Native quant-method reference produced nonfinite output')
            torch.testing.assert_close(result,reference,atol=0,rtol=0)
            return compute
        # Fused routers cannot consume externally selected top-k IDs. Supply
        # separated logits and observe their actual routes before timing.
        logits = torch.full((hidden.shape[0], cfg.num_experts), -20.,
                            device=hidden.device, dtype=self.router_logits_dtype)
        logits.scatter_(1, ids.long(), 20.)
        def compute():
            return self.kernel.apply_monolithic(hidden, self.w13, self.w2, logits,
                                                  **common, **self.routing)
        observed = []
        if (self.experts.routing_replay_capture_fn is not None
                or self.experts._routing_replay_buffer is not None):
            raise ValueError('Native routing replay is already owned by another observer')
        self.experts.set_capture_fn(lambda value: observed.append(value.clone()))
        try:
            compute()
        finally:
            self.experts.set_capture_fn(None)
        if len(observed) != 1 or not torch.equal(observed[0].long().sort(dim=1).values,
                                                ids.long().sort(dim=1).values):
            raise ValueError('Native fused router cannot realize the requested N/A/M routes')
        reference = self.native_method.apply_monolithic(self.native_layer,hidden,logits).clone()
        result = compute()
        if not torch.isfinite(reference).all().item():
            raise ValueError('Native quant-method reference produced nonfinite output')
        torch.testing.assert_close(result,reference,atol=0,rtol=0)
        return compute
