"""Native modular expert region without DP2 or hardware-specific branches.

The caller provides a resolved vLLM config and checkpoint-layout local weights.
Only the post-dispatch expert region is measured; gate, transport and finalize
remain separate. Backend selection and workspace aliasing belong to vLLM.
"""
from dataclasses import asdict
import hashlib
import inspect
from pathlib import Path


class NativeExpertRegion:
    def __init__(self, config, w13, w2):
        import torch
        import vllm
        from vllm.model_executor.layers.fused_moe import modular_kernel as mk
        from vllm.model_executor.layers.fused_moe.config import FUSED_MOE_UNQUANTIZED_CONFIG
        from vllm.model_executor.layers.fused_moe.expert_map_manager import determine_expert_map
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
            convert_to_unquantized_kernel_format, select_unquantized_moe_backend)

        parallel = config.moe_parallel_config
        if (config.has_bias or config.is_lora_enabled or parallel.enable_eplb
                or config.num_experts != config.num_logical_experts):
            raise NotImplementedError("Bias, LoRA, EPLB and redundant experts need separate adapters")
        if config.in_dtype != w13.dtype or w13.dtype != w2.dtype or w13.device != w2.device:
            raise ValueError("Unquantized weights and activation dtype/device must agree")
        if w13.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise NotImplementedError("Quantized experts require a quantization-aware adapter")
        if parallel.tp_size != 1 or parallel.pcp_size != 1:
            raise NotImplementedError("Tensor-sharded experts and PCP are not covered")
        local_e, mapping, _ = determine_expert_map(parallel.ep_size, parallel.ep_rank,
                                                  config.num_experts, "linear")
        inter = config.intermediate_size_per_partition
        if (config.num_local_experts != local_e
                or tuple(w13.shape) != (local_e, config.w13_num_shards * inter, config.hidden_dim)
                or tuple(w2.shape) != (local_e, config.hidden_dim, inter)):
            raise ValueError("Weights do not match the resolved local expert shape")
        backend, cls = select_unquantized_moe_backend(config)
        if cls is None or not issubclass(cls, mk.FusedMoEExpertsModular):
            raise NotImplementedError(f"Native backend {backend} needs a monolithic adapter")
        if cls.activation_format() != mk.FusedMoEActivationFormat.Standard:
            raise NotImplementedError(f"Native backend {backend} needs dispatched token metadata")
        self.config = config
        self.mapping = None if mapping is None else mapping.to(w13.device)
        self.w13, self.w2 = convert_to_unquantized_kernel_format(backend, config, w13, w2)
        self.experts = cls(config, FUSED_MOE_UNQUANTIZED_CONFIG)
        self.impl = mk.FusedMoEKernelModularImpl(None, self.experts)
        sources = (type(self), cls, mk.FusedMoEKernelModularImpl)
        self.contract = dict(
            schema="native-modular-expert-region-v1", vllm_version=vllm.__version__,
            backend=backend.value, expert_class=f"{cls.__module__}.{cls.__name__}",
            expert_parallel_config=asdict(parallel), global_experts=config.num_experts,
            global_top_k=config.experts_per_token, local_experts=local_e,
            expert_map=None if mapping is None else mapping.tolist(),
            hidden_dim=config.hidden_dim, intermediate_size=inter,
            activation=str(config.activation), input_dtype=str(config.in_dtype),
            weight_dtype=str(w13.dtype), workspace="native_shared_manager",
            sources={f"{c.__module__}.{c.__name__}": hashlib.sha256(
                Path(inspect.getfile(c)).read_bytes()).hexdigest() for c in sources},
            excluded=["gate_routing", "collectives", "prepare_finalize", "shared_experts", "cpu_time"])

    def bind(self, hidden, ids, weights, *, output_alias=None):
        """Validate outside measurement; preserve native allocation inside it."""
        import torch
        from vllm.v1.worker.workspace import current_workspace_manager

        cfg = self.config
        if hidden.ndim != 2 or not 1 <= hidden.shape[0] <= cfg.max_num_tokens:
            raise ValueError("Gathered token count is outside the configured measurement range")
        if hidden.shape[1] != cfg.hidden_dim or hidden.dtype != cfg.in_dtype:
            raise ValueError("Hidden shape/dtype differs from the resolved contract")
        if ids.shape != (hidden.shape[0], cfg.experts_per_token) or weights.shape != ids.shape:
            raise ValueError("Routing must retain the complete global top-k width")
        if ids.dtype not in (torch.int32, torch.int64) or weights.dtype != torch.float32:
            raise ValueError("Unexpected routing dtype")
        if any(v.device != self.w13.device for v in (hidden, ids, weights)):
            raise ValueError("All inputs must belong to the measurement device")
        if ids.min().item() < 0 or ids.max().item() >= cfg.num_experts:
            raise ValueError("Global expert ID out of range")
        ordered = ids.sort(dim=1).values
        if cfg.experts_per_token > 1 and (ordered[:, 1:] == ordered[:, :-1]).any().item():
            raise ValueError("A token cannot select the same expert twice")
        if not torch.isfinite(hidden).all().item() or not torch.isfinite(weights).all().item():
            raise ValueError("Non-finite measurement stimulus")
        if current_workspace_manager() is None:
            raise RuntimeError("Initialize the native workspace manager before binding")

        def compute():
            return self.impl._fused_experts(
                in_dtype=hidden.dtype, a1q=hidden, a1q_scale=None,
                w1=self.w13, w2=self.w2, topk_weights=weights, topk_ids=ids,
                activation=cfg.activation, global_num_experts=cfg.num_experts,
                local_num_experts=cfg.num_local_experts, expert_map=self.mapping,
                apply_router_weight_on_input=False, expert_tokens_meta=None,
                output_alias=output_alias)

        return compute
