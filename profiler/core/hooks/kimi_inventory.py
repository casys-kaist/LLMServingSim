"""Untimed native tensor-layout observations for exact Kimi duration replay.

No matrix or cache contents are transferred. Six native MLA scalar scales are
read outside capture; alias/view checks inspect metadata only. This is not a
resident-memory inventory and supplies no timing value.
"""
import hashlib
from pathlib import Path


SOURCES = {
    'kimi_k3/nvidia/model.py': 'models/kimi_k3/nvidia/model.py',
    'kimi_k3/nvidia/kda.py': 'models/kimi_k3/nvidia/kda.py',
    'kimi_k3/nvidia/mla.py': 'models/kimi_k3/nvidia/mla.py',
    'kimi_k3/nvidia/ops/attn_res.py': 'models/kimi_k3/nvidia/ops/attn_res.py',
    'quantization/mxfp4.py': 'model_executor/layers/quantization/mxfp4.py',
    'fused_moe/oracle/mxfp4.py': 'model_executor/layers/fused_moe/oracle/mxfp4.py',
    'fused_moe/experts/deep_gemm_moe.py': 'model_executor/layers/fused_moe/experts/deep_gemm_moe.py',
}


def shape_dtype(tensor):
    if tensor is None or tensor.device.type != 'cuda' or tensor.is_meta:
        raise ValueError('A native CUDA tensor is required for logical layout facts')
    return [list(tensor.shape), str(tensor.dtype)]


def same_view(left, right):
    """Exact aliases, not just two tensors sharing a larger allocation."""
    return (left.device == right.device and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and tuple(left.stride()) == tuple(right.stride())
            and left.storage_offset() == right.storage_offset()
            and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr())


def merge_facts(target, observed):
    for name, value in observed.items():
        if name in target and target[name] != value:
            raise ValueError('Kimi component layouts disagree across native layers: ' + name)
        target[name] = value


def input_id_tensor(runner):
    """Read the native input allocation, never infer a dtype from the model.

    v1/worker/gpu_model_runner.py stores a CpuGpuBuffer; the 0.31 V2
    v1/worker/gpu/model_runner.py stores InputBuffers.input_ids directly.
    Reject ambiguous layouts rather than choosing whichever attribute exists.
    """
    legacy = getattr(runner, 'input_ids', None)
    buffers = getattr(runner, 'input_buffers', None)
    modern = getattr(buffers, 'input_ids', None)
    if (legacy is None) == (modern is None):
        raise ValueError('Missing or ambiguous native input token buffer')
    tensor = modern if modern is not None else getattr(legacy, 'gpu', None)
    shape_dtype(tensor)
    return tensor


def kda_facts(module, name, preparation, config):
    derived = preparation['postload']['derived_buffers']
    matches = [row for row in derived if row['name'] == name]
    initialized = (len(matches) == 1 and matches[0].get('source_copy_equal') is True
                   and matches[0].get('finite') is True)
    return dict(kda_prefill_backend=module.kda_prefill_backend,
        kda_decode_backend=module.kda_decode_backend,
        kda_in_proj_shape=shape_dtype(module.in_proj_qkvgfab.weight)[0],
        kda_conv_shape_dtype=shape_dtype(module.conv1d.weight),
        kda_decode_conv_shape_dtype=shape_dtype(module.decode_conv1d_weight),
        kda_norm_shape_dtype=shape_dtype(module.o_norm.weight),
        kda_decode_norm_shape_dtype=shape_dtype(module.decode_norm_weight),
        kda_A_log_shape_dtype=shape_dtype(module.A_log),
        kda_dt_bias_shape_dtype=shape_dtype(module.dt_bias),
        kda_derived_initialized=initialized,
        kda_state_shapes=[list(shape) for shape in module.get_state_shape()],
        kda_state_dtypes=[str(dtype) for dtype in module.get_state_dtype()],
        kda_spec_tokens=module.num_spec, kda_recoverssm=module.use_recoverssm,
        kda_mamba_cache_mode=config.cache_config.mamba_cache_mode)


def mla_facts(module, config):
    weight = module.kv_b_proj.weight
    shape_dtype(weight)
    if not weight.is_contiguous():
        raise ValueError('Native MLA KV-B must be a contiguous weight')
    heads, latent = module.num_local_heads, module.kv_lora_rank
    key, value = module.qk_nope_head_dim, module.v_head_dim
    # Contiguous view and slicing create tensor metadata only, no CUDA work.
    native = weight.view(heads, key + value, latent)
    uk, uv = module.W_UK_T, module.W_UV
    observed_key, observed_value = shape_dtype(uk), shape_dtype(uv)
    if observed_key[1] != observed_value[1]:
        raise ValueError('Absorbed MLA views have different dtypes')
    spec = module.get_kv_cache_spec(config)
    return dict(mla_nope=module.rotary_emb is None,
        mla_output_gate=module.use_output_gate, mla_backend=module.attn_backend.get_name(),
        mla_impl=type(module.impl).__module__ + '.' + type(module.impl).__name__,
        mla_prefill_backend=type(module.prefill_backend).__module__ + '.' + type(module.prefill_backend).__name__,
        mla_cache_head_dtype=[spec.head_size, str(spec.dtype)],
        mla_cache_quant_mode=spec.kv_quant_mode.name,
        mla_cache_scale_values=[float(getattr(module, name).item()) for name in
            ('_q_scale', '_k_scale', '_v_scale', '_q_scale_inv', '_k_scale_inv', '_one_scale')],
        mla_absorbed_shape_dtype=[observed_key[0], observed_value[0], observed_key[1]],
        mla_absorbed_alias_kv_b=[view.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
                                for view in (uk, uv)],
        mla_absorbed_matches_source=(same_view(uk, native[:, :key])
                                     and same_view(uv, native[:, key:].transpose(1, 2))))


def moe_facts(module):
    runner = module.experts
    experts = runner.routed_experts
    method = experts.quant_method
    kernel = method.moe_kernel.fused_experts
    w13, w2 = experts.w13_weight, experts.w2_weight
    s13, s2 = experts.w13_weight_scale, experts.w2_weight_scale
    shared = runner._shared_experts._layer
    top_k = runner.router.top_k
    if type(top_k) is not int or top_k <= 0 or top_k != runner.moe_config.experts_per_token:
        raise ValueError('Native router and expert kernel top-k disagree')
    aliases = (shared is module.shared_experts
        and runner.routed_output_transform is module.routed_output_transform
        and module.routed_output_transform.norm is module.routed_expert_norm
        and module.routed_output_transform.up_proj is module.routed_expert_up_proj
        and same_view(experts.e_score_correction_bias, module.gate.e_score_correction_bias))
    return dict(moe_backend=method.mxfp4_backend.name,
        moe_backend_configured=method.moe.moe_backend,
        moe_geometry=[module.gate.weight.shape[1], module.moe_hidden_size,
            module.padded_moe_intermediate_size, w13.shape[0], top_k, module.num_shared_experts],
        moe_activation=[method.moe.activation.value, method.moe.activation_situ_beta,
                        method.moe.activation_situ_linear_beta],
        moe_w13_shape_dtype=shape_dtype(w13), moe_w2_shape_dtype=shape_dtype(w2),
        moe_s13_shape_dtype=shape_dtype(s13), moe_s2_shape_dtype=shape_dtype(s2),
        moe_scale_aliases_kernel=same_view(s13, kernel.w1_scale) and same_view(s2, kernel.w2_scale),
        moe_shared_transform_bias_aliases=aliases,
        moe_router_weight_logits_bias_dtypes=[str(module.gate.weight.dtype),
            str(module.gate.out_dtype), str(module.gate.e_score_correction_bias.dtype)])


def ordinary_linear_facts(modules, linear_base):
    # KDA stores convolution weights in ColumnParallelLinear but consumes
    # them with conv kernels. Its float32 layout is observed by kda_facts;
    # it is not one of the ordinary GEMM weights whose precision is checked here.
    conv_ids = {id(module.conv1d) for _, module in modules
                if type(module).__module__ + '.' + type(module).__name__
                == 'vllm.models.kimi_k3.nvidia.kda.KimiK3DeltaAttention'}
    methods, dtypes = set(), set()
    for _, module in modules:
        if isinstance(module, linear_base) and id(module) not in conv_ids:
            methods.add(type(module.quant_method).__name__)
            dtypes.add(shape_dtype(module.weight)[1])
    if len(methods) != 1 or len(dtypes) != 1:
        raise ValueError('Kimi ordinary linear method/precision is not uniform')
    return dict(ordinary_linear_method=methods.pop(), ordinary_weight_dtype=dtypes.pop())


def observe(runner, preparation):
    import torch
    import vllm
    from vllm.model_executor.layers.linear import LinearBase

    model = runner.get_model()
    config = runner.vllm_config
    language = model.language_model
    head, embedding = language.lm_head, language.model.embed_tokens
    head_shape, head_dtype = shape_dtype(head.weight)
    embedding_shape, _ = shape_dtype(embedding.weight)
    processor = language.logits_processor
    if (type(processor).__module__ != 'vllm.model_executor.layers.logits_processor'
            or processor.logits_as_input or processor.soft_cap is not None
            or processor.scale != 1.0
            or type(head.quant_method).__name__ not in ('UnquantizedEmbeddingMethod', 'UnquantizedLinearMethod')):
        raise ValueError('Unreviewed native Kimi head projection route')
    input_tensor = input_id_tensor(runner)
    root = Path(vllm.__file__).parent
    endpoint_source = hashlib.sha256((root / 'model_executor/layers/logits_processor.py').read_bytes()).hexdigest()
    if endpoint_source != '35cf8235cb529ba2d37358fe9b9e5a29f40b5c3cbcc9e6750c8480c51bd7da83':
        raise ValueError('Native logits projection source has changed')
    parallel = config.parallel_config
    facts = dict(endpoint_source_sha256=endpoint_source,
        source_hashes={name: hashlib.sha256((root / path).read_bytes()).hexdigest()
                               for name, path in SOURCES.items()},
        device_capability=list(torch.cuda.get_device_capability()),
        parallel=[parallel.tensor_parallel_size, parallel.pipeline_parallel_size,
                  parallel.data_parallel_size,
                  parallel.tensor_parallel_size * parallel.data_parallel_size if parallel.enable_expert_parallel else 1],
        enforce_eager=config.model_config.enforce_eager, model_dtype=str(config.model_config.dtype),
        input_id_dtype=str(input_tensor.dtype),
        logits_dtype=str(processor.head_dtype) if processor.head_dtype is not None else head_dtype,
        embedding_shape=embedding_shape, head_shape=head_shape,
        head_embedding_alias=head.weight.untyped_storage().data_ptr() == embedding.weight.untyped_storage().data_ptr(),
        norm_dtype=str(language.model.norm.weight.dtype))
    modules = list(model.named_modules())
    for name, module in modules:
        kind = type(module).__module__ + '.' + type(module).__name__
        if kind == 'vllm.models.kimi_k3.nvidia.kda.KimiK3DeltaAttention':
            merge_facts(facts, kda_facts(module, name, preparation, config))
        elif kind == 'vllm.models.kimi_k3.nvidia.mla.MultiHeadLatentAttention':
            merge_facts(facts, mla_facts(module, config))
        elif kind == 'vllm.models.kimi_k3.nvidia.model.KimiMoE':
            merge_facts(facts, moe_facts(module))
    facts.update(ordinary_linear_facts(modules, LinearBase))
    return facts
