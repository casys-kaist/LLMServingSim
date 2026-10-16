"""Prepare assigned KV pages with vLLM's dummy-weight initializer."""

# Bound the temporary FP16 conversion made by vLLM for FP8 tensors. Each
# slab is a separate initializer call, with the upstream per-tensor seed.
FP8_SLAB_BYTES = 16 * 1024**2


def _state_tensors(value):
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _state_tensors(child)
    else:
        raise TypeError("Unsupported recurrent cache container")


def _attention_pages(layer, spec, cache_dtype, num_blocks, used_blocks):
    import torch
    from vllm.platforms import current_platform
    from vllm.v1.kv_cache_interface import KVQuantMode

    cache = layer.kv_cache
    if not isinstance(cache, torch.Tensor) or not cache.numel():
        raise ValueError("Missing typed attention cache")
    if spec.kv_quant_mode not in (KVQuantMode.NONE, KVQuantMode.FP8_PER_TENSOR):
        raise ValueError("Dummy KV initialization does not support packed data/scale caches")
    if not cache.is_floating_point():
        # A byte tensor is not necessarily FP8: sparse indexers and packed
        # quantizers can mix payloads and scales in the same storage.
        if cache.dtype != torch.uint8 or spec.kv_quant_mode != KVQuantMode.FP8_PER_TENSOR:
            raise ValueError("Dummy KV initialization requires a floating-point cache layout")
        if cache_dtype in ("fp8", "fp8_e4m3"):
            dtype = current_platform.fp8_dtype()
        elif cache_dtype == "fp8_e5m2":
            dtype = torch.float8_e5m2
        else:
            raise ValueError(f"Unsupported dummy KV dtype: {cache_dtype}")
        cache = cache.view(dtype)

    # Let the backend identify the logical block axis; do not assume that
    # blocks precede K/V, heads or tokens. Kernel blocks may subdivide each
    # manager block, while the request's IDs always name manager blocks.
    backend = layer.get_attn_backend()
    if not hasattr(backend, "get_kv_cache_shape"):
        # vLLM 0.31 exposes typed logical [B,H,N,C] views, independently of
        # the physical layout/strides. C is bytes in the common shape API.
        from vllm.v1.kv_cache_interface import compute_layer_kv_cache_shape_bytes
        if cache.ndim != 4 or cache.shape[0] % num_blocks:
            raise ValueError("Invalid logical KV cache view")
        factor = cache.shape[0] // num_blocks
        if factor < 1 or spec.block_size % factor:
            raise ValueError("Invalid kernel/manager KV block ratio")
        expected = compute_layer_kv_cache_shape_bytes(
            spec, num_blocks, spec.block_size // factor)
        actual = (*cache.shape[:-1], cache.shape[-1] * cache.element_size())
        if actual != tuple(expected):
            raise ValueError(f"Logical KV cache shape mismatch: {actual} != {expected}")
        return cache.narrow(0, 0, used_blocks * factor), 0
    shape = backend.get_kv_cache_shape(
        -1, spec.storage_block_size, spec.num_kv_heads, spec.head_size,
        cache_dtype_str=cache_dtype)
    axes = [i for i, size in enumerate(shape) if size == -1]
    if len(axes) != 1 or len(shape) != cache.ndim:
        raise ValueError("Cannot identify the attention cache block axis")
    axis = axes[0]
    if cache.shape[axis] % num_blocks:
        raise ValueError("Cache blocks do not match the manager allocation")
    factor = cache.shape[axis] // num_blocks
    if factor < 1:
        raise ValueError("Empty physical cache block allocation")
    return cache.narrow(axis, 0, used_blocks * factor), axis


def initialize_dummy_cache(runner, batch):
    """Initialize only assigned physical pages; keep null/unassigned pages intact."""
    import torch
    from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
    from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode, MambaSpec, UniformTypeKVCacheSpecs
    from .cache_pages import assigned_group_spans, select_pages
    requests = batch.scheduled_new_reqs
    if not requests:
        return
    config = runner.kv_cache_config
    num_blocks = int(config.num_blocks)
    spans = assigned_group_spans(requests, config.kv_cache_groups, num_blocks)
    has_history = any(request.num_computed_tokens > 0 for request in requests)
    context = runner.compilation_config.static_forward_context
    views, seen = [], set()
    for index, group in enumerate(config.kv_cache_groups):
        first, count = spans[index]
        for name in group.layer_names:
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[name]
            layer = context[name]
            if isinstance(spec, MambaSpec):
                selected = []
                for cache in _state_tensors(layer.kv_cache):
                    if not cache.ndim or cache.shape[0] != num_blocks:
                        raise ValueError('Cannot identify recurrent cache pages')
                    selected.append((select_pages(cache, 0, first, count, num_blocks), 0, True))
            elif isinstance(spec, AttentionSpec):
                if not has_history:
                    continue
                cache_dtype = ('auto' if spec.kv_quant_mode == KVQuantMode.NONE else
                    getattr(spec, 'cache_dtype_str', None) or runner.vllm_config.cache_config.cache_dtype)
                # Request the full allocation only as a view; the original
                # resolver validates the installed backend's layout/block axis.
                full, axis = _attention_pages(layer, spec, cache_dtype, num_blocks, num_blocks)
                selected = [(select_pages(full, axis, first, count, num_blocks), axis, False)]
            else:
                raise TypeError(f'Unsupported profiling cache spec: {type(spec).__name__}')
            for view, axis, zero in selected:
                key = (view.device, view.data_ptr(), view.dtype, tuple(view.shape), tuple(view.stride()))
                if key not in seen:
                    seen.add(key)
                    views.append((view, axis, zero))
    with torch.inference_mode():
        for view, axis, zero in views:
            if zero:
                view.zero_()
            elif torch.finfo(view.dtype).bits >= 16:
                initialize_single_dummy_weight(view)
            else:
                elements_per_page = view.numel() // view.shape[axis]
                if elements_per_page > FP8_SLAB_BYTES:
                    raise ValueError('One FP8 cache page exceeds initialization bound')
                step = max(1, FP8_SLAB_BYTES // elements_per_page)
                for start in range(0, view.shape[axis], step):
                    initialize_single_dummy_weight(view.narrow(
                        axis, start, min(step, view.shape[axis] - start)))
    torch.cuda.synchronize()
