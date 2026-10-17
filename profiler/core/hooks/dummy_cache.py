"""Prepare assigned KV pages with vLLM's dummy-weight initializer."""

# Bound source tensors for packed caches and upstream's temporary FP16 FP8
# initialization. Each slab retains the upstream initializer's default seed.
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


def _packed_attention_pages(layer, spec, num_blocks):
    """Resolve only the audited opaque FP8 layouts; never reinterpret integers.

    Format dispatch uses cache specs/backend layout, not the model identity.
    vLLM 0.31 binds these logical [B,1,N,C] views as [B,N,C].
    """
    import torch
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, KVQuantMode

    if not isinstance(spec, MLAAttentionSpec) or not hasattr(spec, 'cache_role'):
        # Older typed MLA caches keep the original resolver authoritative.
        # In particular, vLLM0.28 has no SparseCacheRole declaration.
        return None
    from vllm.v1.kv_cache_interface import SparseCacheRole
    kind = None
    shuffled = False
    if spec.cache_dtype_str == 'fp8_ds_mla':
        kind, width = 'fp8_ds_mla', 656
        if (spec.kv_quant_mode != KVQuantMode.FP8_PER_TENSOR
                or spec.head_size != 576 or spec.state_content_size_bytes != width
                or spec.cache_role != SparseCacheRole.SPARSE):
            raise ValueError('Unsupported fp8_ds_mla spec')
    elif spec.cache_role == SparseCacheRole.INDEXER:
        # Typed floating-point index caches use the ordinary backend layout.
        # INDEXER is a cache role, not a promise of opaque DeepSeek byte records.
        if spec.dtype.is_floating_point:
            return None
        # This exact backend consumes FP8 D128 plus one FP32 scale/token.
        # Other indexer formats (including V4/MXFP4) must not fall through.
        backend = layer.get_attn_backend()
        if (backend.get_name() != 'DEEPSEEK_V32_INDEXER'
                or spec.head_size != 132 or spec.state_content_size_bytes != 132
                or spec.kv_quant_mode != KVQuantMode.NONE
                or spec.cache_dtype_str is not None):
            raise ValueError('Unsupported packed indexer spec/backend')
        kind, width = 'index_fp8_128', 132
        shuffled = getattr(layer, 'uses_shuffled_layout', None)
        if type(shuffled) is not bool:
            raise ValueError('Packed indexer shuffle layout must be explicit')
    else:
        return None
    cache = layer.kv_cache
    if (not isinstance(cache, torch.Tensor) or cache.dtype != torch.uint8
            or spec.dtype != torch.uint8 or cache.ndim != 3
            or type(num_blocks) is not int or num_blocks < 2
            or spec.num_kv_heads != 1 or spec.num_heads != 1
            or spec.head_size_v != 0 or spec.tokens_per_state != 1
            or spec.model_version is not None
            or cache.shape[0] < num_blocks or cache.shape[0] % num_blocks):
        raise ValueError('Unsupported opaque cache geometry')
    ratio = cache.shape[0] // num_blocks
    if (spec.block_size % ratio or cache.shape[1] != spec.block_size // ratio
            or cache.shape[2] != width or cache.stride(2) != 1
            or cache.stride(1) != width
            or cache.stride(0) < cache.shape[1] * width
            or cache.stride(0) % 4 or cache.storage_offset() % 4
            or (ratio > 1 and cache.stride(0) != cache.shape[1] * width)
            or (shuffled and cache.shape[1] % 16)):
        raise ValueError('Unsupported opaque cache strides/block subdivision')
    return cache, kind, shuffled


def _quantize_packed_fp8(values, *, indexer):
    """NVIDIA e4m3 + FP32 power-of-two scales from pinned native writers.

    Native fused_norm_rope uses different clamp positions for main/index KV:
    common/kernels.py:62-73 and348-376. No CPU time enters measurement.
    """
    import torch
    values = values.to(torch.float32)
    amax = values.abs().amax(dim=-1, keepdim=True)
    scale = (amax.clamp_min(1e-4) / 448.0 if indexer
             else (amax / 448.0).clamp_min(1e-4))
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    return (values / scale).to(torch.float8_e4m3fn), scale


def _write_packed_mla(cache, values, rope):
    """Write byte records: 512 FP8 +4 FP32 scales +64 BF16 RoPE."""
    import torch
    blocks, tokens, _ = cache.shape
    if values.shape != (blocks, tokens, 512) or rope.shape != (blocks, tokens, 64):
        raise ValueError('Wrong packed MLA source shape')
    quant, scales = _quantize_packed_fp8(values.reshape(blocks, tokens, 4, 128), indexer=False)
    cache[..., :512].copy_(quant.reshape(blocks, tokens, 512).view(torch.uint8))
    cache[..., 512:528].view(torch.float32).copy_(scales.squeeze(-1))
    cache[..., 528:656].view(torch.bfloat16).copy_(rope)


def _write_packed_index(cache, values, shuffled):
    """Write separate block planes, not 128B+4B interleaved token records."""
    import torch
    blocks, tokens, _ = cache.shape
    if values.shape != (blocks, tokens, 128):
        raise ValueError('Wrong packed index source shape')
    quant, scales = _quantize_packed_fp8(values, indexer=True)
    if shuffled:
        # Native value_off: token//16*16*128 + token%16*16
        #                 + dim//16*16*16 + dim%16.
        quant = quant.reshape(blocks, tokens // 16, 16, 8, 16).transpose(2, 3).contiguous()
    rows = cache.as_strided((blocks, tokens * 132), (cache.stride(0), 1))
    rows[:, :tokens * 128].copy_(quant.reshape(blocks, tokens * 128).view(torch.uint8))
    rows[:, tokens * 128:].view(torch.float32).copy_(scales.squeeze(-1))


def _initialize_packed_cache(cache, kind, shuffled, initializer):
    """Initialize assigned pages only from upstream uniform(-1e-3,1e-3).

    FP8 logical inputs use FP16, as upstream's FP8 dummy initializer does;
    RoPE is BF16. Pack with native scale formulas, not random scale bytes or
    arbitrary unit scales. This is synthetic history, not a native forward.
    """
    import torch
    from vllm.platforms import current_platform

    # NVIDIA E4M3 is the audited format. CPU tensors are used only by tests.
    if cache.device.type != 'cpu' and (
            not current_platform.is_cuda()
            or current_platform.fp8_dtype() != torch.float8_e4m3fn):
        raise ValueError('Packed dummy history supports NVIDIA E4M3 only')
    if kind not in ('fp8_ds_mla', 'index_fp8_128'):
        raise ValueError('Unknown packed dummy cache format')
    width = 512 if kind == 'fp8_ds_mla' else 128
    source_bytes_per_block = cache.shape[1] * (width + (64 if width == 512 else 0)) * 2
    if source_bytes_per_block > FP8_SLAB_BYTES:
        raise ValueError('One packed cache page exceeds initialization bound')
    step = max(1, FP8_SLAB_BYTES // source_bytes_per_block)
    for start in range(0, cache.shape[0], step):
        page = cache.narrow(0, start, min(step, cache.shape[0] - start))
        values = torch.empty((*page.shape[:2], width), dtype=torch.float16, device=page.device)
        initializer(values)
        if kind == 'fp8_ds_mla':
            rope = torch.empty((*page.shape[:2], 64), dtype=torch.bfloat16, device=page.device)
            initializer(rope)
            _write_packed_mla(page, values, rope)
        else:
            _write_packed_index(page, values, shuffled)


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
        from vllm.v1.kv_cache_interface import MLAAttentionSpec, compute_layer_kv_cache_shape_bytes
        # A key-only cache may bind the same storage with its singleton head
        # axis squeezed. Restore that logical axis only for shape validation.
        squeezed_head = (cache.ndim == 3 and isinstance(spec, MLAAttentionSpec)
                         and spec.num_heads == 1 and cache.dtype == spec.dtype
                         and cache.is_floating_point())
        if (cache.ndim != 4 and not squeezed_head) or cache.shape[0] % num_blocks:
            raise ValueError("Invalid logical KV cache view")
        factor = cache.shape[0] // num_blocks
        if factor < 1 or spec.block_size % factor:
            raise ValueError("Invalid kernel/manager KV block ratio")
        expected = compute_layer_kv_cache_shape_bytes(
            spec, num_blocks, spec.block_size // factor)
        actual = (*cache.shape[:-1], cache.shape[-1] * cache.element_size())
        if squeezed_head:
            actual = (actual[0], 1, *actual[1:])
        if actual != tuple(expected):
            raise ValueError(f"Logical KV cache shape mismatch: {actual} != {expected}")
        if squeezed_head:
            # Shape alone cannot establish that logical page IDs own disjoint
            # storage. Allow padded/permuted strides, but require every next
            # axis to begin beyond the extent of the faster varying axes.
            extent = 1
            for stride, size in sorted((stride, size) for size, stride in
                                       zip(cache.shape, cache.stride()) if size > 1):
                if stride < extent:
                    raise ValueError("Overlapping or unsupported typed KV cache strides")
                extent += (size - 1) * stride
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
                    selected.append((select_pages(cache, 0, first, count, num_blocks), 0, True, None))
            elif isinstance(spec, AttentionSpec):
                if not has_history:
                    continue
                packed = _packed_attention_pages(layer, spec, num_blocks)
                if packed is not None:
                    full, kind, shuffled = packed
                    selected = [(select_pages(full, 0, first, count, num_blocks),
                                 0, False, (kind, shuffled))]
                else:
                    cache_dtype = ('auto' if spec.kv_quant_mode == KVQuantMode.NONE else
                        getattr(spec, 'cache_dtype_str', None) or runner.vllm_config.cache_config.cache_dtype)
                    # Request the full allocation only as a view; the original
                    # resolver validates the installed backend's layout/block axis.
                    full, axis = _attention_pages(layer, spec, cache_dtype, num_blocks, num_blocks)
                    selected = [(select_pages(full, axis, first, count, num_blocks), axis, False, None)]
            else:
                raise TypeError(f'Unsupported profiling cache spec: {type(spec).__name__}')
            for view, axis, zero, packed_kind in selected:
                key = (view.device, view.data_ptr(), view.dtype, tuple(view.shape), tuple(view.stride()))
                if key not in seen:
                    seen.add(key)
                    views.append((view, axis, zero, packed_kind))
    with torch.inference_mode():
        for view, axis, zero, packed_kind in views:
            if zero:
                view.zero_()
            elif packed_kind is not None:
                _initialize_packed_cache(view, *packed_kind, initialize_single_dummy_weight)
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
