"""Untimed inspection of the actual reviewed sparse indexer/attend instances."""
from profiler.core import attention_lookup as contract


def runtime_semantic(runner, profile_tp=1):
    cfg = runner.vllm_config.model_config.hf_config.to_dict()
    if cfg.get('model_type') not in ('minimax_m3_vl', 'minimax_m3'):
        return None
    if runner.vllm_config.speculative_config is not None:
        raise ValueError('Native attention lookup contract does not cover speculative acquisition')
    model = runner.model
    modules = [module for module in model.modules()
               if any(base.__name__ == 'MiniMaxM3SparseAttention' for base in type(module).__mro__)]
    if not modules:
        raise ValueError('Native attention lookup requires an actual sparse layer')
    values = []
    for module in modules:
        if (type(module).__module__ != 'vllm.models.minimax_m3.nvidia.model'
                or type(module).__name__ != 'MiniMaxM3SparseAttention'):
            raise ValueError('Unknown sparse attention owner')
        indexer, attend = module.indexer.impl, module.impl
        name = lambda obj: type(obj).__module__ + '.' + type(obj).__name__
        if name(indexer) != contract.INDEXER or name(attend) != contract.ATTEND:
            raise ValueError('Unreviewed selected indexer/attention implementation')
        buffer = module.topk_indices_buffer
        if (buffer is None or buffer is not indexer.topk_indices_buffer
                or buffer.ndim != 3 or str(buffer.dtype) != 'torch.int32'
                or buffer.shape[1] != indexer.num_index_heads
                or buffer.stride(-1) != 1 or not buffer.is_contiguous()
                or buffer.device.type != 'cuda'
                or attend.topk_blocks != indexer.topk_blocks
                or attend.block_size != indexer.block_size
                or attend.use_fp8_kv or str(module.kv_cache_torch_dtype) != 'torch.bfloat16'):
            raise ValueError('Native selected-block storage/layout differs')
        source = dict(indexer_class=name(indexer),
            attend_class=name(attend), topk_blocks=indexer.topk_blocks,
            block_size=indexer.block_size, init_blocks=indexer.init_blocks,
            local_blocks=indexer.local_blocks, topk_width=buffer.shape[-1],
            indexer_dtype=indexer.indexer_kv_dtype, kv_dtype='bfloat16', score_type=indexer.score_type)
        value = dict(schema=contract.SCHEMA, rule=contract.RULE, model_type='minimax_m3_vl', profile_tp=profile_tp,
            key_saturation=indexer.topk_blocks * indexer.block_size,
            saturation_by_layer={'attention': False, 'indexer': False, 'sparse_attention': True},
            source=source)
        values.append(contract.validate_semantic(value, cfg))
    if any(value != values[0] for value in values):
        raise ValueError('Sparse layers disagree on lookup contract')
    return values[0]


def verify(runner, expected):
    if expected is not None:
        contract.validate_semantic(expected)
    value = runtime_semantic(runner, expected['profile_tp'] if expected is not None else 1)
    if contract.semantic_facts(value) != contract.semantic_facts(expected):
        raise ValueError('Live attention lookup semantics differ from requested identity')
    return contract.digest(value) if value is not None else None
