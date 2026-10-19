"""Explicit native DSA acquisition admission, separate from CUDA collection.

The declaration selects an explicit native vLLM capability adapter. No model
ID, hardware name, benchmark, or latency participates in admission. Integration
must carry the returned invariant digest in the existing acquisition identity.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

from profiler.core.metadata_names import PACKED_WEIGHT_POLICY

SCHEMA = 'native-dsa-eager-tail'
MODEL_MODULE = 'vllm.models.deepseek_v32.nvidia.model'
ATTENTION_MODULE = 'vllm.models.deepseek_v32.attention'
DIMENSIONS = ('hidden_size', 'vocab_size', 'num_attention_heads', 'q_lora_rank',
              'kv_lora_rank', 'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim',
              'index_n_heads', 'index_head_dim', 'index_topk', 'intermediate_size',
              'moe_intermediate_size', 'n_routed_experts', 'num_experts_per_tok')
PARALLEL = ('tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size',
            'decode_context_parallel_size', 'prefill_context_parallel_size')
SAMPLER_ROUTES = {
    ('vllm.v1.worker.gpu_model_runner.GPUModelRunner',
     'vllm.v1.sample.sampler.Sampler'): dict(
        dtype='torch.int32', entrypoint='forward'),
    ('vllm.v1.worker.gpu.model_runner.GPUModelRunner',
     'vllm.v1.worker.gpu.sample.sampler.Sampler'): dict(
        dtype='torch.int64', entrypoint='__call__'),
}


def _semantic(value):
    """Keep numerical/layout contracts distinct from implementation provenance."""
    metadata = {'vllm_version', 'torch_version', 'transformers_version', 'native_sources',
        'source_hashes', 'sampler_sources', 'endpoint_source_sha256', 'installed_source_sha256',
        'adapter_source_sha256', 'native_source_sha256', 'runtime_observations',
        'elapsed_preparation_seconds', 'sampled_rows_sha256', 'sampled_row_indices', 'full_tensor_sha256',
        'original_config_sha256'}
    if isinstance(value, dict):
        return {key: _semantic(item) for key, item in value.items() if key not in metadata}
    if isinstance(value, (tuple, list)):
        return [_semantic(item) for item in value]
    return value


def _fail(message):
    raise ValueError(f'{SCHEMA}: {message}')


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _positive(value, name):
    if type(value) is not int or value < 1:
        _fail(f'{name} must be a positive integer')
    return value


def _equal(value, expected, name):
    if type(value) is not type(expected) or _hash(value) != _hash(expected):
        _fail(f'unsupported {name}: {value!r}; expected {expected!r}')


def _normalized_rope(source, parameters):
    """Mirror the reviewed global DeepseekV3Config/native-attention conversion.

    transformers/modeling_rope_utils.py:754-809 supplies legacy aliases/defaults;
    vllm/models/deepseek_v32/attention.py:153-159 selects native scaled RoPE.
    Contradictory explicit aliases are refused instead of certifying a value
    upstream silently overwrites. Other legacy RoPE transformations are not
    covered by this native acquisition contract.
    """
    if parameters is not None and not isinstance(parameters, dict):
        _fail('RoPE parameters must be an explicit global mapping or null')
    result = copy.deepcopy(parameters or {})
    if any(isinstance(value, dict) for value in result.values()):
        _fail('nested RoPE parameters are outside the native acquisition contract')
    for name, default in (('rope_theta', 10000.0), ('partial_rotary_factor', None)):
        value = source.get(name, default)
        if name in source and name in result and value is not None:
            _equal(result[name], value, f'conflicting explicit RoPE alias {name}')
        if value is not None:
            result.setdefault(name, value)
    kind = result.get('rope_type', result.get('type', 'default'))
    if 'type' in result and 'rope_type' in result:
        _equal(result['type'], result['rope_type'], 'conflicting explicit RoPE type aliases')
    if kind not in ('default', 'yarn', 'deepseek_yarn', 'deepseek_llama_scaling'):
        _fail(f'unreviewed source RoPE normalization: {kind!r}')
    result['rope_type'] = kind
    if kind == 'yarn':
        original = source.get('original_max_position_embeddings')
        if original is not None:
            if 'original_max_position_embeddings' in result:
                _equal(result['original_max_position_embeddings'], original,
                       'conflicting original maximum position aliases')
            result['original_max_position_embeddings'] = original
        else:
            result.setdefault('original_max_position_embeddings',
                              source.get('max_position_embeddings', 4096))
    if kind != 'default':
        result['rope_type'] = ('deepseek_llama_scaling'
            if result.get('attention_factor') == 1.0 else 'deepseek_yarn')
    return result


def _validated_effective_config(source, live):
    """Check every explicit input, then retain all resolved semantic defaults.

    DeepseekV3Config attribute_map and __post_init__ normalize expert/MTP aliases
    and a null KV-head count. PreTrainedConfig maps torch_dtype to dtype and its
    to_dict overwrites transformers_version. vLLM maps deepseek_v32 to that
    configuration class. Only loader provenance and separately checked prefix
    depth are exempt, not arbitrary unknown input keys.
    """
    bookkeeping = {'transformers_version', '_name_or_path', 'name_or_path', '_commit_hash'}
    rope_fields = {'rope_scaling', 'rope_parameters', 'rope_theta', 'partial_rotary_factor'}
    aliases = {'torch_dtype': 'dtype', 'num_local_experts': 'n_routed_experts',
               'num_mtp_layers': 'num_nextn_predict_layers'}
    for name, value in source.items():
        if name in bookkeeping or name == 'num_hidden_layers' or name in rope_fields:
            continue
        if name in ('rotary_emb_base', 'rotary_pct', 'rotary_emb_fraction'):
            _fail(f'unreviewed legacy RoPE alias {name}; use rope_parameters explicitly')
        target = aliases.get(name, name)
        if name == 'model_type' and value == 'deepseek_v32':
            value = 'deepseek_v3'
        elif name == 'num_key_value_heads' and value is None:
            value = source['num_attention_heads']
        if target not in live:
            _fail(f'explicit source field is absent from resolved runtime config: {name}')
        _equal(live[target], value, f'live source geometry/config {name}')
    if rope_fields & source.keys():
        legacy, modern = source.get('rope_scaling'), source.get('rope_parameters')
        if any(value is not None and not isinstance(value, dict) for value in (legacy, modern)):
            _fail('RoPE parameters must be an explicit global mapping or null')
        expected = _normalized_rope(source, legacy or modern)
        if legacy and modern:
            other = _normalized_rope(source, modern)
            _equal({k: v for k, v in expected.items() if k != 'type'},
                   {k: v for k, v in other.items() if k != 'type'},
                   'conflicting explicit rope_scaling/rope_parameters')
        _equal(live.get('rope_parameters'), expected, 'resolved RoPE parameters')
        if 'partial_rotary_factor' in source:
            _equal(live.get('partial_rotary_factor'), source['partial_rotary_factor'],
                   'live partial_rotary_factor')
    # Prefix depth and the temporary model directory are the only live
    # differences between the reviewed category boots. Keep actual package
    # version and ALL other generated defaults, not only requested fields.
    result = copy.deepcopy({key: value for key, value in live.items()
                           if key not in ('num_hidden_layers', '_name_or_path')})
    if source.get('model_type') == 'glm_moe_dsa':
        # transformers 5.17 GlmMoeDsaConfig.__post_init__ derives these lists
        # from the instantiated prefix. Validate that prefix, then retain the
        # original-depth schedule so category depths share one identity.
        depth = _positive(live.get('num_hidden_layers'), 'resolved GLM prefix depth')
        schedules = {
            'mlp_layer_types': ['dense' if i < source['first_k_dense_replace']
                                else 'sparse' for i in range(source['num_hidden_layers'])],
            'layer_types': ['deepseek_sparse_attention'] * source['num_hidden_layers'],
            'indexer_types': ['full'] * source['num_hidden_layers'],
        }
        for name, schedule in schedules.items():
            if source.get(name) is None:
                _equal(live.get(name), schedule[:depth], f'derived GLM prefix {name}')
                result[name] = schedule
    return result


def validate_descriptor(descriptor):
    """Validate the host-owned source identity before engine-side preparation.

    source_config is the original TEXT config after explicit user overrides,
    before category depth, TP or EP shrinking. It is not the runtime's already
    sharded config. Keep its original depth for all category descriptors.
    """
    fields = {'schema', 'logical_tp', 'logical_ep', 'source_config'}
    if not isinstance(descriptor, dict) or set(descriptor) != fields:
        _fail('runtime descriptor has missing or unknown fields')
    _equal(descriptor['schema'], SCHEMA, 'descriptor schema')
    _equal(descriptor['logical_tp'], 1, 'logical TP; physical TP1 does not certify TP emulation')
    _equal(descriptor['logical_ep'], 1, 'logical EP; expert-sliced acquisition is separate')
    source = descriptor['source_config']
    if not isinstance(source, dict) or 'text_config' in source:
        _fail('source_config must be the explicitly resolved text configuration')
    _hash(source)  # Reject non-JSON and nonfinite values; retain full source identity.
    depth = _positive(source.get('num_hidden_layers'), 'source num_hidden_layers')
    for name in DIMENSIONS:
        _positive(source.get(name), f'source {name}')
    if source['num_experts_per_tok'] > source['n_routed_experts']:
        _fail('source top-k exceeds routed expert count')
    for name in ('n_shared_experts', 'first_k_dense_replace'):
        value = source.get(name)
        if type(value) is not int or value < 0:
            _fail(f'source {name} must be a nonnegative integer')
    _positive(source.get('moe_layer_freq'), 'source moe_layer_freq')
    quantization = source.get('quantization_config')
    if quantization is None:
        _equal(source.get('dtype', source.get('torch_dtype')), 'bfloat16',
               'unquantized source weight dtype')
    elif not isinstance(quantization, dict) or quantization.get('quant_method') != 'fp8':
        _fail('weight configuration must be unquantized BF16 or explicit FP8')
    # These are packed writer constraints, not checkpoint/model dimensions.
    for name, value in {'kv_lora_rank': 512, 'qk_rope_head_dim': 64,
                        'index_head_dim': 128}.items():
        _equal(source[name], value, f'packed cache {name}')
    # Checking only the short live prefix misses index sharing later in a model.
    # Mirror attention.py:166-177 for the WHOLE source stack, not just this boot.
    frequency = _positive(source.get('index_topk_freq', 1), 'index_topk_freq')
    offset = source.get('index_skip_topk_offset', 2)
    if type(offset) is not int:
        _fail('index_skip_topk_offset must be an integer')
    pattern = source.get('index_topk_pattern')
    if pattern is not None and (not isinstance(pattern, (str, list))
            or any(not isinstance(item, str) for item in pattern)):
        _fail('index_topk_pattern must be an explicit string/list or null')
    skipped = [i for i in range(depth) if (
        max(i - offset + 1, 0) % frequency != 0 if pattern is None
        else i < len(pattern) and pattern[i] == 'S')]
    if skipped:
        _fail('source stack shares/skips indexer work at layers ' + repr(skipped))
    return copy.deepcopy(descriptor)


def profile_descriptor(schema, model_config, hf_overrides, *, logical_tp, logical_ep):
    """Build the original-source identity without importing the engine/runtime.

    Explicit overrides use the engine's recursive mapping/replacement semantics;
    category depth and TP/EP shrinking are deliberately not inputs here.
    """
    from profiler.core.stack import text_config
    if not isinstance(model_config, dict) or (hf_overrides is not None and not isinstance(hf_overrides, dict)):
        _fail('runtime admission requires host source configuration and explicit overrides')

    def merge(base, override):
        out = copy.deepcopy(base)
        for key, value in override.items():
            out[key] = (merge(out[key], value) if key in out
                        and isinstance(out[key], dict) and isinstance(value, dict)
                        else copy.deepcopy(value))
        return out

    source = text_config(merge(model_config, hf_overrides or {}))
    return validate_descriptor(dict(schema=schema, logical_tp=logical_tp,
                                    logical_ep=logical_ep, source_config=source))


def _get(obj, name):
    try:
        return getattr(obj, name)
    except AttributeError as exc:
        raise ValueError(f'{SCHEMA}: missing required live fact {type(obj).__name__}.{name}') from exc


def _class(obj):
    return f'{type(obj).__module__}.{type(obj).__name__}'


def _sampler_route(runner_class, sampler_class):
    # vLLM 0.31 V1 sample/sampler.py:138-146 casts the returned IDs to
    # int32. V2 worker/gpu/sample/sampler.py:214 returns the native sample:
    # its greedy argmax and FlashInfer/Gumbel paths return int64 (337 and
    # worker/gpu/sample/gumbel.py:352,378-381). These are return payloads,
    # not the runners' separate int32 input-ID buffers.
    value = SAMPLER_ROUTES.get((runner_class, sampler_class))
    if value is None:
        _fail('unreviewed runner/sampler return-payload route')
    return value


def _sampler_facts(runner):
    """Bind the callable used by history._execute(), not a default runner mode.

    The profiling facade does not convert outputs. Inspect the native object
    behind it and its callable identity. Checks occur outside the measured
    forwards. Unknown wrappers or modified return paths
    need a new review rather than an assumed token width.
    """
    import inspect
    from .sampler_shim import Sampler as ProfilingSampler

    sampler = _get(runner, 'sampler')
    if type(sampler) is ProfilingSampler:
        sampler = sampler._inner
    runner_class, sampler_class = _class(runner), _class(sampler)
    route = _sampler_route(runner_class, sampler_class)
    method = route['entrypoint']
    bound = _get(sampler, method)
    if (getattr(bound, '__self__', None) is not sampler
            or getattr(bound, '__func__', None) is not getattr(type(sampler), method)):
        _fail('sampler entrypoint is overridden on the live instance')
    native = inspect.unwrap(bound.__func__)
    if (native.__module__ != type(sampler).__module__
            or native.__qualname__ != f'Sampler.{method}' or not callable(native)):
        _fail('sampler entrypoint does not resolve to its native class implementation')
    return dict(runner_class=runner_class, sampler_class=sampler_class,
                sampler_entrypoint=method)


def collect_runtime_facts(runner):
    """Metadata reads only: no forward, device copy, import-based model dispatch.

    An explicit runtime_contract declaration authorizes this adapter. Native
    tensor shape/dtype/device and Python object identity are read, not payloads.
    """
    import torch
    import vllm

    def tensor(value):
        if not isinstance(value, torch.Tensor):
            _fail('expected a live tensor fact')
        return dict(shape=list(value.shape), dtype=str(value.dtype),
                    device_type=value.device.type, stride=list(value.stride()))

    cfg = _get(runner, 'vllm_config')
    model = runner.get_model()
    backbone = _get(model, 'model')
    p, m = cfg.parallel_config, cfg.model_config
    offload = _get(cfg, 'offload_config')
    source = _get(m, 'hf_config').to_dict()
    layers = list(_get(backbone, 'layers'))
    shared = _get(backbone, 'topk_indices_buffer')
    result = dict(schema=SCHEMA, vllm_version=str(vllm.__version__), torch_version=str(torch.__version__),
        **_sampler_facts(runner),
        model_class=_class(model), backbone_class=_class(backbone),
        source_runtime_config=source, depth=len(layers),
        start_layer=_get(backbone, 'start_layer'), end_layer=_get(backbone, 'end_layer'),
        replicated_embed=_get(backbone, 'replicated_embed'),
        sequence_parallel=_get(backbone, 'use_sequence_parallel'),
        parallel={name: _get(p, name) for name in PARALLEL},
        expert_parallel=_get(p, 'enable_expert_parallel'),
        sequence_parallel_moe=_get(p, 'use_sequence_parallel_moe'),
        eplb=_get(p, 'enable_eplb'),
        redundant_experts=_get(_get(p, 'eplb_config'), 'num_redundant_experts'),
        enforce_eager=_get(m, 'enforce_eager'), model_dtype=str(_get(m, 'dtype')),
        quantization=_get(m, 'quantization'),
        speculative=_get(cfg, 'speculative_config') is not None,
        kv_transfer=_get(cfg, 'kv_transfer_config') is not None,
        kv_offload=_get(cfg.cache_config, 'kv_offloading_size'),
        cpu_offload_gb=_get(offload.uva, 'cpu_offload_gb'),
        prefetch_group_size=_get(offload.prefetch, 'offload_group_size'),
        prefix_caching=_get(cfg.cache_config, 'enable_prefix_caching'),
        manager_block_size=_get(cfg.cache_config, 'block_size'),
        load_format=_get(cfg.load_config, 'load_format'),
        aux_hidden_state_layers=list(_get(backbone, 'aux_hidden_state_layers')),
        topk_buffer=tensor(shared), embedding=tensor(backbone.embed_tokens.weight),
        all_parameters_cuda=all(value.device.type == 'cuda' for value in model.parameters()),
        attention=[])
    indexers = []
    for layer in layers:
        attn = _get(layer, 'self_attn')
        indexer = _get(attn, 'indexer')
        if indexer is None:
            _fail('live decoder has no indexer; skip_topk=False is insufficient')
        indexers.append(indexer)
        impl = _get(attn, 'impl')
        result['attention'].append(dict(
            layer_index=_get(layer, 'layer_idx'), layer_class=_class(layer),
            mlp_class=_class(_get(layer, 'mlp')),
            attention_class=_class(attn), indexer_class=_class(indexer),
            backend_class=_class(impl), backend_name=attn.attn_backend.get_name(),
            index_group_class=_class(_get(impl, 'index_group')),
            sequence_parallel=_get(layer, 'use_sequence_parallel'),
            use_mha=_get(layer, 'use_mha'), skip_topk=_get(attn, 'skip_topk'),
            use_pcp=_get(attn, 'use_pcp'), dcp_manager=_get(attn, 'dcp_manager') is not None,
            hisparse=_get(attn, 'hisparse_cache') is not None,
            fp8_query=_get(attn, '_fp8_query'), fp8_kv_needs_view=_get(attn, '_fp8_kv_needs_view'),
            quant_query_support=_get(impl, 'supports_quant_query_input'),
            dense_mha_support=_get(impl, 'supports_dense_mha_prefill'),
            prefill_backend=_class(attn.prefill_backend) if attn.prefill_backend is not None else None,
            kv_cache_dtype=_get(attn, 'kv_cache_dtype'),
            impl_sparse=_get(impl, 'is_sparse'),
            dcp_world_size=_get(impl, 'dcp_world_size'),
            dense_metadata_layer_name=_get(attn, '_dense_mha_metadata_layer_name'),
            layer_name=_get(attn, 'layer_name'),
            topk_aliases=all(value is shared for value in (
                attn.topk_indices_buffer, indexer.topk_indices_buffer,
                indexer.indexer_op.topk_indices_buffer, impl.topk_indices_buffer)),
            topk_backend=str(_get(indexer.indexer_op, 'topk_backend')),
            dimensions={name: _get(attn, name) for name in (
                'num_local_heads', 'q_lora_rank', 'kv_lora_rank', 'qk_nope_head_dim',
                'qk_rope_head_dim', 'v_head_dim')},
            index_dimensions={name: _get(indexer, name) for name in (
                'n_head', 'head_dim', 'topk_tokens', 'quant_block_size', 'scale_fmt')},
            main_cache=tensor(attn.kv_cache), index_cache=tensor(indexer.k_cache.kv_cache),
            index_shuffled=_get(indexer.k_cache, 'uses_shuffled_layout')))
    result['distinct_indexers'] = len({id(value) for value in indexers}) == len(indexers)
    return result


def _validate_shot(shot):
    if shot is None:
        return
    if not isinstance(shot, dict):
        _fail('shot must be a serialized mapping before hydration')
    _equal(shot.get('decode_q_len', 1), 1, 'decode query length; multi-query is not admitted')
    requests = shot.get('requests')
    if not isinstance(requests, list) or not requests:
        _fail('shot has no requests')
    boundary = shot.get('n_prefill', 0)
    if type(boundary) is not int or not 0 <= boundary <= len(requests):
        _fail('shot has invalid prefill boundary')
    for i, row in enumerate(requests):
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            _fail('shot request must be [query, history]')
        new, history = row
        _positive(new, 'query length')
        if type(history) is not int or history < 0:
            _fail('shot history must be a nonnegative integer')
        # Dense/per-sequence constructors use zero-history fresh prompts even
        # with default n_prefill=0. Do not misclassify those as multi-token decode.
        if i >= boundary and history > 0 and new != 1:
            _fail('history-bearing decode contains multiple query tokens')


def admit_runtime(descriptor, facts, *, shot=None):
    """Return validated facts and a depth-independent acquisition-mode digest.

    This is an acquisition boundary only. It does not globally restrict old
    simulation bundles or certify hypothetical deployment/topology semantics.
    """
    descriptor = validate_descriptor(descriptor)
    _validate_shot(shot)
    if not isinstance(facts, dict):
        _fail('missing worker facts')
    _equal(facts.get('schema'), SCHEMA, 'fact schema')
    sampler_route = _sampler_route(facts.get('runner_class'), facts.get('sampler_class'))
    _equal(facts.get('sampler_entrypoint'), sampler_route['entrypoint'], 'native sampler entrypoint')
    _equal(facts.get('model_class'), MODEL_MODULE + '.DeepseekV32ForCausalLM', 'active model source')
    _equal(facts.get('backbone_class'), MODEL_MODULE + '.DeepseekV32Model', 'active backbone source')
    for name in PARALLEL:
        _equal(facts.get('parallel', {}).get(name), 1, f'physical {name}')
    for name in ('expert_parallel', 'sequence_parallel_moe', 'sequence_parallel',
                 'eplb', 'speculative', 'kv_transfer', 'prefix_caching'):
        _equal(facts.get(name), False, name)
    for name in ('enforce_eager', 'replicated_embed', 'distinct_indexers', 'all_parameters_cuda'):
        _equal(facts.get(name), True, name)
    for name in ('redundant_experts', 'prefetch_group_size'):
        _equal(facts.get(name), 0, name)
    value = facts.get('cpu_offload_gb')
    if type(value) not in (int, float) or not math.isfinite(value) or value != 0:
        _fail('weight UVA offload must be disabled')
    if facts.get('kv_offload', 'missing') is not None:
        _fail('KV offload must be absent, not a zero-sized configured connector')
    _equal(facts.get('aux_hidden_state_layers'), [], 'auxiliary/speculative hidden-state output')
    _equal(facts.get('load_format'), 'dummy', 'load format')
    _equal(facts.get('model_dtype'), 'torch.bfloat16', 'model dtype')
    source = descriptor['source_config']
    quantization = source.get('quantization_config')
    _equal(facts.get('quantization'),
           'fp8' if quantization is not None else None, 'resolved weight quantization')
    live = facts.get('source_runtime_config')
    if not isinstance(live, dict):
        _fail('missing resolved runtime config')
    effective_config = _validated_effective_config(source, live)
    for name, default in {'index_topk_freq': 1, 'index_skip_topk_offset': 2,
                          'index_topk_pattern': None}.items():
        _equal(live.get(name, default), source.get(name, default), f'live indexer policy {name}')
    depth = _positive(facts.get('depth'), 'live decoder prefix depth')
    if depth > source['num_hidden_layers']:
        _fail('live prefix is longer than its source model')
    _equal(live.get('num_hidden_layers'), depth, 'resolved prefix depth')
    _equal(facts.get('start_layer'), 0, 'first local layer')
    _equal(facts.get('end_layer'), depth, 'last local layer')
    layers = facts.get('attention')
    if not isinstance(layers, list) or len(layers) != depth:
        _fail('missing target attention layers')
    shared = facts.get('topk_buffer', {})
    if (shared.get('dtype') != 'torch.int32' or shared.get('device_type') != 'cuda'
            or len(shared.get('shape', [])) != 2 or shared['shape'][0] <= 0
            or shared['shape'][1] != source['index_topk']):
        _fail('top-k shared buffer geometry differs from source')
    embedding = facts.get('embedding', {})
    if (embedding.get('shape') != [source['vocab_size'], source['hidden_size']]
            or embedding.get('dtype') != 'torch.bfloat16' or embedding.get('device_type') != 'cuda'):
        _fail('embedding must hold full source vocabulary and hidden width on-device')
    invariant_layers = []
    manager_block = _positive(facts.get('manager_block_size'), 'manager block size')
    for i, entry in enumerate(layers):
        if not isinstance(entry, dict):
            _fail('malformed attention-layer facts')
        _equal(entry.get('layer_index'), i, 'absolute prefix layer index')
        is_moe = i >= source['first_k_dense_replace'] and i % source['moe_layer_freq'] == 0
        _equal(entry.get('mlp_class'), 'vllm.model_executor.models.deepseek_v2.' +
               ('DeepseekV2MoE' if is_moe else 'DeepseekV2MLP'), 'live source prefix MLP composition')
        for name, expected in {
            'layer_class': MODEL_MODULE + '.DeepseekV32DecoderLayer',
            'attention_class': ATTENTION_MODULE + '.DeepseekV32Attention',
            'indexer_class': ATTENTION_MODULE + '.DeepseekV32Indexer',
            'index_group_class': 'vllm.v1.attention.backends.mla.index_group.SparseMLAIndexGroup',
            'kv_cache_dtype': 'fp8_ds_mla', 'prefill_backend': None,
            'dcp_world_size': 1,
        }.items():
            _equal(entry.get(name), expected, name)
        for name in ('sequence_parallel', 'use_mha', 'skip_topk', 'use_pcp',
                     'dcp_manager', 'hisparse', 'fp8_query', 'fp8_kv_needs_view',
                     'quant_query_support', 'dense_mha_support'):
            _equal(entry.get(name), False, name)
        for name in ('impl_sparse', 'topk_aliases'):
            _equal(entry.get(name), True, name)
        if type(entry.get('index_shuffled')) is not bool:
            _fail('missing explicit index-cache shuffle mode')
        for name in ('backend_class', 'backend_name', 'topk_backend'):
            if not isinstance(entry.get(name), str) or not entry[name]:
                _fail(f'missing resolved {name}')
        dimensions = {name: source[name] for name in ('q_lora_rank', 'kv_lora_rank',
            'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim')}
        dimensions['num_local_heads'] = source['num_attention_heads']
        _equal(entry.get('dimensions'), dimensions, 'actual logical TP1 attention dimensions')
        index = dict(n_head=source['index_n_heads'], head_dim=source['index_head_dim'],
                     topk_tokens=source['index_topk'], quant_block_size=128, scale_fmt='ue8m0')
        _equal(entry.get('index_dimensions'), index, 'actual indexer dimensions/packed format')
        for name, width in (('main_cache', 656), ('index_cache', 132)):
            cache = entry.get(name, {})
            shape = cache.get('shape', [])
            if (len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape)
                    or shape[-1] != width or cache.get('dtype') != 'torch.uint8'
                    or cache.get('device_type') != 'cuda'):
                _fail(f'{name} is not the supported assigned-page packed layout')
            strides = cache.get('stride')
            if (not isinstance(strides, list) or len(strides) != 3
                    or any(type(n) is not int or n < 1 for n in strides)
                    or strides[2] != 1 or strides[1] != width
                    or strides[0] < shape[1] * width or manager_block % shape[1]):
                _fail(f'{name} has invalid byte strides or manager/kernel subdivision')
        if entry['index_shuffled'] and entry['index_cache']['shape'][1] % 16:
            _fail('shuffled index blocks must divide into 16-token tiles')
        layer_name = entry.get('layer_name')
        if not isinstance(layer_name, str) or not layer_name:
            _fail('missing live attention layer name')
        # A nonempty metadata name alone does NOT imply dense MHA is enabled:
        # attention.py uses class capability; impl dense_mha_support is separate.
        if entry.get('dense_metadata_layer_name') not in ('', layer_name):
            _fail('short-prefill index metadata points to another layer')
        excluded = {'layer_index', 'mlp_class', 'layer_name', 'dense_metadata_layer_name', 'main_cache', 'index_cache'}
        normalized = {key: value for key, value in entry.items() if key not in excluded}
        for name in ('main_cache', 'index_cache'):
            cache = entry[name]
            # Capacity changes with prefix depth. Physical token/block geometry
            # and strides can change kernels and must remain part of identity.
            normalized[name] = dict(token_shape=cache['shape'][1:], stride=cache['stride'],
                                    dtype=cache['dtype'], device_type=cache['device_type'])
        invariant_layers.append(normalized)
    if any(value != invariant_layers[0] for value in invariant_layers[1:]):
        _fail('one layerwise table cannot mix backend modes or geometry within its prefix')
    invariant = dict(descriptor=descriptor, model_class=facts['model_class'],
        backbone_class=facts['backbone_class'], parallel=facts['parallel'],
        runner_class=facts['runner_class'], sampler_class=facts['sampler_class'],
        sampler_entrypoint=facts['sampler_entrypoint'],
        sampled_token_dtype=sampler_route['dtype'],
        model_dtype=facts['model_dtype'], quantization=facts['quantization'],
        manager_block_size=manager_block, effective_config=effective_config,
        attention=invariant_layers[0])
    invariant = _semantic(invariant)
    return dict(schema=SCHEMA, runtime_sha256=_hash(invariant), invariant_facts=invariant,
                engine_facts=copy.deepcopy(facts))


class HybridResidualProfile:
    """Source-specific hybrid/residual execution adapter, with config-derived geometry.

    This adapter certifies a local eager prefix and observed native layouts.
    Its observed layouts define compatibility; source/version facts are metadata.
    Complete category/context acquisition and scheduler capacity are separate.
    GPU dependencies are imported only by untimed runtime inspection methods.
    """

    schema = 'hybrid-residual-native-local'
    context_layers = ('attn_res',)
    rpc_method = 'native_profile_admission'
    # transformers 5.17 PreTrainedConfig deliberately removes generation
    # parameters. Only absence qualifies; changed explicit values still fail.
    removed_generation_fields = frozenset((
        'diversity_penalty', 'do_sample', 'early_stopping',
        'encoder_no_repeat_ngram_size', 'length_penalty', 'max_length', 'min_length',
        'no_repeat_ngram_size', 'num_beam_groups', 'num_beams', 'num_return_sequences',
        'output_scores', 'remove_invalid_values', 'repetition_penalty',
        'return_dict_in_generate', 'temperature', 'top_k', 'top_p', 'typical_p',
    ))

    @staticmethod
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

    @classmethod
    def digest(cls, value):
        return hashlib.sha256(cls.canonical(value).encode()).hexdigest()

    @staticmethod
    def _positive(value, name):
        if type(value) is not int or value < 1:
            raise ValueError('Hybrid residual ' + name + ' must be a positive integer')
        return value

    @classmethod
    def original_text(cls, config):
        from profiler.core.stack import expert_top_k, resolve_stack, attn_res_roles
        if (type(config) is not dict or config.get('model_type') != 'kimi_k3'
                or type(config.get('text_config')) is not dict):
            raise ValueError('Hybrid residual source requires its native nested wrapper')
        cls.canonical(config)
        text = config['text_config']
        depth = cls._positive(text.get('num_hidden_layers'), 'source depth')
        cls._positive(text.get('hidden_size'), 'hidden width')
        cls._positive(text.get('attn_res_block_size'), 'residual block width')
        experts = cls._positive(text.get('num_experts'), 'routed experts')
        topk = cls._positive(expert_top_k(text), 'expert top-k')
        if topk > experts:
            raise ValueError('Hybrid residual source top-k exceeds its experts')
        stack = resolve_stack(config)
        if len(stack) != depth or not any(row.mlp == 'moe' for row in stack):
            raise ValueError('Hybrid residual source must include native routed blocks')
        # This validates source-derived epsilon, depth and residual schedule.
        attn_res_roles(config)
        return copy.deepcopy(text)

    @staticmethod
    def materialize_prefix(config, overrides):
        from profiler.core.stack import materialize_prefix
        return materialize_prefix(config, overrides)

    @staticmethod
    def roles(config, *, depth=None):
        from profiler.core.stack import attn_res_roles
        return attn_res_roles(config, depth=depth)

    def validate_request(self, args, *, full=False):
        self.original_text(args.model_config)
        if (list(args.tp_degrees) != [1] or args.profile_mtp or args.only_skew
                or (args.dp_degrees is not None and tuple(args.dp_degrees) != (1,))
                or (args.ep_degrees is not None and tuple(args.ep_degrees) != (1,))
                or args.hf_overrides or (full and not args.skip_skew)
                or tuple(args.attention_decode_q_lens) != (1,)):
            raise NotImplementedError('Hybrid residual acquisition requires local TP1/DP1/EP1, q=1, '
                'no MTP or shape overrides; full profiles require --skip-skew')
        if (args.num_hidden_layers not in (None, 1)
                or args.dtype not in (None, 'auto', 'bfloat16')
                or args.kv_cache_dtype not in (None, 'auto')):
            raise NotImplementedError('Hybrid residual acquisition requires automatic category '
                                      'prefixes and native BF16 activations/cache')
        self.context_plan(args.model_config, args.max_num_batched_tokens or 2048,
                          iterations=args.measurement_iterations, warmups=1)

    @staticmethod
    def stack_axes(category):
        return ('mlp',) if category.name == 'moe' else category.stack_axes

    @staticmethod
    def engine_overrides():
        # The admitted native multimodal wrapper is executed as text-only.
        return {'limit_mm_per_prompt': {'image': 0, 'video': 0}}

    @staticmethod
    def validate_context_response(response):
        # Generic same-capture validation joins these counts to the CUDA
        # ledger; this source adapter additionally admits the native route.
        calls = response.get('routes', response.get('evidence', {}).get('per_call'))
        if not isinstance(calls, list) or not calls:
            raise ValueError('Missing native residual callable evidence')
        canonical_calls = [call for call in calls if call.get('canonical') is not None]
        if not canonical_calls or any(call.get('canonical') != 'attn_res'
                or call.get('descendant_counts') != {'_attn_res_kernel': 1}
                for call in canonical_calls):
            raise ValueError('Unreviewed native residual callable route')

    @staticmethod
    def context_plan(config, tokens, *, iterations, warmups):
        from profiler.core.callable_profile import hybrid_residual_plan
        return hybrid_residual_plan(config, tokens, iterations=iterations, warmups=warmups)

    @classmethod
    def runtime_descriptor(cls, config, depth, engine_limits, cache_bytes):
        from profiler.core.stack import resolve_stack
        text = cls.original_text(config)
        cls._positive(depth, 'prefix depth')
        if depth > text['num_hidden_layers']:
            raise ValueError('Hybrid residual prefix exceeds its source depth')
        if not any(row.mlp == 'moe' for row in resolve_stack(config, prefix_layers=depth)):
            raise ValueError('Hybrid residual prefix must include a routed layer')
        fields = {'max_model_len', 'max_num_seqs', 'max_num_batched_tokens'}
        if (type(engine_limits) is not dict or set(engine_limits) != fields
                or any(type(n) is not int or n < 1 for n in engine_limits.values())
                or (cache_bytes is not None and (type(cache_bytes) is not int or cache_bytes < 1))):
            raise ValueError('Hybrid residual requires positive limits and cache budget or automatic cache')
        return dict(schema=cls.schema, original_config=copy.deepcopy(config), engine_depth=depth,
            engine_limits=copy.deepcopy(engine_limits),
            cache_bytes=cache_bytes, tp=1, pp=1, ep=1, dp=1)

    @classmethod
    def validate_descriptor(cls, descriptor):
        if type(descriptor) is not dict:
            raise ValueError('Hybrid residual descriptor must be plain data')
        expected = cls.runtime_descriptor(descriptor.get('original_config'),
            descriptor.get('engine_depth'), descriptor.get('engine_limits'), descriptor.get('cache_bytes'))
        if cls.canonical(_semantic(descriptor)) != cls.canonical(_semantic(expected)):
            raise ValueError('Hybrid residual descriptor differs from its source-bound local scope')
        return expected

    @classmethod
    def admit_runtime(cls, descriptor, facts):
        from profiler.core.stack import resolve_stack
        if type(facts) is not dict:
            raise ValueError('Missing hybrid residual runtime facts')
        facts = copy.deepcopy(facts)
        preparation = facts.get('dummy_weight_preparation') or {}
        preparation.pop('elapsed_preparation_seconds', None)
        for record in preparation.get('tensor_records', []):
            record.pop('elapsed_preparation_seconds', None)
        descriptor = cls.validate_descriptor(descriptor)
        source, depth = descriptor['original_config'], descriptor['engine_depth']
        text = cls.original_text(source)
        expected = cls.materialize_prefix(source, {'num_hidden_layers': depth})['text_config']
        if (facts.get('schema') != cls.schema
                or facts.get('model_class') != 'vllm.models.kimi_k3.nvidia.model.KimiK3ForConditionalGeneration'
                or facts.get('model_type') != 'kimi_k3' or facts.get('text_type') != 'kimi_linear'
                or facts.get('dtype') != 'torch.bfloat16' or facts.get('quantization') != 'mxfp4'
                or facts.get('enforce_eager') is not True):
            raise ValueError('Hybrid residual native model/source/precision differs')
        effective = facts.get('text_config')
        if type(effective) is not dict:
            raise ValueError('Missing resolved hybrid residual text config')
        for key, value in expected.items():
            if key in ('_name_or_path', '_attn_implementation_autoset', 'transformers_version'):
                continue
            if key in cls.removed_generation_fields and key not in effective:
                continue
            if key == 'quantization_config':
                value = dict(value, quant_method='mxfp4')
            if cls.canonical(effective.get(key)) != cls.canonical(value):
                raise ValueError('Hybrid residual actual text config differs at ' + key)
        parallel = facts.get('parallel', {})
        if (set(parallel) != {'tp', 'pp', 'ep', 'dp', 'pcp', 'dcp'}
                or any(type(n) is not int or n != 1 for n in parallel.values())
                or facts.get('expert_parallel') is not False or facts.get('speculative') is not False
                or facts.get('prefix_caching') is not False or facts.get('mamba_cache_mode') != 'none'
                or facts.get('multimodal_limits') != {'image': 0, 'video': 0}
                or facts.get('resident_multimodal_tensors') != 0
                or facts.get('cpu_offload_gb') != 0 or facts.get('prefetch_group_size') != 0
                or facts.get('kv_transfer') is not False or facts.get('kv_offload') is not None):
            raise ValueError('Hybrid residual runtime is outside the admitted local eager scope')
        if (cls.canonical(facts.get('limits')) != cls.canonical(descriptor['engine_limits'])
                or facts.get('kv_cache_memory_bytes') != descriptor['cache_bytes']):
            raise ValueError('Hybrid residual actual engine/cache envelope differs')
        sampler = facts.get('sampler', {})
        route = SAMPLER_ROUTES.get((sampler.get('runner_class'), sampler.get('sampler_class')))
        if (route is None
                or sampler.get('sampler_entrypoint') != route['entrypoint']
                or sampler.get('sampled_token_dtype') != route['dtype']):
            raise ValueError('Hybrid residual native sampler return route is not admitted')
        stack = resolve_stack(source, prefix_layers=depth)
        layers = facts.get('layers', [])
        if len(layers) != depth:
            raise ValueError('Hybrid residual actual decoder prefix differs')
        for index, (layer, spec) in enumerate(zip(layers, stack)):
            attention = ('KimiK3DeltaAttention' if spec.attn == 'linear_attention'
                         else 'MultiHeadLatentAttention')
            wanted = dict(index=index, attention=attention, is_moe=spec.mlp == 'moe',
                hidden=text['hidden_size'], use_attn_res=True, sequence_parallel=False)
            if cls.canonical(layer) != cls.canonical(wanted):
                raise ValueError('Hybrid residual native layer role differs at ' + str(index))
        capacity = (depth + text['attn_res_block_size'] - 1) // text['attn_res_block_size']
        if cls.canonical(facts.get('backbone')) != cls.canonical(
                dict(start=0, end=depth, residual_capacity=capacity)):
            raise ValueError('Hybrid residual native prefix geometry differs')
        contracts = preparation.get('actual_selected_backend_contracts', {})
        paths = {f'language_model.model.layers.{i}.block_sparse_moe.experts.routed_experts'
                 for i, layer in enumerate(stack) if layer.mlp == 'moe'}
        if (preparation.get('policy') != PACKED_WEIGHT_POLICY
                or preparation.get('complete') is not True or preparation.get('forward_verified') is not False
                or preparation.get('postload', {}).get('structure_verified') is not True
                or set(contracts) != paths):
            raise ValueError('Hybrid residual native initialization/postload structure is incomplete')
        for contract in contracts.values():
            if (contract.get('backend') != 'DEEPGEMM_MXFP4' or contract.get('backend_forced') is not False
                    or contract.get('method_class') != 'vllm.model_executor.layers.quantization.mxfp4.Mxfp4MoEMethod'
                    or contract.get('experts_class') != 'vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe.DeepGemmFP4Experts'
                    or [contract.get(key) for key in ('physical_experts', 'logical_hidden', 'logical_intermediate')]
                    != [text['num_experts'], text['routed_expert_hidden_size'], text['moe_intermediate_size']]):
                raise ValueError('Hybrid residual selected native expert backend/layout differs')
        cls.validate_layout(source, facts.get('logical_sizes'), stack)
        return dict(schema=cls.schema, descriptor_sha256=cls.digest(_semantic(descriptor)),
                    facts=facts, facts_sha256=cls.digest(_semantic(facts)), forward_verified=False)

    @classmethod
    def expected_calls(cls, scope):
        from profiler.core.stack import resolve_stack
        descriptor = scope['descriptor']
        stack = resolve_stack(descriptor['original_config'], prefix_layers=descriptor['engine_depth'])
        kda = sum(row.attn == 'linear_attention' for row in stack)
        mla = sum(row.attn == 'full_attention' for row in stack)
        return dict(embedding=1, mlp=sum(row.mlp == 'dense' for row in stack),
            moe=sum(row.mlp == 'moe' for row in stack), final_layernorm=1, lm_head=1, sampler=1,
            kda_in_proj=kda, kda_f_b_proj=kda, kda_core=kda, kda_o_proj=kda,
            attention=mla, mla_gate=mla, mla_o_proj=mla)

    @staticmethod
    def validate_capture_extra(scope, warmup):
        if (warmup.get('finite') is not True or len(warmup.get('checked_attention_layers', []))
                != scope['descriptor']['engine_depth']):
            raise ValueError('Incomplete hybrid residual finite-forward/attention checks')

    @classmethod
    def validate_layout(cls, config, facts, stack):
        """Validate observed shapes against source geometry and actual methods.

        Projection alignment 16 belongs to the reviewed unquantized KDA path;
        expert packing belongs to the selected DeepGemm MXFP4 method. Neither
        is inferred from device marketing name or checkpoint dimensions.
        """
        from serving.core.memory_model import kimi_component_geometry, packed_expert_shapes
        from profiler.core.stack import expert_top_k
        if type(facts) is not dict:
            raise ValueError('Missing observed hybrid residual layouts')
        text = config['text_config']
        common = dict(parallel=[1, 1, 1, 1], enforce_eager=True,
            model_dtype='torch.bfloat16', input_id_dtype='torch.int32', logits_dtype='torch.bfloat16',
            head_embedding_alias=False, ordinary_linear_method='UnquantizedLinearMethod',
            ordinary_weight_dtype='torch.bfloat16', norm_dtype='torch.bfloat16')
        for name, expected in common.items():
            if cls.canonical(facts.get(name)) != cls.canonical(expected):
                raise ValueError('Unreviewed native layout: ' + name)
        shape = facts.get('embedding_shape')
        if not isinstance(shape, list) or len(shape) != 2:
            raise ValueError('Missing observed native vocabulary shape')
        geometry = kimi_component_geometry(config, 1, projection_alignment=16, vocab_rows=shape[0])
        expected = dict(embedding_shape=geometry['shapes']['embedding']['weight'],
                        head_shape=geometry['shapes']['head']['weight'])
        if any(row.attn == 'linear_attention' for row in stack):
            shapes = geometry['shapes']['linear_attention']
            expected.update(kda_prefill_backend='flashkda', kda_decode_backend='native',
                kda_in_proj_shape=shapes['in_proj'],
                kda_conv_shape_dtype=[shapes['conv'], 'torch.float32'],
                kda_decode_conv_shape_dtype=[geometry['kda_decode_buffers']['conv'], 'torch.float32'],
                kda_norm_shape_dtype=[shapes['norm'], 'torch.bfloat16'],
                kda_decode_norm_shape_dtype=[geometry['kda_decode_buffers']['norm'], 'torch.float32'],
                kda_A_log_shape_dtype=[shapes['A_log'], 'torch.float32'],
                kda_dt_bias_shape_dtype=[shapes['dt_bias'], 'torch.float32'],
                kda_derived_initialized=True, kda_state_shapes=geometry['kda_state_shapes'],
                kda_state_dtypes=['torch.bfloat16','torch.float32'], kda_spec_tokens=0,
                kda_recoverssm=False, kda_mamba_cache_mode='none')
        if any(row.attn == 'full_attention' for row in stack):
            expected.update(mla_nope=True, mla_output_gate=True, mla_backend='TRITON_MLA',
                mla_impl='vllm.v1.attention.backends.mla.triton_mla.TritonMLAImpl',
                mla_prefill_backend='vllm.v1.attention.backends.mla.prefill.flash_attn.FlashAttnPrefillBackend',
                mla_cache_head_dtype=[geometry['mla_cache_head_size'],'torch.bfloat16'],
                mla_cache_quant_mode='NONE', mla_cache_scale_values=[1.0]*6,
                mla_absorbed_shape_dtype=[*geometry['mla_absorbed_shapes'], 'torch.bfloat16'],
                mla_absorbed_alias_kv_b=[True,True], mla_absorbed_matches_source=True)
        if any(row.mlp == 'moe' for row in stack):
            packed = packed_expert_shapes(text['num_experts'], text['routed_expert_hidden_size'],
                geometry['moe_padded_intermediate'], weight_pack=2,scale_group_size=32,scales_per_element=4)
            expected.update(moe_backend='DEEPGEMM_MXFP4',moe_backend_configured='auto',
                moe_geometry=[text['hidden_size'],text['routed_expert_hidden_size'],
                    geometry['moe_padded_intermediate'],text['num_experts'],expert_top_k(text),text['num_shared_experts']],
                moe_activation=['situ',text['activation_situ_beta'],text['activation_situ_linear_beta']],
                moe_scale_aliases_kernel=True,moe_shared_transform_bias_aliases=True,
                moe_router_weight_logits_bias_dtypes=['torch.bfloat16','torch.float32','torch.float32'])
            for key, value in packed.items():
                expected['moe_'+key+'_shape_dtype'] = [value, 'torch.int32' if key.startswith('s') else 'torch.uint8']
        for name, value in expected.items():
            if cls.canonical(facts.get(name)) != cls.canonical(value):
                raise ValueError('Native layout differs from source geometry: ' + name)

    @staticmethod
    def _shape_dtype(tensor):
        if tensor is None or tensor.device.type != 'cuda' or tensor.is_meta:
            raise ValueError('A native CUDA tensor is required for logical layout facts')
        return [list(tensor.shape), str(tensor.dtype)]

    @staticmethod
    def _same_view(left, right):
        return (left.device == right.device and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape) and tuple(left.stride()) == tuple(right.stride())
            and left.storage_offset() == right.storage_offset()
            and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr())

    @staticmethod
    def _merge_facts(target, observed):
        for name, value in observed.items():
            if name in target and target[name] != value:
                raise ValueError('Native component layouts disagree across layers: ' + name)
            target[name] = value

    @classmethod
    def _kda_facts(cls, module, name, preparation, config):
        matches = [row for row in preparation['postload']['derived_buffers'] if row['name'] == name]
        initialized = (len(matches) == 1 and matches[0].get('source_copy_equal') is True
                       and matches[0].get('finite') is True)
        shape = cls._shape_dtype
        return dict(kda_prefill_backend=module.kda_prefill_backend, kda_decode_backend=module.kda_decode_backend,
            kda_in_proj_shape=shape(module.in_proj_qkvgfab.weight)[0],
            kda_conv_shape_dtype=shape(module.conv1d.weight),
            kda_decode_conv_shape_dtype=shape(module.decode_conv1d_weight),
            kda_norm_shape_dtype=shape(module.o_norm.weight), kda_decode_norm_shape_dtype=shape(module.decode_norm_weight),
            kda_A_log_shape_dtype=shape(module.A_log), kda_dt_bias_shape_dtype=shape(module.dt_bias),
            kda_derived_initialized=initialized,kda_state_shapes=[list(s) for s in module.get_state_shape()],
            kda_state_dtypes=[str(dtype) for dtype in module.get_state_dtype()],
            kda_spec_tokens=module.num_spec,kda_recoverssm=module.use_recoverssm,
            kda_mamba_cache_mode=config.cache_config.mamba_cache_mode)

    @classmethod
    def _mla_facts(cls, module, config):
        weight = module.kv_b_proj.weight
        cls._shape_dtype(weight)
        if not weight.is_contiguous():
            raise ValueError('Native MLA KV-B must be a contiguous weight')
        native = weight.view(module.num_local_heads,module.qk_nope_head_dim+module.v_head_dim,module.kv_lora_rank)
        uk, uv = module.W_UK_T,module.W_UV
        key, value = cls._shape_dtype(uk),cls._shape_dtype(uv)
        if key[1] != value[1]:
            raise ValueError('Absorbed MLA views have different dtypes')
        spec = module.get_kv_cache_spec(config)
        return dict(mla_nope=module.rotary_emb is None,mla_output_gate=module.use_output_gate,
            mla_backend=module.attn_backend.get_name(), mla_impl=_class(module.impl),
            mla_prefill_backend=_class(module.prefill_backend),mla_cache_head_dtype=[spec.head_size,str(spec.dtype)],
            mla_cache_quant_mode=spec.kv_quant_mode.name,
            mla_cache_scale_values=[float(getattr(module,name).item()) for name in
                ('_q_scale','_k_scale','_v_scale','_q_scale_inv','_k_scale_inv','_one_scale')],
            mla_absorbed_shape_dtype=[key[0],value[0],key[1]],
            mla_absorbed_alias_kv_b=[view.untyped_storage().data_ptr()==weight.untyped_storage().data_ptr() for view in (uk,uv)],
            mla_absorbed_matches_source=(cls._same_view(uk,native[:,:module.qk_nope_head_dim])
                and cls._same_view(uv,native[:,module.qk_nope_head_dim:].transpose(1,2))))

    @classmethod
    def _moe_facts(cls, module):
        runner=module.experts
        experts=runner.routed_experts
        method=experts.quant_method
        kernel=method.moe_kernel.fused_experts
        s13,s2=experts.w13_weight_scale,experts.w2_weight_scale
        topk=runner.router.top_k
        if type(topk) is not int or topk<=0 or topk!=runner.moe_config.experts_per_token:
            raise ValueError('Native router and expert kernel top-k disagree')
        aliases=(runner._shared_experts._layer is module.shared_experts
            and runner.routed_output_transform is module.routed_output_transform
            and module.routed_output_transform.norm is module.routed_expert_norm
            and module.routed_output_transform.up_proj is module.routed_expert_up_proj
            and cls._same_view(experts.e_score_correction_bias,module.gate.e_score_correction_bias))
        shape=cls._shape_dtype
        return dict(moe_backend=method.mxfp4_backend.name,moe_backend_configured=method.moe.moe_backend,
            moe_geometry=[module.gate.weight.shape[1],module.moe_hidden_size,module.padded_moe_intermediate_size,
                experts.w13_weight.shape[0],topk,module.num_shared_experts],
            moe_activation=[method.moe.activation.value,method.moe.activation_situ_beta,method.moe.activation_situ_linear_beta],
            moe_w13_shape_dtype=shape(experts.w13_weight),moe_w2_shape_dtype=shape(experts.w2_weight),
            moe_s13_shape_dtype=shape(s13),moe_s2_shape_dtype=shape(s2),
            moe_scale_aliases_kernel=cls._same_view(s13,kernel.w1_scale) and cls._same_view(s2,kernel.w2_scale),
            moe_shared_transform_bias_aliases=aliases,moe_router_weight_logits_bias_dtypes=[
                str(module.gate.weight.dtype),str(module.gate.out_dtype),str(module.gate.e_score_correction_bias.dtype)])

    @classmethod
    def _observe_layout(cls, runner, preparation):
        import torch
        from vllm.model_executor.layers.linear import LinearBase
        model,config=runner.get_model(),runner.vllm_config
        language=model.language_model
        head,embedding=language.lm_head,language.model.embed_tokens
        head_shape,head_dtype=cls._shape_dtype(head.weight)
        embedding_shape,_=cls._shape_dtype(embedding.weight)
        processor=language.logits_processor
        if (type(processor).__module__!='vllm.model_executor.layers.logits_processor'
                or processor.logits_as_input or processor.soft_cap is not None or processor.scale!=1.0
                or type(head.quant_method).__name__ not in ('UnquantizedEmbeddingMethod','UnquantizedLinearMethod')):
            raise ValueError('Unreviewed native head projection route')
        legacy=getattr(runner,'input_ids',None)
        modern=getattr(getattr(runner,'input_buffers',None),'input_ids',None)
        if (legacy is None)==(modern is None):
            raise ValueError('Missing or ambiguous native input token buffer')
        input_tensor=modern if modern is not None else getattr(legacy,'gpu',None)
        cls._shape_dtype(input_tensor)
        parallel=config.parallel_config
        facts=dict(device_capability=list(torch.cuda.get_device_capability()),
            parallel=[parallel.tensor_parallel_size,parallel.pipeline_parallel_size,parallel.data_parallel_size,
                parallel.tensor_parallel_size*parallel.data_parallel_size if parallel.enable_expert_parallel else 1],
            enforce_eager=config.model_config.enforce_eager,model_dtype=str(config.model_config.dtype),
            input_id_dtype=str(input_tensor.dtype),logits_dtype=str(processor.head_dtype) if processor.head_dtype is not None else head_dtype,
            embedding_shape=embedding_shape,head_shape=head_shape,
            head_embedding_alias=head.weight.untyped_storage().data_ptr()==embedding.weight.untyped_storage().data_ptr(),
            norm_dtype=str(language.model.norm.weight.dtype))
        modules=list(model.named_modules())
        conv_ids=set()
        for name,module in modules:
            kind=_class(module)
            if kind=='vllm.models.kimi_k3.nvidia.kda.KimiK3DeltaAttention':
                conv_ids.add(id(module.conv1d))
                cls._merge_facts(facts,cls._kda_facts(module,name,preparation,config))
            elif kind=='vllm.models.kimi_k3.nvidia.mla.MultiHeadLatentAttention':
                cls._merge_facts(facts,cls._mla_facts(module,config))
            elif kind=='vllm.models.kimi_k3.nvidia.model.KimiMoE':
                cls._merge_facts(facts,cls._moe_facts(module))
        methods,dtypes=set(),set()
        for _,module in modules:
            if isinstance(module,LinearBase) and id(module) not in conv_ids:
                methods.add(type(module.quant_method).__name__)
                dtypes.add(cls._shape_dtype(module.weight)[1])
        if len(methods)!=1 or len(dtypes)!=1:
            raise ValueError('Native ordinary linear method/precision is not uniform')
        facts.update(ordinary_linear_method=methods.pop(),ordinary_weight_dtype=dtypes.pop())
        return facts

    @classmethod
    def inspect_runtime(cls, runner, descriptor):
        import torch
        import vllm
        from .dummy_weights import preparation_facts
        descriptor=cls.validate_descriptor(descriptor)
        model=runner.get_model()
        if _class(model) != 'vllm.models.kimi_k3.nvidia.model.KimiK3ForConditionalGeneration':
            raise ValueError('Actual hybrid residual model source differs')
        cfg=runner.vllm_config
        parallel,cache=cfg.parallel_config,cfg.cache_config
        layers,backbones=[],[]
        for _,module in model.named_modules():
            name=type(module).__name__
            if name=='KimiDecoderLayer':
                if _class(module) != 'vllm.models.kimi_k3.nvidia.model.KimiDecoderLayer':
                    raise ValueError('Foreign hybrid residual decoder implementation')
                layers.append(dict(index=module.layer_idx,attention=type(module.self_attn).__name__,
                    is_moe=module.is_moe_layer,hidden=module.hidden_size,use_attn_res=module.use_attn_res,
                    sequence_parallel=module.use_sequence_parallel))
            elif name=='KimiLinearModel':
                backbones.append(dict(start=module.start_layer,end=module.end_layer,
                                      residual_capacity=module.num_attn_res_blocks))
        if len(backbones)!=1:
            raise ValueError('Expected exactly one hybrid residual text backbone')
        resident_mm=0
        for name in ('vision_tower','mm_projector'):
            tower=getattr(model,name)
            if tower is not None:
                resident_mm+=sum(not tensor.is_meta for tensor in [*tower.parameters(),*tower.buffers()])
        mm=cfg.model_config.get_multimodal_config()
        sampler=_sampler_facts(runner)
        sampler_route=_sampler_route(sampler['runner_class'],sampler['sampler_class'])
        sampler['sampled_token_dtype']=sampler_route['dtype']
        facts=dict(schema=cls.schema,vllm_version=str(vllm.__version__),
            model_class=_class(model),model_type=cfg.model_config.hf_config.model_type,
            text_type=cfg.model_config.hf_text_config.model_type,
            text_config=copy.deepcopy(cfg.model_config.hf_text_config.to_dict()),
            dtype=str(cfg.model_config.dtype),device_capability=list(torch.cuda.get_device_capability()),
            quantization=cfg.model_config.quantization,enforce_eager=cfg.model_config.enforce_eager,
            parallel=dict(tp=parallel.tensor_parallel_size,pp=parallel.pipeline_parallel_size,
                dp=parallel.data_parallel_size,pcp=parallel.prefill_context_parallel_size,
                dcp=parallel.decode_context_parallel_size,
                ep=parallel.tensor_parallel_size*parallel.data_parallel_size if parallel.enable_expert_parallel else 1),
            expert_parallel=parallel.enable_expert_parallel,speculative=cfg.speculative_config is not None,
            prefix_caching=cache.enable_prefix_caching,mamba_cache_mode=cache.mamba_cache_mode,
            multimodal_limits={name:mm.get_limit_per_prompt(name) for name in ('image','video')},
            resident_multimodal_tensors=resident_mm,cpu_offload_gb=cfg.offload_config.uva.cpu_offload_gb,
            prefetch_group_size=cfg.offload_config.prefetch.offload_group_size,
            kv_transfer=cfg.kv_transfer_config is not None,kv_offload=cache.kv_offloading_size,sampler=sampler,
            limits=dict(max_model_len=cfg.model_config.max_model_len,max_num_seqs=cfg.scheduler_config.max_num_seqs,
                max_num_batched_tokens=cfg.scheduler_config.max_num_batched_tokens),
            kv_cache_memory_bytes=cache.kv_cache_memory_bytes,
            layers=sorted(layers,key=lambda row:row['index']),backbone=backbones[0],
            dummy_weight_preparation=preparation_facts(model))
        facts['logical_sizes']=cls._observe_layout(runner,facts['dummy_weight_preparation'])
        return cls.admit_runtime(descriptor,facts)
