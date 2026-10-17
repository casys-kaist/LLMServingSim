"""Explicit native DSA acquisition admission, separate from CUDA collection.

The versioned declaration selects this source-specific vLLM touchpoint. No model
ID, hardware name, benchmark, or latency participates in admission. Integration
must carry the returned invariant digest in the existing acquisition identity.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

SCHEMA = 'native-dsa-eager-tail-v1'
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
        dtype='torch.int32', entrypoint='forward', sources={
            'v1/sample/sampler.py':
                'd299305528479ff505b416b759db208fcf08ca6e86cc1f7cfc38be85bbd0e844'}),
    ('vllm.v1.worker.gpu.model_runner.GPUModelRunner',
     'vllm.v1.worker.gpu.sample.sampler.Sampler'): dict(
        dtype='torch.int64', entrypoint='__call__', sources={
            'v1/worker/gpu/sample/sampler.py':
                'ddbcdab46010bf38a5fab9b9b6a2deed2ec5e883ebfe81e83f6d22f41e546771',
            'v1/worker/gpu/sample/gumbel.py':
                'f24793dbc02d196cb458148b42fb73b7663de55e7e6ded660ea51065b7143853'}),
}


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
    return copy.deepcopy({key: value for key, value in live.items()
                          if key not in ('num_hidden_layers', '_name_or_path')})


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
    if not isinstance(quantization, dict) or quantization.get('quant_method') != 'fp8':
        _fail('initial contract requires explicit FP8 weight configuration')
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
    behind it and its callable's source; file reads and metadata checks occur
    outside the measured forwards. Unknown wrappers or modified return paths
    need a new review rather than an assumed token width.
    """
    import inspect
    from pathlib import Path
    import vllm
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
    package = Path(vllm.__file__).resolve().parent
    expected_file = package / next(iter(route['sources']))
    if (Path(inspect.getsourcefile(type(sampler))).resolve() != expected_file
            or Path(inspect.getsourcefile(native)).resolve() != expected_file
            or native.__qualname__ != f'Sampler.{method}'):
        _fail('sampler entrypoint does not resolve to its reviewed native source')
    return dict(runner_class=runner_class, sampler_class=sampler_class,
                sampler_entrypoint=method,
                sampler_sources={name: hashlib.sha256((package/name).read_bytes()).hexdigest()
                                 for name in route['sources']})


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
    _equal(facts.get('vllm_version'), '0.31.0', 'reviewed native runtime version')
    sampler_route = _sampler_route(facts.get('runner_class'), facts.get('sampler_class'))
    _equal(facts.get('sampler_entrypoint'), sampler_route['entrypoint'], 'native sampler entrypoint')
    _equal(facts.get('sampler_sources'), sampler_route['sources'], 'native sampler return-path sources')
    if not isinstance(facts.get('torch_version'), str) or not facts['torch_version']:
        _fail('missing actual Torch version')
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
    _equal(facts.get('quantization'), 'fp8', 'resolved weight quantization')
    source = descriptor['source_config']
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
        sampler_entrypoint=facts['sampler_entrypoint'], sampler_sources=facts['sampler_sources'],
        sampled_token_dtype=sampler_route['dtype'],
        model_dtype=facts['model_dtype'], quantization=facts['quantization'],
        vllm_version=facts['vllm_version'], torch_version=facts['torch_version'],
        manager_block_size=manager_block, effective_config=effective_config,
        attention=invariant_layers[0])
    return dict(schema=SCHEMA, runtime_sha256=_hash(invariant), invariant_facts=invariant,
                engine_facts=copy.deepcopy(facts))
