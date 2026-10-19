"""Compact serving-execution observations, separate from acquisition settings.

This module is CPU-only. A recorded graph policy describes one deployment; it
neither changes measured CSVs nor turns eager measurements into graph timings.
"""

from copy import deepcopy
import json
from pathlib import Path

from .bundle_metadata import update_bundle


_SCOPE_FIELDS = (
    'tp_size', 'pp_size', 'dp_size', 'ep_size', 'max_num_seqs',
    'max_num_batched_tokens', 'num_speculative_tokens', 'block_size',
    'enable_chunked_prefill', 'enable_prefix_caching', 'max_model_len',
    'activation_dtype', 'kv_cache_dtype', 'quantization',
)
_MODES = {'NONE', 'FULL', 'PIECEWISE', 'FULL_DECODE_ONLY', 'FULL_AND_PIECEWISE'}
_VOLATILE_MODEL_FIELDS = {'transformers_version', 'name_or_path', 'id2label', 'label2id'}
_DTYPES = {'bf16': 'bfloat16', 'fp16': 'float16', 'half': 'float16',
           'fp32': 'float32', 'float': 'float32'}


def canonical_dtype(value):
    if value is None:
        return None
    value = str(value).lower().removeprefix('torch.')
    return _DTYPES.get(value, value)


def model_dimensions(config):
    """Generic semantic config, not a family-specific geometry whitelist.

    Worker snapshots retain effective defaults; import binds those observations
    to the complete declared config and stores only that compact declaration.
    """
    if not isinstance(config, dict) or not config:
        raise ValueError('Execution metadata needs a nonempty model config')

    def normalize(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if type(key) is int:
                    key = str(key)
                elif not isinstance(key, str):
                    raise ValueError('Model config keys must be strings or integers')
                if key.startswith('_') or key in _VOLATILE_MODEL_FIELDS:
                    continue
                normalized_key = 'dtype' if key == 'torch_dtype' else key
                normalized = canonical_dtype(item) if normalized_key == 'dtype' else normalize(item)
                if normalized_key in result:
                    if key not in ('dtype', 'torch_dtype') or result[normalized_key] != normalized:
                        raise ValueError('Conflicting canonical keys in model config')
                result[normalized_key] = normalized
            return result
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise ValueError(f'Unsupported model config value: {type(value).__name__}')

    result = normalize(config)
    if not result:
        raise ValueError('Model config has no semantic fields')
    return result


def _contains(observed, declared):
    if isinstance(declared, dict):
        return isinstance(observed, dict) and all(
            key in observed and _contains(observed[key], value)
            for key, value in declared.items())
    if isinstance(declared, list):
        return isinstance(observed, list) and len(observed) == len(declared) and all(
            _contains(a, b) for a, b in zip(observed, declared))
    return type(observed) is type(declared) and observed == declared


def bind_model_config(observed, declared, resolved_declared):
    """Bind a declaration only after its native config class resolved aliases.

    The worker supplies resolved_declared from its installed config class; the
    offline consumer needs no Transformers dependency or model-specific aliases.
    """
    if not _contains(model_dimensions(observed), model_dimensions(resolved_declared)):
        raise ValueError('Serving model overrides differ from the declared model config')
    return model_dimensions(declared)


def precision_scope(model_config, activation_dtype=None, kv_cache_dtype=None,
                    quantization=None):
    """Precision axes, keeping quantized weights separate from activations."""
    config = model_dimensions(model_config)
    text = config.get('text_config') or config
    activation = canonical_dtype(activation_dtype)
    if activation in (None, 'auto'):
        activation = canonical_dtype(text.get('dtype', config.get('dtype')))
    kv = canonical_dtype(kv_cache_dtype)
    if kv in (None, 'auto'):
        kv = canonical_dtype(text.get('kv_cache_dtype', config.get('kv_cache_dtype')))
    if kv in (None, 'auto'):
        kv = activation
    quant = text.get('quantization_config', config.get('quantization_config')) or {}
    method = quantization if quantization is not None else quant.get('quant_method')
    return dict(activation_dtype=activation, kv_cache_dtype=kv,
                quantization=str(method).lower() if method else None)


def expected_capture_grids(graph, scope):
    """Reconstruct only the fixed-query, no-LoRA/non-DBO dispatch contract."""
    from serving.core.cudagraph import resolve_graph_config
    options = {key: graph[key] for key in ('runner', 'mode', 'enable_sp')}
    if graph['capture_sizes']:
        options['capture_sizes'] = list(graph['capture_sizes'])
    config = resolve_graph_config(options, scope['max_num_seqs'],
                                  scope['max_num_batched_tokens'],
                                  scope['num_speculative_tokens'], scope['tp_size'])
    if tuple(graph['capture_sizes']) != config.capture_sizes:
        raise ValueError('Recorded base grid cannot be represented by this resolver')
    full = (config.capture_sizes if config.mode == 'FULL' else
            config.decode_capture_sizes if config.mode in ('FULL_DECODE_ONLY', 'FULL_AND_PIECEWISE')
            else ())
    piecewise = config.capture_sizes if config.mode in ('PIECEWISE', 'FULL_AND_PIECEWISE') else ()
    return list(full), list(piecewise)


def _positive(value):
    return type(value) is int and value > 0


def validate_execution(record):
    """Validate only the supported dispatch contract, not source/version hashes."""
    if not isinstance(record, dict) or record.get('status') != 'observed':
        raise ValueError('Execution metadata has no complete worker observation')
    if record.get('source') != 'worker_capture':
        raise ValueError('Execution metadata must come from completed worker startup')
    scope, graph = record.get('scope'), record.get('graph')
    if not isinstance(scope, dict) or not isinstance(graph, dict):
        raise ValueError('Execution metadata needs scope and graph mappings')
    for key in _SCOPE_FIELDS:
        value = scope.get(key)
        if key == 'quantization':
            valid = key in scope and (value is None or isinstance(value, str) and bool(value))
        elif key.endswith('_dtype'):
            valid = isinstance(value, str) and bool(value) and value != 'auto' and canonical_dtype(value) == value
        elif key.startswith('enable_'):
            valid = type(value) is bool
        elif key == 'num_speculative_tokens':
            valid = type(value) is int and value >= 0
        else:
            valid = _positive(value)
        if not valid:
            raise ValueError(f'Invalid execution scope field: {key}')
    if model_dimensions(record.get('model_dimensions')) != record['model_dimensions']:
        raise ValueError('Observed model config must use canonical semantic fields')
    if not isinstance(record.get('gpu'), str) or not record['gpu']:
        raise ValueError('Execution metadata needs the observed device name')
    if graph.get('runner') not in ('v1', 'v2') or graph.get('mode') not in _MODES:
        raise ValueError('Unknown graph runner or mode')
    if type(graph.get('enable_sp')) is not bool:
        raise ValueError('Graph compiler SP must be boolean')
    sizes = graph.get('capture_sizes')
    if (not isinstance(sizes, list) or any(not _positive(n) for n in sizes)
            or sorted(set(sizes)) != sizes):
        raise ValueError('Capture sizes must be a sorted, unique integer list')
    if graph['mode'] != 'NONE' and not sizes:
        raise ValueError('An enabled graph mode needs capture sizes')
    if graph['mode'] == 'NONE' and sizes:
        raise ValueError('A disabled graph mode must not claim capture sizes')
    if sizes and sizes[-1] > scope['max_num_batched_tokens']:
        raise ValueError('Capture sizes exceed the serving token budget')
    for key in ('full_capture_sizes', 'piecewise_capture_sizes'):
        values = graph.get(key)
        if (not isinstance(values, list) or any(not _positive(n) for n in values)
                or sorted(set(values)) != values):
            raise ValueError(f'Invalid actual capture grid: {key}')
    if graph.get('implementation') not in ('compiled', 'breakable'):
        raise ValueError('Unknown graph implementation')
    full, piecewise = expected_capture_grids(graph, scope)
    if graph['full_capture_sizes'] != full or graph['piecewise_capture_sizes'] != piecewise:
        raise ValueError('Actual FULL/PIECEWISE captures differ from the supported dispatch contract')
    return record


def merge_workers(workers):
    """Require one matching observation from every TP/PP/DP worker."""
    flat = []
    def flatten(value):
        if isinstance(value, list):
            for item in value:
                flatten(item)
        elif isinstance(value, dict):
            flat.append(value)
        else:
            raise ValueError('Unexpected worker snapshot response')
    flatten(workers)
    if not flat:
        raise ValueError('No workers returned execution metadata')
    first = validate_execution(flat[0])
    expected = first['scope']['tp_size'] * first['scope']['pp_size'] * first['scope']['dp_size']
    if len(flat) != expected:
        raise ValueError(f'Execution snapshot covers {len(flat)} workers, expected {expected}')
    comparable = {key: value for key, value in first.items() if key != 'rank'}
    ranks = set()
    for row in flat:
        validate_execution(row)
        rank = row.get('rank')
        if type(rank) is not int or rank < 0 or rank in ranks:
            raise ValueError('Missing or duplicate execution worker rank')
        ranks.add(rank)
        if {key: value for key, value in row.items() if key != 'rank'} != comparable:
            raise ValueError('Workers disagree on the serving execution policy')
    if ranks != set(range(expected)):
        raise ValueError('Execution snapshot does not cover every global worker rank')
    return deepcopy(comparable)


def _check_bundle_precision(meta, root, model_config, scope):
    """Check standard precision labels without treating fp8 as activation dtype."""
    if meta.get('variant') != root.name:
        raise ValueError('Bundle variant label differs from its directory')
    short = {'bfloat16': 'bf16', 'float16': 'fp16', 'float32': 'fp32',
             'fp8_e4m3': 'fp8', 'fp8_e5m2': 'fp8e5m2'}
    checkpoint = precision_scope(model_config)
    if scope['quantization'] != checkpoint['quantization']:
        raise ValueError('Observed weight quantization differs from the target checkpoint')
    activation = scope['activation_dtype']
    base = short.get(activation, activation)
    if scope['quantization']:
        base = short.get(scope['quantization'], scope['quantization'])
        if (checkpoint['activation_dtype'] not in ('bfloat16', 'float16')
                or activation != checkpoint['activation_dtype']):
            base += '-act' + short.get(activation, activation)
    kv = scope['kv_cache_dtype']
    variants = {base + '-kv' + short.get(kv, kv)}
    if kv == activation:
        variants.add(base)
    # Older bundles may omit precision flags. When present, they must agree;
    # eager acquisition is otherwise deliberately unrelated to graph policy.
    effective = meta.get('engine_effective') or {}
    explicit_precision = True
    for key, observed in (('dtype', activation), ('kv_cache_dtype', kv)):
        declared = canonical_dtype(effective.get(key))
        if declared in (None, 'auto'):
            explicit_precision = False
        if declared not in (None, 'auto') and declared != observed:
            raise ValueError(f'Observed {key} differs from bundle acquisition precision')
    if meta['variant'] not in variants and not explicit_precision:
        raise ValueError('A custom variant needs explicit matching acquisition dtype and kv_cache_dtype')


def import_execution(bench_dir, bundle_root, model_config):
    """Import observed serving facts explicitly; never relabel latency tables."""
    source = Path(bench_dir) / 'meta.json'
    benchmark = json.loads(source.read_text())
    record = deepcopy(validate_execution(benchmark.get('execution')))
    root = Path(bundle_root)
    if not (root / 'meta.yaml').is_file():
        raise ValueError('Import execution into an existing profile bundle')
    if benchmark.get('step_audit'):
        raise ValueError('Perturbed diagnostic runs are not serving references')
    dimensions = model_dimensions(model_config)
    if not _contains(record['model_dimensions'], dimensions):
        raise ValueError('Observed model geometry differs from the target model config')
    record['model_dimensions'] = dimensions
    record['vllm_version'] = benchmark.get('vllm_version')
    record['recorded_at'] = benchmark.get('started_at')

    def apply(meta):
        if benchmark.get('model') != meta.get('model'):
            raise ValueError('Benchmark and bundle model IDs must match')
        if record['gpu'] != meta.get('gpu'):
            raise ValueError('Benchmark device differs from the profile bundle')
        _check_bundle_precision(meta, root, model_config, record['scope'])
        records = meta.setdefault('serving_execution', [])
        if not isinstance(records, list):
            raise ValueError('serving_execution must be a list of deployment observations')
        # Keep one current observation per deployment, not an append-only log.
        records[:] = [old for old in records if old.get('scope') != record['scope']]
        records.append(record)
    update_bundle(root, apply)
    return record


def select_execution(meta, scope, model_config):
    """Return a matching observed policy, or a reason to derive it instead."""
    records = meta.get('serving_execution') or []
    if not isinstance(records, list):
        raise ValueError('serving_execution must be a list')
    if not records:
        return None, None
    dimensions = model_dimensions(model_config)
    matching = []
    invalid = False
    for record in records:
        try:
            validate_execution(record)
        except (ValueError, TypeError, KeyError):
            invalid = True
            continue
        if any(record['scope'].get(k) != scope.get(k) for k in _SCOPE_FIELDS):
            continue
        if not (_contains(record['model_dimensions'], dimensions)
                and _contains(dimensions, record['model_dimensions'])):
            continue
        if record['gpu'] != meta.get('gpu'):
            continue
        matching.append(record)
    if not matching:
        reason = ('recorded serving observation is unsupported or incomplete' if invalid else
                  'recorded serving configuration does not match this deployment')
        return None, reason
    # A mode/grid alone cannot choose between different backend or context limits.
    policies = [r['graph'] for r in matching]
    if any(p != policies[0] for p in policies[1:]):
        return None, 'multiple matching serving graph policies; configure cudagraph explicitly'
    return deepcopy(matching[-1]), None


def graph_options(record):
    graph = validate_execution(record)['graph']
    result = {key: graph[key] for key in ('runner', 'mode', 'enable_sp')}
    if graph['capture_sizes']:
        result['capture_sizes'] = list(graph['capture_sizes'])
    return result
