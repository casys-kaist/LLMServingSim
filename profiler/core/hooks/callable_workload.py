"""Bounded declared native-function workloads using the common CUDA collector.

Only already-loaded, declared functions may run. Recipes are plain data,
not executable expressions. This is isolated synthetic-input acquisition, not
an engine forward or a claim about full-model memory/working-set behavior.
"""
from collections import Counter
import hashlib
import inspect
import json
import math
import sys

from .callable_context import (bind_signature, contextual_catalog, expected_counts,
                               extract_context, validate_context_text)
from .capture_scope_contract import normalize_contexts

MAX_BYTES = 512 * 1024 * 1024
DTYPES = {'torch.bfloat16': 2, 'torch.float16': 2, 'torch.float32': 4}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _plain(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _plain(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _plain(item)
        return
    raise ValueError('Workload must contain finite JSON data only')


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('Invalid bounded ' + name)


def normalize(workload, catalog, scope_contexts):
    """Check the independent context manifest against a bounded argument plan.

    No Torch import, allocations or target resolution occurs here. The caller
    supplies BOTH the invocation plan and its independently derived manifest.
    """
    _plain(workload)
    work = json.loads(canonical(workload))
    required = {'schema', 'scope', 'source', 'shot_tokens', 'iterations', 'warmups',
                'tensors', 'calls', 'expected_contexts'}
    if type(work) is not dict or set(work) != required or type(work['schema']) is not int or work['schema'] != 1:
        raise ValueError('Invalid callable workload schema')
    if type(work['scope']) is not dict or not work['scope']:
        raise ValueError('An independent acquisition/query scope is required')
    if (type(work['source']) is not dict or set(work['source']) != {'module', 'function', 'signature'}
            or any(type(value) is not str or not value for value in work['source'].values())):
        raise ValueError('Module, native function and signature descriptions are required')
    _integer(work['shot_tokens'], 1, 2**31 - 1, 'shot tokens')
    _integer(work['iterations'], 1, 16, 'iterations')
    _integer(work['warmups'], 1, 4, 'warmups')
    contexts = normalize_contexts(scope_contexts, catalog)
    contextual = contextual_catalog(catalog, contexts)
    if len(catalog) != 1 or set(contextual) != set(catalog) or len(contexts.get('callables', [])) != 1:
        raise ValueError('Minimal workload requires exactly one contextual callable')
    declaration = contexts['callables'][0]
    layer = next(iter(catalog))
    if declaration['kind'] != 'function' or declaration.get('within') is not None:
        raise ValueError('Isolated workload cannot fabricate method or runtime-owner ancestry')
    if any(key in catalog[layer] for key in ('within', 'not_within')):
        raise ValueError('Isolated callable catalog cannot require model ancestry')
    tensors = work['tensors']
    if type(tensors) is not dict or not 1 <= len(tensors) <= 32:
        raise ValueError('Expected bounded named tensor slots')
    total = 0
    for name, tensor in tensors.items():
        if not name.isidentifier() or name.startswith('__'):
            raise ValueError('Invalid tensor slot')
        if type(tensor) is not dict or set(tensor) != {'shape', 'dtype', 'fill'}:
            raise ValueError('Only contiguous zero-offset tensor recipes are supported')
        shape = tensor['shape']
        if type(shape) is not list or not 1 <= len(shape) <= 8:
            raise ValueError('Invalid tensor rank')
        for size in shape:
            _integer(size, 1, 2**31 - 1, 'tensor dimension')
        if tensor['dtype'] not in DTYPES:
            raise ValueError('Unsupported workload tensor dtype')
        if type(tensor['fill']) not in (int, float) or not math.isfinite(tensor['fill']) or abs(tensor['fill']) > 1:
            raise ValueError('Synthetic fill must be finite and in [-1, 1]')
        total += math.prod(shape) * DTYPES[tensor['dtype']]
    if total > MAX_BYTES:
        raise ValueError('Explicit tensor slots exceed the 512MiB allocation bound')
    if type(work['calls']) is not list or not 1 <= len(work['calls']) <= 512:
        raise ValueError('Expected bounded ordered calls')
    observed, used, invariants = Counter(), set(), {}
    tensor_arguments = {rule['argument'] for rule in declaration['context']['tensors'].values()}
    for call in work['calls']:
        if type(call) is not dict or not call or any(not key.isidentifier() or key.startswith('__') for key in call):
            raise ValueError('Every call must name its explicit native arguments')
        for value in call.values():
            if type(value) is dict:
                if set(value) != {'tensor'} or type(value['tensor']) is not str or value['tensor'] not in tensors:
                    raise ValueError('Invalid tensor reference')
                used.add(value['tensor'])
            elif value is not None and type(value) not in (bool, int, float):
                raise ValueError('Arguments accept tensor references and finite numeric/None scalars only')
        if any(type(value) is dict and name not in tensor_arguments for name, value in call.items()):
            raise ValueError('Every tensor argument needs declared geometry')
        scalar_identity = canonical({name: value for name, value in call.items() if name not in tensor_arguments})
        fields, geometry, tokens = {}, {}, set()
        for label, rule in declaration['context']['fields'].items():
            if rule['argument'] not in call:
                raise ValueError('Context arguments must be explicit, including defaults')
            value = call[rule['argument']]
            if rule['operation'] == 'int':
                if type(value) is not int:
                    raise ValueError('Context scalar must be an exact integer')
                fields[label] = value
            else:
                fields[label] = value is not None
        for label, rule in declaration['context']['tensors'].items():
            if rule['argument'] not in call:
                raise ValueError('Context tensor arguments must be explicit')
            ref = call[rule['argument']]
            if ref is None and rule['optional']:
                geometry[label] = None
                continue
            if type(ref) is not dict:
                raise ValueError('Required context tensor is not a tensor slot')
            tensor = tensors[ref['tensor']]
            shape = list(tensor['shape'])
            stride = [math.prod(shape[index + 1:]) for index in range(len(shape))]
            if 'token_axis' in rule:
                axis = rule['token_axis']
                if axis >= len(shape):
                    raise ValueError('Token axis exceeds tensor rank')
                tokens.add(shape[axis])
                shape[axis] = '$tokens'
            geometry[label] = dict(shape=shape, stride=stride, dtype=tensor['dtype'], storage_offset=0)
        if len(tokens) != 1:
            raise ValueError('All tensor token axes must agree within each call')
        context = canonical(dict(schema=1, arguments=fields, geometry=geometry))
        validate_context_text(context)
        key = (layer, context, next(iter(tokens)))
        if key in invariants and scalar_identity != invariants[key]:
            raise ValueError('All scalar arguments must be invariant within each context')
        invariants[key] = scalar_identity
        observed[key] += 1
    if used != set(tensors):
        raise ValueError('Unused tensor slots are refused')
    if observed != expected_counts(work['expected_contexts']):
        raise ValueError('Invocation geometry/counts differ from independent expected contexts')
    return work, contexts


def source_identity(module, function):
    """Describe the loaded callable without pinning its implementation bytes."""
    return dict(module=module.__name__, function=function.__module__ + '.' + function.__qualname__,
                signature=str(inspect.signature(function)))


def workload_semantics(work):
    """Execution recipe, excluding descriptive source observations."""
    from profiler.core.measurement import semantic_identity
    return semantic_identity({key: value for key, value in work.items() if key != 'source'})


def measurement(workload, catalog, scope_contexts, *, runtime_versions=None):
    work, contexts = normalize(workload, catalog, scope_contexts)
    from profiler.core.measurement import layerwise_measurement, semantic_identity
    base = layerwise_measurement(catalog, work['iterations'], scope_contexts=contexts,
                                 expected_contexts=work['expected_contexts'], runtime_versions=runtime_versions)
    # Source observations remain workload metadata. Tensor geometry, argument
    # recipes and invocation counts define which timings can be reused.
    semantic_work = workload_semantics(work)
    return dict(measurement_protocol=base['measurement_protocol'] + '-isolated-native-callable',
                measurement_sha256=digest(dict(base=base, workload=semantic_work)))


def execute(workload, catalog, scope_contexts):
    """Run unchanged native functions through collect(); no alternate timer."""
    work, contexts = normalize(workload, catalog, scope_contexts)
    declaration = contexts['callables'][0]
    module = sys.modules.get(declaration['module'])
    if module is None:
        raise ValueError('Declared target module must already be loaded by the trusted runtime')
    function = getattr(module, declaration['attribute'], None)
    if not inspect.isfunction(function) or any(check(function) for check in
            (inspect.iscoroutinefunction, inspect.isgeneratorfunction, inspect.isasyncgenfunction)):
        raise ValueError('Workload target must be a synchronous native Python function')
    # Check the live callable's argument contract below. File contents and
    # annotation formatting are provenance, not an executable compatibility ABI.
    signature = bind_signature(function, declaration['context'])
    for call in work['calls']:
        if set(call) != set(signature.parameters) or any(p.kind not in
                (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) for p in signature.parameters.values()):
            raise ValueError('Every explicit native parameter is required; variadics/positional-only refused')
        signature.bind(**call)
    identity = measurement(work, catalog, contexts)
    import torch
    from .activity_collector import collect
    if torch.cuda.device_count() != 1:
        raise ValueError('Isolated workload requires exactly one visible CUDA device')
    device = torch.device('cuda:0')
    slots, arguments = {}, []
    with torch.inference_mode():
        for name, spec in work['tensors'].items():
            slots[name] = torch.full(spec['shape'], spec['fill'],
                dtype=getattr(torch, spec['dtype'].split('.')[1]), device=device)
        for call in work['calls']:
            arguments.append({key: slots[value['tensor']] if type(value) is dict else value
                              for key, value in call.items()})
        def check_metadata():
            counts = Counter()
            for kwargs in arguments:
                actual = extract_context(signature, declaration['context'], (), kwargs)
                counts[(next(iter(catalog)), actual['context'], actual['context_tokens'])] += 1
            if counts != expected_counts(work['expected_contexts']):
                raise ValueError('Actual CUDA metadata differs from independent workload geometry')
        def forward(_index=0):
            for kwargs in arguments:
                # Resolve each time so the collector's temporary scope wrapper
                # remains in the call path. Do not retain the original alias.
                getattr(module, declaration['attribute'])(**kwargs)
        check_metadata()
        for _ in range(work['warmups']):
            forward()
        torch.cuda.synchronize(device)
        check_metadata()
        class CallableWorkload(torch.nn.Module):
            def forward(self):
                forward()
        harness = CallableWorkload()
        measured = collect({'callable_workload': harness}, work['iterations'], lambda _index: harness(), catalog,
            scope_contexts=contexts, expected_contexts=work['expected_contexts'],
            strict_coverage=True, require_complete=True)
        if getattr(module, declaration['attribute']) is not function or measured['layer_us']:
            raise ValueError('Native target restoration or nonpooled context contract failed')
        torch.cuda.synchronize(device)
    evidence = dict(metadata_calls=measured['metadata']['calls'], per_call=measured['per_call'],
                    activity_ledger=measured['activity_ledger'], audit=measured['audit'],
                    device_capability=list(torch.cuda.get_device_capability(device)))
    rows = [dict(layer=row['layer'], context=row['context'], context_tokens=row['context_tokens'],
                 microseconds=row['microseconds'], **identity) for row in measured['context_samples']]
    return dict(schema=1, workload_sha256=digest(work), scope_sha256=digest(work['scope']),
                source=source_identity(module, function), rows=rows, evidence=json.loads(canonical(evidence)))
