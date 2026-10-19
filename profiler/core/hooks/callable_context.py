"""Explicit callable argument/layout identities; never tensor payloads or CPU time."""
from collections import Counter, defaultdict
import inspect
import json
import math
import re


def _name(value):
    if not isinstance(value, str) or not value.isidentifier() or value.startswith('__'):
        raise ValueError('Context names must be exact non-dunder identifiers')
    return value


def normalize_context(declaration):
    if not isinstance(declaration, dict) or set(declaration) != {'fields', 'tensors'}:
        raise ValueError('Context needs explicit fields and tensors mappings')
    result = {'fields': {}, 'tensors': {}}
    for section, rules in declaration.items():
        if not isinstance(rules, dict):
            raise ValueError('Context section must be a mapping')
        for label, rule in sorted(rules.items()):
            _name(label)
            if not isinstance(rule, dict) or 'argument' not in rule:
                raise ValueError('Context rule needs an exact argument')
            _name(rule['argument'])
            if section == 'fields':
                if set(rule) != {'argument', 'operation'} or rule['operation'] not in ('int', 'present'):
                    raise ValueError('Unsupported context field operation')
                result[section][label] = dict(rule)
            else:
                if set(rule) - {'argument', 'optional', 'token_axis'}:
                    raise ValueError('Unknown tensor context option')
                optional = rule.get('optional', False)
                if type(optional) is not bool:
                    raise ValueError('Tensor optional must be boolean')
                item = dict(argument=rule['argument'], optional=optional)
                if 'token_axis' in rule:
                    axis = rule['token_axis']
                    if type(axis) is not int or axis < 0:
                        raise ValueError('Token axis must be a nonnegative integer')
                    item['token_axis'] = axis
                result[section][label] = item
    if not any('token_axis' in rule and not rule['optional'] for rule in result['tensors'].values()):
        raise ValueError('Context needs a required tensor token axis')
    return result


def bind_signature(function, declaration):
    signature = inspect.signature(function)
    for rules in declaration.values():
        for rule in rules.values():
            parameter = signature.parameters.get(rule['argument'])
            if parameter is None or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                raise ValueError('Declared context argument is not an explicit callable parameter')
    return signature


def extract_context(signature, declaration, args, kwargs):
    import torch
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    arguments, geometry, token_counts = {}, {}, set()
    for label, rule in declaration['fields'].items():
        value = bound.arguments[rule['argument']]
        if rule['operation'] == 'int':
            if type(value) is not int:
                raise ValueError('Context integer must be an exact Python int')
        else:
            value = value is not None
        arguments[label] = value
    for label, rule in declaration['tensors'].items():
        value = bound.arguments[rule['argument']]
        if value is None and rule['optional']:
            geometry[label] = None
            continue
        if not isinstance(value, torch.Tensor) or value.device.type != 'cuda' or value.layout != torch.strided:
            raise ValueError('Context tensors must be actual strided CUDA tensors')
        shape, stride = list(value.shape), list(value.stride())
        offset = value.storage_offset()
        if (len(shape) != len(stride) or any(type(n) is not int or n < 1 for n in shape)
                or any(type(n) is not int or n < 0 for n in stride)
                or type(offset) is not int or offset < 0):
            raise ValueError('Context tensor geometry is not concrete and valid')
        if 'token_axis' in rule:
            axis = rule['token_axis']
            if axis >= len(shape):
                raise ValueError('Token axis exceeds tensor rank')
            token_counts.add(shape[axis])
            shape[axis] = '$tokens'
        geometry[label] = dict(shape=shape, stride=stride, dtype=str(value.dtype), storage_offset=offset)
    if len(token_counts) != 1:
        raise ValueError('All declared tensor token dimensions must agree')
    value = dict(schema=1, arguments=arguments, geometry=geometry)
    return dict(context=json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False),
                context_tokens=next(iter(token_counts)))


def validate_context_text(value):
    if not isinstance(value, str):
        raise ValueError('Context must be canonical JSON text')
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError('Repeated context key')
            result[key] = item
        return result
    item = json.loads(value, object_pairs_hook=unique)
    if (not isinstance(item, dict) or set(item) != {'schema', 'arguments', 'geometry'}
            or type(item['schema']) is not int or item['schema'] != 1
            or not isinstance(item['arguments'], dict) or not isinstance(item['geometry'], dict)):
        raise ValueError('Invalid context payload schema')
    for key, scalar in item['arguments'].items():
        _name(key)
        if type(scalar) not in (bool, int):
            raise ValueError('Invalid context scalar')
    tokens = 0
    for key, geometry in item['geometry'].items():
        _name(key)
        if geometry is None:
            continue
        if not isinstance(geometry, dict) or set(geometry) != {'shape', 'stride', 'dtype', 'storage_offset'}:
            raise ValueError('Invalid context geometry fields')
        shape, stride = geometry['shape'], geometry['stride']
        if (not isinstance(shape, list) or not isinstance(stride, list) or len(shape) != len(stride)
                or any(not (type(n) is int and n > 0) and n != '$tokens' for n in shape)
                or any(type(n) is not int or n < 0 for n in stride)
                or shape.count('$tokens') > 1
                or not isinstance(geometry['dtype'], str)
                or re.fullmatch(r'torch\.[a-z][a-z0-9_]*', geometry['dtype']) is None
                or type(geometry['storage_offset']) is not int or geometry['storage_offset'] < 0):
            raise ValueError('Invalid context geometry')
        tokens += shape.count('$tokens')
    if not tokens or json.dumps(item, sort_keys=True, separators=(',', ':'), allow_nan=False) != value:
        raise ValueError('Context must contain a token dimension and use canonical JSON')
    return item


def contextual_catalog(catalog, contexts):
    declarations = {item['name']: item['context'] for item in contexts.get('callables', ()) if 'context' in item}
    result = {}
    for name, entry in catalog.items():
        selector = entry.get('callable')
        selector = selector.get('name') if isinstance(selector, dict) else selector
        if isinstance(selector, str) and selector in declarations:
            result[name] = declarations[selector]
    for name in result:
        entry = catalog[name]
        if entry.get('normalization') != 'invocations':
            raise ValueError('Context-bearing callables require invocations normalization')
        if type(entry.get('occurrences', 1)) is not int or entry.get('occurrences', 1) < 1:
            raise ValueError('Context occurrence metadata must be a positive integer')
    return result


def expected_counts(manifest):
    if not isinstance(manifest, list) or not manifest:
        raise ValueError('Context acquisition requires an independent nonempty manifest')
    result = Counter()
    for item in manifest:
        if not isinstance(item, dict) or set(item) != {'layer', 'context', 'context_tokens', 'invocations'}:
            raise ValueError('Invalid expected context manifest row')
        if not isinstance(item['layer'], str) or not item['layer']:
            raise ValueError('Manifest layer must be nonempty text')
        validate_context_text(item['context'])
        if any(type(item[key]) is not int or item[key] < 1 for key in ('context_tokens', 'invocations')):
            raise ValueError('Manifest tokens and invocation counts must be positive integers')
        key = (item['layer'], item['context'], item['context_tokens'])
        if key in result:
            raise ValueError('Repeated expected context')
        result[key] = item['invocations']
    return result


def split_context_timings(result, catalog, contexts, iterations, manifest=None, *, diagnostic=False):
    if type(iterations) is not int or iterations < 1:
        raise ValueError('Context acquisition needs positive timed-forward count')
    selected = contextual_catalog(catalog, contexts)
    if not selected:
        if manifest is not None:
            raise ValueError('Unexpected context manifest for a contextless category')
        return result
    expected = expected_counts(manifest) if manifest is not None else None
    if expected is None and not diagnostic:
        raise ValueError('Context acquisition needs its independent expected-context manifest')
    if expected is not None and {key[0] for key in expected} != set(selected):
        raise ValueError('Manifest and contextual catalog layers differ')
    if result['audit']['overlapping_canonical_claims'] and not diagnostic:
        raise ValueError('Context acquisition cannot hide overlapping canonical CUDA claims')
    records = {item['id']: item for item in result['metadata']['calls']}
    grouped = [defaultdict(list) for _ in range(iterations)]
    for call in result['per_call']:
        if call['canonical'] not in selected:
            continue
        record = records[call['id']]
        if record.get('scope_kind') != 'callable' or 'context' not in record:
            raise ValueError('Context measurement is missing its actual callable identity')
        validate_context_text(record['context'])
        tokens = record.get('context_tokens')
        value = call['union_us']
        forward = call['forward']
        if (type(tokens) is not int or tokens < 1 or type(forward) is not int or not 0 <= forward < iterations
                or not math.isfinite(value) or value <= 0):
            raise ValueError('Invalid contextual call geometry or CUDA union')
        grouped[forward][(call['canonical'], record['context'], tokens)].append(value)
    def samples(groups):
        return [dict(layer=key[0], context=key[1], context_tokens=key[2],
                     microseconds=sum(values) / len(values), invocations=len(values))
                for key, values in sorted(groups.items())]
    counts = [Counter({key: len(values) for key, values in groups.items()}) for groups in grouped]
    if any(count != counts[0] for count in counts) or (expected is not None and any(count != expected for count in counts)):
        raise ValueError('Per-forward context invocation counts differ from the manifest')
    combined = defaultdict(list)
    for groups in grouped:
        for key, values in groups.items():
            combined[key].extend(values)
    # Delete every pooled representation from the latency interfaces. The
    # summary tree/per-call ledger remain diagnostics, never timing samples.
    for holder in [result, *result['per_forward']]:
        holder['layer_us'] = {name: value for name, value in holder['layer_us'].items() if name not in selected}
        holder['components'] = [item for item in holder['components'] if item['canonical'] not in selected]
    if 'uncorrected_method_parent_layer_us' in result:
        result['uncorrected_method_parent_layer_us'] = {
            name: value for name, value in result['uncorrected_method_parent_layer_us'].items()
            if name not in selected}
    result['per_forward_layers_us'] = [item['layer_us'] for item in result['per_forward']]
    result['context_samples'] = samples(combined)
    result['per_forward_context_samples'] = [samples(groups) for groups in grouped]
    result['contextual_layers'] = sorted(selected)
    return result
