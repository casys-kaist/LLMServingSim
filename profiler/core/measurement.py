"""Acquisition identity carried atomically with ordinary category CSV rows."""

import hashlib
import json
from pathlib import Path

FIELDS = ('measurement_protocol', 'measurement_sha256')
PROTOCOL = 'kineto-context-scoped-cuda-union-dummy-weights-query-v8'


def layerwise_identity(catalog, iterations, *, scope_contexts=None,
                          component_layout_sha256=None, attention_lookup_sha256=None,
                          runtime_sha256=None, expected_contexts=None, runtime_versions=None,
                          native_profile_sha256=None):
    """Host and worker must agree before timings can acquire this identity."""
    if component_layout_sha256 is not None and (
            not isinstance(component_layout_sha256, str) or len(component_layout_sha256) != 64
            or any(char not in '0123456789abcdef' for char in component_layout_sha256)):
        raise ValueError('Invalid component-layout identity')
    if runtime_sha256 is not None and (
            component_layout_sha256 is None or not isinstance(runtime_sha256, str)
            or len(runtime_sha256) != 64
            or any(char not in '0123456789abcdef' for char in runtime_sha256)):
        raise ValueError('Invalid component runtime identity')
    from .hooks.capture_scope_contract import normalize_contexts
    contexts = normalize_contexts(scope_contexts, catalog)
    if runtime_versions is None:
        import torch
        import vllm
        # Torch exposes a TorchVersion str subclass, not an exact str. Freeze
        # native values to plain JSON strings; explicit replay inputs stay strict.
        runtime_versions = dict(torch=str(torch.__version__), vllm=str(vllm.__version__))
    if (type(runtime_versions) is not dict or set(runtime_versions) != {'torch', 'vllm'}
            or any(type(value) is not str or not value for value in runtime_versions.values())):
        raise ValueError('Explicit runtime versions must name torch and vllm')

    source = Path(__file__)
    hooks = source.parent/'hooks'
    files = [source] + [source.parent/name for name in (
        'engine.py', 'config.py', 'stack.py', 'kimi_profile.py')] + [hooks/name for name in (
        'extension.py', 'cuda_timing.py', 'activity_collector.py', 'runner_capture.py',
        'recurrent_metadata.py', 'tp_shape.py', 'method_scopes.py', 'operator_scopes.py', 'capture_scope_contract.py',
        'skew_measurement.py', 'timings.py', 'batch.py', 'moe_hook.py',
        'sampler_shim.py', 'history.py', 'dummy_cache.py', 'cache_pages.py',
        'dummy_weights.py', 'dummy_weight_layouts.py')]
    if contexts.get('callables'):
        files.append(hooks / 'callable_scopes.py')
    if contexts['schema'] == 3:
        files.append(hooks / 'callable_context.py')
    if expected_contexts is not None:
        files.append(source.parent / 'context_tables.py')
    from .attention_lookup import protocol
    measured_protocol = protocol(PROTOCOL, attention_lookup_sha256)
    if attention_lookup_sha256 is not None:
        files += [source.parent / 'attention_lookup.py', hooks / 'attention_lookup.py']
    identity = dict(protocol=measured_protocol, catalog=catalog, iterations=iterations, scope_contexts=contexts,
                    **runtime_versions,
                    sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    if attention_lookup_sha256 is not None:
        # The shared resolver and worker adapter deliberately have the same
        # basename; retain both rather than colliding in the source map.
        identity['sources']['core/attention_lookup.py'] = hashlib.sha256(
            (source.parent / 'attention_lookup.py').read_bytes()).hexdigest()
    if component_layout_sha256 is not None:
        identity['component_layout_sha256'] = component_layout_sha256
        identity['sources']['component_layouts.py'] = hashlib.sha256(
            (source.parent / 'component_layouts.py').read_bytes()).hexdigest()
    if runtime_sha256 is not None:
        identity['runtime_sha256'] = runtime_sha256
        identity['sources']['runtime_admission.py'] = hashlib.sha256(
            (hooks / 'runtime_admission.py').read_bytes()).hexdigest()
    if native_profile_sha256 is not None:
        if (type(native_profile_sha256) is not str or len(native_profile_sha256) != 64
                or any(c not in '0123456789abcdef' for c in native_profile_sha256)):
            raise ValueError('Invalid native profile identity')
        identity['native_profile_sha256'] = native_profile_sha256
        for name in ('kimi_acquisition.py', 'hooks/kimi_runtime.py', 'hooks/kimi_inventory.py'):
            identity['sources'][name] = hashlib.sha256((source.parent / name).read_bytes()).hexdigest()
    if expected_contexts is not None:
        from .hooks.callable_context import expected_counts, contextual_catalog
        counts = expected_counts(expected_contexts)
        if {key[0] for key in counts} != set(contextual_catalog(catalog, contexts)):
            raise ValueError('Expected contexts differ from the contextual catalog')
        identity['expected_contexts'] = [dict(layer=k[0], context=k[1], context_tokens=k[2], invocations=v)
                                        for k, v in sorted(counts.items())]
    return identity


def layerwise_measurement(catalog, iterations, **kwargs):
    identity = layerwise_identity(catalog, iterations, **kwargs)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return dict(measurement_protocol=identity['protocol'], measurement_sha256=fingerprint)
