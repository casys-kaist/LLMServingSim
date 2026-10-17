"""Descriptive measurement metadata and semantic host/worker agreement."""

import hashlib
import json

FIELDS = ('measurement_protocol', 'measurement_sha256')
PROTOCOL = 'pytorch-kineto-device-active-union'


def method_metadata(iterations):
    return dict(collector='PyTorch profiler / Kineto',
        aggregation='device-active interval union per invocation, normalized over calls',
        ownership='CPU launch correlation; CPU durations excluded',
        preparation=dict(weights='vLLM dummy initialization',
            kv='assigned pages initialized before warmup', recurrent_state='zero-start',
            queries='deterministic token IDs and explicit request phases'),
        iterations=iterations)


def semantic_identity(value):
    """Project nested execution facts, retaining geometry, dtype and backend."""
    if isinstance(value, dict):
        observations = {'runtime_versions', 'sources', 'vllm_version',
            'torch_version', 'native_sources', 'source_hashes', 'adapter_source_sha256',
            'source_sha256', 'elapsed_seconds', 'preparation_seconds', 'prepare_elapsed_seconds',
            'transformers_version', 'sampler_sources', 'endpoint_source_sha256',
            'installed_source_sha256', 'native_source_sha256', 'runtime_observations',
            'elapsed_preparation_seconds', 'sampled_rows_sha256', 'sampled_row_indices', 'full_tensor_sha256',
            'original_config_sha256'}
        if {'protocol', 'catalog', 'iterations'} <= value.keys():
            # These names are version observations only on measurement records;
            # a catalog entry's ``vllm`` selects the operation being measured.
            observations |= {'torch', 'vllm'}
        return {key: semantic_identity(item) for key, item in value.items()
                if key not in observations}
    if isinstance(value, (list, tuple)):
        return [semantic_identity(item) for item in value]
    return value


def measurement_compatibility(identity):
    """Versions describe an acquisition; executable bytes do not define data."""
    return semantic_identity(identity)


def native_scope_token(scope):
    return hashlib.sha256(json.dumps(semantic_identity(scope), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def measurement_token(identity):
    value = measurement_compatibility(identity)
    fingerprint = hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return dict(measurement_protocol=identity['protocol'], measurement_sha256=fingerprint)


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

    from .attention_lookup import protocol
    measured_protocol = protocol(PROTOCOL, attention_lookup_sha256)
    identity = dict(protocol=measured_protocol, catalog=catalog, iterations=iterations, scope_contexts=contexts,
                    **runtime_versions, method=method_metadata(iterations))
    if attention_lookup_sha256 is not None:
        identity['attention_lookup_sha256'] = attention_lookup_sha256
    if component_layout_sha256 is not None:
        identity['component_layout_sha256'] = component_layout_sha256
    if runtime_sha256 is not None:
        identity['runtime_sha256'] = runtime_sha256
    if native_profile_sha256 is not None:
        if (type(native_profile_sha256) is not str or len(native_profile_sha256) != 64
                or any(c not in '0123456789abcdef' for c in native_profile_sha256)):
            raise ValueError('Invalid native profile identity')
        identity['native_profile_sha256'] = native_profile_sha256
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
    return measurement_token(identity)
