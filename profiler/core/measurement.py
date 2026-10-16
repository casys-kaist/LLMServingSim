"""Acquisition identity carried atomically with ordinary category CSV rows."""

import hashlib
import json
from pathlib import Path

FIELDS = ('measurement_protocol', 'measurement_sha256')
PROTOCOL = 'kineto-scoped-cuda-union-dummy-kv-query-v6'


def layerwise_measurement(catalog, iterations):
    """Host and worker must agree before timings can acquire this identity."""
    import torch
    import vllm

    source = Path(__file__)
    hooks = source.parent/'hooks'
    files = [source] + [source.parent/name for name in (
        'engine.py', 'config.py', 'stack.py')] + [hooks/name for name in (
        'extension.py', 'cuda_timing.py', 'activity_collector.py', 'runner_capture.py',
        'recurrent_metadata.py', 'tp_shape.py',
        'skew_measurement.py', 'timings.py', 'batch.py', 'moe_hook.py',
        'sampler_shim.py', 'history.py', 'dummy_cache.py', 'cache_pages.py')]
    identity = dict(protocol=PROTOCOL, catalog=catalog, iterations=iterations,
                    vllm=vllm.__version__, torch=torch.__version__,
                    sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return dict(measurement_protocol=PROTOCOL, measurement_sha256=fingerprint)
