"""Acquisition identity carried atomically with ordinary category CSV rows."""

import hashlib
import json
from pathlib import Path

FIELDS = ('measurement_protocol', 'measurement_sha256')
PROTOCOL = 'cuda-active-union-dummy-kv-query-v4'


def layerwise_measurement(catalog, iterations):
    """Host and worker must agree before timings can acquire this identity."""
    import torch
    import vllm

    source = Path(__file__)
    hooks = source.parent/'hooks'
    files = [source] + [hooks/name for name in (
        'extension.py', 'cuda_timing.py', 'activity_ownership.py',
        'skew_measurement.py', 'timings.py', 'batch.py', 'moe_hook.py',
        'sampler_shim.py', 'history.py', 'dummy_cache.py')]
    identity = dict(protocol=PROTOCOL, catalog=catalog, iterations=iterations,
                    vllm=vllm.__version__, torch=torch.__version__,
                    sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return dict(measurement_protocol=PROTOCOL, measurement_sha256=fingerprint)
