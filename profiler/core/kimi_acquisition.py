"""Source-admitted Kimi ordinary tables and isolated residual contexts.

The standard category planners still choose points. This adapter supplies the
native local execution contract, all-row completion and contextual acquisition;
it does not fit timings, alter workloads or certify end-to-end accuracy.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .kimi_profile import (admit_runtime, canonical, digest, original_text,
                           runtime_descriptor)
from .context_tables import _atomic

SCHEMA = 'kimi-native-categories-v1'
CATEGORIES = ('dense', 'per_sequence', 'attention', 'linear_attention', 'moe')


def catalog_digest():
    """Pin the shipped source-reviewed catalog without importing GPU code."""
    return hashlib.sha256((Path(__file__).parent.parent / 'models/kimi_k3.yaml').read_bytes()).hexdigest()


def applies(config):
    return (config or {}).get('model_type') == 'kimi_k3'


def validate_request(args, *, full=False):
    if not applies(args.model_config):
        return
    original_text(args.model_config)
    if (list(args.tp_degrees) != [1] or args.profile_mtp or args.only_skew
            or args.moe_dp_degrees is not None or tuple(args.moe_ep_degrees or (1,)) != (1,)
            or args.hf_overrides or (full and not args.skip_skew)
            or tuple(args.attention_decode_q_lens) != (1,)):
        raise NotImplementedError('Kimi acquisition requires TP1/EP1, q=1, no MTP/DP/shape overrides; '
                                  'full profiles require --skip-skew')
    if (args.num_hidden_layers not in (None, 1) or args.dtype not in (None, 'auto', 'bfloat16')
            or args.kv_cache_dtype not in (None, 'auto')):
        raise NotImplementedError('Kimi requires automatic category prefixes and native BF16 activation/cache types')
    # Validate isolated argument memory/repetition bounds before engine boot,
    # not after an expensive ordinary sweep has already completed.
    from .kimi_callable import plan
    plan(args.model_config, args.max_num_batched_tokens or 2048,
         iterations=args.measurement_iterations, warmups=1)


def stack_axes(config, category):
    # A four-layer prefix contains three routed blocks. Forced expert routing
    # needs one native router, so its own two-layer prefix ends at the first.
    return ('mlp',) if applies(config) and category.name == 'moe' else category.stack_axes


def read(path):
    raw = Path(path).read_text()
    value = json.loads(raw)
    if raw != canonical(value) + '\n':
        raise ValueError('Kimi contract is not canonical JSON')
    return value


def native_scope(llm, args, tp, limits, architecture):
    if tp != 1 or limits.moe_ep != 1:
        raise NotImplementedError('Kimi native profiles cannot emulate TP/EP shapes')
    from .config import load_architecture
    supported = load_architecture(Path(__file__).parent.parent / 'models/kimi_k3.yaml')
    if architecture != supported.model_dump(exclude_none=True):
        raise ValueError('Kimi native acquisition requires the source-reviewed catalog')
    cfg = llm.llm_engine.vllm_config
    depth = cfg.model_config.hf_text_config.num_hidden_layers
    descriptor = runtime_descriptor(args.model_config, depth,
        dict(max_model_len=cfg.model_config.max_model_len,
             max_num_seqs=cfg.scheduler_config.max_num_seqs,
             max_num_batched_tokens=cfg.scheduler_config.max_num_batched_tokens),
        cfg.cache_config.kv_cache_memory_bytes)
    responses = llm.collective_rpc('kimi_runtime_admission', args=(descriptor,))
    if not responses or len(responses) != 1:
        raise ValueError('Kimi acquisition requires exactly one native worker')
    runtime = responses[0]
    if canonical(admit_runtime(descriptor, runtime['facts'])) != canonical(runtime):
        raise ValueError('Native Kimi runtime response differs from its admission')
    return dict(schema=SCHEMA, descriptor=descriptor, runtime=runtime,
                architecture_sha256=catalog_digest())


def validate_scope(scope):
    if type(scope) is not dict or set(scope) != {'schema', 'descriptor', 'runtime', 'architecture_sha256'}:
        raise ValueError('Invalid Kimi native scope')
    if scope['schema'] != SCHEMA or canonical(admit_runtime(
            scope['descriptor'], scope['runtime']['facts'])) != canonical(scope['runtime']):
        raise ValueError('Kimi native runtime is not admitted')
    if scope['architecture_sha256'] != catalog_digest():
        raise ValueError('Kimi catalog changed since acquisition')


def expected_calls(scope):
    from .stack import resolve_stack
    stack = resolve_stack(scope['descriptor']['original_config'],
                          prefix_layers=scope['descriptor']['engine_depth'])
    kda = sum(row.attn == 'linear_attention' for row in stack)
    mla = sum(row.attn == 'full_attention' for row in stack)
    return dict(embedding=1, mlp=sum(row.mlp == 'dense' for row in stack),
        moe=sum(row.mlp == 'moe' for row in stack), final_layernorm=1, lm_head=1, sampler=1,
        kda_in_proj=kda, kda_f_b_proj=kda, kda_core=kda, kda_o_proj=kda,
        attention=mla, mla_gate=mla, mla_o_proj=mla)


def validate_capture(scope, measured, catalog, iterations, warmup):
    """Every timed forward must execute its native source-derived boundaries."""
    from .hooks.activity_collector import coverage_report
    calls = expected_calls(scope)
    if (not set(catalog) <= set(calls) or set(measured['layer_us']) != set(catalog)
            or not warmup['finite'] or len(warmup['checked_attention_layers']) !=
            scope['descriptor']['engine_depth']):
        raise ValueError('Incomplete Kimi finite-forward or canonical boundary coverage')
    report = coverage_report(measured)['canonical_invocations']
    if report['timed_forwards'] != iterations or len(report['per_forward']) != iterations:
        raise ValueError('Kimi timed-forward report differs')
    for forward in report['per_forward']:
        if forward['missing_canonical'] or set(forward['canonical']) != set(catalog):
            raise ValueError('Kimi native boundary is missing in a timed forward')
        for name, audit in forward['canonical'].items():
            if (not calls[name] or audit['scope_calls'] != calls[name]
                    or audit['positive_cuda_scope_calls'] != calls[name]):
                raise ValueError('Kimi source invocation count differs: ' + name)


class CategoryRecord:
    """One complete planned category, never a union of partially saved layers."""

    def __init__(self, directory, category, arch, shots, catalog, measurement, scope, *,
                 replace, iterations, scope_contexts):
        self.directory = Path(directory)
        self.category = category
        self.path = self.directory / ('kimi_' + category.name + '.json')
        self.keys = set()
        from .hooks.timings import TimingSample
        from .writer import sink_for
        fields = sink_for(category, directory, measurement).key_fields
        for shot in shots:
            for point in category.extract_points(shot,
                    [TimingSample(name, 1.0) for name in catalog], arch, 1):
                row = asdict(point)
                self.keys.add(tuple(row[field] for field in fields))
        if not self.keys:
            raise ValueError('Kimi category has no complete planned coordinates')
        from .measurement import layerwise_identity
        identity = layerwise_identity(catalog, iterations, scope_contexts=scope_contexts,
                                      native_profile_sha256=digest(scope))
        if measurement != dict(measurement_protocol=identity['protocol'], measurement_sha256=digest(identity)):
            raise ValueError('Kimi category identity differs from its native contract')
        self.contract = dict(schema=SCHEMA, category=category.name, scope=scope, identity=identity,
            measurement=measurement, key_fields=fields, expected_keys=sorted(self.keys))
        if self.path.exists() and not replace:
            old = read(self.path)
            if old.get('contract') != json.loads(canonical(self.contract)):
                raise ValueError('Kimi category plan/runtime changed; use fresh output or explicit --force')
            if old.get('status') == 'complete' and old.get('csv_sha256') != hashlib.sha256(
                    (self.directory / category.sink_filename).read_bytes()).hexdigest():
                raise ValueError('Completed Kimi category CSV changed after publication')
        elif (self.directory / category.sink_filename).exists() and not replace:
            raise ValueError('Existing Kimi timings lack a native acquisition contract')
        self.write('writing')

    def write(self, status, **fields):
        _atomic(self.path, canonical(dict(contract=self.contract, status=status, **fields)) + '\n')

    def resume_keys(self, sink):
        actual = set(sink._bucket)
        if actual - self.keys:
            raise ValueError('Kimi table contains unplanned coordinates')
        grouped = {}
        for key in self.keys:
            shot = key[1:] if self.category.name != 'moe' else key
            grouped.setdefault(shot, set()).add(key)
        complete = {shot for shot, keys in grouped.items() if keys <= actual}
        # Drop every partial shot before remeasurement. Never average a fresh
        # complete shot with remnants of an interrupted or edited one.
        for shot, keys in grouped.items():
            if shot not in complete:
                for key in keys:
                    sink._bucket.pop(key, None)
                    sink._row_measurements.pop(key, None)
        return complete

    def complete(self, sink):
        if set(sink._bucket) != self.keys:
            raise ValueError('Kimi planned category is incomplete')
        self.write('complete', csv_sha256=hashlib.sha256(sink.out_path.read_bytes()).hexdigest())


def acquire_contexts(llm, args, shots, directory, *, replace):
    from .kimi_callable import plan
    from .callable_profile import acquire
    from .measurement import layerwise_identity
    tokens = sorted({sum(q for q, _ in shot.requests) for shot in shots})
    plans = [plan(args.model_config, token, iterations=args.measurement_iterations, warmups=1)
             for token in tokens]
    works = [p[0] for p in plans]
    catalog, contexts = plans[0][1:3]
    target = Path(directory) / 'callables' / 'attn_res'
    fired = acquire(llm, target, works, catalog, contexts, replace=replace)
    identity = layerwise_identity(catalog, args.measurement_iterations, scope_contexts=contexts)
    record = dict(schema=SCHEMA, tokens=tokens, iterations=args.measurement_iterations, warmups=1,
                  runtime_versions={key: identity[key] for key in ('torch', 'vllm')})
    _atomic(Path(directory) / 'kimi_contexts.json', canonical(record) + '\n')
    return fired
