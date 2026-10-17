"""Exact isolated-callable acquisition/resume/query seam, separate from engines.

The common API owns publication and exact context lookup. Source adapters may
provide bounded native argument plans; they do not choose a profiling grid or
claim whole-model execution. Targets and manifests are supplied before fire.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path

from .context_tables import (ContextDenseTable, _identity, normalize_manifest,
                             read_contract, read_namespace, update_namespace)
from .context_lookup import FrozenContextLookup
from .hooks.callable_workload import canonical, digest, measurement, normalize, workload_semantics
from .hooks.callable_context import expected_counts
from .hooks.cuda_timing import union_ns


NATIVE_PROFILE_SCHEMA = 'native-profile-categories'


def hybrid_residual_plan(config, tokens, *, iterations, warmups):
    """Plan the source model's native residual calls at one explicit token count.

    The existing native callable runner enforces its allocation, iteration and
    call-count bounds. A source model that exceeds them fails before allocation;
    its residual capacity or schedule is never reduced to fit the envelope.
    Full-depth role coverage here is isolated-callable coverage, not a claim
    that a full-depth model ran or that its resident-memory capacity is known.
    """
    from .hooks.runtime_admission import HybridResidualProfile
    from .hooks.capture_scope_contract import normalize_contexts
    from .hooks.callable_workload import DTYPES, MAX_BYTES

    adapter = HybridResidualProfile()
    text = adapter.original_text(config)
    roles = adapter.roles(config)
    if type(tokens) is not int or not 1 <= tokens <= 2**31 - 1:
        raise ValueError('Residual callable tokens must be a positive bounded integer')
    if text.get('dtype', config.get('dtype')) not in ('bfloat16', 'bf16'):
        raise NotImplementedError('Native residual argument plan requires explicit BF16 source dtype')
    hidden, depth = text['hidden_size'], text['num_hidden_layers']
    capacity = roles[-1]['capacity']
    fields = dict(
        num_blocks=dict(argument='num_blocks', operation='int'),
        block_write_idx=dict(argument='block_write_idx', operation='int'),
        delta_present=dict(argument='delta', operation='present'),
        output_norm_present=dict(argument='output_norm_weight', operation='present'))
    tensor_names = ('prefix', 'delta', 'blocks', 'norm_weight', 'qk_weight', 'output_norm_weight')
    token_tensors = frozenset(('prefix', 'delta', 'blocks'))
    tensors = {name: dict(argument=name,
        **(dict(token_axis=0) if name in token_tensors else {}),
        **(dict(optional=True) if name in ('delta', 'output_norm_weight') else {}))
        for name in tensor_names}
    declaration = dict(name='attn_res', kind='function',
        module='vllm.models.kimi_k3.nvidia.model', attribute='attn_res',
        context=dict(fields=fields, tensors=tensors))
    catalog = dict(attn_res=dict(callable='attn_res', normalization='invocations'))
    contexts = normalize_contexts(catalog=dict(attn_res=dict(catalog['attn_res'], callable=declaration)))
    slots = {name: dict(
        shape=([tokens, capacity, hidden] if name == 'blocks' else
               [tokens, hidden] if name in token_tensors else [hidden]),
        dtype='torch.bfloat16', fill=0.0 if name == 'qk_weight' else 0.125)
        for name in tensor_names}
    allocated = sum(math.prod(slot['shape']) * DTYPES[slot['dtype']] for slot in slots.values())
    if allocated > MAX_BYTES:
        raise ValueError('Source residual argument plan exceeds the common callable allocation bound: '
                         f'{allocated} > {MAX_BYTES} bytes')
    calls, counts, role_contexts = [], Counter(), []
    for role in roles:
        absent = set()
        if not role['delta_present']:
            absent.add('delta')
        if not role['output_norm_present']:
            absent.add('output_norm_weight')
        call = {name: None if name in absent else dict(tensor=name) for name in tensor_names}
        call.update({name: role[name] for name in ('num_blocks', 'block_write_idx', 'eps', 'output_norm_eps')})
        calls.append(call)
        geometry = {}
        for name, slot in slots.items():
            if name in absent:
                geometry[name] = None
                continue
            shape = list(slot['shape'])
            stride = [math.prod(shape[index + 1:]) for index in range(len(shape))]
            if name in token_tensors:
                shape[0] = '$tokens'
            geometry[name] = dict(shape=shape, stride=stride, dtype=slot['dtype'], storage_offset=0)
        context = canonical(dict(schema=1, arguments={name: role[name] for name in fields}, geometry=geometry))
        counts[context] += 1
        role_contexts.append(context)
    signature = ('(prefix: torch.Tensor, delta: torch.Tensor | None, blocks: torch.Tensor, '
                 'norm_weight: torch.Tensor, qk_weight: torch.Tensor, '
                 'output_norm_weight: torch.Tensor | None, num_blocks: int, '
                 'block_write_idx: int, eps: float, output_norm_eps: float) -> torch.Tensor')
    work = dict(schema=1, scope=dict(kind='hybrid-residual-isolated-attn-res',
        execution_depth=depth, capacity=capacity,
        hidden=hidden, tp=1, pp=1, dtype='torch.bfloat16',
        allocation_limit_bytes=MAX_BYTES),
        source=dict(module=declaration['module'], function=declaration['attribute'],
                    signature=signature),
        shot_tokens=tokens, iterations=iterations, warmups=warmups, tensors=slots, calls=calls,
        expected_contexts=[dict(layer='attn_res', context=context, context_tokens=tokens, invocations=count)
                           for context, count in sorted(counts.items())])
    work, contexts = normalize(work, catalog, contexts)
    return work, catalog, contexts, role_contexts


def profile_for(architecture):
    """Select an explicitly declared runtime adapter, never a checkpoint name."""
    if hasattr(architecture, 'model_dump'):
        # Keep declared fields, not unrelated typed-model defaults. Callable
        # context defaults are normalized below with the shared CPU helper.
        architecture = architecture.model_dump(exclude_unset=True, exclude_none=True)
    name = architecture.get('native_profile')
    if name is None:
        return None
    if name == 'hybrid-residual':
        from .hooks.runtime_admission import HybridResidualProfile
        return NativeProfile(name, architecture, HybridResidualProfile())
    raise ValueError('Unknown native acquisition profile: ' + str(name))


class NativeProfile:
    """Common host/worker publication around a narrow source-specific adapter.

    Adapters describe admitted runtime facts and contextual invocation geometry.
    Category planning, complete-table resume, capture counts and callable table
    transactions continue to use the same mechanisms for every architecture.
    """

    def __init__(self, name, architecture, adapter):
        from .hooks.capture_scope_contract import normalize_callable
        self.name = name
        self.architecture = json.loads(canonical(architecture))
        self.architecture['native_profile'] = name
        for entries in self.architecture['catalog'].values():
            for entry in entries.values():
                if entry.get('callable') is not None:
                    entry['callable'] = normalize_callable(entry['callable'])
        self.adapter = adapter

    def validate_request(self, args, *, full=False):
        self.adapter.validate_request(args, full=full)

    def stack_axes(self, category):
        return self.adapter.stack_axes(category)

    def engine_overrides(self):
        return self.adapter.engine_overrides()

    def ordinary_catalog(self, catalog):
        return {name: entry for name, entry in catalog.items()
                if name not in self.adapter.context_layers}

    def native_scope(self, llm, args, tp, limits):
        # A local native profile cannot claim single-worker shape emulation as
        # distributed execution. A later distributed adapter needs its own RPC
        # and rank-agreement contract before these bounds can be extended.
        if tp != 1 or limits.moe_ep != 1:
            raise NotImplementedError('Local native profiles require TP1/EP1')
        cfg = llm.llm_engine.vllm_config
        descriptor = self.adapter.runtime_descriptor(args.model_config,
            cfg.model_config.hf_text_config.num_hidden_layers,
            dict(max_model_len=cfg.model_config.max_model_len,
                 max_num_seqs=cfg.scheduler_config.max_num_seqs,
                 max_num_batched_tokens=cfg.scheduler_config.max_num_batched_tokens),
            cfg.cache_config.kv_cache_memory_bytes)
        responses = llm.collective_rpc(self.adapter.rpc_method, args=(self.architecture, descriptor))
        if type(responses) not in (tuple, list) or len(responses) != 1:
            raise ValueError('Local native acquisition requires exactly one worker')
        scope = dict(schema=NATIVE_PROFILE_SCHEMA, profile=self.name, architecture=self.architecture,
                     descriptor=descriptor, runtime=responses[0])
        self.validate_scope(scope)
        return scope

    def validate_scope(self, scope):
        if (type(scope) is not dict or set(scope) != {'schema', 'profile', 'architecture',
                'descriptor', 'runtime'} or scope['schema'] != NATIVE_PROFILE_SCHEMA
                or scope['profile'] != self.name
                or canonical(scope['architecture']) != canonical(self.architecture)):
            raise ValueError('Native acquisition scope/catalog differs')
        admitted = self.adapter.admit_runtime(scope['descriptor'], scope['runtime']['facts'])
        if canonical(admitted) != canonical(scope['runtime']):
            raise ValueError('Native runtime response differs from its source admission')

    def validate_capture(self, scope, measured, catalog, iterations, warmup):
        from .hooks.activity_collector import coverage_report
        self.adapter.validate_capture_extra(scope, warmup)
        calls = self.adapter.expected_calls(scope)
        if not set(catalog) <= set(calls) or set(measured['layer_us']) != set(catalog):
            raise ValueError('Incomplete native canonical boundary coverage')
        report = coverage_report(measured)['canonical_invocations']
        if report['timed_forwards'] != iterations or len(report['per_forward']) != iterations:
            raise ValueError('Native timed-forward report differs')
        for forward in report['per_forward']:
            if forward['missing_canonical'] or set(forward['canonical']) != set(catalog):
                raise ValueError('Native boundary is missing in a timed forward')
            for name, audit in forward['canonical'].items():
                if (not calls[name] or audit['scope_calls'] != calls[name]
                        or audit['positive_cuda_scope_calls'] != calls[name]):
                    raise ValueError('Native source invocation count differs: ' + name)

    def acquire_contexts(self, llm, args, shots, directory, *, replace):
        from .measurement import layerwise_identity
        tokens = sorted({sum(q for q, _ in shot.requests) for shot in shots})
        plans = [self.adapter.context_plan(args.model_config, token,
                    iterations=args.measurement_iterations, warmups=1) for token in tokens]
        works = [plan[0] for plan in plans]
        catalog, contexts = plans[0][1:3]
        if any(plan[1:3] != plans[0][1:3] for plan in plans):
            raise ValueError('Callable declarations changed across planned token counts')
        fired = acquire(llm, Path(directory) / 'callables', works, catalog, contexts, replace=replace)
        identity = layerwise_identity(catalog, args.measurement_iterations, scope_contexts=contexts)
        record = dict(schema=NATIVE_PROFILE_SCHEMA, profile=self.name, tokens=tokens,
            iterations=args.measurement_iterations, warmups=1,
            runtime_versions={key: identity[key] for key in ('torch', 'vllm')})
        if fired:
            update_namespace(Path(directory) / 'callables', 'callables',
                             lambda old: dict(old or {}, options=record))
        return fired


def validate_response(work, response, stamp):
    """Join same-capture metadata, physical calls and CUDA ledger before saving.

    Generic ownership/count/union checks do not certify a model-specific route
    or logical tensor sizes. Such a consumer must additionally audit the saved
    native activity names against its admitted backend contract.
    """
    if (type(response) is not dict or set(response) !=
            {'schema', 'workload_sha256', 'scope_sha256', 'source', 'rows', 'evidence'}
            or type(response['schema']) is not int or response['schema'] != 1
            or response['workload_sha256'] != digest(work)
            or response['scope_sha256'] != digest(work['scope'])):
        raise ValueError('Worker workload/source/scope identity differs')
    if (type(response['source']) is not dict or set(response['source']) != {'module', 'function', 'signature'}
            or any(type(value) is not str or not value for value in response['source'].values())):
        raise ValueError('Missing callable execution metadata')
    evidence = response['evidence']
    if type(evidence) is not dict or set(evidence) != {'metadata_calls', 'per_call', 'activity_ledger', 'audit', 'device_capability'}:
        raise ValueError('Missing same-acquisition ownership evidence')
    capability = evidence['device_capability']
    if type(capability) is not list or len(capability) != 2 or any(type(x) is not int or x < 0 for x in capability):
        raise ValueError('Invalid actual device capability')
    audit = evidence['audit']
    for key in ('unbound_under_module', 'unscoped_gpu', 'outside_forward', 'overlapping_canonical_claims',
                'missing_canonical', 'zero_canonical', 'invalid_forward_values'):
        if key not in audit or audit[key] != []:
            raise ValueError('Native ownership audit is incomplete or failed')
    metadata = {row['id']: row for row in evidence['metadata_calls']}
    calls = {row['id']: row for row in evidence['per_call']}
    if (len(metadata) != len(evidence['metadata_calls']) or len(calls) != len(evidence['per_call'])
            or set(metadata) != set(calls)
            or any(type(row['id']) is not int or row['id'] < 0
                   for row in evidence['metadata_calls'] + evidence['per_call'])):
        raise ValueError('Duplicate/mismatched metadata and physical-call IDs')
    required = expected_counts(work['expected_contexts'])
    counts = [Counter() for _ in range(work['iterations'])]
    totals, physical = defaultdict(list), {}
    for ident, call in calls.items():
        if call['canonical'] is None:
            continue
        meta = metadata[ident]
        key = (call['canonical'], meta.get('context'), meta.get('context_tokens'))
        forward = call['forward']
        if (key not in required or type(forward) is not int or not 0 <= forward < len(counts)
                or type(meta.get('forward')) is not int or meta['forward'] != forward
                or type(meta.get('context_tokens')) is not int or meta.get('completed') is not True
                or meta.get('scope_kind') != 'callable'):
            raise ValueError('Physical invocation differs from expected forward/context')
        counts[forward][key] += 1
        physical[ident] = key
    if any(count != required for count in counts):
        raise ValueError('Every forward must contain exactly its expected physical contexts')
    activities, names, identities = defaultdict(lambda: defaultdict(list)), defaultdict(Counter), set()
    for activity in evidence['activity_ledger']:
        ident = activity['call']
        identity = activity['identity']
        if (type(identity) is not list or len(identity) != 6
                or type(identity[0]) is not str or any(type(x) is not int for x in identity[1:])
                or identity[3] <= identity[2] or identity[4] < 0
                or type(activity['duration_ns']) is not int
                or activity['duration_ns'] != identity[3] - identity[2]
                or tuple(identity) in identities or ident not in physical
                or activity['forward'] != calls[ident]['forward'] or activity['location'] != 'model'
                or [claim['canonical'] for claim in activity['claims']] != [physical[ident][0]]):
            raise ValueError('Invalid/duplicate/unowned native CUDA activity')
        identities.add(tuple(identity))
        activities[ident][identity[4]].append((identity[2], identity[3]))
        names[ident][identity[0]] += 1
    for ident, key in physical.items():
        native_us = sum(union_ns(spans) for spans in activities[ident].values()) / 1000
        if (not native_us or native_us != calls[ident]['union_us']
                or dict(names[ident]) != calls[ident]['descendant_counts']):
            raise ValueError('Physical-call union/counts differ from the native ledger')
        totals[key].append(native_us)
    rows = response['rows']
    seen = set()
    if type(rows) is not list:
        raise ValueError('Expected plain timing rows')
    for row in rows:
        key = (row['layer'], row['context'], row['context_tokens'])
        value = row['microseconds']
        if (key not in required or key in seen or type(value) not in (int, float)
                or not math.isfinite(value) or value <= 0
                or any(row.get(field) != expected for field, expected in stamp.items())
                or not math.isclose(value, sum(totals[key]) / len(totals[key]), rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError('Timing rows differ from same-acquisition native CUDA unions')
        seen.add(key)
    if seen != set(required):
        raise ValueError('Missing contextual timing rows')


def _plan(workloads, catalog, contexts, context_measurements=None):
    if type(workloads) is not list or not workloads:
        raise ValueError('Explicit bounded workloads are required')
    works = [normalize(work, catalog, contexts)[0] for work in workloads]
    if len({digest(work['scope']) for work in works}) != 1:
        raise ValueError('Callable shots must share one independent execution scope')
    manifest = normalize_manifest(dict(schema=1, shots=[dict(tokens=work['shot_tokens'],
        ordinary_layers=[], contexts=work['expected_contexts']) for work in works]))
    works = {work['shot_tokens']: work for work in works}
    if context_measurements is None:
        stamps = {tokens: measurement(work, catalog, contexts) for tokens, work in works.items()}
    else:
        if (type(context_measurements) is not dict or set(context_measurements) != set(works)
                or any(type(key) is not int for key in context_measurements)):
            raise ValueError('Every exact shot needs an independently admitted acquisition identity')
        stamps = {tokens: _identity(value) for tokens, value in context_measurements.items()}
    # No ordinary rows are acquired here; this stamp identifies the empty side
    # of the existing two-table transaction, not an engine measurement.
    empty_stamp = dict(measurement_protocol='isolated-callable-empty-ordinary',
                       measurement_sha256=digest({str(k): v for k, v in sorted(stamps.items())}))
    contract = dict(schema=1, workloads=[works[k] for k in sorted(works)], catalog=catalog,
                    scope_contexts=contexts, stamps={str(k): v for k, v in sorted(stamps.items())})
    return works, manifest, stamps, empty_stamp, contract


def acquire(llm, directory, workloads, catalog, scope_contexts, *, replace=False):
    """Named default-Msgpack RPC, exact no-fire resume, existing table writer.

    The caller owns runtime loading, resource guards and explicit GPU authority.
    This API does not boot an engine, import target modules or select points.
    """
    directory = Path(directory)
    works, manifest, stamps, empty, contract = _plan(workloads, catalog, scope_contexts)
    saved = read_callable_record(directory)
    if saved is not None and not replace:
        if _plan_semantics(saved['plan']) != _plan_semantics(contract):
            raise ValueError('Isolated workload acquisition contract changed')
    table = ContextDenseTable(directory, manifest, empty, stamps)
    complete = table.preload(replace=replace)
    if saved is None and any(directory.glob('*.csv')) and not replace:
        raise ValueError('Cannot retrofit a workload contract onto existing timings')
    retained_shots = {}
    # Validate retained shots before firing anything. If a historical partial
    # acquisition gains its first root record, carry all validated completions
    # forward together; never strand earlier shots behind the root-first reader.
    for tokens, work in sorted(works.items()):
        if (tokens,) in complete:
            response = _saved_response(directory, saved, tokens)
            _verify_saved(table, tokens, work, response, stamps[tokens])
            retained_shots[str(tokens)] = (response if response.get('kind') == 'callable-completion'
                else _compact_receipt(table, tokens, work, response, stamps[tokens]))
    from .categories import ContextDensePoint
    from .writer import DedupSink
    sink = DedupSink(directory / 'dense.csv', ['layer', 'tokens'], empty)
    if replace:
        # Invalidate completion before replacing any rows. Old evidence files
        # are harmless: only the current manifest and complete receipts count.
        update_namespace(directory, 'callables', lambda old: dict(plan=contract, shots={}))
        table.flush(sink)
    fired = False
    for tokens, work in sorted(works.items()):
        if (tokens,) in complete:
            continue
        raw = llm.collective_rpc('fire_callable', args=(work, catalog, scope_contexts))
        if type(raw) not in (list, tuple) or len(raw) != 1:
            raise ValueError('Isolated workload requires exactly one profiling worker')
        response = raw[0]
        validate_response(work, response, stamps[tokens])
        table.validate_rows(tokens, response['rows'])
        points = [ContextDensePoint(row['layer'], tokens, row['context'], row['context_tokens'],
                                   row['microseconds']) for row in response['rows']]
        table.accept_shot(tokens, points, sink)
        receipt = _compact_receipt(table, tokens, work, response, stamps[tokens])
        retained_shots[str(tokens)] = receipt
        # Compact evidence precedes CSV completion. A crash cannot publish a
        # complete shot until the table transaction verifies its exact digest.
        def publish(old):
            old = dict(old or {})
            old.update(plan=contract, shots=dict(retained_shots))
            return old
        update_namespace(directory, 'callables', publish)
        table.flush(sink)
        fired = True
    return fired


def _plan_semantics(plan):
    from .measurement import semantic_identity
    value = semantic_identity(plan)
    value['workloads'] = [workload_semantics(work) for work in plan['workloads']]
    return value


def read_callable_record(directory):
    """Read root metadata first; old JSON plans remain read-only fallbacks."""
    record = read_namespace(directory, 'callables')
    if record is not None:
        if not {'plan', 'shots'} <= set(record) or set(record) - {'plan', 'shots', 'options'}:
            raise ValueError('Invalid callable metadata')
        if type(record['plan']) is not dict or type(record['shots']) is not dict:
            raise ValueError('Invalid callable plan/receipts')
        return record
    path = Path(directory) / 'callable_workloads.json'
    return dict(plan=read_contract(path), shots=None) if path.exists() else None


def read_callable_options(directory):
    record = read_callable_record(Path(directory) / 'callables')
    if record is not None and record['shots'] is not None:
        if type(record.get('options')) is not dict:
            raise ValueError('Missing callable acquisition options')
        return record['options']
    return read_contract(Path(directory) / 'callable_categories.json')


def _saved_response(directory, record, tokens):
    if record is None:
        raise ValueError('Missing callable acquisition plan')
    if record['shots'] is None:
        return _read_response(Path(directory) / ('callable_evidence_' + str(tokens) + '.json'))
    response = record['shots'].get(str(tokens))
    if type(response) is not dict:
        raise ValueError('Missing compact callable receipt')
    return response


def _compact_receipt(table, tokens, work, response, stamp):
    routes = {canonical(dict(canonical=call['canonical'], descendant_counts=call['descendant_counts']))
              for call in response['evidence']['per_call'] if call['canonical'] is not None}
    return dict(schema=1, kind='callable-completion', workload_sha256=digest(workload_semantics(work)),
        scope_sha256=digest(workload_semantics(dict(scope=work['scope']))['scope']),
        measurement=stamp, csv_shot_sha256=table._shot_hash(tokens),
        contexts=[dict(layer=key[0], context=key[1], context_tokens=key[2],
                       invocations=count * work['iterations'])
                  for key, count in sorted(expected_counts(work['expected_contexts']).items())],
        device_capability=response['evidence']['device_capability'],
        routes=[json.loads(route) for route in sorted(routes)])


def _verify_compact(table, tokens, work, response, stamp):
    if (set(response) != {'schema', 'kind', 'workload_sha256', 'scope_sha256', 'measurement',
                         'csv_shot_sha256', 'contexts', 'device_capability', 'routes'}
            or type(response['schema']) is not int or response['schema'] != 1
            or response['kind'] != 'callable-completion'
            or response['workload_sha256'] != digest(workload_semantics(work))
            or response['scope_sha256'] != digest(workload_semantics(dict(scope=work['scope']))['scope'])
            or response['measurement'] != stamp or response['csv_shot_sha256'] != table._shot_hash(tokens)):
        raise ValueError('Compact callable recipe, identity or CSV digest differs')
    required = {key: count * work['iterations'] for key, count in expected_counts(work['expected_contexts']).items()}
    if expected_counts(response['contexts']) != required:
        raise ValueError('Compact callable physical invocation counts differ')
    capability = response['device_capability']
    if type(capability) is not list or len(capability) != 2 or any(type(x) is not int or x < 0 for x in capability):
        raise ValueError('Invalid actual device capability')
    routes = response['routes']
    if (type(routes) is not list or not routes or any(type(route) is not dict or
            set(route) != {'canonical', 'descendant_counts'} or type(route['canonical']) is not str or
            type(route['descendant_counts']) is not dict or not route['descendant_counts'] or
            any(type(name) is not str or not name or type(count) is not int or count <= 0
                for name, count in route['descendant_counts'].items()) for route in routes)
            or {route['canonical'] for route in routes} != {key[0] for key in required}
            or len({canonical(route) for route in routes}) != len(routes)):
        raise ValueError('Invalid compact native kernel route summaries')


def _read_response(path):
    text = path.read_text()
    response = json.loads(text)
    if canonical(response) + '\n' != text:
        raise ValueError('Evidence must be canonical JSON without duplicate keys')
    return response


def _verify_saved(table, tokens, work, response, stamp):
    if response.get('kind') == 'callable-completion':
        _verify_compact(table, tokens, work, response, stamp)
        return
    validate_response(work, response, stamp)
    table.validate_rows(tokens, response['rows'])
    for row in response['rows']:
        key = (tokens, row['layer'], row['context'], row['context_tokens'])
        if table.rows[key] != float(format(row['microseconds'], '.6g')):
            raise ValueError('Saved timing differs from same-acquisition evidence')


@dataclass(frozen=True)
class ExactQuery:
    scope_sha256: str
    shot_tokens: int
    layer: str
    context: str
    context_tokens: int

    @property
    def copy_key(self):
        return (self.scope_sha256, self.shot_tokens, self.layer, self.context, self.context_tokens)


@dataclass(frozen=True, slots=True, init=False)
class CallableLookup:
    """Scope-bound immutable exact lookup; does not provide tensor sizes."""
    _lookup: FrozenContextLookup
    scope_sha256: str
    _receipts: tuple

    def __init__(self, directory, workloads, catalog, scope_contexts, *, context_measurements):
        # Serving supplies independently admitted bundle identities, as with
        # FrozenContextLookup. Never import Torch/vLLM or infer stamps from rows.
        if context_measurements is None:
            raise ValueError('Exact loading requires independently admitted acquisition identities')
        works, manifest, stamps, empty, contract = _plan(workloads, catalog, scope_contexts, context_measurements)
        directory = Path(directory)
        saved = read_callable_record(directory)
        if saved is None or _plan_semantics(saved['plan']) != _plan_semantics(contract):
            raise ValueError('Loaded workload scope/identity differs')
        table = ContextDenseTable(directory, manifest, empty, stamps)
        if table.preload() != {(tokens,) for tokens in works}:
            raise ValueError('Incomplete isolated callable acquisition')
        receipts = []
        for tokens, work in works.items():
            response = _saved_response(directory, saved, tokens)
            _verify_saved(table, tokens, work, response, stamps[tokens])
            receipts.append((tokens, canonical(response)))
        object.__setattr__(self, '_receipts', tuple(receipts))
        object.__setattr__(self, '_lookup', FrozenContextLookup.load(directory, manifest, empty, stamps))
        object.__setattr__(self, 'scope_sha256', digest(next(iter(works.values()))['scope']))

    def receipt(self, tokens):
        """Return an isolated copy of validated route/device metadata."""
        return json.loads(dict(self._receipts)[tokens])

    def exact(self, query):
        if not isinstance(query, ExactQuery) or query.scope_sha256 != self.scope_sha256:
            raise ValueError('Query execution scope differs from measured callable scope')
        return self._lookup.exact(shot_tokens=query.shot_tokens, layer=query.layer,
                                  context=query.context, context_tokens=query.context_tokens)

    def interpolate(self, *, scope_sha256, tokens, layer, context):
        if scope_sha256 != self.scope_sha256:
            raise ValueError('Query execution scope differs from measured callable scope')
        return self._lookup.interpolate(tokens=tokens, layer=layer, context=context)
