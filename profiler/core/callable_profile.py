"""Exact isolated-callable acquisition/resume/query seam, separate from engines.

This narrow API has no CLI/grid expansion, model planner, size formulas, or
whole-model trace support. Target scopes and manifests are supplied before fire.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path

from .context_tables import ContextDenseTable, _atomic, _identity, normalize_manifest
from .context_lookup import FrozenContextLookup
from .hooks.callable_workload import canonical, digest, measurement, normalize
from .hooks.callable_context import expected_counts
from .hooks.cuda_timing import union_ns


def validate_response(work, response, stamp):
    """Join same-capture metadata, physical calls and CUDA ledger before saving.

    Generic ownership/count/union checks do not certify a model-specific route
    or logical tensor sizes. Such a consumer must additionally audit the saved
    native activity names against its pinned backend/source contract.
    """
    if (type(response) is not dict or set(response) !=
            {'schema', 'workload_sha256', 'scope_sha256', 'source', 'rows', 'evidence'}
            or type(response['schema']) is not int or response['schema'] != 1
            or response['workload_sha256'] != digest(work)
            or response['scope_sha256'] != digest(work['scope']) or response['source'] != work['source']):
        raise ValueError('Worker workload/source/scope identity differs')
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
    empty_stamp = dict(measurement_protocol='isolated-callable-empty-ordinary-v1',
                       measurement_sha256=digest({str(k): v for k, v in sorted(stamps.items())}))
    contract = dict(schema=1, workloads=[works[k] for k in sorted(works)], catalog=catalog,
                    scope_contexts=contexts, stamps={str(k): v for k, v in sorted(stamps.items())})
    return works, manifest, stamps, empty_stamp, contract


def acquire(llm, directory, workloads, catalog, scope_contexts, *, replace=False):
    """Named default-Msgpack RPC, exact no-fire resume, existing table writer.

    The caller owns runtime loading, resource guards and explicit GPU authority.
    This API does not boot an engine, import target modules or select points.
    """
    works, manifest, stamps, empty, contract = _plan(workloads, catalog, scope_contexts)
    directory = Path(directory)
    table = ContextDenseTable(directory, manifest, empty, stamps)
    complete = table.preload(replace=replace)
    plan_path = directory / 'callable_workloads.json'
    if plan_path.exists() and not replace:
        if plan_path.read_text() != canonical(contract) + '\n':
            raise ValueError('Isolated workload acquisition contract changed')
    elif any(directory.glob('*.csv')) and not replace:
        raise ValueError('Cannot retrofit a workload contract onto existing timings')
    from .categories import ContextDensePoint
    from .writer import DedupSink
    sink = DedupSink(directory / 'dense.csv', ['layer', 'tokens'], empty)
    if replace:
        # Invalidate completion before replacing any rows. Old evidence files
        # are harmless: only the current manifest and complete receipts count.
        _atomic(plan_path, canonical(contract) + '\n')
        table.flush(sink)
    fired = False
    for tokens, work in sorted(works.items()):
        evidence_path = directory / ('callable_evidence_' + str(tokens) + '.json')
        if (tokens,) in complete:
            response = _read_response(evidence_path)
            _verify_saved(table, tokens, work, response, stamps[tokens])
            continue
        raw = llm.collective_rpc('fire_callable', args=(work, catalog, scope_contexts))
        if type(raw) not in (list, tuple) or len(raw) != 1:
            raise ValueError('Isolated workload requires exactly one profiling worker')
        response = raw[0]
        validate_response(work, response, stamps[tokens])
        table.validate_rows(tokens, response['rows'])
        # Evidence precedes CSV/receipt. A crash here leaves the shot incomplete;
        # never adopt old rows just because a standalone diagnostic exists.
        _atomic(plan_path, canonical(contract) + '\n')
        _atomic(evidence_path, canonical(response) + '\n')
        points = [ContextDensePoint(row['layer'], tokens, row['context'], row['context_tokens'],
                                   row['microseconds']) for row in response['rows']]
        table.accept_shot(tokens, points, sink)
        table.flush(sink)
        fired = True
    return fired


def _read_response(path):
    text = path.read_text()
    response = json.loads(text)
    if canonical(response) + '\n' != text:
        raise ValueError('Evidence must be canonical JSON without duplicate keys')
    return response


def _verify_saved(table, tokens, work, response, stamp):
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

    def __init__(self, directory, workloads, catalog, scope_contexts, *, context_measurements):
        # Serving supplies independently admitted bundle identities, as with
        # FrozenContextLookup. Never import Torch/vLLM or infer stamps from rows.
        if context_measurements is None:
            raise ValueError('Exact loading requires independently admitted acquisition identities')
        works, manifest, stamps, empty, contract = _plan(workloads, catalog, scope_contexts,
                                                       context_measurements)
        directory = Path(directory)
        if (directory / 'callable_workloads.json').read_text() != canonical(contract) + '\n':
            raise ValueError('Loaded workload scope/identity differs')
        table = ContextDenseTable(directory, manifest, empty, stamps)
        if table.preload() != {(tokens,) for tokens in works}:
            raise ValueError('Incomplete isolated callable acquisition')
        for tokens, work in works.items():
            _verify_saved(table, tokens, work, _read_response(directory / ('callable_evidence_' + str(tokens) + '.json')),
                          stamps[tokens])
        object.__setattr__(self, '_lookup', FrozenContextLookup.load(directory, manifest, empty, stamps))
        object.__setattr__(self, 'scope_sha256', digest(next(iter(works.values()))['scope']))

    def exact(self, query):
        if not isinstance(query, ExactQuery) or query.scope_sha256 != self.scope_sha256:
            raise ValueError('Query execution scope differs from measured callable scope')
        return self._lookup.exact(shot_tokens=query.shot_tokens, layer=query.layer,
                                  context=query.context, context_tokens=query.context_tokens)

    def interpolate(self, *, scope_sha256, tokens, layer, context):
        if scope_sha256 != self.scope_sha256:
            raise ValueError('Query execution scope differs from measured callable scope')
        return self._lookup.interpolate(tokens=tokens, layer=layer, context=context)
