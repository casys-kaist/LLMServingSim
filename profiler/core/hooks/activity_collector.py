"""Eager native CUDA-activity collection with explicit dynamic module scopes.

No vLLM or Torch dependency is imported until capture is requested. Catalog
matching and normalization are shared with the existing timing reader. CPU
scope intervals establish ownership, never latency. Per-call CUDA unions
retain kernels, copies and memsets. Missing measurements are never filled by
zero. Unknown or overlapping work is reported explicitly; callers may require
strict full-catalog coverage, while normal category slices remain supported.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from bisect import bisect_right
import math
import threading
import time

from profiler.core.hooks.cuda_timing import active_ns as _active_ns
from profiler.core.hooks.cuda_timing import activity_identity as _identity
from profiler.core.hooks.timings import _match_slice, _strip_class_name, extract_samples

CALL_PREFIX = 'activity_call_'
FORWARD_PREFIX = 'activity_forward_'
AUX_PREFIX = 'activity_aux_'


def _contained(outer, inner):
    return (outer.device_resource_id() == inner.device_resource_id()
            and outer.start_ns() <= inner.start_ns()
            and inner.end_ns() <= outer.end_ns())


class _ScopeIndex:
    """Validated interval forest keyed by OS thread, queried in O(depth log N).

    Explicit parent ids distinguish true nesting from unrelated overlapping
    scopes. Sibling intervals may touch but must not overlap. No GPU interval
    participates in this index: queries are correlated native CPU launches.
    """
    def __init__(self, events, parents=None):
        self.events = events
        self.groups = defaultdict(list)
        self.starts = {}
        parents = parents or {}
        for index, event in events.items():
            if (type(event.start_ns()) is not int or type(event.end_ns()) is not int
                    or event.end_ns() <= event.start_ns()):
                raise ValueError('Invalid CPU scope interval')
            parent = parents.get(index)
            if parent is not None and (parent not in events or not _contained(events[parent], event)):
                raise ValueError('Invalid CPU scope parent interval')
            self.groups[(event.device_resource_id(), parent)].append(index)
        for key, indices in self.groups.items():
            indices.sort(key=lambda i: (events[i].start_ns(), events[i].end_ns()))
            for left, right in zip(indices, indices[1:]):
                if events[left].end_ns() > events[right].start_ns():
                    raise ValueError('Overlapping unrelated CPU scopes')
            self.starts[key] = [events[i].start_ns() for i in indices]

    def find(self, launch):
        if launch.end_ns() <= launch.start_ns():
            raise ValueError('A CUDA launch needs a positive CPU interval for unique ownership')
        parent = None
        thread = launch.device_resource_id()
        while True:
            key = (thread, parent)
            starts = self.starts.get(key, ())
            position = bisect_right(starts, launch.start_ns()) - 1
            if position < 0:
                return parent
            candidate = self.groups[key][position]
            if not _contained(self.events[candidate], launch):
                return parent
            parent = candidate


def match_catalog(name, ancestors, catalog):
    """Use the catalog reader's exact existing specificity/exclusion rules."""
    return _match_slice(_strip_class_name(name), ancestors, catalog)


def _summary_name(module):
    """vLLM event_module_repr parameter-metadata identity, without PyCall tree."""
    import torch
    parameters = []
    # Torch 2.13 cf30153c profiler_python.cpp:359-378 iterates _parameters
    # directly and retains only THPVariable_CheckExact entries. Its definition
    # (python_variable.h:40-50) accepts exact Tensor or exact Parameter, NOT
    # subclasses. Thus vLLM custom Parameter subclasses are intentionally absent
    # from the LEGACY SUMMARY IDENTITY (not excluded from CUDA measurement).
    # Dict iteration retains distinct alias names and registration order.
    for name, value in module._parameters.items():
        if not isinstance(name, str) or type(value) not in (torch.Tensor, torch.nn.Parameter):
            continue
        dtype = str(value.dtype).replace('torch.', '')
        sizes = ', '.join(str(dimension) for dimension in value.size())
        parameters.append(f'{name}={dtype}[{sizes}]')
    cls = type(module).__name__
    return cls + '(' + ', '.join(parameters) + ')' if parameters else cls


class ScopedModules:
    """Install model-independent pre/post hooks and expose explicit forward scopes.

    Roots are caller-supplied existing nn.Modules, never family-selected here.
    Sampler wrappers or a method called directly instead of through Module.__call__
    must be adapted by the caller; unobserved GPU work is otherwise refused.
    """
    def __init__(self, roots, forward_prefix=FORWARD_PREFIX):
        if not roots:
            raise ValueError('At least one module root is required')
        self.roots = dict(roots)
        self.forward_prefix = forward_prefix
        self._modules = {}
        self._paths = defaultdict(list)
        self._summaries = {}
        self._calls = []
        self._aux = []
        self._handles = []
        self._local = threading.local()
        self._lock = threading.Lock()
        self._forward_ids = set()

        def visit(module, path, active):
            key = id(module)
            if key in active:
                raise ValueError('Cycle in registered module graph')
            if not hasattr(module, 'register_forward_pre_hook'):
                raise TypeError('Root/child is not an nn.Module')
            self._modules[key] = module
            self._paths[key].append(path)
            self._summaries[key] = _summary_name(module)
            for name, child in module._modules.items():
                if child is not None:
                    visit(child, path + '.' + name, active | {key})
        for path, module in self.roots.items():
            visit(module, path, set())

    def _stack(self):
        if not hasattr(self._local, 'stack'):
            self._local.stack = []
        return self._local.stack

    def _before(self, module, args):
        forward = getattr(self._local, 'forward', None)
        if forward is None:
            return
        from torch.profiler import record_function
        stack = self._stack()
        with self._lock:
            call_id = len(self._calls)
            record = dict(id=call_id, parent=stack[-1][0] if stack else None,
                forward=forward, module_id=id(module), class_name=type(module).__name__,
                summary_name=self._summaries[id(module)], thread_id=threading.get_native_id(),
                paths=list(self._paths[id(module)]), completed=False)
            self._calls.append(record)
        scope = record_function(CALL_PREFIX + str(call_id))
        scope.__enter__()
        stack.append((call_id, id(module), scope))

    def _after(self, module, args, result):
        if getattr(self._local, 'forward', None) is None:
            return
        stack = self._stack()
        if not stack or stack[-1][1] != id(module):
            raise ValueError('Module post-hook does not match dynamic call stack')
        call_id, _, scope = stack.pop()
        scope.__exit__(None, None, None)
        self._calls[call_id]['completed'] = True

    def __enter__(self):
        if self._handles:
            raise ValueError('ScopedModules is already installed')
        try:
            for module in self._modules.values():
                self._handles.append(module.register_forward_pre_hook(self._before))
                self._handles.append(module.register_forward_hook(self._after, always_call=True))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if exc_type is None and any(not value['completed'] for value in self._calls):
            raise ValueError('Unclosed module invocation')

    @contextmanager
    def forward(self, index):
        if not self._handles or self._stack() or getattr(self._local, 'forward', None) is not None:
            raise ValueError('Invalid/nested forward boundary')
        if type(index) is not int or index < 0 or index in self._forward_ids:
            raise ValueError('Duplicate/invalid forward index')
        from torch.profiler import record_function
        self._forward_ids.add(index)
        self._local.forward = index
        try:
            with record_function(self.forward_prefix + str(index)):
                yield
        finally:
            self._local.forward = None
            if self._stack():
                raise ValueError('Forward ended with active module scopes')

    @contextmanager
    def auxiliary(self, label):
        """Caller-declared non-model runner preparation/postprocessing boundary."""
        if self._stack() or getattr(self._local, 'forward', None) is None:
            raise ValueError('Auxiliary scope must be inside forward, outside modules')
        from torch.profiler import record_function
        with self._lock:
            index = len(self._aux)
            self._aux.append(dict(id=index, label=str(label),
                                  forward=self._local.forward, thread_id=threading.get_native_id()))
        with record_function(AUX_PREFIX + str(index)):
            yield

    def metadata(self):
        return dict(calls=[dict(value) for value in self._calls],
                    auxiliary=[dict(value) for value in self._aux],
                    module_count=len(self._modules), forward_prefix=self.forward_prefix)


def extract(cpu, gpu, catalog, metadata, iterations,
            forward_prefix=FORWARD_PREFIX, strict_coverage=False,
            require_complete=False, strict_model_coverage=False):
    """Build exact-call CUDA unions, grouped summary, normalized canonical floats.

    metadata may be ScopedModules.metadata() or its ``calls`` list. Repeated
    invocations and heterogeneous layer counts are not constrained. Summary
    groups use full runtime parameter-signature paths, NOT instance identity.
    Parent normalization is the parent GROUP's call count times occurrences;
    invocation normalization uses the matched group's own invocation count.
    """
    if type(iterations) is not int or iterations <= 0 or not catalog:
        raise ValueError('Positive iterations and a nonempty catalog are required')
    if isinstance(metadata, list):
        metadata = dict(calls=metadata, auxiliary=[])
    records = metadata['calls']
    calls = {value['id']: dict(value) for value in records}
    if len(calls) != len(records) or not calls:
        raise ValueError('Duplicate/empty call metadata')
    aux_meta = {value['id']: value for value in metadata.get('auxiliary', [])}
    windows, call_events, auxiliary, launches = {}, {}, {}, defaultdict(list)
    for event in cpu:
        name = event.name()
        if 'graphlaunch' in name.lower() or 'graph_launch' in name.lower():
            raise ValueError('Generic eager collector refuses CUDA graph launches')
        found = None
        for prefix, destination in ((forward_prefix, windows), (CALL_PREFIX, call_events), (AUX_PREFIX, auxiliary)):
            if name.startswith(prefix):
                index = int(name[len(prefix):])
                if index in destination or event.end_ns() <= event.start_ns():
                    raise ValueError('Duplicate/invalid CPU annotation scope')
                destination[index] = event
                found = True
                break
        if found is None and event.activity_type() in ('cuda_runtime', 'cuda_driver'):
            launches[event.correlation_id()].append(event)
    if set(windows) != set(range(iterations)) or set(call_events) != set(calls):
        raise ValueError('Missing/extra forward or module scopes')
    if set(auxiliary) != set(aux_meta):
        raise ValueError('Missing/extra explicit auxiliary scopes')
    forward_index = _ScopeIndex(windows)
    chains = {}

    def chain(index, active=()):
        if index in chains:
            return chains[index]
        if index not in calls or index in active:
            raise ValueError('Missing parent or cycle in runtime call metadata')
        call = calls[index]
        parent = call['parent']
        result = (() if parent is None else chain(parent, active + (index,))) + (index,)
        chains[index] = result
        return result

    for index, call in calls.items():
        ancestry = chain(index)
        event = call_events[index]
        if not call.get('completed', True) or call['forward'] not in windows:
            raise ValueError('Incomplete call or invalid forward')
        if not _contained(windows[call['forward']], event):
            raise ValueError('Module call outside declared forward')
        if call['thread_id'] != event.device_resource_id():
            raise ValueError('Metadata/native CPU thread mismatch')
        if call['parent'] is not None:
            parent = calls[call['parent']]
            if parent['forward'] != call['forward'] or not _contained(call_events[call['parent']], event):
                raise ValueError('Dynamic parent/forward scope mismatch')
        call['ancestors'] = [calls[i]['class_name'] for i in ancestry[:-1]]
        call['canonical'] = match_catalog(call['class_name'], call['ancestors'], catalog)
        call['summary_path'] = tuple(calls[i]['summary_name'] for i in ancestry)
        call['direct'], call['descendants'] = [], []
    for index, event in auxiliary.items():
        info = aux_meta[index]
        if info['forward'] not in windows or not _contained(windows[info['forward']], event):
            raise ValueError('Auxiliary scope outside declared forward')
        if info['thread_id'] != event.device_resource_id():
            raise ValueError('Auxiliary metadata/native thread mismatch')

    call_index = _ScopeIndex(call_events, {i: call['parent'] for i, call in calls.items()})
    aux_index = _ScopeIndex(auxiliary)

    identities = set()
    gaps, unscoped, outside_forward, overlaps, aux_counts = [], [], [], [], Counter()
    forward_counts = [Counter() for _ in range(iterations)]
    raw_nodes = []
    activity_ledger = []
    annotations = 0
    for event in gpu:
        if event.is_user_annotation():
            annotations += 1
            continue
        identity = _identity(event)
        if identity in identities:
            raise ValueError('Duplicate native CUDA activity')
        identities.add(identity)
        if event.duration_ns() <= 0 or event.end_ns() - event.start_ns() != event.duration_ns():
            raise ValueError('Invalid native CUDA activity duration')
        owners = set()
        for launch in launches.get(event.correlation_id(), []):
            owners.add((forward_index.find(launch), call_index.find(launch), aux_index.find(launch)))
        if len(owners) != 1:
            raise ValueError('Missing/ambiguous native CUDA launch correlation')
        forward, deepest, aux = owners.pop()
        ledger = dict(identity=identity, duration_ns=event.duration_ns(), forward=forward,
                      call=deepest, location='outside_forward', ancestors=[], claims=[])
        activity_ledger.append(ledger)
        if forward is None:
            if deepest is not None or aux is not None:
                raise ValueError('Module/auxiliary work outside forward')
            outside_forward.append(dict(identity=identity))
            continue
        forward_counts[forward][event.name()] += 1
        if deepest is None:
            if aux is not None:
                ledger['location'] = 'auxiliary:' + aux_meta[aux]['label']
                aux_counts[aux_meta[aux]['label'] + ':' + event.name()] += 1
            else:
                ledger['location'] = 'unscoped'
                unscoped.append(dict(forward=forward, identity=identity))
            continue
        if aux is not None:
            raise ValueError('Module CUDA work overlaps declared non-model auxiliary scope')
        call = calls[deepest]
        if call['forward'] != forward:
            raise ValueError('Module/forward CUDA ownership disagreement')
        call['direct'].append(event)
        for index in chains[deepest]:
            calls[index]['descendants'].append(event)
        ancestors = [calls[index]['class_name'] for index in chains[deepest]]
        ledger['location'], ledger['ancestors'] = 'model', ancestors
        # Shipped extract_samples strips the first '(' for every summary
        # node, including raw CUDA names. Preserve the full name for summary
        # identity and activity auditing, but match its compatibility label.
        raw_label = event.name().split('(', 1)[0]
        raw_canonical = match_catalog(raw_label, ancestors, catalog)
        raw_nodes.append((deepest, event, raw_canonical))
        module_bindings = [calls[index]['canonical'] for index in chains[deepest]
                           if calls[index]['canonical'] is not None]
        claims = module_bindings + ([raw_canonical] if raw_canonical is not None else [])
        ledger['claims'] = [dict(canonical=calls[index]['canonical'], ancestors=calls[index]['ancestors'])
                            for index in chains[deepest] if calls[index]['canonical'] is not None]
        if raw_canonical is not None:
            ledger['claims'].append(dict(canonical=raw_canonical, ancestors=ancestors))
        if len(claims) > 1:
            overlaps.append(dict(forward=forward, call=deepest, claims=claims, identity=identity))
        if not module_bindings and raw_canonical is None:
            gaps.append(dict(forward=forward, call=deepest, ancestors=ancestors, identity=identity))
    if not identities:
        raise ValueError('No native CUDA activity')

    for call in calls.values():
        call['union_ns'] = _active_ns(call['descendants'])
    aggregate = _summarize(calls, raw_nodes, catalog, iterations)
    per_forward = [_summarize(calls, raw_nodes, catalog, 1, forward=i) for i in range(iterations)]
    layer_us = aggregate['layer_us']
    missing = sorted(set(catalog) - set(layer_us))
    zero = sorted(name for name, value in layer_us.items() if value <= 0 or not math.isfinite(value))
    invalid_forward_values = [dict(forward=i, canonical=name) for i, result in enumerate(per_forward)
        for name, value in result['layer_us'].items() if value <= 0 or not math.isfinite(value)]
    audit = dict(unbound_under_module=gaps, unscoped_gpu=unscoped, outside_forward=outside_forward,
        overlapping_canonical_claims=overlaps,
        explicit_auxiliary_counts=dict(aux_counts), excluded_annotations=annotations,
        per_forward_activity_counts=[dict(values) for values in forward_counts],
        unique_cuda_activities=len(identities), missing_canonical=missing, zero_canonical=zero,
        invalid_forward_values=invalid_forward_values)
    if strict_coverage and (gaps or unscoped or outside_forward or overlaps):
        raise CoverageError('Unbound/unscoped GPU work or overlapping canonical claims are not certified', audit)
    if strict_model_coverage and (gaps or overlaps):
        raise CoverageError('Unbound model work or overlapping canonical claims are not certified', audit)
    if zero or invalid_forward_values or (require_complete and missing):
        raise CoverageError('Missing/zero canonical measurement', audit)
    return dict(**aggregate, per_forward=per_forward, activity_ledger=activity_ledger,
        per_forward_layers_us=[value['layer_us'] for value in per_forward], audit=audit,
        complete=not (gaps or unscoped or outside_forward or overlaps or zero or (require_complete and missing)),
        per_call=[dict(id=call['id'], parent=call['parent'], forward=call['forward'],
            class_name=call['class_name'], summary_name=call['summary_name'],
            scope_kind=call.get('scope_kind', 'module'),
            canonical=call['canonical'], ancestors=call['ancestors'], paths=call.get('paths', []),
            direct_counts=dict(Counter(event.name() for event in call['direct'])),
            descendant_counts=dict(Counter(event.name() for event in call['descendants'])),
            union_us=call['union_ns']/1000) for call in calls.values()],
        limitations=['All actual hook-visible ancestors retained; no kernel-less-module pruning.',
            'Dynamic hooks may see modules absent from the legacy PyCall tree; compare visibility explicitly.',
            'Per-device unions apply inside each call; overlapping canonical claims require explicit catalog review.',
            'Unexecuted catalog alternatives are reported missing; require_complete optionally refuses them.',
            'Parameter summary identity reproduces Torch 2.13 cf30153c PyCall rules, not a promise for every Torch version.',
            'Not a timing-equivalence or production-throughput certification.'])


def _summarize(calls, raw_nodes, catalog, iterations, forward=None):
    groups = {}
    for call in calls.values():
        if forward is not None and call['forward'] != forward:
            continue
        key = call['summary_path']
        if key not in groups:
            groups[key] = dict(name=call['summary_name'], kind='module', class_name=call['class_name'],
                canonical=call['canonical'], invocations=0, cuda_ns=0, calls=[])
        group = groups[key]
        if group['kind'] != 'module' or group['canonical'] != call['canonical']:
            raise ValueError('Inconsistent runtime summary group')
        group['invocations'] += 1
        group['cuda_ns'] += call['union_ns']
        group['calls'].append(call['id'])
    for parent, event, canonical in raw_nodes:
        if forward is not None and calls[parent]['forward'] != forward:
            continue
        key = calls[parent]['summary_path'] + (event.name(),)
        if key not in groups:
            groups[key] = dict(name=event.name(), kind='cuda', class_name=event.name(),
                canonical=canonical, invocations=0, cuda_ns=0, calls=[])
        group = groups[key]
        if group['kind'] != 'cuda' or group['canonical'] != canonical:
            raise ValueError('CUDA/module summary name collision')
        group['invocations'] += 1
        group['cuda_ns'] += event.duration_ns()

    layer_us, components = defaultdict(float), []
    tree_nodes = {path: dict(entry=dict(name=group['name'], cuda_time_us=group['cuda_ns']/1000,
                         invocations=group['invocations']), children=[])
                  for path, group in groups.items()}
    roots = []
    for path, group in groups.items():
        if len(path) == 1:
            roots.append(tree_nodes[path])
            parent_calls = iterations
        else:
            if path[:-1] not in groups:
                raise ValueError('Missing summary parent')
            tree_nodes[path[:-1]]['children'].append(tree_nodes[path])
            parent_calls = groups[path[:-1]]['invocations']
        canonical = group['canonical']
        if canonical is None:
            continue
        spec = catalog[canonical]
        occurrences = spec.get('occurrences') or 1
        if type(occurrences) is not int or occurrences <= 0:
            raise ValueError('Invalid occurrence count')
        mode = spec.get('normalization', 'parent')
        if mode == 'parent':
            denominator = parent_calls * occurrences
        elif mode == 'invocations':
            denominator = group['invocations']
        else:
            raise ValueError('Unknown canonical normalization')
        value = group['cuda_ns'] / (1000 * denominator)
        layer_us[canonical] += value
        components.append(dict(canonical=canonical, summary_path=path, kind=group['kind'],
                               normalization=mode, occurrences=occurrences,
                               invocations=group['invocations'], parent_calls=parent_calls,
                               denominator=denominator, total_us=group['cuda_ns']/1000, normalized_us=value))
    # Reuse the shipped normalization/deduplication implementation itself.
    layer_us = {sample.layer: sample.microseconds for sample in extract_samples(roots, catalog, iterations)}
    return dict(layer_us=layer_us, summary_tree=roots, components=components)


class CoverageError(ValueError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def canonical_invocation_counts(result):
    """Audit physical calls separately from the latency normalization units.

    Only the explicitly captured timed forwards are represented. Counts come
    from matched call records or raw CUDA leaves, never latency or a divisor.
    Transparent contexts can change a group's normalization path/denominator
    without changing how many physical calls actually occurred. Group-level
    divisors therefore remain explicit and are not summed into a fake global
    invocation count. A CPU scope with no CUDA activity counts as observed,
    but not as a positive-CUDA call.
    """
    forwards = result['per_forward']
    physical = [defaultdict(Counter) for _ in forwards]
    positive = [defaultdict(Counter) for _ in forwards]
    for call in result['per_call']:
        canonical = call['canonical']
        if canonical is None:
            continue
        index = call['forward']
        if type(index) is not int or not 0 <= index < len(forwards):
            raise ValueError('Canonical call has an invalid timed-forward index')
        kind = call['scope_kind']
        if kind not in ('module', 'operator', 'callable'):
            raise ValueError('Unsupported measurable scope kind in invocation audit')
        physical[index][canonical][kind] += 1
        if call['union_us'] > 0:
            positive[index][canonical][kind] += 1

    requested = set(result['layer_us']) | set(result['audit']['missing_canonical'])
    rows = []
    for index, forward in enumerate(forwards):
        groups = defaultdict(list)
        for component in forward['components']:
            groups[component['canonical']].append(component)
        canonical_counts = {}
        for canonical, components in sorted(groups.items()):
            modes = {(item['normalization'], item['occurrences']) for item in components}
            if len(modes) != 1:
                raise ValueError('Canonical components have inconsistent normalization metadata')
            mode, occurrences = modes.pop()
            calls = dict(physical[index][canonical])
            positive_calls = dict(positive[index][canonical])
            scope_calls = sum(calls.values())
            component_calls = sum(item['invocations'] for item in components if item['kind'] == 'module')
            if scope_calls != component_calls:
                raise ValueError('Canonical call records and summary invocation counts disagree')
            raw = sum(item['invocations'] for item in components if item['kind'] == 'cuda')
            canonical_counts[canonical] = dict(
                scope_calls=scope_calls, scope_calls_by_kind=calls,
                positive_cuda_scope_calls=sum(positive_calls.values()),
                positive_cuda_scope_calls_by_kind=positive_calls,
                raw_cuda_activities=raw, normalization=mode, occurrences=occurrences,
                normalization_groups=[dict(
                    kind=item['kind'], summary_path=list(item['summary_path']),
                    normalization_path=list(item.get('normalization_path', item['summary_path'])),
                    physical_invocations=item['invocations'], parent_calls=item['parent_calls'],
                    denominator=item['denominator']) for item in components])
        if set(physical[index]) - set(canonical_counts):
            raise ValueError('Matched canonical calls are absent from the summary audit')
        rows.append(dict(forward=index, canonical=canonical_counts,
                         missing_canonical=sorted(requested - set(canonical_counts))))
    return dict(schema=1, timed_forwards=len(forwards), per_forward=rows,
        scope='explicit measured-forward scopes only; warmup excluded',
        semantics='physical scope calls and raw CUDA activities are separate; normalization divisors are not call counts')


def coverage_report(result):
    """Legacy-compatible coverage fields with a conservative CUDA-time ledger.

    Coverage sums native activity durations, NOT module interval unions. Every
    activity appears exactly once in bound-or-gap totals, including unscoped
    runner work. The outermost canonical owner gets the accounting allocation;
    overlapping claims remain a separate explicit diagnostic, never double time.
    """
    bound, gaps, identities, seen_at = defaultdict(int), defaultdict(int), set(), defaultdict(set)
    total = 0
    for activity in result['activity_ledger']:
        identity = tuple(activity['identity'])
        if identity in identities:
            raise ValueError('Coverage ledger repeats a native CUDA identity')
        identities.add(identity)
        duration = activity['duration_ns']
        if type(duration) is not int or duration <= 0:
            raise ValueError('Coverage ledger has an invalid native duration')
        total += duration
        if activity['claims']:
            owner = activity['claims'][0]
            bound[owner['canonical']] += duration
            seen_at[owner['canonical']].add(tuple(owner['ancestors']))
        else:
            name = activity['identity'][0]
            location = activity['location']
            ancestors = ' > '.join(activity['ancestors']) or '(' + location + ')'
            gaps[(name, ancestors)] += duration
    if len(identities) != result['audit']['unique_cuda_activities']:
        raise ValueError('Coverage ledger omitted native CUDA activity')
    if sum(bound.values()) + sum(gaps.values()) != total:
        raise ValueError('Coverage ledger does not conserve CUDA duration')
    over_matches = []
    for canonical, chains in sorted(seen_at.items()):
        first = min(chains, key=len)
        if len(chains) > 1 and (not first or not all(chain and chain[0] == first[0] for chain in chains)):
            over_matches.append([canonical, sorted(' > '.join(chain) or '(top level)' for chain in chains)])
    # Keep overlap failures visible through the existing CoverageReport RPC
    # schema and CLI exit status, not only an extra diagnostic JSON field.
    overlap_groups = defaultdict(set)
    for activity in result['audit']['overlapping_canonical_claims']:
        overlap_groups[tuple(activity['claims'])].add(activity['identity'][0])
    for claims, names in sorted(overlap_groups.items()):
        over_matches.append([' + '.join(claims),
                             ['shared CUDA activity: ' + name for name in sorted(names)]])
    report = dict(total_us=total/1000, bound_us_by_layer={name: value/1000 for name, value in bound.items()},
        gaps=[[name, value/1000, ancestors] for (name, ancestors), value in
              sorted(gaps.items(), key=lambda item: -item[1])], over_matches=over_matches,
        overlapping_activities=result['audit']['overlapping_canonical_claims'],
        accounting='sum of every unique native kernel/copy/memset duration exactly once; not wall time',
        unique_cuda_activities=len(identities))
    if 'callable_audit' in result:
        report['callable_audit'] = result['callable_audit']
    report['canonical_invocations'] = (result['context_source_invocations']
        if 'context_source_invocations' in result else canonical_invocation_counts(result))
    if 'context_samples' in result:
        report['context_samples'] = result['context_samples']
        report['per_forward_context_samples'] = result['per_forward_context_samples']
        report['context_admission'] = 'diagnostic only; not a persisted expected-context manifest'
    return report


Scopes = ScopedModules


def collect(roots, iterations, callback, catalog, *, forward_prefix=FORWARD_PREFIX,
            strict_coverage=False, strict_model_coverage=False, require_complete=False,
            return_events=False, scope_contexts=None, expected_contexts=None, context_diagnostic=False):
    """Collect ``callback(index)`` eagerly without adding warmups or padding.

    ``roots`` is a mapping of existing modules, supplied by a runner adapter.
    The caller owns preparation, device selection, inference/route contexts,
    synchronization policy and lifecycle. Model outputs are not retained.
    ``return_events`` is for bounded diagnostics; do not retain raw contexts
    across a large sweep. No Chrome trace or Python-stack tree is generated.
    """
    if type(iterations) is not int or iterations <= 0:
        raise ValueError('iterations must be a positive integer')
    import torch
    from torch.autograd import DeviceType
    from .capture_scope_contract import normalize_contexts
    from .method_scopes import RuntimeMethodScopes
    from .operator_scopes import extract_with_operators
    contexts = normalize_contexts(scope_contexts, catalog)
    if contexts['schema'] == 3 or expected_contexts is not None:
        from .callable_context import contextual_catalog, expected_counts
        selected = contextual_catalog(catalog, contexts)
        if expected_contexts is not None:
            expected = expected_counts(expected_contexts)
            if {key[0] for key in expected} != set(selected):
                raise ValueError('Expected contexts differ from the contextual catalog')
        elif selected and not context_diagnostic:
            raise ValueError('Context acquisition needs its independent expected-context manifest')
    started = time.perf_counter()
    if contexts.get('callables'):
        from .callable_scopes import DeclaredCallableScopes
        scopes = DeclaredCallableScopes(roots, declarations=contexts['callables'],
            methods=contexts['methods'], forward_prefix=forward_prefix)
    else:
        scopes = RuntimeMethodScopes(roots, methods=contexts['methods'], forward_prefix=forward_prefix)
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, with_stack=False, with_modules=False,
        profile_memory=False, with_flops=False)
    configured = time.perf_counter()
    with scopes:
        with profiler:
            entered = time.perf_counter()
            for index in range(iterations):
                with scopes.forward(index):
                    callback(index)
            forwarded = time.perf_counter()
        closed = time.perf_counter()
    cleaned = time.perf_counter()
    native = profiler.profiler.kineto_results.events()
    cpu = [event for event in native if event.device_type() == DeviceType.CPU]
    gpu = [event for event in native if event.device_type() == DeviceType.CUDA]
    metadata = scopes.metadata()
    result = extract_with_operators(cpu, gpu, catalog, metadata, iterations,
        contexts=contexts['operators'], forward_prefix=forward_prefix, strict_coverage=strict_coverage,
        strict_model_coverage=strict_model_coverage, require_complete=require_complete)
    finished = time.perf_counter()
    result['phases_seconds'] = dict(setup=configured-started, enter=entered-configured,
        forwards=forwarded-entered, exit=closed-forwarded, hook_cleanup=cleaned-closed,
        extract=finished-cleaned)
    result['collection_seconds'] = finished-started
    result.setdefault('metadata', metadata)
    result['scope_contexts'] = contexts
    if (contexts['schema'] == 3 and selected) or expected_contexts is not None:
        from .callable_context import split_context_timings
        # Preserve the physical invocation audit before removing pooled rows.
        result['context_source_invocations'] = canonical_invocation_counts(result)
        split_context_timings(result, catalog, contexts, iterations,
                              expected_contexts, diagnostic=context_diagnostic)
    if return_events:
        result['native_cpu_events'], result['native_gpu_events'] = cpu, gpu
    return result


def roots_from_runner(runner, *, sampler_adapter=None):
    """Common runner interface, intentionally separate from capture/extraction.

    The optional sampler adapter may install an nn.Module facade on runners
    that expose a non-module sampler. No model-family classification is used.
    The caller may instead pass its own roots directly to ``collect``.
    """
    import torch
    if sampler_adapter is not None:
        sampler_adapter(runner)
    getter = getattr(runner, 'get_model', None)
    if not callable(getter):
        raise TypeError('Runner must expose get_model or caller-supplied roots')
    roots = {'model': getter()}
    draft_getter = getattr(runner, 'get_draft_model', None)
    if callable(draft_getter):
        draft = draft_getter()
        if draft is not None:
            roots['draft_model'] = draft
    sampler = getattr(runner, 'sampler', None)
    if sampler is not None:
        roots['sampler'] = sampler
    if any(not isinstance(module, torch.nn.Module) for module in roots.values()):
        raise TypeError('Collector roots must be nn.Modules; apply the runner sampler adapter first')
    return roots
