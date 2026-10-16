"""Runtime ownership for common direct model methods.

Synthetic method scopes are context
for ownership/matching, not additional execution units. A transparent-count
overlay preserves existing parent/invocation units after adding that context.
No model-family branch, static alias selection, or GPU kernel is used.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from functools import wraps
import inspect
import math
import threading

from profiler.core.hooks import activity_collector as base


class RuntimeMethodScopes(base.Scopes):
    def __init__(self, roots, *, methods=('embed_input_ids', 'compute_logits'),
                 include_nested=True, **kwargs):
        super().__init__(roots, **kwargs)
        if type(include_nested) is not bool:
            raise TypeError('include_nested must be a boolean')
        methods = tuple(methods)
        if any(not isinstance(name, str) or not name for name in methods):
            raise TypeError('Method interfaces must be explicit nonempty strings')
        self._method_names = tuple(dict.fromkeys(methods))
        self._method_modules = self._modules if include_nested else {
            id(module): module for module in self.roots.values()}
        self._saved_methods = []

    @contextmanager
    def method_scope(self, module, method):
        stack = self._stack()
        forward = getattr(self._local, 'forward', None)
        # Calling a method from this root's normal forward already has the
        # right runtime ancestor. Never add a second root or denominator.
        if forward is None or any(module_id == id(module) for _, module_id, _ in stack):
            yield
            return
        from torch.profiler import record_function
        summary = self._summaries[id(module)]
        if summary.endswith(')'):
            summary = summary[:-1] + ', method=' + method + ')'
        else:
            summary += '(method=' + method + ')'
        with self._lock:
            call_id = len(self._calls)
            self._calls.append(dict(id=call_id, parent=stack[-1][0] if stack else None,
                forward=forward, module_id=id(module), class_name=type(module).__name__,
                summary_name=summary, thread_id=threading.get_native_id(),
                paths=list(self._paths[id(module)]), completed=False,
                scope_kind='method', method_name=method, normalization_context_only=True))
        scope = record_function(base.CALL_PREFIX + str(call_id))
        scope.__enter__()
        stack.append((call_id, id(module), scope))
        try:
            yield
        finally:
            if not stack or stack[-1][0] != call_id:
                raise ValueError('Unbalanced runtime method scope')
            stack.pop()
            scope.__exit__(None, None, None)
            self._calls[call_id]['completed'] = True

    def __enter__(self):
        super().__enter__()
        visited = set()
        try:
            for module in self._method_modules.values():
                if id(module) in visited:
                    continue
                visited.add(id(module))
                for name in self._method_names:
                    original = getattr(module, name, None)
                    if not callable(original):
                        continue
                    if inspect.iscoroutinefunction(original) or inspect.isgeneratorfunction(original):
                        raise TypeError('Runtime scopes require synchronous methods')
                    present = name in module.__dict__
                    saved = module.__dict__.get(name)

                    def wrap(owner, method, function):
                        @wraps(function)
                        def invoke(*args, **kwargs):
                            with self.method_scope(owner, method):
                                return function(*args, **kwargs)
                        return invoke

                    installed = wrap(module, name, original)
                    setattr(module, name, installed)
                    self._saved_methods.append((module, name, present, saved, installed))
        except BaseException:
            self.__exit__(*__import__('sys').exc_info())
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        errors = []
        try:
            for module, name, present, saved, installed in reversed(self._saved_methods):
                if getattr(module, name, None) is not installed:
                    errors.append('Common method replaced during capture; foreign value preserved')
                    continue
                try:
                    if present:
                        setattr(module, name, saved)
                    else:
                        delattr(module, name)
                except BaseException as error:
                    errors.append(str(error))
            self._saved_methods.clear()
        finally:
            super().__exit__(exc_type, exc, tb)
        if errors:
            raise ValueError('; '.join(errors))


def _transparent_values(result, metadata, catalog, iterations, forward=None):
    calls = {value['id']: value for value in metadata['calls']}
    method_ids = {index for index, call in calls.items() if call.get('scope_kind') == 'method'}
    context_ids = {index for index, call in calls.items()
                   if call.get('normalization_context_only')}
    if not method_ids <= context_ids:
        raise ValueError('Method scopes must declare transparent normalization')
    cache = {}

    def chain(index):
        if index not in cache:
            parent = calls[index]['parent']
            cache[index] = (() if parent is None else chain(parent)) + (index,)
        return cache[index]

    virtual_counts = Counter()
    virtual_sources = defaultdict(set)
    invocation_counts = Counter()
    descriptions = {}
    by_call = {call['id']: call for call in result['per_call']}
    for index, call in calls.items():
        if forward is not None and call['forward'] != forward:
            continue
        ancestry = chain(index)
        full_path = tuple(calls[item]['summary_name'] for item in ancestry)
        virtual_path = tuple(calls[item]['summary_name'] for item in ancestry if item not in context_ids)
        canonical = by_call[index]['canonical']
        if index in method_ids:
            if canonical is not None:
                raise ValueError('Catalog directly binds a synthetic method context; an explicit method operation contract is required')
            continue
        if index in context_ids:
            if call.get('scope_kind') not in ('operator', 'callable'):
                raise ValueError('Unknown transparent context kind')
            # An operator/callable is measurable, but not a new parent
            # execution unit for its descendants. Its own identity remains.
            virtual_path += (call['summary_name'],)
            if canonical is not None:
                invocation_counts[(canonical, virtual_path)] += 1
            descriptions[full_path] = (call['scope_kind'], virtual_path)
            continue
        virtual_counts[virtual_path] += 1
        virtual_sources[virtual_path].add(full_path)
        if canonical is not None:
            invocation_counts[(canonical, virtual_path)] += 1
        descriptions[full_path] = ('module', virtual_path)
    raw_invocations = Counter()
    for activity in result['activity_ledger']:
        if activity['call'] is None or (forward is not None and activity['forward'] != forward):
            continue
        owner = activity['call']
        ancestry = chain(owner)
        full_parent = tuple(calls[item]['summary_name'] for item in ancestry)
        virtual_parent = tuple(calls[item]['summary_name'] for item in ancestry if item not in context_ids)
        raw_name = activity['identity'][0]
        actual_ancestors = [calls[item]['class_name'] for item in ancestry]
        # The production matcher uses the compatibility label before the
        # first '(' for raw CUDA leaves as well as module representations.
        # Keep the full name in grouping/identity, but not in the lookup.
        canonical = base.match_catalog(base._strip_class_name(raw_name), actual_ancestors, catalog)
        full_path = full_parent + (raw_name,)
        virtual_path = virtual_parent + (raw_name,)
        descriptions[full_path] = ('cuda', virtual_path)
        if canonical is not None:
            raw_invocations[(canonical, virtual_path)] += 1

    source = result if forward is None else result['per_forward'][forward]
    totals = defaultdict(float)
    components = []
    for original in source['components']:
        canonical = original['canonical']
        path = tuple(original['summary_path'])
        if path not in descriptions:
            # A matched synthetic context was rejected above; unmatched
            # method containers never produce canonical components.
            raise ValueError('Unknown normalization component')
        kind, virtual_path = descriptions[path]
        mode = catalog[canonical].get('normalization', 'parent')
        if mode == 'parent':
            # An unqualified existing binding must keep its old summary-group
            # denominator after context-only methods are removed. A newly
            # qualified binding can split that old group across target/draft
            # ownership, however. Its intended per-context execution unit is
            # not implied by the existing catalog. Do not silently divide one
            # owner's total by another owner's parent invocations.
            spec = catalog[canonical]
            if (len(virtual_path) > 1
                    and (spec.get('within') or spec.get('not_within'))
                    and len(virtual_sources[virtual_path[:-1]]) > 1):
                raise ValueError('Qualified method contexts merge a parent denominator; explicit normalization contract required')
            parent_calls = virtual_counts[virtual_path[:-1]] if len(virtual_path) > 1 else iterations
            denominator = parent_calls * (catalog[canonical].get('occurrences') or 1)
        elif mode == 'invocations':
            parent_calls = None
            counter = raw_invocations if kind == 'cuda' else invocation_counts
            denominator = counter[(canonical, virtual_path)]
        else:
            raise ValueError('Unknown timing normalization')
        if denominator <= 0:
            raise ValueError('Missing transparent normalization denominator')
        value = original['total_us'] / denominator
        if value <= 0 or not math.isfinite(value):
            raise ValueError('Invalid runtime-method timing')
        totals[canonical] += value
        components.append(dict(original, denominator=denominator, parent_calls=parent_calls,
            normalized_us=value, normalization_path=virtual_path,
            method_contexts_transparent=True))
    return dict(layer_us=dict(totals), components=components)


def extract_with_methods(cpu, gpu, catalog, metadata, iterations, **kwargs):
    """Extract with the base collector and then normalize method contexts out.

    The ordinary summary tree remains an ownership/matching audit tree and
    MUST NOT be passed to unmodified extract_samples: synthetic method counts
    are not new execution units. ``components`` records the actual denominator.
    """
    if isinstance(metadata, list):
        metadata = dict(calls=metadata, auxiliary=[])
    result = base.extract(cpu, gpu, catalog, metadata, iterations, **kwargs)
    corrected = _transparent_values(result, metadata, catalog, iterations)
    per_forward = [_transparent_values(result, metadata, catalog, 1, forward=i)
                   for i in range(iterations)]
    result['uncorrected_method_parent_layer_us'] = result['layer_us']
    result['layer_us'], result['components'] = corrected['layer_us'], corrected['components']
    result['per_forward_layers_us'] = [value['layer_us'] for value in per_forward]
    for original, new in zip(result['per_forward'], per_forward):
        original.update(new)
    result['normalization_contract'] = 'method contexts retained for matching, removed from invocation identity'
    result['limitations'].append('Context-only parents participate in ownership; use returned normalized values/components, not legacy summary-tree replay.')
    return result
