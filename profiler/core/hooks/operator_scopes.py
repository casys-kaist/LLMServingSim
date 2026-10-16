"""Exact native CPU-operator contexts for the common eager collector.

Serialized architecture-catalog declaration::

    {'core': {'operator': 'namespace::fused_core', 'within': 'Attention'}}

The native CPU operator interval establishes launch ownership, never latency.
Every duration still comes from native CUDA kernel/copy/memset events. Operator
contexts participate in ancestry matching but are transparent normalization
parents. Context declarations may be supplied separately from a category slice.
"""
from collections import defaultdict
import copy

from .method_scopes import base, extract_with_methods


class _Annotation:
    def __init__(self, event, name):
        self.event, self.scope_name = event, name

    def name(self):
        return self.scope_name

    def __getattr__(self, name):
        return getattr(self.event, name)


def operator_contexts(catalog):
    """Exact names needed even when the requested category omits their costs."""
    names = set()
    for spec in catalog.values():
        if 'operator' not in spec:
            continue
        name = spec['operator']
        if not isinstance(name, str) or not name or '*' in name or '(' in name:
            raise ValueError('Operator binding requires one exact native CPU name')
        if 'vllm' in spec or 'callable' in spec:
            raise ValueError('One binding cannot claim both operator and module/kernel kinds')
        names.add(name)
    return sorted(names)


def _translate_catalog(catalog):
    operator_contexts(catalog)  # Validate all declarations first.
    result = {}
    for canonical, spec in catalog.items():
        value = dict(spec)
        if len(set(value) & {'operator', 'callable', 'vllm'}) != 1:
            raise ValueError('A binding requires exactly one operator/callable/module kind')
        if 'operator' in value:
            value['vllm'] = value.pop('operator')
        elif 'callable' in value:
            name = value.pop('callable')
            if not isinstance(name, str) or not name or '*' in name or '(' in name:
                raise ValueError('Callable binding requires one exact semantic name')
            value['vllm'] = name
        result[canonical] = value
    return result


def add_operator_contexts(cpu, gpu, metadata, names, *, forward_prefix=base.FORWARD_PREFIX):
    """Insert observed operator intervals into the existing dynamic call tree.

    Exact ties/crossing scopes and unscoped operator work are refused, never
    resolved using arbitrary static paths or GPU timestamps. Existing real
    module ancestry must remain identical after operator contexts are erased.
    """
    names = set(names)
    if any(not isinstance(name, str) or not name or '*' in name or '(' in name for name in names):
        raise ValueError('Operator context requires one exact native CPU name')
    if isinstance(metadata, list):
        metadata = dict(calls=metadata, auxiliary=[])
    metadata = copy.deepcopy(metadata)
    records = {value['id']: value for value in metadata['calls']}
    if len(records) != len(metadata['calls']):
        raise ValueError('Duplicate module call metadata')
    # Until binding kinds are supported in the shipped matcher, explicitly
    # refuse collisions rather than letting an operator declaration claim a
    # module or native CUDA leaf bearing the same string.
    if names & {value['class_name'] for value in records.values()}:
        raise ValueError('Operator name collides with a module class')
    if names & {base._strip_class_name(event.name()) for event in gpu}:
        raise ValueError('Operator name collides with a native CUDA leaf')
    windows, events, operators = {}, {}, []
    for event in cpu:
        name = event.name()
        if name.startswith(forward_prefix):
            index = int(name[len(forward_prefix):])
            if index in windows:
                raise ValueError('Duplicate forward annotation')
            windows[index] = event
        elif name.startswith(base.CALL_PREFIX):
            index = int(name[len(base.CALL_PREFIX):])
            if index in events:
                raise ValueError('Duplicate module annotation')
            events[index] = event
        elif name in names:
            if event.activity_type() in ('cuda_runtime', 'cuda_driver'):
                raise ValueError('A CUDA launch API cannot be declared an operator scope')
            operators.append(event)
    if set(events) != set(records):
        raise ValueError('Module metadata/native scopes disagree')
    if not operators:
        return list(cpu), metadata, dict(operator_calls=0, observed_operators=[],
                                        inactive_operator_contexts=sorted(names))
    forwards = base._ScopeIndex(windows)
    original_parents = {index: value['parent'] for index, value in records.items()}
    # Validate old hierarchy before supplementing it, including OS threads.
    base._ScopeIndex(events, original_parents)
    next_id = max(records, default=-1) + 1
    operator_ids = set()
    augmented_cpu = list(cpu)
    for event in operators:
        forward = forwards.find(event)
        if forward is None:
            raise ValueError('Declared operator executes outside an observed forward thread/boundary')
        index = next_id
        next_id += 1
        operator_ids.add(index)
        event_name = event.name()
        records[index] = dict(id=index, parent=None, forward=forward,
            module_id=-(index + 1), class_name=event_name, summary_name=event_name,
            thread_id=event.device_resource_id(), paths=[], completed=True,
            scope_kind='operator', operator_name=event_name,
            normalization_context_only=True)
        annotation = _Annotation(event, base.CALL_PREFIX + str(index))
        events[index] = annotation
        augmented_cpu.append(annotation)

    by_thread = defaultdict(list)
    for index, event in events.items():
        by_thread[event.device_resource_id()].append(index)
    for indices in by_thread.values():
        indices.sort(key=lambda index: (events[index].start_ns(), -events[index].end_ns()))
        stack = []
        for index in indices:
            event = events[index]
            if event.end_ns() <= event.start_ns():
                raise ValueError('Invalid operator/module CPU interval')
            while stack and events[stack[-1]].end_ns() <= event.start_ns():
                stack.pop()
            if stack:
                parent = stack[-1]
                enclosing = events[parent]
                if (enclosing.start_ns(), enclosing.end_ns()) == (event.start_ns(), event.end_ns()):
                    raise ValueError('Equal CPU intervals do not establish unique operator/module ancestry')
                if not base._contained(enclosing, event):
                    raise ValueError('Crossing operator/module CPU intervals')
                if records[parent]['forward'] != records[index]['forward']:
                    raise ValueError('Operator/module scopes cross forward boundaries')
            else:
                parent = None
            records[index]['parent'] = parent
            stack.append(index)
    for index, old_parent in original_parents.items():
        parent = records[index]['parent']
        while parent in operator_ids:
            parent = records[parent]['parent']
        if parent != old_parent:
            raise ValueError('Operator scopes would alter the observed real module ancestry')
    metadata['calls'] = list(records.values())
    observed = {records[index]['operator_name'] for index in operator_ids}
    return augmented_cpu, metadata, dict(operator_calls=len(operator_ids),
        observed_operators=sorted(observed), inactive_operator_contexts=sorted(names - observed))


def extract_with_operators(cpu, gpu, catalog, metadata, iterations, *, contexts=(),
                           forward_prefix=base.FORWARD_PREFIX, **kwargs):
    """Extract canonical operator/module/raw costs with transparent contexts."""
    declared = operator_contexts(catalog)
    cpu, metadata, audit = add_operator_contexts(cpu, gpu, metadata, set(contexts) | set(declared),
                                                forward_prefix=forward_prefix)
    translated = _translate_catalog(catalog)
    result = extract_with_methods(cpu, gpu, translated, metadata, iterations,
                                  forward_prefix=forward_prefix, **kwargs)
    operator_ids = {record['id'] for record in metadata['calls']
                    if record.get('scope_kind') == 'operator'}
    operator_canonicals = {name for name, spec in catalog.items() if 'operator' in spec}
    callable_ids = {record['id'] for record in metadata['calls']
                    if record.get('scope_kind') == 'callable'}
    callable_canonicals = {name for name, spec in catalog.items() if 'callable' in spec}
    # Include inactive category contexts in collision checks: a raw/module
    # selector must not accidentally bind one merely because its own timing
    # is being collected by a different category.
    callable_names = {record['class_name'] for record in metadata['calls']
                      if record['id'] in callable_ids}
    callable_names.update(catalog[name]['callable'] for name in callable_canonicals)
    if callable_names & {record['class_name'] for record in metadata['calls']
                          if record['id'] not in callable_ids}:
        raise ValueError('Callable semantic name collides with a module/operator class')
    if callable_names & {base._strip_class_name(event.name()) for event in gpu}:
        raise ValueError('Callable semantic name collides with a native CUDA leaf')
    for call in result['per_call']:
        if call['canonical'] is not None and ((call['id'] in operator_ids)
                                             != (call['canonical'] in operator_canonicals)):
            raise ValueError('Catalog binding kind does not match operator/module scope kind')
        if call['canonical'] is not None and ((call['id'] in callable_ids)
                                             != (call['canonical'] in callable_canonicals)):
            raise ValueError('Catalog binding kind does not match callable/module scope kind')
    result['operator_audit'] = audit
    result['metadata'] = metadata
    result['normalization_contract'] = 'method/operator/callable contexts affect matching, not descendant parent execution units'
    if 'callable_audit' in metadata:
        result['callable_audit'] = metadata['callable_audit']
    return result
