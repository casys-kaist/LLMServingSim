"""Repair CUDA-leaf ownership using CPU launches, never GPU time containment.

The Python-call tree can attach a kernel beside the module that launched it.
Module CPU intervals identify the call; native runtime/driver correlation IDs
connect that call to CUDA activity. CPU duration is never a latency input.
"""

from collections import defaultdict
import copy


def reparent_cuda_activity(roots, is_module, get_gpu_event, cpu_events):
    """Return an ownership-corrected copy, preserving every event and duration.

    ``cpu_events`` contains native CPU Kineto events. Python module events and
    CUDA API events do not necessarily share a logical thread-ID namespace;
    match each module's exact Python interval to its OS resource ID first.
    Ambiguous or missing launch ownership invalidates the measurement.
    """
    python_resources = defaultdict(set)
    launches = defaultdict(list)
    for event in cpu_events:
        activity = event.activity_type()
        if activity == 'python_function':
            python_resources[(event.start_ns(), event.end_ns(),
                              event.start_thread_id())].add(event.device_resource_id())
        elif activity in ('cuda_runtime', 'cuda_driver'):
            launches[event.correlation_id()].append(event)

    nodes, parents, seen = [], [], set()

    def visit(node, parent):
        if id(node) in seen:
            raise ValueError('Repeated node in CUDA ownership tree')
        seen.add(id(node))
        index = len(nodes)
        nodes.append(node)
        parents.append(parent)
        for child in node.children:
            visit(child, index)

    for root in roots:
        visit(root, None)
    modules = defaultdict(list)
    for index, node in enumerate(nodes):
        if not is_module(node.event):
            continue
        event = node.event
        start, duration = event.start_time_ns, event.duration_time_ns
        if not isinstance(start, int) or not isinstance(duration, int) or duration < 0:
            raise ValueError('Invalid module CPU interval for CUDA ownership')
        resources = python_resources[(start, start + duration, event.start_tid)]
        if len(resources) != 1:
            raise ValueError('Module has no unique native CPU thread mapping')
        modules[next(iter(resources))].append((start, start + duration, index))

    proposed = list(parents)
    correlated = 0
    for index, node in enumerate(nodes):
        if node.children:
            continue
        event = get_gpu_event(node)
        if event is None or event.is_user_annotation():
            continue
        owners = set()
        for launch in launches[event.correlation_id()]:
            candidates = [
                (right - left, module)
                for left, right, module in modules[launch.device_resource_id()]
                if left <= launch.start_ns() and launch.end_ns() <= right
            ]
            if not candidates:
                raise ValueError('CUDA launch is outside every recorded module call')
            candidates.sort()
            if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
                raise ValueError('CUDA launch has ambiguous innermost module ownership')
            owners.add(candidates[0][1])
        if len(owners) != 1:
            raise ValueError('CUDA activity has no unique native launch owner')
        proposed[index] = next(iter(owners))
        correlated += 1

    changed = sum(a != b for a, b in zip(parents, proposed))
    audit = dict(correlated_cuda_activities=correlated,
                 reparented_cuda_activities=changed)
    if not changed:
        return roots, audit
    cloned = [copy.copy(node) for node in nodes]
    for node in cloned:
        node.children, node.parent = [], None
    result = []
    for index, node in enumerate(cloned):
        parent = proposed[index]
        if parent is None:
            result.append(node)
        else:
            node.parent = cloned[parent]
            cloned[parent].children.append(node)

    rebuilt = []

    def collect(node):
        rebuilt.append(id(node.event))
        for child in node.children:
            collect(child)

    for root in result:
        collect(root)
    if sorted(rebuilt) != sorted(id(node.event) for node in nodes):
        raise ValueError('CUDA ownership repair changed event coverage')
    return result, audit
