"""Per-call CUDA unions; CPU scopes establish ownership, never latency."""

from collections import defaultdict
import copy
from types import MethodType


def union_ns(intervals):
    spans = sorted(intervals)
    if not spans:
        return 0
    if any(type(a) is not int or type(b) is not int or a >= b for a, b in spans):
        raise ValueError('CUDA intervals must have positive integer duration')
    left, right = spans[0]
    total = 0
    for start, end in spans[1:]:
        if start > right:
            total += right - left
            left, right = start, end
        else:
            right = max(right, end)
    return total + right - left


def activity_identity(event):
    return (event.name(), event.correlation_id(), event.start_ns(), event.end_ns(),
            event.device_index(), event.device_resource_id())


def active_ns(events):
    devices = defaultdict(list)
    for event in events:
        devices[event.device_index()].append((event.start_ns(), event.end_ns()))
    return sum(union_ns(spans) for spans in devices.values())


def exact_resolver(native_gpu_events, is_kineto, native_cpu_events=()):
    """Match an activity, not ownership, with the native device interval."""
    by_key = defaultdict(list)
    for event in native_gpu_events:
        by_key[(event.correlation_id(), event.name(), event.start_ns(), event.end_ns())].append(event)
    cpu_by_interval = defaultdict(list)
    for event in native_cpu_events:
        cpu_by_interval[(event.correlation_id(), event.start_ns(), event.end_ns())].append(event)

    def resolve(node):
        event = node.event
        if not is_kineto(event):
            return None
        start, duration = event.start_time_ns, event.duration_time_ns
        candidates = by_key.get((event.correlation_id, event.name, start, start + duration), ())
        if not candidates and any(e.name() == event.name for e in cpu_by_interval.get(
                (event.correlation_id, start, start + duration), ())):
            return None
        if len(candidates) != 1:
            raise ValueError('CUDA tree leaf has no unique exact native activity')
        return candidates[0]

    return resolve


def reaggregate(original, is_module, native_gpu_events, cpu_events, is_kineto):
    from profiler.core.hooks.skew_measurement import canonical_module_tree
    from profiler.core.hooks.activity_ownership import reparent_cuda_activity

    resolve = exact_resolver(native_gpu_events, is_kineto, cpu_events)
    removed = []
    seen_nodes = set()

    def clean(node, parent=None):
        if id(node) in seen_nodes:
            raise ValueError('Repeated node or cycle in native module tree')
        seen_nodes.add(id(node))
        event = resolve(node) if not node.children else None
        if event is not None and event.is_user_annotation():
            removed.append(activity_identity(event))
            return None
        result = copy.copy(node)
        result.parent = parent
        result.children = [v for child in node.children if (v := clean(child, result)) is not None]
        return result

    roots = [v for root in original._module_tree if (v := clean(root)) is not None]
    roots, module_audit = canonical_module_tree(roots, is_module)
    roots, ownership_audit = reparent_cuda_activity(roots, is_module, resolve, cpu_events)
    memo = {}

    def activities(node):
        if id(node) not in memo:
            if node.children:
                values = tuple(e for child in node.children for e in activities(child))
            else:
                event = resolve(node)
                values = () if event is None else (event,)
            memo[id(node)] = values
        return memo[id(node)]

    all_events = tuple(e for root in roots for e in activities(root))
    identities = [activity_identity(e) for e in all_events]
    if len(set(identities)) != len(identities):
        raise ValueError('Native CUDA activity is claimed more than once')
    for event in all_events:
        if (event.is_user_annotation() or event.duration_ns() <= 0
                or event.duration_ns() != event.end_ns() - event.start_ns()):
            raise ValueError('Invalid retained native CUDA activity')
    if not all_events:
        raise ValueError('No native CUDA activity retained')

    # Build the union before vLLM merges same-class calls. Each invocation
    # keeps its own latency even if it overlaps a different invocation.
    result = copy.copy(original)
    result._module_tree = roots
    result._get_kineto_gpu_event = MethodType(lambda self, node: resolve(node), result)
    result._cumulative_cuda_time = MethodType(
        lambda self, node: active_ns(activities(node)) / 1000, result)
    result._build_stats_trees()
    audit = dict(module_repair=module_audit, activity_ownership=ownership_audit,
                 excluded_annotations=len(removed), retained_activities=len(all_events),
                 excluded_gpu_annotations=[identity[0] for identity in removed],
                 kernel_sum_us=sum(e.duration_ns() for e in all_events) / 1000,
                 root_call_union_us=sum(active_ns(activities(root)) for root in roots) / 1000,
                 global_device_union_us=active_ns(all_events) / 1000)
    return result, audit


def from_vllm(results):
    from torch.autograd import DeviceType
    from torch._C._profiler import _EventType
    from vllm.profiler.utils import event_has_module
    events = results._kineto_results.events()
    return reaggregate(results, event_has_module,
        [e for e in events if e.device_type() == DeviceType.CUDA],
        [e for e in events if e.device_type() == DeviceType.CPU],
        lambda e: e.tag == _EventType.Kineto)
