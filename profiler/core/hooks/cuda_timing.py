"""Per-call CUDA unions; CPU scopes establish ownership, never latency."""

from collections import defaultdict


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
