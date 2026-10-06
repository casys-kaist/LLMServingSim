"""Attribute an isolated component by native CPU launch correlation IDs.

CPU scopes establish ownership only. Durations come exclusively from CUDA
activities; module tree summaries and CPU/GPU temporal containment are unused.
"""
from collections import Counter, defaultdict
import math

from .skew_measurement import MARKER


class NativeOwnershipError(ValueError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def extract_eager_forwards(kineto, iterations):
    from torch.autograd import DeviceType
    if type(iterations) is not int or iterations < 1:
        raise ValueError('Positive eager forward count is required')
    events = list(kineto.events())
    windows = {}
    for event in events:
        if (event.device_type() == DeviceType.CPU and event.is_user_annotation()
                and event.name().startswith(MARKER)):
            index = int(event.name()[len(MARKER):])
            if index in windows or event.duration_ns() <= 0:
                raise ValueError('Duplicate or empty eager forward marker')
            windows[index] = event
    if set(windows) != set(range(iterations)):
        raise ValueError('Missing eager forward marker')
    ordered = sorted(windows.values(), key=lambda e:e.start_ns())
    if any(a.end_ns() > b.start_ns() for a,b in zip(ordered,ordered[1:])):
        raise ValueError('Overlapping isolated forward markers')
    activity = [e for e in events if e.device_type() == DeviceType.CUDA
                and not e.is_user_annotation()]
    correlations = {e.correlation_id() for e in activity}
    owners = defaultdict(set)
    thread_mismatches = Counter()
    for launch in events:
        # Torch operator IDs and CUDA runtime/driver IDs are different
        # namespaces. Equal integers do not connect an aten op to a kernel.
        if (launch.device_type() != DeviceType.CPU
                or launch.activity_type() not in ('cuda_runtime','cuda_driver')
                or launch.correlation_id() not in correlations):
            continue
        if launch.name() == 'cudaGraphLaunch':
            raise ValueError('Graph replay requires graph activity attribution')
        for index, window in windows.items():
            # Compare the OS thread identity recorded for both CPU scopes
            # and CUDA API activities, not Torch's logical thread namespace.
            if (launch.device_resource_id() == window.device_resource_id()
                    and window.start_ns() <= launch.start_ns()
                    and launch.end_ns() <= window.end_ns()):
                owners[launch.correlation_id()].add(index)
                if launch.start_thread_id() != window.start_thread_id():
                    thread_mismatches[launch.name()] += 1
    values, counts = [0.]*iterations, [Counter() for _ in range(iterations)]
    for event in activity:
        found = owners[event.correlation_id()]
        if len(found) != 1:
            def describe(e):
                return dict(name=e.name(),device=str(e.device_type()),start_ns=e.start_ns(),
                    end_ns=e.end_ns(),duration_ns=e.duration_ns(),correlation=e.correlation_id(),
                    linked=e.linked_correlation_id() if hasattr(e,'linked_correlation_id') else None,
                    logical_thread=e.start_thread_id(),resource=e.device_resource_id(),
                    activity=str(e.activity_type()) if hasattr(e,'activity_type') else None)
            audit=dict(unowned=describe(event),windows={i:describe(w) for i,w in windows.items()},
                matching_cpu=[describe(e) for e in events if e.device_type()==DeviceType.CPU
                              and e.correlation_id()==event.correlation_id()],
                launches=[describe(e) for e in events if e.name().startswith(('cudaLaunch','cuLaunch'))])
            raise NativeOwnershipError(f'Native CUDA activity has no unique CPU launch owner: {event.name()}, correlation={event.correlation_id()}',audit)
        index = next(iter(found))
        us = event.duration_ns()/1000
        if not math.isfinite(us) or us <= 0:
            raise ValueError('Invalid native CUDA duration')
        values[index] += us
        counts[index][event.name()] += 1
    if not all(counts) or any(c != counts[0] for c in counts[1:]):
        raise ValueError('Repeated isolated eager calls have missing or inconsistent CUDA activity')
    total = sum(e.duration_ns()/1000 for e in activity)
    if not math.isclose(sum(values), total, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError('Native eager CUDA activity was omitted or counted twice')
    excluded = [e.name() for e in events if e.device_type() == DeviceType.CUDA and e.is_user_annotation()]
    return dict(per_forward_us=values, cuda_kernel_counts=[dict(c) for c in counts],
                conservation_delta_us={'component':sum(values)-total},
                excluded_gpu_annotations=excluded, logical_thread_id_mismatches=dict(thread_mismatches),
                attribution='native_eager_launch_correlation_v1')
