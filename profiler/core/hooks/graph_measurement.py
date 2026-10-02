"""One isolated captured component per forward, attributed by launch ID.

CUDA graphs share a launch correlation ID across their kernels. Kernel names
are not unique within a launch. Read each native activity exactly once;
neither module-tree membership nor CPU/GPU time containment assigns it.
"""
from collections import Counter
import math

from .skew_measurement import MARKER


def extract_graph_forwards(kineto, iterations):
    from torch.autograd import DeviceType
    events = list(kineto.events())
    windows = {}
    for event in events:
        if (event.device_type() == DeviceType.CPU and event.is_user_annotation()
                and event.name().startswith(MARKER)):
            index = int(event.name()[len(MARKER):])
            if index in windows:
                raise ValueError('Duplicate graph forward marker')
            windows[index] = event
    if set(windows) != set(range(iterations)):
        raise ValueError('Missing graph forward marker')
    launches = [e for e in events if e.name() == 'cudaGraphLaunch' and e.device_type() == DeviceType.CPU]
    if len(launches) != iterations:
        raise ValueError('Exactly one isolated graph launch is required per timed forward')
    owner = {}
    for launch in launches:
        candidates = [i for i,w in windows.items() if w.start_ns() <= launch.start_ns()
                      and launch.end_ns() <= w.end_ns()]
        if len(candidates) != 1 or launch.correlation_id() in owner:
            raise ValueError('Graph launch does not have unique forward ownership')
        owner[launch.correlation_id()] = candidates[0]
    if set(owner.values()) != set(range(iterations)):
        raise ValueError('Timed forward has no graph launch')
    values, counts, excluded = [0.]*iterations, [Counter() for _ in range(iterations)], []
    activity = [e for e in events if e.device_type() == DeviceType.CUDA]
    for event in activity:
        if event.is_user_annotation():
            excluded.append(event.name())
            continue
        if event.correlation_id() not in owner:
            raise ValueError('Unowned CUDA activity in isolated graph measurement')
        index = owner[event.correlation_id()]
        us = event.duration_ns()/1000
        if not math.isfinite(us) or us <= 0:
            raise ValueError('Invalid CUDA activity duration')
        values[index] += us
        counts[index][event.name()] += 1
    if not all(counts) or any(v <= 0 for v in values):
        raise ValueError('Graph forward has no CUDA activity')
    # Graph structure is fixed; a missing kernel invalidates the entire shot.
    if any(c != counts[0] for c in counts[1:]):
        raise ValueError('Captured graph activity counts differ across replays')
    native_sum = sum(e.duration_ns()/1000 for e in activity if not e.is_user_annotation())
    if not math.isclose(sum(values), native_sum, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError('Graph CUDA activity was duplicated or omitted')
    return dict(per_forward_us=values, cuda_kernel_counts=[dict(c) for c in counts],
                conservation_delta_us={'component':sum(values)-native_sum},
                excluded_gpu_annotations=excluded,
                attribution='native_graph_launch_correlation_v1')
