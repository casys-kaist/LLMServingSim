"""Native per-forward skew timing for vLLM 0.28, with CPU-scope ownership.

CUDA activity is neither shifted in time nor scaled. CPU call intervals and
native launch correlations establish ownership; GPU annotations are excluded.
"""

from collections import Counter
from contextlib import contextmanager
import copy
import math

from .timings import extract_samples

def canonical_module_tree(roots, is_module):
    nodes, parents, aliases = [], [], []
    seen = set()

    def visit(node, parent):
        if id(node) in seen:
            raise ValueError('Repeated node or cycle in native module tree')
        seen.add(id(node))
        if (parent is not None and is_module(node.event)
                and node.event is nodes[parent].event):
            if node.children:
                raise ValueError('Nonempty self-alias module cannot be discarded')
            aliases.append(node)
            return
        index = len(nodes)
        nodes.append(node)
        parents.append(parent)
        for child in node.children:
            visit(child, index)

    for root in roots:
        visit(root, None)
    modules = [i for i, node in enumerate(nodes) if is_module(node.event)]
    proposed = list(parents)
    intervals = {}
    for i in modules:
        event = nodes[i].event
        start, duration, tid = event.start_time_ns, event.duration_time_ns, event.start_tid
        if (not all(isinstance(v, int) for v in (start, duration, tid)) or duration < 0):
            raise ValueError('Module ownership needs valid CPU intervals')
        intervals[i] = (start, start+duration, tid)
    for i in modules:
        start, end, tid = intervals[i]
        candidates = []
        for j in modules:
            if i == j:
                continue
            left, right, other_tid = intervals[j]
            if tid != other_tid:
                continue
            contains = left <= start and end <= right
            inside = start <= left and right <= end
            if max(left, start) < min(right, end) and not (contains or inside):
                raise ValueError('Crossing CPU module intervals are ambiguous')
            if contains:
                if right-left == end-start:
                    raise ValueError('Distinct module calls have identical CPU intervals')
                candidates.append((right-left, j))
        if candidates:
            candidates.sort()
            if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
                raise ValueError('CPU module call has no unique closest container')
            proposed[i] = candidates[0][1]
        else:
            proposed[i] = None
    changed = [i for i in modules if proposed[i] != parents[i]]
    if not changed and not aliases:
        return roots, dict(reparented_modules=0, removed_empty_self_aliases=0)
    cloned = [copy.copy(node) for node in nodes]
    result = []
    for node in cloned:
        node.children = []
        node.parent = None
    for i, node in enumerate(cloned):
        parent = proposed[i]
        if parent is None:
            result.append(node)
        else:
            node.parent = cloned[parent]
            cloned[parent].children.append(node)
    # Every nonmodule node, including every CUDA leaf, is kept exactly once.
    original_activity = sorted(id(n.event) for n in nodes if not is_module(n.event))
    rebuilt_activity = []

    def activity(node):
        if not is_module(node.event):
            rebuilt_activity.append(id(node.event))
        for child in node.children:
            activity(child)

    for root in result:
        activity(root)
    if sorted(rebuilt_activity) != original_activity:
        raise ValueError('Canonical module tree changed activity coverage')
    return result, dict(reparented_modules=len(changed), removed_empty_self_aliases=len(aliases))


MARKER = 'skew_dynamic_forward_'
CPU_TAG = '_EventType.TorchOp'


def partition_cpu_roots(roots, iterations, windows=None):
    if (type(iterations) is not int or iterations < 1 or windows is None
            or len(windows) != iterations):
        raise ValueError('Every timed forward requires an explicit CPU scope')
    for index, window in enumerate(windows):
        if (str(window.tag) != CPU_TAG or window.name != MARKER + str(index)
                or window.duration_time_ns <= 0):
            raise ValueError('Only indexed positive-duration CPU TorchOp scopes are authoritative')
    for index, left in enumerate(windows):
        for right in windows[index + 1:]:
            if (left.start_tid == right.start_tid
                    and max(left.start_time_ns, right.start_time_ns) < min(
                        left.start_time_ns + left.duration_time_ns,
                        right.start_time_ns + right.duration_time_ns)):
                raise ValueError('Timed CPU scopes overlap')
    groups = [[] for _ in windows]
    for root in roots:
        event = root.event
        candidates = [index for index, window in enumerate(windows)
            if event.duration_time_ns >= 0 and window.start_tid == event.start_tid
            and window.start_time_ns <= event.start_time_ns
            and event.start_time_ns + event.duration_time_ns <=
                window.start_time_ns + window.duration_time_ns]
        if len(candidates) != 1:
            raise ValueError('Module call has no unique complete CPU-forward interval')
        index = candidates[0]
        ancestor, seen = event, {}
        while ancestor is not None:
            if id(ancestor) in seen or len(seen) >= 256:
                raise ValueError('CPU ancestry is cyclic or exceeds the bounded audit')
            # Pybind parent access can create short-lived Python wrappers.
            # Retain them so allocator address reuse cannot resemble a cycle.
            seen[id(ancestor)] = ancestor
            if str(ancestor.tag) == CPU_TAG and ancestor.name.startswith(MARKER):
                window = windows[index]
                if any(getattr(ancestor, field) != getattr(window, field) for field in
                       ('name', 'start_tid', 'start_time_ns', 'duration_time_ns')):
                    raise ValueError('CPU marker ancestry contradicts complete CPU containment')
            ancestor = ancestor.parent
        groups[index].append(root)
    if any(not group for group in groups):
        raise ValueError('A timed forward has no module events')
    return groups


def extract_forwards(results, catalog, iterations):
    from torch._C._profiler import _EventType
    from .cuda_timing import from_vllm

    windows = {}

    def find_markers(event):
        # CUDA-enabled Kineto can also expose a second, correlated event
        # with the annotation's name. Only the CPU TorchOp is the scope.
        if event.tag == _EventType.TorchOp and event.name.startswith(MARKER):
            index = int(event.name[len(MARKER):])
            if index in windows:
                raise ValueError(f'Duplicate CPU timed-forward scope: {index}')
            windows[index] = event
        for child in event.children:
            find_markers(child)

    for event in results._kineto_results.experimental_event_tree():
        find_markers(event)
    if set(windows) != set(range(iterations)):
        raise ValueError('Missing timed-forward scopes')
    results, activity_audit = from_vllm(results)
    groups = partition_cpu_roots(results._module_tree, iterations, [windows[i] for i in range(iterations)])
    forwards, kernels = [], []
    for roots in groups:
        # Reuse upstream's exact tree construction and this repo's existing
        # normalization. Only the unmerged input roots are partitioned.
        one = copy.copy(results)
        one._module_tree = roots
        one._build_stats_trees()
        summary = one.convert_stats_to_dict()['summary_stats']
        values = {s.layer: s.microseconds for s in extract_samples(summary, catalog, 1)}
        if set(values) != set(catalog):
            raise ValueError(f'Missing catalog entries: {set(catalog)-set(values)}')
        if any(not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError('Invalid per-invocation CUDA timing')
        counts = Counter()

        def leaves(node):
            if not node.children:
                event = one._get_kineto_gpu_event(node)
                if event is not None:
                    counts[event.name()] += 1
            for child in node.children:
                leaves(child)

        for root in roots:
            leaves(root)
        forwards.append(values)
        kernels.append(dict(counts))
    aggregate = {s.layer: s.microseconds for s in extract_samples(
        results.convert_stats_to_dict()['summary_stats'], catalog, iterations)}
    deltas = {}
    for name, value in aggregate.items():
        mean = sum(f[name] for f in forwards)/iterations
        deltas[name] = mean-value
        if not math.isclose(mean, value, rel_tol=1e-6, abs_tol=1e-5):
            raise ValueError(f'Per-forward attribution does not conserve {name}: {mean} vs {value}')
    from profiler.core.skew import PROTOCOL, measurement_fingerprint
    return dict(per_forward_us=forwards, aggregate_us=aggregate,
                measurement_protocol=PROTOCOL, measurement_sha256=measurement_fingerprint(),
                conservation_delta_us=deltas, cuda_kernel_counts=kernels,
                excluded_gpu_annotations=activity_audit['excluded_gpu_annotations'],
                activity_ownership=activity_audit['activity_ownership'],
                cuda_activity=activity_audit)


def initialize(runner):
    """Initialize history once; never time zeroing or leave undefined KV data."""
    import torch
    caches = getattr(runner, "kv_caches", None)
    if not caches:
        raise ValueError("No KV cache available for skew measurement")
    visited, total = set(), 0

    def clear(value):
        nonlocal total
        if isinstance(value, torch.Tensor):
            key = (value.device, value.data_ptr())
            if key not in visited:
                visited.add(key)
                value.zero_()
                total += value.numel() * value.element_size()
        elif isinstance(value, dict):
            for child in value.values():
                clear(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                clear(child)
        else:
            raise ValueError("Unsupported KV cache container")
    with torch.inference_mode():
        clear(caches)
    torch.cuda.synchronize()
    return dict(cache_bytes=total, protocol="native-skew-per-forward-v2",
                kv_initialization="zeroed once; native query writes thereafter")


def _verify_request_geometry(metadata, requests):
    starts = getattr(metadata, "query_start_loc", None)
    lengths = getattr(metadata, "seq_lens", None)
    if starts is None or lengths is None:
        raise ValueError("Attention geometry requires query starts and sequence lengths")
    starts, lengths = starts.cpu().tolist(), lengths.cpu().tolist()
    if (len(starts) != len(lengths) + 1 or not starts or starts[0] != 0
            or any(b < a or h < b - a
                   for a, b, h in zip(starts, starts[1:], lengths))):
        raise ValueError("Invalid attention query/history boundaries")
    pairs = [(b - a, h - (b - a)) for a, b, h
             in zip(starts, starts[1:], lengths) if b > a]
    if Counter(pairs) != Counter(map(tuple, requests)):
        raise ValueError("Executed query/history geometry differs from requested shot")


@contextmanager
def _verified_builder_outputs(runner, requests, state_layers):
    """Verify common geometry before a backend discards query boundaries.

    Sparse indexer metadata can omit query_start_loc after splitting prefill
    and decode. Retain exact output-object identities, not a backend-name
    exemption. Wrappers exist only during the untimed warmup.
    """
    outputs, patched = {}, []
    try:
        for groups in getattr(runner, "attn_groups", ()):
            for group in groups:
                if all(name in state_layers for name in group.layer_names):
                    continue
                for builder in group.metadata_builders:
                    if any(builder is entry[0] for entry in patched):
                        continue
                    original = builder.build
                    had_local = "build" in vars(builder)
                    local = vars(builder).get("build")

                    def build(*args, _original=original, **kwargs):
                        # GPUModelRunner passes common metadata by keyword.
                        common = kwargs.get("common_attn_metadata")
                        if common is None:
                            raise ValueError("Missing common attention metadata at build")
                        _verify_request_geometry(common, requests)
                        result = _original(*args, **kwargs)
                        # Keep the object alive so id reuse cannot validate
                        # unrelated metadata produced later in the warmup.
                        outputs[id(result)] = result
                        return result

                    builder.build = build
                    patched.append((builder, had_local, local))
        yield outputs
    finally:
        for builder, had_local, local in reversed(patched):
            if had_local:
                builder.build = local
            else:
                del builder.build


def measure(runner, shot_dict, catalog, iterations=3):
    """Warm up, verify executed geometry and retain individual kernel times."""
    import torch
    from torch.profiler import record_function
    from vllm.forward_context import get_forward_context
    from vllm.profiler.layerwise_profile import layerwise_profile
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs
    from .batch import Shot, assemble_scheduler_output
    from .sampler_shim import wrap_sampler_for_profiling

    if type(iterations) is not int or iterations < 1:
        raise ValueError("Positive timed-forward count required")
    shot = Shot.hydrate(shot_dict)
    wrap_sampler_for_profiling(runner)
    checked, finite = [], []
    # State-space metadata has no KV history axis. Validate the paged
    # attention groups of a hybrid, not its recurrent state representation.
    state_layers = set()
    for group in runner.kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, MambaSpec):
            state_layers.update(group.layer_names)
        elif isinstance(spec, UniformTypeKVCacheSpecs):
            state_layers.update(name for name, item in spec.kv_cache_specs.items()
                                if isinstance(item, MambaSpec))

    def before(_module, _args):
        metadata = get_forward_context().attn_metadata
        for bundle in metadata if isinstance(metadata, list) else [metadata]:
            if not isinstance(bundle, dict):
                raise ValueError("Unsupported attention metadata container")
            for name, item in bundle.items():
                if name in state_layers:
                    continue
                if (getattr(item, "query_start_loc", None) is not None
                        and getattr(item, "seq_lens", None) is not None):
                    _verify_request_geometry(item, shot.requests)
                elif id(item) not in verified_outputs:
                    raise ValueError(f"Cannot verify attention geometry for {name}")
                checked.append(name)

    def check_tensor(value):
        if isinstance(value, torch.Tensor):
            finite.append(bool(torch.isfinite(value).all().item()))
        elif isinstance(value, (tuple, list)):
            for item in value:
                check_tensor(item)
        else:
            raise ValueError("Cannot verify finite model output")

    def after(_module, _args, output):
        check_tensor(output)

    def fire():
        batch, _ = assemble_scheduler_output(shot, runner)
        if runner.execute_model(batch) is None:
            runner.sample_tokens(None)

    with torch.inference_mode():
        handles = [runner.model.register_forward_pre_hook(before),
                   runner.model.register_forward_hook(after)]
        try:
            with _verified_builder_outputs(runner, shot.requests, state_layers) as verified_outputs:
                fire()
        finally:
            for handle in handles:
                handle.remove()
        if not checked or not finite or not all(finite):
            raise ValueError("Skew warmup geometry/finite-value control failed")
        torch.cuda.synchronize()
        with layerwise_profile() as hook:
            for index in range(iterations):
                with record_function(MARKER + str(index)):
                    fire()
        torch.cuda.synchronize()
    return extract_forwards(hook.results, catalog, iterations)
