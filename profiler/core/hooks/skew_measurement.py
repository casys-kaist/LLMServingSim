"""Per-forward heterogeneous CUDA acquisition and untimed geometry checks."""

from collections import Counter
from contextlib import contextmanager
import math


MARKER = "skew_dynamic_forward_"


def initialize(runner):
    """Inspect storage; each context separately initializes its assigned pages."""
    import torch
    from profiler.core.skew import PROTOCOL
    caches = getattr(runner, "kv_caches", None)
    if not caches:
        raise ValueError("No KV cache available for skew measurement")
    visited, total = set(), 0

    def inspect(value):
        nonlocal total
        if isinstance(value, torch.Tensor):
            key = (value.device, value.data_ptr())
            if key not in visited:
                visited.add(key)
                total += value.numel() * value.element_size()
        elif isinstance(value, dict):
            for child in value.values():
                inspect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                inspect(child)
        else:
            raise ValueError("Unsupported KV cache container")
    with torch.inference_mode():
        inspect(caches)
    torch.cuda.synchronize()
    return dict(cache_bytes=total, protocol=PROTOCOL,
                kv_initialization="per-context vLLM dummy random KV; zero-start recurrent state")


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


def measure(runner, shot_dict, catalog, iterations=3, *, scope_contexts=None):
    """Capture individual forwards through the common eager CUDA collector."""
    import torch
    from .batch import Shot
    from .history import complete_forward, prepare_history
    from .runner_capture import capture, verified_warmup
    from .sampler_shim import wrap_sampler_for_profiling
    from profiler.core.skew import PROTOCOL, measurement_fingerprint
    from .capture_scope_contract import normalize_contexts

    scope_contexts = normalize_contexts(scope_contexts, catalog)

    if type(iterations) is not int or iterations < 1:
        raise ValueError("Positive timed-forward count required")
    shot = Shot.hydrate(shot_dict)
    wrap_sampler_for_profiling(runner)
    fresh_batch = prepare_history(runner, shot)

    def fire():
        complete_forward(runner, fresh_batch())

    with torch.inference_mode():
        verified_warmup(runner, shot, fire)
        measured = capture(runner, fire, catalog, iterations, forward_prefix=MARKER,
                           scope_contexts=scope_contexts)
    forwards = measured["per_forward_layers_us"]
    aggregate = measured["layer_us"]
    deltas = {}
    for values in forwards:
        if set(values) != set(catalog):
            raise ValueError(f"Missing per-forward attention entries: {set(catalog)-set(values)}")
        if any(not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("Invalid per-forward attention timing")
    for name, value in aggregate.items():
        mean = sum(values[name] for values in forwards) / iterations
        deltas[name] = mean - value
        if not math.isclose(mean, value, rel_tol=1e-6, abs_tol=1e-5):
            raise ValueError(f"Per-forward attribution does not conserve {name}")
    audit = measured["audit"]
    return dict(per_forward_us=forwards, aggregate_us=aggregate,
                measurement_protocol=PROTOCOL, measurement_sha256=measurement_fingerprint(scope_contexts, catalog=catalog),
                conservation_delta_us=deltas,
                cuda_kernel_counts=audit["per_forward_activity_counts"],
                excluded_gpu_annotations=audit["excluded_annotations"],
                activity_ownership=audit, cuda_activity=audit)
