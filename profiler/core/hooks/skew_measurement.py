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


def measure(runner, shot_dict, catalog, iterations=3, *, scope_contexts=None,
            attention_lookup_sha256=None, component_layout_contract=None):
    """Capture individual forwards through the common eager CUDA collector."""
    import torch
    from .batch import Shot
    from .history import complete_forward, prepare_history
    from .runner_capture import capture, verified_warmup
    from .sampler_shim import wrap_sampler_for_profiling
    from profiler.core.attention_lookup import protocol
    from profiler.core.skew import PROTOCOL, measurement_fingerprint, capture_catalog_for_skew
    from profiler.core import component_layouts as layouts
    from .activity_collector import coverage_report
    from .capture_scope_contract import normalize_contexts

    scope_contexts = normalize_contexts(scope_contexts, catalog)

    if type(iterations) is not int or iterations < 1:
        raise ValueError("Positive timed-forward count required")
    capture_catalog = capture_catalog_for_skew(catalog, component_layout_contract)
    layout_sha256 = None if component_layout_contract is None else component_layout_contract["layout_sha256"]
    invocation_contract = layouts.prepare_invocation_contract(
        component_layout_contract, layout_sha256, capture_catalog,
        runner.vllm_config if component_layout_contract is not None else None,
        scope_contexts=scope_contexts)
    runtime_sha256 = None
    if component_layout_contract is not None:
        from .runtime_admission import collect_runtime_facts, admit_runtime
        runtime = component_layout_contract['runtime']
        admitted = admit_runtime(runtime['descriptor'], collect_runtime_facts(runner), shot=shot_dict)
        runtime_sha256 = admitted['runtime_sha256']
        if runtime_sha256 != runtime['runtime_sha256']:
            raise ValueError('Live skew runtime changed before this shot')
    shot = Shot.hydrate(shot_dict)
    wrap_sampler_for_profiling(runner)
    fresh_batch = prepare_history(runner, shot)

    def fire():
        complete_forward(runner, fresh_batch())

    with torch.inference_mode():
        verified_warmup(runner, shot, fire)
        measured = capture(runner, fire, capture_catalog, iterations, forward_prefix=MARKER,
                           scope_contexts=scope_contexts)
    report = None
    if invocation_contract is not None:
        architecture, choices, expected = invocation_contract
        from profiler.core.categories import _split_linear_attention_shot
        prefill, decode = _split_linear_attention_shot(shot)
        regime = "mixed" if prefill and decode else "prefill" if prefill else "decode"
        report = coverage_report(measured)
        layouts.validate_invocation_report(architecture, choices, report, expected,
                                           iterations, capture_catalog, regime)
    # The same captured probes validate ownership/counts but are not skew rows.
    forwards, aggregate = measured["per_forward_layers_us"], measured["layer_us"]
    if invocation_contract is not None:
        forwards = [{name: values[name] for name in catalog if name in values} for values in forwards]
        aggregate = {name: aggregate[name] for name in catalog if name in aggregate}
    if len(forwards) != iterations or set(aggregate) != set(catalog):
        raise ValueError("Incomplete skew timed-forward set or aggregate")
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
    result = dict(per_forward_us=forwards, aggregate_us=aggregate,
                measurement_protocol=protocol(PROTOCOL, attention_lookup_sha256),
                measurement_sha256=measurement_fingerprint(scope_contexts, catalog=catalog,
                    component_layout_contract=component_layout_contract,
                    attention_lookup_sha256=attention_lookup_sha256),
                conservation_delta_us=deltas,
                cuda_kernel_counts=audit["per_forward_activity_counts"],
                excluded_gpu_annotations=audit["excluded_annotations"],
                activity_ownership=audit, cuda_activity=audit)
    if report is not None:
        result.update(component_layout_sha256=layout_sha256,
                      runtime_sha256=runtime_sha256,
                      canonical_invocations=report["canonical_invocations"])
    return result
