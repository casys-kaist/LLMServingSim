"""Shared eager runner capture; inputs and correctness checks are untimed."""

import math
import logging


def verified_warmup(runner, shot, fire):
    """Check actual query/history geometry and finite outputs before capture.

    Validation executes one ordinary warmup, with all checks removed before
    timing. Verify recurrent query/history and assigned state pages through
    common builder metadata and its final per-layer representation; paged
    attention groups use their native specs and builders.
    """
    import torch
    from vllm.forward_context import get_forward_context
    from .skew_measurement import _verified_builder_outputs, _verify_request_geometry
    from .recurrent_metadata import RecurrentMetadataGuard

    recurrent = RecurrentMetadataGuard(runner, shot)
    state_layers = set(recurrent.layer_group)
    checked, finite = [], []

    def before(_module, _args):
        metadata = get_forward_context().attn_metadata
        for bundle in metadata if isinstance(metadata, list) else [metadata]:
            if not isinstance(bundle, dict):
                raise ValueError('Unsupported attention metadata container')
            for name, item in bundle.items():
                if name in state_layers:
                    recurrent.validate(name, item)
                elif (getattr(item, 'query_start_loc', None) is not None
                        and getattr(item, 'seq_lens', None) is not None):
                    _verify_request_geometry(item, shot.requests)
                elif id(item) not in verified_outputs:
                    raise ValueError(f'Cannot verify attention geometry for {name}')
                checked.append(name)

    def check(value):
        if isinstance(value, torch.Tensor):
            finite.append(bool(torch.isfinite(value).all().item()))
        elif isinstance(value, (tuple, list)):
            for item in value:
                check(item)
        elif isinstance(value, dict):
            for item in value.values():
                check(item)
        else:
            raise ValueError(f'Cannot verify finite model output: {type(value).__name__}')

    def after(_module, _args, output):
        check(output)

    model = runner.get_model()
    handles = []
    try:
        handles.append(model.register_forward_pre_hook(before))
        handles.append(model.register_forward_hook(after))
        with recurrent, _verified_builder_outputs(runner, shot.requests, state_layers) as verified_outputs:
            fire()
            recurrent.finish()
    finally:
        for handle in handles:
            handle.remove()
    if not checked or not finite or not all(finite):
        raise ValueError('Warmup geometry/finite-value control failed; no timings may be saved')
    torch.cuda.synchronize()
    return dict(requests=[list(row) for row in shot.requests],
                checked_attention_layers=sorted(set(checked)),
                finite_output_tensors=len(finite), finite=True)


def capture(runner, fire, catalog, iterations, *, forward_prefix=None, diagnostic=False,
            scope_contexts=None, expected_contexts=None):
    """Collect native GPU activities without Python stacks or global shape tracing.

    Schema-3 callable declarations may record explicit argument/layout metadata;
    tensor contents and CPU scope durations never become timing samples.

    Model roots come from the runner interfaces, not architecture names. This
    function does not change padding, weights, samplers or backend selection.
    Runner work outside a module remains an explicit audit gap; it is never
    silently added to a layer or interpreted as CPU latency.
    """
    from .activity_collector import collect, roots_from_runner, FORWARD_PREFIX
    from .sampler_shim import wrap_sampler_for_profiling

    if type(iterations) is not int or iterations < 1:
        raise ValueError('A positive timed-forward count is required')
    if not runner.vllm_config.model_config.enforce_eager:
        raise ValueError('Layer acquisition requires enforce_eager; graph components use their own contract')
    result = collect(roots_from_runner(runner, sampler_adapter=wrap_sampler_for_profiling),
                     iterations, lambda _index: fire(), catalog,
                     forward_prefix=forward_prefix or FORWARD_PREFIX, scope_contexts=scope_contexts,
                     expected_contexts=expected_contexts, context_diagnostic=diagnostic)
    values = result['layer_us']
    if ((not values and not result.get('context_samples'))
            or any(not math.isfinite(v) or v <= 0 for v in values.values())):
        raise ValueError('Capture contains no valid canonical GPU timings')
    forwards = result['per_forward_layers_us']
    if len(forwards) != iterations or any(
            set(row) != set(values) or any(not math.isfinite(v) or v <= 0 for v in row.values())
            for row in forwards):
        raise ValueError('Timed forwards have inconsistent active canonical layers')
    audit = result['audit']
    # Whole-architecture declarations include version/backend alternatives.
    # Keep unresolved targets visible even when the timing-only RPC discards
    # diagnostic metadata. A missing alternative is not itself a zero timing
    # or proof that a positively selected component has been measured.
    resolution = result.get('callable_audit', {}).get('resolution', ())
    inactive = {(entry['name'], tuple(entry['reasons'])) for entry in resolution
                if not entry['resolved'] or entry['reasons']}
    reported = getattr(runner, '_profile_callable_alternatives', frozenset())
    if inactive - reported:
        logging.getLogger(__name__).warning(
            'Unavailable callable capture alternatives (no timing is invented): %s. '
            'Selected component layouts must establish positive CUDA coverage.',
            '; '.join(name + ': ' + ','.join(reasons) for name, reasons in sorted(inactive - reported)))
        runner._profile_callable_alternatives = reported | inactive
    if audit['overlapping_canonical_claims'] and not diagnostic:
        raise ValueError('Catalog assigns the same CUDA activity to multiple canonical layers; '
                         'inspect profiler coverage before saving timings')
    # Category slices deliberately leave other model layers unbound. Only
    # work outside all module scopes is a runner-boundary gap here; full
    # catalog coverage provides its detailed, duration-conserving ledger.
    gaps = audit['unscoped_gpu'] + audit['outside_forward']
    if gaps:
        names = frozenset(item['identity'][0] for item in gaps)
        reported = getattr(runner, '_profile_unscoped_names', frozenset())
        if names - reported:
            logging.getLogger(__name__).warning(
                'CUDA activities outside module scopes are excluded from layer tables: %s. '
                'Use profiler coverage for the complete GPU-work ledger.',
                ', '.join(sorted(names - reported)))
            runner._profile_unscoped_names = reported | names
    return result
