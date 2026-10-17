"""Read-only acquisition runtime facts for host-side sampling plans.

These facts describe the loaded acquisition stack, not a completed serving
capture or the backend choices of an unmodified full-depth deployment.
"""


_SUPPORT = {'NEVER': 0, 'UNIFORM_SINGLE_TOKEN_DECODE': 1,
            'UNIFORM_BATCH': 2, 'ALWAYS': 3}
_RUNNERS = {'vllm.v1.worker.gpu_model_runner': 'v1',
            'vllm.v1.worker.gpu.model_runner': 'v2'}


def _runner_identity(runner):
    """Identify the instantiated runner, including an extension subclass."""
    identities = {_RUNNERS[base.__module__] for base in type(runner).__mro__
                  if base.__module__ in _RUNNERS}
    if len(identities) != 1:
        raise ValueError('Acquisition runner has no unambiguous supported base class')
    return identities.pop()


def _support_name(value):
    name = getattr(value, 'name', None)
    if name not in _SUPPORT:
        raise ValueError('Unknown attention CUDA graph support value')
    return name


def _attention_support(runner, identity):
    """Query existing builders; never build metadata or resolve/capture graphs.

    vLLM 0.31 v1/worker/gpu/attn_utils.py::get_attn_cg_support uses
    each group's first builder and the group's original KV spec. V2 then
    narrows that result with model_state.get_additional_cg_support().
    """
    groups = getattr(runner, 'attn_groups', None)
    if not groups or any(not group_list for group_list in groups):
        raise ValueError('Acquisition attention groups are unavailable or incomplete')
    supports = []
    for group_list in groups:
        for group in group_list:
            builder = group.get_metadata_builder(0)
            supports.append(_support_name(builder.get_cudagraph_support(
                runner.vllm_config, group.kv_cache_spec)))
    if not supports:
        raise ValueError('No acquisition attention builders were observed')
    if identity == 'v2':
        extra, _backend = runner.model_state.get_additional_cg_support()
        supports.append(_support_name(extra))
    elif identity != 'v1':
        raise ValueError('Unknown runner cannot establish additional graph restrictions')
    return min(supports, key=_SUPPORT.__getitem__)


def snapshot(worker):
    """Observe an already initialized worker without changing its execution."""
    from profiler.core.execution_metadata import canonical_dtype
    import torch

    runner = worker.model_runner
    cfg = runner.vllm_config
    model, cache, scheduler = cfg.model_config, cfg.cache_config, cfg.scheduler_config
    activation = canonical_dtype(model.dtype)
    kv = canonical_dtype(cache.cache_dtype)
    if kv == 'auto':
        kv = activation
    result = dict(source='acquisition_runtime', scope='acquisition_stack',
                  gpu=None, compute_capability=None, runner=None,
                  activation_dtype=activation, kv_cache_dtype=kv,
                  quantization=model.quantization,
                  enable_chunked_prefill=scheduler.enable_chunked_prefill,
                  enable_prefix_caching=cache.enable_prefix_caching,
                  max_model_len=model.max_model_len,
                  attention_cg_support=None)
    try:
        device = torch.cuda.get_device_properties(worker.device)
        result.update(gpu=device.name,
                      compute_capability=f'{device.major}.{device.minor}')
    except Exception as exc:
        result['device_reason'] = f'{type(exc).__name__}: {exc}'
    try:
        result['runner'] = _runner_identity(runner)
    except Exception as exc:
        result['runner_reason'] = f'{type(exc).__name__}: {exc}'
    try:
        result['attention_cg_support'] = _attention_support(runner, result['runner'])
    except Exception as exc:
        result['attention_cg_support_reason'] = f'{type(exc).__name__}: {exc}'
    return result
