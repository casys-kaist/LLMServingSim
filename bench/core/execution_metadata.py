"""Read-only worker startup observations; never instrument request timing."""


class ExecutionObserver:
    def serving_execution_snapshot(self, declared_config=None):
        """Called once after engine initialization, before workload submission."""
        try:
            return snapshot_worker(self, declared_config)
        except Exception as exc:
            return {'status': 'unknown', 'reason': f'{type(exc).__name__}: {exc}'}


def snapshot_worker(worker, declared_config=None):
    from profiler.core.execution_metadata import (
        bind_model_config, expected_capture_grids, model_dimensions, precision_scope,
        validate_execution,
    )
    import torch

    runner = worker.model_runner
    cfg = runner.vllm_config
    model, parallel, scheduler = cfg.model_config, cfg.parallel_config, cfg.scheduler_config
    manager = getattr(runner, 'cudagraph_manager', None)
    if manager is None:
        raise ValueError('This runner has no supported capture registry')
    if (cfg.lora_config is not None or parallel.enable_dbo
            or getattr(manager, 'varlen_decode', False)
            or getattr(manager, 'ubatch_runner', None) is not None
            or getattr(parallel, 'decode_context_parallel_size', 1) != 1
            or getattr(parallel, 'prefill_context_parallel_size', 1) != 1):
        raise ValueError('LoRA, DBO, variable decode and context parallel capture are not modeled')
    speculative = cfg.speculative_config
    if speculative is not None and speculative.uses_dynamic_speculative_decoding():
        raise ValueError('Dynamic speculative capture is not modeled')
    if manager.decode_query_len != cfg.num_speculative_tokens + 1:
        raise ValueError('Worker decode query length differs from the supported fixed query')
    mode = manager.cudagraph_mode.name
    if mode != 'NONE' and not manager._graphs_captured:
        raise ValueError('Worker graph capture has not completed')
    if getattr(manager, '_max_full_descs_to_capture', None) is not None:
        raise ValueError('A profiling-only graph capture is not a serving observation')
    cc = cfg.compilation_config
    sizes = sorted(set(cc.cudagraph_capture_sizes or [])) if mode != 'NONE' else []
    full = sorted({desc.num_tokens for desc in manager.graphs})
    piecewise = sorted({desc.num_tokens for group in manager._capture_descs.values()
                        for desc in group if desc.cg_mode.name == 'PIECEWISE'})
    device = torch.cuda.get_device_properties(worker.device)
    model_config = model.hf_config.to_dict()
    dimensions = model_dimensions(model_config)
    if declared_config is not None:
        from copy import deepcopy
        resolved = type(model.hf_config).from_dict(deepcopy(declared_config)).to_dict()
        dimensions = bind_model_config(model_config, declared_config, resolved)
    scope = dict(tp_size=parallel.tensor_parallel_size, pp_size=parallel.pipeline_parallel_size,
                 dp_size=parallel.data_parallel_size,
                 ep_size=(parallel.tensor_parallel_size * parallel.data_parallel_size
                          if parallel.enable_expert_parallel else 1),
                 max_num_seqs=scheduler.max_num_seqs,
                 max_num_batched_tokens=scheduler.max_num_batched_tokens,
                 num_speculative_tokens=cfg.num_speculative_tokens,
                 block_size=cfg.cache_config.block_size,
                 enable_chunked_prefill=scheduler.enable_chunked_prefill,
                 enable_prefix_caching=cfg.cache_config.enable_prefix_caching,
                 max_model_len=model.max_model_len)
    scope.update(precision_scope(model_config, model.dtype,
                                 cfg.cache_config.cache_dtype, model.quantization))
    from vllm.distributed import get_world_group
    world = get_world_group()
    # WORLD already includes the DP offset (also for external launchers).
    rank = world.rank
    record = dict(status='observed', source='worker_capture', rank=rank,
                  gpu=device.name, model_dimensions=dimensions,
                  scope=scope,
                  graph=dict(runner='v2', mode=mode, capture_sizes=sizes,
                             enable_sp=cc.pass_config.enable_sp,
                             full_capture_sizes=full, piecewise_capture_sizes=piecewise,
                             implementation=('breakable' if manager.use_breakable_cg else 'compiled')))
    expected_full, expected_piecewise = expected_capture_grids(record['graph'], scope)
    query = cfg.num_speculative_tokens + 1
    # Token sets alone cannot distinguish dynamic-q, request-padding, LoRA or
    # microbatch variants. Refuse those rather than silently dropping axes.
    def descriptor_key(desc):
        return (desc.cg_mode.name, desc.num_tokens, desc.num_reqs,
                desc.uniform_token_count, desc.max_query_len,
                desc.num_active_loras, desc.num_ubatches)

    expected = set()
    for n in expected_full:
        reqs = min(n, scheduler.max_num_seqs) if mode == 'FULL' else n // query
        uniform = None if mode == 'FULL' else query
        expected.add(('FULL', n, reqs, uniform, None, 0, 1))
    expected.update(('PIECEWISE', n, None, None, None, 0, 1)
                    for n in expected_piecewise)
    planned = {descriptor_key(desc) for group in manager._capture_descs.values()
               for desc in group}
    if planned != expected:
        raise ValueError('Worker capture descriptors exceed the supported dispatch contract')
    actual_full = {descriptor_key(desc) for desc in manager.graphs}
    if actual_full != {desc for desc in expected if desc[0] == 'FULL'}:
        raise ValueError('Worker has not captured every expected FULL descriptor')
    return validate_execution(record)


async def capture_execution(engine):
    """Unsupported observations remain unknown without failing a benchmark."""
    import asyncio
    import json
    from pathlib import Path
    from profiler.core.execution_metadata import merge_workers
    try:
        model = getattr(getattr(getattr(engine, 'vllm_config', None), 'model_config', None), 'model', None)
        args = ()
        if isinstance(model, str):
            # Use the user's declaration, not a handwritten config-key whitelist.
            # Native workers resolve its aliases/defaults before binding it to
            # their effective config. Missing local declarations are not guessed.
            paths = (Path(model) / 'config.json',
                     Path(__file__).resolve().parents[2] / 'configs' / 'model' / (model + '.json'))
            for path in paths:
                if path.is_file():
                    args = (json.loads(path.read_text()),)
                    break
        client = getattr(engine, 'engine_core', None)
        identities = getattr(client, 'core_engines', None)
        call_one = getattr(client, '_call_utility_async', None)
        if isinstance(identities, (list, tuple)) and len(identities) > 1 and callable(call_one):
            # The load-balancing client's public collective RPC invokes every
            # DP engine but returns only the first engine's result. Preserve
            # each result here; merge_workers still requires global coverage,
            # so a client managing only some DP engines cannot claim success.
            workers = await asyncio.gather(*(
                call_one('collective_rpc', 'serving_execution_snapshot', 30, args, {},
                         engine=identity)
                for identity in tuple(identities)
            ), return_exceptions=True)
            for result in workers:
                if isinstance(result, BaseException):
                    raise result
        else:
            # Single-engine and unsupported client implementations retain the
            # public API. Partial DP responses fail the same coverage checks.
            workers = await engine.collective_rpc('serving_execution_snapshot', timeout=30, args=args)
        return merge_workers(workers)
    except Exception as exc:
        return {'status': 'unknown', 'reason': f'{type(exc).__name__}: {exc}'}
