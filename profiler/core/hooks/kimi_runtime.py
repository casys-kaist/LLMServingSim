"""Untimed, read-only native Kimi prefix admission through a named RPC."""
import copy
import hashlib
import inspect
from pathlib import Path

from profiler.core.kimi_profile import NATIVE_SOURCES, SCHEMA, admit_runtime, validate_descriptor


def inspect_runtime(runner, descriptor):
    import torch
    import vllm
    from .dummy_weights import preparation_facts
    from .runtime_admission import _sampler_facts, _sampler_route

    descriptor = validate_descriptor(descriptor)
    root = Path(vllm.__file__).resolve().parent
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in NATIVE_SOURCES}
    if sources != NATIVE_SOURCES:
        raise ValueError("Installed Kimi source differs from reviewed native source")
    model = runner.get_model()
    model_source = root / "models/kimi_k3/nvidia/model.py"
    if Path(inspect.getfile(type(model))).resolve() != model_source:
        raise ValueError("Actual Kimi model was not constructed from the reviewed source")
    cfg = runner.vllm_config
    parallel, cache = cfg.parallel_config, cfg.cache_config
    modules = list(model.named_modules())
    layers, backbones = [], []
    for _, module in modules:
        name = type(module).__name__
        if name == "KimiDecoderLayer":
            if Path(inspect.getfile(type(module))).resolve() != model_source:
                raise ValueError("Foreign Kimi decoder implementation")
            layers.append(dict(index=module.layer_idx, attention=type(module.self_attn).__name__,
                is_moe=module.is_moe_layer, hidden=module.hidden_size,
                use_attn_res=module.use_attn_res, sequence_parallel=module.use_sequence_parallel))
        elif name == "KimiLinearModel":
            backbones.append(dict(start=module.start_layer, end=module.end_layer,
                                  residual_capacity=module.num_attn_res_blocks))
    if len(backbones) != 1:
        raise ValueError("Expected exactly one Kimi text backbone")
    resident_mm = 0
    for name in ("vision_tower", "mm_projector"):
        tower = getattr(model, name)
        if tower is not None:
            resident_mm += sum(not tensor.is_meta for tensor in
                               [*tower.parameters(), *tower.buffers()])
    mm = cfg.model_config.get_multimodal_config()
    sampler = _sampler_facts(runner)
    sampler_route = _sampler_route(sampler["runner_class"], sampler["sampler_class"])
    if sampler["sampler_sources"] != sampler_route["sources"]:
        raise ValueError("Kimi native sampler source differs from the reviewed return-dtype route")
    sampler["sampled_token_dtype"] = sampler_route["dtype"]
    facts = dict(schema=SCHEMA, vllm_version=vllm.__version__, native_sources=sources,
        model_class=type(model).__module__ + "." + type(model).__name__,
        model_type=cfg.model_config.hf_config.model_type,
        text_type=cfg.model_config.hf_text_config.model_type,
        text_config=copy.deepcopy(cfg.model_config.hf_text_config.to_dict()),
        dtype=str(cfg.model_config.dtype), device_capability=list(torch.cuda.get_device_capability()),
        quantization=cfg.model_config.quantization, enforce_eager=cfg.model_config.enforce_eager,
        parallel=dict(tp=parallel.tensor_parallel_size, pp=parallel.pipeline_parallel_size,
            dp=parallel.data_parallel_size, pcp=parallel.prefill_context_parallel_size,
            dcp=parallel.decode_context_parallel_size,
            ep=parallel.tensor_parallel_size * parallel.data_parallel_size if parallel.enable_expert_parallel else 1),
        expert_parallel=parallel.enable_expert_parallel,
        speculative=cfg.speculative_config is not None,
        prefix_caching=cache.enable_prefix_caching, mamba_cache_mode=cache.mamba_cache_mode,
        multimodal_limits={name: mm.get_limit_per_prompt(name) for name in ("image", "video")},
        resident_multimodal_tensors=resident_mm,
        cpu_offload_gb=cfg.offload_config.uva.cpu_offload_gb,
        prefetch_group_size=cfg.offload_config.prefetch.offload_group_size,
        kv_transfer=cfg.kv_transfer_config is not None,
        kv_offload=cache.kv_offloading_size,
        sampler=sampler,
        limits=dict(max_model_len=cfg.model_config.max_model_len,
            max_num_seqs=cfg.scheduler_config.max_num_seqs,
            max_num_batched_tokens=cfg.scheduler_config.max_num_batched_tokens),
        kv_cache_memory_bytes=cache.kv_cache_memory_bytes,
        layers=sorted(layers, key=lambda row: row["index"]), backbone=backbones[0],
        dummy_weight_preparation=preparation_facts(model))
    from .kimi_inventory import observe
    facts['logical_sizes'] = observe(runner, facts['dummy_weight_preparation'])
    return admit_runtime(descriptor, facts)
