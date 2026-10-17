"""CPU source contracts for native Kimi-K3 profiling and local trace annotation.

This is not a grid, interpolation contract or resident-memory model. The
original checkpoint is immutable; only native engine construction takes a
prefix. Runtime/CSV completion and callable route evidence are separate gates.
"""
from __future__ import annotations

import copy
import hashlib
import json

from .stack import expert_top_k, resolve_stack

SCHEMA = "kimi-k3-native-local-v1"
CHECKPOINT_SHA256 = "2a5cb51c92f3b64e68f7670042e4d8cfff0acd4bcf1e3c5f7190d69609381de6"
NATIVE_SOURCES = {
    "models/kimi_k3/nvidia/model.py": "2b1d9629f810926a98e9ae05582afcceda02e372bb0a4afd7aa95c60d4931c74",
    "models/kimi_k3/nvidia/kda.py": "54c9939f1c11128fcdfa3cf8eaabec4e66471ec3183b5642b6509e09b5b8f8d6",
    "models/kimi_k3/nvidia/mla.py": "613d03689525acadbd3c9d271ede3615fe043ff9417dfb1230a7d86581cf336a",
    "models/kimi_k3/nvidia/ops/attn_res.py": "38c4d67cdb8366cd7b959305ddd251c6b9c9fa408e7481d3dac05ebf3b44028c",
}

# Transformers 5.17 configuration_utils.py:322-324 deliberately pops generation
# parameters instead of storing them in PreTrainedConfig. These are precisely
# the non-null fields removed from this pinned checkpoint by KimiK3Config.
# Source SHA256: 02826ac9e987487b645135d8f1d747f60a7e567d1b8f72327de8514d02b6f15d.
# Only missing fields qualify; changed values or another serializer still fail.
_REMOVED_GENERATION_FIELDS = frozenset((
    'diversity_penalty', 'do_sample', 'early_stopping',
    'encoder_no_repeat_ngram_size', 'length_penalty', 'max_length', 'min_length',
    'no_repeat_ngram_size', 'num_beam_groups', 'num_beams', 'num_return_sequences',
    'output_scores', 'remove_invalid_values', 'repetition_penalty',
    'return_dict_in_generate', 'temperature', 'top_k', 'top_p', 'typical_p',
))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def original_text(config):
    """Admit the reviewed original wrapper, not a reconstructed flat config."""
    if type(config) is not dict or digest(config) != CHECKPOINT_SHA256:
        raise ValueError("Bounded Kimi requires the reviewed original93 checkpoint configuration")
    text = config["text_config"]
    if expert_top_k(text) != 16:
        raise ValueError("Kimi top-k differs from its native16 slots")
    stack = resolve_stack(config)
    if (len(stack) != 93 or sum(s.attn == "linear_attention" for s in stack) != 69
            or sum(s.mlp == "moe" for s in stack) != 92):
        raise ValueError("Kimi original decoder schedule differs")
    return copy.deepcopy(text)


def materialize_prefix(config, overrides):
    """Materialize and validate a Kimi prefix before native config resolution.

    Generic shaping still owns final native config construction. User schedule
    overrides must describe the original depth; validate them before clipping.
    The source wrapper and the caller's override mapping are never modified.
    """
    result = copy.deepcopy(config)
    if "text_config" in overrides:
        raise ValueError("Kimi backbone overrides must be explicit top-level fields")
    resolve_stack(config)
    inner = result.get("text_config", result)
    source_depth = inner["num_hidden_layers"]
    for name, value in overrides.items():
        if name in inner or name in ("num_hidden_layers", "linear_attn_config",
                                      "layer_types", "full_attention_interval"):
            inner[name] = copy.deepcopy(value)
    depth = overrides.get("num_hidden_layers", source_depth)
    full = dict(inner, model_type="kimi_linear", num_hidden_layers=source_depth)
    resolve_stack(full, prefix_layers=depth)
    inner["num_hidden_layers"] = depth
    linear = inner.get("linear_attn_config")
    if linear is not None:
        for name in ("kda_layers", "full_attn_layers"):
            linear[name] = [index for index in linear[name] if index <= depth]
    if "layer_types" in inner:
        inner["layer_types"] = inner["layer_types"][:depth]
    return result


def roles(config, *, depth=93):
    """Source model.py:967-969,999-1060,1374-1386, in forward order."""
    text = original_text(config)
    if type(depth) is not int or not 1 <= depth <= 93:
        raise ValueError("Kimi context depth must be within the original93 layers")
    width = text["attn_res_block_size"]
    capacity = (depth + width - 1) // width
    result = []
    for index in range(depth):
        boundary = index % width == 0
        previous = (index + width - 1) // width
        common = dict(capacity=capacity, hidden=text["hidden_size"], eps=text["rms_norm_eps"])
        result.append(dict(layer=index, role="pre_attn", num_blocks=previous,
            block_write_idx=index // width if boundary else -1,
            delta_present=index != 0, output_norm_present=True,
            output_norm_eps=text["rms_norm_eps"], **common))
        result.append(dict(layer=index, role="post_attn", num_blocks=previous + int(boundary),
            block_write_idx=-1, delta_present=not boundary, output_norm_present=True,
            output_norm_eps=text["rms_norm_eps"], **common))
    result.append(dict(layer=None, role="final", num_blocks=capacity, block_write_idx=-1,
        delta_present=True, output_norm_present=False, output_norm_eps=0.0,
        capacity=capacity, hidden=text["hidden_size"], eps=text["rms_norm_eps"]))
    return result


def runtime_descriptor(config, depth, engine_limits, cache_bytes):
    original_text(config)
    if type(depth) is not int or not 2 <= depth <= 93:
        raise ValueError("Native Kimi prefix must include a routed layer within original93")
    fields = {"max_model_len", "max_num_seqs", "max_num_batched_tokens"}
    if (type(engine_limits) is not dict or set(engine_limits) != fields
            or any(type(n) is not int or n < 1 for n in engine_limits.values())
            or (cache_bytes is not None and (type(cache_bytes) is not int or cache_bytes < 1))):
        raise ValueError("Native Kimi requires positive engine limits and a positive or automatic cache budget")
    return dict(schema=SCHEMA, original_config=copy.deepcopy(config),
        original_config_sha256=CHECKPOINT_SHA256, engine_depth=depth,
        native_sources=dict(NATIVE_SOURCES), engine_limits=copy.deepcopy(engine_limits),
        cache_bytes=cache_bytes, tp=1, pp=1, ep=1, dp=1)


def validate_descriptor(descriptor):
    if type(descriptor) is not dict:
        raise ValueError("Kimi runtime descriptor must be plain data")
    expected = runtime_descriptor(descriptor.get("original_config"), descriptor.get("engine_depth"),
                                  descriptor.get("engine_limits"), descriptor.get("cache_bytes"))
    if canonical(descriptor) != canonical(expected):
        raise ValueError("Kimi runtime descriptor differs from its source-bound local scope")
    return expected


def admit_runtime(descriptor, facts):
    """Admit actual metadata; this does not certify a forward or acquired rows."""
    facts = copy.deepcopy(facts)
    # Preparation wall time is telemetry, not an execution invariant or GPU
    # latency. The full receipt remains in engine metadata. Retain every other
    # initialization fact, including source/layout pins and sampled-byte hashes.
    preparation = facts.get('dummy_weight_preparation') or {}
    preparation.pop('elapsed_preparation_seconds', None)
    for record in preparation.get('tensor_records', []):
        record.pop('elapsed_preparation_seconds', None)
    descriptor = validate_descriptor(descriptor)
    depth = descriptor["engine_depth"]
    text = original_text(descriptor["original_config"])
    expected = materialize_prefix(descriptor["original_config"], {"num_hidden_layers": depth})["text_config"]
    if (facts.get("schema") != SCHEMA or facts.get("vllm_version") != "0.31.0"
            or facts.get("native_sources") != NATIVE_SOURCES
            or facts.get("model_class") != "vllm.models.kimi_k3.nvidia.model.KimiK3ForConditionalGeneration"
            or facts.get("model_type") != "kimi_k3" or facts.get("text_type") != "kimi_linear"
            or facts.get("dtype") != "torch.bfloat16" or facts.get("device_capability") != [12, 0]
            or facts.get("quantization") != "mxfp4" or facts.get("enforce_eager") is not True):
        raise ValueError("Kimi actual model/source/device/precision differs")
    effective = facts["text_config"]
    # Native config adds defaults, drops legacy generation options and rewrites
    # the checkpoint quant_method. Architectural fields must still agree.
    for key, value in expected.items():
        if key in ("_name_or_path", "_attn_implementation_autoset", "transformers_version"):
            continue
        if (key in _REMOVED_GENERATION_FIELDS and key not in effective
                and effective.get('transformers_version') == '5.17.0'):
            continue
        if key == "quantization_config":
            value = dict(value, quant_method="mxfp4")
        if canonical(effective.get(key)) != canonical(value):
            raise ValueError("Kimi actual text config differs at " + key)
    parallel = facts.get("parallel", {})
    if (set(parallel) != {"tp", "pp", "ep", "dp", "pcp", "dcp"}
            or any(type(value) is not int or value != 1 for value in parallel.values())
            or facts.get("expert_parallel") is not False
            or facts.get("speculative") is not False
            or facts.get("prefix_caching") is not False
            or facts.get("mamba_cache_mode") != "none"
            or facts.get("multimodal_limits") != {"image": 0, "video": 0}
            or facts.get("resident_multimodal_tensors") != 0
            or facts.get("cpu_offload_gb") != 0
            or facts.get("prefetch_group_size") != 0
            or facts.get("kv_transfer") is not False
            or facts.get("kv_offload") is not None):
        raise ValueError("Kimi actual execution mode is outside the reviewed local scope")
    if canonical(facts.get("limits")) != canonical(descriptor["engine_limits"]):
        raise ValueError("Kimi actual engine envelope differs")
    if facts.get("kv_cache_memory_bytes") != descriptor["cache_bytes"]:
        raise ValueError("Kimi actual cache budget differs")
    from .hooks.runtime_admission import SAMPLER_ROUTES
    sampler = facts.get("sampler", {})
    route = SAMPLER_ROUTES.get((sampler.get("runner_class"), sampler.get("sampler_class")))
    if (route is None or sampler.get("sampler_sources") != route["sources"]
            or sampler.get("sampler_entrypoint") != route["entrypoint"]
            or sampler.get("sampled_token_dtype") != route["dtype"]):
        raise ValueError("Kimi native sampler return-dtype route is not admitted")
    stack = resolve_stack(descriptor["original_config"], prefix_layers=depth)
    layers = facts.get("layers", [])
    if len(layers) != depth:
        raise ValueError("Kimi actual decoder prefix differs")
    for index, (layer, spec) in enumerate(zip(layers, stack)):
        attention = "KimiK3DeltaAttention" if spec.attn == "linear_attention" else "MultiHeadLatentAttention"
        if (canonical(layer) != canonical(dict(index=index, attention=attention, is_moe=spec.mlp == "moe",
                          hidden=7168, use_attn_res=True, sequence_parallel=False))):
            raise ValueError("Kimi actual layer role differs at " + str(index))
    if canonical(facts.get("backbone")) != canonical(dict(start=0, end=depth, residual_capacity=(depth+11)//12)):
        raise ValueError("Kimi actual residual prefix geometry differs")
    preparation = facts.get("dummy_weight_preparation") or {}
    contracts = preparation.get("actual_selected_backend_contracts", {})
    paths = {f"language_model.model.layers.{i}.block_sparse_moe.experts.routed_experts" for i in range(1, depth)}
    if (preparation.get("policy") != "native-mxfp4-dummy-bf16-cuda-row-chunks-v1"
            or preparation.get("complete") is not True or preparation.get("forward_verified") is not False
            or preparation.get("postload", {}).get("structure_verified") is not True
            or set(contracts) != paths):
        raise ValueError("Kimi native initialization/postload structure is incomplete")
    for contract in contracts.values():
        if (contract.get("backend") != "DEEPGEMM_MXFP4" or contract.get("backend_forced") is not False
                or [contract.get(key) for key in ("physical_experts", "logical_hidden", "logical_intermediate")]
                != [text["num_experts"], text["routed_expert_hidden_size"], text["moe_intermediate_size"]]):
            raise ValueError("Kimi selected native expert layout differs")
    return dict(schema=SCHEMA, descriptor_sha256=digest(descriptor),
                facts=copy.deepcopy(facts), facts_sha256=digest(facts), forward_verified=False)
