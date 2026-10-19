"""Resolve native attention dimensions before single-GPU TP emulation.

This acquisition-only adapter loads config objects, never model weights or
process groups. It preserves vLLM's head width while the host shrinks heads.
"""
from __future__ import annotations

import copy


def _converter(hf_config):
    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS, ModelArchConfigConvertorBase,
    )
    text = hf_config.get_text_config()
    cls = MODEL_ARCH_CONFIG_CONVERTORS.get(hf_config.model_type,
                                          ModelArchConfigConvertorBase)
    return cls(hf_config, text), text


def preserve_resolved_head_dim(original_hf_config, tp_overrides, *, reload_config):
    """Preserve native scalar head width, refusing an unaudited special case.

    User shape overrides must already be resolved in the original config.
    ``reload_config`` must materialize the raw configuration and reload it
    through vLLM, exactly as engine creation does. Updating a constructed
    config retains derived defaults and cannot validate a new config file.
    Neither the original config nor the TP override dictionary is mutated.
    """
    # Initialize the public package first: converter-first imports are circular
    # in the supported vLLM releases.
    from vllm.config import ModelConfig  # noqa: F401
    from vllm.transformers_utils.model_arch_config_convertor import ModelArchConfigConvertorBase

    original = copy.deepcopy(original_hf_config)
    before_converter, before_text = _converter(original)
    before = before_converter.get_head_size()
    result = copy.deepcopy(tp_overrides)
    modified = reload_config(result)
    after_converter, _ = _converter(modified)
    after = after_converter.get_head_size()
    evidence = dict(before_head_size=before, uncorrected_head_size=after,
                    corrected_head_size=after, added_override=False,
                    converter=type(before_converter).__name__)
    if before == after:
        return result, evidence
    if type(before) is not int or before < 1:
        raise ValueError('Cannot preserve a nonpositive/noninteger resolved head size')
    if before_converter.is_deepseek_mla() or after_converter.is_deepseek_mla():
        raise ValueError('TP overrides changed MLA head/cache size; generic head_dim pin is unsafe')
    if (before_converter.get_per_layer_hf_configs() is not None
            or after_converter.get_per_layer_hf_configs() is not None):
        raise ValueError('TP overrides changed a heterogeneous head size; scalar pin is unsafe')
    if (type(before_converter).get_head_size is not ModelArchConfigConvertorBase.get_head_size
            or type(after_converter).get_head_size is not ModelArchConfigConvertorBase.get_head_size):
        raise ValueError('Specialized architecture head-size converter requires explicit support')
    if not hasattr(before_text, 'head_dim'):
        raise ValueError('Resolved config does not declare the head_dim interface')
    if 'head_dim' in result:
        raise ValueError('Resolve explicit head_dim changes before TP emulation')
    result['head_dim'] = before
    modified = reload_config(result)
    if _converter(modified)[0].get_head_size() != before:
        raise ValueError('Pinned head_dim does not preserve native resolved head size')
    evidence.update(corrected_head_size=before, added_override=True)
    return result, evidence


def resolved_tp_overrides(model_config, unsharded_overrides, tp, shard_fields,
                          materialize):
    """Load the local native config, then shard its resolved unsharded values.

    Use the engine's nested/depth-prefix materialization without constructing
    an engine or contacting the hub. Divisibility remains strict; this helper
    does not emulate replicated KV heads or arbitrary architecture dimensions.
    """
    import json
    from pathlib import Path
    import tempfile
    from vllm.transformers_utils.config import get_config

    if type(tp) is not int or tp < 1:
        raise ValueError('TP must be a positive integer')

    def load_native(extra_overrides):
        # Materialization consumes nested overrides. Start every candidate
        # from fresh inputs, never the previous candidate's forwarded kwargs
        # or a resolved config whose derived defaults are already populated.
        config_kwargs = {'hf_overrides': {
            **copy.deepcopy(unsharded_overrides), **copy.deepcopy(extra_overrides)}}
        written = materialize(copy.deepcopy(model_config), config_kwargs)
        with tempfile.TemporaryDirectory(prefix='profiler_shape_') as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(written))
            resolved = get_config(path, trust_remote_code=False, config_format='hf',
                                  hf_overrides_kw=config_kwargs['hf_overrides'],
                                  local_files_only=True)
        return resolved, written

    original, written = load_native({})
    text = original.get_text_config()
    raw = written.get('text_config')
    declared = raw if isinstance(raw, dict) else written
    fields = {}
    for name in shard_fields:
        if name not in declared and name not in unsharded_overrides:
            continue
        value = getattr(text, name)
        if type(value) is not int or value < 1:
            raise TypeError(f'shard field {name!r} must be a positive int; got {value!r}')
        if value % tp:
            raise ValueError(f'model config field {name!r}={value} is not divisible by tp={tp}')
        fields[name] = value // tp

    def reload_config(overrides):
        resolved, _ = load_native(overrides)
        text = resolved.get_text_config()
        for name, value in overrides.items():
            if getattr(text, name, None) != value:
                raise ValueError(f'Native configuration did not preserve TP override {name!r}={value!r}')
        return resolved

    return preserve_resolved_head_dim(original, fields, reload_config=reload_config)[0]
