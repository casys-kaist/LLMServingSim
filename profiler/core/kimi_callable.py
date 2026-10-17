"""Source-derived original93 AttnRes arguments, not a measurement-point grid."""
from collections import Counter
import math

from .kimi_profile import CHECKPOINT_SHA256, NATIVE_SOURCES, canonical, digest, roles


SIGNATURE = ('(prefix: torch.Tensor, delta: torch.Tensor | None, blocks: torch.Tensor, '
             'norm_weight: torch.Tensor, qk_weight: torch.Tensor, '
             'output_norm_weight: torch.Tensor | None, num_blocks: int, '
             'block_write_idx: int, eps: float, output_norm_eps: float) -> torch.Tensor')


def plan(config, tokens, *, iterations, warmups):
    """Describe full93 roles at an explicitly selected token point.

    No model engine is built and no prompt/point policy is selected. This
    source plan covers a standalone native function, not full93 execution.
    """
    from .hooks.capture_scope_contract import normalize_contexts
    from .hooks.callable_workload import normalize
    schedule = roles(config)
    fields = dict(num_blocks=dict(argument='num_blocks', operation='int'),
        block_write_idx=dict(argument='block_write_idx', operation='int'),
        delta_present=dict(argument='delta', operation='present'),
        output_norm_present=dict(argument='output_norm_weight', operation='present'))
    tensors = {name: dict(argument=name,
        **(dict(token_axis=0) if name in ('prefix', 'delta', 'blocks') else {}),
        **(dict(optional=True) if name in ('delta', 'output_norm_weight') else {}))
        for name in ('prefix', 'delta', 'blocks', 'norm_weight', 'qk_weight', 'output_norm_weight')}
    declaration = dict(name='attn_res', kind='function', module='vllm.models.kimi_k3.nvidia.model',
                       attribute='attn_res', context=dict(fields=fields, tensors=tensors))
    catalog = dict(attn_res=dict(callable='attn_res', normalization='invocations'))
    contexts = normalize_contexts(catalog=dict(attn_res=dict(catalog['attn_res'], callable=declaration)))
    slots = {name: dict(shape=([tokens, 8, 7168] if name == 'blocks' else
        [tokens, 7168] if name in ('prefix', 'delta') else [7168]), dtype='torch.bfloat16',
        fill=0.0 if name == 'qk_weight' else 0.125) for name in tensors}
    calls, counts, role_contexts = [], Counter(), []
    for role in schedule:
        absent = set()
        if not role['delta_present']:
            absent.add('delta')
        if not role['output_norm_present']:
            absent.add('output_norm_weight')
        call = {name: None if name in absent else dict(tensor=name) for name in tensors}
        call.update({name: role[name] for name in ('num_blocks', 'block_write_idx', 'eps', 'output_norm_eps')})
        calls.append(call)
        geometry = {}
        for name, spec in slots.items():
            if name in absent:
                geometry[name] = None
                continue
            shape = list(spec['shape'])
            stride = [math.prod(shape[index+1:]) for index in range(len(shape))]
            if name in ('prefix', 'delta', 'blocks'):
                shape[0] = '$tokens'
            geometry[name] = dict(shape=shape, stride=stride, dtype=spec['dtype'], storage_offset=0)
        context = canonical(dict(schema=1, arguments={name: role[name] for name in fields}, geometry=geometry))
        counts[context] += 1
        role_contexts.append(context)
    work = dict(schema=1, scope=dict(kind='kimi-original93-isolated-attn-res-v1',
        original_config_sha256=CHECKPOINT_SHA256, execution_depth=93, capacity=8,
        tp=1, pp=1, dtype='torch.bfloat16', device_capability=[12, 0]),
        source=dict(module=NATIVE_SOURCES['models/kimi_k3/nvidia/model.py'],
            function=NATIVE_SOURCES['models/kimi_k3/nvidia/ops/attn_res.py'], signature=digest(SIGNATURE)),
        shot_tokens=tokens, iterations=iterations, warmups=warmups, tensors=slots, calls=calls,
        expected_contexts=[dict(layer='attn_res', context=context, context_tokens=tokens, invocations=count)
                           for context, count in sorted(counts.items())])
    normalize(work, catalog, contexts)
    return work, catalog, contexts, role_contexts
