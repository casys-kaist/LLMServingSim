"""Native MoE component ordering and deployment-aware table selection."""
from pathlib import Path

from profiler.core.moe_deployment import MoeTarget
from .communication import dtype_bytes
from .config_builder import get_device
from .logger import get_logger
from .moe_components import MoeComponentTable, MoeCoverageError, component_index
from .power_model import total_ring_data

_warned = set()
_log = get_logger('MoeComponents')


def _warn(ctx, reason):
    key = (ctx.perf_db['root'], ctx.tp_size, ctx.ep_total, reason)
    if key not in _warned:
        _warned.add(key)
        _log.warning('Native MoE profile unavailable: %s. Falling back to the legacy '
                     'EP table; DP/backend matching is not verified and accuracy may degrade.', reason)


def _legacy_fallback(ctx, reason, *, warn=True):
    """Only retained whole-block data can provide historical fallback."""
    if 'legacy_moe_present' not in ctx.perf_db:
        tps = ctx.perf_db['available_tps']
        legacy_tp = 1 if 1 in tps else tps[0]
        ctx.perf_db['legacy_moe_present'] = (
            Path(ctx.perf_db['root']) / f'tp{legacy_tp}' / 'moe.csv').is_file()
    if not ctx.perf_db['legacy_moe_present']:
        raise MoeCoverageError('Native MoE components are required for this bundle: '
                               + reason + '. Complete a matching native MoE acquisition.')
    if warn:
        _warn(ctx, reason)
    return False


def _collective(kind, dims, size):
    suffix = '' if dims is None else ':'+','.join(str(int(v)) for v in dims)
    return f'{kind}{suffix} {size}'


def emit_native_components(ctx, bctx, lines, power_acc, layer_num, batch_id, batch_tag):
    root = Path(ctx.perf_db['root'])/f'tp{ctx.tp_size}'
    available = ctx.perf_db.setdefault('native_moe_index_present', {})
    if ctx.tp_size not in available:
        available[ctx.tp_size] = component_index(root) is not None
    if not available[ctx.tp_size]:
        return _legacy_fallback(ctx, 'no native component table is installed',
                                warn=bool(ctx.dp_sum_total_len))
    counts = ctx.dp_token_counts
    if not counts:
        if ctx.dp_sum_total_len:
            raise ValueError('Native DP+EP needs the complete padded DP token vector')
        counts = (bctx.total_len,)
    elif (sum(counts) != ctx.dp_sum_total_len or min(counts) != ctx.dp_min_total_len
            or counts[ctx.dp_rank] != bctx.total_len):
        raise ValueError('MoE token vector differs from this DP wave or sub-batch')
    dp = len(counts)
    if ctx.local_ep != ctx.tp_size or ctx.ep_total != ctx.tp_size*dp:
        if dp > 1:
            raise ValueError('Native MoE requires full TP*DP expert parallelism')
        return _legacy_fallback(ctx, 'native tables cover full TP*DP expert parallelism, not tensor-sharded experts')
    entries = ctx.perf_db['architecture']['catalog'].get('moe') or {}
    if len(entries) != 1:
        raise ValueError('Native MoE needs one unambiguous catalog component binding')
    entry = next(iter(entries.values()))
    if ctx.tp_size > 1 and dp > 1 and type(entry.get('sequence_parallel')) is not bool:
        raise ValueError('TP+DP MoE needs an audited sequence-parallel catalog flag')
    sp = bool(ctx.tp_size > 1 and dp > 1 and entry.get('sequence_parallel'))
    expected = dict(model=ctx.model, hardware=ctx.hardware, variant=ctx.perf_db['variant'],
                    global_experts=ctx.gate.E, global_top_k=ctx.gate.k,
                    hidden_dim=ctx.config['hidden_size'],
                    intermediate_size=(ctx.config.get('moe_intermediate_size')
                                       or ctx.config['intermediate_size']))
    cache = ctx.perf_db.setdefault('native_moe_components', {})
    tables, domains = [], []
    for rank in range(ctx.dp_rank*ctx.local_ep, (ctx.dp_rank+1)*ctx.local_ep):
        target = MoeTarget(ctx.tp_size, dp, ctx.ep_total, rank, sp)
        key = (ctx.tp_size, dp, rank, sp)
        if key not in cache:
            try:
                cache[key] = MoeComponentTable(root, target=vars(target), expected=expected)
            except MoeCoverageError:
                # Bundles are immutable during a simulation. Cache absent
                # coverage too; otherwise every layer rereads the same metadata.
                # Invalid contracts still raise instead of becoming a fallback.
                cache[key] = None
        if cache[key] is None:
            return _legacy_fallback(ctx, f'TP={ctx.tp_size}, DP={dp}, EP={ctx.ep_total}, rank={rank}')
        tables.append(cache[key])
        domains.append(target.token_domains(counts))
    domain = domains[0]
    if any(d != domain for d in domains):
        raise ValueError('Local TP ranks disagree on the modeled token domains')
    if (ctx.gate.routing_policy not in ('BALANCED', 'CUSTOM')
            and any(not table.independent_assignments for table in tables)):
        return _legacy_fallback(ctx, f'policy={ctx.gate.routing_policy} requires independently measured assignment counts')
    mode_value = getattr(bctx.batch, 'cudagraph_mode', 0)
    if mode_value not in (0,1,2):
        raise ValueError('Unknown model-forward execution mode')
    local, gathered = domain['gate_rows'], domain['expert_rows']
    routing = ctx.gate.route_ep(layer_num, batch_id, gathered, ctx.ep_total, native=True)
    h, k = expected['hidden_dim'], expected['global_top_k']
    weight_loc = get_device(ctx.placement, layer_num, 'moe', 'weights')
    parts = []
    payloads = None
    components = tables[0].components
    if any(table.components != components for table in tables):
        raise ValueError('MoE ranks disagree on their native component boundaries')
    for table in tables:
        contract = table.contract
        if dtype_bytes(contract['input_dtype']) != ctx.fp:
            raise ValueError('Native MoE activation dtype differs from runtime')
        this_payloads = (h*ctx.fp, k*dtype_bytes(contract['topk_weights_dtype']),
                         k*dtype_bytes(contract['topk_ids_dtype']))
        if payloads is not None and this_payloads != payloads:
            raise ValueError('MoE ranks disagree on dispatch tensor dtypes')
        payloads = this_payloads
        pairs = routing.local_assignments[table.target.rank]
        active = routing.activated_experts[table.target.rank]
        mode = table.execution_mode(bool(mode_value))
        times = tuple(table.expert_ns(mode, gathered, active, pairs) if name == 'experts'
                      else table.local_ns(name, mode, local) for name in components)
        gate_bytes = contract.get('gate_weight_bytes')
        expert_bytes = contract.get('expert_weight_bytes')
        if type(gate_bytes) is not int or type(expert_bytes) is not int:
            raise ValueError('Native MoE profile must record measured component weight sizes')
        weights = dict(gate_routing=gate_bytes, experts=expert_bytes,
                       finalize_copy=0, wrapper=contract.get('wrapper_weight_bytes', 0))
        widths = dict(gate_routing=contract.get('gate_hidden_dim', h),
                      experts=contract.get('expert_hidden_dim', h), finalize_copy=h, wrapper=h)
        parts.append((times, tuple(weights[name] for name in components),
                      tuple(widths[name] for name in components)))
    group_size = ctx.ep_total if sp else dp
    if dp == 1:
        dispatch_dims = None
    elif sp or ctx.tp_size == 1:
        dispatch_dims = ctx.ep_dim
    else:
        if ctx.ep_dim is None or ctx.tp_dim is None or len(ctx.ep_dim) != len(ctx.tp_dim):
            raise ValueError('Non-SP DP communication needs explicit topology dimensions')
        dispatch_dims = [e and not t for e,t in zip(ctx.ep_dim,ctx.tp_dim)]
        if not any(dispatch_dims):
            raise ValueError('DP collective has no network dimension')
    chunks = domain['dispatch_rows']
    # Retain the existing analytical ragged Ring envelope. This is an
    # equivalent network payload, not a claim that NCCL packs the tensors.
    dispatch_sizes, combine_size = [], 0
    dispatch = combine = 'NONE 0'
    if dp > 1:
        remote_rows = gathered-min(chunks)
        dispatch_sizes = [(remote_rows*b+group_size-2)//(group_size-1) for b in payloads]
        combine_size = ((remote_rows*h*ctx.fp+group_size-2)//(group_size-1))*group_size
        dispatch = ' '.join(_collective('ALLGATHER',dispatch_dims,size) for size in dispatch_sizes)
        combine = _collective('REDUCESCATTER',dispatch_dims,combine_size)
    restore = 'NONE 0'
    if ctx.tp_size > 1:
        restore = _collective('ALLGATHER' if sp else 'ALLREDUCE',ctx.tp_dim,local*h*ctx.fp)

    # DP1 has no dispatch/combine or finalize copy. The final expert marker
    # still carries its optional TP reduction and preserves PP boundaries.
    phases = [(0,'moe_gate_routing',dispatch), (1,'moe_experts',restore)]
    if dp > 1:
        phases = [(0,'moe_gate_routing',dispatch), (1,'moe_experts',combine),
                  (2,'moe_finalize_copy',restore)]
    elif 'wrapper' in components:
        # Wrapper activities can surround the routed call in the live model.
        # This serial replay preserves their summed native CUDA activity duration;
        # the boundary does not claim overlap or allocator-accurate lifetimes.
        phases = [(0,'moe_gate_routing',dispatch), (1,'moe_experts','NONE 0'),
                  (2,'moe_wrapper',restore)]
    for phase, name, end in phases:
        for local_rank, (times, weights, widths) in enumerate(parts):
            lines.append((f'EXPERT {local_rank} NONE 0',))
            rows = gathered if phase == 1 else local
            activation_bytes = rows*widths[phase]*ctx.fp
            lines.append((name,str(times[phase]),'LOCAL',str(activation_bytes),weight_loc,
                          str(weights[phase]),'LOCAL',str(activation_bytes),'NONE','0',batch_tag))
        lines.append((f'EXPERT END {end}',))
    if power_acc is not None:
        power_acc.npu_latencies_ns.append(max(sum(p[0]) for p in parts))
        if weight_loc != 'LOCAL':
            power_acc.dram_weight_bytes += sum(sum(weights) for _,weights,_ in parts)
        power_acc.link_data_bytes += sum(total_ring_data(s,group_size,collective='allgather') for s in dispatch_sizes)
        if dp > 1:
            power_acc.link_data_bytes += total_ring_data(combine_size,group_size,collective='reducescatter')
        if ctx.tp_size > 1:
            power_acc.link_data_bytes += total_ring_data(local*h*ctx.fp,ctx.tp_size,
                collective='allgather' if sp else 'allreduce')
    return True
