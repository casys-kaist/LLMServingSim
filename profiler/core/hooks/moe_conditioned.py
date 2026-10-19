"""Measure complete native expert reuse cycles and retain both controls."""
from collections import Counter
import hashlib
from pathlib import Path

import torch

from ..moe_conditioning import expert_conditioning
from .eager_measurement import extract_eager_forwards
from .moe_expert_region import NativeExpertRegion
from .skew_measurement import MARKER


@torch.inference_mode()
def measure_experts(owner, point, iterations, failure_dir=None):
    if point['mode'] != 'eager':
        raise ValueError('Native MoE component acquisition requires eager mode')

    from .moe_components import ProfileCall

    n, active, pairs = (point[k] for k in ('tokens','activated_experts','local_assignments'))
    plan = expert_conditioning(owner.contract, active, iterations)
    maximum = max(plan['bank_counts'])
    # The allocation guard includes one extra conversion-sized buffer and
    # an independent output reference. Never relax it after an OOM.
    additional = maximum-len(owner.expert_banks)
    reserve = max(2**30, 8*n*owner.cfg.experts_per_token*owner.cfg.hidden_dim*4)
    free, total = torch.cuda.mem_get_info(owner.device)
    live = owner.contract['expert_region']['schema'] == 'native-live-expert-region'
    bank_bytes = owner.contract['expert_bank_bytes'] if live else owner.contract['expert_weight_bytes']
    # A live bank clones already processed native tensors once, without a
    # second conversion buffer. Account all persistent scales in its footprint
    # and recheck free memory before each bank. The legacy conversion path
    # keeps its original double-buffer and quarter-device guards.
    needed = max(0,additional)*bank_bytes*(1 if live else 2)+reserve
    if needed > free or (not live and max(0,additional)*owner.contract['expert_weight_bytes'] > total//4):
        raise MemoryError('Native expert conditioning exceeds safe free GPU memory; lower engine memory utilization or use a larger GPU')
    for _ in range(additional):
        if live:
            free, _ = torch.cuda.mem_get_info(owner.device)
            if bank_bytes+reserve > free:
                raise MemoryError('Native processed weight bank plus workspace reserve exceeds free GPU memory')
            owner.expert_banks.append(owner.expert.clone())
        else:
            w13,w2 = (v.clone() for v in owner.checkpoint_weights)
            owner.expert_banks.append(NativeExpertRegion(owner.cfg,w13,w2))
    generator = torch.Generator(device=owner.device).manual_seed(n)
    hidden = torch.randn((n,owner.cfg.hidden_dim),device=owner.device,
                         dtype=owner.cfg.in_dtype,generator=generator)
    routes = owner.place.balanced_route(n,active,pairs)
    base_ids = torch.tensor(routes,device=owner.device,dtype=owner.indices_dtype)
    local = torch.tensor(owner.place.expert_map,device=owner.device)[base_ids.long()] >= 0
    first = owner.place.local_global_ids[0]
    ids = [torch.where(local,(base_ids-first+offset)%owner.cfg.num_local_experts+first,base_ids)
           for offset in plan['offsets']]
    weights = torch.full(base_ids.shape,1/owner.cfg.experts_per_token,device=owner.device,dtype=torch.float32)
    histogram = Counter(e for r in routes for e in r if owner.place.expert_map[e]>=0)
    if len(histogram)!=active or sum(histogram.values())!=pairs:
        raise ValueError('Conditioned routing changed the requested local work')
    # NoDPEP writes the already-reduced expert result directly into the
    # caller's output. A distinct final buffer would add a non-native copy.
    output_alias = torch.empty_like(hidden) if owner.target.dp == 1 and not live else None
    calls = [[ProfileCall(bank.bind(hidden,route,weights,output_alias=output_alias)) for route in ids]
             for bank in owner.expert_banks[:maximum]]
    for bank_calls in calls:
        for r,call in enumerate(bank_calls):
            # Clone before another bank reuses the native output workspace.
            reference = calls[0][r]().clone()
            for _ in range(3): result=call()
            if output_alias is not None and result is not output_alias:
                raise ValueError('Native DP1 expert did not preserve its output alias')
            if not torch.isfinite(reference).all().item():
                raise ValueError('Non-finite native expert reference')
            torch.testing.assert_close(result,reference,atol=0,rtol=0)
    results=[]
    for arm,banks in enumerate(plan['bank_counts']):
        cycle = plan['cycles'][arm]
        count = plan['forward_counts'][arm]
        def choose(index):
            position=index%cycle
            return calls[position%banks][position//banks]
        for j in range(plan['warmup_forwards']): choose(j)()
        torch.cuda.synchronize()
        context=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                 torch.profiler.ProfilerActivity.CUDA])
        with context as hook:
            for j in range(count):
                with torch.profiler.record_function(MARKER+str(j)):
                    choose(j)()
        torch.cuda.synchronize()
        try:
            raw=extract_eager_forwards(hook.profiler.kineto_results,count)
        except Exception as exc:
            if failure_dir is not None:
                import tempfile
                Path(failure_dir).mkdir(parents=True,exist_ok=True)
                path=Path(tempfile.mkdtemp(prefix='conditioned-',dir=failure_dir))/'trace.json'
                hook.export_chrome_trace(str(path))
                if hasattr(exc,'audit'):
                    import json
                    path.with_name('native_ownership.json').write_text(json.dumps(exc.audit,indent=2))
                raise ValueError(f'{exc}; original CUDA trace: {path}') from exc
            raise
        results.append(dict(raw,verified=True))
    geometry=dict(global_top_k=owner.cfg.experts_per_token,
        local_histogram=[histogram[e] for e in owner.place.local_global_ids],
        routing_sha256=hashlib.sha256(base_ids.cpu().numpy().tobytes()).hexdigest(),
        conditioning=plan,independent_outputs_verified=True)
    return dict(results[1],geometry=geometry,conditioning_control=results[0])
