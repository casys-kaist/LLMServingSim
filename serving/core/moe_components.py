"""Strict, cached lookup of deployment-matched native MoE components.

The expert surface describes balanced local assignment totals, not an
arbitrary routing histogram. The runtime selects a matching published table;
unsupported coverage is distinct from a corrupt measurement contract.
"""
from bisect import bisect_left
from collections import defaultdict
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

from profiler.core.moe_deployment import MoeTarget
from profiler.core.moe_geometry import ExpertPlacement
from .communication import dtype_bytes


class MoeCoverageError(ValueError):
    pass


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _linear(pairs, x):
    if not pairs[0][0] <= x <= pairs[-1][0]:
        raise MoeCoverageError(f'MoE coordinate {x} is outside measured [{pairs[0][0]}, {pairs[-1][0]}]')
    index = bisect_left([p[0] for p in pairs], x)
    if pairs[index][0] == x:
        return pairs[index][1]
    (a, fa), (b, fb) = pairs[index-1:index+1]
    return fa+(fb-fa)*(x-a)/(b-a)


class MoeComponentTable:
    def __init__(self, tp_folder, *, target, expected):
        target = MoeTarget(**target)
        root = Path(tp_folder).resolve()
        index = json.loads((root/'moe_components.json').read_text())
        if index['schema'] != 'moe-components-v1':
            raise ValueError('Unknown MoE component index schema')
        matching = [e for e in index['entries'] if e['target'] == vars(target)]
        if not matching:
            raise MoeCoverageError('A matching TP/DP/EP/rank/sequence-parallel contract is required')
        if len(matching) != 1:
            raise ValueError('Duplicate native MoE deployment contracts in the component index')
        entry = matching[0]
        folder = (root/entry['path']).resolve()
        if not folder.is_relative_to(root):
            raise ValueError('MoE component path leaves its TP bundle')
        contract = json.loads((folder/'contract.json').read_text())
        coverage = json.loads((folder/'coverage.json').read_text())
        identity = _hash(contract)
        if (contract['schema'] != index['schema'] or coverage['schema'] != index['schema']
                or contract['target'] != vars(target) or entry['contract_id'] != identity
                or coverage['contract_id'] != identity or coverage['complete'] is not True):
            raise ValueError('Incomplete or inconsistent MoE component contract')
        if contract.get('graph_attribution') != 'native_graph_launch_correlation_v1':
            raise ValueError('Graph data needs launch-correlated acquisition; module-tree graph timings are not supported')
        if ((contract.get('expert_conditioning') or {}).get('schema') != 'rotated-weight-cycles-v2'
                or coverage.get('quality_supported') is not True
                or contract.get('eager_attribution') != 'native_eager_launch_correlation_v1'
                or contract.get('graph_memory_pool') != 'independent_per_variant'
                or contract.get('graph_output_reference') != 'independent_pre_capture_clone'):
            raise ValueError('Native MoE data requires matched-work conditioning controls, native launch attribution and independently verified graph variants')
        required = {'model', 'hardware', 'variant', 'global_experts', 'global_top_k', 'hidden_dim'}
        if not required.issubset(expected) or any(contract.get(k) != v for k, v in expected.items()):
            raise ValueError('Model, hardware or tensor geometry differs from the measured MoE contract')
        placement = ExpertPlacement(contract['global_experts'],contract['global_top_k'],
                                    target.ep,target.rank)
        if (target.dp < 2 or contract.get('placement') != 'linear'
                or contract.get('expert_map') != list(placement.expert_map)
                or contract.get('local_experts') != len(placement.local_global_ids)):
            raise ValueError('MoE expert ownership differs from the target rank')
        parallel = contract['expert_parallel_config']
        expected_parallel = dict(tp_size=1,tp_rank=0,pcp_size=1,pcp_rank=0,
            dp_size=target.dp,dp_rank=target.dp_rank,ep_size=target.ep,ep_rank=target.rank,
            sp_size=target.tp if target.sequence_parallel else 1,use_ep=True,
            all2all_backend='allgather_reducescatter',enable_eplb=False)
        if any(parallel.get(k) != v for k,v in expected_parallel.items()):
            raise ValueError('MoE backend or parallel configuration differs from the supported execution path')
        region = contract['expert_region']
        if (region.get('schema') != 'native-modular-expert-region-v1'
                or region.get('expert_parallel_config') != parallel
                or any(region.get(k) != contract[k] for k in
                       ('global_experts','global_top_k','local_experts','expert_map',
                        'hidden_dim','intermediate_size','activation','input_dtype'))
                or not region.get('backend') or not region.get('expert_class')):
            raise ValueError('Native expert kernel contract and component table disagree')
        if (contract['topk_weights_dtype'] != 'torch.float32'
                or contract['topk_ids_dtype'] not in ('torch.int32','torch.int64')
                or contract['input_dtype'] not in ('torch.float16','torch.bfloat16','torch.float32')
                or contract.get('gate_weight_bytes') != contract['global_experts']*contract['hidden_dim']*dtype_bytes(contract['input_dtype'])
                or type(contract.get('expert_weight_bytes')) is not int
                or contract['expert_weight_bytes'] <= 0
                or contract['expert_weight_bytes'] % contract['local_experts']):
            raise ValueError('Invalid native component dtype or measured weight-byte contract')
        table = folder/'components.csv'
        checksum = hashlib.sha256(table.read_bytes()).hexdigest()
        if checksum != entry['table_sha256'] or checksum != coverage['table_sha256']:
            raise ValueError('MoE component CSV checksum mismatch')
        self.local, self.expert, self.zero = defaultdict(list), defaultdict(list), defaultdict(list)
        self.exact, plan = {}, []
        with table.open() as stream:
            for row in csv.DictReader(stream):
                component, mode = row['component'], row['mode']
                n, a, p = (int(row[k]) for k in ('tokens', 'activated_experts', 'local_assignments'))
                us = float(row['time_us'])
                key = (component, mode, n, a, p)
                if (component not in ('gate_routing', 'experts', 'finalize_copy')
                        or mode not in ('eager', 'graph') or n < 1
                        or not math.isfinite(us) or us <= 0 or key in self.exact):
                    raise ValueError('Invalid or duplicate native MoE coordinate')
                self.exact[key] = us
                plan.append(dict(component=component, mode=mode, tokens=n,
                                 activated_experts=a, local_assignments=p))
                if component != 'experts':
                    if (a, p) != (-1, -1):
                        raise ValueError('Local component has an expert coordinate')
                    self.local[component, mode].append((n, us))
                elif a == p == 0:
                    self.zero[mode].append((n, us))
                else:
                    balanced = (n*contract['global_top_k']*contract['local_experts']+contract['global_experts']-1)//contract['global_experts']
                    if p != balanced or not 1 <= a <= min(contract['local_experts'], p) or p > n*a:
                        raise ValueError('Expert row does not describe feasible balanced local assignments')
                    self.expert[mode, n].append((a, us))
        if (not plan or _hash(plan) != contract['point_plan_sha256']
                or coverage['points'] != len(plan)
                or coverage['expected_rounds'] != len(plan)*contract['protocol']['rounds']
                or coverage['completed_rounds'] != coverage['expected_rounds']):
            raise ValueError('Native MoE plan and completed coverage disagree')
        for pairs in (*self.local.values(), *self.expert.values(), *self.zero.values()):
            pairs.sort()
        self.tokens = {mode: sorted(n for m,n in self.expert if m == mode) for mode in ('eager', 'graph')}
        if any(not self.tokens[m] or any((c,m) not in self.local for c in ('gate_routing','finalize_copy'))
               for m in ('eager','graph')):
            raise ValueError('Missing component or execution mode')
        self.contract, self.target = contract, target

    @lru_cache(maxsize=4096)
    def local_ns(self, component, mode, tokens):
        if component not in ('gate_routing', 'finalize_copy') or mode not in self.tokens:
            raise ValueError('Invalid local component or mode')
        if type(tokens) is not int or tokens < 1:
            raise ValueError('Positive integer local token count required')
        return max(1, round(_linear(self.local[component, mode], tokens)*1000))

    @lru_cache(maxsize=8192)
    def expert_ns(self, mode, tokens, activated, local_assignments):
        if (mode not in self.tokens or any(type(v) is not int for v in (tokens, activated, local_assignments))
                or tokens < 1 or activated < 0 or local_assignments < 0):
            raise ValueError('Invalid expert query')
        if activated == local_assignments == 0:
            if mode not in self.zero:
                raise MoeCoverageError('This deployment has no all-remote routing support')
            return max(1, round(_linear(self.zero[mode], tokens)*1000))
        c = self.contract
        balanced = (tokens*c['global_top_k']*c['local_experts']+c['global_experts']-1)//c['global_experts']
        if local_assignments != balanced or not 1 <= activated <= min(c['local_experts'], balanced) or balanced > tokens*activated:
            raise MoeCoverageError('This surface covers balanced assignment totals only; arbitrary routing needs additional measurements')
        axis = self.tokens[mode]
        index = bisect_left(axis, tokens)
        if index < len(axis) and axis[index] == tokens:
            return max(1, round(_linear(self.expert[mode,tokens], activated)*1000))
        if index == 0 or index == len(axis):
            raise MoeCoverageError('Expert token count is outside the measured range')
        lo, hi = axis[index-1:index+1]
        fraction = (tokens-lo)/(hi-lo)
        low, high = self.expert[mode,lo], self.expert[mode,hi]
        candidates = []
        # A valid surface can collapse to a line (e.g. one local expert).
        # Such support needs two vertices, not an artificial third point.
        for a in low:
            for b in high:
                if math.isclose((1-fraction)*a[0]+fraction*b[0], activated, rel_tol=0, abs_tol=1e-9):
                    candidates.append(((abs(b[0]-a[0]),0,-1,a[0],b[0]),
                                       (1-fraction)*a[1]+fraction*b[1]))
        for side, pairs, opposite, mass in ((0,low,high,1-fraction),(1,high,low,fraction)):
            for a,b in zip(pairs,pairs[1:]):
                for c in opposite:
                    effective = (activated-(1-mass)*c[0])/mass
                    if not a[0]-1e-9 <= effective <= b[0]+1e-9:
                        continue
                    mix = min(1., max(0., (effective-a[0])/(b[0]-a[0])))
                    value = mass*((1-mix)*a[1]+mix*b[1])+(1-mass)*c[1]
                    score = (max(a[0],b[0],c[0])-min(a[0],b[0],c[0]),b[0]-a[0],side,a[0],c[0])
                    candidates.append((score,value))
        if not candidates:
            raise MoeCoverageError('Expert query is outside the measured feasible support')
        return max(1, round(min(candidates, key=lambda item:item[0])[1]*1000))
