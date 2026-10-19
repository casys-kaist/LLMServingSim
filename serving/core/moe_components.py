"""Strict, cached lookup of deployment-matched native MoE components.

The expert surface separates input rows, active local experts and local
assignments; within-rank histograms remain balanced profiling stimuli.
Historical surfaces retain their balanced-total lookup. The runtime selects a matching published table;
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

from profiler.core.moe_deployment import MoeTarget, component_names
from profiler.core.moe_geometry import ExpertPlacement
from profiler.core.bundle_metadata import read_tp
from profiler.core.metadata_names import (
    EAGER_ATTRIBUTION, GRAPH_ATTRIBUTION, MOE_CONDITIONING_POLICY)
from .communication import dtype_bytes


class MoeCoverageError(ValueError):
    pass


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def component_index(tp_folder):
    """Read the root-owned record, or an unmigrated historical JSON index."""
    missing = object()
    record = read_tp(tp_folder, 'moe_components', missing)
    canonical = record is not missing
    if record is missing:
        historical = Path(tp_folder)/'moe_components.json'
        if not historical.is_file():
            return None
        record = json.loads(historical.read_text())
    if (not isinstance(record, dict) or record.get('schema') != 'moe-components'
            or not isinstance(record.get('entries'), list)):
        raise ValueError('Unknown MoE component index schema')
    for entry in record['entries']:
        if not isinstance(entry, dict) or not isinstance(entry.get('target'), dict):
            raise ValueError('Malformed MoE component target entry')
        MoeTarget(**entry['target'])
        if not isinstance(entry.get('path'), str):
            raise ValueError('Malformed MoE component table path')
        if canonical and (not isinstance(entry.get('contract'), dict)
                          or not isinstance(entry.get('coverage'), dict)):
            raise ValueError('Root MoE metadata must embed its contract and coverage')
    return record


def _linear(pairs, x):
    if not pairs[0][0] <= x <= pairs[-1][0]:
        raise MoeCoverageError(f'MoE coordinate {x} is outside measured [{pairs[0][0]}, {pairs[-1][0]}]')
    index = bisect_left([p[0] for p in pairs], x)
    if pairs[index][0] == x:
        return pairs[index][1]
    (a, fa), (b, fb) = pairs[index-1:index+1]
    return fa+(fb-fa)*(x-a)/(b-a)


class _ExpertSurface:
    """Prebuilt simplices in physical (N, A, M) coordinates.

    Fixed per-axis scaling conditions the geometry without changing affine
    interpolation. SVD removes only affine dependencies, such as M=k*N at
    EP1, so a plane, line or single measured point needs no invented extent.
    """
    tolerance = 1e-9

    def __init__(self, samples):
        import numpy as np

        coordinates = np.asarray([point for point, _ in samples], dtype=float)
        self.values = np.asarray([value for _, value in samples], dtype=float)
        self.origin = coordinates[0]
        self.scale = np.ptp(coordinates, axis=0)
        self.scale[self.scale == 0] = 1.
        normalized = (coordinates-self.origin)/self.scale
        _, singular, vectors = np.linalg.svd(normalized, full_matrices=False)
        threshold = max(normalized.shape)*np.finfo(float).eps*singular[0]
        self.dimension = int(np.count_nonzero(singular > threshold))
        self.basis = vectors[:self.dimension]
        projected = normalized @ self.basis.T
        if self.dimension == 1:
            self.line = sorted(zip(projected[:, 0].tolist(), self.values.tolist()))
        elif self.dimension > 1:
            from scipy.spatial import Delaunay, QhullError
            try:
                self.triangulation = Delaunay(projected)
            except QhullError as exc:
                raise ValueError('Native MoE coordinates cannot form a supported interpolation surface') from exc
            # Materialize SciPy's lazy barycentric transforms during loading.
            self.transforms = self.triangulation.transform
            self.vertices = self.values[self.triangulation.simplices]

    def __call__(self, coordinates):
        normalized = (coordinates-self.origin)/self.scale
        projected = normalized @ self.basis.T
        if abs(normalized-projected @ self.basis).max() > self.tolerance:
            raise MoeCoverageError('Expert query is outside the measured affine support')
        if self.dimension == 0:
            return float(self.values[0])
        if self.dimension == 1:
            x = float(projected[0])
            if x < self.line[0][0]-self.tolerance or x > self.line[-1][0]+self.tolerance:
                raise MoeCoverageError('Expert query is outside the measured line support')
            return _linear(self.line, min(self.line[-1][0], max(self.line[0][0], x)))
        simplex = int(self.triangulation.find_simplex(projected, tol=self.tolerance))
        if simplex < 0:
            raise MoeCoverageError('Expert query is outside the measured convex support')
        transform = self.transforms[simplex]
        weights = (transform[:self.dimension] @ (projected-transform[self.dimension])).tolist()
        weights.append(1.-sum(weights))
        if any(not math.isfinite(weight) or weight < -self.tolerance
               or weight > 1.+self.tolerance for weight in weights):
            raise MoeCoverageError('Expert query has no nonnegative interpolation weights')
        weights = [max(0., min(1., weight)) for weight in weights]
        total = sum(weights)
        return sum(weight*value for weight, value in zip(weights, self.vertices[simplex]))/total


class MoeComponentTable:
    def __init__(self, tp_folder, *, target, expected):
        target = MoeTarget(**target)
        root = Path(tp_folder).resolve()
        index = component_index(root)
        if index is None:
            raise MoeCoverageError('No native MoE component metadata is installed')
        matching = [e for e in index['entries'] if e['target'] == vars(target)]
        if not matching:
            raise MoeCoverageError('A matching TP/DP/EP/rank/sequence-parallel contract is required')
        if len(matching) != 1:
            raise ValueError('Duplicate native MoE deployment contracts in the component index')
        entry = matching[0]
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('MoE component path leaves its TP bundle')
        path = (root/relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('MoE component path leaves its TP bundle')
        embedded = 'contract' in entry or 'coverage' in entry
        if embedded:
            contract, coverage = entry['contract'], entry['coverage']
            table = path
        else:
            # Historical bundles are read-only: their old identities are
            # checked against their own records, never against current code.
            contract = json.loads((path/'contract.json').read_text())
            coverage = json.loads((path/'coverage.json').read_text())
            identity = _hash(contract)
            if entry['contract_id'] != identity or coverage['contract_id'] != identity:
                raise ValueError('Inconsistent historical MoE component identity')
            table = path/'components.csv'
        if (contract['schema'] != index['schema']
                or coverage['schema'] != index['schema']
                or MoeTarget(**contract['target']) != target or coverage['complete'] is not True):
            raise ValueError('Incomplete or inconsistent MoE component contract')
        modes = contract.get('modes')
        self.independent_assignments = 'expert_sampling' in contract
        if self.independent_assignments:
            sampling = contract['expert_sampling']
            if (not isinstance(sampling, dict)
                    or sampling.get('axes') != ['tokens', 'activated_experts', 'local_assignments']
                    or type(sampling.get('grid_factor')) not in (int, float)
                    or not math.isfinite(sampling['grid_factor']) or sampling['grid_factor'] <= 1
                    or modes != ['eager']
                    or contract.get('expert_output_reference') != 'independent_eager_clone'):
                raise ValueError('Independent MoE axes require an eager acquisition contract')
        elif (modes != ['eager', 'graph']
                or contract.get('graph_attribution') != GRAPH_ATTRIBUTION
                or contract.get('graph_memory_pool') != 'independent_per_variant'
                or contract.get('graph_output_reference') != 'independent_pre_capture_clone'):
            raise ValueError('Historical graph data requires verified launch-correlated acquisition')
        if ((contract.get('expert_conditioning') or {}).get('schema') != MOE_CONDITIONING_POLICY
                or coverage.get('quality_supported') is not True
                or contract.get('eager_attribution') != EAGER_ATTRIBUTION):
            raise ValueError('Native MoE data requires matched-work conditioning and native launch attribution')
        required = {'model', 'hardware', 'variant', 'global_experts', 'global_top_k', 'hidden_dim'}
        if not required.issubset(expected) or any(contract.get(k) != v for k, v in expected.items()):
            raise ValueError('Model, hardware or tensor geometry differs from the measured MoE contract')
        placement = ExpertPlacement(contract['global_experts'],contract['global_top_k'],
                                    target.ep,target.rank)
        expected_map = None if target.ep == 1 else list(placement.expert_map)
        if (contract.get('placement') != 'linear'
                or contract.get('expert_map') != expected_map
                or contract.get('local_experts') != len(placement.local_global_ids)):
            raise ValueError('MoE expert ownership differs from the target rank')
        parallel = contract['expert_parallel_config']
        expected_parallel = dict(tp_size=1,tp_rank=0,pcp_size=1,pcp_rank=0,
            dp_size=target.dp,dp_rank=target.dp_rank,ep_size=target.ep,ep_rank=target.rank,
            sp_size=target.tp if target.sequence_parallel else 1,use_ep=target.ep > 1,
            all2all_backend='allgather_reducescatter',enable_eplb=False)
        if any(parallel.get(k) != v for k,v in expected_parallel.items()):
            raise ValueError('MoE backend or parallel configuration differs from the supported execution path')
        self.components = component_names(contract)
        region = contract['expert_region']
        self.live = region['schema'] == 'native-live-expert-region'
        if self.live:
            capabilities = contract.get('native_capabilities')
            unsupported = {'deferred', 'has_bias', 'lora', 'eplb', 'redundant_experts',
                           'input_weighted', 'fused_shared_gate'}
            if (not isinstance(capabilities, dict) or set(capabilities) != unsupported
                    or any(value is not False for value in capabilities.values())
                    or type(contract.get('skip_final_all_reduce')) is not bool):
                raise ValueError('Live components lack supported native capability evidence')
            if (not self.independent_assignments
                    or contract.get('local_execution') != dict(
                        prepare='included_in_experts', finalize='included_in_experts')
                    or contract.get('gate_routing_boundary') not in (
                        'projection_only', 'projection_and_routing')
                    or contract.get('wrapper_boundary') != 'native_parent_excluding_gate_router_and_routed_kernel'
                    or contract.get('wrapper_overlap') != 'sum_of_unique_cuda_activity_durations'
                    or contract.get('native_method_reference') != 'exact_output_at_each_routing_stimulus'
                    or region.get('weight_layout') != 'live_processed_native'
                    or not isinstance(region.get('quant_method'), str)
                    or not region['quant_method']
                    or region.get('routing') != ('fused' if contract.get('gate_routing_boundary') == 'projection_only'
                                                 else 'external')
                    or type(contract.get('wrapper_weight_bytes')) is not int
                    or contract['wrapper_weight_bytes'] < 0
                    or type(contract.get('expert_bank_bytes')) is not int
                    or contract['expert_bank_bytes'] < contract.get('expert_weight_bytes', 0)):
                raise ValueError('Live components require explicit local native boundaries')
            for key in ('expert_hidden_dim', 'expert_intermediate_size', 'gate_hidden_dim'):
                if type(contract.get(key)) is not int or contract[key] < 1:
                    raise ValueError('Live components require physical gate and expert geometry')
            if (region.get('hidden_dim') != contract['expert_hidden_dim']
                    or region.get('intermediate_size') != contract['expert_intermediate_size']):
                raise ValueError('Live expert tensor geometry differs from the component contract')
        elif target.dp == 1 and contract.get('local_execution') != dict(
                prepare='identity', finalize='output_alias_noop', expert_output_alias=True):
            raise ValueError('DP1 requires verified identity preparation and aliased no-op finalization')
        geometry = () if self.live else ('hidden_dim', 'intermediate_size')
        if (region.get('expert_parallel_config') != parallel
                or any(region.get(k) != contract[k] for k in
                       ('global_experts','global_top_k','local_experts','expert_map',
                        'activation','input_dtype') + geometry)
                or not region.get('backend') or not region.get('expert_class')):
            raise ValueError('Native expert kernel contract and component table disagree')
        gate_dtype = contract.get('gate_weight_dtype', contract['input_dtype'])
        gate_hidden = contract.get('gate_hidden_dim', contract['hidden_dim'])
        gate_bytes = contract.get('gate_weight_bytes')
        if (contract['topk_weights_dtype'] != 'torch.float32'
                or contract['topk_ids_dtype'] not in ('torch.int32','torch.int64')
                or contract['input_dtype'] not in ('torch.float16','torch.bfloat16','torch.float32')
                or gate_dtype not in ('torch.float16','torch.bfloat16','torch.float32')
                or type(gate_hidden) is not int or gate_hidden < 1
                or type(gate_bytes) is not int
                or type(contract.get('expert_weight_bytes')) is not int
                or contract['expert_weight_bytes'] <= 0
                or (not self.live and contract['expert_weight_bytes'] % contract['local_experts'])):
            raise ValueError('Invalid native component dtype or measured weight-byte contract')
        gate_matrix = contract['global_experts']*gate_hidden*dtype_bytes(gate_dtype)
        if (gate_bytes < gate_matrix if self.live else gate_bytes != gate_matrix):
            raise ValueError('Measured gate storage does not cover its projection matrix')
        checksum = hashlib.sha256(table.read_bytes()).hexdigest()
        if checksum != coverage['table_sha256'] or (not embedded and checksum != entry['table_sha256']):
            raise ValueError('MoE component CSV checksum mismatch')
        self.local, self.expert, self.zero = defaultdict(list), defaultdict(list), defaultdict(list)
        self.assignments, self.active = defaultdict(list), defaultdict(list)
        self.exact, plan = {}, []
        with table.open() as stream:
            for row in csv.DictReader(stream):
                component, mode = row['component'], row['mode']
                n, a, p = (int(row[k]) for k in ('tokens', 'activated_experts', 'local_assignments'))
                us = float(row['time_us'])
                key = (component, mode, n, a, p)
                if (component not in self.components
                        or mode not in modes or n < 1
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
                    if placement.assignment_bounds(n, a)[0] != 0:
                        raise ValueError('All-remote work is impossible for this placement')
                    self.zero[mode].append((n, us))
                elif self.independent_assignments:
                    lower, upper = placement.assignment_bounds(n, a)
                    if a < 1 or not lower <= p <= upper:
                        raise ValueError('Infeasible local expert assignments')
                    self.assignments[mode, n, a].append((p, us))
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
        summary = coverage.get('conditioning_summary')
        tolerance = contract['expert_conditioning'].get('quality_relative_tolerance')
        expert_points = sum(point['component'] == 'experts' for point in plan)
        if (not isinstance(summary, dict)
                or set(summary) != {'checked_points', 'failed_points', 'max_relative_change'}
                or type(summary['checked_points']) is not int
                or summary['checked_points'] != expert_points or expert_points < 1
                or type(summary['failed_points']) is not int or summary['failed_points'] != 0
                or type(tolerance) not in (int, float) or not math.isfinite(tolerance)
                or not 0 < tolerance < 1
                or type(summary['max_relative_change']) not in (int, float)
                or not math.isfinite(summary['max_relative_change'])
                or not 0 <= summary['max_relative_change'] <= tolerance):
            raise ValueError('Native MoE conditioning summary disagrees with supported coverage')
        for pairs in (*self.local.values(), *self.expert.values(), *self.zero.values(), *self.assignments.values()):
            pairs.sort()
        if self.independent_assignments:
            for (mode, n, a), pairs in self.assignments.items():
                if (pairs[0][0], pairs[-1][0]) != placement.assignment_bounds(n, a):
                    raise ValueError('Independent assignment grid must include feasible endpoints')
                self.active[mode, n].append(a)
            for (mode, n), axis in self.active.items():
                axis.sort()
                if (axis[0], axis[-1]) != placement.active_bounds(n):
                    raise ValueError('Active-expert grid must include feasible endpoints')
            self.tokens = {m: sorted(n for mode, n in self.active if mode == m) for m in modes}
        else:
            self.tokens = {mode: sorted(n for m,n in self.expert if m == mode) for mode in modes}
        if any(not self.tokens[m] or any((c,m) not in self.local for c in self.components if c != 'experts')
               for m in modes):
            raise ValueError('Missing component or execution mode')
        self.contract, self.target, self.placement = contract, target, placement
        if self.independent_assignments:
            self.surfaces = {mode: _ExpertSurface(sorted(
                ((n, a, p), us) for (component, m, n, a, p), us in self.exact.items()
                if component == 'experts' and m == mode and a > 0)) for mode in modes}

    def execution_mode(self, graph):
        # Acquisition mode is not a serving padding decision. New tables use
        # eager kernels at the already-resolved (possibly padded) row count.
        return 'graph' if graph and not self.independent_assignments else 'eager'

    @lru_cache(maxsize=4096)
    def local_ns(self, component, mode, tokens):
        if component == 'experts' or component not in self.components or mode not in self.tokens:
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
        if self.independent_assignments:
            return self._independent_ns(mode, tokens, activated, local_assignments)
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

    def _independent_ns(self, mode, tokens, activated, assignments):
        """Interpolate physical work coordinates using the prebuilt surface."""
        place = self.placement
        try:
            amin, amax = place.active_bounds(tokens)
            lower, upper = place.assignment_bounds(tokens, activated)
        except ValueError as exc:
            raise MoeCoverageError(str(exc)) from exc
        if not amin <= activated <= amax or not lower <= assignments <= upper:
            raise MoeCoverageError('Expert query has infeasible active experts or assignments')
        exact = self.exact.get(('experts', mode, tokens, activated, assignments))
        if exact is not None:
            return max(1, round(exact*1000))
        result = self.surfaces[mode]((tokens, activated, assignments))
        return max(1, round(result*1000))
