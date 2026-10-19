"""Eager native MoE acquisition with independent local-work coordinates."""
from dataclasses import asdict, replace
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from .config import HOST_ENGINE_DEFAULTS, load_architecture, probe_moe_params
from .moe_deployment import MoeTarget, component_names
from .moe_geometry import ExpertPlacement
from .moe_conditioning import expert_conditioning
from .bundle_metadata import read_tp, update_tp, update_bundle
from .stack import text_config

SCHEMA = 'moe-components'
KEYS = ('component', 'mode', 'tokens', 'activated_experts', 'local_assignments')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


_OBSERVATION_FIELDS = {'vllm_version', 'torch_version', 'cuda_version', 'source_sha256',
                       'sources', 'planning_source_sha256', 'model_config_sha256', 'sampling_graph',
                       'engine_effective', 'observations'}


def _contract_observations(contract):
    """Small acquisition context; it does not claim any completed row count."""
    value = {key: item for key, item in contract.items() if key in _OBSERVATION_FIELDS}
    region = contract.get('expert_region')
    if isinstance(region, dict):
        value['expert_region'] = {key: item for key, item in region.items()
                                  if key in _OBSERVATION_FIELDS}
    if isinstance(contract.get('gpu'), dict) and 'uuid' in contract['gpu']:
        value['gpu'] = dict(uuid=contract['gpu']['uuid'])
    return value


def _contract_conditions(contract):
    """Select numerical conditions independently of execution observations."""
    observations = _OBSERVATION_FIELDS
    value = {key: item for key, item in contract.items() if key not in observations}
    if isinstance(value.get('expert_region'), dict):
        value['expert_region'] = {key: item for key, item in value['expert_region'].items()
                                  if key not in observations}
    if isinstance(value.get('gpu'), dict):
        value['gpu'] = {key: item for key, item in value['gpu'].items() if key != 'uuid'}
    return value


def semantic_contract(contract):
    """Numerical acquisition identity without runtime-version restrictions."""
    return _contract_conditions(contract)


def _replace_entry(previous, entry):
    block = dict(schema=SCHEMA, entries=[]) if previous is None else previous
    if (not isinstance(block, dict) or block.get('schema') != SCHEMA
            or not isinstance(block.get('entries'), list)):
        raise ValueError('Unknown native MoE metadata schema')
    matching = [item for item in block['entries'] if item['target'] == entry['target']]
    if len(matching) > 1:
        raise ValueError('Duplicate native MoE deployment contracts')
    retained = [item for item in block['entries'] if item['target'] != entry['target']]
    return dict(block, schema=SCHEMA, entries=retained + [{**(matching[0] if matching else {}), **entry}])


def grid(maximum, factor=2.0, minimum=1):
    if type(maximum) is not int or maximum < 1:
        raise ValueError("Grid maximum must be a positive integer")
    if not math.isfinite(factor) or factor <= 1 or not 1 <= minimum <= maximum:
        raise ValueError("Grid needs a finite factor > 1 and valid positive bounds")
    values, value = {minimum, maximum}, 1.
    while value < maximum:
        point = max(1, round(value))
        if minimum <= point <= maximum:
            values.add(point)
        value *= factor
    return sorted(values)


def physical_token_budget(plan, budget):
    """MRV1 compiler SP also pads eager forwards beyond a logical budget."""
    return max([budget] + [((budget+g.sp_multiple-1)//g.sp_multiple)*g.sp_multiple
                           for g in plan.graphs.values() if g.runner == 'v1'])


def token_sampling(target, plan, budget):
    """Capture-first token axes, with eager fallback in target DP/SP domains.

    Successful DP graph rounds use one padded count on every member. A round
    outside graph coverage may instead have unequal counts; capture anchors
    alone cannot describe its gathered expert rows or its peers' local rows.
    """
    domains = [target.token_domains((n,)*target.dp) for n in plan.forward_sizes]
    anchors = dict(local=sorted({d['gate_rows'] for d in domains}),
                   experts=sorted({d['expert_rows'] for d in domains}))
    logical_budget = budget
    budget = physical_token_budget(plan, budget)
    maximum = target.token_domains((budget,)*target.dp)
    graphs = list(plan.graphs.values())
    # Decode-only graphs leave prefill/mixed eager throughout the token range.
    partial = not graphs or any(g.mode not in ('FULL', 'PIECEWISE', 'FULL_AND_PIECEWISE')
                               or not g.capture_sizes for g in graphs)
    ranges = dict(local=[], experts=[])
    if partial:
        ranges = dict(local=[[1, maximum['gate_rows']]],
                      experts=[[1, maximum['expert_rows']]])
    else:
        cap = min(g.capture_sizes[-1] for g in graphs)
        if cap < budget:
            # One eager DP member can cancel graph replay on a small peer.
            # V2 also removes that peer's padding; retain the full local axis.
            first = target.token_domains((cap+1,)+(1,)*(target.dp-1))
            local_start = 1 if target.dp > 1 else first['gate_rows']
            ranges = dict(local=[[local_start, maximum['gate_rows']]],
                          experts=[[first['expert_rows'], maximum['expert_rows']]])
    return dict(logical_token_budget=logical_budget, capture_tokens=anchors,
                geometric_token_ranges=ranges)


def _token_points(sampling, axis, maximum):
    # Endpoints support interpolation even when the smallest capture pads N=1.
    points = {1, maximum}
    points.update(n for n in sampling.get('capture_tokens', {}).get(axis, ())
                  if 0 < n <= maximum)
    # Preserve the plans of already-started three-axis acquisitions. New plans
    # explicitly distinguish an empty eager range from absent plan metadata.
    ranges = sampling.get('geometric_token_ranges', {}).get(axis, [[1, maximum]])
    for lower, upper in ranges:
        if (type(lower) is not int or type(upper) is not int
                or not 1 <= lower <= upper <= maximum):
            raise ValueError('MoE geometric token range exceeds its target domain')
        points.update(grid(upper, sampling['grid_factor'], lower))
    return points


def points_for(contract):
    target = MoeTarget(**contract['target'])
    components = component_names(contract)
    place = ExpertPlacement(contract['global_experts'], contract['global_top_k'], target.ep, target.rank)
    factor = contract['expert_sampling']['grid_factor']
    maximum_tokens = contract['maximum_expert_tokens']
    token_points = _token_points(contract['expert_sampling'], 'experts', maximum_tokens)
    local_e = len(place.local_global_ids)
    # A <= min(local_E, N*k) changes slope at local_E/k. Preserve both
    # integer neighbours so interpolation does not cut off feasible work.
    for per_token in (place.global_top_k,
                      max(0, place.global_top_k-(place.global_experts-local_e))):
        if per_token:
            for corner in (local_e//per_token, (local_e+per_token-1)//per_token):
                if 1 <= corner <= maximum_tokens:
                    token_points.add(corner)
    local = target.token_domains((contract['local_token_budget'],)*target.dp)['gate_rows']
    for mode in contract['modes']:
        for component in components:
            if component == 'experts':
                continue
            local_points = _token_points(contract['expert_sampling'], 'local', local)
            for n in sorted(local_points):
                yield dict(component=component, mode=mode, tokens=n,
                           activated_experts=-1, local_assignments=-1)
        for n in sorted(token_points):
            balanced = (n*place.global_top_k*local_e+place.global_experts-1)//place.global_experts
            minimum, maximum = place.active_bounds(n)
            active_points = set(grid(maximum, factor, minimum))
            # Preserve the corners where the feasible assignment envelope
            # changes slope; these are geometry, not fitted latency knobs.
            for corner in (place.global_top_k,
                           n*max(0, place.global_top_k-(place.global_experts-local_e))):
                if minimum <= corner <= maximum:
                    active_points.add(corner)
            for active in sorted(active_points):
                lower, upper = place.assignment_bounds(n, active)
                assignments = set(grid(upper, factor, lower))
                if lower <= balanced <= upper:
                    assignments.add(balanced)
                for pairs in sorted(assignments):
                    yield dict(component='experts', mode=mode, tokens=n,
                               activated_experts=active, local_assignments=pairs)
            try:
                lower, upper = place.assignment_bounds(n, 0)
            except ValueError:
                continue
            if lower == 0:
                yield dict(component='experts', mode=mode, tokens=n,
                           activated_experts=0, local_assignments=0)


def point_key(point):
    return tuple(point[k] for k in KEYS)


def validate_result(result, iterations):
    values = result['per_forward_us']
    if (result['verified'] is not True or len(values) != iterations
            or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in values)):
        raise ValueError('Incomplete or non-finite measurement repetitions')
    deltas=result['conservation_delta_us']
    if (set(deltas)!={'component'}
            or any(not math.isfinite(v) or abs(v)>1e-5 for v in deltas.values())):
        raise ValueError('CUDA event attribution does not conserve measured work')
    kernels=result['cuda_kernel_counts']
    if (len(kernels)!=iterations or any(not counts or any(
            type(count) is not int or count<1 for count in counts.values()) for counts in kernels)):
        raise ValueError('Missing per-forward CUDA kernel counts')


class MeasurementStore:
    """Exclusive, durable raw repetitions; incomplete data is never published."""
    def __init__(self, parent, contract, points):
        if contract.get('schema') != SCHEMA:
            raise ValueError('Unknown native MoE measurement schema')
        self.contract = contract
        self._incoming_observations = _contract_observations(self.contract)
        self._context_recorded = True
        self.identity = fingerprint(semantic_contract(contract))
        target = MoeTarget(**contract['target'])
        self.tp_dir = Path(parent).parent
        if self.tp_dir.name != f'tp{target.tp}':
            raise ValueError('MoE target TP differs from its table directory')
        self.folder = Path(parent)/f'dp{target.dp}_ep{target.ep}_rank{target.rank}_sp{int(target.sequence_parallel)}'
        self.folder.mkdir(parents=True, exist_ok=True)
        self.lock = (self.folder/'writer.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.rounds = contract['protocol']['rounds']
            self.iterations = contract['protocol']['iterations']
            if min(self.rounds, self.iterations) < 1:
                raise ValueError("Positive measurement repetitions are required")
            self.points = {point_key(p): p for p in points}
            if not self.points or len(self.points) != len(points):
                raise ValueError("Empty or duplicate measurement plan")
            missing = object()
            metadata = read_tp(self.tp_dir, 'moe_components', missing)
            if metadata is not missing:
                if (not isinstance(metadata, dict) or metadata.get('schema') != SCHEMA
                        or not isinstance(metadata.get('entries'), list)):
                    raise ValueError('Unknown native MoE metadata schema')
                existing = [entry for entry in metadata['entries'] if entry['target'] == vars(target)]
                if len(existing) > 1:
                    raise ValueError('Duplicate native MoE deployment contracts')
                if existing and semantic_contract(existing[0]['contract']) != semantic_contract(contract):
                    raise ValueError('Native MoE measurement conditions changed; use a new output root')
                if existing:
                    # Reused timings keep the original acquisition observations.
                    # A later runtime is recorded only if it produces new work.
                    self.contract = existing[0]['contract']
                    contexts = existing[0].get('acquisition_contexts', [])
                    if not isinstance(contexts, list) or any(not isinstance(item, dict) for item in contexts):
                        raise ValueError('Invalid native MoE acquisition contexts')
                    self._context_recorded = (
                        self._incoming_observations == _contract_observations(self.contract)
                        or self._incoming_observations in contexts)
                if (existing and existing[0]['coverage'].get('complete')
                        and not (self.folder/'samples.jsonl').exists()):
                    raise ValueError('Published MoE data has no acquisition checkpoints; use a new output root to remeasure')
            self.rows = {}
            raw = self.folder/'samples.jsonl'
            if raw.exists():
                with raw.open('rb') as stream:
                    offset = 0
                    for line in stream:
                        if not line.endswith(b'\n'):
                            archive = self.folder/('interrupted-tail-'+hashlib.sha256(line).hexdigest()+'.bin')
                            if not archive.exists():
                                archive.write_bytes(line)
                            with raw.open('r+b') as repair:
                                repair.truncate(offset)
                            break
                        self._validate(json.loads(line))
                        offset += len(line)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.lock.close()

    def _validate(self, row):
        key, repeat = point_key(row['point']), row['round']
        result = row['result']
        if row['contract_id'] != self.identity or key not in self.points:
            raise ValueError("Measurement identity or point differs from its plan")
        if type(repeat) is not int or not 0 <= repeat < self.rounds:
            raise ValueError("Invalid measurement round")
        if ((self.contract.get('expert_region') or {}).get('schema') == 'native-live-expert-region'
                and result.get('geometry', {}).get('native_state_unchanged') is not True):
            raise ValueError('Live native measurement did not verify unchanged runtime state')
        if self.contract.get('expert_conditioning') and row['point']['component']=='experts':
            plan=expert_conditioning(self.contract,row['point']['activated_experts'],self.iterations)
            if (result.get('geometry',{}).get('conditioning') != plan
                    or result['geometry'].get('independent_outputs_verified') is not True):
                raise ValueError('Expert measurement did not cover the declared complete weight cycles')
            validate_result(result,plan['forward_counts'][1])
            validate_result(result['conditioning_control'],plan['forward_counts'][0])
        else:
            validate_result(result,self.iterations)
        if self.contract.get('eager_attribution'):
            expected = self.contract[row['point']['mode']+'_attribution']
            if result.get('attribution') != expected or ('conditioning_control' in result
                    and result['conditioning_control'].get('attribution') != expected):
                raise ValueError('Measurement attribution differs from the declared contract')
        if (key, repeat) in self.rows:
            raise ValueError("Duplicate complete raw measurement")
        self.rows[key, repeat] = row

    def append(self, point, repeat, result):
        row = dict(contract_id=self.identity, point=point, round=repeat, result=result)
        self._validate(row)
        if not self._context_recorded:
            def record_context(block):
                entries = [entry for entry in block['entries']
                           if entry['target'] == self.contract['target']]
                if len(entries) != 1:
                    raise ValueError('Native MoE acquisition metadata changed during collection')
                entry = dict(entries[0])
                contexts = list(entry.get('acquisition_contexts', []))
                if self._incoming_observations not in contexts:
                    contexts.append(self._incoming_observations)
                entry['acquisition_contexts'] = contexts
                return _replace_entry(block, entry)
            # The result is already validated. Persist its execution context
            # before its raw row: interruption can leave an unused context,
            # never a durable row whose runtime observations have been lost.
            update_tp(self.tp_dir, 'moe_components', record_context)
            self._context_recorded = True
        with (self.folder/'samples.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())

    def publish(self):
        expected = len(self.points)*self.rounds
        status = dict(schema=SCHEMA, complete=len(self.rows)==expected,
                      completed_rounds=len(self.rows), expected_rounds=expected, points=len(self.points))
        if status['complete'] and self.contract.get('expert_conditioning'):
            summary = dict(checked_points=0, failed_points=0, max_relative_change=0.0)
            tolerance=self.contract['expert_conditioning']['quality_relative_tolerance']
            for key,point in self.points.items():
                if point['component']!='experts': continue
                selected=[v for r in range(self.rounds) for v in self.rows[key,r]['result']['per_forward_us']]
                controls=[v for r in range(self.rounds) for v in self.rows[key,r]['result']['conditioning_control']['per_forward_us']]
                high,low=statistics.median(selected),statistics.median(controls)
                change=abs(high/low-1)
                summary['checked_points'] += 1
                summary['failed_points'] += int(not change <= tolerance)
                summary['max_relative_change'] = max(summary['max_relative_change'], change)
            status['conditioning_summary'] = summary
            status['quality_supported'] = summary['checked_points'] > 0 and summary['failed_points'] == 0
            status['complete']=status['quality_supported']
        if status['complete']:
            path = self.folder/'components.csv'
            temporary = path.with_suffix('.csv.tmp')
            with temporary.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=[*KEYS, 'time_us'])
                writer.writeheader()
                for key, point in self.points.items():
                    medians = [statistics.median(self.rows[key,r]['result']['per_forward_us'])
                               for r in range(self.rounds)]
                    writer.writerow(dict(point, time_us=statistics.median(medians)))
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
            status['table_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        entry = dict(target=self.contract['target'],
            path=str((self.folder/'components.csv').relative_to(self.tp_dir)),
            contract=self.contract, coverage=status)
        update_tp(self.tp_dir, 'moe_components', lambda previous: _replace_entry(previous, entry))
        return status


def validate_request(arch, args):
    if args.dp_degrees is None or not probe_moe_params(text_config(args.model_config or {})):
        raise ValueError("Native component profiling requires a MoE checkpoint and resolved DP")
    if not arch.catalog.moe or len(arch.catalog.moe) != 1:
        raise NotImplementedError("One catalog MoE block is required for this component adapter")
    if args.force:
        raise ValueError("Native contracts are immutable; use a new output root instead of --force")
    if args.profile_mtp or args.only_skew:
        raise ValueError("Native components do not profile a drafter or skew")
    if not args.dp_degrees or any(type(dp) is not int or dp < 1 for dp in args.dp_degrees):
        raise ValueError("Native component target DP degrees must be positive integers")
    if args.ep_degrees is not None and (not args.ep_degrees or any(
            type(ep) is not int or ep < 1 for ep in args.ep_degrees)):
        raise ValueError("Native component target EP degrees must be positive integers")
    if args.moe_rounds < 1 or args.measurement_iterations < 1:
        raise ValueError("Measurement repetitions must be positive")
    if not math.isfinite(args.moe_grid_factor) or args.moe_grid_factor <= 1:
        raise ValueError("MoE grid factor must be finite and greater than one")
    # Quantization alone does not determine the expert execution API. The
    # worker admits its actual native method, tensors and component boundaries
    # before returning a contract; unsupported methods still fail there.


def resolve_targets(arch, args, tps):
    """Plan full-EP targets and reject unsupported selections before engine boot."""
    validate_request(arch, args)
    tps = tuple(tps)
    if not tps or any(type(tp) is not int or tp < 1 for tp in tps):
        raise ValueError("Native component target TP degrees must be positive integers")
    pairs = sorted({(tp, dp) for tp in tps for dp in args.dp_degrees})
    if args.ep_degrees is not None:
        requested = set(args.ep_degrees)
        available = {tp*dp for tp, dp in pairs}
        missing = requested - available
        if missing:
            raise ValueError(
                f"Requested --ep degrees {sorted(missing)} do not match the selected "
                f"TP*DP targets {sorted(available)}. Set --tp/--dp accordingly; "
                "a MoE slice uses --tp-refresh, not every session TP degree. "
                "Only full expert parallelism (EP=TP*DP) is supported.")
        pairs = [(tp, dp) for tp, dp in pairs if tp*dp in requested]
    entry = next(iter(arch.catalog.moe.values()))
    if (any(tp > 1 and dp > 1 for tp, dp in pairs)
            and entry.sequence_parallel is None):
        raise ValueError("TP+DP requires an audited MoE sequence_parallel catalog binding")
    experts, _ = probe_moe_params(text_config(args.model_config))
    if any(tp*dp > experts for tp, dp in pairs):
        raise ValueError("Requested EP degree exceeds the global expert count")
    return [MoeTarget(tp, dp, tp*dp, rank, bool(entry.sequence_parallel and tp>1 and dp>1))
            for tp, dp in pairs for rank in range(tp*dp)]


def run_components(arch_path, args, tps, variant_root, *, kv_cache_memory_bytes=1024**3):
    """Acquire isolated components under an explicit native-validated KV cap.

    The internal cache budget may be raised for a context or cache layout that
    needs it; the default 1 GiB is a resource bound, not a capacity guarantee.
    Ordinary acquisition engines and their recorded settings are independent.
    """
    from . import logger as log
    from .engine import spin_up, spin_down, probe_limits
    from .sampling import resolve_sampling
    from .bundle_metadata import read_bundle
    from .callable_profile import profile_for

    if type(kv_cache_memory_bytes) is not int or kv_cache_memory_bytes < 1:
        raise ValueError('Component KV memory budget must be a positive integer')
    arch = load_architecture(arch_path)
    targets = resolve_targets(arch, args, tps)
    engine_args = replace(args, tp_degrees=[1], profile_mtp=False)
    budget = args.max_num_batched_tokens or HOST_ENGINE_DEFAULTS['max_num_batched_tokens']
    variant_root.mkdir(parents=True, exist_ok=True)
    def record_bundle(meta):
        for key, value in (('model', args.model), ('hardware', args.hardware),
                           ('variant', args.effective_variant)):
            if key in meta and meta[key] != value:
                raise ValueError('Native MoE output differs from the existing bundle')
            meta[key] = value
    update_bundle(variant_root, record_bundle)
    # Components need the first routed MLP, not every attention alternative.
    # Preserve a declared profile's native engine settings (e.g. text-only
    # acquisition) while keeping this engine separate from ordinary tables.
    # The isolated component engine never consumes attention history. Leave
    # the requested context/token/sequence limits intact, but reserve memory
    # for independently conditioned native weight banks instead of assigning
    # the usual GPU-utilization remainder to an unused KV cache. vLLM still
    # checks that this explicit cache can hold its configured max_model_len;
    # insufficient native capacity is an error, never a smaller context.
    log.info('Native MoE engine KV budget: %d bytes; vLLM must admit the configured context length',
             kv_cache_memory_bytes)
    llm, _, temporary = spin_up(engine_args, 1, ('mlp',), native_profile=profile_for(arch),
                                engine_overrides={'kv_cache_memory_bytes': kv_cache_memory_bytes})
    try:
        limits = probe_limits(llm, engine_args)
        effective = llm.llm_engine.vllm_config
        engine_effective = dict(
            max_model_len=effective.model_config.max_model_len,
            max_num_batched_tokens=effective.scheduler_config.max_num_batched_tokens,
            max_num_seqs=effective.scheduler_config.max_num_seqs,
            kv_cache_memory_bytes=effective.cache_config.kv_cache_memory_bytes,
            block_size=effective.cache_config.block_size,
            num_gpu_blocks=effective.cache_config.num_gpu_blocks)
        for target in targets:
            plan = resolve_sampling(args, limits, target.tp, dp=target.dp,
                                    meta=read_bundle(variant_root))
            physical_budget = physical_token_budget(plan, budget)
            native = llm.collective_rpc('moe_component_initialize',
                                        args=(asdict(target), physical_budget))[0]
            contract = dict(native, model=args.model, hardware=args.hardware, variant=args.effective_variant,
                engine_effective=engine_effective,
                expert_sampling=dict(axes=['tokens', 'activated_experts', 'local_assignments'],
                                     grid_factor=args.moe_grid_factor,
                                     **token_sampling(target, plan, budget)),
                sampling_graph=plan.metadata(),
                protocol=dict(rounds=args.moe_rounds, iterations=args.measurement_iterations,
                              representative='median_of_forward_medians', timing='cuda_kernel_sum'))
            contract = json.loads(json.dumps(contract, default=str))
            points = list(points_for(contract))
            contract['point_plan_sha256'] = fingerprint(points)
            parent = variant_root/f'tp{target.tp}'/'moe_components'
            store = MeasurementStore(parent, contract, points)
            try:
                store.publish()
                log.info("Native MoE TP=%d DP=%d EP=%d rank=%d: %d points, %d/%d rounds present",
                         target.tp, target.dp, target.ep, target.rank, len(points),
                         len(store.rows), len(points)*args.moe_rounds)
                for index, point in enumerate(points):
                    for repeat in range(args.moe_rounds):
                        if (point_key(point), repeat) in store.rows:
                            continue
                        try:
                            result = llm.collective_rpc('moe_component_measure',
                                                       args=(point, args.measurement_iterations,
                                                             str(store.folder/'failures')))[0]
                        except Exception as exc:
                            failure = dict(contract_id=store.identity, point=point, round=repeat,
                                           epoch_s=time.time(), error=str(exc))
                            with (store.folder/'failures.jsonl').open('a') as stream:
                                stream.write(json.dumps(failure, allow_nan=False)+'\n')
                                stream.flush()
                                os.fsync(stream.fileno())
                            store.publish()
                            raise
                        store.append(point, repeat, result)
                    if index % max(1,len(points)//20) == 0 or index+1 == len(points):
                        log.info("Native MoE rank=%d: %d/%d points", target.rank, index+1, len(points))
                status = store.publish()
                if not status['complete']:
                    failed = status.get('conditioning_summary', {}).get('failed_points', 0)
                    raise RuntimeError(f"Native MoE acquisition not publishable: {failed} conditioning controls failed; see {variant_root/'meta.yaml'}")
            finally:
                store.close()
                llm.collective_rpc('moe_component_release')
    finally:
        spin_down(llm, temporary)
