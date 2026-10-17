"""Native MoE acquisition; legacy whole-block tables stay intact."""
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
from .moe_deployment import MoeTarget
from .moe_geometry import ExpertPlacement
from .moe_conditioning import expert_conditioning
from .bundle_metadata import read_tp, update_tp, update_bundle

SCHEMA = 'moe-components'
KEYS = ('component', 'mode', 'tokens', 'activated_experts', 'local_assignments')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


_OBSERVATION_FIELDS = {'vllm_version', 'torch_version', 'cuda_version', 'source_sha256',
                       'sources', 'planning_source_sha256', 'model_config_sha256',
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


def grid(maximum):
    if type(maximum) is not int or maximum < 1:
        raise ValueError("Grid maximum must be a positive integer")
    values, value = {1, maximum}, 1.
    while value < maximum:
        values.add(max(1, round(value)))
        value *= math.sqrt(2)
    return sorted(values)


def points_for(contract):
    target = MoeTarget(**contract['target'])
    place = ExpertPlacement(contract['global_experts'], contract['global_top_k'], target.ep, target.rank)
    local = target.token_domains((contract['local_token_budget'],)*target.dp)['gate_rows']
    for mode in contract['modes']:
        for component in ('gate_routing', 'finalize_copy'):
            for n in grid(local):
                yield dict(component=component, mode=mode, tokens=n,
                           activated_experts=-1, local_assignments=-1)
        for n in grid(contract['maximum_expert_tokens']):
            local_e = len(place.local_global_ids)
            pairs = (n*place.global_top_k*local_e+place.global_experts-1)//place.global_experts
            minimum, maximum = (pairs+n-1)//n, min(local_e, pairs)
            for active in sorted(set(grid(maximum)+[minimum])):
                try:
                    lower, upper = place.assignment_bounds(n, active)
                except ValueError:
                    continue
                if lower <= pairs <= upper:
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
    if args.moe_dp_degrees is None or not probe_moe_params(args.model_config or {}):
        raise ValueError("Native component profiling requires a MoE checkpoint and explicit DP")
    if not arch.catalog.moe or len(arch.catalog.moe) != 1:
        raise NotImplementedError("One catalog MoE block is required for this component adapter")
    if args.force:
        raise ValueError("Native contracts are immutable; use a new output root instead of --force")
    if args.profile_mtp or args.only_skew:
        raise ValueError("Native components do not profile a drafter or skew")
    if any(type(dp) is not int or dp < 2 for dp in args.moe_dp_degrees):
        raise NotImplementedError("Native DP+EP components require DP>=2; DP1 prepare/finalize support is separate")
    if args.moe_rounds < 1 or args.measurement_iterations < 1:
        raise ValueError("Measurement repetitions must be positive")
    config = args.model_config or {}
    for holder in (config, config.get('text_config') or {}):
        if holder.get('quantization_config'):
            raise NotImplementedError("Native component acquisition currently supports unquantized experts")


def run_components(arch_path, args, tps, variant_root):
    from . import logger as log
    from .engine import spin_up, spin_down

    arch = load_architecture(arch_path)
    validate_request(arch, args)
    entry = next(iter(arch.catalog.moe.values()))
    if any(tp > 1 for tp in tps) and entry.sequence_parallel is None:
        raise ValueError("TP+DP requires an audited MoE sequence_parallel catalog binding")
    targets = [MoeTarget(tp, dp, tp*dp, rank, bool(entry.sequence_parallel and tp>1))
               for tp in tps for dp in args.moe_dp_degrees for rank in range(tp*dp)]
    experts, _ = probe_moe_params(args.model_config)
    if any(target.ep > experts for target in targets):
        raise ValueError("Requested EP degree exceeds the global expert count")
    engine_args = replace(args, tp_degrees=[1], moe_ep_degrees=(1,), profile_mtp=False)
    budget = args.max_num_batched_tokens or HOST_ENGINE_DEFAULTS['max_num_batched_tokens']
    variant_root.mkdir(parents=True, exist_ok=True)
    def record_bundle(meta):
        for key, value in (('model', args.model), ('hardware', args.hardware),
                           ('variant', args.effective_variant)):
            if key in meta and meta[key] != value:
                raise ValueError('Native MoE output differs from the existing bundle')
            meta[key] = value
    update_bundle(variant_root, record_bundle)
    llm, _, temporary = spin_up(engine_args, 1)
    try:
        for target in targets:
            native = llm.collective_rpc('moe_component_initialize', args=(asdict(target), budget))[0]
            contract = dict(native, model=args.model, hardware=args.hardware, variant=args.effective_variant,
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
