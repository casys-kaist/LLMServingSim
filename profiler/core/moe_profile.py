"""Versioned native MoE acquisition; legacy whole-block tables stay intact."""
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

SCHEMA = 'moe-components-v1'
KEYS = ('component', 'mode', 'tokens', 'activated_experts', 'local_assignments')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


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
        self.contract = contract
        self.identity = fingerprint(contract)
        self.folder = Path(parent)/self.identity
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
            path = self.folder/'contract.json'
            if path.exists():
                if json.loads(path.read_text()) != contract:
                    raise ValueError("Corrupt measurement contract")
            else:
                atomic_json(path, contract)
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
        with (self.folder/'samples.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())

    def publish(self):
        expected = len(self.points)*self.rounds
        status = dict(schema=SCHEMA, contract_id=self.identity, complete=len(self.rows)==expected,
                      completed_rounds=len(self.rows), expected_rounds=expected, points=len(self.points))
        if status['complete'] and self.contract.get('expert_conditioning'):
            comparisons=[]
            tolerance=self.contract['expert_conditioning']['quality_relative_tolerance']
            for key,point in self.points.items():
                if point['component']!='experts': continue
                selected=[v for r in range(self.rounds) for v in self.rows[key,r]['result']['per_forward_us']]
                controls=[v for r in range(self.rounds) for v in self.rows[key,r]['result']['conditioning_control']['per_forward_us']]
                high,low=statistics.median(selected),statistics.median(controls)
                change=abs(high/low-1)
                comparisons.append(dict(point=point,selected_us=high,control_us=low,
                                        relative_change=change,supported=change<=tolerance))
            status['conditioning_controls']=comparisons
            status['quality_supported']=bool(comparisons) and all(c['supported'] for c in comparisons)
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
        atomic_json(self.folder/'coverage.json', status)
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
    from .writer import persist_moe_component_meta

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
    llm, kwargs, temporary = spin_up(engine_args, 1)
    index_entries = {}
    try:
        for target in targets:
            native = llm.collective_rpc('moe_component_initialize', args=(asdict(target), budget))[0]
            contract = dict(native, model=args.model, hardware=args.hardware, variant=args.effective_variant,
                model_config_sha256=fingerprint(args.model_config),
                protocol=dict(rounds=args.moe_rounds, iterations=args.measurement_iterations,
                              representative='median_of_forward_medians', timing='cuda_kernel_sum'),
                engine_effective={k:v for k,v in kwargs.items() if k not in ('model','tokenizer')},
                planning_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
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
                    failed = [c for c in status.get('conditioning_controls', []) if not c['supported']]
                    raise RuntimeError(f"Native MoE acquisition not publishable: {len(failed)} conditioning controls failed; see {store.folder/'coverage.json'}")
                index_entries.setdefault(target.tp, []).append(dict(target=asdict(target),
                    path=str(store.folder.relative_to(variant_root/f'tp{target.tp}')),
                    contract_id=store.identity, table_sha256=status['table_sha256']))
            finally:
                store.close()
                llm.collective_rpc('moe_component_release')
        for tp, entries in index_entries.items():
            path = variant_root/f'tp{tp}'/'moe_components.json'
            prior = json.loads(path.read_text()) if path.exists() else dict(schema=SCHEMA, entries=[])
            if prior['schema'] != SCHEMA:
                raise ValueError("Unknown existing MoE component index schema")
            replaced = {(e['target']['dp'], e['target']['rank']) for e in entries}
            retained = [e for e in prior['entries'] if (e['target']['dp'],e['target']['rank']) not in replaced]
            atomic_json(path, dict(schema=SCHEMA, entries=retained+entries))
        persist_moe_component_meta(args, variant_root, sorted(index_entries))
    finally:
        spin_down(llm, temporary)
