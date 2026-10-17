"""CPU-only acquisition grids anchored to target serving capture shapes.

These are eager measurement coordinates, never a request to capture graphs.
Observed serving policies, acquisition capabilities and derived candidates have
different provenance; only a matching serving observation proves capture.
"""
from dataclasses import dataclass
import math
from pathlib import Path

from serving.core.cudagraph import resolve_graph_config
from .execution_metadata import graph_options, precision_scope, select_execution


def geometric(maximum, factor=2.0, *, start=1, zero=False):
    if not math.isfinite(factor) or factor <= 1:
        raise ValueError('Grid factor must be finite and greater than one')
    if type(start) is not int or start < 1:
        raise ValueError('Grid start must be a positive integer')
    if type(maximum) is not int or maximum < 0:
        raise ValueError('Grid maximum must be a nonnegative integer')
    points = {0} if zero else set()
    if maximum:
        points.add(maximum)
    value = float(start)
    while value < maximum:
        points.add(max(1, round(value)))
        value *= factor
    return sorted(points)


def union_grid(maximum, factor, anchors=(), *, start=1, zero=False):
    return sorted(set(geometric(maximum, factor, start=start, zero=zero))
                  | {n for n in anchors if 0 < n <= maximum})


def full_sizes(graph):
    if graph.mode == 'FULL':
        return graph.capture_sizes
    if graph.mode not in ('FULL_DECODE_ONLY', 'FULL_AND_PIECEWISE'):
        return ()
    if graph.runner == 'v2':
        return graph.decode_capture_sizes
    q = graph.query_length
    return tuple(n for n in graph.capture_sizes
                 if n % q == 0 and q <= n <= graph.max_num_seqs*q)


def forward_sizes(graph):
    if graph.mode == 'FULL_AND_PIECEWISE':
        # V2 full-decode descriptors round base buckets to a query multiple;
        # those model-row counts need not occur in the piecewise base grid.
        return tuple(sorted(set(graph.capture_sizes) | set(full_sizes(graph))))
    if graph.mode in ('FULL', 'PIECEWISE'):
        return graph.capture_sizes
    return full_sizes(graph)


@dataclass
class SamplingPlan:
    graphs: dict
    records: list
    stable_forward_sizes: tuple = ()

    @property
    def forward_sizes(self):
        return tuple(sorted({n for graph in self.graphs.values() for n in forward_sizes(graph)}
                            | set(self.stable_forward_sizes)))

    def decode_counts(self, q=1):
        graph = self.graphs[q]
        return tuple(n//q for n in full_sizes(graph)
                     if n % q == 0 and 1 <= n//q <= graph.max_num_seqs)

    def prefill_sizes(self):
        return tuple(sorted({n for graph in self.graphs.values() if graph.mode == 'FULL'
                             for n in graph.capture_sizes}))

    def mtp_counts(self):
        """Request-coordinate anchors, not a new first-pass MTP token axis."""
        counts = set()
        for q, graph in self.graphs.items():
            if q <= 1:
                continue
            if graph.runner == 'v1':
                if graph.mode in ('FULL', 'PIECEWISE', 'FULL_AND_PIECEWISE'):
                    counts.update(n for n in graph.capture_sizes if n <= graph.max_num_seqs)
            else:
                # V2 first-pass FULL descriptors carry padded request counts.
                counts.update(min(n, int(graph.max_num_seqs)) if graph.mode == 'FULL' else n//q
                              for n in full_sizes(graph))
                # N=1 has no subsequent single-query drafter graph at all.
                if q > 2 and graph.mode in ('FULL', 'FULL_DECODE_ONLY', 'FULL_AND_PIECEWISE'):
                    counts.update(n for n in graph.capture_sizes if n <= graph.max_num_seqs)
        return tuple(sorted(n for n in counts if n > 0))

    def metadata(self):
        value = dict(execution='eager', graph_role='sampling_coordinates', targets=self.records)
        if self.stable_forward_sizes:
            value['tp_stable_forward_tokens'] = list(self.stable_forward_sizes)
        return value


def resolve_sampling(args, limits, tp, *, dp=1, meta=None, include_stable=False):
    """Resolve target shapes from logical limits, never enlarged engine buffers."""
    runtime = getattr(limits, 'sampling_runtime', None) or {}
    meta = meta or {}
    capability = runtime.get('compute_capability')
    if capability is None:
        import yaml
        path = Path(__file__).resolve().parents[1]/'perf'/args.hardware/'hardware.yaml'
        if path.is_file():
            capability = (yaml.safe_load(path.read_text()) or {}).get('spec', {}).get('compute_capability')
    queries = sorted({1, *args.attention_decode_q_lens})
    if args.profile_mtp and not any(q > 1 for q in queries):
        queries.append(2)  # the acquisition engine builds one speculative step
    graphs, records, stable = {}, [], set()
    tps = sorted(set(args.tp_degrees) | {tp}) if include_stable and tp == 1 else [tp]
    for target_tp in tps:
        for q in queries:
            scope = dict(tp_size=target_tp, pp_size=1, dp_size=dp,
                         ep_size=target_tp*dp if _has_experts(args) else 1,
                         max_num_seqs=limits.max_num_seqs,
                         max_num_batched_tokens=limits.max_num_batched_tokens,
                         num_speculative_tokens=q-1, block_size=limits.block_size,
                         max_model_len=limits.max_model_len,
                         enable_chunked_prefill=runtime.get('enable_chunked_prefill', True),
                         enable_prefix_caching=runtime.get('enable_prefix_caching', False))
            scope.update(precision_scope(args.model_config or {'model_type': args.architecture},
                runtime.get('activation_dtype', args.dtype),
                runtime.get('kv_cache_dtype', args.kv_cache_dtype), runtime.get('quantization')))
            options = (None if args.cudagraph is None else
                       dict(runner=runtime.get('runner') or 'v1') | args.cudagraph)
            source, reason = 'explicit target options', None
            if options is None:
                if runtime.get('gpu') is not None and meta.get('gpu') is not None and runtime['gpu'] != meta['gpu']:
                    observation, reason = None, 'bundle device does not match acquisition device'
                else:
                    observation, reason = select_execution(meta, scope,
                        args.model_config or {'model_type': args.architecture})
                if observation is not None:
                    options, source = graph_options(observation), 'observed worker capture'
                else:
                    source = 'derived capture candidates'
                    options = dict(runner=runtime.get('runner') or 'v1')
                    support = runtime.get('attention_cg_support')
                    if support == 'NEVER' or (q > 1 and support == 'UNIFORM_SINGLE_TOKEN_DECODE'):
                        options['mode'] = 'PIECEWISE'
            graph = resolve_graph_config(options, limits.max_num_seqs,
                limits.max_num_batched_tokens, q-1, target_tp, capability)
            record = dict(tp=target_tp, dp=dp, query_length=q, source=source,
                          runner=graph.runner, mode=graph.mode,
                          capture_sizes=list(graph.capture_sizes),
                          full_capture_sizes=list(full_sizes(graph)),
                          compute_capability=capability)
            if source == 'derived capture candidates':
                record['attention_cg_support'] = runtime.get('attention_cg_support')
                record['capability_scope'] = 'acquisition backend, not full-model capture'
                record['assumptions'] = 'default compiled serving with piecewise support; no LoRA, DBO or context parallelism'
            if source == 'observed worker capture':
                record['device_verified'] = runtime.get('gpu') is not None
            if reason:
                record['observation_fallback'] = reason
            records.append(record)
            if target_tp == tp:
                graphs[q] = graph
            if include_stable and tp == 1:
                stable.update(forward_sizes(graph))
    return SamplingPlan(graphs, records, tuple(sorted(stable)))


def _has_experts(args):
    from .config import probe_moe_params
    return probe_moe_params(args.model_config or {}) is not None


def for_category(args, limits, tp, *, include_stable=False):
    """Host composition also works in CPU planning without a live engine."""
    plan = getattr(limits, 'sampling_plan', None)
    return plan if plan is not None else resolve_sampling(args, limits, tp, include_stable=include_stable)


def record_category(out_dir, category, args, plan, shots):
    """Record the feasible plan, not a claim that its measurements completed."""
    from .bundle_metadata import update_tp
    record = plan.metadata()
    record.update(grid_factor=args.grid_factor, feasible_shots=len(shots))
    if category.name == 'attention':
        record['factors'] = dict(chunk=args.attention_chunk_factor,
                                kv=args.attention_kv_factor, n=args.attention_n_factor)
        record['graph_anchors'] = 'pure regimes only; mixed retains geometric axes'
    elif category.name == 'mtp':
        record['graph_role'] = 'drafter request anchors; first-pass query/token layout is not represented by this table'
        record['capture_request_anchors'] = list(plan.mtp_counts())
    elif category.name == 'per_sequence':
        record['graph_role'] = 'none; target logits and sampler use actual output rows'
        record.pop('targets')
        record.pop('tp_stable_forward_tokens', None)
    if category.name == 'dense':
        actual = {sum(n for n, _ in shot.requests) for shot in shots}
        missing = set(plan.forward_sizes) - actual
        if missing:
            record['infeasible_capture_tokens'] = sorted(missing)
            from . import logger as log
            log.warning('Dense capture coordinates excluded by runtime limits: %s', sorted(missing))
    update_tp(out_dir, 'sampling', lambda previous: dict(previous or {}, **{category.name: record}))
