"""Deterministic, workload-independent heterogeneous attention acquisition.

Only one expanded batch is kept in memory at a time. The axes follow the
requested envelope and resolved engine limits, including the KV page size.
Distribution families and orderings are coverage probes, not fitted priors.
"""

from collections import Counter
import hashlib
import json
import math
import random

from .hooks.batch import Shot


SCHEMA = "heterogeneous-skew-sweep-v1"
FAMILIES = ("bimodal", "outliers", "trimodal", "ramp", "lognormal", "pareto", "near_uniform", "uniform")


def geometric(start, stop, factor):
    if not math.isfinite(factor) or factor <= 1:
        raise ValueError("Skew axis factors must be finite and greater than one")
    if stop < 1:
        return []
    values, value = set(), float(min(start, stop))
    while round(value) <= stop:
        values.add(max(1, round(value)))
        value *= factor
    return sorted(values | {stop})


def grid(args, limits):
    tokens = min(args.max_num_batched_tokens or limits.max_num_batched_tokens,
                 limits.max_num_batched_tokens)
    seqs = min(args.max_num_seqs or limits.max_num_seqs, limits.max_num_seqs)
    context = min(args.attention_max_kv or limits.max_model_len, limits.max_model_len - 2)
    if min(tokens, seqs, context, limits.block_size, limits.num_cache_tokens) < 1:
        raise ValueError("Skew acquisition requires positive engine limits")
    queries = sorted(set(args.attention_decode_q_lens))
    if any(type(q) is not int or q < 1 for q in queries):
        raise ValueError("Decode query lengths must be positive integers")
    return dict(tokens=tokens, seqs=seqs, context=context,
                n=geometric(2, seqs, args.skew_n_factor),
                pc=[0] + geometric(16, tokens, args.skew_pc_factor),
                kp=[0] + geometric(limits.block_size, context, args.skew_kp_factor),
                kv=geometric(limits.block_size, context, args.skew_kvs_factor),
                q=queries)


def _distribution(family, n, cap, rng):
    low = rng.randrange(max(1, cap // 2))
    if family == "bimodal":
        count = max(1, min(n - 1, round(n * rng.choice((.0625, .125, .25, .5, .75, .9)))))
        values = [cap] * count + [low] * (n - count)
    elif family == "outliers":
        count = min(n - 1, rng.choice((1, 2, 3, 4)))
        values = [cap] * count + [rng.randrange(max(1, cap // 16)) for _ in range(n - count)]
    elif family == "trimodal":
        values = [rng.choice((low, (low + cap) // 2, cap)) for _ in range(n)]
    elif family == "ramp":
        values = [round(cap * i / (n - 1)) for i in range(n)]
    elif family in ("lognormal", "pareto"):
        shape = rng.uniform(.6, 1.6) if family == "lognormal" else rng.uniform(1.25, 3)
        draws = [rng.lognormvariate(0, shape) if family == "lognormal"
                 else rng.paretovariate(shape) - 1 for _ in range(n)]
        maximum = max(draws)
        values = [round(cap * value / maximum) for value in draws]
    elif family == "near_uniform":
        values = [rng.randint(max(0, cap - max(1, cap // 16)), cap) for _ in range(n)]
    elif family == "uniform":
        values = [rng.randint(0, cap) for _ in range(n)]
    else:
        raise ValueError("Unknown skew distribution family")
    values[0], values[-1] = cap, min(values[-1], cap - 1)
    return values


def feasible(shot, limits, axes):
    if not shot.requests or len(shot.requests) > axes["seqs"]:
        return False
    if sum(q for q, _ in shot.requests) > axes["tokens"]:
        return False
    page = limits.block_size
    allocated = 0
    for query, history in shot.requests:
        if query < 1 or history < 0 or history > axes["context"]:
            return False
        if query + history + 1 > limits.max_model_len:
            return False
        allocated += ((query + history + page - 1) // page) * page
    return allocated <= limits.num_cache_tokens


def iter_cases(args, limits, *, cell_filter=None):
    """Yield unique feasible (Shot, family) pairs in breadth-first sample order.

    Each geometric operating cell gets the same number of distribution draws.
    Mixed shots also cover the token frontier and equal/unequal multi-prefill
    splits. Seeds depend only on coordinates, so widening another axis does
    not change existing draws. No benchmark file or latency enters planning.
    """
    axes = grid(args, limits)
    samples = args.skew_samples_per_cell
    if type(samples) is not int or samples < len(FAMILIES):
        raise ValueError(f"At least {len(FAMILIES)} skew samples per cell are required")
    seen = set()
    for draw in range(samples):
        family = FAMILIES[draw % len(FAMILIES)]
        for q in axes["q"]:
            max_n = min(axes["seqs"], axes["tokens"] // q)
            n_values = sorted(set(n for n in axes["n"] if 2 <= n <= max_n)
                              | ({max_n} if max_n >= 2 else set())
                              | ({max_n - 1} if max_n > 2 else set()))
            for n in n_values:
                for pc_target in axes["pc"]:
                    budget = min(pc_target, axes["tokens"] - n * q)
                    if pc_target and (budget <= q or n == axes["seqs"]):
                        continue
                    if cell_filter is not None and not cell_filter(q, n, budget):
                        continue
                    for cap_target in axes["kv"]:
                        cap = min(cap_target, limits.max_model_len - q - 1)
                        if cap < 1:
                            continue
                        seed = json.dumps([args.skew_seed, q, n, pc_target, cap_target, draw])
                        rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest(), 16))
                        histories = _distribution(family, n, cap, rng)
                        histories.sort()
                        # Independent draws avoid confounding a history family
                        # with one ordering or one prefill split.
                        order = rng.randrange(4)
                        if order == 1:
                            histories.reverse()
                        elif order == 2:
                            histories = histories[::2] + histories[1::2][::-1]
                        elif order == 3:
                            rng.shuffle(histories)
                        prefill = []
                        if budget:
                            count = min(rng.choice((1, 2, 4, 8)),
                                        axes["seqs"] - n, budget // (q + 1))
                            weights = [1.0] * count if rng.randrange(2) == 0 else [rng.random() + .01 for _ in range(count)]
                            remainder = budget - count * (q + 1)
                            chunks = [q + 1 + int(remainder * w / sum(weights)) for w in weights]
                            chunks[-1] += budget - sum(chunks)
                            if any(chunk + 1 > limits.max_model_len for chunk in chunks):
                                continue
                            for chunk in chunks:
                                candidates = [h for h in axes["kp"] if h <= cap_target
                                              and chunk + h + 1 <= limits.max_model_len]
                                history = 0 if rng.randrange(2) == 0 else rng.choice(candidates)
                                prefill.append((chunk, history))
                        shot = Shot(requests=prefill + [(q, h) for h in histories],
                                    n_prefill=len(prefill), decode_q_len=q)
                        if not feasible(shot, limits, axes):
                            continue
                        # Preserve the order that the native runner receives.
                        from .skew_calibration import measurement_shape
                        key = measurement_shape(dict(requests_json=json.dumps(shot.requests),
                            n_prefill=shot.n_prefill, decode_q_len=q))[4]
                        if key not in seen:
                            seen.add(key)
                            yield shot, family, key


def summarize(args, limits, completed=()):
    counts, query_counts, regimes = Counter(), Counter(), Counter()
    completed = set(completed)
    reused, max_prefill = 0, 0
    for shot, family, key in iter_cases(args, limits):
        counts[family] += 1
        reused += key in completed
        query_counts[shot.decode_q_len] += 1
        regimes["mixed" if shot.n_prefill else "decode"] += 1
        max_prefill = max(max_prefill, sum(query for query, _ in shot.requests[:shot.n_prefill]))
    return dict(schema=SCHEMA, grid=grid(args, limits), families=dict(counts),
                cases=sum(counts.values()), reusable_cases=reused,
                remaining_cases=sum(counts.values()) - reused,
                query_cases=dict(query_counts), regimes=dict(regimes),
                seed=args.skew_seed, samples_per_cell=args.skew_samples_per_cell,
                rounds=args.skew_rounds, timed_forwards_per_round=args.measurement_iterations,
                block_size=limits.block_size, num_cache_tokens=limits.num_cache_tokens,
                max_model_len=limits.max_model_len, max_prefill_tokens=max_prefill)
