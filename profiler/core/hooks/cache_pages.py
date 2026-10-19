"""Physical page allocation shared by synthetic requests and KV preparation."""


def allocate_counts(counts_by_group, num_blocks):
    """Group-major contiguous allocations from one physical pool, reserving 0.

    Return request-major tuple-of-group block IDs, matching NewRequestData.
    Counts include every group's actual physical page need, not a token proxy.
    """
    if type(num_blocks) is not int or num_blocks < 2 or not counts_by_group:
        raise ValueError('A shared cache pool with a reserved null block is required')
    n_requests = len(counts_by_group[0])
    if not n_requests or any(len(row) != n_requests for row in counts_by_group):
        raise ValueError('Inconsistent group/request page counts')
    if any(type(n) is not int or n < 1 for row in counts_by_group for n in row):
        raise ValueError('Positive integer page counts required')
    required = sum(sum(row) for row in counts_by_group)
    if required + 1 > num_blocks:
        raise ValueError(f'Shared cache capacity exceeded: {required} assigned + 1 null > {num_blocks}')
    result = [[] for _ in range(n_requests)]
    cursor = 1
    for row in counts_by_group:
        for request, count in enumerate(row):
            result[request].append(list(range(cursor, cursor + count)))
            cursor += count
    return [tuple(groups) for groups in result]


def allocate_shot_blocks(shot, runner):
    """Use installed generic cache specs; no model-family or version branch."""
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs
    cache = runner.kv_cache_config
    counts = []
    for group in cache.kv_cache_groups:
        group_spec = group.kv_cache_spec
        specs = (list(group_spec.kv_cache_specs.values())
                 if isinstance(group_spec, UniformTypeKVCacheSpecs) else [group_spec])
        per_request = []
        for new_tokens, history in shot.requests:
            if (type(new_tokens) is not int or type(history) is not int
                    or new_tokens < 1 or history < 0):
                raise ValueError('Invalid synthetic request geometry')
            needs = []
            for spec in specs:
                size = int(spec.block_size)
                if size < 1:
                    raise ValueError('Invalid group cache block size')
                count = (new_tokens + history + size - 1) // size
                if isinstance(spec, MambaSpec):
                    mode = runner.vllm_config.cache_config.mamba_cache_mode
                    extra = int(spec.num_speculative_blocks)
                    if extra < 0:
                        raise ValueError('Invalid speculative state count')
                    if mode == 'none':
                        count = 1 + extra
                    elif mode == 'all':
                        count += extra
                    else:
                        # align needs sparse/null historical entries and checkpoint
                        # placement. Prefix-disabled profiler boots use none.
                        raise ValueError(f'Unsupported synthetic Mamba state cache mode: {mode}')
                needs.append(count)
            per_request.append(max(needs))
        counts.append(per_request)
    return allocate_counts(counts, int(cache.num_blocks))


def assigned_group_spans(requests, groups, num_blocks):
    """Validate disjoint group spans, IDs in range, and the actual total budget."""
    spans, seen = [], set()
    for index, _group in enumerate(groups):
        pages = [p for request in requests for p in request.block_ids[index]]
        if any(type(page) is not int for page in pages):
            raise ValueError('Physical cache IDs must be integers')
        if (not pages or pages[0] < 1 or pages[-1] >= num_blocks
                or pages != list(range(pages[0], pages[0] + len(pages)))):
            raise ValueError('Assigned cache pages must form a non-null in-range group span')
        if seen.intersection(pages):
            raise ValueError('Synthetic cache groups alias physical pages')
        seen.update(pages)
        spans.append((pages[0], len(pages)))
    if len(seen) + 1 > num_blocks:
        raise ValueError('Assigned shared-pool pages exceed capacity')
    return spans


def select_pages(cache, axis, first_block, count, num_blocks):
    """Select the actual physical span, including backend-subdivided blocks.

    The caller obtains axis from the existing backend/logical-layout resolver.
    For recurrent state, axis=0 and cache.shape[0]=num_blocks.
    """
    if (type(first_block) is not int or type(count) is not int or
            first_block < 1 or count < 1 or first_block + count > num_blocks):
        raise ValueError('Invalid physical page span')
    if cache.shape[axis] % num_blocks:
        raise ValueError('Backend block count is not a multiple of manager blocks')
    factor = cache.shape[axis] // num_blocks
    if factor < 1:
        raise ValueError('No physical cache blocks')
    return cache.narrow(axis, first_block * factor, count * factor)
