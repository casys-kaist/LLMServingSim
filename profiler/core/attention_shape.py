"""Attention coordinates shared by profiling and simulation, without GPU imports."""


def prefill_key(requests, cap=None):
    """Query-weighted causal key length for ``(chunk, history)`` pairs.

    Keep the grid's continuous ``history + chunk / 2`` convention. Equal
    chunks and single-request shots retain their old coordinate. A cap, when
    supplied, clips each request's effective window before weighting; this
    retains the existing sparse-window approximation, not an exact per-query
    integral through a window boundary.
    """
    total = 0
    work = 0.0
    for chunk, history in requests:
        if chunk < 0 or history < 0:
            raise ValueError("prefill chunks and histories must be non-negative")
        key = history + chunk / 2.0
        if cap is not None:
            key = min(key, cap)
        total += chunk
        work += chunk * key
    return work / total if total else 0.0


def partition_q1_requests(requests):
    """Use serving's non-speculative classification, independent of input labels.

    A one-query prompt tail and a one-query decode have the same attention
    shape. Serving places both in the decode pool. This is deliberately not
    a speculative-query classifier: those batches also carry verification
    state which cannot be recovered from query counts alone.
    """
    prefill, decode = [], []
    for query, history in requests:
        if (type(query) is not int or type(history) is not int or
                query < 1 or history < 0):
            raise ValueError('Expected positive query lengths and nonnegative histories')
        if query == 1:
            decode.append(history)
        else:
            prefill.append((query, history))
    return prefill, decode
