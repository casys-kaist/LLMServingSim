"""Aggregate a real gate's distinct-expert count into ``gate_stats.json``.

The simulator prices an MoE block from ``moe.csv`` at an ``activated_experts``
coordinate it derives from a **uniform** gate::

    activated = E * (1 - ((E - k) / E) ** n)

A trained gate concentrates on popular experts, so the real distinct count is
lower -- measured on Qwen3-30B-A3B, 0.87x of the uniform model through the
middle of the range and 0.94x at a saturated decode. The closed form cannot
know that: the concentration is in the weights.

The alternative to guessing is to *measure* it. `bench run --record-gate-stats`
turns on the ``VLLM_MOE_ACTIVATED_LOG`` source patch, which logs
``(tokens, distinct, top_k)`` for every ``select_experts`` call, and this module
reduces that log to one curve the simulator reads back. Nothing here is fitted
-- the curve is the measurement, interpolated between the token counts the run
actually visited.

**Why not force the truth to be uniform instead.** A bench mode that replaced
the gate's assignment was tried and does not answer the question: on a non-EP
configuration under cudagraphs it reads **0.700x** because flattening the
per-expert histogram lands the grouped GEMM on a *different kernel variant*
(an in-situ profile of 8 real decode steps shows a ``MoeFCGemm`` instantiation
appearing with 144 calls and another dropping to zero, with attention
unchanged). It cannot hold "everything but the count" fixed, so it measures the
schedule rather than the count. Measuring the count directly has no such
coupling.

**Requires ``--enforce-eager``.** The patch calls ``.unique()`` on the router's
output, which is a data-dependent shape and cannot be captured into a cudagraph.
That costs nothing in fidelity here: the gate's top-k output is a function of
the weights and the input, not of how the forward is executed, so a curve
recorded eagerly describes the compiled run's gate exactly.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

FILENAME = "gate_stats.json"


def _is_uniform_token_batch(tokens: int, distinct: int, top_k: int) -> bool:
    """vLLM's dummy batches repeat one token id, so they route as one token.

    A profiling or cudagraph-capture forward submits ``n`` copies of the same
    token, and every copy selects the same ``top_k`` experts -- Qwen3-30B-A3B
    logged ``distinct = 8`` at 128 tokens, where a real batch of that size
    measures 120.6. Those rows describe vLLM's warmup, not a workload, and
    averaging them in drags the curve down at exactly the capture sizes.

    Real text never lands here: at ``n = top_k`` the same model measures 47.6
    distinct against this rule's ceiling of 8.
    """
    return tokens > top_k and distinct <= top_k


def aggregate(log_path: str | Path) -> dict[str, Any] | None:
    """Reduce a ``VLLM_MOE_ACTIVATED_LOG`` jsonl to one curve.

    Returns ``None`` when the log is missing or holds no usable row -- the
    caller then writes nothing and the simulator falls back to its closed
    form, which is the documented behaviour when no measurement exists.
    """
    path = Path(log_path)
    if not path.exists():
        return None

    by_tokens: dict[int, list[int]] = defaultdict(list)
    top_k = 0
    n_dropped = 0
    n_bad = 0
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                tokens = int(rec["tokens"])
                distinct = int(rec["distinct"])
                k = int(rec["top_k"])
            except (ValueError, KeyError, TypeError):
                n_bad += 1
                continue
            top_k = top_k or k
            if _is_uniform_token_batch(tokens, distinct, k):
                n_dropped += 1
                continue
            by_tokens[tokens].append(distinct)

    if not by_tokens:
        return None

    curve = [
        [n, sum(v) / len(v), len(v)]
        for n, v in sorted(by_tokens.items())
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "num_experts_per_tok": top_k,
        "n_calls": sum(c[2] for c in curve),
        "n_dropped_uniform_token": n_dropped,
        "n_unparsed": n_bad,
        # tokens, mean distinct experts, samples. ``tokens`` is the router's
        # own input length, i.e. the post-all-gather count under EP/DP -- the
        # same quantity the simulator's ``_balanced_route_ep`` is handed.
        "curve": curve,
    }


def write(output_dir: str | Path, log_path: str | Path,
          model: str, num_experts: int) -> dict[str, Any] | None:
    """Aggregate and write ``gate_stats.json`` beside the run's other output.

    ``model`` and ``num_experts`` are recorded so a reader can refuse a curve
    measured on a different checkpoint: the count is a property of *these*
    weights and ``E``, and applying one model's curve to another is worse than
    the closed form, which at least knows the right ``E``.
    """
    payload = aggregate(log_path)
    if payload is None:
        return None
    payload["model"] = model
    payload["num_experts"] = int(num_experts)
    out = Path(output_dir) / FILENAME
    out.write_text(json.dumps(payload, indent=2))
    return payload
