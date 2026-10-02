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
This observes routing, not unperturbed end-to-end latency. A curve describes
the recorded inputs and weights; different batching or numerics need controls.

Startup and workload calls are separated by explicit phase markers from the
driver. Concentrated routing is valid workload data, not evidence of warmup.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2

FILENAME = "gate_stats.json"


def mark_phase(log_path: str | Path, phase: str) -> None:
    """Delimit workload observation outside request execution."""
    if phase not in ("workload_start", "workload_end"):
        raise ValueError("Unknown gate-observation phase")
    with Path(log_path).open("a") as stream:
        stream.write(json.dumps({"event": phase, "schema_version": SCHEMA_VERSION}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def aggregate(log_path: str | Path) -> dict[str, Any] | None:
    """Reduce a ``VLLM_MOE_ACTIVATED_LOG`` jsonl to one curve.

    Returns ``None`` when the log is missing or holds no usable row -- the
    caller then writes nothing and the simulator falls back to its closed
    form, which is the documented behaviour when no measurement exists.
    Unmarked, interrupted or multiply delimited logs are refused: expert
    counts alone cannot distinguish warmup from real concentrated routing.
    """
    path = Path(log_path)
    if not path.exists():
        return None

    by_tokens: dict[int, list[int]] = defaultdict(list)
    top_k = 0
    phase = "startup"
    n_startup = n_shutdown = 0
    n_bad = 0
    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                event = rec.get("event")
                if event in ("workload_start", "workload_end"):
                    expected = "startup" if event == "workload_start" else "workload"
                    if phase != expected or rec.get("schema_version") != SCHEMA_VERSION:
                        raise RuntimeError("Invalid or repeated gate-observation boundary")
                    phase = "workload" if event == "workload_start" else "shutdown"
                    continue
                tokens = int(rec["tokens"])
                distinct = int(rec["distinct"])
                k = int(rec["top_k"])
                if tokens < 1 or k < 1 or not k <= distinct <= tokens * k:
                    raise ValueError("Invalid gate observation")
            except (ValueError, KeyError, TypeError):
                n_bad += 1
                continue
            if phase == "startup":
                n_startup += 1
                continue
            if phase == "shutdown":
                n_shutdown += 1
                continue
            if top_k and top_k != k:
                raise ValueError("Mixed top-k values cannot form one routing curve")
            top_k = k
            by_tokens[tokens].append(distinct)

    if phase != "shutdown":
        raise ValueError("Gate log lacks complete workload boundaries; re-record with --record-gate-stats")
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
        "observation_scope": "explicit_workload_boundaries",
        "n_dropped_startup": n_startup,
        "n_dropped_shutdown": n_shutdown,
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
