"""Synthetic batch construction for profiling shots.

A ``Shot`` is one unit of work we hand to the vLLM worker: a list of
(new_tokens, history) pairs describing the shape of every request in a
synthetic batch, plus optional MoE routing hints. A ``Shot`` is
serialized to a plain dict for cross-process transport (via
``llm.collective_rpc``) and rehydrated inside the worker.

``assemble_scheduler_output`` turns a ``Shot`` into a fully-formed
``SchedulerOutput`` that vLLM's ``model_runner.execute_model`` can
consume. We bypass the vLLM scheduler entirely so that the shapes of
the requests are exactly what the grid generators asked for — no
risk of the scheduler splitting, chunking, or reordering.

Key trick: setting ``num_computed_tokens = history`` tells vLLM
"pretend the first `history` tokens are already computed and their KV
is in the cache". Combined with ``prompt_token_ids = [1] * (new_tokens
+ history)`` this gives the engine a request that attends to
``history`` preloaded tokens while newly computing ``new_tokens``.
Exactly the shape needed to sweep attention at arbitrary
(prefill_chunk, kv_cache) configurations.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Shot:
    """One profiling shot — the minimal description of a batch to run.

    Attributes:
        requests: Ordered list of ``(new_tokens, history)`` pairs. One
            entry per synthetic request. ``new_tokens`` is what we'll
            feed into ``execute_model`` this step; ``history`` is the
            KV the engine should treat as preloaded.
        experts: Optional MoE payload. When set, the worker activates
            exactly ``experts["activated"]`` experts via forced
            routing (see moe_hook.force_moe_routing). None for
            non-MoE shots.
    """

    requests: list[tuple[int, int]]
    experts: dict[str, Any] | None = None

    # Serialization roundtrip: these helpers keep cross-process
    # transport simple. collective_rpc serializes args as pickle, so
    # plain dicts are safer than dataclass instances.
    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def hydrate(cls, raw: dict[str, Any]) -> "Shot":
        return cls(
            requests=[tuple(r) for r in raw.get("requests", [])],
            experts=raw.get("experts"),
        )

    # -----------------------------------------------------------------
    # Convenience constructors used by categories.py
    # -----------------------------------------------------------------

    @classmethod
    def dense(cls, total_tokens: int) -> "Shot":
        """One request carrying ``total_tokens`` new tokens, no KV history."""
        return cls(requests=[(total_tokens, 0)])

    @classmethod
    def per_sequence(cls, num_sequences: int) -> "Shot":
        """``num_sequences`` one-token requests.

        Used for lm_head / sampler profiling where cost scales with the
        number of output sequences, not total tokens.
        """
        return cls(requests=[(1, 0)] * num_sequences)

    @classmethod
    def attention(
        cls,
        prefill_chunk: int,
        kv_prefill: int,
        n_decode: int,
        kv_decode: int,
    ) -> "Shot":
        """Mixed prefill+decode batch for unified attention profiling.

        At most one prefill + ``n_decode`` decode requests. Either
        component can be absent (``prefill_chunk=0`` or ``n_decode=0``).
        """
        reqs: list[tuple[int, int]] = []
        if prefill_chunk > 0:
            reqs.append((prefill_chunk, kv_prefill))
        if n_decode > 0:
            reqs.extend([(1, kv_decode)] * n_decode)
        if not reqs:
            raise ValueError("attention Shot must have at least one request")
        return cls(requests=reqs)

    @classmethod
    def moe(cls, total_tokens: int, activated_experts: int) -> "Shot":
        """Dense-style batch tagged with MoE routing metadata."""
        return cls(
            requests=[(total_tokens, 0)],
            experts={"activated": activated_experts},
        )


# ---------------------------------------------------------------------------
# SchedulerOutput assembly (worker-side)
# ---------------------------------------------------------------------------
#
# Imports are deferred to function-call time so that this module can be
# imported from the host side (where vLLM internals may not be the
# version we run in the worker) without pulling in every vLLM symbol.


def assemble_scheduler_output(shot: Shot, model_runner):
    """Build a ``SchedulerOutput`` describing the shot's synthetic batch.

    Returns:
        A tuple ``(scheduler_output, req_ids)``. The second element is
        the set of request IDs we created so callers can correlate
        with ``execute_model``'s output if needed.
    """
    # Local imports: these symbols must come from whatever vLLM is
    # actually installed in the worker — not from a cached import at
    # host-side module load time.
    from vllm import SamplingParams
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        NewRequestData,
        SchedulerOutput,
    )

    # Greedy single-step sampling — we don't care about token quality,
    # only about kernel shapes.
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=1,
    )

    # vLLM may have multiple KV-cache groups (cross-layer managers,
    # hybrid architectures). We honor all of them by reading the
    # worker's live block_table list.
    block_tables = model_runner.input_batch.block_table.block_tables
    block_sizes = [
        bt.block_size * bt.blocks_per_kv_block
        for bt in block_tables
    ]
    num_kv_groups = len(block_sizes)

    scheduled: list = []
    num_scheduled_tokens: dict[str, int] = {}
    total_num_scheduled_tokens = 0

    # Track the next free block index per KV group. We assign blocks
    # greedily in arrival order; since these are fresh dummy requests
    # there's no overlap concern.
    block_cursor = [0] * num_kv_groups
    req_ids: list[str] = []

    for idx, (new_tokens, history) in enumerate(shot.requests):
        req_id = f"r{idx}"
        total_len = new_tokens + history

        # For each KV group, reserve enough blocks to cover total_len.
        # We round up because partial blocks still consume one slot.
        group_block_ids: list[list[int]] = []
        for g, bs in enumerate(block_sizes):
            num_blocks = math.ceil(total_len / bs) if total_len else 1
            ids = list(range(block_cursor[g], block_cursor[g] + num_blocks))
            block_cursor[g] += num_blocks
            group_block_ids.append(ids)

        scheduled.append(
            NewRequestData(
                req_id=req_id,
                # Contents don't matter — we use token id 1 uniformly.
                # Length must equal `history + new_tokens` so vLLM
                # thinks it's handling a real sequence.
                prompt_token_ids=[1] * total_len,
                mm_features=[],
                sampling_params=sampling_params,
                pooling_params=None,
                block_ids=tuple(group_block_ids),
                # This is the "KV cache already contains `history`
                # tokens" marker — the crux of how we inject arbitrary
                # kv_cache shapes without actually prefilling.
                num_computed_tokens=history,
                lora_request=None,
            )
        )
        num_scheduled_tokens[req_id] = new_tokens
        total_num_scheduled_tokens += new_tokens
        req_ids.append(req_id)

    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=scheduled,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0] * num_kv_groups,
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    return scheduler_output, set(req_ids)
