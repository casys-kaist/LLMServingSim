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
    # Query tokens per decode request. Carried rather than re-derived from
    # ``requests``, because with more than one query token a decode is
    # indistinguishable by shape from a prefill chunk -- which is exactly the
    # confusion this axis exists to remove.
    decode_q_len: int = 1

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
            decode_q_len=int(raw.get("decode_q_len") or 1),
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
        decode_q_len: int = 1,
    ) -> "Shot":
        """Mixed prefill+decode batch for unified attention profiling.

        At most one prefill + ``n_decode`` decode requests. Either
        component can be absent (``prefill_chunk=0`` or ``n_decode=0``).

        ``decode_q_len`` is how many query tokens each decode request submits.
        It is 1 for ordinary decoding and ``1 + num_speculative_tokens`` for a
        speculative-decoding verification step, which is a shape the other four
        axes cannot express: *n* sequences each submitting *k+1* queries against
        **their own** KV is neither one prefill chunk of ``n*(k+1)`` tokens nor
        ``n*(k+1)`` single-token decodes, because the k+1 queries of one
        sequence share that sequence's KV read. FlashAttention runs it as a
        varlen batch of uniform query length, which is a different tile shape
        again.
        """
        reqs: list[tuple[int, int]] = []
        if prefill_chunk > 0:
            reqs.append((prefill_chunk, kv_prefill))
        if n_decode > 0:
            reqs.extend([(max(1, decode_q_len), kv_decode)] * n_decode)
        if not reqs:
            raise ValueError("attention Shot must have at least one request")
        return cls(requests=reqs, decode_q_len=max(1, decode_q_len))

    # History given to a linear-attention shot's decode requests. The value is
    # arbitrary as far as cost goes -- a gated-DeltaNet state is fixed-size
    # regardless of position, and a 64x spread in this number moves the
    # measured time 1.1%. It is not arbitrary as far as *classification* goes;
    # see ``Shot.linear_attention``. One block covers it at every block size
    # the hybrid page unification produces, so it is free.
    LINEAR_ATTN_DECODE_HISTORY = 256

    @classmethod
    def linear_attention(
        cls,
        prefill_tokens: int,
        n_decode: int,
        decode_history: int | None = None,
    ) -> "Shot":
        """Mixed prefill+decode batch for a linear-attention (mamba / GDN) sweep.

        Unlike ``Shot.attention`` there is no kv **axis**: a gated-DeltaNet
        layer keeps a fixed-size conv state and a fixed-size recurrent state
        per sequence, neither a function of position, so cost does not depend
        on how long the sequences are — measured, a 64x spread in kv length
        moves it 1.1% and a skewed batch is indistinguishable from a uniform
        one. There is no skew correction to apply either.

        The decodes still carry history, though, because vLLM's *classification*
        depends on it even where the cost does not.
        ``split_decodes_and_prefills`` assumes a decodes-first batch and
        short-circuits:

            if query_lens[0].item() > decode_threshold:
                # first request is not decode, so no decode requests
                return 0, num_reqs, 0, num_tokens

        A pure batch of 1-token requests takes an earlier fast path
        (``max_query_len <= threshold`` => all decodes) and reaches the decode
        kernel with no history at all. A **mixed** batch does not: with
        zero-history decodes the whole batch was classified as prefill and the
        decode kernel never ran, so the mixed-regime rows came out empty.
        Since GDN runs a *different* kernel in the mixed regime than in the
        pure one, those rows are exactly the ones that cannot be inferred from
        anywhere else.
        """
        history = (
            cls.LINEAR_ATTN_DECODE_HISTORY if decode_history is None
            else decode_history
        )
        reqs: list[tuple[int, int]] = []
        if prefill_tokens > 0:
            reqs.append((prefill_tokens, 0))
        if n_decode > 0:
            reqs.extend([(1, history)] * n_decode)
        if not reqs:
            raise ValueError(
                "linear_attention Shot needs prefill_tokens or n_decode"
            )
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


def _kv_group_block_sizes(model_runner) -> list[int]:
    """Block size of each KV cache group, in tokens, as the KV manager counts them.

    A model can have several KV cache groups — cross-layer managers, and any
    hybrid stack, where full-attention layers page KV per token while mamba /
    linear-attention layers hold one fixed-size state per sequence. Each group
    has its own block size, and a request occupies blocks in *every* group at
    once, so the caller has to reserve for all of them.

    Read from ``kv_cache_config`` rather than from the worker's block tables:
    v0.28 ships two GPU model runners and defaults to the newer one, which
    keeps no persistent ``input_batch`` (it builds one per step) and holds its
    block tables under a different type. Both runners derive their tables from
    ``kv_cache_config``, so that is the version-independent source — and it is
    the KV-manager block size directly, with no kernel-block arithmetic.
    """
    kv_cache_config = getattr(model_runner, "kv_cache_config", None)
    if kv_cache_config is not None:
        groups = getattr(kv_cache_config, "kv_cache_groups", None)
        if groups:
            return [int(g.kv_cache_spec.block_size) for g in groups]

    # Legacy fallback: the V1 runner's persistent input batch. ``block_size``
    # there is the *kernel* block size, which equals the manager's only when
    # blocks aren't subdivided — hence the multiply.
    block_tables = model_runner.input_batch.block_table.block_tables
    return [int(bt.block_size) * int(bt.blocks_per_kv_block) for bt in block_tables]


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

    block_sizes = _kv_group_block_sizes(model_runner)
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
                # V2-model-runner only, and it asserts rather than defaults:
                # ``add_requests`` passes this straight through as the
                # request's ``all_token_ids``. vLLM's own scheduler fills it
                # with ``req._all_token_ids`` (prompt plus everything
                # generated so far), which for a fresh synthetic request is
                # just the prompt again.
                prefill_token_ids=[1] * total_len,
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

    # Start from vLLM's own empty instance rather than naming every field.
    # SchedulerOutput grows a field or two most releases (v0.28 alone added
    # eight, plus spec-decode bookkeeping); constructing positionally means
    # each of those is a breakage we'd have to chase. ``make_empty`` is
    # maintained alongside the dataclass, so it always fills whatever the
    # installed version requires, and we override only what a shot defines.
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_new_reqs = scheduled
    scheduler_output.num_scheduled_tokens = num_scheduled_tokens
    scheduler_output.total_num_scheduled_tokens = total_num_scheduled_tokens
    scheduler_output.num_common_prefix_blocks = [0] * num_kv_groups
    return scheduler_output, set(req_ids)
