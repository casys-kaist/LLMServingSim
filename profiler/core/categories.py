"""Profile categories.

Each category knows three things:
  1. How to generate the list of Shots that make up its sweep
     (``compose_shots``).
  2. How to convert the raw per-layer timings returned by the worker
     into CSV-bound Points (``extract_points``).
  3. Which slice of the ModelSpec's catalog it cares about
     (``catalog_slice``).

Four concrete categories:
    DenseCategory       token-parameterized layers (embedding, qkv_proj, ...)
    SequenceCategory    sequence-parameterized layers (lm_head, sampler)
    AttentionCategory   the unified prefill+decode+mixed attention grid
    ExpertCategory      MoE block (tokens × activated_experts)

Adding a new profile kind is a matter of subclassing ``Category`` and
registering it in ``categories_for()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Iterator

from profiler.core.config import (
    Architecture,
    LayerEntry,
    ProfileArgs,
    declares_moe,
)
from profiler.core.engine import RuntimeLimits
from profiler.core.hooks.batch import Shot
from profiler.core.stack import ALL_AXES, ATTENTION_AXES
from profiler.core.hooks.timings import TimingSample


# ---------------------------------------------------------------------------
# Point types — one per category, shaped by the CSV schema for that kind.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DensePoint:
    layer: str
    tokens: int
    microseconds: float


@dataclass(frozen=True)
class SequencePoint:
    layer: str
    sequences: int
    microseconds: float


@dataclass(frozen=True)
class MtpPoint:
    layer: str
    sequences: int
    microseconds: float


@dataclass(frozen=True)
class AttentionPoint:
    # 4D keyed, one column per axis, plus microseconds -- and a layer name,
    # because the group can legitimately hold more than one kernel now. A
    # sparse-attention model runs an indexer over the whole KV before the
    # top-k selection, so it keys on exactly these axes and runs on every
    # attention layer, but it is a different kernel with a different cost.
    layer: str
    prefill_chunk: int
    kv_prefill: int
    n_decode: int
    kv_decode: int
    # Query tokens per decode sequence: 1 for ordinary decoding, 1 + N for a
    # speculative-decoding verification step. Neither a prefill chunk of the
    # same token count nor that many single-token decodes -- the k+1 queries of
    # one sequence share that sequence's KV read.
    decode_q_len: int
    microseconds: float


@dataclass(frozen=True)
class LinearAttentionPoint:
    # Carries ``layer`` unlike AttentionPoint: a linear-attention block runs
    # several distinct kernels that key on the same axes -- the chunked prefill
    # scan, and two different decode recurrences depending on whether the batch
    # also holds a prefill -- and they are not interchangeable.
    layer: str
    prefill_tokens: int
    n_decode: int
    microseconds: float


@dataclass(frozen=True)
class ExpertPoint:
    tokens: int
    activated_experts: int
    microseconds: float


# Union alias for writer.py's benefit.
Point = (
    DensePoint | SequencePoint | MtpPoint | AttentionPoint | LinearAttentionPoint
    | ExpertPoint
)


# ---------------------------------------------------------------------------
# Grid helpers (shared)
# ---------------------------------------------------------------------------

def _power_of_two_grid(max_value: int) -> list[int]:
    """``[1, 2, 4, 8, ..., largest_pow2_le_max]``."""
    values: list[int] = []
    v = 1
    while v <= max_value:
        values.append(v)
        v *= 2
    return values


def _token_grid(max_tokens: int) -> list[int]:
    """Dense grid used for both dense and per_sequence sweeps.

    Fine points at the low end (where decode-sized batches live),
    coarser at the high end. Matches the shape of vLLM's typical
    runtime load — small batches dominate.
    """
    if max_tokens < 1:
        return []
    pts: list[int] = []
    # 1 .. 15 — every integer. Short-decode regime.
    pts.extend(range(1, min(16, max_tokens + 1)))
    # 16 .. 63 — step of 4. Transition regime.
    pts.extend(range(16, min(64, max_tokens + 1), 4))
    # 64 .. max — step of 16. Longer-chunk regime.
    pts.extend(range(64, max_tokens + 1, 16))
    # max might not be captured by the strided range above.
    if pts[-1] != max_tokens:
        pts.append(max_tokens)
    return pts


# ---------------------------------------------------------------------------
# Category base
# ---------------------------------------------------------------------------

class Category(ABC):
    """Abstract base. Subclasses fill in ``name``, ``sink_filename``,
    ``label``, and the three methods below."""

    # ``name`` is the ``kind`` string the worker's ``fire(...)`` method
    # sees; ``sink_filename`` is the CSV output file; ``label`` is what
    # shows up in the progress bar.
    name: ClassVar[str]
    sink_filename: ClassVar[str]
    label: ClassVar[str]

    #: Which axes of the layer stack this category's measurements depend on.
    #: The engine is shrunk to the smallest prefix instantiating every value of
    #: these, and a category should not pay for an axis it does not measure:
    #: the profile tree merges same-class siblings, so a second layer of a type
    #: already present adds no information, only its whole op count on every
    #: shot -- 372 ms of profiling overhead per forward at 4 layers of
    #: DeepSeek-V3.2 against 94 ms at one. The default is every axis, and it is
    #: the safe answer for anything new.
    #:
    #: Axes are a **conservative proxy** for what a category needs, which is
    #: really "every block its entries live in". They agree where it matters
    #: and the proxy over-shrinks nowhere, so no data is ever lost -- but it
    #: does under-shrink: DeepSeek-V3.2's ``dense`` entries sit only in
    #: ``attn.full_attention`` and ``mlp.dense`` (``moe`` is its own category),
    #: so layer 0 alone would do, while the all-axes answer is 4 because the
    #: MLP axis turns to MoE at layer 3. That slack is deliberate: ``attention``
    #: is 8,643 shots against ``dense``'s 152, so the axis rule already
    #: captures every minute that matters, and an entry-to-block resolver would
    #: have to cross-reference ``blocks`` with ``catalog`` to save three.
    stack_axes: ClassVar[tuple[str, ...]] = ALL_AXES

    @abstractmethod
    def compose_shots(
        self,
        arch: Architecture,
        args: ProfileArgs,
        limits: RuntimeLimits,
        tp: int,
    ) -> Iterator[Shot]:
        """Yield the Shots that make up this category's sweep."""

    @abstractmethod
    def extract_points(
        self,
        shot: Shot,
        timings: list[TimingSample],
        arch: Architecture,
        tp: int,
    ) -> Iterator[Point]:
        """Turn one shot's timings into CSV-bound Points."""

    @abstractmethod
    def catalog_slice(self, arch: Architecture) -> dict[str, dict]:
        """Which layers the worker should report timings for.

        Returns a plain dict (not pydantic) because it crosses the
        host↔worker RPC boundary.
        """

    @abstractmethod
    def shot_key(self, shot: Shot) -> tuple:
        """Return the shot's identity tuple for resume matching.

        Must align with the CSV row key minus any ``layer`` column:
        if a prior CSV row shares this key the shot is considered
        already profiled and resume mode skips firing it.
        """


def _entry_dict(
    entries: dict[str, LayerEntry],
    arch: Architecture,
) -> dict[str, dict]:
    """Serialize a catalog group for RPC transport.

    ``occurrences`` rides along because the worker cannot compute it: the
    profiled tree only knows how many times a *module* was called, not how
    many trace nodes the architecture emits for it. See ``hooks/timings.py``.
    """
    occurrences = arch.layer_occurrences()
    return {
        name: {
            "vllm": e.vllm,
            "within": e.within,        # str, list[str] or None; see LayerEntry
            "not_within": e.not_within,
            "tp_stable": e.tp_stable,
            "occurrences": occurrences.get(name, 1),
        }
        for name, e in entries.items()
    }


# ---------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------

class DenseCategory(Category):
    """Token-linear layers: embedding, qkv_proj, MLP, layernorm, ..."""

    name = "dense"
    sink_filename = "dense.csv"
    label = "dense"

    def compose_shots(self, arch, args, limits, tp):
        for n in _token_grid(limits.max_num_batched_tokens):
            # Guard against absurdly small KV caches where even a
            # dense prefill wouldn't fit in a single block budget.
            bs = limits.block_size
            if ((n + bs - 1) // bs) * bs > limits.num_cache_tokens:
                continue
            # Context-length bound: a single request of length n must
            # leave room for the sampler's +1 write.
            if n >= limits.max_model_len:
                continue
            yield Shot.dense(total_tokens=n)

    def extract_points(self, shot, timings, arch, tp):
        # Dense shots are one request with all the new tokens packed
        # together — total_tokens is just the sum across requests.
        total_tokens = sum(new for new, _ in shot.requests)
        for sample in timings:
            yield DensePoint(
                layer=sample.layer,
                tokens=total_tokens,
                microseconds=sample.microseconds,
            )

    def catalog_slice(self, arch):
        return _entry_dict(arch.catalog.dense, arch)

    def shot_key(self, shot):
        return (sum(new for new, _ in shot.requests),)


# ---------------------------------------------------------------------------
# Per-sequence
# ---------------------------------------------------------------------------

class SequenceCategory(Category):
    """Sequence-linear layers: lm_head, sampler.

    These operate on one row per output sequence (the "last token of
    each prompt") rather than per input token. Their cost scales with
    batch cardinality.
    """

    name = "per_sequence"
    sink_filename = "per_sequence.csv"
    label = "per_sequence"

    def compose_shots(self, arch, args, limits, tp):
        for n in _token_grid(limits.max_num_seqs):
            # N single-token requests → N new tokens + N blocks used.
            if n > limits.max_num_batched_tokens:
                continue
            if n * limits.block_size > limits.num_cache_tokens:
                continue
            yield Shot.per_sequence(num_sequences=n)

    def extract_points(self, shot, timings, arch, tp):
        # Shot.per_sequence packs N single-token requests — number of
        # requests == number of sequences == what lm_head/sampler see.
        num_sequences = len(shot.requests)
        for sample in timings:
            yield SequencePoint(
                layer=sample.layer,
                sequences=num_sequences,
                microseconds=sample.microseconds,
            )

    def catalog_slice(self, arch):
        return _entry_dict(arch.catalog.per_sequence, arch)

    def shot_key(self, shot):
        return (len(shot.requests),)


class MtpCategory(Category):
    """The model's own drafter (MTP), keyed on the decode batch size.

    One axis, not four. vLLM runs the drafter N times per step and every pass
    is decode-shaped at a fixed query length: ``llm_base_proposer.py`` sets
    ``common_attn_metadata.max_query_len = 1`` and
    ``num_actual_tokens = batch_size``. So the only thing that varies is how
    many sequences are drafting, which makes this the same shape as
    ``per_sequence`` and costs tens of shots rather than the attention grid's
    thousands.

    The kernels arrive for free: the drafter runs inside ``sample_tokens()``
    (``propose_draft_token_ids`` -> ``drafter.propose``), which the fire path
    already calls inside ``layerwise_profile``. What this category adds is the
    *axis* -- without it the drafter's time would be measured once, at whatever
    batch the other categories happened to use.

    Only runs when the engine was booted with ``--profile-mtp``; without it the
    module does not exist and the catalog slice matches nothing.
    """

    name = "mtp"
    sink_filename = "mtp.csv"
    label = "mtp"

    def compose_shots(self, arch, args, limits, tp):
        for n in _token_grid(limits.max_num_seqs):
            # N single-token decode requests, the shape a drafter pass sees.
            if n > limits.max_num_batched_tokens:
                continue
            if n * limits.block_size > limits.num_cache_tokens:
                continue
            yield Shot.per_sequence(num_sequences=n)

    def extract_points(self, shot, timings, arch, tp):
        num_sequences = len(shot.requests)
        for sample in timings:
            yield MtpPoint(
                layer=sample.layer,
                sequences=num_sequences,
                microseconds=sample.microseconds,
            )

    def catalog_slice(self, arch):
        return _entry_dict(arch.catalog.mtp, arch)

    def shot_key(self, shot):
        return (len(shot.requests),)


# ---------------------------------------------------------------------------
# Attention (unified prefill+decode+mixed)
# ---------------------------------------------------------------------------

# Starting points for each axis (smallest non-zero value).
# Grids double from here up to an axis-specific cap.
_ATTN_CHUNK_START = 16      # smallest prefill chunk we profile
_ATTN_N_DECODE_START = 1    # smallest decode batch
_ATTN_KV_START = 16        # smallest KV context (for both prefill & decode)

# Block-aligned KV-budget feasibility checks read ``limits.block_size``, not a
# constant: what we request in HOST_ENGINE_DEFAULTS is not always what the
# engine uses. A hybrid stack makes vLLM enlarge the attention block until an
# attention page is at least as many bytes as a mamba state page (784 tokens
# on Qwen3.8-27B against the 16 we asked for), and a filter off by 49x either
# emits shots the cache cannot hold or silently drops ones it can.


def _geometric_grid(max_value: int, start: int, factor: float = 2.0) -> list[int]:
    """Geometric grid ``[0, start, start*f, start*f^2, ...]`` capped at
    ``max_value``. ``factor=2.0`` is the default (doubling); smaller
    factors give denser sampling at the cost of more shots.

    Always prepends 0 as the "axis absent" sentinel. Values are
    deduplicated (round-to-int can collide at small sizes) and the
    exact ``max_value`` is appended when it isn't already on the grid.
    """
    if factor <= 1.0:
        raise ValueError(f"factor must be > 1.0; got {factor}")
    if max_value < start:
        return [0]
    values: list[int] = [0, start]
    v: float = float(start)
    while True:
        v *= factor
        iv = int(round(v))
        if iv > max_value:
            break
        if iv != values[-1]:  # skip dupes at small scales when factor close to 1
            values.append(iv)
    if values[-1] != max_value:
        values.append(max_value)
    return values


class AttentionCategory(Category):
    """Unified attention profile covering pure-prefill, pure-decode,
    and mixed kernel shapes in a single 4D grid.

    Profiles exactly the kernel shape vLLM's chunked-prefill scheduler
    produces: one prefill chunk + N decode requests in a single
    FlashAttention varlen call. Pure-prefill rows drop the decodes
    (n_decode=0); pure-decode rows drop the prefill (prefill_chunk=0).
    """

    name = "attention"
    # Only the attention axes: which MLP a layer runs cannot change
    # an attention kernel's cost, so a stack shrunk for the MLP axis
    # is measuring the same kernel several times over.
    stack_axes = ATTENTION_AXES
    sink_filename = "attention.csv"
    label = "attention"

    def compose_shots(self, arch, args, limits, tp):
        # Axes are generated from the live runtime limits so the grid
        # naturally scales with each model's max_num_batched_tokens /
        # max_num_seqs. The KV axes are additionally capped by
        # ``args.attention_max_kv`` (CLI-configurable) to keep
        # profile time bounded on long-context models.
        # prefill_chunk and kv axes both default to 2.0 (doubling);
        # override via --attention-chunk-factor / --attention-kv-factor
        # if you want denser sampling. n_decode stays on doubling.
        chunk_vals = _geometric_grid(
            limits.max_num_batched_tokens, _ATTN_CHUNK_START,
            factor=args.attention_chunk_factor,
        )
        n_dec_vals = _geometric_grid(
            limits.max_num_seqs, _ATTN_N_DECODE_START,
        )
        # ``runner`` resolves this against the live engine before any grid
        # is composed, so None here means a caller bypassed that -- and a
        # silently wrong cap is a whole sweep at the wrong resolution.
        assert args.attention_max_kv is not None, (
            "attention_max_kv is unresolved; call "
            "engine.resolve_attention_max_kv(args, limits) after probe_limits"
        )
        kv_cap = min(args.attention_max_kv, limits.max_model_len)
        kv_vals = _geometric_grid(
            kv_cap, _ATTN_KV_START, factor=args.attention_kv_factor,
        )
        # Query tokens per decode sequence. [1] by default -- the axis
        # multiplies the whole sweep, and it only matters for speculative
        # decoding, whose verification step submits 1 + N queries per sequence.
        # Set --attention-decode-q-lens to the 1+N values you intend to
        # simulate; the published N for the four modern families are 3, 4 and
        # 5, so "1,2,4,6,8" brackets them.
        q_vals = sorted({max(1, int(v)) for v in args.attention_decode_q_lens})

        for chunk in chunk_vals:
            for kv_p in kv_vals:
                # When there's no prefill, sweeping kv_prefill would
                # only produce duplicate rows. Collapse to kv_p=0.
                if chunk == 0 and kv_p != 0:
                    continue
                for n_dec in n_dec_vals:
                    for kv_d in kv_vals:
                        if n_dec == 0 and kv_d != 0:
                            continue
                        # A "decode" step by definition has prior
                        # history in the KV cache. (q=1, history=0)
                        # is a 1-token prefill in disguise — not a
                        # shape vLLM's scheduler ever produces, so
                        # profiling it wastes shots on a degenerate
                        # attention case.
                        if n_dec > 0 and kv_d == 0:
                            continue
                        # Empty batch — skip entirely.
                        if chunk == 0 and n_dec == 0:
                            continue
                        # -------- Infeasibility filters --------
                        # The shot bypasses the vLLM scheduler via
                        # ``assemble_scheduler_output``, so MNBT is
                        # advisory — its only role here is bounding
                        # the grid. We allow ``chunk + n_dec`` to grow
                        # up to ``MNBT + MSQ`` so chunk=MNBT can still
                        # pair with the full n_decode axis (filling
                        # the top-half corner that pure geometric
                        # doubling otherwise leaves empty for mixed
                        # batches). ``n_reqs`` is the hard cap —
                        # vLLM V1 pre-allocates ``input_batch`` for
                        # ``max_num_seqs`` sequences and crashes at
                        # the boundary (observed during skew sweeps),
                        # so we stay strictly below.
                        #
                        # 2. Request count vs max_num_seqs. vLLM V1
                        # pre-allocates input_batch for MSQ sequences;
                        # MSQ itself fits, MSQ+1 overflows the buffer.
                        n_reqs = (1 if chunk > 0 else 0) + n_dec
                        if n_reqs > limits.max_num_seqs:
                            continue
                        # 3. Per-request sequence length vs max_model_len
                        # (hardware position-embedding index).
                        if chunk > 0 and chunk + kv_p + 1 > limits.max_model_len:
                            continue
                        bs = limits.block_size

                        def _aligned(total_len: int, bs: int = bs) -> int:
                            return ((total_len + bs - 1) // bs) * bs
                        prefill_block_toks = (
                            _aligned(chunk + kv_p) if chunk > 0 else 0
                        )
                        for q in q_vals:
                            # q > 1 only makes sense where there are decodes
                            # to widen; with none it would duplicate the
                            # pure-prefill row.
                            if q > 1 and n_dec == 0:
                                continue
                            # The remaining bounds are per **token**, and a
                            # decode request submits ``q`` of them rather than
                            # one -- so they have to sit inside this loop.
                            # They used to sit outside it, which counted every
                            # decode as a single token: a shot at chunk=MNBT
                            # whose n_dec*q ran past MSQ then passed the filter
                            # and overflowed vLLM's own buffer, surfacing hours
                            # into a sweep as ``operands could not be broadcast
                            # together with shapes (2304,) (2432,) (2304,)`` --
                            # 2304 being MNBT + MSQ and 2432 the real token
                            # count. At q=1 every bound below is identical to
                            # what it was, so existing grids are unchanged.
                            #
                            # 1. Combined sum bound (advisory).
                            if chunk + n_dec * q > (
                                limits.max_num_batched_tokens
                                + limits.max_num_seqs
                            ):
                                continue
                            # 3b. A decode request's own sequence length.
                            if n_dec > 0 and q + kv_d + 1 > limits.max_model_len:
                                continue
                            # 4. KV cache block budget. Each request rounds up
                            # to a whole block, so block-aligned totals can be
                            # up to ~2x the raw KV tokens for tiny requests.
                            # Compute exactly.
                            decode_block_toks = (
                                n_dec * _aligned(q + kv_d) if n_dec > 0 else 0
                            )
                            if (prefill_block_toks + decode_block_toks
                                    > limits.num_cache_tokens):
                                continue
                            yield Shot.attention(
                                prefill_chunk=chunk,
                                kv_prefill=kv_p,
                                n_decode=n_dec,
                                kv_decode=kv_d,
                                decode_q_len=q,
                            )

    def extract_points(self, shot, timings, arch, tp):
        # Shot.attention encodes the 4D key in its request list:
        #   requests[0] = (prefill_chunk, kv_prefill)   if chunk>0
        #   requests[k] = (1, kv_decode) for each decode, k in [1..n_decode]
        reqs = shot.requests
        q = max(1, getattr(shot, "decode_q_len", 1))
        # Reconstruct the key from the shot shape. The decode query length has
        # to come from the shot rather than the shapes: at q > 1 a decode
        # request looks exactly like a prefill chunk in ``requests``, which is
        # the ambiguity the axis exists to remove.
        if reqs and reqs[0][0] > q:
            # First request is the prefill.
            prefill_chunk, kv_prefill = reqs[0]
            decode_reqs = reqs[1:]
        elif reqs and reqs[0][0] == q and len(reqs) > 0:
            # No prefill; everything is a decode.
            prefill_chunk, kv_prefill = 0, 0
            decode_reqs = reqs
        else:
            raise RuntimeError(f"Unexpected attention shot shape: {reqs!r}")

        n_decode = len(decode_reqs)
        # All decodes share kv_decode by construction.
        kv_decode = decode_reqs[0][1] if decode_reqs else 0

        # One point per matched layer. This used to average every sample into
        # a single point, which was right while the catalog was required to
        # declare exactly one attention entry -- it compensated for a
        # multi-layer test model handing back several samples of the same
        # kernel. Two things changed: the timing extractor now normalizes by
        # parent invocations, so a multi-layer run yields one sample per
        # canonical layer rather than several, and a sparse-attention catalog
        # declares two genuinely different kernels here. Averaging them
        # produced a number describing neither -- measured on
        # DeepSeek-V3.2-Exp, MLAAttention and SparseAttnIndexer collapsed into
        # one value per key.
        for sample in timings:
            yield AttentionPoint(
                layer=sample.layer,
                prefill_chunk=prefill_chunk,
                kv_prefill=kv_prefill,
                n_decode=n_decode,
                kv_decode=kv_decode,
                decode_q_len=q,
                microseconds=sample.microseconds,
            )

    def catalog_slice(self, arch):
        return _entry_dict(arch.catalog.attention, arch)

    def shot_key(self, shot):
        reqs = shot.requests
        q = max(1, getattr(shot, "decode_q_len", 1))
        if reqs and reqs[0][0] > q:
            pc, kp = reqs[0]
            decodes = reqs[1:]
        else:
            pc, kp = 0, 0
            decodes = reqs
        n_dec = len(decodes)
        kv_dec = decodes[0][1] if decodes else 0
        return (pc, kp, n_dec, kv_dec, q)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------

def _chunk_aware_grid(
    max_value: int,
    chunk: int | None,
    start: int = _ATTN_CHUNK_START,
    factor: float = 2.0,
) -> list[int]:
    """Grid for an axis whose cost is a staircase in ``chunk``, not a line.

    A linear-attention prefill scan works in fixed chunks, and the measured
    cost tracks the chunk count. On Qwen3.8-27B one token past a 64-boundary
    costs **13.5% more** than the boundary itself, and the whole interval to
    the next boundary is nearly flat. A plain geometric grid lands only on
    boundaries (64, 128, 256 ...), so interpolating between two samples
    underestimates every token count just past one -- which is most of what a
    chunked-prefill scheduler actually produces.

    So sample three things: points inside the first chunk (cost still varies
    with tokens there), each chunk boundary, and the single token past each
    boundary, which pins the step. Falls back to the plain geometric grid when
    the chunk length is unknown.
    """
    if max_value < 1:
        return [0]
    if not chunk or chunk < 2:
        return _geometric_grid(max_value, start, factor)

    values: set[int] = {0}
    # Inside the first chunk.
    v = start
    while v < min(chunk, max_value):
        values.add(v)
        v = max(v + 1, int(round(v * factor)))
    # Boundaries, and the token that starts the next chunk.
    c = 1
    while c * chunk <= max_value:
        values.add(c * chunk)
        if c * chunk + 1 <= max_value:
            values.add(c * chunk + 1)
        nxt = max(c + 1, int(round(c * factor)))
        if nxt <= c:
            break
        c = nxt
    values.add(max_value)
    return sorted(values)


class LinearAttentionCategory(Category):
    """Linear-attention (mamba / gated-DeltaNet) block, keyed by
    ``(prefill_tokens, n_decode)``.

    Two axes rather than attention's four, because there is no kv axis: the
    state is fixed-size per sequence regardless of position, so cost is
    independent of sequence length (measured: 1.1% over a 64x kv spread) and
    no skew correction applies either.

    Two axes rather than one, though, even though the prefill and decode
    kernels *are* additive -- because **which** kernel runs depends on the mix.
    A pure-decode batch runs a recurrent kernel; add a prefill chunk and vLLM
    switches to a fused-gating one instead, 4.5% apart at the same decode
    count. A pair of 1-D tables cannot represent a kernel-identity switch.
    Same argument that justifies the unified 4-D attention grid, and cheap
    here: ~100 shots against attention's ~1300.
    """

    name = "linear_attention"
    # Only the attention axes: which MLP a layer runs cannot change
    # an attention kernel's cost, so a stack shrunk for the MLP axis
    # is measuring the same kernel several times over.
    stack_axes = ATTENTION_AXES
    sink_filename = "linear_attention.csv"
    label = "linear_attention"

    def compose_shots(self, arch, args, limits, tp):
        pre_vals = _chunk_aware_grid(
            limits.max_num_batched_tokens, limits.linear_attn_chunk,
        )
        dec_vals = _geometric_grid(limits.max_num_seqs, _ATTN_N_DECODE_START)
        bs = limits.block_size
        for pre in pre_vals:
            for n_dec in dec_vals:
                if pre == 0 and n_dec == 0:
                    continue
                # Same feasibility rules as the attention grid, minus the kv
                # ones: MNBT is advisory (the shot bypasses the scheduler) but
                # bounds the grid, while max_num_seqs is a hard cap because
                # vLLM V1 preallocates input_batch for exactly that many.
                if pre + n_dec > (
                    limits.max_num_batched_tokens + limits.max_num_seqs
                ):
                    continue
                n_reqs = (1 if pre > 0 else 0) + n_dec
                if n_reqs > limits.max_num_seqs:
                    continue
                if pre > 0 and pre + 1 > limits.max_model_len:
                    continue
                hist = Shot.LINEAR_ATTN_DECODE_HISTORY
                if n_dec > 0 and hist + 1 + 1 > limits.max_model_len:
                    continue
                # Block budget: every request rounds up to a whole block.
                blocks = 0
                if pre > 0:
                    blocks += ((pre + bs - 1) // bs) * bs
                blocks += n_dec * (((hist + 1 + bs - 1) // bs) * bs)
                if blocks > limits.num_cache_tokens:
                    continue
                yield Shot.linear_attention(
                    prefill_tokens=pre, n_decode=n_dec,
                )

    def extract_points(self, shot, timings, arch, tp):
        pre, n_dec = _split_linear_attention_shot(shot)
        for sample in timings:
            yield LinearAttentionPoint(
                layer=sample.layer,
                prefill_tokens=pre,
                n_decode=n_dec,
                microseconds=sample.microseconds,
            )

    def catalog_slice(self, arch):
        return _entry_dict(arch.catalog.linear_attention, arch)

    def shot_key(self, shot):
        return _split_linear_attention_shot(shot)


def _split_linear_attention_shot(shot: Shot) -> tuple[int, int]:
    """Recover ``(prefill_tokens, n_decode)`` from a shot's request list.

    ``Shot.linear_attention`` puts at most one multi-token request first and
    then the 1-token decodes, so a single pass over the requests is enough.
    """
    pre = 0
    n_dec = 0
    for new, _history in shot.requests:
        if new > 1:
            pre += new
        else:
            n_dec += 1
    return (pre, n_dec)


class ExpertCategory(Category):
    """MoE block (gate + grouped experts), keyed by
    (tokens, activated_experts)."""

    name = "moe"
    sink_filename = "moe.csv"
    label = "moe"

    def compose_shots(self, arch, args, limits, tp):
        # MoE parameters come from the live HF config via RuntimeLimits,
        # not from the yaml. If catalog.moe.* entries exist but the
        # live config didn't expose num_experts / top_k, fail loudly.
        if limits.num_experts is None or limits.top_k is None:
            # One catalog now serves a whole family, so a ``catalog.moe`` entry
            # says "this family has MoE checkpoints", not "this checkpoint is
            # MoE". A dense member declares nothing MoE and simply has no
            # expert sweep to run.
            model_config = args.model_config or {}
            if not declares_moe(model_config):
                return
            raise RuntimeError(
                "catalog.moe entries are declared and the model config "
                "mentions MoE, but num_experts / top_k could not both be "
                "read from it. If this model uses a non-standard field "
                "name, add it to MOE_NUM_EXPERTS_KEYS / MOE_TOP_K_KEYS in "
                "profiler/core/config.py."
            )
        num_experts = limits.num_experts
        top_k = limits.top_k

        for n_tokens in _power_of_two_grid(limits.max_num_batched_tokens):
            # Cheap guards: n_tokens must fit context (with sampler
            # +1 headroom) + cache.
            if n_tokens >= limits.max_model_len:
                continue
            bs = limits.block_size
            if ((n_tokens + bs - 1) // bs) * bs > limits.num_cache_tokens:
                continue
            for activated in _power_of_two_grid(num_experts):
                # Minimum activations per call is top_k (every token
                # votes for top_k experts).
                if activated < top_k:
                    continue
                # Upper bound: each token contributes top_k distinct
                # activations, so more than n_tokens*top_k is impossible.
                if activated > min(num_experts, n_tokens * top_k):
                    continue
                yield Shot.moe(
                    total_tokens=n_tokens,
                    activated_experts=activated,
                )

    def extract_points(self, shot, timings, arch, tp):
        total_tokens = sum(new for new, _ in shot.requests)
        assert shot.experts is not None
        activated = int(shot.experts["activated"])
        if not timings:
            return
        sample = timings[0]
        yield ExpertPoint(
            tokens=total_tokens,
            activated_experts=activated,
            microseconds=sample.microseconds,
        )

    def catalog_slice(self, arch):
        return _entry_dict(arch.catalog.moe, arch)

    def shot_key(self, shot):
        total_tokens = sum(new for new, _ in shot.requests)
        assert shot.experts is not None
        return (total_tokens, int(shot.experts["activated"]))


# ---------------------------------------------------------------------------
# Category registry
# ---------------------------------------------------------------------------

def categories_for(arch: Architecture, tp: int) -> list[Category]:
    """Return the list of categories that should run for this (arch, tp).

    Excludes:
      * Any category whose catalog slice is empty (e.g., ExpertCategory
        for a dense model).
      * ExpertCategory for tp != 1 (MoE is profiled once at tp=1;
        simulator scales per-expert time by ep_size).
      * Any category for which every matching layer is tp_stable AND
        tp != 1 (replicate_tp_stable will fill it in from tp=1).
    """
    result: list[Category] = []
    registry = [
        (DenseCategory(), arch.catalog.dense),
        (SequenceCategory(), arch.catalog.per_sequence),
        (AttentionCategory(), arch.catalog.attention),
        (LinearAttentionCategory(), arch.catalog.linear_attention),
        (ExpertCategory(), arch.catalog.moe),
        (MtpCategory(), arch.catalog.mtp),
    ]
    for cat, entries in registry:
        if not entries:
            continue
        if isinstance(cat, ExpertCategory) and tp != 1:
            continue
        if tp != 1 and all(e.tp_stable for e in entries.values()):
            continue
        result.append(cat)
    return result


# Name→class map used by the `slice` CLI subcommand.
CATEGORY_BY_NAME: dict[str, type[Category]] = {
    DenseCategory.name: DenseCategory,
    SequenceCategory.name: SequenceCategory,
    AttentionCategory.name: AttentionCategory,
    LinearAttentionCategory.name: LinearAttentionCategory,
    ExpertCategory.name: ExpertCategory,
    MtpCategory.name: MtpCategory,
}
