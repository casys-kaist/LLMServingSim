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

from profiler.core.config import Architecture, LayerEntry, ProfileArgs
from profiler.core.engine import RuntimeLimits
from profiler.core.hooks.batch import Shot
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
class AttentionPoint:
    # Shape mirrors attention.csv (§ DESIGN doc): 4D keyed, one column
    # per axis, plus microseconds.
    prefill_chunk: int
    kv_prefill: int
    n_decode: int
    kv_decode: int
    microseconds: float


@dataclass(frozen=True)
class ExpertPoint:
    tokens: int
    activated_experts: int
    microseconds: float


# Union alias for writer.py's benefit.
Point = DensePoint | SequencePoint | AttentionPoint | ExpertPoint


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
            "within": e.within,
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
            if ((n + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _BLOCK_SIZE > limits.num_cache_tokens:
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
            if n * _BLOCK_SIZE > limits.num_cache_tokens:
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


# ---------------------------------------------------------------------------
# Attention (unified prefill+decode+mixed)
# ---------------------------------------------------------------------------

# Starting points for each axis (smallest non-zero value).
# Grids double from here up to an axis-specific cap.
_ATTN_CHUNK_START = 16      # smallest prefill chunk we profile
_ATTN_N_DECODE_START = 1    # smallest decode batch
_ATTN_KV_START = 16        # smallest KV context (for both prefill & decode)

# Must match HOST_ENGINE_DEFAULTS["block_size"] (16). Used for
# block-aligned KV-budget feasibility checks so shots are only
# generated when the paged cache can actually hold them.
_BLOCK_SIZE = 16


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
        kv_cap = min(args.attention_max_kv, limits.max_model_len)
        kv_vals = _geometric_grid(
            kv_cap, _ATTN_KV_START, factor=args.attention_kv_factor,
        )

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
                        # 1. Combined sum bound (advisory).
                        if chunk + n_dec > (
                            limits.max_num_batched_tokens + limits.max_num_seqs
                        ):
                            continue
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
                        if n_dec > 0 and 1 + kv_d + 1 > limits.max_model_len:
                            continue
                        # 4. KV cache block budget. Each request
                        # rounds up to a whole block, so block-aligned
                        # totals can be up to ~2× the raw KV tokens
                        # for tiny requests. Compute exactly.
                        def _aligned(total_len: int) -> int:
                            return ((total_len + _BLOCK_SIZE - 1)
                                    // _BLOCK_SIZE) * _BLOCK_SIZE
                        prefill_block_toks = (
                            _aligned(chunk + kv_p) if chunk > 0 else 0
                        )
                        decode_block_toks = (
                            n_dec * _aligned(1 + kv_d) if n_dec > 0 else 0
                        )
                        if (prefill_block_toks + decode_block_toks
                                > limits.num_cache_tokens):
                            continue
                        yield Shot.attention(
                            prefill_chunk=chunk,
                            kv_prefill=kv_p,
                            n_decode=n_dec,
                            kv_decode=kv_d,
                        )

    def extract_points(self, shot, timings, arch, tp):
        # Shot.attention encodes the 4D key in its request list:
        #   requests[0] = (prefill_chunk, kv_prefill)   if chunk>0
        #   requests[k] = (1, kv_decode) for each decode, k in [1..n_decode]
        reqs = shot.requests
        # Reconstruct the key from the shot shape.
        if reqs and reqs[0][0] > 1:
            # First request is the prefill.
            prefill_chunk, kv_prefill = reqs[0]
            decode_reqs = reqs[1:]
        elif reqs and reqs[0][0] == 1 and len(reqs) > 0:
            # No prefill; everything is a decode.
            prefill_chunk, kv_prefill = 0, 0
            decode_reqs = reqs
        else:
            raise RuntimeError(f"Unexpected attention shot shape: {reqs!r}")

        n_decode = len(decode_reqs)
        # All decodes share kv_decode by construction.
        kv_decode = decode_reqs[0][1] if decode_reqs else 0

        # Attention category has exactly one layer (the attention
        # kernel). We expect at most one sample per shot. If multiple
        # show up (e.g., the test model accidentally has >1 layer),
        # average them so the profile still makes sense.
        if not timings:
            return
        total_us = sum(t.microseconds for t in timings) / len(timings)
        yield AttentionPoint(
            prefill_chunk=prefill_chunk,
            kv_prefill=kv_prefill,
            n_decode=n_decode,
            kv_decode=kv_decode,
            microseconds=total_us,
        )

    def catalog_slice(self, arch):
        return _entry_dict(arch.catalog.attention, arch)

    def shot_key(self, shot):
        reqs = shot.requests
        if reqs and reqs[0][0] > 1:
            pc, kp = reqs[0]
            decodes = reqs[1:]
        else:
            pc, kp = 0, 0
            decodes = reqs
        n_dec = len(decodes)
        kv_dec = decodes[0][1] if decodes else 0
        return (pc, kp, n_dec, kv_dec)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------

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
            raise RuntimeError(
                "catalog.moe entries are declared but the HF config did "
                "not expose num_experts / top_k. If this model uses a "
                "non-standard field name, add it to MOE_NUM_EXPERTS_KEYS "
                "/ MOE_TOP_K_KEYS in profiler/config.py."
            )
        num_experts = limits.num_experts
        top_k = limits.top_k

        for n_tokens in _power_of_two_grid(limits.max_num_batched_tokens):
            # Cheap guards: n_tokens must fit context (with sampler
            # +1 headroom) + cache.
            if n_tokens >= limits.max_model_len:
                continue
            if ((n_tokens + _BLOCK_SIZE - 1) // _BLOCK_SIZE) * _BLOCK_SIZE > limits.num_cache_tokens:
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
        (ExpertCategory(), arch.catalog.moe),
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
    ExpertCategory.name: ExpertCategory,
}
