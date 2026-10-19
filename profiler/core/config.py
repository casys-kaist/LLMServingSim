"""Architecture spec loader + profile session args.

The profiler pairs **two independent pieces of state** at every run:

1. ``Architecture`` — static vLLM class catalog describing one model
   family (llama3 / qwen3 / qwen3-moe / mixtral / phi-moe / ...).
   Stored as a yaml under ``profiler/models/``. Shared
   between profiler and (future) trace_generator.

2. ``ProfileArgs`` — per-session settings that change between runs:
   which checkpoint, which hardware label, TP sweep, dtype, KV cache
   dtype, attention grid cap, etc. Passed as CLI arguments. No yaml.

Per-checkpoint dimensions (hidden_size / num_heads / ...) live in
LLMServingSim's ``configs/model/*.json`` and flow into vLLM via the
HF ID or local path provided at the CLI. The profiler does not
duplicate those fields.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from profiler.core.catalog_path import resolve_architecture_path


# ---------------------------------------------------------------------------
# Constants shared with engine.py
# ---------------------------------------------------------------------------

# HF config fields divided by TP to emulate a single rank of a
# multi-TP deployment (see engine.fuse_engine_kwargs). The same list
# covers every common dense + MoE architecture, so it's a module
# constant rather than per-architecture data.
# Config fields divided by the TP degree to emulate one rank's shapes on a
# single GPU. Every field vLLM shards has to appear here, or that part of the
# model is measured at full size while the rest is measured per-rank -- and a
# mixed measurement is worse than either.
#
# The linear-attention pair is here because vLLM's
# ``MambaStateShapeCalculator.gated_delta_net_state_shape`` shards the
# recurrent state: ``divide(conv_dim, tp)`` where
# ``conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads``, and
# ``divide(num_v_heads, tp)``. Dividing the two head counts reproduces both,
# and keeps their ratio (Qwen3.8-27B is 16 / 48) so the kernel's grouping is
# unchanged. Without them, TP>1 on a hybrid measured a rank holding the
# **full** GDN state while its attention was already per-rank: the engine then
# resolved ``block_size`` 1568 instead of 784 on Qwen3.8-27B, because only the
# attention page had halved.
SHARD_FIELDS: list[str] = [
    "intermediate_size",
    # MiniMax's dense MLP reads this separate width; intermediate_size is its
    # expert width (native minimax_m3/nvidia/model.py:743).
    "dense_intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
    "linear_num_key_heads",
    "linear_num_value_heads",
]


# HF config field-name variants for MoE parameters. Different model
# families spell these differently; we probe all variants and use the
# first hit.
MOE_NUM_EXPERTS_KEYS: tuple[str, ...] = (
    "num_local_experts",     # Mixtral, PhiMoE
    "num_experts",            # Qwen3 MoE
    "n_routed_experts",       # DeepSeek V2/V3
)
MOE_TOP_K_KEYS: tuple[str, ...] = (
    "num_experts_per_tok",    # Mixtral, PhiMoE, Qwen3 MoE
    "num_experts_per_token",  # some variants
    "moe_k",                  # edge cases
)


# HF config field-name variants for the linear-attention prefill chunk length.
# Mamba2-style configs declare it; gated-DeltaNet ones don't, and take it from
# flash-linear-attention's fixed constant instead.
LINEAR_ATTN_CHUNK_KEYS: tuple[str, ...] = (
    "chunk_size",        # Mamba2 / Nemotron-H
    "mamba_chunk_size",  # some hybrids
)


# HF config field-name variants for the model's weight dtype.
MODEL_DTYPE_KEYS: tuple[str, ...] = ("torch_dtype", "dtype")


def model_config_weight_dtype(hf_cfg: dict[str, Any]) -> Any | None:
    """The weight precision a model config declares, or None.

    A ``quantization_config`` wins over the dtype fields, because for a
    quantized checkpoint those describe the *activation* dtype, not the
    weights. DeepSeek-V3.2-Exp ships FP8 block-quantized with
    ``torch_dtype: bfloat16``; naming its variant folder ``bf16`` would both
    mislabel what was measured (FP8 GEMMs) and collide with a genuine bf16
    release of the same model.

    HuggingFace renamed this field: ``torch_dtype`` is the legacy spelling
    and ``dtype`` the current one (Qwen3.8's config carries only the latter).
    Both are accepted. ``torch_dtype`` is consulted first because every
    profile bundle already committed was named from it, so a config that
    somehow carried both would keep pointing at its existing folder rather
    than silently renaming it.

    Kept next to the other config probes so the profiler and the simulator's
    ``trace_generator.resolve_variant`` can agree — if they disagree the
    simulator looks in a variant folder the profiler never wrote.
    """
    quant = hf_cfg.get("quantization_config")
    if isinstance(quant, dict):
        method = quant.get("quant_method")
        if method:
            return method
    for key in MODEL_DTYPE_KEYS:
        v = hf_cfg.get(key)
        if v:
            return v
    return None


def probe_linear_attn_chunk(hf_cfg: dict[str, Any]) -> int | None:
    """Chunk length the linear-attention prefill scan works in.

    Resolution order, most authoritative first:

    1. the model config, when the architecture declares it — Mamba2-style
       models do, and their value is not the same as anyone else's;
    2. vLLM's ``FLA_CHUNK_SIZE``, the fixed ``BT`` every gated-DeltaNet kernel
       compiles with (64). Read from the installed vLLM rather than copied,
       since it is a kernel property and can change with the release.

    Returns None when neither is available, which is the signal to fall back
    to a plain geometric grid. The caller applies a CLI override on top.
    """
    for key in LINEAR_ATTN_CHUNK_KEYS:
        if key in hf_cfg:
            try:
                v = int(hf_cfg[key])
            except (TypeError, ValueError):
                continue
            if v > 0:
                return v
    try:
        from vllm.third_party.flash_linear_attention.ops.utils import (
            FLA_CHUNK_SIZE,
        )
    except Exception:
        return None
    return int(FLA_CHUNK_SIZE) or None


def declares_moe(hf_cfg: dict[str, Any]) -> bool:
    """True if the config mentions MoE at all, however partially.

    Distinguishes the two reasons ``probe_moe_params`` can come back None. A
    **dense checkpoint of a family whose catalog covers both shapes** declares
    nothing MoE and should simply skip the expert sweep — that is normal now
    that one catalog serves a family. A config that declares *some* MoE field
    but not the pair we need is a different thing: almost certainly a spelling
    this repo doesn't know yet, and worth failing on.
    """
    keys = set(MOE_NUM_EXPERTS_KEYS) | set(MOE_TOP_K_KEYS)
    return any(k in hf_cfg for k in keys)


def probe_moe_params(hf_cfg: dict[str, Any]) -> tuple[int, int] | None:
    """Extract (num_experts, top_k) from a HuggingFace config dict.

    Returns None if the config doesn't declare both — caller decides
    how to react (usually: fail if the model is supposed to be MoE).
    """
    num_experts = next(
        (hf_cfg[k] for k in MOE_NUM_EXPERTS_KEYS if k in hf_cfg),
        None,
    )
    from .stack import expert_top_k
    top_k = expert_top_k(hf_cfg)
    if num_experts is None or top_k is None:
        return None
    return (int(num_experts), int(top_k))


# ---------------------------------------------------------------------------
# Catalog (loaded from architecture yaml)
# ---------------------------------------------------------------------------

class CallableScope(BaseModel):
    """Exact runtime target; instrumentation never imports or replaces its code."""
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["function", "method"]
    module: str | None = None
    attribute: str | None = None
    owner: str | None = None
    path: str | None = None
    within: str | list[str] | None = None
    context: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check_target(self):
        from .hooks.capture_scope_contract import normalize_callable
        normalized = normalize_callable(self.model_dump(exclude_none=True))
        self.context = normalized.get('context')
        return self


class LayerEntry(BaseModel):
    """One row of the catalog.

    Each entry binds a canonical name (the YAML key) to a vLLM class/kernel
    selector or one exact native CPU operator scope, plus optional
    disambiguation (``within``) and TP-invariance (``tp_stable``).
    """
    # extra="forbid" catches typos in YAML early (e.g., `tp_stabe: true`).
    model_config = ConfigDict(extra="forbid")

    vllm: str | list[str] | None = None
    """Name the CUDA profiler reports for this layer, e.g.
    ``"QKVParallelLinear"``, ``"RMSNorm"``, ``"Attention"``.

    Usually a vLLM leaf class, but a **raw CUDA kernel name** works exactly as
    well: matching strips a trailing ``(...)`` and compares the rest, and a
    kernel node has no parentheses. That is the only way to reach layers vLLM
    never wraps in a module — gated DeltaNet's conv and decode recurrence, and
    MiniMax-M3's sparse attention, among them.

    A trailing ``*`` matches by **prefix**. Needed for a fused kernel, whose
    reported name inlines its template arguments and so carries the dtypes:
    ``fusedMiniMaxM3QNormRopeKVInsertKernel<c10::BFloat16, ...`` would stop
    matching on the fp8 variant, which is a run we deliberately make. No class
    or Triton-kernel name contains ``*``, so the sigil is unambiguous.

    A **list** means "every one of these that the checkpoint has", and covers
    two situations. A family may swap the class by checkpoint rather than by
    structure: Llama 3 uses ``Llama3RotaryEmbedding`` for its extended rope
    scaling where Llama 1/2 and Mistral use the base ``RotaryEmbedding``, with
    the layer playing the same role either way — listing both lets one catalog
    cover the family instead of quietly measuring nothing on half of it. Or one
    canonical layer may genuinely be several kernels, as MiniMax-M3's sparse
    attention is (prefill kernel, decode kernel, merge), in which case the
    matches are summed."""

    operator: str | None = None
    """One exact native CPU operator name, instead of ``vllm``.

    The operator interval determines which native CUDA launches belong to
    this component. Latency still uses CUDA activity intervals, not CPU time.
    Its scope is transparent to descendant parent-call normalization.
    """

    callable: CallableScope | None = None
    """One explicitly declared imported function or live-object method.

    The wrapper only establishes runtime ancestry and CUDA launch ownership.
    An inactive alternative is reported, never populated with a zero value.
    """

    normalization: Literal["parent", "invocations"] | None = None
    """Override the category default (parent, or invocations for whole MoE)."""

    @model_validator(mode="after")
    def _check_binding_kind(self):
        if sum(value is not None for value in (self.vllm, self.operator, self.callable)) != 1:
            raise ValueError("A layer requires exactly one of vllm, operator or callable")
        if self.operator is not None:
            if not self.operator or any(char in self.operator for char in "*("):
                raise ValueError("operator must be one exact native CPU name")
        elif self.vllm is not None and (not self.vllm_names() or any(not name for name in self.vllm_names())):
            raise ValueError("vllm requires nonempty class/kernel names")
        return self

    def vllm_names(self) -> list[str]:
        """``vllm`` as a list."""
        if isinstance(self.vllm, str):
            return [self.vllm]
        return list(self.vllm or [])

    within: str | list[str] | None = None
    """Optional ancestor class name to disambiguate when the same ``vllm``
    class appears multiple times in the model (most commonly RMSNorm, which
    shows up as input / post / final layernorm).

    A **list** means "any of these", which is what lets one catalog serve a
    whole family: vLLM names the same structural class differently per
    checkpoint shape, so Qwen3's decoder layer is ``Qwen3DecoderLayer`` for a
    dense checkpoint and ``Qwen3MoeDecoderLayer`` for a MoE one, with
    everything else identical. Without alternatives the two need duplicate
    catalogs, and a fix to one silently misses the other. Matching takes the
    deepest alternative present in the ancestor chain, so listing a name that
    this checkpoint does not have costs nothing."""

    not_within: str | list[str] | None = None
    """Ancestor class(es) that **disqualify** a node from matching this entry.

    Needed when one class plays two roles that ``within`` cannot separate
    because it is the immediate parent in both. DeepSeek's shared expert is a
    ``DeepseekV2MLP``, exactly like the dense-MLP layers' own ``mlp``, so
    ``gate_up_proj`` with ``within: DeepseekV2MLP`` matches both — an 18432-wide
    GEMM and a 2048-wide one, whose mean describes neither. The deepest-
    ``within`` rule cannot help: ``DeepseekV2MLP`` is the closest ancestor
    either way. ``not_within: DeepseekV2MoE`` excludes the shared-expert copy,
    whose cost is already inside the ``moe`` entry that binds the whole block.
    """

    def within_names(self) -> list[str]:
        """``within`` as a list, empty when unset."""
        if self.within is None:
            return []
        if isinstance(self.within, str):
            return [self.within]
        return list(self.within)

    def not_within_names(self) -> list[str]:
        """``not_within`` as a list, empty when unset."""
        if self.not_within is None:
            return []
        if isinstance(self.not_within, str):
            return [self.not_within]
        return list(self.not_within)

    tp_stable: bool = False
    """If True, profile this layer only at TP=1 and replicate the
    results into every tp{N}/ folder."""

    sequence_parallel: bool | None = None
    """Whether the MoE wrapper sequence-shards its input under DP+TP."""

    attention_sequence_parallel: bool | None = None
    """Whether TP+DP MoE at PP1 keeps sequence shards across attention blocks.

    This requires a different trace from a wrapper-only sequence-parallel MoE.
    Until implemented, serving refuses that deployment rather than emitting
    attention AllReduce and a per-block MoE restore that the engine omits.
    """

    tp_output_reduction: Literal["all_reduce"] | None = None
    """Ordinary TP row-parallel output reduction, after this component."""

    key_saturates: bool = False
    """If True, this kernel's cost stops growing once a sequence's key window
    passes the checkpoint's bound (``probe_key_saturation``), so the
    simulator caps the key length it looks the kernel up at.

    A property of the computation, and it varies *within* a model rather than
    between models. DeepSeek-V3.2's ``attention`` is sparse MLA and saturates
    -- a 16-token prefill costs 139.3 us at 2048 of context and 138.3 at 8192
    -- while its ``indexer`` scores the whole KV to make the selection and
    keeps growing (52 -> 101 us from kv 2048 to 16384). MiniMax-M3 has both
    kinds of attention in one stack, sparse from its fourth layer on.

    Why it is capped at lookup rather than at sweep time: the two kernels
    share one grid, so a capped column would collapse the indexer's
    large-key rows onto one cell. The grid records the key length uncapped
    and each kernel reads it at its own resolution."""


class Catalog(BaseModel):
    """The full layer catalog, grouped by profile kind.

    Grouping is at the top level (rather than as an ``as:`` field on
    each entry) so that the file reads as four coherent blocks.
    """
    model_config = ConfigDict(extra="forbid")

    dense: dict[str, LayerEntry] = Field(default_factory=dict)
    per_sequence: dict[str, LayerEntry] = Field(default_factory=dict)
    attention: dict[str, LayerEntry] = Field(default_factory=dict)
    linear_attention: dict[str, LayerEntry] = Field(default_factory=dict)
    moe: dict[str, LayerEntry] = Field(default_factory=dict)
    # The model's own drafter. A separate group rather than entries in `dense`
    # because it only exists when the engine boots with speculative_config, and
    # because it is emitted N times per step rather than once per layer.
    mtp: dict[str, LayerEntry] = Field(default_factory=dict)

    def all_entries(self) -> list[tuple[str, str, LayerEntry]]:
        """Flatten to ``[(profile_kind, layer_name, entry), ...]``."""
        out: list[tuple[str, str, LayerEntry]] = []
        for kind in ("dense", "per_sequence", "attention", "linear_attention",
                     "moe"):
            for name, entry in getattr(self, kind).items():
                out.append((kind, name, entry))
        return out


class AttnBlock(BaseModel):
    """One attention block type's layer order, around its attention kernel."""
    model_config = ConfigDict(extra="forbid")

    pre_attn: list[str] = Field(default_factory=list)
    post_attn: list[str] = Field(default_factory=list)

    def all_layers(self) -> list[str]:
        return [*self.pre_attn, *self.post_attn]


class Blocks(BaseModel):
    """Heterogeneous stack: layer order keyed by **axis**, not by block name.

    A layer's identity in a modern stack is a tuple, not a name — Qwen3.8
    varies the attention type along ``layer_types``, GLM and DeepSeek vary the
    MLP along ``first_k_dense_replace``, MiniMax-M3 varies both. Naming every
    combination explodes; keying each axis separately does not, and the
    per-layer values come from the checkpoint's own config rather than from
    this file.

    Keys under ``attn`` are the values the model config uses (e.g.
    ``linear_attention`` / ``full_attention`` from ``layer_types``), so the
    resolver can look a layer up directly.
    """
    model_config = ConfigDict(extra="forbid")

    attn: dict[str, AttnBlock] = Field(default_factory=dict)
    sparse_attn: dict[str, AttnBlock] = Field(default_factory=dict)
    """Overlay used on layers that run a sparse-attention selection branch.

    A third axis rather than more keys under ``attn`` because vendors vary it
    independently of the attention type: MiniMax-M3 keeps full attention on
    every layer and switches only sparsity, swapping
    ``MiniMaxM3Attention`` for ``MiniMaxM3SparseAttention`` with a different
    qkv projection and an indexer subtree. Keyed by the same attention-type
    names as ``attn``, so a model varying both stays expressible. Empty means
    sparse layers use the plain ``attn`` block, which is right for
    DeepSeek-V3.2 and GLM-5: they are sparse on every layer, so there is
    nothing to distinguish."""

    mlp: dict[str, list[str]] = Field(default_factory=dict)

    def all_layers(self) -> list[str]:
        out: list[str] = []
        for block in self.attn.values():
            out.extend(block.all_layers())
        for block in self.sparse_attn.values():
            out.extend(block.all_layers())
        for layers in self.mlp.values():
            out.extend(layers)
        return out

    def occurrences(self) -> dict[str, int]:
        """Emissions per block instance, maxed across block types.

        Per block type, because that is what the profiled node's parent
        invocation count is relative to: a layernorm emitted twice inside
        every decoder layer has 2 regardless of which attention type the layer
        carries, while a gated-DeltaNet projection emitted once inside the GDN
        block has 1 — and taking the max is exactly right, since a layer
        belongs to one block type or is shared by all of them with the same
        count.
        """
        counts: dict[str, int] = {}
        groups: list[list[str]] = [b.all_layers() for b in self.attn.values()]
        groups += [b.all_layers() for b in self.sparse_attn.values()]
        groups += [list(v) for v in self.mlp.values()]
        for group in groups:
            per_group: dict[str, int] = {}
            for name in group:
                per_group[name] = per_group.get(name, 0) + 1
            for name, c in per_group.items():
                counts[name] = max(counts.get(name, 0), c)
        return counts


class Shared(BaseModel):
    """Layers outside the repeated stack: once per iteration, not per block."""
    model_config = ConfigDict(extra="forbid")

    prologue: list[str] = Field(default_factory=list)
    head: list[str] = Field(default_factory=list)

    def all_layers(self) -> list[str]:
        return [*self.prologue, *self.head]

    def occurrences(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for name in self.all_layers():
            counts[name] = counts.get(name, 0) + 1
        return counts


class DrafterBlock(BaseModel):
    """The decoder block one drafter pass wraps.

    Declared rather than resolved from the checkpoint, because vLLM's MTP
    modules **force** it per family and the forcing lives in each family's
    ``*_mtp.py`` rather than in the config:

    * DeepSeek/GLM (``deepseek_mtp.py``) build the block at layer index
      ``num_hidden_layers``, which is past ``first_k_dense_replace`` (MoE) and
      inherits the model-wide ``index_topk`` (sparse).
    * Qwen3.5 (``qwen3_5_mtp.py``) passes ``layer_type="full_attention"``
      explicitly, overriding whatever ``layer_types`` says.
    * MiniMax-M3 (``minimax_m3/*/mtp.py``) passes ``force_sparse_attn=True,
      force_moe=True``.

    Indexing the checkpoint's own stack gets all three wrong: the stack has
    exactly ``num_hidden_layers`` entries, so index ``num_hidden_layers`` wraps
    to layer 0 -- dense for DeepSeek/GLM, non-sparse for M3, linear attention
    for Qwen3.8. The values here name axes of ``blocks:``, exactly as
    ``stack.LayerSpec`` does.

    An axis left **unset** means "whatever the checkpoint says", and the
    simulator then requires that axis to be uniform across the target's stack
    -- otherwise there is no single answer to inherit and the catalog has to
    name one. Qwen3.5 is the case that needs it: ``qwen3_5.py`` picks the MLP
    off ``model_type`` rather than per layer, so the drafter's MLP is the
    model's, whichever that is.
    """
    model_config = ConfigDict(extra="forbid")

    attn: str | None = None
    mlp: str | None = None
    sparse: bool | None = None


class MtpSection(BaseModel):
    """``mtp:`` -- what one drafter pass emits.

    vLLM runs the drafter **N times per step** for N speculative tokens: once,
    then ``num_speculative_tokens - 1`` more inside
    ``llm_base_proposer.py``'s loop. The first pass is target-shaped; the rest
    are decode-shaped at ``max_query_len = 1``.

    Split the way ``shared:`` is, around the decoder block the wrapper holds:
    ``prologue`` runs before it (the norms and the 2h->h combine), ``head``
    after (DeepSeek/GLM's ``SharedHead``; Qwen and M3 keep ``lm_head`` outside
    the predictor and so declare none).
    """
    model_config = ConfigDict(extra="forbid")

    prologue: list[str] = Field(default_factory=list)
    decoder_block: DrafterBlock | None = None
    head: list[str] = Field(default_factory=list)


class ComponentLayoutLayer0(BaseModel):
    """One exact target-layer-zero prefix consumed by a fused prologue."""
    model_config = ConfigDict(extra="forbid")
    path: list[str]
    expected_prefix: list[str]
    replacement_prefix: list[str]
    consumed_by: str


class ComponentLayoutVariant(BaseModel):
    """Observed canonical members and a one-pass block-order substitution."""
    model_config = ConfigDict(extra="forbid")
    observed: dict[str, list[str]]
    replace: dict[str, list[str]] = Field(default_factory=dict)
    layer0: list[ComponentLayoutLayer0] | None = None
    invocation_contract: Literal["site-scope-calls"] | None = None
    runtime_contract: Literal["native-dsa-eager-tail"] | None = None


class ComponentLayoutGroup(BaseModel):
    """Mutually exclusive, runtime-observed representations of one component."""
    model_config = ConfigDict(extra="forbid")
    historical: str
    categories: list[str]
    variants: dict[str, ComponentLayoutVariant]
    sites: list[list[str]] | None = None


class Architecture(BaseModel):
    """Parsed architecture yaml.

    Holds ``catalog`` (vLLM class bindings — used by the profiler) and the
    layer order the simulator's trace generator walks: ``blocks`` keyed by
    axis, plus ``shared`` for what sits outside the repeated stack.

    There used to be a second form, ``sequence``, for a uniform stack. It was
    the same thing flattened -- its ``pre_attn`` / ``post_attn`` were the
    single implicit ``attn.full_attention`` block and its ``mlp_dense`` /
    ``mlp_moe`` were the ``mlp`` axis with the value baked into the key name.
    Two forms meant two code paths, and the simulator only ever implemented
    the flat one, so half the catalogs could not be simulated at all. Worse,
    the flattening encoded a claim that is not true: baking the axis into the
    key removes the per-layer question, and the MLP choice was resolved once
    per *model* -- which modelled DeepSeek-V3.2's and GLM-5's first three
    dense layers as MoE. One form, and the per-layer values come from the
    checkpoint's config (``layer_types``, ``first_k_dense_replace``, ...),
    never from here.
    """
    model_config = ConfigDict(extra="forbid")

    catalog: Catalog
    model_types: list[str] = Field(default_factory=list)
    blocks: Blocks | None = None
    shared: Shared | None = None
    # What one drafter pass emits, in order. Present only for a model with MTP
    # modules; `mtp_block` is deliberately not listed, because it is the same
    # class as a target decoder layer and the simulator replays the target's
    # own block sequence for it.
    mtp: MtpSection | None = None
    component_layouts: dict[str, ComponentLayoutGroup] = Field(default_factory=dict)
    native_profile: Literal['hybrid-residual'] | None = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_catalog(self) -> "Architecture":
        # Canonical names must be unique across ALL catalog groups.
        seen: set[str] = set()
        for kind, name, entry in self.catalog.all_entries():
            if name in seen:
                raise ValueError(f"Layer name {name!r} appears twice in catalog")
            seen.add(name)
            if entry.callable is not None and entry.callable.context is not None:
                if kind != 'dense' or entry.normalization != 'invocations':
                    raise ValueError('Context-bearing callables currently require dense/invocations')
            if entry.attention_sequence_parallel and (
                    kind != "moe" or entry.sequence_parallel is not True):
                raise ValueError("attention_sequence_parallel requires a sequence-parallel MoE entry")
            if entry.tp_output_reduction is not None and kind != "dense":
                raise ValueError("tp_output_reduction requires a dense projection entry")

        # At least one softmax-attention entry, unless the architecture is
        # purely linear-attention. More than one is allowed: a sparse-attention
        # model runs an indexer kernel alongside the attention kernel, and both
        # are keyed on the same axes, so both belong in this category.
        if not self.catalog.attention and not self.catalog.linear_attention:
            raise ValueError(
                "catalog must declare at least one attention or "
                "linear_attention entry"
            )

        # Binding kind/name and ancestry guards are globally unique so matching is
        # deterministic. Several profile-tree nodes may still match one entry,
        # which is fine and sometimes the point: their timings are summed,
        # because one canonical name is one trace node.
        pairs: dict[tuple, str] = {}
        for _, name, entry in self.catalog.all_entries():
            key = (
                "callable" if entry.callable is not None else "operator" if entry.operator is not None else "vllm",
                (entry.callable.name,) if entry.callable is not None else
                (entry.operator,) if entry.operator is not None else tuple(sorted(entry.vllm_names())),
                tuple(sorted(entry.within_names())),
                tuple(sorted(entry.not_within_names())),
            )
            if key in pairs:
                raise ValueError(
                    f"Ambiguous layer binding: {name!r} and {pairs[key]!r} "
                    f"both resolve to ({key[0]}={key[1]!r}, "
                    f"within={entry.within!r})"
                )
            pairs[key] = name

        # One semantic callable name has one target across category slices.
        # This also rejects operator/callable context-name collisions before
        # any worker can change a live callable attribute.
        from .hooks.capture_scope_contract import contexts_from_architecture
        contexts_from_architecture(self)

        # One layer-order form, and it has to be there: the trace generator
        # has nothing to walk otherwise.
        if self.blocks is None:
            raise ValueError(
                "architecture must declare 'blocks' (layer order keyed by "
                "axis: attn.<layer_types value>, mlp.dense|moe, and "
                "sparse_attn.<...> when a layer's sparse flag applies)"
            )
        if not self.blocks.attn:
            raise ValueError(
                "'blocks.attn' must declare at least one attention block type"
            )

        # Every referenced name must be a canonical name declared in the
        # catalog — catches typos before a profile/simulation run.
        catalog_names = {n for _, n, _ in self.catalog.all_entries()}
        referenced: list[str] = []
        if self.blocks is not None:
            referenced += self.blocks.all_layers()
        if self.shared is not None:
            referenced += self.shared.all_layers()
        unknown = [n for n in referenced if n not in catalog_names]
        if unknown:
            raise ValueError(
                f"layer order references names not in catalog: "
                f"{sorted(set(unknown))}"
            )

        from profiler.core.component_layouts import validate_definitions
        validate_definitions(self.model_dump(exclude_none=True))
        return self

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def has_moe(self) -> bool:
        return bool(self.catalog.moe)

    def layer_occurrences(self) -> dict[str, int]:
        """How many trace nodes each canonical layer contributes per block.

        The profiler needs this to normalize a profiled node's total CUDA
        time: vLLM merges same-class siblings into one node, so the reported
        ``invocations`` counts module calls, not trace nodes. See
        ``hooks/timings.py``.

        A name absent from the layer order is reported as 1 by the caller's
        ``or 1``, which is the right default — a layer nothing emits has
        nothing to divide.
        """
        counts: dict[str, int] = {}
        for source in (self.blocks, self.shared):
            if source is None:
                continue
            for name, c in source.occurrences().items():
                counts[name] = max(counts.get(name, 0), c)
        return counts

    def has_tp_dependent_work(self, tp: int) -> bool:
        """True iff this TP pass has any non-tp_stable layers to profile."""
        if tp == 1:
            return True
        for _, _, entry in self.catalog.all_entries():
            if not entry.tp_stable:
                return True
        return False


def load_architecture(path: Path) -> Architecture:
    """Parse an architecture yaml into an ``Architecture``."""
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: top-level must be a mapping, got {type(raw).__name__}"
        )
    return Architecture.model_validate(raw)


def architecture_hash(path: Path) -> str:
    """SHA-256 of the raw yaml bytes, for meta.yaml provenance."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Architecture auto-resolution from a HuggingFace model config
# ---------------------------------------------------------------------------

def _load_model_config(path: Path) -> dict[str, Any]:
    """Parse a model's config.json and return it as a dict.

    Raises with a clear message if the file is missing or malformed.
    """
    import json

    if not path.is_file():
        raise FileNotFoundError(
            f"Model config not found: {path}. Place the HuggingFace "
            f"config.json at this path before profiling."
        )
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from e


def detect_model_type(model_config_path: Path) -> str:
    """Extract ``model_type`` from a HuggingFace-style config.json.

    Raises if the file lacks ``model_type``.
    """
    cfg = _load_model_config(model_config_path)
    mt = cfg.get("model_type")
    if not mt:
        raise ValueError(
            f"{model_config_path} has no ``model_type`` field. Use a "
            f"HuggingFace config.json."
        )
    return str(mt)


def read_model_config(model_config_path: Path) -> dict[str, Any]:
    """Read the full model config.json as a dict.

    Returns every field verbatim — architectures, model_type,
    dimensions, rope_scaling, etc. vLLM ingests this directly from a
    local directory when the profiler spins up, so HF hub access is
    never required at profile time. Users profiling custom shapes
    just edit ``configs/model/<path>.json`` and re-run.
    """
    return _load_model_config(model_config_path)


def resolve_architecture_by_model_type(
    model_type: str,
    arch_dir: Path,
) -> Path:
    """Find the architecture yaml serving ``model_type``.

    Thin wrapper over ``catalog_path.resolve_architecture_path``, which holds
    the naming rule. That rule lives in its own dependency-free module because
    the **simulator** resolves catalogs too, and two implementations of it
    already drifted once — aliasing landed here and not there, and every MoE
    scenario broke the moment two ``model_type`` values shared one file.
    """
    return Path(resolve_architecture_path(model_type, str(arch_dir)))


# ---------------------------------------------------------------------------
# Profile session args (CLI, no yaml)
# ---------------------------------------------------------------------------

#: Default geometric factor for attention request counts.
#:
#: Smaller geometric factors trade more shots for finer resolution. A doubling
#: axis is nested in the sqrt(2) axis: even powers retain its coordinates.
#: Reuse still requires compatible acquisition identities and matching complete
#: shot keys; changing another axis can remove those matches.
#:
#: For a fixed range, the number of intervals scales as 1/log(factor).
#: Changing 2.0 to sqrt(2) roughly doubles intervals on EACH axis, not 1.4x.
#: Attention combines four shape axes, with one KV factor controlling two of
#: them. Their costs multiply before rounding, degenerate-axis deduplication
#: and feasibility filters. Count the composed plan rather than assuming a
#: fixed multiplier or wall time. Denser sampling alone does not establish
#: better end-to-end accuracy on every axis or workload.
#:
#: A bundle records the factor it was swept at, per axis, in
#: ``meta.yaml::attention_grid``. **Refreshing one axis of an existing bundle
#: means restating the others**, or the defaults silently re-grid them: the
#: RTXPRO6000 Llama/Qwen bundles are at 1.5 on chunk and kv, everything else
#: at 2.0.
SQRT2 = 2.0 ** 0.5


@dataclass(frozen=True)
class ProfileArgs:
    """One profiling session's settings.

    Built from CLI arguments in ``__main__.py``. Immutable after
    construction so downstream code can treat it as a value.

    Attributes:
        architecture: Name of the architecture yaml (e.g. "llama3").
            Used for meta.yaml and banner display. The actual yaml is
            resolved by ``__main__.py`` before this object is built.
        model: HF model ID or local directory path forwarded to
            ``vllm.LLM(model=...)``.
        hardware: Free-form hardware label that becomes an output
            folder name (e.g. "H100", "A6000", "RTXPRO6000").
        tp_degrees: Positive TP shardings to sweep. Ordinary category sweeps
            require 1 for TP-stable replication; skew-only work is independent.
        variant: Free-form output folder label under the model's
            directory. If omitted at the CLI, auto-derived from
            checkpoint precision and explicit activation/KV dtype choices.
            HF precision or nested text-config overrides need an explicit label.
        dtype / kv_cache_dtype / max_num_batched_tokens / max_num_seqs:
            Common vLLM engine kwargs. None means "use defaults"
            (HOST_ENGINE_DEFAULTS for max_*, vLLM default for dtype).
        attention_max_kv: Cap for the attention and skew grids' KV axes.
            Each grid retains its own starting point and factor. ``None`` means
            the model's own context, and ``runner`` resolves it against the
            live engine's ``max_model_len`` (see
            ``engine.resolve_attention_max_kv``) before any grid is composed --
            so every reader downstream, including the meta the run records,
            sees one concrete number and never ``None``.
        dp_degrees: Target DP degrees for native MoE components. None
            selects DP1 automatically for MoE acquisition; other categories
            and coverage are unaffected. EP is TP times DP.
        ep_degrees: Optional EP filter over those TP/DP targets. None keeps
            every target; an explicit degree must match a selected TP times DP.
        profile_mtp: When set, boot with ``speculative_config`` so the drafter
            is built and its kernels land in the profile tree. The drafter
            runs inside ``sample_tokens()`` (``propose_draft_token_ids`` ->
            ``drafter.propose``), which the fire path already calls inside
            the native CUDA collector. Always at ``num_speculative_tokens=1``, so
            what lands in ``mtp.csv`` is **one** drafter pass -- the unit the
            simulator multiplies by its own N.
        hf_overrides: Extra HF config overrides, merged under the
            profiler's own (num_hidden_layers=1) + TP sharding.
    """

    # Required
    architecture: str
    model: str
    hardware: str

    # TP sweep
    tp_degrees: list[int] = field(default_factory=lambda: [1])

    # Output variant
    variant: str | None = None

    # Engine kwargs (optional overrides)
    dtype: str | None = None
    kv_cache_dtype: str | None = None
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    block_size: int | None = None
    """KV block size in tokens, vLLM's own ``--block-size``. None uses
    vLLM's backend-dependent resolution. Exposed because the simulator has the same knob
    and the two have to agree: a profile measured at one block size describes a
    different paging regime than a simulation run at another. It also stops
    mattering only for uniform models — on a hybrid stack vLLM *overrides*
    whatever is requested here to unify attention and mamba page sizes, and
    ``probe_limits`` reports what it settled on."""

    gpu_memory_utilization: float | None = None
    """vLLM's own ``--gpu-memory-utilization``; the simulator spells the same
    quantity ``--npu-memory-utilization``. None uses HOST_ENGINE_DEFAULTS
    (0.9). It decides ``num_gpu_blocks`` and therefore ``num_cache_tokens``,
    which every shot-feasibility filter is measured against — so it changes
    *which* shots a sweep contains, not just how fast it runs."""

    max_model_len: int | None = None
    """Cap the engine's context length. None lets vLLM take it from the model
    config. Bounds the kv axes and the per-shot length checks, so lowering it
    on a long-context model cuts profile time; ``--attention-max-kv`` caps the
    same axes from the grid side."""

    num_hidden_layers: int | None = None
    # Boot the engine with a speculative_config so vLLM also builds the
    # model's own MTP module. Needed to profile the drafter at all: the MTP
    # config model_type (deepseek_mtp / qwen3_5_mtp / minimax_m3_mtp) is
    # produced by SpeculativeConfig.hf_config_override and is unknown to HF
    # Transformers, so the module cannot be loaded standalone.
    dp_degrees: tuple[int, ...] | None = None
    """Target DP degrees for native, separately timed MoE components.

    None selects DP1 automatically for MoE acquisition. This is target metadata, not
    the number of actual profiler processes. EP is resolved as TP times DP.
    DP1 measures local gate/routing and experts without DP dispatch/combine.
    """
    ep_degrees: tuple[int, ...] | None = None
    """Optional full-EP target filter. None selects all TP times DP targets.

    This does not change expert tensor sharding or infer additional TP/DP degrees.
    """
    moe_rounds: int = 3
    moe_grid_factor: float = 2.0
    profile_mtp: bool = False
    """Layers to instantiate. None uses HOST_ENGINE_DEFAULTS (1), which is
    right for a uniform stack: every block is identical, so profiling one
    captures the per-block cost. A **hybrid** stack needs the smallest count
    that instantiates every distinct block type — 4 for Qwen3.8-27B, whose
    ``layer_types`` runs gated-DeltaNet x3 then full attention — or the
    catalog can only ever see one of them."""

    linear_attn_chunk: int | None = None
    """Chunk length the linear-attention (gated-DeltaNet / Mamba) prefill
    scan works in, used to place grid points. None resolves it, in order,
    from the model config's ``chunk_size`` and then from vLLM's own constant.

    This is a *grid placement* knob, not an engine one. Measured cost tracks
    the chunk count, not the token count: on Qwen3.8-27B one token past a
    64-boundary costs 13.5% more than the boundary itself, and the interval
    between boundaries is nearly flat. Sampling on a plain geometric grid
    lands only on boundaries and interpolating across them underestimates by
    that much."""

    hf_overrides: dict[str, Any] | None = None
    """CLI-specified hf_overrides applied on top of the model config
    at vLLM load time."""

    model_config: dict[str, Any] | None = None
    """Full parsed ``configs/model/<path>.json`` — the source of
    truth for the model's shape. At profile time the engine writes
    this dict to a temporary directory as ``config.json`` and points
    vLLM there, so HF hub access is never required and custom-shape
    profiling is one file-edit away."""

    # Target sampling policy, separate from the eager acquisition engine.
    grid_factor: float = 2.0
    """Geometric spacing for ordinary non-graph token/request coordinates.
    Attention, skew and native MoE use their own axis factors."""
    cudagraph: dict[str, Any] | None = None
    """Target serving graph options passed to the shared capture-grid resolver.
    None selects its defaults; these options never enable graph acquisition."""

    # Attention grid
    attention_max_kv: int | None = None
    attention_decode_q_lens: tuple[int, ...] = (1,)
    """Query tokens per decode sequence to sweep on the attention axis.

    ``(1,)`` is ordinary decoding and the default. A speculative-decoding
    verification step submits ``1 + num_speculative_tokens`` queries per
    sequence against that sequence's own KV -- a shape the other four axes
    cannot express, since it is neither a prefill chunk of the same token
    count nor that many single-token decodes. Sweeping it multiplies the
    attention grid, so it is opt-in: pass the ``1 + N`` values you intend to
    simulate. Published N for the four modern families are 3, 4 and 5.
    """
    attention_chunk_factor: float = 2.0
    """Geometric factor for the prefill-token axis. Default 2.0;
    override via --attention-chunk-factor.

    Denser sampling increases acquisition cost without an accuracy guarantee.
    When refreshing one axis, restate the bundle's recorded factors and limits
    for the others; resume also requires compatible identities and shot keys.
    """
    attention_kv_factor: float = 2.0
    """Geometric factor for the prefill-key and kv_decode axes. Default
    2.0; override via --attention-kv-factor. See the note above."""
    attention_n_factor: float = SQRT2
    """Geometric factor for the n_decode axis. Default ``SQRT2``;
    override via --attention-n-factor.

    Request counts retain finer spacing because mixed attention can change
    sharply between doubling points even when pure-decode cost is smooth.
    The floating-point geometric grid interleaves doubling coordinates;
    row reuse still requires matching acquisition identities and full keys.
    """
    # Measurement averaging
    measurement_iterations: int = 3
    """Timed forwards per ordinary shot. The collector averages GPU activity
    using the catalog's per-call normalization; repeated forwards reduce
    sampling noise without providing a hardware-independent error bound."""

    skip_skew: bool = False
    """If True, skip the skew profiling step (skew.csv will not be
    written and alpha fit cannot run). Useful for quick profile runs
    that only need uniform attention data."""

    # Acquisition density is independent of the supported-N fit partitions.
    skew_n_factor: float = 2.0
    skew_pc_factor: float = 2.0
    skew_kp_factor: float = 2.0
    skew_kvs_factor: float = 2.0
    skew_samples_per_cell: int = 32
    """Distribution draws per geometric N/prefill/history operating cell."""
    skew_rounds: int = 3
    """Independent profile contexts; target is the median of forward medians."""
    skew_seed: int = 0
    """Deterministic acquisition seed, shared by every hardware/model."""

    only_skew: bool = False
    """If True, skip dense/per_sequence/attention/moe categories and
    run ONLY the skew profiling step. Useful when the uniform
    attention sweep has already been done and you want to add skew
    data without reprofiling from scratch."""

    force: bool = False
    """If True, wipe existing CSVs before profiling rather than
    resuming. Default (False) preloads existing rows and skips shots
    whose keys are already measured, so a re-run after a feasibility
    change adds only the newly-eligible cases. Applies to both the
    main loop categories (dense/per_sequence/attention/moe) and skew."""

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def effective_variant(self) -> str:
        """Resolved variant — explicit override or auto-derived name.

        Omitted ``dtype`` and ``dtype='auto'`` retain the checkpoint's
        established folder name. On a quantized checkpoint ``--dtype``
        controls floating activations/unquantized modules, not quantized
        weights. A different activation dtype gets a distinct ``-act``
        suffix; matching BF16/FP16 keeps the checkpoint name. FP32 is not
        considered an auto match because vLLM normally downcasts it.
        HF precision fields and any nested ``text_config`` override need an
        explicit ``--variant``; config-class-specific updates are not guessed.

        Examples (typical BF16 models like Llama 3.x):

            no flags                   → "bf16"          (from torch_dtype)
            --kv-cache-dtype fp8       → "bf16-kvfp8"
            --dtype float16            → "fp16"
            FP8 checkpoint + BF16      → "fp8"       (BF16 config)
            FP8 checkpoint + FP16      → "fp8-actfp16" (BF16 config)
        """
        if self.variant is not None:
            return self.variant

        # Do not infer the effect of arbitrary HF config-class overrides.
        # An explicit label remains the escape hatch for deliberate variants.
        if self.hf_overrides and set(self.hf_overrides) & {
                "dtype", "torch_dtype", "quantization_config", "text_config"}:
            raise ValueError("Precision or nested HF overrides require an explicit --variant")

        config = self.model_config or {}
        checkpoint_dtype = model_config_weight_dtype(config)
        base = _short_dtype(checkpoint_dtype) if checkpoint_dtype else "default"
        requested = None if self.dtype is None else str(self.dtype).lower()
        if requested not in (None, "auto"):
            quant = config.get("quantization_config")
            method = quant.get("quant_method") if isinstance(quant, dict) else None
            inner = config.get("text_config")
            nested_quant = inner.get("quantization_config") if isinstance(inner, dict) else None
            if not method and isinstance(nested_quant, dict) and nested_quant.get("quant_method"):
                raise ValueError("Nested-only quantization needs an explicit --variant with --dtype")
            if method:
                activation = _checkpoint_activation_dtype(config)
                selected = _short_dtype(requested)
                if activation not in ("bf16", "fp16") or selected != activation:
                    base += f"-act{selected}"
            else:
                base = _short_dtype(requested)
        parts = [base]

        if self.kv_cache_dtype and self.kv_cache_dtype != "auto":
            parts.append(f"kv{_short_dtype(self.kv_cache_dtype)}")
        return "-".join(parts)


# Short-form dtype names used in variant folder names.
_DTYPE_SHORT: dict[str, str] = {
    "bfloat16": "bf16",
    "bf16": "bf16",
    "float16": "fp16",
    "half": "fp16",
    "fp16": "fp16",
    "float32": "fp32",
    "float": "fp32",
    "fp32": "fp32",
    "fp8": "fp8",
    "fp8_e4m3": "fp8",
    "fp8_e5m2": "fp8e5m2",
    "int8": "int8",
    "int4": "int4",
}


def _short_dtype(d: str) -> str:
    """Return a compact form suitable for folder names.

    Unknown dtypes fall through unchanged — keeps the function from
    silently rewriting quantization scheme names we don't know about.
    """
    return _DTYPE_SHORT.get(str(d), str(d))


def _checkpoint_activation_dtype(config: dict[str, Any]) -> str | None:
    """Unambiguous declared floating dtype, root first then text backbone.

    vLLM's auto resolver may use weight metadata or a platform-specific fallback
    when this is absent/FP32; do not claim an explicit dtype matches that case.
    Conflicting legacy/current aliases also need an intentional variant label.
    This is naming only and never changes the engine's dtype or quantization.
    """
    for node in (config, config.get("text_config")):
        if not isinstance(node, dict):
            continue
        values = {_short_dtype(str(node[key]).lower().removeprefix("torch."))
                  for key in MODEL_DTYPE_KEYS if node.get(key)}
        if len(values) > 1:
            raise ValueError("Conflicting dtype/torch_dtype aliases require an explicit --variant")
        if values:
            return next(iter(values))
    return None


# ---------------------------------------------------------------------------
# vLLM engine default kwargs
# ---------------------------------------------------------------------------
#
# Placed here (not engine.py) so that callers reasoning about config
# merging don't have to import vLLM. These are profiler-critical — user
# overrides via ProfileArgs are merged on top, but most of these should
# not be changed (changing them breaks profiling correctness).

HOST_ENGINE_DEFAULTS: dict[str, Any] = {
    # Don't download checkpoints; we only measure kernel latency.
    "load_format": "dummy",
    # Disable CUDA graphs so every launch is an independently timeable
    # event. The native CUDA collector requires this.
    "enforce_eager": True,
    # Skip tokenizer init — our synthetic batches never tokenize.
    "skip_tokenizer_init": True,
    # Profiling must be deterministic; prefix caching is not.
    "enable_prefix_caching": False,
    # Silences a "generation_config not set" warning; harmless otherwise.
    "generation_config": "vllm",
    # Default TP; actual engine always spins up single-GPU (we emulate
    # multi-TP via shrunk hf_overrides, see engine.fuse_engine_kwargs).
    "tensor_parallel_size": 1,
    # Paging block size is deliberately NOT set: vLLM's platform layer picks
    # one the model's attention backend can actually use, and forcing a value
    # can leave it with none. DeepSeek-V3.2's sparse MLA is the case in point --
    # requesting 16 fails backend selection outright
    # ("FLASHINFER_MLA_SPARSE_SM120: [block_size not supported]") where letting
    # vLLM choose gives 64. It only sizes the synthetic block table and never
    # changes kernel time, so there was nothing to gain by pinning it.
    # ``--block-size`` still overrides, and ``probe_limits`` reports whatever
    # the engine settled on so the feasibility filters use the real value.
    # KV cache fraction of GPU memory. 0.9 is generous for the
    # 1-decoder-layer dummy model.
    "gpu_memory_utilization": 0.9,
    # Batch budget defaults when user doesn't specify.
    "max_num_batched_tokens": 2048,
    "max_num_seqs": 256,
    # Only one decoder layer — all blocks are identical, so profiling
    # one captures the per-block cost and keeps profiling cheap.
    "hf_overrides": {"num_hidden_layers": 1},
}
