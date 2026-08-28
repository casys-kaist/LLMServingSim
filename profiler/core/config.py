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
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Constants shared with engine.py
# ---------------------------------------------------------------------------

# HF config fields divided by TP to emulate a single rank of a
# multi-TP deployment (see engine.fuse_engine_kwargs). The same list
# covers every common dense + MoE architecture, so it's a module
# constant rather than per-architecture data.
SHARD_FIELDS: list[str] = [
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
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
    """The weight dtype a model config declares, or None.

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


def probe_moe_params(hf_cfg: dict[str, Any]) -> tuple[int, int] | None:
    """Extract (num_experts, top_k) from a HuggingFace config dict.

    Returns None if the config doesn't declare both — caller decides
    how to react (usually: fail if the model is supposed to be MoE).
    """
    num_experts = next(
        (hf_cfg[k] for k in MOE_NUM_EXPERTS_KEYS if k in hf_cfg),
        None,
    )
    top_k = next(
        (hf_cfg[k] for k in MOE_TOP_K_KEYS if k in hf_cfg),
        None,
    )
    if num_experts is None or top_k is None:
        return None
    return (int(num_experts), int(top_k))


# ---------------------------------------------------------------------------
# Catalog (loaded from architecture yaml)
# ---------------------------------------------------------------------------

class LayerEntry(BaseModel):
    """One row of the catalog.

    Each entry binds a canonical layer name (the YAML key) to the vLLM
    Python class that the CUDA profiler will report, plus optional
    disambiguation (``within``) and TP-invariance (``tp_stable``).
    """
    # extra="forbid" catches typos in YAML early (e.g., `tp_stabe: true`).
    model_config = ConfigDict(extra="forbid")

    vllm: str
    """vLLM leaf class name as reported by the CUDA profiler, e.g.
    ``"QKVParallelLinear"``, ``"RMSNorm"``, ``"Attention"``."""

    within: str | None = None
    """Optional immediate-parent class name to disambiguate when the
    same ``vllm`` class appears multiple times in the model (most
    commonly RMSNorm, which shows up as input/post/final layernorm)."""

    tp_stable: bool = False
    """If True, profile this layer only at TP=1 and replicate the
    results into every tp{N}/ folder."""


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

    def all_entries(self) -> list[tuple[str, str, LayerEntry]]:
        """Flatten to ``[(profile_kind, layer_name, entry), ...]``."""
        out: list[tuple[str, str, LayerEntry]] = []
        for kind in ("dense", "per_sequence", "attention", "linear_attention",
                     "moe"):
            for name, entry in getattr(self, kind).items():
                out.append((kind, name, entry))
        return out


class Sequence(BaseModel):
    """Ordered pipeline the simulator's ``trace_generator`` walks to
    emit one iteration. Not used by the profiler itself — only
    validated here so typos in the yaml fail loudly before profiling.

    This is the **uniform-stack** form: one attention block type, repeated
    ``num_hidden_layers`` times. A heterogeneous stack uses ``blocks:``
    instead (see :class:`Blocks`); the two are mutually exclusive and an
    architecture declares exactly one.
    """
    model_config = ConfigDict(extra="forbid")

    prologue: list[str] = Field(default_factory=list)
    pre_attn: list[str] = Field(default_factory=list)
    post_attn: list[str] = Field(default_factory=list)
    mlp_dense: list[str] = Field(default_factory=list)
    mlp_moe: list[str] = Field(default_factory=list)
    head: list[str] = Field(default_factory=list)

    def all_layers(self) -> list[str]:
        return [
            *self.prologue, *self.pre_attn, *self.post_attn,
            *self.mlp_dense, *self.mlp_moe, *self.head,
        ]

    def occurrences(self) -> dict[str, int]:
        """Emissions per block, for the timing denominator.

        ``prologue`` and ``head`` fire once per *iteration* rather than once
        per block, but they are also emitted once, so a plain count is right
        for every section here.
        """
        counts: dict[str, int] = {}
        for name in self.all_layers():
            counts[name] = counts.get(name, 0) + 1
        return counts


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
    mlp: dict[str, list[str]] = Field(default_factory=dict)

    def all_layers(self) -> list[str]:
        out: list[str] = []
        for block in self.attn.values():
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


class Architecture(BaseModel):
    """Parsed architecture yaml.

    Holds ``catalog`` (vLLM class bindings — used by the profiler) and the
    layer order the simulator's trace generator walks. That order comes in one
    of two forms, and an architecture declares exactly one:

    * ``sequence`` — a uniform stack, every block identical. What the five
      original architectures use.
    * ``blocks`` + ``shared`` — a heterogeneous stack, keyed by axis. The
      per-layer values come from the checkpoint's config
      (``layer_types``, ``first_k_dense_replace``, ...), not from here.
    """
    model_config = ConfigDict(extra="forbid")

    catalog: Catalog
    model_types: list[str] = Field(default_factory=list)
    sequence: Sequence | None = None
    blocks: Blocks | None = None
    shared: Shared | None = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _check_catalog(self) -> "Architecture":
        # Canonical names must be unique across ALL catalog groups.
        seen: set[str] = set()
        for _, name, _ in self.catalog.all_entries():
            if name in seen:
                raise ValueError(f"Layer name {name!r} appears twice in catalog")
            seen.add(name)

        # At least one softmax-attention entry, unless the architecture is
        # purely linear-attention. More than one is allowed: a sparse-attention
        # model runs an indexer kernel alongside the attention kernel, and both
        # are keyed on the same axes, so both belong in this category.
        if not self.catalog.attention and not self.catalog.linear_attention:
            raise ValueError(
                "catalog must declare at least one attention or "
                "linear_attention entry"
            )

        # (vllm, within) pairs globally unique so layer matching is
        # deterministic. (Multiple catalog-tree nodes can match one
        # entry, that's fine — their timings get averaged by the sink.)
        pairs: dict[tuple[str, str | None], str] = {}
        for _, name, entry in self.catalog.all_entries():
            key = (entry.vllm, entry.within)
            if key in pairs:
                raise ValueError(
                    f"Ambiguous layer binding: {name!r} and {pairs[key]!r} "
                    f"both resolve to (vllm={entry.vllm!r}, "
                    f"within={entry.within!r})"
                )
            pairs[key] = name

        # Exactly one layer-order form. Declaring both would leave it
        # ambiguous which one the trace generator walks, and declaring
        # neither leaves it with nothing to walk.
        if self.sequence is not None and self.blocks is not None:
            raise ValueError(
                "declare either 'sequence' (uniform stack) or 'blocks' "
                "(heterogeneous stack), not both"
            )
        if self.blocks is None and self.shared is not None:
            raise ValueError("'shared' only applies alongside 'blocks'")

        # Every referenced name must be a canonical name declared in the
        # catalog — catches typos before a profile/simulation run.
        catalog_names = {n for _, n, _ in self.catalog.all_entries()}
        referenced: list[str] = []
        if self.sequence is not None:
            referenced += self.sequence.all_layers()
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
        for source in (self.sequence, self.blocks, self.shared):
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


def _declared_model_types(path: Path) -> list[str]:
    """The ``model_types:`` list a yaml declares, or [] — cheaply.

    Deliberately does not build an ``Architecture``: this runs over every file
    in the directory during a miss, and a malformed *other* file must not stop
    the one we are looking for from resolving.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except Exception:
        return []
    if not isinstance(raw, dict):
        return []
    declared = raw.get("model_types") or []
    if not isinstance(declared, list):
        return []
    return [str(v) for v in declared]


def resolve_architecture_by_model_type(
    model_type: str,
    arch_dir: Path,
) -> Path:
    """Find the architecture yaml serving ``model_type``.

    **Naming rule.** The file is named after the HF ``model_type`` it primarily
    serves: ``<model_type>.yaml``. ``model_type`` is read from the top level of
    the ``configs/model/<org>/<name>.json`` handed to the profiler, so for a
    wrapped checkpoint the config-authoring convention decides it — store the
    **text tower flattened to top level**, with ``architectures`` set to the
    text-only class, and the recorded ``model_type`` is the text tower's
    (``qwen3_5_text``, not the VL wrapper's ``qwen3_5``). One architecture, one
    name.

    **One file, several model_types.** Some ``model_type`` values are different
    checkpoints of one implementation — GLM-5 (``glm_moe_dsa``) and
    DeepSeek-V3.2 (``deepseek_v32``) both run vLLM's ``deepseek_v2`` path, and
    a wrapped checkpoint answers to both its own and its text tower's name.
    Rather than duplicate a catalog or lean on symlinks, a yaml lists every
    value it serves in ``model_types:``. Resolution tries the filename first
    and only scans for a declaration on a miss, so the common path stays a
    single stat.

    Two files must not claim the same ``model_type``; that is an error rather
    than a silent first-wins, because which catalog you got would then depend
    on directory order.
    """
    candidate = arch_dir / f"{model_type}.yaml"
    if candidate.is_file():
        return candidate.resolve()

    matches = [
        path for path in sorted(arch_dir.glob("*.yaml"))
        if model_type in _declared_model_types(path)
    ]
    if len(matches) > 1:
        raise ValueError(
            f"model_type={model_type!r} is declared by more than one "
            f"architecture yaml: {[str(m) for m in matches]}. Each "
            f"model_type must resolve to exactly one catalog."
        )
    if matches:
        return matches[0].resolve()

    # List everything resolvable, not just filenames, so the message is
    # actionable when the name lives in a model_types: list.
    available: dict[str, list[str]] = {}
    for path in sorted(arch_dir.glob("*.yaml")):
        available[path.stem] = _declared_model_types(path)
    listing = "\n".join(
        f"  {stem}.yaml" + (f" (also serves: {', '.join(extra)})" if extra else "")
        for stem, extra in available.items()
    )
    raise FileNotFoundError(
        f"No architecture yaml found for model_type={model_type!r}. "
        f"Tried {candidate}, and no yaml declares it under 'model_types:'.\n"
        f"Available architectures:\n{listing}\n"
        f"To add support, either create {candidate.name} under {arch_dir} with "
        f"a catalog matching this model family's vLLM classes, or -- if an "
        f"existing catalog already describes this implementation -- add "
        f"{model_type!r} to that file's 'model_types:' list."
    )


# ---------------------------------------------------------------------------
# Profile session args (CLI, no yaml)
# ---------------------------------------------------------------------------

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
        tp_degrees: Which TP shardings to sweep. Must include 1.
        variant: Free-form output folder label under the model's
            directory. If omitted at the CLI, auto-derived from
            ``dtype`` + ``kv_cache_dtype`` so that profiles with
            different engine kwargs don't collide.
        dtype / kv_cache_dtype / max_num_batched_tokens / max_num_seqs:
            Common vLLM engine kwargs. None means "use defaults"
            (HOST_ENGINE_DEFAULTS for max_*, vLLM default for dtype).
        attention_max_kv: Cap for attention grid's KV axes. Doubles
            from 512 up to min(this, max_model_len).
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
    HOST_ENGINE_DEFAULTS (16). Exposed because the simulator has the same knob
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

    # Attention grid
    attention_max_kv: int = 16384
    attention_chunk_factor: float = 2.0
    """Geometric factor for the prefill_chunk axis. Default 2.0
    (doubling). Override via --attention-chunk-factor."""
    attention_kv_factor: float = 2.0
    """Geometric factor for the kv_prefill / kv_decode axes. Default
    2.0 (doubling). Override via --attention-kv-factor."""

    # Measurement averaging
    measurement_iterations: int = 3
    """N timed forwards per shot, averaged."""

    skip_skew: bool = False
    """If True, skip the skew profiling step (skew.csv will not be
    written and alpha fit cannot run). Useful for quick profile runs
    that only need uniform attention data."""

    # Skew grid density. Mirrors the attention factor knobs — the
    # default 2.0 (doubling) is what ships today; crank higher
    # (e.g. 4.0) to coarsen the sweep and cut profile time when the
    # target workload doesn't stress every axis.
    skew_n_factor: float = 2.0
    """Geometric factor for the skew n (total decode count) axis.
    Default 2.0 (doubling). Override via --skew-n-factor."""
    skew_pc_factor: float = 2.0
    """Geometric factor for the skew pc (prefill chunk) axis.
    Default 2.0. Override via --skew-pc-factor."""
    skew_kp_factor: float = 2.0
    """Geometric factor for the skew kp (prefill history) axis.
    Default 2.0. Override via --skew-kp-factor."""
    skew_kvs_factor: float = 2.0
    """Geometric factor for the skew kvs (small-decode kv) axis.
    Default 2.0. Override via --skew-kvs-factor."""

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
    """Number of timed forward passes per shot, averaged by vLLM's
    layerwise_profile via its ``invocations`` count. A single sample
    can swing 15-25% on large GEMMs due to DVFS / clock-state jitter;
    N=3 cuts that jitter to ~5% at ~3x profile time."""

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def effective_variant(self) -> str:
        """Resolved variant — explicit override or auto-derived name.

        When the user doesn't pass ``--variant``, we name the folder
        after the engine flags that actually change kernel timings:
        ``dtype`` and ``kv_cache_dtype``. The weight dtype defaults to
        the model config's ``torch_dtype`` / ``dtype`` so the folder carries
        meaningful info (no bare "default").

        Examples (typical BF16 models like Llama 3.x):

            no flags                   → "bf16"          (from torch_dtype)
            --kv-cache-dtype fp8       → "bf16-kvfp8"
            --dtype fp8                → "fp8"
            --dtype fp8 --kv... fp8    → "fp8-kvfp8"
        """
        if self.variant is not None:
            return self.variant

        # Weight dtype: CLI wins; otherwise probe the model config.
        weight_dtype = self.dtype
        if weight_dtype is None and self.model_config:
            weight_dtype = model_config_weight_dtype(self.model_config)
        parts = [_short_dtype(weight_dtype) if weight_dtype else "default"]

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
    # event. layerwise_profile requires this.
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
    # Paging block size. Only affects how we size the synthetic block
    # table; does not change kernel time.
    "block_size": 16,
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
