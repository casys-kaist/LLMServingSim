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
    moe: dict[str, LayerEntry] = Field(default_factory=dict)

    def all_entries(self) -> list[tuple[str, str, LayerEntry]]:
        """Flatten to ``[(profile_kind, layer_name, entry), ...]``."""
        out: list[tuple[str, str, LayerEntry]] = []
        for kind in ("dense", "per_sequence", "attention", "moe"):
            for name, entry in getattr(self, kind).items():
                out.append((kind, name, entry))
        return out


class Sequence(BaseModel):
    """Ordered pipeline the simulator's ``trace_generator`` walks to
    emit one iteration. Not used by the profiler itself — only
    validated here so typos in the yaml fail loudly before profiling.
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


class Architecture(BaseModel):
    """Parsed architecture yaml.

    Holds ``catalog`` (vLLM class bindings — used by the profiler) and
    ``sequence`` (ordered canonical names — used by the simulator's
    trace generator).
    """
    model_config = ConfigDict(extra="forbid")

    catalog: Catalog
    sequence: Sequence | None = None

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

        # Exactly one attention entry.
        if len(self.catalog.attention) != 1:
            raise ValueError(
                f"catalog.attention must have exactly 1 entry; got "
                f"{len(self.catalog.attention)}"
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

        # Every sequence entry must be a canonical name declared in the
        # catalog — catches typos before a profile/simulation run.
        if self.sequence is not None:
            catalog_names = {n for _, n, _ in self.catalog.all_entries()}
            unknown = [n for n in self.sequence.all_layers() if n not in catalog_names]
            if unknown:
                raise ValueError(
                    f"sequence references names not in catalog: {sorted(set(unknown))}"
                )

        return self

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def has_moe(self) -> bool:
        return bool(self.catalog.moe)

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
    """Find the architecture yaml matching ``model_type``.

    Convention: architecture yaml filename equals ``<model_type>.yaml``.
    So ``model_type == "qwen3_moe"`` resolves to
    ``<arch_dir>/qwen3_moe.yaml``.
    """
    candidate = arch_dir / f"{model_type}.yaml"
    if candidate.is_file():
        return candidate.resolve()

    # List available architectures to help the user decide what to do.
    available = sorted(p.stem for p in arch_dir.glob("*.yaml"))
    raise FileNotFoundError(
        f"No architecture yaml found for model_type={model_type!r}. "
        f"Tried {candidate}.\n"
        f"Available architectures: {available}\n"
        f"To add support, create {candidate.name} under {arch_dir} "
        f"with a catalog matching this model family's vLLM classes."
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
        the model config's ``torch_dtype`` so the folder always carries
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
            weight_dtype = self.model_config.get("torch_dtype")
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
