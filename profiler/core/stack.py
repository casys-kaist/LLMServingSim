"""Resolve a checkpoint's per-layer block composition from its HF config.

A modern decoder stack is not N identical blocks. Qwen3.5/3.6/3.8 interleave
gated DeltaNet with full attention, GLM and DeepSeek run a dense MLP for the
first few layers and MoE after, and a checkpoint can vary both at once. The
profiler needs to know that for one reason above all: it shrinks the model to
save time, and shrinking to a single layer on a hybrid stack means the catalog
only ever sees whichever block type happens to come first. Everything else
about that layer type then reads as free.

So this module answers two questions from the config alone:

* ``resolve_stack`` — what block does each layer run?
* ``minimal_layer_count`` — the smallest prefix that instantiates every
  distinct block, which is what the profiler should shrink to.

**Only rules read out of vLLM's own source are implemented.** The field names
in this area are not standardised and their conventions genuinely disagree —
DeepSeek's MoE test is ``layer_idx % moe_layer_freq`` while Qwen3-MoE's is
``(layer_idx + 1) % decoder_sparse_step``, an off-by-one in opposite
directions — so a plausible-looking guess is a wrong answer, not a
near-miss. An unrecognised layout falls back to "uniform" and says so, which
costs a hybrid nothing worse than the behaviour before this module existed.

Kept free of third-party imports so the simulator can share it: the
trace generator needs the same per-layer answer to emit the right block, and
two implementations of this would drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Attention-type value used when a config gives no basis to differentiate.
FULL_ATTENTION = "full_attention"

MLP_DENSE = "dense"
MLP_MOE = "moe"

# Config keys that carry the expert count, mirroring config.MOE_NUM_EXPERTS_KEYS.
# Duplicated deliberately rather than imported: config.py pulls in pydantic,
# and this module stays importable from the simulator container.
_EXPERT_COUNT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")


@dataclass(frozen=True)
class LayerSpec:
    """What one decoder layer is made of.

    ``sparse`` is whether this layer runs a sparse-attention selection branch
    on top of its attention kernel. It is a third axis rather than part of
    ``attn`` because vendors vary it independently of the attention type.
    """

    attn: str
    mlp: str
    sparse: bool = False


def text_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """The sub-dict holding the text backbone's hyperparameters.

    A wrapped checkpoint may be stored either way, because the right shape is
    whatever the model's own config class reads. Qwen3.5's class resolves a
    flat config, so that one is flattened; MiniMax-M3's is a wrapper that
    builds its backbone from a ``text_config`` key, and flattening it makes the
    wrapper fall back to an all-defaults backbone -- silently, with every value
    we wrote swallowed by ``**kwargs``.

    So every reader of a model config goes through here rather than indexing
    the top level, or a nested config looks empty and each helper quietly
    answers for a model nobody asked about. Top-level keys still win when both
    exist, since that is where a flat config puts them.
    """
    inner = cfg.get("text_config")
    if isinstance(inner, dict) and inner:
        merged = dict(inner)
        merged.update({k: v for k, v in cfg.items() if k != "text_config"})
        # A wrapper's own top-level keys (model_type, architectures) must not
        # shadow the backbone's, but the backbone has no such keys, so the
        # update above is safe -- except for the two the wrapper always sets.
        for key in ("num_hidden_layers", "num_local_experts", "num_experts",
                    "n_routed_experts", "moe_layer_freq", "layer_types",
                    "sparse_attention_config", "hidden_size"):
            if key in inner:
                merged[key] = inner[key]
        return merged
    return cfg


def _num_layers(cfg: dict[str, Any]) -> int:
    try:
        n = int(cfg.get("num_hidden_layers") or 0)
    except (TypeError, ValueError):
        return 0
    return max(n, 0)


def _experts(cfg: dict[str, Any]) -> int:
    for key in _EXPERT_COUNT_KEYS:
        if key in cfg:
            try:
                return int(cfg[key] or 0)
            except (TypeError, ValueError):
                continue
    return 0


def _attn_types(cfg: dict[str, Any], n: int) -> list[str]:
    """Attention type per layer.

    Verified against vLLM:

    * ``layer_types`` is indexed directly —
      ``qwen3_5.py``: ``layer_type=config.layer_types[extract_layer_index(prefix)]``.
    * when absent, ``full_attention_interval`` generates it —
      ``transformers_utils/configs/qwen3_5.py``:
      ``"linear_attention" if bool((i + 1) % interval) else "full_attention"``.
    """
    declared = cfg.get("layer_types")
    if isinstance(declared, list) and declared:
        # A config may declare more entries than the (possibly shrunk) layer
        # count; index into it exactly as vLLM does.
        return [str(declared[i % len(declared)]) for i in range(n)]

    interval = cfg.get("full_attention_interval")
    if isinstance(interval, int) and interval > 0:
        return [
            "linear_attention" if (i + 1) % interval else FULL_ATTENTION
            for i in range(n)
        ]

    return [FULL_ATTENTION] * n


def _mlp_types(cfg: dict[str, Any], n: int) -> list[str]:
    """MLP type per layer.

    Verified against vLLM, and note the two rules disagree on the off-by-one:

    * ``qwen3_moe.py``: MoE iff
      ``layer_idx not in mlp_only_layers and num_experts > 0
      and (layer_idx + 1) % decoder_sparse_step == 0``
    * ``deepseek_v2.py``: MoE iff
      ``n_routed_experts is not None and layer_idx >= first_k_dense_replace
      and layer_idx % moe_layer_freq == 0``

    A **list**-valued ``moe_layer_freq`` is a different convention again — a
    per-layer 0/1 flag, as MiniMax-M3 ships — and is read as such. That one is
    inferred from the config's own shape rather than from source, since the
    implementation is out-of-tree; being wrong costs a wrong layer count, which
    surfaces as an empty CSV column rather than a bad number.
    """
    experts = _experts(cfg)
    if experts <= 0:
        return [MLP_DENSE] * n

    freq = cfg.get("moe_layer_freq")
    if isinstance(freq, list) and freq:
        return [
            MLP_MOE if freq[i % len(freq)] else MLP_DENSE for i in range(n)
        ]

    step = cfg.get("decoder_sparse_step")
    if isinstance(step, int) and step > 0:
        only_mlp = set(cfg.get("mlp_only_layers") or [])
        return [
            MLP_DENSE if (i in only_mlp or (i + 1) % step) else MLP_MOE
            for i in range(n)
        ]

    first_dense = cfg.get("first_k_dense_replace")
    if isinstance(first_dense, int):
        stride = freq if isinstance(freq, int) and freq > 0 else 1
        return [
            MLP_MOE if (i >= first_dense and i % stride == 0) else MLP_DENSE
            for i in range(n)
        ]

    return [MLP_MOE] * n


def _sparse_flags(cfg: dict[str, Any], n: int) -> list[bool]:
    """Whether each layer runs a sparse-attention selection branch.

    Verified against two implementations, which share no field names:

    * MiniMax-M3 (``vllm/models/minimax_m3/nvidia/model.py``):
      ``{i for i, f in enumerate(sparse_attention_config["sparse_attention_freq"])
      if f != 0}``.
    * DeepSeek-V3.2 / GLM-5 (``deepseek_v2.py``): the model is sparse at all
      iff it declares ``index_topk``, and a layer *skips* the top-k when
      ``index_topk_pattern[layer_id] == "S"``, or -- when no pattern is given --
      when ``max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq
      != 0``. ``index_topk_freq`` defaults to 1, under which that is never
      true, so a config declaring neither field is sparse on every layer.
    """
    sac = cfg.get("sparse_attention_config")
    if isinstance(sac, dict):
        freq = sac.get("sparse_attention_freq")
        if isinstance(freq, list) and freq:
            return [bool(freq[i % len(freq)]) for i in range(n)]
        return [bool(sac.get("use_sparse_attention"))] * n

    if "index_topk" not in cfg:
        return [False] * n

    pattern = cfg.get("index_topk_pattern")
    if isinstance(pattern, (str, list)) and len(pattern):
        return [
            not (pattern[i] == "S") if i < len(pattern) else True
            for i in range(n)
        ]
    try:
        topk_freq = int(cfg.get("index_topk_freq", 1) or 1)
    except (TypeError, ValueError):
        topk_freq = 1
    try:
        offset = int(cfg.get("index_skip_topk_offset", 2))
    except (TypeError, ValueError):
        offset = 2
    if topk_freq <= 1:
        return [True] * n
    return [
        max(i - offset + 1, 0) % topk_freq == 0 for i in range(n)
    ]


def resolve_stack(cfg: dict[str, Any]) -> list[LayerSpec]:
    """One :class:`LayerSpec` per decoder layer, in order.

    Empty when the config declares no layer count.
    """
    cfg = text_config(cfg)
    n = _num_layers(cfg)
    if n == 0:
        return []
    attn = _attn_types(cfg, n)
    mlp = _mlp_types(cfg, n)
    sparse = _sparse_flags(cfg, n)
    return [
        LayerSpec(attn=attn[i], mlp=mlp[i], sparse=sparse[i])
        for i in range(n)
    ]


def is_uniform(cfg: dict[str, Any]) -> bool:
    """True when every layer is the same block, i.e. the old assumption holds."""
    stack = resolve_stack(cfg)
    return len(set(stack)) <= 1


#: The axes of a :class:`LayerSpec`, and the subsets a category can depend on.
#: ``attn`` and ``sparse`` travel together: a sparse-attention layer runs a
#: different kernel from a non-sparse one of the same attention type.
ATTENTION_AXES = ("attn", "sparse")
MLP_AXES = ("mlp",)
ALL_AXES = ATTENTION_AXES + MLP_AXES


def minimal_layer_count_for(
    cfg: dict[str, Any], axes: tuple[str, ...] = ALL_AXES
) -> int:
    """Smallest layer count that instantiates every value of ``axes``.

    The smallest **prefix**, not the smallest subset: the profiler shrinks by
    setting ``num_hidden_layers``, which keeps layers ``0..N-1``, so a value
    that first appears at layer 40 forces N to 41 whether we like it or not.

    ``axes`` is what makes this per **category**. Every layer costs the
    profiler its whole op count on every shot -- measured at 372 ms of
    profiling overhead per forward on a 4-layer DeepSeek-V3.2 against 94 ms at
    one layer -- so a category should not pay for an axis it does not measure.
    DeepSeek-V3.2 and GLM-5 need 4 layers only because their MLP turns from
    dense to MoE at ``first_k_dense_replace``; every one of their layers has
    the same attention, so the attention sweep needs **1**. Qwen3.8-27B is the
    mirror image: 4 for attention (gated DeltaNet three times before the first
    full-attention layer) and 1 for the MLP.

    Returns 1 for a stack that is uniform on the requested axes, and never
    more than the model's own layer count.
    """
    stack = resolve_stack(cfg)
    if not stack:
        return 1

    def key(spec: "LayerSpec") -> tuple:
        return tuple(getattr(spec, a) for a in axes)

    wanted = {key(spec) for spec in stack}
    seen: set[tuple] = set()
    for i, spec in enumerate(stack):
        seen.add(key(spec))
        if seen == wanted:
            return i + 1
    return len(stack)


def minimal_layer_count(cfg: dict[str, Any]) -> int:
    """Smallest layer count that instantiates every distinct block type.

    The all-axes case of :func:`minimal_layer_count_for`, which is what a
    category measuring both axes (``dense``) needs, and the safe default.
    """
    return minimal_layer_count_for(cfg, ALL_AXES)


def describe(cfg: dict[str, Any]) -> str:
    """One-line summary for the run log, so the choice is visible.

    The count matters enough to state rather than leave implicit: it decides
    whether a block type gets profiled at all.
    """
    stack = resolve_stack(cfg)
    if not stack:
        return "layer stack: unknown (config declares no num_hidden_layers)"
    n = minimal_layer_count(cfg)
    if len(set(stack)) <= 1:
        only = stack[0]
        return (
            f"layer stack: uniform, {len(stack)} x "
            f"({only.attn}, {only.mlp} mlp"
            f"{', sparse' if only.sparse else ''}) -> profiling {n} layer"
        )
    counts: dict[LayerSpec, int] = {}
    for spec in stack:
        counts[spec] = counts.get(spec, 0) + 1
    shapes = ", ".join(
        f"{c}x({s.attn}, {s.mlp}{', sparse' if s.sparse else ''})"
        for s, c in sorted(counts.items(), key=lambda kv: -kv[1])
    )
    return (
        f"layer stack: heterogeneous over {len(stack)} layers -- {shapes} "
        f"-> profiling {n} layers to reach every block type"
    )
