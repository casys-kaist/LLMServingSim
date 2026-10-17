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
import copy
import math
from typing import Any

# Attention-type value used when a config gives no basis to differentiate.
FULL_ATTENTION = "full_attention"

MLP_DENSE = "dense"
MLP_MOE = "moe"

# Config keys that carry the expert count, mirroring config.MOE_NUM_EXPERTS_KEYS.
# Duplicated deliberately rather than imported: config.py pulls in pydantic,
# and this module stays importable from the simulator container.
_EXPERT_COUNT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")


def materialize_prefix(config, overrides):
    """Materialize backbone overrides and validate schedules before truncation.

    Returns a new config without mutating either input. For a wrapper, only
    fields its text config owns (and explicit decoder/schedule fields) move
    into that text config; the engine still owns forwarding/consuming overrides.
    It must forward an explicitly overridden layer_types/linear_attn_config
    from this returned config, so native config loading cannot undo clipping.
    """
    kimi = config.get("model_type") in ("kimi_k3", "kimi_linear")
    if kimi and "text_config" in overrides:
        raise ValueError("Backbone overrides must be explicit top-level fields")
    result = copy.deepcopy(config)
    inner = result.get("text_config", result)
    if not isinstance(inner, dict):
        raise ValueError("text_config must be a mapping")
    source_depth = inner.get("num_hidden_layers")
    for name, value in overrides.items():
        if (inner is result and name != "text_config") or name in inner or name in ("num_hidden_layers", "linear_attn_config",
                                     "layer_types", "full_attention_interval"):
            inner[name] = copy.deepcopy(value)
    depth = overrides.get("num_hidden_layers", source_depth)
    if depth is None:
        return result
    if type(depth) is not int or depth <= 0:
        raise ValueError("Profiling depth must be a positive integer")
    if type(source_depth) is not int or source_depth <= 0:
        source_depth = depth
    if kimi and depth > source_depth:
        raise ValueError("Kimi prefix must be within the original decoder depth")
    # Keep the whole effective source schedule visible before clipping it.
    # Uniform non-Kimi configs retain their previous ability to change depth;
    # an indexed schedule must cover both the source and requested depths.
    validation = copy.deepcopy(result)
    validation.get("text_config", validation)["num_hidden_layers"] = max(source_depth, depth)
    resolve_stack(validation)
    inner["num_hidden_layers"] = depth
    if kimi and inner.get("linear_attn_config") is not None:
        for name in ("kda_layers", "full_attn_layers"):
            inner["linear_attn_config"][name] = [
                index for index in inner["linear_attn_config"][name] if index <= depth]
    if isinstance(inner.get("layer_types"), list):
        inner["layer_types"] = inner["layer_types"][:depth]
    return result


def attn_res_roles(config, *, depth=None):
    """Kimi's local source roles, independent of a checkpoint or device.

    vLLM 0.31 kimi_k3/nvidia/model.py:962-1060,1163-1166,1373-1386.
    This describes PP1, without sequence parallelism; it does not certify that
    the runtime can execute any particular dimensions, dtype or backend.
    """
    if config.get("model_type") not in ("kimi_k3", "kimi_linear"):
        raise ValueError("AttnRes roles require the Kimi source rule")
    inner = config.get("text_config", config)
    stack = resolve_stack(config, prefix_layers=depth)
    depth = len(stack)
    width, hidden = inner.get("attn_res_block_size"), inner.get("hidden_size")
    if type(width) is not int or width <= 0 or type(hidden) is not int or hidden <= 0:
        raise ValueError("AttnRes block size and hidden size must be positive integers")
    eps = inner.get("rms_norm_eps")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("AttnRes normalization epsilon must be positive and finite")
    capacity = (depth + width - 1) // width
    common = dict(capacity=capacity, hidden=hidden, eps=eps)
    result = []
    for index in range(depth):
        boundary = index % width == 0
        previous = (index + width - 1) // width
        result.append(dict(layer=index, role="pre_attn", num_blocks=previous,
            block_write_idx=index // width if boundary else -1,
            delta_present=index != 0, output_norm_present=True,
            output_norm_eps=eps, **common))
        result.append(dict(layer=index, role="post_attn", num_blocks=previous + int(boundary),
            block_write_idx=-1, delta_present=not boundary, output_norm_present=True,
            output_norm_eps=eps, **common))
    result.append(dict(layer=None, role="final", num_blocks=capacity, block_write_idx=-1,
        delta_present=True, output_norm_present=False, output_norm_eps=0.0, **common))
    return result


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


def expert_top_k(cfg: dict[str, Any], default: int | None = None) -> int | None:
    """One top-k across profiler and serving; refuse conflicting aliases.

    KimiLinearConfig calls this num_experts_per_token (vLLM 0.31
    models/kimi_k3/nvidia/model.py:570), unlike DeepSeek/Qwen/Mixtral.
    Declared values must be positive integers, not bools or coerced strings.
    """
    values = [(name, cfg[name]) for name in
              ("num_experts_per_tok", "num_experts_per_token", "moe_k") if name in cfg]
    if not values:
        return default
    if any(type(value) is not int or value < 1 for _, value in values):
        raise ValueError("Expert top-k must be a positive integer")
    if len({value for _, value in values}) != 1:
        raise ValueError("Conflicting expert top-k aliases: " + repr(values))
    return values[0][1]


def _kimi_attn_types(cfg: dict[str, Any], n: int) -> list[str]:
    """Kimi's original one-based schedule, restricted to a leading prefix.

    vLLM 0.31 kimi_linear.py:157-161 selects KDA by membership in
    ``kda_layers``; kimi_k3/nvidia/model.py:881-928 takes MLA otherwise.
    The full list validation below is deliberately stricter than that predicate:
    an inconsistent checkpoint must not silently acquire a different stack.
    ``num_hidden_layers`` is the original depth, never inferred from list maxima.
    This describes attention/MLP composition, not AttnRes execution coverage.
    """
    depth = cfg.get("num_hidden_layers")
    if type(depth) is not int or depth < 1:
        raise ValueError("Kimi num_hidden_layers must be a positive integer")
    if type(n) is not int or not 1 <= n <= depth:
        raise ValueError("Kimi prefix must be within the original model depth")

    linear = cfg.get("linear_attn_config")
    if linear is None:
        expected = [FULL_ATTENTION] * depth
    else:
        if not isinstance(linear, dict):
            raise ValueError("Kimi linear_attn_config must be a mapping or null")
        schedules = {}
        for key in ("kda_layers", "full_attn_layers"):
            values = linear.get(key)
            if not isinstance(values, list):
                raise ValueError(f"Kimi {key} must be an explicit list")
            if any(type(value) is not int or not 1 <= value <= depth for value in values):
                raise ValueError(f"Kimi {key} must contain one-based indices within the original depth")
            if len(values) != len(set(values)):
                raise ValueError(f"Kimi {key} repeats an index")
            schedules[key] = set(values)
        kda, full = schedules["kda_layers"], schedules["full_attn_layers"]
        if kda & full or kda | full != set(range(1, depth + 1)):
            raise ValueError("Kimi attention lists must partition the original decoder stack")
        expected = ["linear_attention" if index in kda else FULL_ATTENTION
                    for index in range(1, depth + 1)]

    # Native Kimi does not dispatch on these generic keys. Accept redundant
    # declarations only when they agree, rather than letting them take precedence.
    if "layer_types" in cfg and cfg["layer_types"] != expected:
        raise ValueError("Kimi layer_types contradicts its native attention schedule")
    if "full_attention_interval" in cfg:
        interval = cfg["full_attention_interval"]
        if type(interval) is not int or interval < 1 or any(
                kind != ("linear_attention" if (index + 1) % interval else FULL_ATTENTION)
                for index, kind in enumerate(expected)):
            raise ValueError("Kimi full_attention_interval contradicts its native attention schedule")
    return expected[:n]


def _attn_types(cfg: dict[str, Any], n: int) -> list[str]:
    """Attention type per layer.

    Verified against vLLM:

    * ``layer_types`` is indexed directly —
      ``qwen3_5.py``: ``layer_type=config.layer_types[extract_layer_index(prefix)]``.
    * when absent, ``full_attention_interval`` generates it —
      ``transformers_utils/configs/qwen3_5.py``:
      ``"linear_attention" if bool((i + 1) % interval) else "full_attention"``.
    """
    if cfg.get("model_type") in ("kimi_k3", "kimi_linear"):
        return _kimi_attn_types(cfg, n)
    declared = cfg.get("layer_types")
    if isinstance(declared, list):
        # A config may declare more entries than the (possibly shrunk) layer
        # count; index into it exactly as vLLM does.
        if len(declared) < n:
            raise ValueError("layer_types must cover the original decoder depth")
        values = [str(declared[i]) for i in range(n)]
        # Native DSA config classes use a cache-dispatch label. Sparse indexer
        # selection is a separate axis in our catalog, not an attention block
        # name (transformers configuration_glm_moe_dsa.py:157-159 and
        # configuration_deepseek_v32.py:137-139).
        if cfg.get('model_type') in ('deepseek_v32', 'glm_moe_dsa'):
            values = [FULL_ATTENTION if value == 'deepseek_sparse_attention' else value
                      for value in values]
        return values

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

    A list-valued ``moe_layer_freq`` is a per-layer flag: MiniMax-M3's native
    ``minimax_m3/nvidia/model.py:109-114`` indexes the declared layer directly
    and compares it with zero. It does not repeat a shorter schedule.
    """
    experts = _experts(cfg)
    if experts <= 0:
        return [MLP_DENSE] * n

    freq = cfg.get("moe_layer_freq")
    if isinstance(freq, list):
        if len(freq) < n:
            raise ValueError("moe_layer_freq must cover the original decoder depth")
        return [
            MLP_MOE if freq[i] != 0 else MLP_DENSE for i in range(n)
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
        if isinstance(freq, list):
            # Native MiniMax model.py:98-106 enumerates the selected layer
            # IDs. A missing suffix is non-sparse, unlike the indexed MoE list.
            return [i < len(freq) and freq[i] != 0 for i in range(n)]
        if cfg.get("model_type") in ("minimax_m3_vl", "minimax_m3") and freq is None:
            return [False] * n
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


def resolve_stack(cfg: dict[str, Any], *, prefix_layers: int | None = None) -> list[LayerSpec]:
    """One :class:`LayerSpec` per decoder layer, in order.

    Empty when a non-Kimi config declares no layer count. A prefix never
    changes the source configuration or hides malformed later indexed layers.
    """
    if cfg.get("model_type") == "kimi_k3" and "text_config" in cfg:
        inner = cfg["text_config"]
        if not isinstance(inner, dict):
            raise ValueError("Kimi text_config must be an explicit mapping")
        cfg = dict(inner, model_type="kimi_linear")
    else:
        cfg = text_config(cfg)
    n = _num_layers(cfg)
    if cfg.get("model_type") in ("kimi_k3", "kimi_linear"):
        if type(cfg.get("num_hidden_layers")) is not int or n < 1:
            raise ValueError("Kimi num_hidden_layers must be a positive integer")
    if prefix_layers is not None:
        if type(prefix_layers) is not int or not 1 <= prefix_layers <= n:
            raise ValueError("prefix_layers must be within the original depth")
    if n == 0:
        return []
    # Resolve the original declaration before taking the engine prefix. A
    # short valid boot must not hide an invalid source model's later layers.
    attn = _attn_types(cfg, n)
    mlp = _mlp_types(cfg, n)
    sparse = _sparse_flags(cfg, n)
    if prefix_layers is not None:
        n = prefix_layers
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


def probe_key_saturation(hf_cfg: dict) -> int | None:
    """How many key tokens a query can attend to, when the model bounds it.

    A sparse-attention query reads at most a fixed number of selected
    positions, so its cost stops growing once a sequence's key window passes
    that bound -- measured on DeepSeek-V3.2, a 16-token prefill costs 968.8 us
    at no context, 742.4 at 512, and then 139.3 at 2048 and 138.3 at 8192,
    flat from exactly ``index_topk`` onward. A dense query reads its whole
    causal window, so the same sweep rises throughout (Llama-3.1-8B: 8.5,
    18.8, 51.2, 181.1 us) and there is nothing to bound. Returns None for
    that case.

    Each family declares the bound its own way, and both are checkpoint
    fields:

    * DeepSeek-V3.2 / GLM-5 select ``index_topk`` individual tokens.
    * MiniMax-M3 selects ``sparse_topk_blocks`` blocks of
      ``sparse_block_size``, plus the always-included initial and local
      blocks, so the bound is their product.

    Note this is a property of a *kernel*, not of a model: M3's
    ``sparse_attention_freq`` leaves its first three layers non-sparse, so one
    checkpoint has both bounded and unbounded attention. The caller applies
    the value only to the catalog entries whose computation is sparse.
    """
    topk = hf_cfg.get("index_topk")
    if topk:
        return int(topk)
    sparse = hf_cfg.get("sparse_attention_config") or {}
    if isinstance(sparse, dict) and sparse.get("use_sparse_attention"):
        blocks = (int(sparse.get("sparse_topk_blocks", 0) or 0)
                  + int(sparse.get("sparse_init_block", 0) or 0)
                  + int(sparse.get("sparse_local_block", 0) or 0))
        block_size = int(sparse.get("sparse_block_size", 0) or 0)
        if blocks > 0 and block_size > 0:
            return blocks * block_size
    return None
