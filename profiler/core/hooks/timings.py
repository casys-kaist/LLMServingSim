"""Extract canonical-layer timings from vLLM's layerwise-profile tree.

vLLM's ``layerwise_profile()`` context manager records CUDA kernel
events per nn.Module invocation and returns a nested tree where each
node has:
    entry["entry"]["name"]       str like "QKVParallelLinear(...)"
    entry["entry"]["cuda_time_us"]  total CUDA time across invocations
    entry["entry"]["invocations"]   how many times the module was called
    entry["children"]             list of child entries

We walk that tree in DFS order, strip the ``(...)`` argument suffix
from each class name, and try to match every node against the
catalog slice the host passed in. A match produces a ``TimingSample``
(layer_name, microseconds for **one trace node**).

Matching rule: a catalog entry matches a node iff
    node_class is entry.vllm (or one of them, when it is a list)
AND (entry.within is None OR some ancestor_class == entry.within)

The DFS path carries the list of ancestor class names, so the
``within`` check is just a membership test.

Normalization rule: vLLM merges every same-class sibling under one parent
into a *single* node, summing ``cuda_time_us`` and counting calls in
``invocations``. So ``invocations`` is not the number of trace nodes — it is
``parent_invocations x modules_of_this_class_per_parent``. Dividing by it is
only correct when those sibling modules are interchangeable.

They are not always. Qwen3.5/3.8's gated-DeltaNet block holds two
``MergedColumnParallelLinear`` children, ``in_proj_qkvz`` (5120 -> 16384) and
``in_proj_ba`` (5120 -> 96); the profiler reports one node with
``invocations = 2 x layers``, and dividing by that yields the mean of a large
GEMM and a tiny one, which describes neither. There is no discriminator to
recover — one node is all vLLM ever emits — so the catalog models the pair as
one canonical layer and we want their **sum**.

The denominator that gets both cases right is

    parent_invocations  x  occurrences of the layer in one block's sequence

which the host passes in as ``occurrences``. For a layer the sequence emits
twice per block (an input and a post-attention layernorm) this is the
per-call mean, as before; for a fused pair emitted once it is their sum. When
``invocations == parent_invocations x occurrences`` — every homogeneous
model — it is identical to dividing by ``invocations``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class TimingSample:
    """One layer-level CUDA timing extracted from a shot.

    ``microseconds`` is the cost of **one trace node**: the profiled total
    divided by the number of parent invocations times the number of times
    the block sequence emits this layer. Profiling several decoder layers at
    once (which hybrid stacks require) therefore still yields a per-layer
    number, and a canonical layer covering two fused sibling modules yields
    their sum rather than their mean. See the module docstring.
    """

    layer: str
    microseconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _strip_class_name(raw: str) -> str:
    """Turn ``"QKVParallelLinear(in_features=4096, ...)"`` → ``"QKVParallelLinear"``.

    vLLM's profiler stringifies nn.Module instances as ``ClassName(repr)``.
    We only need the class name for matching.
    """
    paren = raw.find("(")
    return raw if paren < 0 else raw[:paren]


def _match_slice(
    node_class: str,
    ancestors: list[str],
    slice_: dict[str, dict[str, Any]],
) -> str | None:
    """Return the canonical layer name that matches, or None.

    ``slice_`` is the host-to-worker serialized form of a ``Catalog``
    group — ``{canonical_name: {"vllm": cls, "within": parent_cls_or_None,
    "tp_stable": ..., "occurrences": int}}``.

    Ambiguity rule: when several catalog entries match the same node
    (same ``vllm`` class, several ``within`` candidates all present in
    the ancestor chain), the one whose ``within`` is **deepest** in the
    ancestor chain wins. That disambiguates cases like Qwen3's two
    RMSNorms — one inside ``Qwen3DecoderLayer`` (input/post layernorm)
    and one inside ``Qwen3Attention`` (qk_norm) — so the inner match
    (``Qwen3Attention``) doesn't get swallowed by the outer catalog
    entry purely because of YAML ordering. Entries without ``within``
    are treated as the lowest-specificity fallback.
    """
    best_name: str | None = None
    best_depth = -2  # within=None → depth -1; any match wins over no match
    for canonical, spec in slice_.items():
        # ``vllm`` may be a list of alternatives, for a family that swaps the
        # class by checkpoint rather than by structure (Llama 3's
        # ``Llama3RotaryEmbedding`` where Llama 1/2 use plain
        # ``RotaryEmbedding``).
        wanted = spec["vllm"]
        if isinstance(wanted, str):
            if wanted != node_class:
                continue
        elif node_class not in wanted:
            continue
        within = spec.get("within")
        if within is None:
            depth = -1
        else:
            # ``within`` may be a list of alternatives, so one catalog can
            # serve a family whose class names differ by shape (Qwen3's
            # ``Qwen3DecoderLayer`` vs ``Qwen3MoeDecoderLayer``). Take the
            # deepest alternative that is actually present; alternatives this
            # checkpoint doesn't have simply don't match.
            candidates = [within] if isinstance(within, str) else list(within)
            depths = [
                ancestors.index(w) for w in candidates if w in ancestors
            ]
            if not depths:
                continue
            depth = max(depths)
        if depth > best_depth:
            best_depth = depth
            best_name = canonical
    return best_name


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def extract_samples(
    tree: list[dict[str, Any]],
    slice_: dict[str, dict[str, Any]],
) -> list[TimingSample]:
    """Walk the profiler tree; emit samples for nodes matching the slice.

    The slice is typically category-scoped (only the layers relevant
    to the category being profiled right now — passing the full
    catalog works too but produces samples the caller will discard).
    """
    samples: list[TimingSample] = []

    def walk(
        nodes: list[dict[str, Any]],
        ancestors: list[str],
        parent_invocations: int,
    ) -> None:
        for node in nodes:
            raw_name = str(node["entry"]["name"])
            cls = _strip_class_name(raw_name)
            invocations = max(1, int(node["entry"]["invocations"]))

            # Try to match this node against the requested slice.
            canonical = _match_slice(cls, ancestors, slice_)
            if canonical is not None:
                cuda_us = float(node["entry"]["cuda_time_us"])
                occurrences = max(
                    1, int(slice_[canonical].get("occurrences") or 1)
                )
                samples.append(
                    TimingSample(
                        layer=canonical,
                        microseconds=cuda_us / (parent_invocations * occurrences),
                    )
                )

            # Always recurse, even after a match. Some catalog entries
            # are defined by parent-class; their actual kernel time is
            # in a leaf that we want to reach independently.
            children = node.get("children") or []
            walk(children, ancestors + [cls], invocations)

    walk(tree, ancestors=[], parent_invocations=1)
    return samples
