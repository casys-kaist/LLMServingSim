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
    node_class is entry.vllm (or one of them, when it is a list; a name
    ending in ``*`` matches by prefix)
AND (entry.within is None OR some ancestor_class is in entry.within)
AND no ancestor_class is in entry.not_within

The DFS path carries the list of ancestor class names, so the
``within`` check is just a membership test.

Several nodes may match one canonical name, and then the samples are
**summed**: a canonical layer is one trace node, so every profile node bound
to it is one of that node's parts. MiniMax-M3's sparse attention is written
as three Triton kernels launched straight from the module (a prefill kernel,
a decode kernel, and a merge), with no ``Attention`` wrapper to aggregate
them, so the block's cost only exists as their sum. This is a no-op for a
catalog whose entries each match one node, which is every homogeneous model.

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
    the block sequence emits this layer, summed over every profile node the
    catalog binds to this name. Profiling several decoder layers at once
    (which hybrid stacks require) therefore still yields a per-layer number;
    a canonical layer covering two fused sibling modules yields their sum
    rather than their mean; and one written as several kernels yields the sum
    of the kernels. See the module docstring.
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


def _name_matches(wanted: str, node_class: str) -> bool:
    """Whether one catalog ``vllm`` name matches a node's reported name.

    A trailing ``*`` matches by prefix, which is the only workable binding for
    a fused CUDA kernel. Those are reported with their template arguments
    inlined -- MiniMax-M3's q-norm/rope/KV-insert kernel comes back as
    ``void vllm::minimax_m3_fused_ops::fusedMiniMaxM3QNormRopeKVInsertKernel<
    c10::BFloat16, __nv_bfloat16, (...)`` -- so the exact string carries the
    dtypes, and the dtype is a variant axis we deliberately re-profile. An
    exact binding would silently stop matching on the fp8 run.

    Class and Triton-kernel names never contain ``*``, so the sigil is
    unambiguous.
    """
    if wanted.endswith("*"):
        return node_class.startswith(wanted[:-1])
    return wanted == node_class


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
        # ``RotaryEmbedding``), or that spreads one canonical layer over
        # several kernels (MiniMax-M3's sparse attention). A trailing ``*``
        # matches by prefix.
        wanted = spec["vllm"]
        names = [wanted] if isinstance(wanted, str) else list(wanted)
        if not any(_name_matches(w, node_class) for w in names):
            continue
        # An exclusion beats any match: one class can play two roles that
        # ``within`` cannot separate, because it is the immediate parent in
        # both (DeepSeek's shared expert is a DeepseekV2MLP, exactly like a
        # dense layer's own mlp).
        excluded = spec.get("not_within")
        if excluded is not None:
            bad = [excluded] if isinstance(excluded, str) else list(excluded)
            if any(w in ancestors for w in bad):
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
    iterations: int = 1,
) -> list[TimingSample]:
    """Walk the profiler tree; emit samples for nodes matching the slice.

    The slice is typically category-scoped (only the layers relevant
    to the category being profiled right now — passing the full
    catalog works too but produces samples the caller will discard).

    One sample per matched canonical name, in first-match order. When several
    profile nodes bind to the same name their normalized costs are **summed**,
    because a canonical name is one trace node and each matched profile node
    is one of its parts — see the module docstring.

    ``iterations`` is how many timed forwards ran inside the one
    ``layerwise_profile`` context, and it is the **top level's** invocation
    count: every node divides by its parent's invocations to get a per-call
    figure, and the top level has no parent node to read that from. Passing 1
    where 3 forwards ran inflates every top-level binding 3x -- which is
    exactly what happened to ``embedding``, ``lm_head``, ``sampler`` and
    Qwen3.5's whole drafter under vLLM 0.28, where a top-level node is
    reported per forward rather than merged into one wrapper.
    """
    totals: dict[str, float] = {}

    def walk(
        nodes: list[dict[str, Any]],
        ancestors: list[str],
        parent_invocations: int,
    ) -> None:
        # vLLM merges same-class siblings into one node whose ``cuda_time_us``
        # is their sum and whose ``invocations`` counts them all. Under a
        # parent that is itself a node it then reports that merged node
        # **once**; at the top level, where the owning module launched no
        # kernel of its own and so was flattened away, it reports the same
        # merged node once **per sibling module**. Qwen3.5's drafter holds
        # three GemmaRMSNorms under a wrapper that launches nothing, and they
        # arrive as three identical top-level entries -- summing them charged
        # the drafter's norms 3x. Two entries identical in class, time and
        # invocation count cannot be distinct work, so the repeats are dropped.
        seen: set[tuple[str, float, int]] = set()
        for node in nodes:
            raw_name = str(node["entry"]["name"])
            cls = _strip_class_name(raw_name)
            invocations = max(1, int(node["entry"]["invocations"]))
            fingerprint = (cls, float(node["entry"]["cuda_time_us"]),
                           invocations)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)

            # Try to match this node against the requested slice.
            canonical = _match_slice(cls, ancestors, slice_)
            if canonical is not None:
                cuda_us = float(node["entry"]["cuda_time_us"])
                occurrences = max(
                    1, int(slice_[canonical].get("occurrences") or 1)
                )
                totals[canonical] = totals.get(canonical, 0.0) + (
                    cuda_us / (parent_invocations * occurrences)
                )

            # Always recurse, even after a match. Some catalog entries
            # are defined by parent-class; their actual kernel time is
            # in a leaf that we want to reach independently.
            children = node.get("children") or []
            walk(children, ancestors + [cls], invocations)

    walk(tree, ancestors=[], parent_invocations=max(1, int(iterations)))
    return [
        TimingSample(layer=name, microseconds=us) for name, us in totals.items()
    ]

# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

@dataclass
class CoverageReport:
    """How much of a shot's CUDA time the catalog actually binds.

    The check exists because a catalog entry can look right and measure
    nothing. vLLM's profile tree only holds modules that launch a kernel of
    their own, so an entry bound to a class whose work has been fused
    elsewhere silently yields no node -- and the module tree, which is what
    one naturally reads a catalog off, still shows the class. Every one of
    MiniMax-M3's q-norm, Qwen3.5's q-norm/rope/gate and DeepSeek's indexer
    rope was bound plausibly and measured zero; nothing but this told us.

    Accounting: a bound node's ``cuda_time_us`` already includes its whole
    subtree, so ``bound`` sums only bound nodes with no bound ancestor, and
    ``total`` is the root-level sum. ``gaps`` locates the difference at the
    shallowest node whose subtree binds nothing at all, which is the level a
    catalog author can act on.
    """

    total_us: float
    bound_us_by_layer: dict[str, float]
    gaps: list[tuple[str, float, str]]
    """``(node_class, cuda_us, ancestor_chain)``, largest first."""
    over_matches: list[tuple[str, list[str]]]
    """``(canonical_name, ancestor_chains)`` for entries claiming nodes in
    structurally unrelated places -- the failure ``gaps`` is blind to."""

    @property
    def bound_us(self) -> float:
        return sum(self.bound_us_by_layer.values())

    @property
    def fraction(self) -> float:
        return self.bound_us / self.total_us if self.total_us else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_us": self.total_us,
            "bound_us_by_layer": self.bound_us_by_layer,
            "gaps": [list(g) for g in self.gaps],
            "over_matches": [[n, list(c)] for n, c in self.over_matches],
        }

    @classmethod
    def hydrate(cls, raw: dict[str, Any]) -> "CoverageReport":
        return cls(
            total_us=float(raw["total_us"]),
            bound_us_by_layer=dict(raw["bound_us_by_layer"]),
            gaps=[(str(a), float(b), str(c)) for a, b, c in raw["gaps"]],
            over_matches=[(str(n), [str(x) for x in c])
                          for n, c in raw.get("over_matches") or []],
        )


def attribute_tree(
    tree: list[dict[str, Any]],
    slice_: dict[str, dict[str, Any]],
) -> CoverageReport:
    """Account for every microsecond in the profile tree against ``slice_``.

    Uses the same :func:`_match_slice` the profiler itself matches with, so a
    clean report means the catalog binds what a real run will bind -- not
    something a second implementation of the rules happens to agree on.
    """
    bound: dict[str, float] = {}
    gaps: list[tuple[str, float, str]] = []
    # Where each entry's matches sat, so an entry claiming two unrelated places
    # can be reported. ``gaps`` cannot see this: over-matching leaves nothing
    # unbound, so coverage reads 100% while a number is silently the sum of two
    # roles. Qwen3.5's ``mtp_norms`` measured the target's norms as well as the
    # drafter's and read 1287 us for two RMSNorms; its ``embedding`` summed the
    # target's table with the drafter's own and read 2.1x. Both passed coverage.
    seen_at: dict[str, set[tuple[str, ...]]] = {}

    def binds_anything(node: dict[str, Any], ancestors: list[str]) -> bool:
        cls = _strip_class_name(str(node["entry"]["name"]))
        if _match_slice(cls, ancestors, slice_) is not None:
            return True
        return any(
            binds_anything(child, ancestors + [cls])
            for child in (node.get("children") or [])
        )

    def walk(nodes: list[dict[str, Any]], ancestors: list[str]) -> None:
        for node in nodes:
            entry = node["entry"]
            cls = _strip_class_name(str(entry["name"]))
            cuda_us = float(entry["cuda_time_us"])
            canonical = _match_slice(cls, ancestors, slice_)
            if canonical is not None:
                # This node's cost covers its subtree; do not descend, or the
                # children would be counted twice.
                bound[canonical] = bound.get(canonical, 0.0) + cuda_us
                seen_at.setdefault(canonical, set()).add(tuple(ancestors))
                continue
            if not binds_anything(node, ancestors):
                gaps.append((cls, cuda_us, " > ".join(ancestors[-3:])))
                continue
            walk(node.get("children") or [], ancestors + [cls])

    walk(tree, ancestors=[])

    # An entry may legitimately match several nodes -- MiniMax-M3's sparse
    # attention is three Triton kernels, and a ``*``-prefixed glue entry is
    # many at::native functors -- but those all sit under one parent, so their
    # ancestor chains share a prefix. Matches with **no** common prefix are in
    # structurally unrelated places, which is what an entry doing two jobs
    # looks like.
    over: list[tuple[str, list[str]]] = []
    for canonical, chains in sorted(seen_at.items()):
        if len(chains) < 2:
            continue
        first = min(chains, key=len)
        shared = 0
        for i, name in enumerate(first):
            if all(len(c) > i and c[i] == name for c in chains):
                shared = i + 1
            else:
                break
        if shared == 0:
            over.append((canonical,
                         sorted(" > ".join(c) or "(top level)" for c in chains)))

    total = sum(float(n["entry"]["cuda_time_us"]) for n in tree)
    return CoverageReport(
        total_us=total,
        bound_us_by_layer=bound,
        gaps=sorted(gaps, key=lambda g: -g[1]),
        over_matches=over,
    )
