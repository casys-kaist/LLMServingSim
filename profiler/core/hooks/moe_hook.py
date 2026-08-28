"""MoE forced-routing hook.

To profile the MoE block cleanly across the (tokens, activated_experts)
grid we need to control which experts receive tokens — relying on the
(dummy-weighted) learned gate would give unpredictable activation
patterns and poor grid coverage.

This module provides:

* ``ExpertRoute.forge``: build the ``(topk_weights, topk_ids)`` tensors
  that would force a given number of experts to be activated over a
  given number of tokens.

* ``force_moe_routing``: a context manager that swaps ``_compute_routing``
  on the live router **instance** for the duration of the block, so that
  ``router.select_experts`` returns our forged tensors instead of whatever
  the learned gate produces. The original bound method is restored on exit.

vLLM v0.28 restructured MoE: the ``FusedMoE`` module this hook used to patch
no longer exists. Models now call ``FusedMoEFactory(...)``, which returns a
``MoERunner`` owning a ``router`` (a ``BaseRouter``) and a ``RoutedExperts``.
The routing template is ``select_experts`` → ``_select_experts`` →
``_compute_routing``, and only the innermost one is ours to replace:
``_select_experts`` still has to run so EPLB mapping and index-dtype
conversion happen exactly as they would in production.

Because the swap is per-instance rather than on the class, there is no need
to guard on ``layer_name`` — other MoE layers keep their own routers
untouched. Every symbol named here is a vLLM internal API; a version bump
may move them.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class ExpertRoute:
    """Precomputed tensors that pin routing to a specific expert set.

    Attributes:
        router: The live router instance whose ``_compute_routing`` will be
            swapped. Held so ``force_moe_routing`` doesn't have to re-find it.
        layer_name: Identifier of the MoE layer this route targets. Carried
            for error messages only.
        weights: Tensor of shape (num_tokens, top_k); each row is a
            uniform ``1/top_k`` distribution over the chosen experts.
        ids: Integer tensor of shape (num_tokens, top_k) naming the
            experts each token is routed to.
    """

    router: object
    layer_name: str
    weights: torch.Tensor
    ids: torch.Tensor

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def forge(
        cls,
        runner,
        num_tokens: int,
        activated_experts: int,
    ) -> "ExpertRoute":
        """Allocate ``weights`` and ``ids`` for a single MoE layer.

        Args:
            runner: The ``MoERunner`` instance we'll patch. Used to read
                ``top_k`` and to find a device.
            num_tokens: Number of tokens being routed this call.
            activated_experts: How many distinct experts should receive
                at least one token. Must satisfy
                ``top_k <= activated_experts <= num_tokens * top_k``.
        """
        router = runner.router
        top_k = int(router.top_k)
        if activated_experts < top_k:
            raise ValueError(
                f"activated_experts ({activated_experts}) must be >= "
                f"top_k ({top_k})"
            )
        if activated_experts > num_tokens * top_k:
            raise ValueError(
                f"activated_experts ({activated_experts}) cannot exceed "
                f"num_tokens*top_k ({num_tokens * top_k})"
            )

        ids_rows = _cycle_expert_ids(num_tokens, top_k, activated_experts)

        device = _runner_device(runner)

        # The concrete index dtype the kernel wants is handed to
        # ``_compute_routing`` as its ``indices_type`` argument, so we forge
        # in int32 here and cast at call time — no need to guess up front.
        ids = torch.tensor(ids_rows, device=device, dtype=torch.int32)
        expected_shape = (num_tokens, top_k)
        if tuple(ids.shape) != expected_shape:
            raise ValueError(
                f"Forged topk_ids shape mismatch: expected {expected_shape}, "
                f"got {tuple(ids.shape)}"
            )

        weights = torch.full(
            (num_tokens, top_k),
            1.0 / top_k,
            device=device,
            dtype=torch.float32,
        )
        return cls(
            router=router,
            layer_name=str(getattr(runner, "layer_name", "<unknown>")),
            weights=weights,
            ids=ids,
        )


def _runner_device(runner) -> torch.device:
    """Best-effort device lookup for a ``MoERunner``.

    Prefers a real expert parameter; falls back to the current CUDA device
    when the runner holds no parameters directly (some quantized paths keep
    weights on a nested method object).
    """
    for param in runner.parameters():
        return param.device
    for buf in runner.buffers():
        return buf.device
    return torch.device(torch.cuda.current_device())


def _cycle_expert_ids(
    num_tokens: int,
    top_k: int,
    activated_experts: int,
) -> list[list[int]]:
    """Assign expert ids deterministically so exactly ``activated_experts``
    distinct ids appear, cycled across the token dimension.

    The specific assignment doesn't matter for latency — only the count
    of distinct activations does. We use the simplest pattern:
    ``id = (token_idx * top_k + offset) % activated_experts``.
    """
    return [
        [
            (token_idx * top_k + offset) % activated_experts
            for offset in range(top_k)
        ]
        for token_idx in range(num_tokens)
    ]


# ---------------------------------------------------------------------------
# Context manager: live router patch
# ---------------------------------------------------------------------------

@contextmanager
def force_moe_routing(route: ExpertRoute | None) -> Iterator[None]:
    """Swap ``_compute_routing`` on ``route``'s router for the block's duration.

    If ``route`` is None the function is a no-op (useful for dense
    profile categories where we still pass through the MoE-aware
    execute path).

    Patching ``_compute_routing`` rather than ``_select_experts`` keeps the
    surrounding template intact: EPLB validation and mapping, the routing
    capture hook, and the index-dtype conversion all still run, so the
    kernel sees exactly the tensor shapes and dtypes production would give
    it — only the *choice* of experts is ours.
    """
    if route is None:
        yield
        return

    router = route.router
    original_compute_routing = router._compute_routing

    def forced_compute_routing(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ):
        # Args are deliberately ignored — the whole point of forced routing
        # is that we return pre-forged values regardless of the learned
        # gate's logits. The one thing we do check is that the batch the
        # kernel actually got matches the batch we forged for; a padded or
        # split call would otherwise fail deep inside the kernel with a
        # much less informative message.
        expected = (hidden_states.shape[0], int(router.top_k))
        if tuple(route.ids.shape) != expected:
            raise ValueError(
                f"Forged topk_ids shape mismatch for {route.layer_name}: "
                f"expected {expected}, got {tuple(route.ids.shape)}"
            )
        ids = route.ids
        if indices_type is not None and ids.dtype != indices_type:
            ids = ids.to(indices_type)
        return route.weights, ids

    # Bind on the instance so sibling MoE layers keep their own routing.
    router._compute_routing = forced_compute_routing
    try:
        yield
    finally:
        # Restore by deleting the instance attribute, which re-exposes the
        # class's bound method. Assigning the captured bound method back
        # would leave a permanent instance attribute behind.
        try:
            del router._compute_routing
        except AttributeError:
            router._compute_routing = original_compute_routing


# ---------------------------------------------------------------------------
# Helpers that run worker-side
# ---------------------------------------------------------------------------

def single_moe_runner(model_runner):
    """Return the model's lone MoE runner.

    We run profiling with ``hf_overrides.num_hidden_layers`` shrunk to the
    smallest stack that still instantiates every distinct block type, so a
    non-hybrid MoE model has exactly one MoE layer. If there are zero or
    more than one, raise so the caller can investigate rather than forge
    the wrong route.
    """
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
        MoERunner,
    )

    model = model_runner.get_model()
    runners = [m for m in model.modules() if isinstance(m, MoERunner)]
    if len(runners) != 1:
        raise RuntimeError(
            f"Expected exactly one MoE runner in the test model, "
            f"got {len(runners)}"
        )
    return runners[0]
