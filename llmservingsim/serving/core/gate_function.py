import random
from dataclasses import dataclass
from .logger import get_logger


@dataclass
class RoutingResult:
    """Per-EP-rank routing information for a single MoE layer."""
    local_tokens: list     # [rank] -> token count routed to this rank
    activated_experts: list # [rank] -> number of distinct experts activated on this rank
    source_tokens: list    # [rank] -> token count originating from this rank before dispatch


class GateRouter:
    """Simulator-side model of the MoE gate + expert dispatch.

    Policies:
        BALANCED  (default) — closed-form pigeonhole approximation of
                              a trained learned gate with load-balancing
                              auxiliary loss. Deterministic.
        RR                  — deterministic round-robin per token.
        RAND                — uniform random per token (seedable).
        CUSTOM              — user-supplied ``_custom_gate_function``.

    ``block_copy``: simulator-side optimization that emits one
    transformer block's trace and replays it ``num_hidden_layers``
    times instead of re-computing the routing every layer. Cuts
    trace-generation time by roughly ``num_hidden_layers`` × on MoE
    models. Safe whenever every layer's routing produces the same
    (local_tokens, activated_experts) pair — which is true for
    BALANCED (deterministic), and a harmless approximation for
    RR / RAND (per-layer variance in activated-count is small once
    the batch is at saturation). Default True for speed; CUSTOM
    policies that legitimately need per-layer variance can set
    ``block_copy=False`` in the constructor.
    """

    _SUPPORTED_POLICIES = ("BALANCED", "RR", "RAND", "CUSTOM")

    def __init__(
        self,
        node_id,
        instance_id,
        num_local_experts,
        num_experts_per_tok=1,
        routing_policy="BALANCED",
        seed=42,
        block_copy=True,
    ):
        self.instance_id = instance_id
        self.E = int(num_local_experts)
        self.k = max(1, min(int(num_experts_per_tok), self.E))
        self.routing_policy = routing_policy.upper()
        self.seed = seed
        self.rnd = random.Random(seed) if seed is not None else random
        self.block_copy = bool(block_copy)

        if self.routing_policy == "RR":
            self.routing_fn = self._rr_routing
        elif self.routing_policy == "RAND":
            self.routing_fn = self._rand_routing
        elif self.routing_policy == "BALANCED":
            # ``route_ep`` bypasses ``routing_fn`` for this policy —
            # the per-rank (local_tokens, activated_experts) pair is
            # computed analytically. ``routing_fn`` still points at a
            # valid function for the few non-route_ep call sites.
            self.routing_fn = self._rand_routing
        elif self.routing_policy == "CUSTOM":
            self.routing_fn = self._custom_gate_function
        else:
            raise ValueError(
                f"Unknown routing_policy {routing_policy!r}. "
                f"Supported: {', '.join(self._SUPPORTED_POLICIES)}"
            )
        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)

    @staticmethod
    def expert_owner(expert_id, ep_size, num_experts):
        """Determine which EP rank owns a given expert. Even distribution across ranks."""
        return min(int(expert_id * ep_size // num_experts), ep_size - 1)

    def _rr_routing(self, token_idx, E, k):
        base = token_idx % E
        return [(base + o) % E for o in range(k)]

    def _rand_routing(self, token_idx, E, k):
        return self.rnd.sample(range(E), k)

    def _custom_gate_function(self, token_idx, E, k):
        raise NotImplementedError("Implement custom gate function.")

    def route(self, layer_num, batch_id, total_len):
        """Returns flat token counts per expert (used when EP=1)."""
        counts = [0] * self.E
        for t in range(int(total_len)):
            exps = self.routing_fn(t, self.E, self.k)
            for e in exps:
                counts[e] += 1

        self.logger.info(
            "layer=%d policy=%s E=%d k=%d batch=%s tokens=%d assigned=%s",
            layer_num, self.routing_policy, self.E, self.k,
            batch_id, total_len, counts,
        )
        return counts

    def route_ep(self, layer_num, batch_id, total_len, ep_size):
        """EP-aware routing: returns per-rank token counts and activated experts.

        Tokens are distributed evenly across EP ranks before dispatch
        (matching vLLM's EP execution model). Each token selects k
        experts; the owning rank receives the token for local
        execution. Expert-to-rank assignment uses even partitioning:
        ``expert_id * ep // num_experts``.

        BALANCED short-circuits the per-token draw and uses the
        pigeonhole expression — trained MoE gates (Qwen3, Mixtral,
        DeepSeek, …) are load-balance-regularised so per-expert
        traffic is approximately uniform at serving time.
        """
        total_len = int(total_len)
        ep_size = max(1, int(ep_size))

        # Distribute source tokens evenly across EP ranks
        base = total_len // ep_size
        remainder = total_len % ep_size
        source_tokens = [base + (1 if r < remainder else 0) for r in range(ep_size)]

        if self.routing_policy == "BALANCED":
            local_tokens, activated_counts = self._balanced_route_ep(
                total_len, ep_size, source_tokens,
            )
        else:
            local_tokens = [0] * ep_size
            activated_experts = [set() for _ in range(ep_size)]

            for src_rank in range(ep_size):
                for _ in range(source_tokens[src_rank]):
                    selected = self.routing_fn(0, self.E, self.k)
                    dest_ranks = set()
                    for expert_id in selected:
                        owner = self.expert_owner(expert_id, ep_size, self.E)
                        activated_experts[owner].add(expert_id)
                        dest_ranks.add(owner)
                    for owner in dest_ranks:
                        local_tokens[owner] += 1

            activated_counts = [len(s) for s in activated_experts]

        self.logger.info(
            "layer=%d policy=%s E=%d k=%d ep=%d batch=%s tokens=%d local=%s activated=%s",
            layer_num, self.routing_policy, self.E, self.k, ep_size,
            batch_id, total_len, local_tokens, activated_counts,
        )

        return RoutingResult(
            local_tokens=local_tokens,
            activated_experts=activated_counts,
            source_tokens=source_tokens,
        )

    def _balanced_route_ep(self, total_len, ep_size, source_tokens):
        """Closed-form per-rank load for a perfectly-balanced learned gate.

        Pigeonhole model:
          * total expert-token pairs = ``total_len * k``
          * split evenly across EP ranks
          * ``pairs_per_rank       = total_len * k / ep_size``
          * ``activated_per_rank   = min(pairs_per_rank, experts_per_rank)``
            — each owned expert fires as long as there are enough
            pairs to go around; beyond saturation the count is
            capped at ``E / ep_size``.

        Per-rank token count uses the probability a given token hits
        at least one of rank r's experts:

          P(token hits r) = 1 − ((ep − 1) / ep) ** k

        which collapses to ~1 when ``k ≫ ep`` (so every token reaches
        every rank) and degrades gracefully for small-k / large-ep.
        """
        k = self.k
        E_rank = max(1, self.E // ep_size)

        pairs_per_rank = (total_len * k) / ep_size
        activated_per_rank = min(int(round(pairs_per_rank)), E_rank)
        activated_counts = [activated_per_rank] * ep_size

        if ep_size <= 1:
            hit_prob = 1.0
        else:
            hit_prob = 1.0 - ((ep_size - 1) / ep_size) ** k
        tokens_to_r = int(round(total_len * hit_prob))
        local_tokens = [tokens_to_r] * ep_size
        return local_tokens, activated_counts
