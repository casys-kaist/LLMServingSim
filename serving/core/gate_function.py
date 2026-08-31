import random
from dataclasses import dataclass
from math import comb
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
        n_group=1,
        topk_group=1,
    ):
        self.instance_id = instance_id
        self.E = int(num_local_experts)
        self.k = max(1, min(int(num_experts_per_tok), self.E))
        # Group-limited routing (DeepSeek-V3/V3.2, GLM). A token's k experts
        # are drawn only from ``topk_group`` of ``n_group`` groups, so it
        # reaches fewer EP ranks than an unrestricted gate would. Absent
        # fields mean one group, which is the unrestricted case.
        self.n_group = max(1, int(n_group or 1))
        self.topk_group = max(1, min(int(topk_group or 1), self.n_group))
        if self.E % self.n_group:
            raise ValueError(
                f"n_group={self.n_group} does not divide num_experts={self.E}; "
                f"vLLM's grouped_topk reshapes the scores to "
                f"(tokens, n_group, experts_per_group), which requires it"
            )
        self._hit_cache = {}
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

    def _rank_group_overlap(self, ep_size):
        """``m[r][g]`` -- how many of EP rank r's experts lie in group g.

        Groups are contiguous expert ranges (vLLM reshapes the score vector to
        ``(tokens, n_group, experts_per_group)``, so group g holds experts
        ``[g * E/n_group, (g+1) * E/n_group)``), and ``expert_owner``
        partitions the same range contiguously, so a rank either spans whole
        groups or sits inside one.
        """
        per_group = self.E // self.n_group
        m = [[0] * self.n_group for _ in range(ep_size)]
        for e in range(self.E):
            m[self.expert_owner(e, ep_size, self.E)][e // per_group] += 1
        return m

    def _hit_probs(self, ep_size):
        """Per-rank P(a token sends at least one of its k experts to rank r).

        Exact under a balanced gate, by enumerating how many of rank r's
        experts fall inside the token's selected groups. The subset
        distribution is built with a DP over groups (``n_group`` is 1-8 in
        practice, so this is microseconds) rather than by drawing samples,
        because the simulator must stay deterministic.

        Without replacement, which is what ``torch.topk`` does: a token
        selects k *distinct* experts. The previous closed form
        ``1 - ((ep-1)/ep)**k`` modelled k independent draws, which double
        counts and reads ~1% low even with no grouping at all.
        """
        key = ep_size
        cached = self._hit_cache.get(key)
        if cached is not None:
            return cached

        E, k, ng, tg = self.E, self.k, self.n_group, self.topk_group
        reach = E * tg // ng            # experts the selected groups hold
        m = self._rank_group_overlap(ep_size)
        subsets = comb(ng, tg)
        probs = []
        for r in range(ep_size):
            # dp[(groups_taken, experts_of_r_reached)] -> subset count
            dp = {(0, 0): 1}
            for g in range(ng):
                nxt = {}
                for (c, a), w in dp.items():
                    nxt[(c, a)] = nxt.get((c, a), 0) + w
                    if c < tg:
                        hit = (c + 1, a + m[r][g])
                        nxt[hit] = nxt.get(hit, 0) + w
                dp = nxt
            miss = 0.0
            for (c, a), w in dp.items():
                if c != tg:
                    continue
                avail = reach - a
                if avail >= k:
                    miss += (w / subsets) * (comb(avail, k) / comb(reach, k))
            probs.append(1.0 - miss)

        self._hit_cache[key] = probs
        return probs

    def _token_experts(self, token_idx):
        """One token's k experts, honouring group-limited routing."""
        if self.n_group == 1:
            return self.routing_fn(token_idx, self.E, self.k)
        per_group = self.E // self.n_group
        if self.routing_policy == "RR":
            first = (token_idx * self.topk_group) % self.n_group
            groups = [(first + i) % self.n_group for i in range(self.topk_group)]
        else:
            groups = self.rnd.sample(range(self.n_group), self.topk_group)
        pool = [g * per_group + i for g in groups for i in range(per_group)]
        picked = self.routing_fn(token_idx, len(pool), self.k)
        return [pool[i] for i in picked]

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
            exps = self._token_experts(t)
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
                    selected = self._token_experts(0)
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

        Pigeonhole model for the activated-expert count:
          * total expert-token pairs = ``total_len * k``
          * split evenly across EP ranks
          * ``pairs_per_rank       = total_len * k / ep_size``
          * ``activated_per_rank   = min(pairs_per_rank, experts_per_rank)``
            — each owned expert fires as long as there are enough
            pairs to go around; beyond saturation the count is
            capped at ``E / ep_size``.

        Group-limited routing does not change that: it restricts *which*
        experts a given token may pick, not how many pairs the batch
        produces, and a balanced gate still spreads those pairs over every
        expert. What it does change is how many ranks one token reaches,
        which is ``_hit_probs``.
        """
        k = self.k
        E_rank = max(1, self.E // ep_size)

        pairs_per_rank = (total_len * k) / ep_size
        activated_per_rank = min(int(round(pairs_per_rank)), E_rank)
        activated_counts = [activated_per_rank] * ep_size

        if ep_size <= 1:
            local_tokens = [total_len] * ep_size
        else:
            # Floored at one whenever the batch has tokens at all. A token's k
            # experts are owned by *some* ranks, so at least one rank runs it,
            # but this model gives every rank the same count and cannot say
            # "some" -- and ``round`` drops it to zero on every rank as soon as
            # ``total_len * p < 0.5``. That is a single-token decode step on
            # anything with a low hit probability: DeepSeek-V3.2 at EP=8
            # (0.454, grouped), Mixtral at EP=8 (0.250), any model at EP=16.
            # The MoE block would then be priced at zero tokens on every rank.
            #
            # Rounding up is the right error direction because the ranks run in
            # parallel behind the ALLTOALL barrier, so what the trace needs is
            # the **critical path**, i.e. the max over ranks -- and that is
            # ``latency(1 token)`` whether one rank holds the token or all of
            # them do. Rounding down loses it; rounding up only over-states
            # ranks that are off the critical path anyway. The ALLTOALL size is
            # computed separately from the group-wide padded total, so nothing
            # else reads this as a sum.
            floor = 1 if total_len > 0 else 0
            local_tokens = [
                max(floor, int(round(total_len * p)))
                for p in self._hit_probs(ep_size)
            ]
        return local_tokens, activated_counts
