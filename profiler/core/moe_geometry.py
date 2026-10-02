"""Hardware-independent EP placement and valid global routing constructions.

These structures describe measured work, not its latency. In particular,
local assignment count is not a replacement for global top-k: each generated
row always contains the model's original number of distinct global experts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertPlacement:
    global_experts: int
    global_top_k: int
    ep: int
    rank: int = 0
    strategy: str = "linear"

    def __post_init__(self):
        if not 1 <= self.global_top_k <= self.global_experts:
            raise ValueError("Expected 1 <= global_top_k <= global_experts")
        if not 1 <= self.ep <= self.global_experts or not 0 <= self.rank < self.ep:
            raise ValueError("Invalid expert-parallel placement")
        if self.strategy not in ("linear", "round_robin"):
            raise ValueError("Unknown expert placement strategy")

    @property
    def local_global_ids(self) -> tuple[int, ...]:
        if self.strategy == "round_robin":
            return tuple(range(self.rank, self.global_experts, self.ep))
        width, remainder = divmod(self.global_experts, self.ep)
        first = self.rank * width + min(self.rank, remainder)
        count = width + int(self.rank < remainder)
        return tuple(range(first, first + count))

    @property
    def expert_map(self) -> tuple[int, ...]:
        ids = {global_id: local_id for local_id, global_id in enumerate(self.local_global_ids)}
        return tuple(ids.get(index, -1) for index in range(self.global_experts))

    def assignment_bounds(self, tokens: int, activated_local: int) -> tuple[int, int]:
        """Exact feasible total local assignments at fixed N and active A.

        A token cannot select an expert twice. Remote capacity gives the lower
        per-token local bound; local active capacity gives the upper bound.
        Activating A distinct local experts additionally requires >=A pairs.
        Zero active local experts is valid when every choice can be remote.
        """
        if tokens < 1:
            raise ValueError("A measured gathered batch must contain tokens")
        local = len(self.local_global_ids)
        if not 0 <= activated_local <= local:
            raise ValueError("Activated expert count outside local placement")
        remote = self.global_experts - local
        lower = max(activated_local, tokens * max(0, self.global_top_k - remote))
        upper = tokens * min(self.global_top_k, activated_local)
        if lower > upper:
            raise ValueError("No valid global top-k routing exists for this point")
        return lower, upper

    def balanced_route(self, tokens: int, activated_local: int,
                       local_assignments: int) -> tuple[tuple[int, ...], ...]:
        """Construct a balanced local histogram with an explicit assignment total.

        This is a profiling stimulus, not a learned routing model. Local
        assignments per token differ by at most one; the resulting histogram
        over A active experts also differs by at most one. Remote IDs fill the
        remaining slots without reducing the global top-k width.
        """
        lower, upper = self.assignment_bounds(tokens, activated_local)
        if not lower <= local_assignments <= upper:
            raise ValueError(f"Local assignments outside feasible [{lower}, {upper}]")
        local_ids = self.local_global_ids[:activated_local]
        local_set = set(self.local_global_ids)
        remote_ids = tuple(e for e in range(self.global_experts) if e not in local_set)
        per_token, extra = divmod(local_assignments, tokens)
        local_offset = remote_offset = 0
        rows = []
        for index in range(tokens):
            local_count = per_token + int(index < extra)
            remote_count = self.global_top_k - local_count
            local = tuple(local_ids[(local_offset + j) % len(local_ids)] for j in range(local_count))
            remote = tuple(remote_ids[(remote_offset + j) % len(remote_ids)] for j in range(remote_count))
            rows.append(local + remote)
            local_offset += local_count
            remote_offset += remote_count
        return tuple(rows)
