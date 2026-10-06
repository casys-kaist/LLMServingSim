"""Explicit target geometry for isolated native MoE component profiling.

The profiler and simulator share these token domains. Target ranks describe
the simulated deployment, not the process ranks of the single-GPU profiler.
"""
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class MoeTarget:
    tp: int
    dp: int
    ep: int
    rank: int
    sequence_parallel: bool = False

    def __post_init__(self):
        for value in (self.tp, self.dp, self.ep):
            if type(value) is not int or value < 1:
                raise ValueError("Parallel degrees must be positive integers")
        if self.ep != self.tp * self.dp:
            raise ValueError("Only full expert parallelism is covered")
        if type(self.rank) is not int or not 0 <= self.rank < self.ep:
            raise ValueError("Invalid target EP rank")
        if type(self.sequence_parallel) is not bool:
            raise ValueError("Sequence parallelism must be explicit")
        if self.sequence_parallel and (self.tp == 1 or self.dp == 1):
            raise ValueError("This sequence-parallel path requires TP and DP")

    @property
    def dp_rank(self):
        return self.rank // self.tp

    @property
    def tp_rank(self):
        return self.rank % self.tp

    def token_domains(self, padded_dp_tokens):
        """Receive already-padded counts, never infer graph padding here."""
        if len(padded_dp_tokens) != self.dp:
            raise ValueError("One token count is required per DP member")
        if any(type(n) is not int or n < 1 for n in padded_dp_tokens):
            raise ValueError("Active or dummy model calls need positive token counts")
        if self.sequence_parallel:
            chunks = tuple((n + self.tp - 1) // self.tp for n in padded_dp_tokens)
            local = chunks[self.dp_rank]
            gathered = sum(chunks) * self.tp
            dispatch_rows = tuple(n for n in chunks for _ in range(self.tp))
        else:
            local = padded_dp_tokens[self.dp_rank]
            gathered = sum(padded_dp_tokens)
            dispatch_rows = tuple(padded_dp_tokens) if self.dp > 1 else ()
        return dict(gate_rows=local, expert_rows=gathered,
                    finalize_rows=local, dispatch_rows=dispatch_rows)

    def resolve(self, unsharded_config, local_token_budget):
        """Retain global E/k; let vLLM determine local expert ownership."""
        from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
        from vllm.model_executor.layers.fused_moe.expert_map_manager import determine_expert_map

        if type(local_token_budget) is not int or local_token_budget < 1:
            raise ValueError("Token budget must be a positive integer")
        original = unsharded_config.moe_parallel_config
        if any(getattr(original, key) != 1 for key in ("tp_size", "dp_size", "ep_size", "pcp_size")):
            raise ValueError("Resolve from an unsharded live configuration, not a sliced checkpoint")
        if unsharded_config.num_experts < self.ep:
            raise ValueError("Ranks without any physical experts are not covered")
        local_e, mapping, _ = determine_expert_map(self.ep, self.rank,
                                                  unsharded_config.num_experts, "linear")
        parallel = FusedMoEParallelConfig(
            tp_size=1, tp_rank=0, pcp_size=1, pcp_rank=0,
            dp_size=self.dp, dp_rank=self.dp_rank, ep_size=self.ep, ep_rank=self.rank,
            sp_size=self.tp if self.sequence_parallel else 1,
            use_ep=self.ep > 1, all2all_backend="allgather_reducescatter", enable_eplb=False)
        maximum = self.token_domains((local_token_budget,) * self.dp)["expert_rows"]
        cfg = replace(unsharded_config, moe_parallel_config=parallel,
                      num_local_experts=local_e, max_num_tokens=maximum)
        assert cfg.num_experts == unsharded_config.num_experts
        assert cfg.experts_per_token == unsharded_config.experts_per_token
        metadata = dict(target=asdict(self), expert_parallel_config=asdict(parallel),
                        local_token_budget=local_token_budget, maximum_expert_tokens=maximum,
                        global_experts=cfg.num_experts, global_top_k=cfg.experts_per_token,
                        local_experts=local_e, placement="linear",
                        expert_map=None if mapping is None else mapping.tolist(),
                        acquisition_processes=1, acquisition_tp=1, acquisition_dp=1,
                        measurement_execution="single_rank_emulation",
                        collective_execution="none")
        return cfg, metadata
