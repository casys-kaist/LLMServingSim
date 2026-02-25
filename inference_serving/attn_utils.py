import math
import time
import statistics
from typing import List, Dict, Any

def compute_statistics(values: List[int]) -> Dict[str, float]:
    """Compute basic statistics for a list of integers."""
    return {
        "mean": statistics.mean(values) if len(values) > 0 else 0.0,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "max": max(values) if len(values) > 0 else 0,
        "min": min(values) if len(values) > 0 else 0,
    }

def _num_splits_heuristic(
    batch_nheads_mblocks: int,
    num_sms: int,
    num_n_blocks: int,
    max_splits: int,
) -> int:
    """
    Python port of FlashAttention's num_splits_heuristic in flash_api.cpp.

    Args:
        batch_nheads_mblocks: batch_size * num_heads * num_m_blocks
        num_sms: number of SMs * 2  (FA2 uses num_sm * 2 for 128-thread blocks)
        num_n_blocks: number of blocks along K dimension
        max_splits: upper bound for num_splits (typically 128)

    Returns:
        Chosen num_splits according to the same efficiency heuristic.
    """

    # If we have enough parallel work to almost fill SMs with 1 split, use 1.
    if batch_nheads_mblocks >= 0.8 * num_sms:
        return 1

    max_splits = min(max_splits, num_sms, num_n_blocks)
    if max_splits <= 0:
        return 1

    def ceildiv(a: int, b: int) -> int:
        return (a + b - 1) // b

    def is_split_eligible(num_splits: int) -> bool:
        # Ineligible if ceil(num_n_blocks / s) equals ceil(num_n_blocks / (s - 1))
        # (see C++ comment in num_splits_heuristic).
        if num_splits == 1:
            return True
        return ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1)

    efficiencies: List[float] = []
    max_eff = 0.0

    # First pass: compute efficiency for all candidate num_splits
    for s in range(1, max_splits + 1):
        if not is_split_eligible(s):
            efficiencies.append(0.0)
            continue
        n_waves = float(batch_nheads_mblocks * s) / float(num_sms)
        eff = n_waves / math.ceil(n_waves)
        efficiencies.append(eff)
        if eff > max_eff:
            max_eff = eff

    # Second pass: pick the smallest num_splits that achieves >= 85% of max_eff
    threshold = 0.85 * max_eff
    for s in range(1, max_splits + 1):
        if not is_split_eligible(s):
            continue
        if efficiencies[s - 1] >= threshold:
            return s

    return 1

def fa2_num_splits(
    batch_size: int,
    num_heads: int,
    head_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_sm: int,
    max_splits: int = 128,
) -> tuple[int, dict]:
    """
    Heuristic to select num_splits for FlashAttention2 and expose basic tiling metadata.

    Returns:
        num_splits: chosen split factor along K dimension.
        meta: dict with block sizes and block counts.
    """
    if head_size <= 64:
        block_n = 256
    elif head_size <= 128:
        block_n = 128
    else:
        block_n = 64

    block_m = 64

    # num_n_blocks = ceil(max_seqlen_k / block_n)
    num_n_blocks = (max_seqlen_k + block_n - 1) // block_n
    # num_m_blocks = ceil(max_seqlen_q / block_m)
    num_m_blocks = (max_seqlen_q + block_m - 1) // block_m

    batch_nheads_mblocks = batch_size * num_heads * num_m_blocks

    # FA2 calls:
    #   num_splits_heuristic(batch_size * num_heads * num_m_blocks,
    #                        num_sm * 2,
    #                        num_n_blocks,
    #                        128);
    num_splits = _num_splits_heuristic(
        batch_nheads_mblocks=batch_nheads_mblocks,
        num_sms=num_sm * 2,
        num_n_blocks=num_n_blocks,
        max_splits=max_splits,
    )

    meta = {
        "block_m": block_m,
        "block_n": block_n,
        "num_m_blocks": num_m_blocks,
        "num_n_blocks": num_n_blocks,
        "batch_nheads_mblocks": batch_nheads_mblocks,
    }
    return num_splits, meta

def make_attn_metadata(
    hardware: str,
    num_sm: int,
    model: str,
    head_size: int,
    batch_size: int,
    num_prefill: int,
    num_decode: int,
    Lq_list: List[int],
    Lk_list: List[int],
    tensor_parallel_degree: int,
    num_heads_per_shard: int,
    num_kv_heads_per_shard: int,
    latency_ns: float,
):
    """
    Profile configuration metadata for FlashAttention-like workloads.
    This function only collects metadata/statistics about (Lq, Lk) pairs
    and does not execute any attention kernel internally.
    """

    assert len(Lq_list) == batch_size
    assert len(Lk_list) == batch_size

    # ------------------------------------------------------------
    # Compute per-request QK pair counts (Lq * Lk)
    # ------------------------------------------------------------
    qk_pairs_list = [Lq_list[i] * Lk_list[i] for i in range(batch_size)]
    qk_stats = compute_statistics(qk_pairs_list)

    # ------------------------------------------------------------
    # Compute statistics for Lq and Lk separately
    # ------------------------------------------------------------
    Lq_stats = compute_statistics(Lq_list)
    Lk_stats = compute_statistics(Lk_list)

    sum_Lq = sum(Lq_list)
    sum_Lk = sum(Lk_list)
    sum_qk = sum(qk_pairs_list)

    Lq_max = Lq_stats["max"]
    Lk_max = Lk_stats["max"]

    # ------------------------------------------------------------
    # Determine tiling parameters using FA2 heuristic
    # ------------------------------------------------------------

    num_splits, tiling_meta = fa2_num_splits(
        batch_size=batch_size,
        num_heads=num_heads_per_shard,
        head_size=head_size,
        max_seqlen_q=Lq_stats["max"],
        max_seqlen_k=Lk_stats["max"],
        num_sm=num_sm,
    )

    block_m = tiling_meta["block_m"]
    block_n = tiling_meta["block_n"]
    num_m_blocks = tiling_meta["num_m_blocks"]
    num_n_blocks = tiling_meta["num_n_blocks"]
    batch_nheads_mblocks = tiling_meta["batch_nheads_mblocks"]

    # ------------------------------------------------------------
    # Compute padding ratios (useful for tile efficiency analysis)
    # ------------------------------------------------------------
    physical_q = block_m * num_m_blocks
    physical_k = block_n * num_n_blocks

    wasted_q = max(physical_q - Lq_max, 0)
    wasted_k = max(physical_k - Lk_max, 0)

    q_padding_ratio = (
        float(wasted_q) / float(physical_q) if physical_q > 0 else 0.0
    )
    k_padding_ratio = (
        float(wasted_k) / float(physical_k) if physical_k > 0 else 0.0
    )

    # ------------------------------------------------------------
    # Tile-level stats
    # ------------------------------------------------------------
    total_tiles = (
        batch_size
        * num_heads_per_shard
        * num_m_blocks
        * num_n_blocks
        * num_splits
    )
    tiles_per_sm = float(total_tiles) / float(num_sm) if num_sm > 0 else 0.0
    waves = float(batch_nheads_mblocks * num_splits) / float(num_sm) if num_sm > 0 else 0.0
    wave_eff = waves / math.ceil(waves) if waves > 0.0 else 0.0

    # ------------------------------------------------------------
    # Waste ratio based on tile granularity
    # waste_ratio_qk = padded / actual
    # ------------------------------------------------------------
    logical_qk_pairs = Lq_max * Lk_max
    tile_qk_pairs = physical_q * physical_k
    
    if tile_qk_pairs > 0:
        waste_ratio_qk = float(tile_qk_pairs - logical_qk_pairs) / float(tile_qk_pairs)
    else:
        waste_ratio_qk = 0.0

    # ------------------------------------------------------------
    # Construct final metadata object
    # ------------------------------------------------------------
    meta = {
        "hardware": hardware,
        "model": model,

        # Batch related statistics
        "batch_size": batch_size,
        "num_prefill": num_prefill,
        "num_decode": num_decode,

        # Query/Key Length statistics
        "sum_Lq": sum_Lq,
        "sum_Lk": sum_Lk,
        "sum_qk": sum_qk,

        "mean_Lq": Lq_stats["mean"],
        "std_Lq": Lq_stats["std"],
        "max_Lq": Lq_stats["max"],
        "min_Lq": Lq_stats["min"],

        "mean_Lk": Lk_stats["mean"],
        "std_Lk": Lk_stats["std"],
        "max_Lk": Lk_stats["max"],
        "min_Lk": Lk_stats["min"],

        # Q·K statistics -> Representing FLOPs 
        "mean_qk": qk_stats["mean"],
        "std_qk": qk_stats["std"],
        "max_qk": qk_stats["max"],
        "min_qk": qk_stats["min"],

        # Tensor parallelism statistics
        "tensor_parallel_degree": tensor_parallel_degree,
        "num_heads_per_shard": num_heads_per_shard,
        "num_kv_heads_per_shard": num_kv_heads_per_shard,

        # FA2 specific statistics
        "block_m": block_m,
        "block_n": block_n,
        "num_m_blocks": num_m_blocks,
        "num_n_blocks": num_n_blocks,
        "batch_nheads_mblocks": batch_nheads_mblocks,
        "num_splits": num_splits,
        "total_tiles": total_tiles,
        "tiles_per_sm": tiles_per_sm,
        "waves": waves,
        "wave_eff": wave_eff,
        "q_padding_ratio": q_padding_ratio,
        "k_padding_ratio": k_padding_ratio,
        "waste_ratio_qk": waste_ratio_qk,

        "latency(ns)": latency_ns,
    }

    return meta
