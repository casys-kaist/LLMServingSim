import enum
from itertools import product
from math import floor
from typing import Any

import torch

from .attention_input import AttentionInput

# Modified from https://github.com/microsoft/vidur

def get_num_tokens_to_profile(
    max_num_tokens: int,
):
    NUM_TOKENS_SPACE = (
        list([1, 2, 4])
        + list(range(8, 1024, 8))
        + list(range(1024, 2 * 1024 + 1, 16))
        + list(range(2 * 1024, 4 * 1024 + 1, 32))
        + list(range(4 * 1024, 8 * 1024 + 1, 64))
        + list(range(8 * 1024, 16 * 1024 + 1, 128))
        + list(range(16 * 1024, 32 * 1024 + 1, 256))
        + list(range(32 * 1024, 64 * 1024 + 1, 512))
        + list(range(64 * 1024, 128 * 1024 + 1, 1024))
    )
    num_tokens_to_profile = []
    for num_tokens in NUM_TOKENS_SPACE:
        if num_tokens <= max_num_tokens:
            num_tokens_to_profile.append(num_tokens)
        else:
            break
    num_tokens_to_profile.sort(reverse=True)

    return num_tokens_to_profile


def get_attention_batch_sizes_to_profile(min_batch_size: int, max_batch_size: int):
    BATCH_SIZE_SPACE = list(range(1, 128 + 1, 1)) + list(range(128, 1024 + 1, 8))
    return list(
        filter(
            lambda x: (x >= min_batch_size and x <= max_batch_size), BATCH_SIZE_SPACE
        )
    )


def get_attention_prefill_chunk_sizes_to_profile(max_seq_len: int):
    # PREFILL_CHUNK_SIZE_SPACE = [64, 128, 256, 512, 768, 1024, 1536, 2048, 3076, 4096, 8192, 16384]
    # PREFILL_CHUNK_SIZE_SPACE = range(128, 128 * 1024, 128)
    PREFILL_CHUNK_SIZE_SPACE = (
        list(range(32, 128 + 1, 32))
        + list(range(128, 1024 + 1, 32))
        + list(range(1024, 4 * 1024 + 1, 64))
        + list(range(4 * 1024, 16 * 1024 + 1, 128))
        + list(range(16 * 1024, 64 * 1024 + 1, 256))
    )
    prefill_chunk_sizes_to_profile = []
    for prefill_chunk_size in PREFILL_CHUNK_SIZE_SPACE:
        if prefill_chunk_size <= max_seq_len:
            prefill_chunk_sizes_to_profile.append(prefill_chunk_size)
        else:
            break
    return prefill_chunk_sizes_to_profile


def get_seq_lengths_to_profile(max_seq_len: int):
    SEQ_LENGTH_SIZE_SPACE = (
        list(range(0, 1024 + 1, 32))
        + list(range(1024, 4 * 1024 + 1, 64))
        + list(range(4 * 1024, 64 * 1024 + 1, 256))
    )
    seq_lengths_to_profile = []
    for seq_length in SEQ_LENGTH_SIZE_SPACE:
        if seq_length < max_seq_len:
            seq_lengths_to_profile.append(seq_length)
        else:
            break
    return seq_lengths_to_profile


def get_attention_input_combinations(
    max_seq_len: int,
    max_model_len: int,
    min_batch_size: int,
    max_batch_size: int,
    profile_only_prefill: bool,
    profile_only_decode: bool,
):
    input_combinations = []
    # Chunked Prefills
    prefill_chunk_sizes_to_profile = get_attention_prefill_chunk_sizes_to_profile(
        max_seq_len
    )
    for prefill_chunk_size in prefill_chunk_sizes_to_profile:
        num_partitions = max_seq_len // prefill_chunk_size
        kv_cache_sizes_to_profile = [
            partition_index * prefill_chunk_size
            for partition_index in range(num_partitions)
        ]
        input_combinations.extend(
            product([prefill_chunk_size], kv_cache_sizes_to_profile, [1], [True])
        )
    # Full prefills
    prefill_lengths_to_profile = get_seq_lengths_to_profile(max_seq_len)
    input_combinations.extend(product(prefill_lengths_to_profile, [0], [1], [True]))
    # Decodes
    kv_cache_sizes_to_profile = get_seq_lengths_to_profile(max_model_len)
    batch_sizes_to_profile = get_attention_batch_sizes_to_profile(
        min_batch_size, max_batch_size
    )
    input_combinations.extend(
        product([0], kv_cache_sizes_to_profile, batch_sizes_to_profile, [False])
    )

    valid_input_combinations = []
    for input_combination in input_combinations:
        prefill_chunk_size, kv_cache_size, batch_size, is_prefill = input_combination

        if is_prefill and profile_only_decode:
            continue

        if not is_prefill and profile_only_prefill:
            continue

        attention_input = AttentionInput(
            prefill_chunk_size,
            kv_cache_size,
            batch_size,
            is_prefill,
        )

        if attention_input.is_valid(max_seq_len, max_model_len):
            valid_input_combinations.append(attention_input)
    return valid_input_combinations


"""
    For a given model and parallel config, get the maximum number of blocks that can be allocated.
    This doesn't take into account the weights and activations.
"""


def get_max_num_blocks(
    model_config: Any,
    tensor_parallel_size: int,
    block_size: int,
    dtype: torch.dtype,
    memory_utilization: float = 0.9,
    max_pipeline_parallel_size: int = 1,
):
    element_size = torch.randn(1, dtype=dtype).element_size()
    block_memory_size = (
        2
        * block_size
        * (getattr(model_config, "num_key_value_heads", getattr(model_config, "num_attention_heads", 32)) // tensor_parallel_size)
        * (getattr(model_config, "hidden_size", 4096) // getattr(model_config, "num_attention_heads", 32))
        * element_size
    )
    assert getattr(model_config, "num_hidden_layers", 32) % max_pipeline_parallel_size == 0
    block_memory_total = block_memory_size * (
        getattr(model_config, "num_hidden_layers", 32) // max_pipeline_parallel_size
    )
    return floor(
        (torch.cuda.mem_get_info()[1] * memory_utilization) / (block_memory_total)
    )
