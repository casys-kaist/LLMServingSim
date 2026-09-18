"""Tensor/collective contracts, independent of latency measurements."""


def dtype_bytes(dtype, default=2):
    if dtype is None or str(dtype) == "auto":
        return default
    name = str(dtype).removeprefix("torch.")
    sizes = {"bfloat16": 2, "bf16": 2, "float16": 2, "half": 2,
             "float32": 4, "float": 4, "fp32": 4, "int32": 4, "int64": 8}
    if name not in sizes:
        raise ValueError(f"Unsupported communication dtype: {dtype}")
    return sizes[name]


def vocab_shard_size(config, tp, padding=64):
    """vLLM pads vocabulary before partitioning it, not after."""
    padded = ((int(config["vocab_size"]) + padding - 1) // padding) * padding
    if tp < 1 or padded % tp:
        raise ValueError(f"Padded vocabulary {padded} is not divisible by TP={tp}")
    return padded // tp
