import enum

# Modified from https://github.com/microsoft/vidur

class ProfileMethod(enum.Enum):
    CUDA_EVENT = "cuda_event"
    KINETO = "kineto"
    PERF_COUNTER = "perf_counter"
    RECORD_FUNCTION = "record_function"

def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

def validate_tp_size(tp_size: int, num_heads: int) -> None:

    if tp_size <= 0:
        return False

    if not _is_power_of_two(tp_size):
        return False

    if num_heads % tp_size != 0:
        return False