"""Native MoE runner discovery for component acquisition."""


def single_moe_runner(model_runner):
    """Return the model's lone MoE runner.

    We run profiling with ``hf_overrides.num_hidden_layers`` shrunk to the
    smallest stack that still instantiates every distinct block type, so a
    non-hybrid MoE model has exactly one MoE layer. If there are zero or
    more than one, raise before measuring an ambiguous component.
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
