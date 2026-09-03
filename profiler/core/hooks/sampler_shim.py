"""Give vLLM's non-module sampler a module scope so it can be profiled.

vLLM 0.28 sends every **dense** model down the V2 model runner
(``VllmConfig._is_default_v2_model_runner_model``: ``is_default_v2_architecture
or not model_config.is_moe``), and that runner's sampler is
``vllm.v1.worker.gpu.sample.sampler.Sampler`` -- a plain object, where the V1
runner's is an ``nn.Module``. ``layerwise_profile`` builds its tree from module
events, so the V2 sampler never becomes a node at all: the ``per_sequence``
sweep writes no ``sampler`` row and the simulator then rejects the bundle with
``Missing per-sequence profile for layer=sampler``.

The cost it hides is real. Timed with CUDA events on RTXPRO6000 /
Llama-3.1-8B, one call runs 29.5 us at a single sequence and 90.3 us at 256 --
within a few percent of the ``nn.Module`` sampler's own numbers in the
vllm=0.19 bundle (24.7 us and, at 128 sequences where both were measured,
63.69 against 63.77).

Every bundle profiled on 0.28 before this was MoE, which keeps the V1 runner,
which is why the defect surfaced only with the first dense 0.28 refresh.

Wrapping puts the same class name in the tree on both runners, so each
catalog's ``sampler: {vllm: Sampler}`` binds unchanged -- the alternative was
a second spelling in every catalog that can run a dense model.
"""

from typing import Any

from torch import nn

__all__ = ["Sampler", "wrap_sampler_for_profiling"]


class Sampler(nn.Module):
    """An ``nn.Module`` facade over a sampler that is not one.

    Named ``Sampler`` deliberately: a profile node carries its module's class
    name, and that name is what a catalog entry binds. This stands in for
    vLLM's own ``Sampler`` on the runner where that class is a plain object,
    so one canonical name covers both.
    """

    def __init__(self, inner: Any) -> None:
        super().__init__()
        self._inner = inner

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # ``nn.Module.__getattr__`` runs only after normal lookup fails, so
        # this sees exactly the names the wrapped sampler owns. The V2 runner
        # reads several off it directly -- ``sampling_states``,
        # ``penalties_state``, ``add_request``, ``apply_staged_writes`` -- so
        # the facade has to be transparent, not just callable.
        try:
            inner = object.__getattribute__(self, "_inner")
        except AttributeError:  # pragma: no cover - during __init__ only
            raise AttributeError(name) from None
        return getattr(inner, name)


def wrap_sampler_for_profiling(model_runner: Any) -> None:
    """Install the facade when the runner's sampler is not a module.

    Idempotent: after wrapping, the sampler *is* an ``nn.Module``. Installed
    for the engine's whole life rather than per shot, so the warm-up forward
    and the measured ones run the identical call path.
    """
    sampler = getattr(model_runner, "sampler", None)
    if sampler is None or isinstance(sampler, nn.Module):
        return
    model_runner.sampler = Sampler(sampler)
