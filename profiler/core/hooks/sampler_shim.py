"""Give a non-module sampler the same dynamic scope as an nn.Module sampler.

The facade forwards native arguments, outputs and sampler attributes unchanged.
It is shared by warmup and collection for the worker's lifetime, so catalog
bindings keep the common Sampler class without architecture-specific branches.
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
