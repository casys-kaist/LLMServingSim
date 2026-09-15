"""vLLM-internal integration layer.

Every module in this subpackage couples to a vLLM *internal* API —
classes and functions that are not part of vLLM's stable public
surface (``SchedulerOutput``, ``FusedMoE.forward_native``,
``model_runner``, ``layerwise_profile``). If a vLLM upgrade breaks the
profiler, the fix almost always lives here.

Modules:
    extension.py    Extension class registered via worker_extension_cls
    batch.py        synthetic SchedulerOutput builder
    timings.py      layerwise_profile tree parser
    moe_hook.py     FusedMoE forced-routing context manager
"""
