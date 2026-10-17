"""vLLM-internal integration layer.

Every module in this subpackage couples to a vLLM *internal* API —
classes and functions that are not part of vLLM's stable public
surface (``SchedulerOutput``, ``MoERunner``/``BaseRouter``,
``model_runner``). If a vLLM upgrade breaks the
profiler, the fix almost always lives here.

Modules:
    extension.py    Extension class registered via worker_extension_cls
    batch.py        synthetic SchedulerOutput builder
    activity_collector.py  native CUDA capture with explicit module scopes
    runner_capture.py      common warmup/capture validation
    cache_pages.py         physical shared-pool request allocation
    timings.py             catalog matching and normalization
    moe_hook.py     native MoE runner discovery
    moe_components.py      eager native component measurement
"""
