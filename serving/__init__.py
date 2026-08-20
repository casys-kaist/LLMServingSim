"""LLMServingSim 2.0 — cycle-level LLM serving simulator.

Python frontend that drives the ASTRA-Sim C++ backend. The top-level
package only exposes the CLI (``python -m serving``); all simulator
internals live under ``serving.core``.

Module map:
    __main__.py                 simulation entry point + main loop
    core/                       simulator internals
        scheduler.py            vLLM-style continuous batching scheduler
        trace_generator.py      builds execution traces from profiled latencies
        memory_model.py         memory tracking, KV cache, tensor sizes
        graph_generator.py      Chakra protobuf graph generation
        controller.py           IPC with ASTRA-Sim subprocess
        router.py               request routing across instances
        gate_function.py        MoE expert token routing
        config_builder.py       cluster config -> ASTRA-Sim input files
        power_model.py          power / energy estimation
        pim_model.py            PIM device model
        request.py              Request / Batch data classes
        block_pool.py           per-tier KV block pool + prefix-cache index
        kv_cache_manager.py     tiered KV cache manager (block hashing, allocation)
        utils.py                model config loading, formatting
        logger.py               rich-based logger + stdio capture
    run.sh                      example invocations across cluster configs
"""
