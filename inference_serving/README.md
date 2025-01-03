# Inference Serving
Where the techniques for LLM inference are located.

## `request.py`
Class of the request. Includes the information about a request.

## `scheduler.py`
Class of the scheduler. Manages iteration-level scheduling.

**You can add your own scheduler here.**

## `memory_model.py`
Memory model of LLMServingSim. Calculating KV cache sizes and weight sizes are located here.

## `control.py`
Class that controls the flow between ASTRA-Sim and Scheduler.

## `generate_graph.py`
Calls Chakra Graph Converter to convert trace into execution graph.

## `generate_trace.py`
Lookup perf_model to generate model trace. Also uses memory_model for tensor sizes.

## `config_generator.py`
Generates network and memory config json file automatically. You can change it according to your needs.

## `pim.py`
Gets the pim trace and add the pim operator in the trace.

## `utils.py`
Where various helper functions are located. Especially model config and writing format of the trace file.
