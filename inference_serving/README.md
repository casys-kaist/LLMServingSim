# Inference Serving
Where the techniques for LLM inference are located.

## `model_reference`
Reference model traces to figure out the weight of the model.

## `request.py`
Class of the request. Includes the information about a request.

## `scheduler.py`
Class of the scheduler. Manages iteration-level scheduling using `kv_manage.py`.

**You can add your own scheduler here.**

## `kv_manage.py`
Where management of KV cache happens.

## `control.py`
Controls the flow between ASTRA-Sim and Scheduler.

## `generate_graph.py`
Calls Chakra Graph Converter to convert trace into execution graph.

## `generate_text.py`
Calls Execution Engine to make model trace.

## `pim.py`
Gets the pim trace and add the pim operator in the trace.

## `utils.py`
Where various helper functions are located.
