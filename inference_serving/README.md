# inference_serving

This directory contains the core Python modules for LLM inference simulation.

## Modules

### `request.py`
Defines the `Request` and `Batch` data classes. Tracks per-request state and latency
metrics (TTFT, TPOT, ITL).

### `scheduler.py`
Per-instance scheduler implementing vLLM-style continuous batching. Manages request queuing,
memory-constrained batch formation, KV cache block eviction and swapping to CPU, and prefix
cache lookup. Add custom scheduling policies here.

### `router.py`
Routes incoming requests across instances using configurable policies (Round Robin, Random,
Custom). Handles request transfer in Prefill/Decode disaggregation mode. Add custom routing
policies here.

### `gate_function.py`
Routes tokens to MoE experts according to configurable policies (Round Robin, Random, Fast,
Custom). Add custom expert routing policies here.

### `memory_model.py`
Tracks NPU, CPU, and CXL memory usage. Manages KV cache block allocation and the RadixCache
for prefix caching. Contains `calculate_sizes` and `get_weight` for per-layer tensor size
computation — modify these when adding a new model architecture.

### `radix_tree.py`
Radix tree data structure for token-level prefix matching, used by the prefix cache. Ported
from SGLang.

### `power_model.py`
Estimates power and energy consumption per node, covering NPU, CPU, DRAM, interconnect, NIC,
and storage.

### `controller.py`
Manages the IPC protocol with the ASTRA-Sim subprocess. Writes workload graph paths to
ASTRA-Sim stdin and parses iteration timing from stdout.

### `graph_generator.py`
Invokes the Chakra converter to transform text-format execution traces into protobuf workload
graphs consumed by ASTRA-Sim.

### `trace_generator.py`
Core performance estimator. Reads pre-profiled latency data from
`llm_profile/perf_models/{hardware}/{model}/tp{N}/` and constructs per-iteration execution
traces. Handles tensor parallelism (ALLREDUCE placement), MoE expert routing, PIM attention
offloading, and sub-batch interleaving. Contains `synthesize_trace` — modify this when adding
a new model architecture.

### `config_builder.py`
Parses the user-provided `cluster_config` JSON and generates the ASTRA-Sim input files:
`astra-sim/inputs/network/network.yml`, `astra-sim/inputs/memory/memory_expansion.json`, and
`astra-sim/inputs/system/system.json`.

### `pim_model.py`
Parses PIM device INI configuration files from `pim_config/`. Derives bandwidth, latency, and
power parameters used by the trace generator for PIM-offloaded attention.

### `attn_utils.py`
Computes attention feature vectors used as input to the scikit-learn attention latency predictor.

### `utils.py`
Helper functions for loading model configs, constructing workload paths, and formatting
terminal output.

### `logger.py`
Configures the LLMServingSim logger. Log level is set via `--log-level` in `main.py`.
