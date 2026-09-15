# llm_profile v0 (Pytorch profiler)

A PyTorch-based profiling tool for measuring LLM layer latencies, attention latencies, and
GPU/system-level power consumption. The outputs are used by LLMServingSim as performance and
power models.

To profile a new model or hardware target for use with LLMServingSim, follow the steps below.
See also the [Adding a New Model & Hardware](../README.md#adding-a-new-model--hardware) section
in the top-level README.

## Overview

`llm_profile` loads models from Hugging Face and inserts PyTorch profiler hooks into key
layers to measure execution time on GPU. It supports dense and MoE architectures and
produces per-layer latency CSVs and a scikit-learn-based attention latency predictor.
GPU and system-level power consumption are measured via `nvidia-smi` and `ipmitool`,
and the results feed into LLMServingSim's power model.

## Usage

### 1. Environment

Run inside the provided Docker container or a native PyTorch + CUDA environment:

```bash
./docker.sh
```

For models that require access approval (e.g., LLaMA), provide your Hugging Face token
as described in `docker.sh`.

### 2. Profile layers and attention

```bash
./profile_layers.sh    # Measures compute latency for non-attention layers
./profile_attn.sh      # Measures attention latency across batch sizes and sequence lengths
```

To reduce profiling time and memory usage, decrease the number of layers via `--num-layer`
in the respective profiling scripts.

### 3. Profile power (optional)

For power measurement, we provide example scripts under `profiler/power/` that use
`nvidia-smi` to measure GPU power consumption and `ipmitool` to measure system-level power:

```bash
./profiler/power/profile_gpu_power.sh      # GPU power via nvidia-smi
./profiler/power/profile_server_power.sh   # System-level power via ipmitool
```

Power profiling results are used by LLMServingSim's power model when a cluster config with
power settings is provided (e.g., `cluster_config/single_node_power_instance.json`).

### 4. Build the attention predictor

```bash
./build_predictor.sh
```

This trains a scikit-learn model on the profiled attention data to support real-time latency
prediction during simulation (`--enable-attn-prediction`). The inference space covered by
the predictor can be controlled via `--max-batch` and `--max-len`.

## Output structure

Results are written to:

```
perf_models/{hardware}/{model}/tp{tp_size}/
  layers.csv                              # Per-layer compute latency
  attention.csv                           # Attention latency by (batch_size, seq_len)
  predictions/
    attn_decode_predictions.csv           # Predictor output for decode attention
    attn_prefill_predictions.csv          # Predictor output for prefill attention
```

These files are loaded automatically by LLMServingSim at runtime.

## Supported models

Model-specific profiling code is located in `models/`:

- `llama.py` — Llama architecture (Llama-3.1-8B, Llama-3.1-70B)
- `mixtral.py` — Mixtral-8x7B (MoE)
- `phimoe.py` — Phi-mini-MoE-instruct (MoE)

## Adding a new model or hardware

1. Add a model profiling script in `models/` following the existing examples.
2. Set the target hardware name and model identifier in the profiling shell scripts.
3. Run the profiling and predictor build steps above.
4. Create a `cluster_config` entry referencing the new hardware name.
