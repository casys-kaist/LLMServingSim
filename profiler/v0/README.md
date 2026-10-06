# profiler/v0

**Archived implementation, retained for reference.** This directory is not the
vLLM 0.28 profiling path and its CSV/predictor formats are not current simulator
inputs. Use the [current profiler](../README.md) for new acquisitions. The
commands below describe the legacy workflow, not a supported current setup.

## Overview

`llm_profile` loads models from Hugging Face and inserts PyTorch profiler hooks into key
layers to measure execution time on GPU. It supports dense and MoE architectures and
produces per-layer latency CSVs and a scikit-learn-based attention latency predictor.
GPU and system-level power consumption are measured via `nvidia-smi` and `ipmitool`,
and were inputs to the legacy simulator's power model.

## Usage

### 1. Environment

Run these legacy commands from `profiler/v0/`, using its own Docker setup or
a compatible historical PyTorch + CUDA environment:

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

This trains the legacy scikit-learn attention predictor. The historical
`--enable-attn-prediction` simulator flag is not part of the current CLI.
The inference space covered by
the predictor can be controlled via `--max-batch` and `--max-len`.

## Output schema

Results are written to:

```text
perf_models/{hardware}/{model}/tp{tp_size}/
├── layers.csv                          per-layer compute latency
├── attention.csv                       latency by batch_size and seq_len
└── predictions/
    ├── attn_decode_predictions.csv     legacy decode predictor output
    └── attn_prefill_predictions.csv    legacy prefill predictor output
```

These are legacy outputs. The current simulator expects the per-category
bundle documented in the [current output guide](../../docs/docs/profiler/output-bundle.md),
not this layout or predictor.

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
