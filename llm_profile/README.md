# LLM-Profile
LLM-Profile is a profiling tool designed to support GPU execution within [LLMServingSim](https://github.com/casys-kaist/LLMServingSim). 

## Overview
This repository loads LLMs from Hugging Face and inserts PyTorch profiler hooks into key layers to measure their execution time on GPU. 
It is intended for performance analysis and simulation purposes, especially in conjunction with LLMServingSim.

## How to Use
### 1. Environment Setting
Run the appropriate PyTorch Docker container based on your GPU and CUDA version.
See `docker.sh` for an example.

Alternatively, if you already have PyTorch and CUDA installed natively, Docker is not required.

### 2. Set Hugging Face Token
For models that require access approval (e.g., LLaMA), you must provide your Hugging Face token.

- Modify the `run_profile.py` `__main__` section directly
```python
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "<your_token>")
```
- Or, set the token via an environment variable before running the script
```bash
export HF_TOKEN=<your_token>
```

### 3. Run the Profiler
Refer to the `run_profile` function in `run_profile.py` to start profiling.

If you're working with a large model, you can reduce `num_layers` in the configuration to ensure the model fits within a single GPU.

Once the configuration is set, insert it into the `__main__` section and simply run:
```bash
python3 run_profile.py
```
The profiling results will be saved in a CSV file named `<HARDWARE>.csv`.

An example profiling result for RTX 3090 is provided in `RTX3090.csv`.

### 4. Validation
The `validation.py` module provides functionality to automatically compute and apply a scaling factor to adjust the estimated latency.

When profiling a model with a reduced number of layers, GPUs are often underutilized, leading to artificially slower per-layer latency. 
To correct this discrepancy, we compute a scaling factor that reflects the utilization difference. 
As the number of profiled layers approaches the full model depth, the scaling factor converges to 1.


- `compute_average_scaling_factor()` estimates the scaling factor based on the difference between real-model latency and estimated-model latency.

- `apply_scaling_to_latency_csv()` adjusts the latency values in the CSV using the computed scaling factor.

For implementation details, see `validation.py`.
