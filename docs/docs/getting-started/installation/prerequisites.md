---
sidebar_position: 1
title: Prerequisites
---

# Prerequisites

LLMServingSim runs on Linux with Docker. The simulator side runs on
CPU, including comparison against recorded benchmarks. GPU acquisition and
new vLLM benchmark recordings need an NVIDIA GPU.

## System

| | Required for Simulator | Required for Profiler / Bench |
| --- | --- | --- |
| **OS** | Linux (Ubuntu 22.04+ tested) | Linux (Ubuntu 22.04+ tested) |
| **Docker** | ✓ | ✓ (or bare-metal install) |
| **NVIDIA GPU** |  | ✓ |
| **NVIDIA Container Toolkit** |  | ✓ (for GPU passthrough into Docker) |
| **CUDA driver** |  | 13.x for the default `vllm/vllm-openai:v0.28.0` image; 12.9 hosts need the `v0.28.0-cu129` tag instead |
| **Disk** | Space for the container, backend build and run outputs | Space for the vLLM image, model-dependent weight/compile caches and profile outputs; full benchmark weights can occupy tens or hundreds of GB |
| **RAM** | 16 GB | 32 GB+ recommended |

If you only plan to run pre-profiled simulations (e.g., the bundled
RTXPRO6000 profiles), you do **not** need a GPU.

## Install Docker

If you don't already have Docker:

```bash
# Ubuntu, official quick-install script
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
```

Verify:

```bash
docker run --rm hello-world
```

## Install NVIDIA Container Toolkit

Required only for GPU containers (profiler / bench). On Ubuntu:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU listed. If not, see
[Troubleshooting → GPU not detected](../troubleshooting#gpu-not-detected).

## Hugging Face token (optional)

Some model configs (e.g., Llama 3.x, gated Qwen variants) live behind
HF authentication. The profiler can auto-fetch these if you set:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxx"
```

Running pre-profiled simulations never needs a token. Hub access needs
authentication for gated/private resources; public resources may be fetched
without a token. Check access for:

- **Profiling** a gated model, where the profiler auto-fetches its
  `config.json` on first run.
- **`bench run`**, which loads real weights — a much larger download
  than a config file.
- **`workloads.generators`**, which pulls the source dataset (and, with
  `--use-vllm`, the model too).

`scripts/docker-vllm.sh` forwards `HF_TOKEN` from your shell into the
container and mounts `~/.cache/huggingface`, so a download is shared
with the host and happens once.

Get a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Next

You're ready to install. Continue with **[Simulator setup](./simulator)**
- this is the main install path that everyone needs.
