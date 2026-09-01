#!/bin/bash

# Launch vLLM Docker for profiler / bench / validate.
#
# Mounts the LLMServingSim repo root as /workspace so the profiler,
# bench, datasets generators, and shared model configs are all visible:
#
#     /workspace/profiler/            profiler package + scripts
#     /workspace/bench/               bench + validate
#     /workspace/workloads/            workload JSONLs and generators
#     /workspace/configs/model/       HF model configs
#
# The working directory defaults to /workspace so any of the modules
# can be run via ``python -m profiler``, ``python -m bench``, etc.
#
# The official vllm/vllm-openai image already provides vllm, pydantic,
# pyyaml, rich, and huggingface_hub. Three extras are installed on start:
# pandas (the profiler's alpha fit and the simulator's tables), matplotlib
# (bench plots) and datasets (the workload generators).
#
# `nvidia-nccl-cu13` is pinned because installing `datasets` pulls a newer one
# than the image's torch declares (`torch 2.13.0+cu130 requires
# nvidia-nccl-cu13==2.29.7`), and pip reports the conflict but installs it
# anyway.
#
# scripts/patches/ is applied after the installs. Right now that is one line
# backporting vLLM PR #51395, without which DeepSeek-V3.2 / GLM-5 crash
# mid-sweep on any Blackwell card. Each patch is idempotent and a no-op on a
# vLLM that already carries the fix.

set -euo pipefail

# Resolve the repo root regardless of where this script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../scripts
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                    # .../LLMServingSim

# Which GPUs to expose. Defaults to every GPU on the host; set VLLM_GPUS to a
# docker device spec to narrow it on a shared machine. Note the inner double
# quotes — they are required, and are part of the value:
#     VLLM_GPUS='"device=2,3"' ./scripts/docker-vllm.sh
# Without them docker splits the value on the comma and reads the second field
# as a GPU *count*, failing with "cannot set both Count and DeviceIDs".
docker run --name vllm_docker \
  --gpus "${VLLM_GPUS:-all}" \
  -it \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$REPO_ROOT":/workspace \
  --volume "$HOME/.cache/huggingface":/root/.cache/huggingface \
  --shm-size=16g \
  -w /workspace \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.28.0 \
  -c "pip install datasets matplotlib pandas 'nvidia-nccl-cu13==2.29.7' \
      && for p in scripts/patches/*.py; do python3 \"\$p\"; done \
      && exec bash"
