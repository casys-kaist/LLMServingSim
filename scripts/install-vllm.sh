#!/bin/bash
# Local CUDA 13 vLLM install helper (requires uv with --torch-backend cu130).
#
# The vllm runs inside the vLLM Docker container by default
# (see docker-vllm.sh). This script is for bare-metal setups where you
# want to install everything into a local uv venv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Python 3.12 is supported by vLLM 0.31.0 and its pinned Torch/Triton.
uv venv --python 3.12 .venv

# Extra deps the profiler / bench / generators import directly:
# datasets (workloads.generators), matplotlib (bench.core.plots), pandas
# (profiler.core.skew + fit_alpha). pandas arrives transitively with
# datasets today, but naming it keeps a datasets release that drops the
# dependency from silently breaking the skew sweep.
# Resolve in one transaction, explicitly targeting this interpreter and the
# CUDA 13 PyTorch packages. NCCL matches torch 2.13.0+cu130's declared pin;
# it does not force the runtime to use NCCL-only communication.
VLLM_USE_PRECOMPILED=1 uv pip install --python .venv/bin/python \
  --torch-backend cu130 --verbose --no-build-isolation \
  vllm==0.31.0 datasets matplotlib pandas 'nvidia-nccl-cu13==2.29.7'

# Match the container's namespace fix and inert-by-default observers.
for patch in "$SCRIPT_DIR"/patches/*.py; do
  .venv/bin/python "$patch"
done
