#!/bin/bash
# Local install helper for the vllm.
#
# The vllm runs inside the vLLM Docker container by default
# (see docker-vllm.sh). This script is for bare-metal setups where you
# want to install everything into a local uv venv.

set -euo pipefail

uv venv --python 3.12

# vLLM brings in torch, pydantic, pyyaml, and rich as transitive
# dependencies — no need to list them separately.
VLLM_USE_PRECOMPILED=1 uv pip install vllm==0.28.0 --verbose --no-build-isolation

# Extra deps the profiler / bench / generators import directly:
# datasets (workloads.generators), matplotlib (bench.core.plots), pandas
# (profiler.core.skew + fit_alpha). pandas arrives transitively with
# datasets today, but naming it keeps a datasets release that drops the
# dependency from silently breaking the skew sweep.
uv pip install datasets matplotlib pandas
