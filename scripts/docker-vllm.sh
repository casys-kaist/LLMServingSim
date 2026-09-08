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
# ~/.cache/vllm is mounted alongside the HF cache so torch.compile artifacts
# survive a container recreate. Without it every fresh container pays a cold
# compile -- minutes of CPU with the GPU allocated and idle, which reads as a
# hang.
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
# scripts/patches/ is applied after the installs -- every .py in the directory,
# unconditionally. Each is idempotent and a no-op on a vLLM that already carries
# the fix. Two are corrections the container cannot run without:
#
#   vllm_sm120_sparse_mla.py     backport of vLLM PR #51395; without it
#                                DeepSeek-V3.2 / GLM-5 crash mid-sweep on any
#                                Blackwell card
#   vllm_m3_mtp_layer_name.py    separates MiniMax-M3's MTP layer-name prefix;
#                                without it the model cannot start with
#                                speculative decoding at all
#
# The other two are **instruments**, inert unless their environment variable is
# set, and nothing in profiler/ or bench/ sets it -- they are run by hand when
# a cost-model residual needs localising:
#
#   vllm_step_shape_log.py       VLLM_STEP_SHAPE_LOG -- per-step shape and
#                                interval. Works under cudagraphs.
#   vllm_moe_activated_log.py    VLLM_MOE_ACTIVATED_LOG -- the distinct-expert
#                                count a real gate fires. Needs
#                                --enforce-eager: it calls .unique() inside
#                                the router, which graph capture refuses, and
#                                under replay the Python never runs anyway.

set -euo pipefail

# Resolve the repo root regardless of where this script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../scripts
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                    # .../LLMServingSim

# Extra host paths to mount, for booting a checkpoint that lives outside the
# repo (real weights, so an MoE model's router is the trained one rather than
# a random projection). Space-separated docker -v specs; mount a local model
# store at the same path on both sides so symlinks inside it still resolve:
#     VLLM_EXTRA_MOUNTS="/data/model:/data/model /srv/models:/srv/models" \
#         ./scripts/docker-vllm.sh
# Then pass the directory to bench as --model, which vLLM takes verbatim.
VLLM_EXTRA_MOUNT_ARGS=()
for spec in ${VLLM_EXTRA_MOUNTS:-}; do
  VLLM_EXTRA_MOUNT_ARGS+=(-v "$spec")
done

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
  --volume "$HOME/.cache/vllm":/root/.cache/vllm \
  "${VLLM_EXTRA_MOUNT_ARGS[@]+"${VLLM_EXTRA_MOUNT_ARGS[@]}"}" \
  --shm-size=16g \
  -w /workspace \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.28.0 \
  -c "pip install datasets matplotlib pandas 'nvidia-nccl-cu13==2.29.7' \
      && for p in scripts/patches/*.py; do python3 \"\$p\"; done \
      && exec bash"
