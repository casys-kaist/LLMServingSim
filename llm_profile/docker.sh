#!/bin/bash

# launch pytorch docker
docker run --name llm_profile \
  --gpus all \
  -it \
  -e HUGGING_FACE_HUB_TOKEN="<your_token>" \
  -v $(pwd):/workspace \
  --volume ~/.cache/huggingface:/root/.cache/huggingface \
  --shm-size=16g \
  nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -lc 'set -euo pipefail; \
    apt-get update && apt-get install -y git ninja-build cmake && \
    pip install -U pip setuptools wheel packaging transformers==4.57.3 && \
    exec bash'