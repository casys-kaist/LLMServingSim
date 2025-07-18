#!/bin/bash

# launch pytorch docker
docker run --name llm_profile \
  --gpus '"device=0"' \
  -it \
  -v $(pwd):/workspace \
  --shm-size=16g \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel