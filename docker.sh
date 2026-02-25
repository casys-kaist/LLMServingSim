#!/bin/bash

# launch docker
docker run --name servingsim_docker \
  -it \
  -v $PWD:/app/LLMServingSim \
  -w /app/LLMServingSim \
  astrasim/tutorial-micro2024 \
  bash -c "pip3 install pyyaml pyinstrument transformers datasets \
  msgspec scikit-learn xgboost==3.1.2 && exec bash"