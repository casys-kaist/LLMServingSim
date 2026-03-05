#!/bin/bash

# launch docker
docker run --name servingsim_docker \
  -it \
  -v $PWD:/app/LLMServingSim \
  -w /app/LLMServingSim \
  astrasim/tutorial-micro2024 \
  bash -c "pip3 install pyyaml pyinstrument transformers datasets \
  msgspec scikit-learn xgboost==3.1.2 matplotlib==3.5.3 pandas==1.5.3 \
  numpy==1.23.5 && exec bash" 