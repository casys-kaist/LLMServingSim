#!/bin/bash

#######################  Profile attention layers  #######################

CUDA_VISIBLE_DEVICES=0 \
python -m profiler.attention.main \
  --model "meta-llama/Llama-3.1-8B" \
  --hardware A6000 \
  --max-len 2048 \
  --tp-size "1, 2" \
  --warmup 10 \
  --repeat 50 \
  --device cuda