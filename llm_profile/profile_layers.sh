#!/bin/bash

#####################  Profile non-attention layers  #####################

CUDA_VISIBLE_DEVICES=0 \
python3 -m profiler.layers.main \
  --hardware A6000 \
  --model "meta-llama/Llama-3.1-8B" \
  --num-layers 1 \
  --tp-size "1, 2" \
  --warmup 10 \
  --repeat 30 \
  --max-len 10 \
  --device cuda
