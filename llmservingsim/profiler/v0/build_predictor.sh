#!/bin/bash

#######################  Build attention predictor  ######################

# python -m profiler.predictor.main \
#   --model "meta-llama/Llama-3.1-8B" \
#   --hardware A6000 \
#   --tp-size "1, 2" \
#   --kv-granularity 64 \
#   --chunk-granularity 32 \
#   --max-len 2048 \
#   --max-batch 256 

python -m profiler.predictor.main \
  --model "meta-llama/Llama-3.1-8B" \
  --hardware TPU-v6e-1 \
  --tp-size "1" \
  --kv-granularity 64 \
  --chunk-granularity 32 \
  --max-len 2048 \
  --max-batch 256 
