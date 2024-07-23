#!/bin/bash

# move to home directory
cd ..

# clean the compiled model
COMPILE_DIR="execution_engine/compiled_result"
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} +
find "$COMPILE_DIR" -name "gpt3-13b*" -exec rm -rf {} +
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} +

# GPT3-7B
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 1 --npu_group 1 --npu_mem 24 \
    --dataset 'dataset/simulation-time-bs32-seq512.tsv' \
    --output 'evaluation/evaluation3/gpt7b' > 'evaluation/evaluation3/gpt7b.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} +

# GPT3-13B
python3 -u main.py --model_name 'gpt3-13b' --npu_num 1 --npu_group 1 --npu_mem 24 \
    --dataset 'dataset/simulation-time-bs32-seq512.tsv' \
    --output 'evaluation/evaluation3/gpt13b' > 'evaluation/evaluation3/gpt13b.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-13b*" -exec rm -rf {} +

# GPT3-30B
python3 -u main.py --model_name 'gpt3-30b' --npu_num 1 --npu_group 1 --npu_mem 40 \
    --dataset 'dataset/simulation-time-bs32-seq512.tsv' \
    --output 'evaluation/evaluation3/gpt30b' > 'evaluation/evaluation3/gpt30b.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} +