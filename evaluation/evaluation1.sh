#!/bin/bash

# move to home directory
cd ..

# GPT3-7B
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 1 --npu_group 1 --npu_mem 24 \
    --dataset 'dataset/share-gpt-req100-rate10.tsv' --network 'gpu_n1.json' \
    --output 'evaluation/evaluation1/gpt7b' --fast_run > evaluation/evaluation1/gpt7b.txt 2>&1

# GPT3-30B
python3 -u main.py --model_name 'gpt3-30b' --npu_num 4 --npu_group 1 --npu_mem 24 \
    --dataset 'dataset/share-gpt-req100-rate10.tsv' --network 'gpu_n4.json' \
    --output 'evaluation/evaluation1/gpt30b' --fast_run > evaluation/evaluation1/gpt30b.txt 2>&1

# LLaMA-7B
python3 -u main.py --model_name 'llama-7b' --npu_num 1 --npu_group 1 --npu_mem 24 \
    --dataset 'dataset/share-gpt-req100-rate10.tsv' --network 'gpu_n1.json' \
    --output 'evaluation/evaluation1/llama7b' --fast_run > evaluation/evaluation1/llama7b.txt 2>&1

# LLaMA-30B
python3 -u main.py --model_name 'llama-30b' --npu_num 4 --npu_group 1 --npu_mem 24 \
    --dataset 'dataset/share-gpt-req100-rate10.tsv' --network 'gpu_n4.json' \
    --output 'evaluation/evaluation1/llama30b' --fast_run > evaluation/evaluation1/llama30b.txt 2>&1
