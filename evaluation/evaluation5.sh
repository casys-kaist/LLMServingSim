#!/bin/bash

# move to home directory
cd ..

# clean the compiled model
COMPILE_DIR="execution_engine/compiled_result"
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} +
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} +
find "$COMPILE_DIR" -name "gpt3-175b*" -exec rm -rf {} +

# GPT3-7B with 8 NPUs
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 8 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt7b-n8' --gen > 'evaluation/evaluation5/gpt7b-n8.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} +

# GPT3-7B with 16 NPUs
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 16 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt7b-n16' --gen > 'evaluation/evaluation5/gpt7b-n16.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} +

# GPT3-7B with 32 NPUs
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 32 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt7b-n32' --gen > 'evaluation/evaluation5/gpt7b-n32.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} +

# GPT3-7B with 64 NPUs
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 64 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt7b-n64' --gen > 'evaluation/evaluation5/gpt7b-n64.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} +

# GPT3-7B with 128 NPUs
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 128 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt7b-n128' --gen > 'evaluation/evaluation5/gpt7b-n128.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} +

# GPT3-7B with 256 NPUs
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 256 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt7b-n256' --gen > 'evaluation/evaluation5/gpt7b-n256.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} +

# GPT3-7B with 512 NPUs
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 512 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt7b-n512' --gen > 'evaluation/evaluation5/gpt7b-n512.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} +

# GPT3-7B with 1024 NPUs
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 1024 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt7b-n1024' --gen > 'evaluation/evaluation5/gpt7b-n1024.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} +

# GPT3-7B with 2048 NPUs
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 2048 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt7b-n2048' --gen > 'evaluation/evaluation5/gpt7b-n2048.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} +

#####################################################################################
# GPT3-30B with 8 NPUs
python3 -u main.py --model_name 'gpt3-30b' --npu_num 8 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt30b-n8' --gen > 'evaluation/evaluation5/gpt30b-n8.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} +

# GPT3-30B with 16 NPUs
python3 -u main.py --model_name 'gpt3-30b' --npu_num 16 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt30b-n16' --gen > 'evaluation/evaluation5/gpt30b-n16.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} +

# GPT3-30B with 32 NPUs
python3 -u main.py --model_name 'gpt3-30b' --npu_num 32 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt30b-n32' --gen > 'evaluation/evaluation5/gpt30b-n32.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} +

# GPT3-30B with 64 NPUs
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt30b-n64' --gen > 'evaluation/evaluation5/gpt30b-n64.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} +

# GPT3-30B with 128 NPUs
python3 -u main.py --model_name 'gpt3-30b' --npu_num 128 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt30b-n128' --gen > 'evaluation/evaluation5/gpt30b-n128.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} +

# GPT3-30B with 256 NPUs
python3 -u main.py --model_name 'gpt3-30b' --npu_num 256 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt30b-n256' --gen > 'evaluation/evaluation5/gpt30b-n256.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} +

# GPT3-30B with 512 NPUs
python3 -u main.py --model_name 'gpt3-30b' --npu_num 512 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt30b-n512' --gen > 'evaluation/evaluation5/gpt30b-n512.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} +

# GPT3-30B with 1024 NPUs
python3 -u main.py --model_name 'gpt3-30b' --npu_num 1024 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt30b-n1024' --gen > 'evaluation/evaluation5/gpt30b-n1024.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} +

# GPT3-30B with 2048 NPUs
python3 -u main.py --model_name 'gpt3-30b' --npu_num 2048 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt30b-n2048' --gen > 'evaluation/evaluation5/gpt30b-n2048.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} +

#####################################################################################
# GPT3-175B with 8 NPUs
python3 -u main.py --model_name 'gpt3-175b' --npu_num 8 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt175b-n8' --gen > 'evaluation/evaluation5/gpt175b-n8.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-175b*" -exec rm -rf {} +

# GPT3-175B with 16 NPUs
python3 -u main.py --model_name 'gpt3-175b' --npu_num 16 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt175b-n16' --gen > 'evaluation/evaluation5/gpt175b-n16.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-175b*" -exec rm -rf {} +

# GPT3-175B with 32 NPUs
python3 -u main.py --model_name 'gpt3-175b' --npu_num 32 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt175b-n32' --gen > 'evaluation/evaluation5/gpt175b-n32.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-175b*" -exec rm -rf {} +

# GPT3-175B with 64 NPUs
python3 -u main.py --model_name 'gpt3-175b' --npu_num 64 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt175b-n64' --gen > 'evaluation/evaluation5/gpt175b-n64.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-175b*" -exec rm -rf {} +

# GPT3-175B with 128 NPUs
python3 -u main.py --model_name 'gpt3-175b' --npu_num 128 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt175b-n128' --gen > 'evaluation/evaluation5/gpt175b-n128.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-175b*" -exec rm -rf {} +

# GPT3-175B with 256 NPUs
python3 -u main.py --model_name 'gpt3-175b' --npu_num 256 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt175b-n256' --gen > 'evaluation/evaluation5/gpt175b-n256.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-175b*" -exec rm -rf {} +

# GPT3-175B with 512 NPUs
python3 -u main.py --model_name 'gpt3-175b' --npu_num 512 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt175b-n512' --gen > 'evaluation/evaluation5/gpt175b-n512.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-175b*" -exec rm -rf {} +

# GPT3-175B with 1024 NPUs
python3 -u main.py --model_name 'gpt3-175b' --npu_num 1024 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt175b-n1024' --gen > 'evaluation/evaluation5/gpt175b-n1024.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-175b*" -exec rm -rf {} +

# GPT3-175B with 2048 NPUs
python3 -u main.py --model_name 'gpt3-175b' --npu_num 2048 --npu_group 1 --npu_mem 100 \
    --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation5/gpt175b-n2048' --gen > 'evaluation/evaluation5/gpt175b-n2048.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-175b*" -exec rm -rf {} +