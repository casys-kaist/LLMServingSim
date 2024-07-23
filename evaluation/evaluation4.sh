#!/bin/bash

# move to home directory
cd ..

# clean the compiled model
COMPILE_DIR="execution_engine/compiled_result"
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} +

# GPT3-30B (64,1) without reuse
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 1 --npu_mem 24 \
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation4/(64,1)-wo-reuse' --gen > 'evaluation/evaluation4/(64,1)-wo-reuse.txt' 2>&1

# GPT3-30B (64,1) with reuse
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 1 --npu_mem 24 \
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation4/(64,1)-w-reuse' --gen > 'evaluation/evaluation4/(64,1)-w-reuse.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} +

# GPT3-30B (16,4) without reuse
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 4 --npu_mem 24 \
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation4/(16,4)-wo-reuse' --gen > 'evaluation/evaluation4/(16,4)-wo-reuse.txt' 2>&1

# GPT3-30B (16,4) with reuse
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 4 --npu_mem 24 \
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation4/(16,4)-w-reuse' --gen > 'evaluation/evaluation4/(16,4)-w-reuse.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} +

# GPT3-30B (8,8) without reuse
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 8 --npu_mem 24 \
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation4/(8,8)-wo-reuse' --gen > 'evaluation/evaluation4/(8,8)-wo-reuse.txt' 2>&1

# GPT3-30B (8,8) with reuse
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 8 --npu_mem 24 \
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation4/(8,8)-w-reuse' --gen > 'evaluation/evaluation4/(8,8)-w-reuse.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} +

# GPT3-30B (4,16) without reuse
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 16 --npu_mem 24 \
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation4/(4,16)-wo-reuse' --gen > 'evaluation/evaluation4/(4,16)-wo-reuse.txt' 2>&1

# GPT3-30B (4,16) with reuse
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 16 --npu_mem 24 \
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation4/(4,16)-w-reuse' --gen > 'evaluation/evaluation4/(4,16)-w-reuse.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} +

# GPT3-30B (1,64) without reuse
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 64 --npu_mem 24 \
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation4/(1,64)-wo-reuse' --gen > 'evaluation/evaluation4/(1,64)-wo-reuse.txt' 2>&1

# GPT3-30B (1,64) with reuse
python3 -u main.py --model_name 'gpt3-30b' --npu_num 64 --npu_group 64 --npu_mem 24 \
    --pim_type 'local' --sub_batch --dataset 'dataset/simulation-time-bs64-seq1024.tsv' \
    --output 'evaluation/evaluation4/(1,64)-w-reuse' --gen > 'evaluation/evaluation4/(1,64)-w-reuse.txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} +