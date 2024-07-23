#!/bin/bash

# move to home directory
cd ..

# clean the compiled model
COMPILE_DIR="execution_engine/compiled_result"

# GPT3-7B (4, 1)
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 32 --npu_group 1 --npu_mem 4 \
    --pim_type 'local' --sub_batch --dataset 'dataset/alpaca-bs256-ms7B-tp4-pp1.tsv' \
    --network 'neupims_(4,1).json' --output 'evaluation/evaluation2/gpt7b-(4,1)' --gen > 'evaluation/evaluation2/gpt7b-(4,1).txt' 2>&1

# GPT3-7B (2, 2)
python3 -u main.py --model_name 'gpt3-6.7b' --npu_num 32 --npu_group 2 --npu_mem 4 \
    --pim_type 'local' --sub_batch --dataset 'dataset/alpaca-bs256-ms7B-tp2-pp2.tsv' \
    --network 'neupims_(2,2).json' --output 'evaluation/evaluation2/gpt7b-(2,2)' --gen > 'evaluation/evaluation2/gpt7b-(2,2).txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-6.7b*" -exec rm -rf {} +

# GPT3-13B (8, 1)
python3 -u main.py --model_name 'gpt3-13b' --npu_num 64 --npu_group 1 --npu_mem 4 \
    --pim_type 'local' --sub_batch --dataset 'dataset/alpaca-bs256-ms13B-tp8-pp1.tsv' \
    --network 'neupims_(8,1).json' --output 'evaluation/evaluation2/gpt13b-(8,1)' --gen > 'evaluation/evaluation2/gpt13b-(8,1).txt' 2>&1

# GPT3-13B (4, 2)
python3 -u main.py --model_name 'gpt3-13b' --npu_num 64 --npu_group 2 --npu_mem 4 \
    --pim_type 'local' --sub_batch --dataset 'dataset/alpaca-bs256-ms13B-tp4-pp2.tsv' \
    --network 'neupims_(4,2).json' --output 'evaluation/evaluation2/gpt13b-(4,2)' --gen > 'evaluation/evaluation2/gpt13b-(4,2).txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-13b*" -exec rm -rf {} +

# GPT3-30B (8, 2)
python3 -u main.py --model_name 'gpt3-30b' --npu_num 128 --npu_group 2 --npu_mem 4 \
    --pim_type 'local' --sub_batch --dataset 'dataset/alpaca-bs256-ms30B-tp8-pp2.tsv' \
    --network 'neupims_(8,2).json' --output 'evaluation/evaluation2/gpt30b-(8,2)' --gen > 'evaluation/evaluation2/gpt30b-(8,2).txt' 2>&1

# GPT3-30B (4, 4)
python3 -u main.py --model_name 'gpt3-30b' --npu_num 128 --npu_group 4 --npu_mem 4 \
    --pim_type 'local' --sub_batch --dataset 'dataset/alpaca-bs256-ms30B-tp4-pp4.tsv' \
    --network 'neupims_(4,4).json' --output 'evaluation/evaluation2/gpt30b-(4,4)' --gen > 'evaluation/evaluation2/gpt30b-(4,4).txt' 2>&1
find "$COMPILE_DIR" -name "gpt3-30b*" -exec rm -rf {} +