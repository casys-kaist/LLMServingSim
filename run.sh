#!/bin/bash

python main.py --model_name 'gpt3-6.7b' --hardware 'RTX3090' --npu_num 1 --npu_group 1 --npu_mem 40 \
    --local_bw 1024 --remote_bw 512 --link_bw 256 --fp 16 --block_size 4 \
    --dataset 'dataset/share-gpt-req100-rate10.tsv' --output 'output/example_run.csv' \
    --verbose --req_num 10