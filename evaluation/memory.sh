# #!/bin/bash
cd ..


# python main.py --cluster-config 'evaluation/memory/SD_config.json' --enable-prefix-caching \
#     --dataset '/dataset/sharegpt_pulse_req50_n6_delay15_pc.jsonl' --output 'evaluation/memory/SD_result.csv' \
#     --fp 16 --block-size 16 --num-req 300 --log-interval 1 > 'evaluation/memory/SD_output.txt'

python main.py --cluster-config 'evaluation/memory/MD_config.json' --enable-prefix-caching --prefix-storage "CPU" --enable-prefix-sharing \
    --dataset '/dataset/sharegpt_pulse_req50_n6_delay15_pc.jsonl' --output 'evaluation/memory/MD_sharing_result.csv' \
    --fp 16 --block-size 16 --num-req 300 --log-interval 1 > 'evaluation/memory/MD_sharing_output.txt'