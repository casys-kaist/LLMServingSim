# #!/bin/bash
cd ..

###############################################
#     Evaluation of LLMServingSim2.0 - PIM    #
###############################################

# only-gpu
# python main.py --cluster-config 'evaluation/pim/gpu_only_config.json' \
#     --dataset '/dataset/fixed_in128_out512_req256_rate10.jsonl' --output 'evaluation/pim/gpu_only_b256_result.csv' \
#     --fp 16 --block-size 16 --num-req 256 --log-interval 1  > 'evaluation/pim/gpu_only_b256_output.txt'

# # with naive pim
# python main.py --cluster-config 'evaluation/pim/pim_config.json' --enable-attn-offloading \
#     --dataset '/dataset/fixed_in128_out512_req256_rate10.jsonl' --output 'evaluation/pim/pim_b256_result.csv' \
#     --fp 16 --block-size 16 --num-req 256 --log-interval 1 > 'evaluation/pim/pim_b256_output.txt'

# # with pim & sub-batch interleaving
# python main.py --cluster-config 'evaluation/pim/pim_config.json' --enable-attn-offloading --enable-sub-batch-interleaving \
#     --dataset '/dataset/fixed_in128_out512_req256_rate10.jsonl' --output 'evaluation/pim/pim_sbi_b256_result.csv' \
#     --fp 16 --block-size 16 --num-req 256 --log-interval 1  > 'evaluation/pim/pim_sbi_b256_output.txt'

# only-gpu
python main.py --cluster-config 'evaluation/pim/gpu_only_config.json' \
    --dataset '/dataset/fixed_in128_out512_req512_rate10.jsonl' --output 'evaluation/pim/gpu_only_b512_result.csv' \
    --fp 16 --block-size 16 --num-req 512 --log-interval 1  > 'evaluation/pim/gpu_only_b512_output.txt'

# with naive pim
python main.py --cluster-config 'evaluation/pim/pim_config.json' --enable-attn-offloading \
    --dataset '/dataset/fixed_in128_out512_req512_rate10.jsonl' --output 'evaluation/pim/pim_b512_result.csv' \
    --fp 16 --block-size 16 --num-req 512 --log-interval 1 > 'evaluation/pim/pim_b512_output.txt'

# with pim & sub-batch interleaving
python main.py --cluster-config 'evaluation/pim/pim_config.json' --enable-attn-offloading --enable-sub-batch-interleaving \
    --dataset '/dataset/fixed_in128_out512_req512_rate10.jsonl' --output 'evaluation/pim/pim_sbi_b512_result.csv' \
    --fp 16 --block-size 16 --num-req 512 --log-interval 1  > 'evaluation/pim/pim_sbi_b512_output.txt'
