# #!/bin/bash
cd ..

###############################################
# Evaluation of LLMServingSim2.0 - Validation #
###############################################

#################### A6000 ####################
# SD (Single Dense)
python main.py --cluster-config 'evaluation/validation/A6000/SD_config.json' \
    --dataset '/dataset/sharegpt_req300_rate10_llama.jsonl' --output 'evaluation/validation/A6000/SD_result.csv' \
    --fp 16 --block-size 16 --num-req 300 --log-interval 1 > 'evaluation/validation/A6000/SD_output.txt'

# # MD (Multi-instane Dense)
# python main.py --cluster-config 'evaluation/validation/A6000/MD_config.json' \
#     --dataset '/dataset/sharegpt_req300_rate10_llama.jsonl' --output 'evaluation/validation/A6000/MD_result.csv' \
#     --fp 16 --block-size 16 --num-req 300 --log-interval 1 > 'evaluation/validation/A6000/MD_output.txt'

# # PDD (Prefill/Decode Disaggregation Dense)
# python main.py --cluster-config 'evaluation/validation/A6000/PDD_config.json' \
#     --dataset '/dataset/sharegpt_req300_rate10_llama.jsonl' --output 'evaluation/validation/A6000/PDD_result.csv' \
#     --fp 16 --block-size 16 --num-req 300 --log-interval 1 > 'evaluation/validation/A6000/PDD_output.txt'

# # SD+PC (Single Dense with Prefix Caching)
# python main.py --cluster-config 'evaluation/validation/A6000/SD+PC_config.json' --enable-prefix-caching \
#     --dataset '/dataset/sharegpt_req300_rate10_llama.jsonl' --output 'evaluation/validation/A6000/SD+PC_result.csv' \
#     --fp 16 --block-size 16 --num-req 300 --log-interval 1 > 'evaluation/validation/A6000/SD+PC_output.txt'

# SM (Single MoE)
# python main.py --cluster-config 'evaluation/validation/A6000/SM_config.json' \
#     --dataset '/dataset/sharegpt_req300_rate10_phi.jsonl' --output 'evaluation/validation/A6000/SM_result.csv' \
#     --fp 16 --block-size 16 --num-req 300 --log-interval 1 > 'evaluation/validation/A6000/SM_output.txt'



##################### H100 ####################
# # SD (Single Dense)
# python main.py --cluster-config 'evaluation/validation/H100/SD_config.json' \
#     --dataset '/dataset/sharegpt_req300_rate10_llama.jsonl' --output 'evaluation/validation/H100/SD_result.csv' \
#     --fp 16 --block-size 16 --num-req 300 --log-interval 1 > 'evaluation/validation/H100/SD_output.txt'

# # MD (Multi-instane Dense)
# python main.py --cluster-config 'evaluation/validation/H100/MD_config.json' \
#     --dataset '/dataset/sharegpt_req300_rate10_llama.jsonl' --output 'evaluation/validation/H100/MD_result.csv' \
#     --fp 16 --block-size 16 --num-req 300 --log-interval 1 > 'evaluation/validation/H100/MD_output.txt'

# # PD (Prefill/Decode Disaggregation)
# python main.py --cluster-config 'evaluation/validation/H100/PDD_config.json' \
#     --dataset '/dataset/sharegpt_req300_rate10_llama.jsonl' --output 'evaluation/validation/H100/PDD_result.csv' \
#     --fp 16 --block-size 16 --num-req 300 --log-interval 1 > 'evaluation/validation/H100/PDD_output.txt'

# # SD+PC (Single Dense with Prefix Caching)
# python main.py --cluster-config 'evaluation/validation/H100/SD+PC_config.json' --enable-prefix-caching \
#     --dataset '/dataset/sharegpt_req300_rate10_llama.jsonl' --output 'evaluation/validation/H100/SD+PC_result.csv' \
#     --fp 16 --block-size 16 --num-req 300 --log-interval 1 > 'evaluation/validation/H100/SD+PC_output.txt'

# # SM (Single MoE)
# python main.py --cluster-config 'evaluation/validation/H100/SM_config.json' \
#     --dataset '/dataset/sharegpt_req300_rate10_mixtral.jsonl' --output 'evaluation/validation/H100/SM_result.csv' \
#     --fp 16 --block-size 16 --num-req 300 --log-interval 1 > 'evaluation/validation/H100/SM_output.txt'

##################### TPU-v6e-1 ####################
# SD (Single Dense)
# python main.py --cluster-config 'evaluation/validation/TPU-v6e-1/SD_config.json' \
#     --dataset '/dataset/sharegpt_req300_rate10_llama.jsonl' --output 'evaluation/validation/TPU-v6e-1/SD_result.csv' \
#     --fp 16 --block-size 16 --num-req 300 --log-interval 1 > 'evaluation/validation/TPU-v6e-1/SD_output.txt'