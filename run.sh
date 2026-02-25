# #!/bin/bash

# Single instance example
python main.py --cluster-config 'cluster_config/single_node_single_instance.json' \
    --fp 16 --block-size 16 \
    --dataset '/dataset/example_trace.jsonl' --output 'output/example_single_run.csv' \
    --num-req 100 --log-interval 1.0

# Multi instance example
python main.py --cluster-config 'cluster_config/single_node_multi_instance.json' \
    --fp 16 --block-size 16 \
    --dataset 'dataset/example_trace.jsonl' --output 'output/example_multi_run.csv' \
    --num-req 10

# PD example
python main.py --cluster-config 'cluster_config/single_node_pd_instance.json' \
    --fp 16 --block-size 16 \
    --dataset 'dataset/example_trace.jsonl' --output 'output/example_pd_run.csv' \
    --num-req 10

# MoE example
python main.py --cluster-config 'cluster_config/single_node_moe_single_instance.json' \
    --fp 16 --block-size 16 \
    --dataset 'dataset/example_trace.jsonl' --output 'output/example_moe_run.csv' \
    --num-req 10

# Prefix caching example
python main.py --cluster-config 'cluster_config/single_node_single_instance.json' \
    --fp 16 --block-size 16 --enable-prefix-caching \
    --dataset 'dataset/example_trace.jsonl' --output 'output/example_prefix_run.csv' \
    --num-req 10

# CXL example
python main.py --cluster-config 'cluster_config/single_node_cxl_instance.json' \
    --fp 16 --block-size 16 \
    --dataset 'dataset/example_trace.jsonl' --output 'output/example_cxl_run.csv' \
    --num-req 10

# Prefix cache with cpu mem example
python main.py --cluster-config 'cluster_config/single_node_single_instance.json' \
    --fp 16 --block-size 16 --enable-prefix-caching \
    --dataset '/dataset/example_trace.jsonl' --output 'output/example_prefix_cpu_mem_run.csv' \
    --num-req 10

# Prefix cache with CPU Prefix Cache Pool example (Single Node)
python main.py --cluster-config 'cluster_config/single_node_multi_instance.json' \
     --fp 16 --block-size 16 \
    --enable-prefix-caching --enable-prefix-sharing --prefix-storage CPU \
    --dataset '/dataset/example_trace.jsonl' --output 'output/example_prefix_cpu_mem_pool_run.csv' \
    --num-req 10

# Prefix cache with CPU Prefix Cache Pool example (Dual Node)
python main.py --cluster-config 'cluster_config/dual_node_multi_instance.json' \
    --fp 16 --block-size 16 \
    --enable-prefix-caching --enable-prefix-sharing --prefix-storage CPU \
    --dataset '/dataset/example_trace.jsonl' --output 'output/example_dual_prefix_cpu_mem_pool_run.csv' \
    --num-req 10

# Power model example
python main.py --cluster-config 'cluster_config/single_node_power_instance.json' \
    --fp 16 --block-size 16 \
    --dataset 'dataset/example_trace.jsonl' --output 'output/example_power_run.csv' \
    --num-req 10 --log-interval 0.1

# PIM example
python main.py --cluster-config 'cluster_config/single_node_pim_instance.json' \
    --fp 16 --block-size 16 --enable-attn-offloading \
    --dataset 'dataset/example_trace.jsonl' --output 'output/example_pim_run.csv' \
    --num-req 10 --log-interval 1 --log-level WARNING

# Sub-batch interleaving example
python main.py --cluster-config 'cluster_config/single_node_pim_instance.json' \
    --fp 16 --block-size 16 --enable-attn-offloading --enable-sub-batch-interleaving \
    --dataset 'dataset/example_trace.jsonl' --output 'output/example_pim_sub_batch_run.csv' \
    --num-req 10 --log-interval 1 --log-level WARNING

# NS-3 example
# Note: NS-3 integration is currently a work in progress. The following command is a placeholder and may not work until the NS-3 integration is complete.
# python main.py --cluster-config 'cluster_config/single_node_single_instance.json' \
#     --fp 16 --block-size 16 --network-backend 'ns3' \
#     --dataset 'dataset/example_trace.jsonl' --output 'output/example_ns3_run.csv' \
#     --num-req 10 