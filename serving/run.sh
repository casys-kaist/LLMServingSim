#!/bin/bash

# Single instance example (prefix caching in xPU memory is default now)
python -m serving --cluster-config 'configs/cluster/single_node_single_instance.json' \
    --dtype bfloat16 --block-size 16 \
    --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_single_run.csv' \
    --num-reqs 10

# Multi instance example
# python -m serving --cluster-config 'configs/cluster/single_node_multi_instance.json' \
#     --dtype bfloat16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_multi_run.csv' \
#     --num-reqs 10

# # PD example
# python -m serving --cluster-config 'configs/cluster/single_node_pd_instance.json' \
#     --dtype bfloat16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_pd_run.csv' \
#     --num-reqs 10

# # CXL example
# python -m serving --cluster-config 'configs/cluster/single_node_cxl_instance.json' \
#     --dtype bfloat16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_cxl_run.csv' \
#     --num-reqs 10

# # Prefix cache with CPU Prefix Cache Pool example (Single Node)
# python -m serving --cluster-config 'configs/cluster/single_node_multi_instance.json' \
#      --dtype bfloat16 --block-size 16 \
#     --enable-prefix-caching --enable-prefix-sharing --prefix-storage CPU \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_prefix_cpu_mem_pool_run.csv' \
#     --num-reqs 10

# # Prefix cache with CPU Prefix Cache Pool example (Dual Node)
# python -m serving --cluster-config 'configs/cluster/dual_node_multi_instance.json' \
#     --dtype bfloat16 --block-size 16 \
#     --enable-prefix-caching --enable-prefix-sharing --prefix-storage CPU \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_dual_prefix_cpu_mem_pool_run.csv' \
#     --num-reqs 10

# # Power model example
# python -m serving --cluster-config 'configs/cluster/single_node_power_instance.json' \
#     --dtype bfloat16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_power_run.csv' \
#     --num-reqs 10 --log-interval 0.1

# # PIM example
# python -m serving --cluster-config 'configs/cluster/single_node_pim_instance.json' \
#     --dtype bfloat16 --block-size 16 --enable-attn-offloading \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_pim_run.csv' \
#     --num-reqs 10 --log-level WARNING

# # Sub-batch interleaving example
# python -m serving --cluster-config 'configs/cluster/single_node_pim_instance.json' \
#     --dtype bfloat16 --block-size 16 --enable-attn-offloading --enable-sub-batch-interleaving \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_pim_sub_batch_run.csv' \
#     --num-reqs 10 --log-level WARNING

# # Pipeline parallelism example (4 GPUs as pp=4, one stage each)
# python -m serving --cluster-config 'configs/cluster/single_node_pp_instance.json' \
#     --dtype bfloat16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_pp_run.csv' \
#     --num-reqs 10

# # Tensor + pipeline parallelism example (4 GPUs as tp=2 x pp=2)
# python -m serving --cluster-config 'configs/cluster/single_node_tp_pp_instance.json' \
#     --dtype bfloat16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_tp_pp_run.csv' \
#     --num-reqs 10

# # MoE with tensor + pipeline + expert parallelism (4 GPUs as tp=2 x pp=2, ep=2)
# python -m serving --cluster-config 'configs/cluster/single_node_moe_pp_instance.json' \
#     --dtype bfloat16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_moe_pp_run.csv' \
#     --num-reqs 10



# MoE example
# python -m serving --cluster-config 'configs/cluster/single_node_moe_single_instance.json' \
#     --dtype bfloat16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_moe_run.csv' \
#     --num-reqs 10

# MoE DP+EP with agentic session example (SWE-bench)
# python -m serving --cluster-config 'configs/cluster/single_node_moe_dp_ep_instance.json' \
#     --dtype bfloat16 --block-size 16 \
#     --dataset 'workloads/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl' --output 'outputs/example_moe_dp_ep_run.csv' \
#     --num-reqs 1 # session count in agentic workload


# -----------------------------------------------------------------------------------------------
#    Deprecated examples (may not be up to date with the latest codebase, kept for reference)
# -----------------------------------------------------------------------------------------------


# Prefix caching example (disabled: prefix caching in xPU memory is default now)
# python -m serving --cluster-config 'configs/cluster/single_node_single_instance.json' \
#     --dtype bfloat16 --block-size 16 --enable-prefix-caching \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_prefix_run.csv' \
#     --num-reqs 10

# NS-3 example
# Note: NS-3 integration is currently a work in progress. The following command is a placeholder and may not work until the NS-3 integration is complete.
# python -m serving --cluster-config 'configs/cluster/single_node_single_instance.json' \
#     --dtype bfloat16 --block-size 16 --network-backend 'ns3' \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_ns3_run.csv' \
#     --num-reqs 10 