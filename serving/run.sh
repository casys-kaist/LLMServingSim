#!/bin/bash
# One example per feature. The first is active; uncomment whichever you want.
#
# This file is a menu, not a test suite -- it deliberately shows one
# representative invocation per feature rather than every combination.
# For exhaustive coverage (every config, flag and parallelism shape, checked
# against recorded results) use ./serving/validate.sh.

# Single instance. Prefix caching in xPU memory is on by default.
python -m serving --cluster-config 'configs/cluster/single_node_single_instance.json' \
    --dtype bfloat16 --block-size 16 \
    --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_single_run.csv' \
    --num-reqs 10

# Multiple instances on one node, requests routed across them
# python -m serving --cluster-config 'configs/cluster/single_node_multi_instance.json' \
#     --dtype bfloat16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_multi_run.csv' \
#     --num-reqs 10

# Prefix cache spilling to a shared CPU pool
# python -m serving --cluster-config 'configs/cluster/single_node_multi_instance.json' \
#     --dtype bfloat16 --block-size 16 \
#     --enable-prefix-caching --enable-prefix-sharing --prefix-storage CPU \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_prefix_cpu_mem_pool_run.csv' \
#     --num-reqs 10

# Prefill/decode disaggregation
# python -m serving --cluster-config 'configs/cluster/single_node_pd_instance.json' \
#     --dtype bfloat16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_pd_run.csv' \
#     --num-reqs 10

# CXL memory expansion
# python -m serving --cluster-config 'configs/cluster/single_node_cxl_instance.json' \
#     --dtype bfloat16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_cxl_run.csv' \
#     --num-reqs 10

# PIM attention offloading
# python -m serving --cluster-config 'configs/cluster/single_node_pim_instance.json' \
#     --dtype bfloat16 --block-size 16 --enable-attn-offloading \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_pim_run.csv' \
#     --num-reqs 10

# Power and energy modelling
# python -m serving --cluster-config 'configs/cluster/single_node_power_instance.json' \
#     --dtype bfloat16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_power_run.csv' \
#     --num-reqs 10 --log-interval 0.1

# Tensor + pipeline parallelism (tp=2 x pp=2 -> 4 GPUs)
# python -m serving --cluster-config 'configs/cluster/single_node_tp_pp_instance.json' \
#     --dtype bfloat16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_tp_pp_run.csv' \
#     --num-reqs 10

# MoE with expert parallelism (tp=2, ep=2)
# python -m serving --cluster-config 'configs/cluster/single_node_moe_single_instance.json' \
#     --dtype bfloat16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_moe_run.csv' \
#     --num-reqs 10

# Data parallelism (dp=2 x ep=2 -> 2 GPUs, wave-synchronized)
# python -m serving --cluster-config 'configs/cluster/single_node_moe_dp_ep_instance.json' \
#     --dtype bfloat16 --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_moe_dp_ep_run.csv' \
#     --num-reqs 10

# Agentic sessions: LLM calls chained through tool calls (SWE-bench)
# python -m serving --cluster-config 'configs/cluster/single_node_moe_dp_ep_instance.json' \
#     --dtype bfloat16 --block-size 16 \
#     --dataset 'workloads/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl' \
#     --output 'outputs/example_agentic_run.csv' \
#     --num-reqs 1   # session count, not request count

# -----------------------------------------------------------------------------------------------

# NS-3 network backend. Work in progress: needs the ns-3 submodule built, and
# the placeholder below may not run until that integration lands.
# python -m serving --cluster-config 'configs/cluster/single_node_single_instance.json' \
#     --dtype bfloat16 --block-size 16 --network-backend 'ns3' \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_ns3_run.csv' \
#     --num-reqs 10
