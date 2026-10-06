#!/bin/bash
# One example per feature. The first is active; uncomment whichever you want.
#
# This file is a menu, not a test suite -- it deliberately shows one
# representative invocation per feature rather than every combination.
# For exhaustive coverage (every config, flag and parallelism shape, checked
# against recorded results) use ./serving/validate.sh.

# Single instance. Prefix caching in xPU memory is on by default.
python -m serving --cluster-config 'configs/cluster/single_node_single_instance.json' \
    --block-size 16 \
    --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_single_run.csv' \
    --num-reqs 10

# Multiple instances on one node, requests routed across them
# python -m serving --cluster-config 'configs/cluster/single_node_multi_instance.json' \
#     --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_multi_run.csv' \
#     --num-reqs 10

# Prefix cache spilling to a shared CPU pool
# python -m serving --cluster-config 'configs/cluster/single_node_multi_instance.json' \
#     --block-size 16 \
#     --enable-prefix-caching --enable-prefix-sharing --prefix-storage CPU \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_prefix_cpu_mem_pool_run.csv' \
#     --num-reqs 10

# Prefill/decode disaggregation
# python -m serving --cluster-config 'configs/cluster/single_node_pd_instance.json' \
#     --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_pd_run.csv' \
#     --num-reqs 10

# CXL memory expansion
# python -m serving --cluster-config 'configs/cluster/single_node_cxl_instance.json' \
#     --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_cxl_run.csv' \
#     --num-reqs 10

# PIM attention offloading
# python -m serving --cluster-config 'configs/cluster/single_node_pim_instance.json' \
#     --block-size 16 --enable-attn-offloading \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_pim_run.csv' \
#     --num-reqs 10

# Power and energy modelling
# python -m serving --cluster-config 'configs/cluster/single_node_power_instance.json' \
#     --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_power_run.csv' \
#     --num-reqs 10 --log-interval 0.1

# Tensor + pipeline parallelism (tp=2 x pp=2 -> 4 GPUs)
# python -m serving --cluster-config 'configs/cluster/single_node_tp_pp_instance.json' \
#     --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_tp_pp_run.csv' \
#     --num-reqs 10

# MoE with expert parallelism (tp=2, ep=2)
# python -m serving --cluster-config 'configs/cluster/single_node_moe_single_instance.json' \
#     --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_moe_run.csv' \
#     --num-reqs 10

# Data parallelism (dp=2 x ep=2 -> 2 GPUs, wave-synchronized)
# python -m serving --cluster-config 'configs/cluster/single_node_moe_dp_ep_instance.json' \
#     --block-size 16 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_moe_dp_ep_run.csv' \
#     --num-reqs 10

# Agentic sessions: LLM calls chained through tool calls (SWE-bench)
# python -m serving --cluster-config 'configs/cluster/single_node_moe_dp_ep_instance.json' \
#     --block-size 16 \
#     --dataset 'workloads/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl' \
#     --output 'outputs/example_agentic_run.csv' \
#     --num-reqs 1   # session count, not request count

# Speculative decoding. --num-speculative-tokens is vLLM's own flag name, and
# the acceptance rate defaults to the model's own published measurement
# (configs/spec_decode.json). Llama-3.1-8B has no published figure, so it has
# to be given one -- and since it declares no MTP modules it drafts with a
# separate model or n-gram, which this simulator has nothing to charge for, so
# the run warns that draft *time* is not counted and the speedup is an upper
# bound. On a model with MTP modules (DeepSeek-V3.2, GLM-5, MiniMax-M3,
# Qwen3.8) the run instead **refuses** until its catalog names both
# `mtp.prologue` and `mtp.decoder_block` and the bundle has an `mtp.csv`.
# python -m serving --cluster-config 'configs/cluster/single_node_single_instance.json' \
#     --block-size 16 --num-speculative-tokens 4 --spec-acceptance-rate 0.6 \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_spec_run.csv' \
#     --num-reqs 10

# Hybrid linear-attention model (gated DeltaNet). Needs a profiled bundle for
# the model, which the repo does not ship yet -- profile it first with
# MODEL="Qwen/Qwen3.8-27B" ./profiler/profile.sh. Omitting --block-size is
# deliberate here: vLLM raises the block size until one attention page covers
# one mamba state page (784 on this model), and the bundle records what it
# settled on.
# python -m serving --cluster-config 'configs/cluster/single_node_single_instance.json' \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_hybrid_run.csv' \
#     --num-reqs 10

# Inspect what was emitted: --save-trace-text writes the human-readable trace
# next to the .et files and implies --keep-inputs, so the run's ASTRA-Sim
# inputs survive under astra-sim/inputs/runs/<run-id>/. Both can leave
# gigabytes behind.
# python -m serving --cluster-config 'configs/cluster/single_node_single_instance.json' \
#     --block-size 16 --save-trace-text --run-id 'inspect_me' \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_trace_run.csv' \
#     --num-reqs 4

# -----------------------------------------------------------------------------------------------

# NS-3 network backend. Work in progress: needs the ns-3 submodule built, and
# the placeholder below may not run until that integration lands.
# python -m serving --cluster-config 'configs/cluster/single_node_single_instance.json' \
#     --block-size 16 --network-backend 'ns3' \
#     --dataset 'workloads/example_trace.jsonl' --output 'outputs/example_ns3_run.csv' \
#     --num-reqs 10
