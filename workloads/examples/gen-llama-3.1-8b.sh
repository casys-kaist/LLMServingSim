#!/bin/bash
# Generate sharegpt-llama-3.1-8b-300-sps10.jsonl via real vLLM (free-gen).
#
# Run inside the vLLM Docker (scripts/docker-vllm.sh) at /workspace.
# Llama-3.1-8B is gated — HF_TOKEN must be set in the container env.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"

python -m llmservingsim.workloads.generators sharegpt \
    --model "$MODEL" \
    --source shibing624/sharegpt_gpt4 \
    --num-reqs 300 --sps 10 --seed 42 \
    --min-input-toks 256 --min-output-toks 512 \
    --output workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl

    # Use these to use vLLM for generation
    # --use-vllm \
    # --vllm-tp 2 \
    # --vllm-dtype bfloat16 \
    # --vllm-max-num-seqs 1024 \
    # --vllm-max-num-batched-tokens 16384

