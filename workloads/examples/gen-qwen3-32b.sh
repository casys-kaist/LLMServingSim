#!/bin/bash
# Generate sharegpt-qwen3-32b-300-sps10.jsonl via real vLLM (free-gen).
#
# Run inside the vLLM Docker (scripts/docker-vllm.sh) at /workspace.
# TP=2 across two visible GPUs for faster throughput on the dense 32B
# model. ``--model`` accepts an HF id (auto-downloaded into the cache)
# or a local checkpoint path.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL="${MODEL:-Qwen/Qwen3-32B}"

python -m llmservingsim.workloads.generators sharegpt \
    --model "$MODEL" \
    --source shibing624/sharegpt_gpt4 \
    --num-reqs 300 --sps 10 --seed 42 \
    --min-input-toks 256 --min-output-toks 512 \
    --output workloads/sharegpt-qwen3-32b-300-sps10.jsonl

    # Use these to use vLLM for generation
    # --use-vllm \
    # --vllm-tp 2 \
    # --vllm-dtype bfloat16 \
    # --vllm-max-num-seqs 1024 \
    # --vllm-max-num-batched-tokens 16384
