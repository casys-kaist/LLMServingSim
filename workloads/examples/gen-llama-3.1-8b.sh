#!/bin/bash
# Generate sharegpt-llama-3.1-8b-300-sps10.jsonl with tokenizer-only defaults.
# Optional --use-vllm generates new replies; see the flags below.
#
# Run inside the vLLM Docker (scripts/docker-vllm.sh) at /workspace.
# Llama-3.1-8B is gated — HF_TOKEN must be set in the container env.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"

python3 -m workloads.generators sharegpt \
    --model "$MODEL" \
    --source shibing624/sharegpt_gpt4 \
    --num-reqs 300 --sps 10 --seed 42 \
    --min-input-toks 256 --min-output-toks 512 \
    --output workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl

    # To enable vLLM generation, append these flags to the Python command:
    # add a trailing backslash to its --output line, then place the flag
    # lines immediately after it with their comment prefixes removed.
    # Do not leave blank or comment lines inside the continued command.
    # --use-vllm \
    # --vllm-tp 2 \
    # --vllm-dtype bfloat16 \
    # --vllm-max-num-seqs 1024 \
    # --vllm-max-num-batched-tokens 16384

