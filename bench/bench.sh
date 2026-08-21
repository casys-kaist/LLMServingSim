#!/bin/bash
# Run a vLLM benchmark and write results under bench/results/<run_id>/.
#
# This is a thin host-side wrapper that:
#   1. Resolves the repo root.
#   2. Launches python -m bench run inside the vLLM Docker container
#      (or the local uv venv from scripts/install-vllm.sh).
#
# Edit the variables below for your run, then execute:
#
#     ./bench/bench.sh
#
# Every setting below is also an environment override, so a different
# model / hardware target needs no separate copy of this script:
#
#     MODEL=meta-llama/Llama-3.1-8B \
#     DATASET=workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
#     TP=1 MAX_NUM_SEQS=256 MAX_MODEL_LEN=32768 \
#     ./bench/bench.sh
#
# (that one is the RTX 4090 / Llama-3.1-8B run committed under
# bench/examples/RTX4090/Llama-3.1-8B/ -- MAX_MODEL_LEN is capped so the
# KV cache fits in 24 GB.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../bench
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# =============================================================================
# EDIT THESE
# =============================================================================
MODEL="${MODEL:-Qwen/Qwen3-32B}"
DATASET="${DATASET:-workloads/sharegpt-qwen3-32b-300-sps10.jsonl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-bench/results/$RUN_ID}"

TP="${TP:-2}"
DP="${DP:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"   # blank => use the model default
DTYPE="${DTYPE:-bfloat16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
SEED="${SEED:-42}"
TICK_SECONDS="${TICK_SECONDS:-1.0}"
NUM_REQS="${NUM_REQS:-0}"            # 0 => replay the full dataset
LOG_LEVEL="${LOG_LEVEL:-INFO}"

EXPERT_PARALLEL="${EXPERT_PARALLEL:-0}"   # 1 to enable for MoE

# =============================================================================
# EXECUTE
# =============================================================================
mkdir -p "$OUTPUT_DIR"

cmd=(python3 -m bench run
    --model "$MODEL"
    --dataset "$DATASET"
    --output-dir "$OUTPUT_DIR"
    --tensor-parallel-size "$TP"
    --data-parallel-size "$DP"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --dtype "$DTYPE"
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --seed "$SEED"
    --tick-seconds "$TICK_SECONDS"
    --num-reqs "$NUM_REQS"
    --log-level "$LOG_LEVEL"
)

[[ -n "$MAX_MODEL_LEN" ]] && cmd+=(--max-model-len "$MAX_MODEL_LEN")
[[ "$EXPERT_PARALLEL" == "1" ]] && cmd+=(--enable-expert-parallel)

echo "Running: ${cmd[*]}"
"${cmd[@]}"
echo "Done. Results in: $OUTPUT_DIR"
