#!/bin/bash
# Compare a finished bench run against simulator output.
#
# Usage:
#     ./llmservingsim/bench/validate.sh <bench_dir> <sim_csv> <sim_log> [prefix]
#
# Example:
#     ./llmservingsim/bench/validate.sh \
#         bench/results/20260426-001234 \
#         outputs/qwen3-32b-sharegpt/sim.csv \
#         outputs/qwen3-32b-sharegpt/sim.log \
#         eval
#
# Optional environment overrides:
#     OUTPUT_SUBDIR=validation
#     TITLE="vLLM vs LLMServingSim"
#     LOG_LEVEL=INFO

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <bench_dir> <sim_csv> <sim_log> [prefix]" >&2
    exit 2
fi

BENCH_DIR="$1"
SIM_CSV="$2"
SIM_LOG="$3"
PREFIX="${4:-}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-validation}"
TITLE="${TITLE:-vLLM vs LLMServingSim}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

cmd=(python -m llmservingsim.bench validate
    --bench-dir "$BENCH_DIR"
    --sim-csv "$SIM_CSV"
    --sim-log "$SIM_LOG"
    --output-subdir "$OUTPUT_SUBDIR"
    --title "$TITLE"
    --log-level "$LOG_LEVEL"
)
[[ -n "$PREFIX" ]] && cmd+=(--prefix "$PREFIX")

echo "Running: ${cmd[*]}"
"${cmd[@]}"
