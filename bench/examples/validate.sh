#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"

OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-../validation}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
PREFIX="${PREFIX:-}"
TITLE_PREFIX="${TITLE_PREFIX:-vLLM vs LLMServingSim}"

# Examples are keyed by <hardware>/<model>, matching the directory layout
# under this folder.
DEFAULT_EXAMPLES=(
    "RTXPRO6000/Llama-3.1-8B"
    "RTXPRO6000/Qwen3-32B"
    "RTXPRO6000/Qwen3-30B-A3B-Instruct-2507"
    "RTX4090/Llama-3.1-8B"
)

repo_relative_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        case "$path" in
            "$REPO_ROOT"/*)
                printf '%s\n' "${path#"$REPO_ROOT"/}"
                ;;
            *)
                echo "Path must live under the repo root: $path" >&2
                exit 1
                ;;
        esac
    else
        printf '%s\n' "$path"
    fi
}

validate_example() {
    local model_dir="$1"   # <hardware>/<model>
    local vllm_dir="$SCRIPT_DIR/$model_dir/vllm"
    local sim_csv="$SCRIPT_DIR/$model_dir/outputs/sim.csv"
    local sim_log="$SCRIPT_DIR/$model_dir/outputs/sim.log"
    local vllm_dir_rel
    local sim_csv_rel
    local sim_log_rel

    [[ -d "$vllm_dir" ]] || { echo "Missing vLLM bench dir: $vllm_dir" >&2; exit 1; }
    [[ -f "$sim_csv" ]] || { echo "Missing sim CSV: $sim_csv" >&2; exit 1; }
    [[ -f "$sim_log" ]] || { echo "Missing sim log: $sim_log" >&2; exit 1; }

    vllm_dir_rel="$(repo_relative_path "$vllm_dir")"
    sim_csv_rel="$(repo_relative_path "$sim_csv")"
    sim_log_rel="$(repo_relative_path "$sim_log")"

    local cmd=(
        "$PYTHON" -m bench validate
        --bench-dir "$vllm_dir_rel"
        --sim-csv "$sim_csv_rel"
        --sim-log "$sim_log_rel"
        --output-subdir "$OUTPUT_SUBDIR"
        --title "$TITLE_PREFIX - $model_dir"
        --log-level "$LOG_LEVEL"
    )

    [[ -n "$PREFIX" ]] && cmd+=(--prefix "$PREFIX")

    echo "============================================================"
    echo "Example:    $model_dir"
    echo "Bench dir:  $vllm_dir_rel"
    echo "Sim CSV:    $sim_csv_rel"
    echo "Sim log:    $sim_log_rel"
    echo "Output dir: $model_dir/validation"
    echo "Running: ${cmd[*]}"

    (
        cd "$REPO_ROOT"
        "${cmd[@]}"
    )
}

if [[ $# -eq 0 ]]; then
    set -- "${DEFAULT_EXAMPLES[@]}"
fi

for example in "$@"; do
    if [[ -d "$SCRIPT_DIR/$example" ]]; then
        validate_example "$example"
    else
        echo "Unknown example: $example" >&2
        echo "Known examples: ${DEFAULT_EXAMPLES[*]}" >&2
        exit 2
    fi
done
