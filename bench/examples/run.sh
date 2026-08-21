#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"

BLOCK_SIZE="${BLOCK_SIZE:-16}"
LOG_LEVEL="${LOG_LEVEL:-WARNING}"
NETWORK_BACKEND="${NETWORK_BACKEND:-analytical}"
TERM="${TERM:-xterm-256color}"
LANG="${LANG:-C.UTF-8}"
FORCE_COLOR="${FORCE_COLOR:-1}"

export TERM LANG FORCE_COLOR

# Examples are keyed by <hardware>/<model>, matching the directory layout
# under this folder. Each one carries its own config.json, so nothing has
# to be kept in sync with a parallel configs/ tree.
DEFAULT_EXAMPLES=(
    "RTXPRO6000/Llama-3.1-8B"
    "RTXPRO6000/Qwen3-32B"
    "RTXPRO6000/Qwen3-30B-A3B-Instruct-2507"
    "RTX4090/Llama-3.1-8B"
)

json_get() {
    local json_path="$1"
    local key_path="$2"
    "$PYTHON" - "$json_path" "$key_path" <<'PY'
import json
import sys

path = sys.argv[1]
key = sys.argv[2]

with open(path, encoding="utf-8") as f:
    obj = json.load(f)

for part in key.split("."):
    obj = obj[part]

if obj is None:
    sys.exit(0)
if isinstance(obj, bool):
    print("true" if obj else "false")
else:
    print(obj)
PY
}

resolve_repo_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf '%s\n' "$path"
    else
        printf '%s/%s\n' "$REPO_ROOT" "$path"
    fi
}

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

run_example() {
    local model_dir="$1"   # <hardware>/<model>
    local meta="$SCRIPT_DIR/$model_dir/vllm/meta.json"
    local config="$SCRIPT_DIR/$model_dir/config.json"
    local config_rel
    local output_dir="$SCRIPT_DIR/$model_dir/outputs"
    local output_dir_rel

    [[ -f "$meta" ]] || { echo "Missing meta: $meta" >&2; exit 1; }
    [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 1; }

    local dataset_rel
    local dataset_cli
    local dataset
    local num_reqs
    local dtype
    local kv_cache_dtype
    local max_num_seqs
    local max_num_batched_tokens

    dataset_rel="$(json_get "$meta" "dataset_path")"
    dataset_cli="$(repo_relative_path "$dataset_rel")"
    dataset="$(resolve_repo_path "$dataset_rel")"
    num_reqs="$(json_get "$meta" "num_requests")"
    dtype="$(json_get "$meta" "engine_kwargs.dtype")"
    kv_cache_dtype="$(json_get "$meta" "engine_kwargs.kv_cache_dtype")"
    max_num_seqs="$(json_get "$meta" "engine_kwargs.max_num_seqs")"
    max_num_batched_tokens="$(json_get "$meta" "engine_kwargs.max_num_batched_tokens")"
    config_rel="$(repo_relative_path "$config")"
    output_dir_rel="$(repo_relative_path "$output_dir")"

    [[ -f "$dataset" ]] || { echo "Missing dataset: $dataset" >&2; exit 1; }
    mkdir -p "$output_dir"

    local cmd=(
        "$PYTHON" -m serving
        --cluster-config "$config_rel"
        --dataset "$dataset_cli"
        --output "$output_dir_rel/sim.csv"
        --num-reqs "$num_reqs"
        --dtype "$dtype"
        --kv-cache-dtype "$kv_cache_dtype"
        --block-size "$BLOCK_SIZE"
        --max-num-seqs "$max_num_seqs"
        --max-num-batched-tokens "$max_num_batched_tokens"
        --log-level "$LOG_LEVEL"
        --network-backend "$NETWORK_BACKEND"
    )

    if [[ -n "${NPU_MEMORY_UTILIZATION:-}" ]]; then
        cmd+=(--npu-memory-utilization "$NPU_MEMORY_UTILIZATION")
    fi

    echo "============================================================"
    echo "Example: $model_dir"
    echo "Dataset: $dataset_cli"
    echo "Config:  $config_rel"
    echo "Output:  $output_dir_rel"
    echo "Running: ${cmd[*]}"

    (
        cd "$REPO_ROOT"
        "${cmd[@]}"
    ) 2>&1 | tee "$output_dir/sim.log"
}

if [[ $# -eq 0 ]]; then
    set -- "${DEFAULT_EXAMPLES[@]}"
fi

for example in "$@"; do
    if [[ -d "$SCRIPT_DIR/$example" && -f "$SCRIPT_DIR/$example/config.json" ]]; then
        run_example "$example"
    else
        echo "Unknown example: $example" >&2
        echo "Known examples: ${DEFAULT_EXAMPLES[*]}" >&2
        exit 2
    fi
done
