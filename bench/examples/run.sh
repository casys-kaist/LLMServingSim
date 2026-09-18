#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"

BLOCK_SIZE="${BLOCK_SIZE:-}"
LOG_LEVEL="${LOG_LEVEL:-WARNING}"
NETWORK_BACKEND="${NETWORK_BACKEND:-analytical}"
TERM="${TERM:-xterm-256color}"
LANG="${LANG:-C.UTF-8}"
FORCE_COLOR="${FORCE_COLOR:-1}"

export TERM LANG FORCE_COLOR

# Examples are keyed by <hardware>/<model>, matching the directory layout
# under this folder. Each one carries its own config.json, so nothing has
# to be kept in sync with a parallel configs/ tree.
DEFAULT_EXAMPLES=()
for config in "$SCRIPT_DIR"/*/*/config.json; do
    [[ -f "$config" ]] || continue
    example=${config#"$SCRIPT_DIR/"}
    DEFAULT_EXAMPLES+=("${example%/config.json}")
done

json_get() {
    local json_path="$1"
    local key_path="$2"
    "$PYTHON" - "$json_path" "$key_path" "${3:-required}" <<'PY'
import json
import sys

path = sys.argv[1]
key = sys.argv[2]

with open(path, encoding="utf-8") as f:
    obj = json.load(f)

for part in key.split("."):
    if sys.argv[3] == "optional" and (not isinstance(obj, dict) or part not in obj):
        sys.exit(0)
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
    local max_num_seqs
    local max_num_batched_tokens
    local block_size

    dataset_rel="$(json_get "$meta" "dataset_path")"
    dataset_cli="$(repo_relative_path "$dataset_rel")"
    dataset="$(resolve_repo_path "$dataset_rel")"
    num_reqs="$(json_get "$meta" "num_requests")"
    # dtype and kv_cache_dtype are deliberately not read back: the simulator
    # derives both from the model config now, so passing vLLM's recorded
    # values would be an input it no longer has. They agree on every bundled
    # example (bfloat16 / auto).
    max_num_seqs="$(json_get "$meta" "engine_kwargs.max_num_seqs")"
    max_num_batched_tokens="$(json_get "$meta" "engine_kwargs.max_num_batched_tokens")"
    # Match the recorded engine, not a dense-only global default. Sparse
    # backends may resolve a different page size (64 for the shrunk DSA run).
    block_size="${BLOCK_SIZE:-$(json_get "$meta" "kv_cache.block_size" optional)}"
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
        --max-num-seqs "$max_num_seqs"
        --max-num-batched-tokens "$max_num_batched_tokens"
        --log-level "$LOG_LEVEL"
        --network-backend "$NETWORK_BACKEND"
    )

    [[ -n "$block_size" ]] && cmd+=(--block-size "$block_size")

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
