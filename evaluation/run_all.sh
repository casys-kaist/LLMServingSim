#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
format_duration() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

TOTAL_START=$(date +%s)
echo "[Evaluation] Starting full evaluation run."

for script in \
    "figure_5.sh" \
    "figure_6.sh" \
    "figure_7.sh" \
    "figure_8.sh" \
    "figure_9.sh" \
    "figure_10.sh"
do
    (
        cd "$SCRIPT_DIR"
        bash "$script"
    )
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))
echo "[Evaluation] Full evaluation run completed in $(format_duration "$TOTAL_ELAPSED")."
