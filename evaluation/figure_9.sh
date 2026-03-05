# #!/bin/bash
cd ..
format_duration() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

SCRIPT_START=$(date +%s)
THROUGHPUT_PARSER="evaluation/parser/parser_throughput.py"
LATENCY_PARSER="evaluation/parser/parser_latency.py"
FIG_DIR="evaluation/figure_9"
CONFIG_DIR="$FIG_DIR/config"
DATASET="/dataset/sharegpt_req300_rate10_llama.jsonl"
FIGURE_SCRIPT="$FIG_DIR/figure_9.py"
mkdir -p "$FIG_DIR/logs" "$FIG_DIR/results" "$FIG_DIR/parsed"
echo "[Figure 9] Starting evaluation."

###############################################
#    Evaluation of LLMServingSim2.0 - TPU     #
###############################################

##################### TPU-v6e-1 ####################
# SD (Single Dense)
python main.py --cluster-config "$CONFIG_DIR/SD_config.json" \
    --dataset "$DATASET" --output "$FIG_DIR/results/SD_result.csv" \
    --fp 16 --block-size 16 --num-req 300 --log-interval 1 > "$FIG_DIR/logs/SD_output.txt"
python "$THROUGHPUT_PARSER" "$FIG_DIR/logs/SD_output.txt" "$FIG_DIR/parsed/SD_throughput.tsv"
python "$LATENCY_PARSER" "$FIG_DIR/logs/SD_output.txt" "$FIG_DIR/parsed/SD_latency.tsv"

# Generates Figure 9 PDF.
python "$FIGURE_SCRIPT"
SCRIPT_END=$(date +%s)
SCRIPT_ELAPSED=$((SCRIPT_END - SCRIPT_START))
echo "[Figure 9] Evaluation completed in $(format_duration "$SCRIPT_ELAPSED")."
echo "[Figure 9] Outputs are ready to compare against the reference files."
