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
FIG_DIR="evaluation/figure_7"
CONFIG_DIR="$FIG_DIR/config"
DATASET="/dataset/sharegpt_pulse_req50_n6_delay15_pc.jsonl"
PARSER="evaluation/parser/parser_memory.py"
FIGURE_SCRIPT="$FIG_DIR/figure_7.py"
mkdir -p "$FIG_DIR/logs" "$FIG_DIR/results" "$FIG_DIR/parsed"
echo "[Figure 7] Starting evaluation."

###############################################
#    Evaluation of LLMServingSim2.0 - Memory  #
###############################################

# SD+PC (Single Dense with Prefix Caching)
python main.py --cluster-config "$CONFIG_DIR/SD+PC_config.json" --enable-prefix-caching \
    --dataset "$DATASET" --output "$FIG_DIR/results/SD+PC_result.csv" \
    --fp 16 --block-size 16 --num-req 300 --log-interval 1 > "$FIG_DIR/logs/SD+PC_output.txt"
python "$PARSER" "$FIG_DIR/logs/SD+PC_output.txt" "$FIG_DIR/parsed/SD+PC.tsv"

# MD+PC+PS (Multi Dense with Prefix Caching and Sharing)
python main.py --cluster-config "$CONFIG_DIR/MD+PC+PS_config.json" --enable-prefix-caching --prefix-storage "CPU" --enable-prefix-sharing \
    --dataset "$DATASET" --output "$FIG_DIR/results/MD+PC+PS_result.csv" \
    --fp 16 --block-size 16 --num-req 300 --log-interval 1 > "$FIG_DIR/logs/MD+PC+PS_output.txt"
python "$PARSER" "$FIG_DIR/logs/MD+PC+PS_output.txt" "$FIG_DIR/parsed/MD+PC+PS.tsv"

# Generates Figure 7 PDF
python "$FIGURE_SCRIPT"
SCRIPT_END=$(date +%s)
SCRIPT_ELAPSED=$((SCRIPT_END - SCRIPT_START))
echo "[Figure 7] Evaluation completed in $(format_duration "$SCRIPT_ELAPSED")."
echo "[Figure 7] Outputs are ready to compare against the reference files."
