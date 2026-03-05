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
FIG_DIR="evaluation/figure_6"
CONFIG_DIR="$FIG_DIR/config"
DATASET="/dataset/sharegpt_pulse_req10_n3_delay60.jsonl"
PARSER="evaluation/parser/parser_power.py"
COMPONENT_PARSER="evaluation/parser/parser_component_power.py"
FIGURE_SCRIPT="$FIG_DIR/figure_6.py"
mkdir -p "$FIG_DIR/logs" "$FIG_DIR/results" "$FIG_DIR/parsed"
echo "[Figure 6] Starting evaluation."

###############################################
#    Evaluation of LLMServingSim2.0 - Power   #
###############################################

# TP 1
python main.py --cluster-config "$CONFIG_DIR/power_config_tp1.json" \
    --dataset "$DATASET" --output "$FIG_DIR/results/power_result_tp1.csv" \
    --fp 16 --block-size 16 --num-req 30 --log-interval 1  > "$FIG_DIR/logs/power_output_tp1.txt"
python "$PARSER" "$FIG_DIR/logs/power_output_tp1.txt" "$FIG_DIR/parsed/power_tp1.tsv"

# TP 2
python main.py --cluster-config "$CONFIG_DIR/power_config_tp2.json" \
    --dataset "$DATASET" --output "$FIG_DIR/results/power_result_tp2.csv" \
    --fp 16 --block-size 16 --num-req 30 --log-interval 1  > "$FIG_DIR/logs/power_output_tp2.txt"
python "$PARSER" "$FIG_DIR/logs/power_output_tp2.txt" "$FIG_DIR/parsed/power_tp2.tsv"

python "$COMPONENT_PARSER" \
    --labels "TP 1" "TP 2" \
    -o "$FIG_DIR/parsed/component_energy.tsv" \
    "$FIG_DIR/logs/power_output_tp1.txt" \
    "$FIG_DIR/logs/power_output_tp2.txt"

# Generates Figure 6a, 6b, and 6c PDFs.
python "$FIGURE_SCRIPT"
SCRIPT_END=$(date +%s)
SCRIPT_ELAPSED=$((SCRIPT_END - SCRIPT_START))
echo "[Figure 6] Evaluation completed in $(format_duration "$SCRIPT_ELAPSED")."
echo "[Figure 6] Outputs are ready to compare against the reference files."
