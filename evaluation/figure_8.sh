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
FIG_DIR="evaluation/figure_8"
CONFIG_DIR="$FIG_DIR/config"
DATASET_DENSE="/dataset/sharegpt_req300_rate10_llama.jsonl"
DATASET_MOE="/dataset/sharegpt_req300_rate10_mixtral.jsonl"
LATENCY_PARSER="evaluation/parser/parser_latency.py"
SIM_TIME_PARSER="evaluation/parser/parser_sim_time.py"
FIGURE_SCRIPT="$FIG_DIR/figure_8.py"
mkdir -p "$FIG_DIR/logs" "$FIG_DIR/results" "$FIG_DIR/parsed"
echo "[Figure 8] Starting evaluation."

###############################################
# Evaluation of LLMServingSim2.0 - Prior Work #
###############################################

# SD (Single Dense)
python main.py --cluster-config "$CONFIG_DIR/SD_config.json" \
    --dataset "$DATASET_DENSE" --output "$FIG_DIR/results/SD_result.csv" \
    --fp 16 --block-size 16 --num-req 300 --log-interval 1 > "$FIG_DIR/logs/SD_output.txt"
python "$LATENCY_PARSER" "$FIG_DIR/logs/SD_output.txt" "$FIG_DIR/parsed/SD_latency.tsv"
python "$SIM_TIME_PARSER" "$FIG_DIR/logs/SD_output.txt" "$FIG_DIR/parsed/SD_sim_time.tsv"

# MD (Multi-instane Dense)
python main.py --cluster-config "$CONFIG_DIR/MD_config.json" \
    --dataset "$DATASET_DENSE" --output "$FIG_DIR/results/MD_result.csv" \
    --fp 16 --block-size 16 --num-req 300 --log-interval 1 > "$FIG_DIR/logs/MD_output.txt"
python "$LATENCY_PARSER" "$FIG_DIR/logs/MD_output.txt" "$FIG_DIR/parsed/MD_latency.tsv"
python "$SIM_TIME_PARSER" "$FIG_DIR/logs/MD_output.txt" "$FIG_DIR/parsed/MD_sim_time.tsv"

# PD (Prefill/Decode Disaggregation)
python main.py --cluster-config "$CONFIG_DIR/PDD_config.json" \
    --dataset "$DATASET_DENSE" --output "$FIG_DIR/results/PDD_result.csv" \
    --fp 16 --block-size 16 --num-req 300 --log-interval 1 > "$FIG_DIR/logs/PDD_output.txt"
python "$LATENCY_PARSER" "$FIG_DIR/logs/PDD_output.txt" "$FIG_DIR/parsed/PDD_latency.tsv"
python "$SIM_TIME_PARSER" "$FIG_DIR/logs/PDD_output.txt" "$FIG_DIR/parsed/PDD_sim_time.tsv"

# SM (Single MoE)
python main.py --cluster-config "$CONFIG_DIR/SM_config.json" \
    --dataset "$DATASET_MOE" --output "$FIG_DIR/results/SM_result.csv" \
    --fp 16 --block-size 16 --num-req 300 --log-interval 1 > "$FIG_DIR/logs/SM_output.txt"
python "$LATENCY_PARSER" "$FIG_DIR/logs/SM_output.txt" "$FIG_DIR/parsed/SM_latency.tsv"
python "$SIM_TIME_PARSER" "$FIG_DIR/logs/SM_output.txt" "$FIG_DIR/parsed/SM_sim_time.tsv"

# Generates Figure 8a and 8b PDFs.
python "$FIGURE_SCRIPT"
SCRIPT_END=$(date +%s)
SCRIPT_ELAPSED=$((SCRIPT_END - SCRIPT_START))
echo "[Figure 8] Evaluation completed in $(format_duration "$SCRIPT_ELAPSED")."
echo "[Figure 8] Outputs are ready to compare against the reference files."
