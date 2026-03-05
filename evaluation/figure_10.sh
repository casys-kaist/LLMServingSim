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
COMPONENT_PARSER="evaluation/parser/parser_component_power.py"
ENERGY_PER_TOKEN_PARSER="evaluation/parser/parser_energy_per_token.py"
FIG_DIR="evaluation/figure_10"
CONFIG_DIR="$FIG_DIR/config"
DATASET="/dataset/fixed_in128_out512_req256_rate10.jsonl"
FIGURE_SCRIPT="$FIG_DIR/figure_10.py"
mkdir -p "$FIG_DIR/logs" "$FIG_DIR/results" "$FIG_DIR/parsed"
echo "[Figure 10] Starting evaluation."

###############################################
#     Evaluation of LLMServingSim2.0 - PIM    #
###############################################

# only-gpu
python main.py --cluster-config "$CONFIG_DIR/gpu_only_config.json" \
    --dataset "$DATASET" --output "$FIG_DIR/results/gpu_only_b256_result.csv" \
    --fp 16 --block-size 16 --num-req 256 --log-interval 1  > "$FIG_DIR/logs/gpu_only_b256_output.txt"
python "$THROUGHPUT_PARSER" "$FIG_DIR/logs/gpu_only_b256_output.txt" "$FIG_DIR/parsed/gpu_only_b256_throughput.tsv"

# with naive pim
python main.py --cluster-config "$CONFIG_DIR/pim_config.json" --enable-attn-offloading \
    --dataset "$DATASET" --output "$FIG_DIR/results/pim_b256_result.csv" \
    --fp 16 --block-size 16 --num-req 256 --log-interval 1 > "$FIG_DIR/logs/pim_b256_output.txt"
python "$THROUGHPUT_PARSER" "$FIG_DIR/logs/pim_b256_output.txt" "$FIG_DIR/parsed/pim_b256_throughput.tsv"

# with pim & sub-batch interleaving
python main.py --cluster-config "$CONFIG_DIR/pim_config.json" --enable-attn-offloading --enable-sub-batch-interleaving \
    --dataset "$DATASET" --output "$FIG_DIR/results/pim_sbi_b256_result.csv" \
    --fp 16 --block-size 16 --num-req 256 --log-interval 1  > "$FIG_DIR/logs/pim_sbi_b256_output.txt"
python "$THROUGHPUT_PARSER" "$FIG_DIR/logs/pim_sbi_b256_output.txt" "$FIG_DIR/parsed/pim_sbi_b256_throughput.tsv"

python "$COMPONENT_PARSER" \
    --labels "gpu-only" "pim" "pim-sbi" \
    -o "$FIG_DIR/parsed/component_energy.tsv" \
    "$FIG_DIR/logs/gpu_only_b256_output.txt" \
    "$FIG_DIR/logs/pim_b256_output.txt" \
    "$FIG_DIR/logs/pim_sbi_b256_output.txt"

python "$ENERGY_PER_TOKEN_PARSER" \
    --labels "gpu-only" "pim" "pim-sbi" \
    -o "$FIG_DIR/parsed/energy_per_token.tsv" \
    "$FIG_DIR/logs/gpu_only_b256_output.txt" \
    "$FIG_DIR/logs/pim_b256_output.txt" \
    "$FIG_DIR/logs/pim_sbi_b256_output.txt"

# Generates Figure 10 PDF.
python "$FIGURE_SCRIPT"
SCRIPT_END=$(date +%s)
SCRIPT_ELAPSED=$((SCRIPT_END - SCRIPT_START))
echo "[Figure 10] Evaluation completed in $(format_duration "$SCRIPT_ELAPSED")."
echo "[Figure 10] Outputs are ready to compare against the reference files."
