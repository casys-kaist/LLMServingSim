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
PARSER="evaluation/parser/parser_throughput.py"
FIGURE_SCRIPT="evaluation/figure_5/figure_5.py"
A6000_DIR="evaluation/figure_5/A6000"
H100_DIR="evaluation/figure_5/H100"
A6000_CONFIG_DIR="$A6000_DIR/config"
H100_CONFIG_DIR="$H100_DIR/config"
DENSE_DATASET="/dataset/sharegpt_req300_rate10_llama.jsonl"
A6000_MOE_DATASET="/dataset/sharegpt_req300_rate10_phi.jsonl"
H100_MOE_DATASET="/dataset/sharegpt_req300_rate10_mixtral.jsonl"
mkdir -p "$A6000_DIR/logs" "$A6000_DIR/results" "$A6000_DIR/parsed" \
    "$H100_DIR/logs" "$H100_DIR/results" "$H100_DIR/parsed"

# Generates Figure 5 PDF.
echo "[Figure 5] Starting evaluation."

###############################################
# Evaluation of LLMServingSim2.0 - Validation #
###############################################

#################### A6000 ####################
# MD (Multi-instane Dense)
python main.py --cluster-config "$A6000_CONFIG_DIR/MD_config.json" \
    --dataset "$DENSE_DATASET" --output "$A6000_DIR/results/MD_result.csv" \
    --fp 16 --block-size 16 --num-req 300 --log-interval 1 > "$A6000_DIR/logs/MD_output.txt"
python "$PARSER" "$A6000_DIR/logs/MD_output.txt" "$A6000_DIR/parsed/MD_throughput.tsv"

# SD+PC (Single Dense with Prefix Caching)
python main.py --cluster-config "$A6000_CONFIG_DIR/SD+PC_config.json" --enable-prefix-caching \
    --dataset "$DENSE_DATASET" --output "$A6000_DIR/results/SD+PC_result.csv" \
    --fp 16 --block-size 16 --num-req 300 --log-interval 1 > "$A6000_DIR/logs/SD+PC_output.txt"
python "$PARSER" "$A6000_DIR/logs/SD+PC_output.txt" "$A6000_DIR/parsed/SD+PC_throughput.tsv"

# PDD (Prefill/Decode Disaggregation Dense)
python main.py --cluster-config "$A6000_CONFIG_DIR/PDD_config.json" \
    --dataset "$DENSE_DATASET" --output "$A6000_DIR/results/PDD_result.csv" \
    --fp 16 --block-size 16 --num-req 300 --log-interval 1 > "$A6000_DIR/logs/PDD_output.txt"
python "$PARSER" "$A6000_DIR/logs/PDD_output.txt" "$A6000_DIR/parsed/PDD_throughput.tsv"

# SM (Single MoE)
python main.py --cluster-config "$A6000_CONFIG_DIR/SM_config.json" \
    --dataset "$A6000_MOE_DATASET" --output "$A6000_DIR/results/SM_result.csv" \
    --fp 16 --block-size 16 --num-req 300 --log-interval 1 > "$A6000_DIR/logs/SM_output.txt"
python "$PARSER" "$A6000_DIR/logs/SM_output.txt" "$A6000_DIR/parsed/SM_throughput.tsv"


##################### H100 ####################
# MD (Multi-instane Dense)
python main.py --cluster-config "$H100_CONFIG_DIR/MD_config.json" \
    --dataset "$DENSE_DATASET" --output "$H100_DIR/results/MD_result.csv" \
    --fp 16 --block-size 16 --num-req 300 --log-interval 1 > "$H100_DIR/logs/MD_output.txt"
python "$PARSER" "$H100_DIR/logs/MD_output.txt" "$H100_DIR/parsed/MD_throughput.tsv"

# SD+PC (Single Dense with Prefix Caching)
python main.py --cluster-config "$H100_CONFIG_DIR/SD+PC_config.json" --enable-prefix-caching \
    --dataset "$DENSE_DATASET" --output "$H100_DIR/results/SD+PC_result.csv" \
    --fp 16 --block-size 16 --num-req 300 --log-interval 1 > "$H100_DIR/logs/SD+PC_output.txt"
python "$PARSER" "$H100_DIR/logs/SD+PC_output.txt" "$H100_DIR/parsed/SD+PC_throughput.tsv"

# PDD (Prefill/Decode Disaggregation)
python main.py --cluster-config "$H100_CONFIG_DIR/PDD_config.json" \
    --dataset "$DENSE_DATASET" --output "$H100_DIR/results/PDD_result.csv" \
    --fp 16 --block-size 16 --num-req 300 --log-interval 1 > "$H100_DIR/logs/PDD_output.txt"
python "$PARSER" "$H100_DIR/logs/PDD_output.txt" "$H100_DIR/parsed/PDD_throughput.tsv"

# SM (Single MoE)
python main.py --cluster-config "$H100_CONFIG_DIR/SM_config.json" \
    --dataset "$H100_MOE_DATASET" --output "$H100_DIR/results/SM_result.csv" \
    --fp 16 --block-size 16 --num-req 300 --log-interval 1 > "$H100_DIR/logs/SM_output.txt"
python "$PARSER" "$H100_DIR/logs/SM_output.txt" "$H100_DIR/parsed/SM_throughput.tsv"

python "$FIGURE_SCRIPT"
SCRIPT_END=$(date +%s)
SCRIPT_ELAPSED=$((SCRIPT_END - SCRIPT_START))
echo "[Figure 5] Evaluation completed in $(format_duration "$SCRIPT_ELAPSED")."
echo "[Figure 5] Outputs are ready to compare against the reference files."
