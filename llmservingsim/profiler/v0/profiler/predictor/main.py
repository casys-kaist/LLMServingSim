import os
import argparse
from transformers import AutoConfig

from profiler.utils import *
from profiler.utils.logger import *
from .build_sklearn_predictor_and_pred import *

def parse_args():
    parser = argparse.ArgumentParser(description="Build attention predictors and prediction tables")
    # Model parameters
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name (e.g., meta-llama/Llama-3.1-8B).")
    parser.add_argument("--hardware", type=str,  required=True,
                        help="Hardware name for metadata logging.")
    # Tensor parallelism configuration
    parser.add_argument("--tp-size", type=str, default="1",
                        help="Comma-separated list of tensor parallel degrees (e.g., '1,2,4').")
    # Batch conditions
    parser.add_argument("--kv-granularity", type=int, default=64,
                        help="KV cache size grid granularity.")
    parser.add_argument("--chunk-granularity", type=int, default=32,
                        help="Prefill chunk size grid granularity.")
    parser.add_argument("--max-len", type=int, default=2048,
                        help="Maximum request length.")
    parser.add_argument("--max-batch", type=int, default=256,
                        help="Maximum batch size.")
    # Overhead
    parser.add_argument("--prefill-overhead", type=float, default=1.0)
    parser.add_argument("--decode-overhead", type=float, default=1.0)
    # Prediction parameters
    parser.add_argument("--rf-estimators", type=int, nargs="+", default=[250, 500, 750])
    parser.add_argument("--rf-max-depth", type=int, nargs="+", default=[8, 16, 24, 32])
    parser.add_argument("--rf-min-samples-split", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)

    return parser.parse_args()

def main():
    args = parse_args()

    # Convert tp_size string into list
    tp_sizes = [int(x.strip()) for x in args.tp_size.split(",")]

    # Load model config once per script run
    model_config = AutoConfig.from_pretrained(args.model)
    max_model_len = getattr(model_config, "max_position_embeddings", 2048)

    for tp_size in tp_sizes:
        if validate_tp_size(tp_size, model_config.num_attention_heads):
            log_warning(f"Skipping invalid TP degree {tp_size}.")
            continue
        
        input_csv = f"perf_models/{args.hardware}/{args.model}/tp{tp_size}/attention.csv"
        output_dir = f"perf_models/{args.hardware}/{args.model}/tp{tp_size}/predictions"
        os.makedirs(output_dir, exist_ok=True)

        rf_grid = {
            "n_estimators": args.rf_estimators,
            "max_depth": args.rf_max_depth,
            "min_samples_split": args.rf_min_samples_split,
        }

        log_info(f"Starting building attention predictor for hardware={args.hardware}, model={args.model}, tp_size={tp_size}")
        prefill_df, decode_df = load_and_split_attention_csv(input_csv)
        if args.hardware.lower().startswith("tpu"):
            pf_model = train_model("attn_prefill_predictor_model", prefill_df,
                                            ["kv_cache_size", "prefill_chunk_size"],
                                            "p50_ns",
                                            rf_grid, args.cv, args.n_jobs, output_dir)
        else:
            pf_model = train_model("attn_prefill_predictor_model", prefill_df,
                                            ["kv_cache_size", "prefill_chunk_size"],
                                            "time_stats.attn_prefill.median",
                                            rf_grid, args.cv, args.n_jobs, output_dir)
        log_success(f"[{args.hardware}/{args.model} TP={tp_size} PREFILL] Storing trained model to {output_dir}/attn_prefill_predictor_model.pkl")
        if args.hardware.lower().startswith("tpu"):
            dc_model = train_model("attn_decode_predictor_model", decode_df,
                                        ["batch_size", "kv_cache_size"],
                                        "p50_ns",
                                        rf_grid, args.cv, args.n_jobs, output_dir)
        else:
            dc_model = train_model("attn_decode_predictor_model", decode_df,
                                        ["batch_size", "kv_cache_size"],
                                        "time_stats.attn_decode.median",
                                        rf_grid, args.cv, args.n_jobs, output_dir)
        log_success(f"[{args.hardware}/{args.model} TP={tp_size} DECODE] Storing trained model to {output_dir}/attn_decode_predictor_model.pkl")
        log_info(f"Starting building prediction csv for hardware={args.hardware}, model={args.model}, tp_size={tp_size}")
        prefill_X, decode_X = build_grids(args.max_len, args.kv_granularity,
                                        args.chunk_granularity,
                                        max_model_len, args.max_batch)

        predict_and_save("attn_prefill", pf_model, prefill_X, output_dir, args.prefill_overhead)
        log_success(f"[{args.hardware}/{args.model} TP={tp_size} PREFILL] Storing prediction csv to {output_dir}/attn_prefill_predictions.csv")
        predict_and_save("attn_decode", dc_model, decode_X, output_dir, args.decode_overhead)
        log_success(f"[{args.hardware}/{args.model} TP={tp_size} DECODE] Storing prediction csv to {output_dir}/attn_decode_predictions.csv")

if __name__ == "__main__":
    main()