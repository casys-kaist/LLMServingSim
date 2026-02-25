import csv
import argparse
import torch
import os
from tqdm.auto import tqdm
import pandas as pd
from transformers import AutoConfig

from profiler.utils import *
from profiler.utils.logger import *
from .batch_sampling import *
from .attention_profiler import profile_flash_attention


def parse_args():
    parser = argparse.ArgumentParser(description="FlashAttention Profiling")

    # Model parameters
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name (e.g., meta-llama/Llama-3.1-8B).")
    parser.add_argument("--hardware", type=str,  required=True,
                        help="Hardware name for metadata logging.")

    # Tensor parallelism configuration
    parser.add_argument("--tp-size", type=str, default="1",
                        help="Comma-separated list of tensor parallel degrees (e.g., '1,2,4').")

    # Batch sampling conditions
    parser.add_argument("--max-len", type=int, default=2048,
                        help="Maximum request length.")
    parser.add_argument("--min-batch-size", type=int, default=1,
                        help="Maximum decode batch size")
    parser.add_argument("--max-batch-size",type=int, default=256,
                        help="Maximum decode batch size")
    parser.add_argument("--profile-only-decode", action="store_true",
                        help="Only profile the decode")
    parser.add_argument("--profile-only-prefill", action="store_true",
                        help="Only profile the prefill")
    parser.add_argument("--block-size", type=int, default=16,
                        help="Block size for paged attention")
    
    # Profiling parameters
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup iterations.")
    parser.add_argument("--repeat", type=int, default=30,
                        help="Number of repeated profiling steps.")
    # Profiling method
    parser.add_argument("--profile-method", default="record_function", choices=[e.value for e in ProfileMethod],
        help="Method to use for measuring time taken by operations (default: %(default)s)")

    # Device selection
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' or 'cpu'. FlashAttention requires GPU.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Convert tp_size string into list
    tp_sizes = [int(x.strip()) for x in args.tp_size.split(",")]

    # Load model config once per script run
    model_config = AutoConfig.from_pretrained(args.model)
    model_config.dtype = torch.float16
    
    for tp_size in tp_sizes:
        if validate_tp_size(tp_size, model_config.num_attention_heads):
            log_warning(f"Skipping invalid TP degree {tp_size}.")
            continue
            
        input_combinations = get_attention_input_combinations(
            max_seq_len=args.max_len,
            max_model_len=getattr(model_config, "max_position_embeddings", 2048),
            min_batch_size=args.min_batch_size,
            max_batch_size=args.max_batch_size,
            profile_only_prefill=args.profile_only_prefill,
            profile_only_decode=args.profile_only_decode,
        )
        
        max_num_blocks = get_max_num_blocks(
            model_config=model_config,
            tensor_parallel_size=tp_size,
            block_size=args.block_size,
            dtype=torch.float16,
            memory_utilization=0.9,
            max_pipeline_parallel_size=1
        )

        total_combos = list(
            filter(
                lambda input_combination: input_combination.is_under_memory_limit(
                    max_num_blocks * args.block_size
                ),
                input_combinations,
            )
        )

        output_path = f"perf_models/{args.hardware}/{args.model}/tp{tp_size}/attention.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        all_results = []
        log_info(f"Starting Attention profiling for hardware={args.hardware}, model={args.model}, tp_size={tp_size}")
        with tqdm(total=len(total_combos), desc="Profiling Attention", unit="config") as pbar:
            
            for attention_input in total_combos:
                result_dict = profile_flash_attention(
                    hardware=args.hardware,
                    model_name=args.model,
                    model_config=model_config,
                    tp_size=tp_size,
                    attention_input=attention_input,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    profile_method=args.profile_method,
                    device=args.device
                )
                all_results.append(result_dict)
                pbar.update(1)

        # filter all none results
        df = pd.DataFrame(list(filter(None, all_results)))
        time_stats_df = pd.json_normalize(df["time_stats"])
        time_stats_df.columns = [f"time_stats.{c}" for c in time_stats_df.columns]

        df_flat = pd.concat(
            [time_stats_df, df.drop(columns=["time_stats"])],
            axis=1,
        )

        df_flat.to_csv(output_path, index=False)
        

        log_success(f"Profiling completed. Results saved to: {output_path}")

if __name__ == "__main__":
    main()
