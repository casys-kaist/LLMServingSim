import torch

from collections import defaultdict
from tqdm import tqdm
import csv
import os
import gc
import argparse

from transformers import AutoConfig
from transformers.utils import logging
from transformers.cache_utils import DynamicCache

from profiler.common.timer_stats_store import TimerStatsStore
from profiler.utils import *
from profiler.utils.record_function_tracer import RecordFunctionTracer
from profiler.utils.logger import *


logging.set_verbosity_error()   # error only to avoid warnings from transformers

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Non-Attention Layer profiler")

    # Model parameters
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name (e.g., meta-llama/Llama-3.1-8B).")
    parser.add_argument("--hardware", type=str,  required=True,
                        help="Hardware name for metadata logging.")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="Number of transformer layers to profile.")
    # Tensor parallelism configuration
    parser.add_argument("--tp-size", type=str, default="1",
                        help="Comma-separated list of tensor parallel degrees (e.g., '1,2,4').")
    # Batch conditions
    parser.add_argument("--max-len", type=int, default=2048,
                        help="Maximum request length.")
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
    
    parser.add_argument("--append", action="store_true", help="Append to CSV instead of overwrite")

    parser.add_argument("--verbose", action="store_true")
    # Which stages to run
    parser.add_argument("--legacy", action="store_true", help="Use legacy attention with no prediction (not recommended)")

    return parser.parse_args()


def _create_past_key_values(config, kv_len, device):
    """
    Create a DynamicCache with preallocated KV tensors for a single TP rank.

    - tp_size is the logical tensor-parallel degree.
    - We allocate KV tensors with num_key_value_heads / tp_size heads, i.e.
      the per-rank KV-head count in a tensor-parallel setup.

    This matches the per-rank KV cache shape used in vLLM / Megatron-style TP:
        K, V: [batch, num_kv_heads_local, kv_len, head_dim]
    """
    num_layers = config.num_hidden_layers

    # Total KV heads in the (global) config
    num_kv_heads_total = config.num_key_value_heads // config.tp_size

    # Head dim is always based on total attention heads (not KV heads)
    head_dim = config.hidden_size // (config.num_attention_heads) # config has been already divided by tp_size

    # Choose dtype from config
    if getattr(config, "dtype", None) == torch.float16:
        dtype = torch.float16
    else:
        # Fallback; you can extend this if you want bfloat16, etc.
        dtype = torch.float32

    # Preallocate per-rank KV tensors:
    key_states = torch.zeros(
        (1, num_kv_heads_total, kv_len, head_dim),
        device=device,
        dtype=dtype,
    )
    value_states = torch.zeros(
        (1, num_kv_heads_total, kv_len, head_dim),
        device=device,
        dtype=dtype,
    )

    # Dummy input to satisfy rotary embedding forward
    dummy_x = torch.zeros((1, kv_len, head_dim), device=device, dtype=dtype)
    position_ids = torch.arange(kv_len, device=device).unsqueeze(0)  # shape: (1, kv_len)

    if "llama" in config.model_type:
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        rope = LlamaRotaryEmbedding(config, device=device)
        cos, sin = rope(dummy_x, position_ids)
    elif "mixtral" in config.model_type:
        from transformers.models.mixtral.modeling_mixtral import MixtralRotaryEmbedding
        rope = MixtralRotaryEmbedding(config, device=device)
        cos, sin = rope(dummy_x, position_ids)
    elif "phimoe" in config.model_type:
        from transformers.models.phimoe.modeling_phimoe import PhimoeRotaryEmbedding
        rope = PhimoeRotaryEmbedding(config)
        cos, sin = rope(dummy_x, kv_len)
    else:
        raise NotImplementedError("Only LLaMA, Mixtral, Phi-MoE models are supported in profiling. We will add more models soon.")

    cache = DynamicCache()
    for layer_idx in range(num_layers):
        cache.update(
            key_states,
            value_states,
            layer_idx,
            {
                "cos": cos,
                "sin": sin,
                "cache_position": position_ids,
            },
        )
    return cache

def run_profile(
    hardware="A6000",
    model_name="meta-llama/Llama-3.1-8B",
    num_layers=None,
    input_lengths=[128, 256, 512, 1024],
    is_prefill=True,
    tp_size=1,
    device="cuda",
    warmup=10,
    repeat=100,
    profile_method="record_function",
    csv_append=True,
    verbose=False
):

    config = AutoConfig.from_pretrained(model_name)
    original_num_layers = config.num_hidden_layers
    config.num_hidden_layers = num_layers
    config.dtype = torch.float16
    config.pad_token_id = 1
    config.tp_size = tp_size
    # Call singletone instance TimerStatsStore to set profile method
    timer_stats_store = TimerStatsStore(profile_method=profile_method)

    if 'llama' in config.model_type:
        from models.llama import LlamaForCausalLM
        model = LlamaForCausalLM(config)
    elif 'mixtral' in config.model_type:
        from models.mixtral import MixtralForCausalLM
        # If you want to collect router stats during profiling, turn this on
        config.collect_router_stats = True
        model = MixtralForCausalLM(config)
    elif 'phimoe' in config.model_type:
        from models.phimoe import PhimoeForCausalLM
        # If you want to collect router stats during profiling, turn this on
        config.collect_router_stats = False
        model = PhimoeForCausalLM(config)
    else:
        raise NotImplementedError("Only LLaMA, Mixtral, Phi-MoE models are supported in profiling. We will add more models soon.")
    
    model.eval()
    model.to(config.dtype)
    model.to(device)

    if is_prefill:
        kv_lengths = [0]
    else:
        kv_lengths = input_lengths # should run all possible input/kv combinations for decode
        raise log_warning(f"This deprecated decode profiling will profile {len(kv_lengths) * len(input_lengths)} configurations.")
    results = defaultdict(float)
   
    total_tasks = len(input_lengths) * len(kv_lengths)
    csv_rows = []
    log_info(f"Starting profiling for hardware={hardware}, model={model_name}, tp_size={tp_size}")
    for (input_len, kv_len) in tqdm([(l, k) for l in input_lengths for k in kv_lengths], total=total_tasks, desc="Profiling configs"):
        if input_len + kv_len > config.max_position_embeddings:
            continue  # Skip if input length exceeds max position embeddings
        if verbose:
            log_info(f"Running input={input_len}, kv={kv_len}, tp={tp_size}")

        input_ids = torch.randint(low=0, high=config.vocab_size // tp_size, size=(1, input_len), device=device)

        num_layers = config.num_hidden_layers
        past_key_values = _create_past_key_values(config, kv_len, device)

        # Warm-up phase
        for _ in range(warmup):
            past_key_values = _create_past_key_values(config, kv_len, device)
            with torch.no_grad():
                _ = model(input_ids, past_key_values=past_key_values, use_cache=True)

        torch.cuda.synchronize()
        timer_stats_store.clear_stats()

        if profile_method == ProfileMethod.RECORD_FUNCTION.value:
            
            trace_output_dir = f"perf_models/{hardware}/{model_name}/tp{tp_size}"
            record_function_tracer = RecordFunctionTracer(trace_output_dir)

            # Profiling phase
            with record_function_tracer:
                for _ in range(repeat):
                    past_key_values = _create_past_key_values(config, kv_len, device)
                    with torch.no_grad():
                        _ = model(input_ids, past_key_values=past_key_values, use_cache=True)
                
            torch.cuda.synchronize()
            time_stats = record_function_tracer.get_operation_time_stats()
            record_function_tracer.clean_up() # remove trace file after processing

        else:
            # Profiling phase
            for _ in range(repeat):
                past_key_values = _create_past_key_values(config, kv_len, device)
                with torch.no_grad():
                    _ = model(input_ids, past_key_values=past_key_values, use_cache=True)
            
            torch.cuda.synchronize()
            time_stats = timer_stats_store.get_stats()


        profile_keys = ["embedding", "input_layernorm", "q_proj", "k_proj", "v_proj", "rope", "attn", "o_proj", "post_layernorm", "gate_proj", "up_proj", "act_fn", "down_proj", "final_layernorm", "lm_head"]
        if 'mixtral' in config.model_type or 'phimoe' in config.model_type:
            profile_keys += ["gate", "expert.w1", "expert.w2", "expert.w3"]

        for key, value in time_stats.items():
            if key in profile_keys:
                if verbose:
                    log_info(f"input={input_len}, kv={kv_len}, tp={tp_size}, layer={key}, time={value['median']*1000:.2f} us")
                results_key = (input_len, kv_len, key)
                results[results_key] = value['median']*1000
                csv_rows.append({
                    "layer_name": key,
                    "input": input_len,
                    "kv_cache": kv_len,
                    "tp_size": tp_size,
                    "latency(ns)": int(value['median']* 1000_000)  # convert ms to ns
                })
            else:
                log_warning(f"Skipping layer={key} not in profile keys.")

        embedding = results.get((input_len, kv_len, "embedding"), 0.0)
        final_norm = results.get((input_len, kv_len, "final_layernorm"), 0.0)
        lm_head = results.get((input_len, kv_len, "lm_head"), 0.0)
        if 'llama' in config.model_type:
            block_components = ["input_layernorm", "q_proj", "k_proj", "v_proj", "rope", "attn", "o_proj", "post_layernorm", "gate_proj", "up_proj", "act_fn", "down_proj"]
        elif 'mixtral' in config.model_type or 'phimoe' in config.model_type:
            block_components = ["input_layernorm", "q_proj", "k_proj", "v_proj", "rope", "attn", "o_proj", "post_layernorm", "gate"]       
        else:
            raise NotImplementedError("Only LLaMA, Mixtral, Phi-MoE models are supported in profiling. We will add more models soon.")
        
        per_block_time = sum(results.get((input_len, kv_len, comp), 0.0) for comp in block_components)

        # Runs experts sequentially in huggungface implementation
        if 'mixtral' in config.model_type or 'phimoe' in config.model_type:
            moe_components = ["expert.w1", "expert.w2", "expert.w3", "act_fn"]
            n_tok = max(input_len // config.num_local_experts // tp_size, 1)
            for moe_comp in moe_components:
                per_block_time += results.get((n_tok, kv_len, moe_comp), 0.0) * (config.num_local_experts // tp_size)

        full_latency_estimate = embedding + final_norm + lm_head + per_block_time * original_num_layers

        if verbose:
            log_info(f"Estimated latency: {(full_latency_estimate / 1000):.2f} ms")

    output_path = f"perf_models/{hardware}/{model_name}/tp{tp_size}/layers.csv"
    if csv_append:
        mode = "a"
    else:
        mode = "w"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["layer_name", "input", "kv_cache", "tp_size", "latency(ns)"])
        if not csv_append:
            writer.writeheader()
        writer.writerows(csv_rows)
    log_success(f"[{hardware}/{model_name} TP={tp_size}] Writing profiled results to {output_path}")

def main():

    args = parse_args()

    # Convert tp_size string into list
    tp_sizes = [int(x.strip()) for x in args.tp_size.split(",")]

    # Load model config once per script run
    model_config = AutoConfig.from_pretrained(args.model)

    for tp_size in tp_sizes:
        if validate_tp_size(tp_size, model_config.num_attention_heads):
            log_warning(f"Skipping invalid TP degree {tp_size}.")
            continue

        # ---------- Prefill sweep ----------
        run_profile(
            hardware=args.hardware,
            model_name=args.model,
            input_lengths=range(1, args.max_len + 1),
            is_prefill=True,
            num_layers=args.num_layers,
            tp_size=tp_size,
            device=args.device,
            warmup=args.warmup,
            repeat=args.repeat,
            profile_method=args.profile_method,
            csv_append=False, # prefill first
            verbose=args.verbose,
        )
        torch.cuda.empty_cache()
        gc.collect()

        # ---------- Decode sweep (legacy) ----------
        if args.legacy:
            log_warning(
                "Deprecated: running legacy profiler. "
                "We recommend using attention prediction instead."
            )
            run_profile(
                hardware=args.hardware,
                model_name=args.model,
                input_lengths=range(1, args.max_len + 1),
                is_prefill=False,
                num_layers=args.num_layers,
                tp_size=tp_size,
                device=args.device,
                warmup=args.warmup,
                repeat=args.repeat,
                profile_method=args.profile_method,
                csv_append=True, # append decode results
                verbose=args.verbose,
            )
            torch.cuda.empty_cache()
            gc.collect()

    
if __name__ == "__main__":
    main()