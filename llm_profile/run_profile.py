import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.profiler import profile, ProfilerActivity
from patch_model import *
from validation import *
from collections import defaultdict
from tqdm import tqdm
import csv
import os
import gc

def run_profile(
    hardware="RTX3090",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    num_layers=None,
    input_lengths=[128, 256, 512, 1024],
    kv_cache_lengths=[0, 128, 512, 1024],
    device="cuda",
    num_warm_up=10,
    num_profile=100,
    csv_append=True,
    verbose=False
):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.eval()

    # Reduce model layers if specified
    if num_layers is not None:
        if 'llama' in model_name.lower():
            model.model.layers = model.model.layers[:num_layers]
        else:
            model.model.decoder.layers = model.model.decoder.layers[:num_layers]
        model.config.num_hidden_layers = num_layers

    model.to(device)

    patch_model(model, model.config)

    results = defaultdict(float)
    total_tasks = len(input_lengths) * len(kv_cache_lengths)
    csv_rows = []

    for (input_len, kv_len) in tqdm([(l, k) for l in input_lengths for k in kv_cache_lengths], total=total_tasks, desc="Profiling configs"):
        if input_len + kv_len > model.config.max_position_embeddings:
            continue  # Skip if input length exceeds max position embeddings
        if verbose:
            print(f"\nRunning input_len={input_len}, kv_len={kv_len}",flush=True)

        input_ids = torch.randint(low=0, high=tokenizer.vocab_size, size=(1, input_len), device=device)

        num_layers = model.config.num_hidden_layers
        kv_head = model.config.num_key_value_heads if hasattr(model.config, 'num_key_value_heads') else model.config.num_attention_heads
        head_dim = model.config.hidden_size // model.config.num_attention_heads

        if 'llama' in model_name.lower():
            past_key_values = create_llama_past_key_values(model.config, kv_len, device)
        else:
            # OPT-compatible tuple
            past_key_values = tuple([
                torch.zeros((1, kv_head, kv_len, head_dim), device=device),
                torch.zeros((1, kv_head, kv_len, head_dim), device=device)
            ])
        
        # Warm-up phase
        for _ in range(num_warm_up):
            if 'llama' in model_name.lower():
                past_key_values = create_llama_past_key_values(model.config, kv_len, device)
            with torch.no_grad():
                _ = model(input_ids, past_key_values=past_key_values, use_cache=True)

        # Profiling phase
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
            with_modules=False,
        ) as prof:
            for _ in range(num_profile):
                if 'llama' in model_name.lower():
                    past_key_values = create_llama_past_key_values(model.config, kv_len, device)
                with torch.no_grad():
                    _ = model(input_ids, past_key_values=past_key_values, use_cache=True)
                prof.step()


        for evt in prof.key_averages():
            if evt.key.startswith("aten::"):
                continue
            if any(kw in evt.key for kw in ["embedding", "layernorm", "proj", "fc", "matmul", "softmax", "act_fn", "rope", "attn", "lm_head"]):
                time_us = evt.cpu_time_total / evt.count
                if time_us > 0:
                    if verbose:
                        print(f"input={input_len}, kv={kv_len}, block={evt.key}, time={time_us:.2f} us")
                    key = (input_len, kv_len, evt.key)
                    results[key] = time_us
                    csv_rows.append({
                        "hardware": hardware,
                        "model": model_name,
                        "layer_name": evt.key,
                        "input": input_len,
                        "kv_cache": kv_len,
                        "latency(ns)": int(time_us * 1000)  # convert us to ns
                    })
        

        embedding = results.get((input_len, kv_len, "embedding"), 0.0)
        final_norm = results.get((input_len, kv_len, "final_layernorm"), 0.0)
        lm_head = results.get((input_len, kv_len, "lm_head"), 0.0)
        if 'llama' in model_name.lower():
            block_components = [
                "input_layernorm",
                "q_proj",
                "k_proj",
                "v_proj",
                "rope",
                "attn",
                "o_proj",
                "post_layernorm",
                "gate_proj",
                "up_proj",
                "act_fn",
                "down_proj"
            ]
        else:  # OPT/TransformerDecoder
            block_components = [
                "input_layernorm",
                "q_proj",
                "k_proj",
                "v_proj",
                "qk_matmul",
                "softmax",
                "sv_matmul",
                "o_proj",
                "post_layernorm",
                "fc1",
                "act_fn",
                "fc2"
            ]
        per_block_time = sum(results.get((input_len, kv_len, comp), 0.0) for comp in block_components)
        full_latency_estimate = embedding + final_norm + lm_head + per_block_time * num_layers

        if verbose:
            print(f"Estimated latency: {(full_latency_estimate / 1000):.2f} ms")

    output_path = f"{hardware}.csv"
    if csv_append:
        mode = "a"
    else:
        mode = "w"
    file_exists = os.path.exists(output_path)
    with open(output_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["hardware", "model", "layer_name", "input", "kv_cache", "latency(ns)"])
        if not file_exists or not csv_append:
            writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Writing profiled results to {output_path}")

if __name__ == "__main__":
    # Add your Hugging Face token if needed
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "<your_token>")

    # For prefill phase: input=[1,2,3 ...], kv=0
    run_profile(hardware="RTX3090", model_name="meta-llama/Llama-3.1-8B-Instruct",
            input_lengths=range(1, 2049),
            kv_cache_lengths=range(0, 1),
            num_layers=1,
            num_warm_up=10, num_profile=100, csv_append=False, verbose=True)
    
    # Prevent OOM
    torch.cuda.empty_cache()
    gc.collect()

    # For decode phase: input=1, kv=[0,1,2, ...]
    run_profile(hardware="RTX3090", model_name="meta-llama/Llama-3.1-8B-Instruct",
            input_lengths=range(1, 2),
            kv_cache_lengths=range(0, 2048),
            num_layers=1,
            num_warm_up=10, num_profile=100, csv_append=True, verbose=True)
    
    # Prevent OOM
    torch.cuda.empty_cache()
    gc.collect()

    # Start validating the result
    # Reducing the number of layers for profiling often leads to artificially slower per-layer latency due to GPU execution behavior. 
    # This is because GPUs are not fully utilized when the model is shallow. 
    # To compensate for this under-utilization, we measure a scaling factor and apply it to the estimated latency. 
    # As the number of profiled layers approaches the actual model depth, the scaling factor converges to 1.
    scaling_factor = compute_average_scaling_factor(hardware="RTX3090", model_name="meta-llama/Llama-3.1-8B-Instruct",
            input_lengths=range(128, 1025, 128),
            output_lengths=range(128, 1025, 128),
            num_trials=3,
            verbose=True)
    
    # Apply to the csv file
    apply_scaling_to_latency_csv(hardware="RTX3090", scaling_factor=scaling_factor, output_path=None, overwrite=True)