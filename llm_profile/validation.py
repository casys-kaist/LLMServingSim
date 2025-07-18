import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.utils import logging
from collections import defaultdict
import csv
import os

logging.set_verbosity_error()

def measure_generation_latency(
    model,
    tokenizer,
    input_length=10,
    output_length=5,
    num_warmup=1,
    device=None,
    verbose=False
):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dummy input
    dummy_text = "Hello " * input_length
    inputs = tokenizer(dummy_text, return_tensors="pt", truncation=True, max_length=input_length).to(device)

    # Warm-up
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=output_length,
                do_sample=False,
                use_cache=True
            )

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=output_length,
            do_sample=False,
            use_cache=True
        )

    torch.cuda.synchronize()
    end = time.time()

    total_latency_ms = (end - start) * 1000

    if verbose:
        print(f"Measured latency: {total_latency_ms:.2f} ms")

    return total_latency_ms

def estimate_total_latency(
    hardware="RTX3090",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    num_layers=32,
    input_length=10,
    output_length=5,
    verbose=False
):
    # Load latency data from CSV
    latency_db = defaultdict(dict)  # (input, kv) -> {block_name -> latency_ns}
    csv_path = f"{hardware}.csv"
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            input_size = int(row["input"])
            kv_size = int(row["kv_cache"])
            block = row["layer_name"]
            latency_ns = int(row["latency(ns)"])
            latency_db[(input_size, kv_size)][block] = latency_ns

    # === Select block components by model type ===
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
    else:  # OPT
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

    total_latency_ns = 0

    # --- Prefill phase ---
    prefill_key = (input_length, 0)
    if prefill_key not in latency_db:
        raise ValueError(f"Missing latency data for input={input_length}, kv=0")

    block_latencies = latency_db[prefill_key]
    block_sum = sum(block_latencies.get(comp, 0) for comp in block_components)
    total_latency_ns += num_layers * block_sum

    # Add one-time components
    for comp in ["embedding", "final_layernorm", "lm_head"]:
        total_latency_ns += block_latencies.get(comp, 0)

    # --- Decode phase ---
    for i in range(output_length - 1):
        kv = input_length + i
        decode_key = (1, kv)
        if decode_key not in latency_db:
            raise ValueError(f"Missing latency data for input=1, kv={kv}")
        block_latencies = latency_db[decode_key]
        block_sum = sum(block_latencies.get(comp, 0) for comp in block_components)
        total_latency_ns += num_layers * block_sum
        total_latency_ns += block_latencies.get("lm_head", 0)

    total_latency_ms = total_latency_ns / 1e6  # convert ns to ms

    if verbose:
        print(f"Estimated latency: {total_latency_ms:.2f} ms")

    return total_latency_ms

def compute_average_scaling_factor(
    hardware="RTX3090",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    num_layers=None,
    input_lengths=[10, 20, 30],
    output_lengths=[2, 4, 6],
    num_trials=3,
    device=None,
    verbose=False
):
    scaling_factors = []
        
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model.eval()

    # Reduce number of layers if specified
    if num_layers is not None:
        if 'llama' in model_name.lower():
            model.model.layers = model.model.layers[:num_layers]
        else:
            model.model.decoder.layers = model.model.decoder.layers[:num_layers]
        model.config.num_hidden_layers = num_layers
    else:
        num_layers = model.config.num_hidden_layers
        
    model.to(device)

    for input_len in input_lengths:
        for output_len in output_lengths:
            measured_latencies = []
            estimated_latencies = []

            for trial in range(num_trials):
                # real latency
                try:
                    measured = measure_generation_latency(
                        model,
                        tokenizer,
                        input_length=input_len,
                        output_length=output_len,
                        verbose=verbose
                    )
                except ValueError as e:
                    if verbose:
                        print(f"Skipping ({input_len}, {output_len}): {e}")
                    continue
                # estimated latency
                try:
                    estimated = estimate_total_latency(
                        hardware=hardware,
                        model_name=model_name,
                        num_layers=num_layers,
                        input_length=input_len,
                        output_length=output_len,
                        verbose=verbose
                    )
                except ValueError as e:
                    if verbose:
                        print(f"Skipping ({input_len}, {output_len}): {e}")
                    continue

                measured_latencies.append(measured)
                estimated_latencies.append(estimated)

                if verbose:
                    print(f"[input={input_len}, output={output_len}] Measured: {measured:.2f} ms, Estimated: {estimated:.2f} ms, Factor: {measured/estimated:.3f}")

            if len(measured_latencies) > 0:
                avg_measured = sum(measured_latencies) / len(measured_latencies)
                avg_estimated = sum(estimated_latencies) / len(estimated_latencies)
                scaling_factor = avg_measured / avg_estimated if avg_estimated > 0 else float('inf')
                scaling_factors.append(scaling_factor)

    if len(scaling_factors) == 0:
        raise RuntimeError("No valid data points to compute scaling factor.")

    avg_scaling = sum(scaling_factors) / len(scaling_factors)

    if verbose:
        print(f"\nAverage Scaling Factor (measured / estimated): {avg_scaling:.3f}")

    return avg_scaling


def apply_scaling_to_latency_csv(
    hardware="RTX3090",
    scaling_factor=1,
    output_path=None,
    overwrite=False):

    csv_path = f"{hardware}.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found")

    if output_path is None:
        base, ext = os.path.splitext(csv_path)
        output_path = f"{base}_scaled{ext}"

    scaled_rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames.copy()

        # If overwriting latency, don't add extra column
        if not overwrite and "scaled_latency(ns)" not in fieldnames:
            fieldnames.append("scaled_latency(ns)")

        for row in reader:
            original_latency = int(row["latency(ns)"])
            scaled_latency = int(original_latency * scaling_factor)
            if overwrite:
                row["latency(ns)"] = scaled_latency
            else:
                row["scaled_latency(ns)"] = scaled_latency
            scaled_rows.append(row)

    with open(output_path, mode="w", newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scaled_rows)

    print(f"Scaled CSV written to: {output_path}")
    return output_path