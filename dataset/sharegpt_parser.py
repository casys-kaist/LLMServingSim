from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import random
import json
import os

# --------- HF token ----------
HF_TOKEN = "<your token>"  # Replace with your Hugging Face token
os.environ["HF_TOKEN"] = HF_TOKEN

# --------- Repro ----------
random.seed(42)
np.random.seed(42)

# --------- Config ----------
dataset_name = "shibing624/sharegpt_gpt4"
tokenizer_name = "meta-llama/Llama-3.1-8B"
# tokenizer_name = "microsoft/Phi-mini-MoE-instruct"
# tokenizer_name = "mistralai/Mixtral-8x7B-v0.1"
request_per_sec = 10
room_for_decode = 0 # leave room for decode input tokens (200 for only moe models)
max_input_length = 2048 - room_for_decode 
max_output_length = 2048
max_kv_length = 2048 
max_sessions = 1000 # 100, 300
max_requests = 512 # 300
output_path = f"sharegpt_req{max_requests}_rate{request_per_sec}.jsonl"
first_arrival_time = 0 # first arrival time in seconds

fix_len = True  # if True, use fixed length inputs/outputs
if fix_len:
    fix_input_length = 128
    fix_output_length = 512
    output_path = f"fixed_in{fix_input_length}_out{fix_output_length}_req{max_requests}_rate{request_per_sec}.jsonl"

pulse = False  # if True, generate pulse requests
use_poisson_in_pulse = False # if True, use Poisson process also in pulse mode
if pulse:
    num_req_pulse = 10 # 50
    first_arrival_time = 1 # give some delay before starting
    delay_seconds = 60 # 15, delay after sending the pulse
    output_path = f"sharegpt_pulse_req{num_req_pulse}_n{max_requests//num_req_pulse}_delay{delay_seconds}.jsonl"

# --------- Load ----------
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
dataset = load_dataset(dataset_name, split="train").select(range(max_sessions))


# --------- Parse sessions ----------
if not fix_len:
    sessions = []
    for row in tqdm(dataset, desc="Parsing dataset into sessions"):
        conversations = row["conversations"]
        context = ""
        turns = []

        for i in range(0, len(conversations) - 1, 2):
            if conversations[i]["from"] == "human" and conversations[i+1]["from"] == "gpt":
                prompt = conversations[i]["value"]
                response = conversations[i+1]["value"]

                if context:
                    input_text = context.strip() + " " + prompt.strip()
                else:
                    input_text = prompt.strip()

                output_text = response.strip()
                turns.append((input_text, output_text))

                current_turn = prompt.strip() + " " + response.strip()
                
                if context:
                    context += " " + current_turn
                else:
                    context = current_turn

        if turns:
            sessions.append(turns)
    session_indices = [0] * len(sessions) 


time_offset_ns = first_arrival_time * 1_000_000_000
request_count = 0


# --------- Generate requests & write JSONL ----------
with open(output_path, "w", encoding="utf-8") as fout:
    while request_count < max_requests:
        if not fix_len:
            available_sessions = [i for i, idx in enumerate(session_indices) if idx < len(sessions[i])]
            if not available_sessions:
                break
            
            sid = random.choice(available_sessions)
            input_text, output_text = sessions[sid][session_indices[sid]]
            session_indices[sid] += 1

            input_tokens = tokenizer(input_text, add_special_tokens=False)["input_ids"]
            output_tokens = tokenizer(output_text, add_special_tokens=False)["input_ids"]

            if len(input_tokens) > max_input_length or len(output_tokens) > max_output_length or len(input_tokens) + len(output_tokens) > max_kv_length:
                continue
        else:
            # fixed length inputs/outputs
            vocab_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 32000
            input_tokens = [random.randint(0, vocab_size - 1) for _ in range(fix_input_length)]  # random token ids
            output_tokens = [random.randint(0, vocab_size - 1) for _ in range(fix_output_length)]  # random token ids

        if pulse and request_count % num_req_pulse == 0 and request_count > 0:
            # generate pulse
            time_offset_ns += delay_seconds * 1_000_000_000
        elif not pulse or (pulse and use_poisson_in_pulse):
            interval_ns = int(np.random.exponential(scale=1e9 / request_per_sec))
            time_offset_ns += interval_ns

        record = {
            "input_toks": len(input_tokens),
            "output_toks": len(output_tokens),
            "arrival_time_ns": time_offset_ns,
            "input_tok_ids": input_tokens,
            "output_tok_ids": output_tokens
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        request_count += 1

print(f"Finished writing {request_count} requests to {output_path}")
