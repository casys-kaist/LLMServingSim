import os
import sys
import random
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_serving.request import *
from inference_serving.generate_trace import generate_trace

def generate_dummy_request(model, req_id):
    """Generate a dummy Request for a given batch."""
    input_size = random.randint(64, 512)
    output_size = input_size + random.randint(64, 512)
    # arrival_time = random.randint(0, 1000)
    arrival_time = 0

    return Request(
        id=req_id,
        model=model,
        input=input_size,
        output=output_size,
        arrival=arrival_time
    )

def generate_dummy_batch(model):
    """Generate a dummy Batch instance."""
    input_size = random.randint(64, 512)
    init_cnt = random.randint(1, 10)
    batch_size = random.randint(1, 10)
    batch_time = random.randint(0, 1000)
    kv_size = random.randint(1, 20)
    evict = random.randint(100, 200)
    load = random.randint(100, 200)
    is_orca = random.choice([True, False])

    batch = Batch(
        batch_id=0,
        model=model,
        input=input_size,
        init_cnt=init_cnt,
        batch_size=batch_size,
        batch_time=batch_time,
        kv_size=kv_size,
        evict=evict,
        load=load,
        is_orca=is_orca
    )

    # Add random requests to the batch
    num_requests = batch_size
    for i in range(num_requests):
        batch.requests.append(generate_dummy_request(model, i))

    return batch

def calculate_latency(path):
    # Set the directory path
    dir_path = Path(path)

    # Initialize total nanoseconds accumulator
    total_time_ns = 0

    # Iterate over all .txt files in the directory
    for file in dir_path.glob("*.txt"):
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[1].isdigit():
                    total_time_ns += int(parts[1])

    # Total computation time
    total_time_ms = total_time_ns / 1_000_000
    total_time_sec = total_time_ns / 1_000_000_000

    print(f"Total time (ns): {total_time_ns}")
    print(f"Total time (ms): {total_time_ms}")
    print(f"Total time (s): {total_time_sec}")

if __name__ == "__main__":
    generated_batch = generate_dummy_batch('meta-llama/Llama-3.1-8B-Instruct')
    # print(generated_batch)
    generate_trace(generated_batch, hardware="RTX3090", npu_num=4, npu_group=2)
    calculate_latency("inputs/trace/RTX3090_meta-llama")
