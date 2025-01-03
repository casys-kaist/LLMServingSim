import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference_serving.request import *
from inference_serving.generate_trace import generateTrace

def generate_dummy_request(batch_id, model):
    """Generate a dummy Request for a given batch."""
    input_size = random.randint(64, 512)
    output_size = input_size + random.randint(64, 512)
    # arrival_time = random.randint(0, 1000)
    arrival_time = 0

    return Request(
        id=f"req-{batch_id}-{random.randint(0, 100)}",
        model=model,
        input=input_size,
        output=output_size,
        arrival=arrival_time
    )

def generate_dummy_batch(batch_id):
    """Generate a dummy Batch instance."""
    model = "gpt3-6.7b"
    input_size = random.randint(64, 512)
    init_cnt = random.randint(1, 10)
    batch_size = random.randint(1, 10)
    batch_time = random.randint(0, 1000)
    kv_size = random.randint(1, 20)
    evict = random.randint(100, 200)
    load = random.randint(100, 200)
    is_orca = random.choice([True, False])

    batch = Batch(
        batch_id=batch_id,
        model=model,
        input=input_size,
        init_cnt=init_cnt,
        batch_size=batch_size,
        batch_time=batch_time,
        kv_size=kv_size,
        evict=evict,
        load=load,
        isORCA=is_orca
    )

    # Add random requests to the batch
    num_requests = batch_size
    for _ in range(num_requests):
        batch.requests.append(generate_dummy_request(batch_id, model))

    return batch



if __name__ == "__main__":
    generated_batch = generate_dummy_batch(0)
    # print(generated_batch)

    generateTrace(generated_batch, hardware="RTX3090", npu_num=4, npu_group=2)