import os
import subprocess
from time import time
from .request import *
from .logger import get_logger
from .run_paths import input_path

logger = get_logger("GraphGenerator")

def generate_graph(batch, hardware, num_npus, node_id=0, instance_id=0, npu_offset=0, enable_local_offloading=False, event=False, workload_name=None, inputs_root=None, cleanup_trace=True):

    cwd = os.getcwd()
    chakra = os.path.join(cwd, "extern/graph_frontend/chakra")
    if inputs_root is None:
        inputs_root = os.path.join(cwd, "inputs")

    if event:
        file_name = 'event_handler'
    else:
        file_name = f'{hardware}/{batch.model}/instance{instance_id}_batch{batch.batch_id}'

    # For DP groups, all instances write .et files to a shared workload folder
    output_name = workload_name if workload_name else file_name

    trace_path = input_path(inputs_root, "trace", f"{file_name}.txt")
    output_path = input_path(inputs_root, "workload", output_name, "llm")
    workload_dir = os.path.dirname(output_path)
    os.makedirs(workload_dir, exist_ok=True)

    cmd = [
        'python', '-m', 'chakra.src.converter.converter', 'LLM',
        '--input', trace_path,
        '--output', output_path,
        '--num-npus', str(num_npus),
        '--npu-offset', str(npu_offset),
    ]

    if enable_local_offloading:
        cmd.append('--local-offloading')

    logger.debug("Generating graph with command: %s", " ".join(cmd), extra={"node_id": node_id, "instance_id": instance_id})

    subprocess.run(cmd, cwd=chakra, text=True, check=True)
    if cleanup_trace:
        try:
            os.remove(trace_path)
        except FileNotFoundError:
            pass
    return
