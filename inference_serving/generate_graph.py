import os
import subprocess
from time import time
from .request import *

def generateGraph(batch, hardware, total_num, event=False):

    cwd = os.getcwd()
    chakra = os.path.join(cwd, "extern/graph_frontend/chakra")
    os.chdir(chakra)

    if event:
        file_name = 'event_handler'
    else:
        file_name = f'{hardware}_{batch.model}_batch{batch.batch_id}'

    workload_dir = f'../../../inputs/workload/{file_name}'
    os.makedirs(workload_dir, exist_ok=True)

    cmd = f'python -m chakra.src.converter.converter LLM ' \
            f'--input ../../../inputs/trace/{file_name}.txt ' \
            f'--output ../../../inputs/workload/{file_name}/llm ' \
            f'--num-npus {total_num}'

    cmd = cmd.split()
    subprocess.run(cmd, text=True)
    os.chdir(cwd)
    return