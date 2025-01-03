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

    if not os.path.isdir(f'../../../inputs/custom_workload/{file_name}'):
        os.mkdir(f'../../../inputs/custom_workload/{file_name}')

    cmd = f'python -m chakra.et_converter.et_converter --input_type LLM ' \
            f'--input_filename ../../../inputs/custom_workload/{file_name}.txt ' \
            f'--output_filename ../../../inputs/custom_workload/{file_name}/llm ' \
            f'--num_npus {total_num} --num_dims 1 --num_passes 1'

    cmd = cmd.split()
    subprocess.run(cmd, text=True)
    os.chdir(cwd)
    return