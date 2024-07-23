import os
import subprocess
from time import time
from .request import *

def generateGraph(batch, parallel, npu_group, npu_num):
    # print("generateGraph start")
    cwd = os.getcwd()
    chakra = os.path.join(cwd, "extern/graph_frontend/chakra")
    os.chdir(chakra)

    model = batch.model
    batch_size = batch.batch_size
    input_len = batch.input
    output_len = batch.output
    npu_group = str(npu_group)

    if not batch.isORCA:
        out_dir = f"../../../inputs/custom_workload/{model}_b{batch_size}_s{input_len}-{output_len}_{parallel}_n{npu_group}"
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        cmd = f'python -m chakra.et_converter.et_converter --input_type LLM ' \
                f'--input_filename ../../../inputs/custom_workload/{model}_b{batch_size}_s{input_len}-{output_len}_{parallel}_n{npu_group}.txt ' \
                f'--output_filename ../../../inputs/custom_workload/{model}_b{batch_size}_s{input_len}-{output_len}_{parallel}_n{npu_group}/llm ' \
                f'--num_npus {npu_num} --num_dims 1 --num_passes 1'
    else:
        out_dir = f"../../../inputs/custom_workload/{model}_b{batch_size}_s{input_len}_orca_n{npu_group}"
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        cmd = f'python -m chakra.et_converter.et_converter --input_type LLM ' \
                f'--input_filename ../../../inputs/custom_workload/{model}_b{batch_size}_s{input_len}_orca_n{npu_group}.txt ' \
                f'--output_filename ../../../inputs/custom_workload/{model}_b{batch_size}_s{input_len}_orca_n{npu_group}/llm ' \
                f'--num_npus {npu_num} --num_dims 1 --num_passes 1'
    # print(cmd)
    cmd = cmd.split()
    start = time()
    subprocess.run(cmd, text=True)
    end = time()
    os.chdir(cwd)
    # print("generateGraph done")
    return (end-start)*1000