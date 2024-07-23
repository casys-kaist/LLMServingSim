import os
import subprocess
import re
from .request import *

def getTime(result):
    compile = 0
    simulate = 0
    splited = result.stdout.split('\n')
    for l in splited:
        if 'compile' in l:
            compile += int(l.split()[-1])
        elif 'simulate' in l:
            simulate += int(l.split()[-1])
    return compile, simulate

def generateText(batch, parallel, npu_group, fast_run=False, network=None):
    # print("generateText start")
    cwd = os.getcwd()
    input = os.path.join(cwd, "../execution_engine")

    model = batch.model
    batch_size = batch.batch_size
    npu_group = str(npu_group)

    # vllm: add load or eviction in the txt file
    load_size = batch.load
    evict_size = batch.evict

    if not batch.isORCA:
        input_len = batch.input
        output_len = batch.output
        os.chdir(input)
        stdout = subprocess.run(["./run.sh", model, batch_size, input_len, output_len, parallel, npu_group], text=True, capture_output=True)
        compile, simulate = getTime(stdout)
        os.chdir(cwd)

    else: # ORCA
        input_len = batch.input
        init_cnt = batch.output
        parallel = 'hybrid'
        attn = []
        for req in batch.requests:
            attn.append(str(req.input))
        orca = " ".join(attn)
        os.chdir(input)

        print(f"model: {model}, num requests: {len(attn)}, total length: {input_len}, prompt/kv_cache length: {attn}")
        if network == None:
            overhead = 'none'
        elif 'gpu' in network:
            overhead = 'gpu'
        elif 'neupims' in network:
            overhead = 'neupims'
        else:
            overhead = 'none'
            
        command = ["./run.sh", model, batch_size, input_len, init_cnt, parallel, npu_group, orca, overhead]
        if fast_run:
            command.append("fast_run")
        # print(' '.join(command))

        stdout = subprocess.run(command, text=True, capture_output=True)
        compile, simulate = getTime(stdout)
        output_path = f"inputs/custom_workload/{model}_b{batch_size}_s{input_len}_orca_n{npu_group}.txt"
        os.chdir(cwd)
        # vllm: open output txt file and add load, evict mem
        if load_size != 0 or evict_size != 0:
            with open(output_path, 'r') as f:
                dic = []
                for line in f.readlines():
                    split = re.findall(r'\S+', line)
                    dic.append(split)
                
            # make load and evict
            mem = []
            if load_size != 0:
                load = ["vllm_load_kv", '0', 'LOCAL', '0', 'REMOTE', str(load_size), 'REMOTE', '0', 'NONE', '0', 'NONE']
                mem.append(load)
            if evict_size != 0:
                evict = ["vllm_evict_kv", '0', 'LOCAL', '0', 'REMOTE', str(evict_size), 'REMOTE', '0', 'NONE', '0', 'NONE']
                mem.append(evict)

            result = dic[:3] + mem + dic[3:]
            result[1][0] = str(len(result) - 3)

            with open(output_path, 'w') as f:
                f.write(f"ORCA\t\tmodel_parallel_NPU_group: {npu_group}\n")
                f.write(result[1][0]+'\n')
                f.write(header())

                # write body
                for i in range(3, len(result)):
                    if "ATTENTION" not in result[i][0]:
                        name = '_'.join(result[i][0].split('_')[:-1])
                        new_string = f'{name}_{i-3}'
                        f.write(formatter(new_string, *result[i][1:], parallel))
                    else:
                        f.write(formatter(' '.join(result[i]),'','','','','','','','','','', parallel))

    # print("generateText done")
    return compile, simulate