import os
import subprocess
import re
from .request import *
from .utils import *
import pandas as pd
from .memory_model import calculateSizes

def generateTrace(batch, hardware, npu_num, npu_group, fp=16):

    model = batch.model
    tp = True
    if npu_num == npu_group:
        # full pipeline
        tp = False
    npu_group = str(npu_group)
    fp = fp // 8 # bit -> byte of floating point

    # vllm: add load or eviction in the txt file
    load_size = batch.load
    evict_size = batch.evict

    input_len = batch.input
    parallel = 'hybrid' # default
    attn = []
    init = []
    for req in batch.requests:
        attn.append(req.input)
        init.append(req.isInit)
    # orca = " ".join(attn)

    print(f"Trace: batch #{batch.batch_id}: model: {model}, num requests: {len(attn)}, total length: {input_len}, prompt/kv_cache length: {sum(attn)}")

    output_path = f"inputs/custom_workload/{hardware}_{batch.model}_batch{batch.batch_id}.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # make trace
    synthsizeTrace(hardware, model, input_len, attn, init, output_path, tp, fp)


    with open(output_path, 'r') as f:
        dic = []
        for line in f.readlines():
            split = re.findall(r'\S+', line)
            dic.append(split)

    # vllm: open output txt file and add load, evict mem 
    mem = []
    if load_size != 0:
        load = ["vllm_load_kv", '0', 'LOCAL', '0', 'REMOTE', str(load_size), 'REMOTE', '0', 'NONE', '0', 'NONE']
        mem.append(load)
    if evict_size != 0:
        evict = ["vllm_evict_kv", '0', 'LOCAL', '0', 'REMOTE', str(evict_size), 'REMOTE', '0', 'NONE', '0', 'NONE']
        mem.append(evict)

    result = mem + dic

    with open(output_path, 'w') as f:
        f.write(f"ORCA\t\tmodel_parallel_NPU_group: {npu_group}\n")
        f.write(str(len(result))+'\n')
        f.write(header())

        # add layer_number at the end of the layer_name
        for i in range(0, len(result)):
            if "ATTENTION" not in result[i][0]:
                # name = '_'.join(result[i][0].split('_')[:-1])
                new_string = f'{result[i][0]}_{i}'
                f.write(formatter(new_string, *result[i][1:], parallel))
            else:
                f.write(formatter(' '.join(result[i]),'','','','','','','','','','', parallel))
    return

# makes trace for the batch
# change it as needed
def synthsizeTrace(hardware, model, total_len, attn, init, output_path, tp, fp=2):
    file_path = f"../perf_model/{hardware}.csv"
    df = pd.read_csv(file_path, sep=',')
    n_embd, n_layer, n_head, vocab_size = getConfig(model)

    with open(output_path, 'w') as f:
        # f.write(header())
        # write embedding
        embedding_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "vocab_embedding")]
        emb_input, emb_weight, emb_output = calculateSizes(model, embedding_matching_row["layer_name"].values[0], total_len)
        f.write(formatter(str(embedding_matching_row["layer_name"].values[0]), str(embedding_matching_row['latency(ns)'].values[0]), 'REMOTE',
             str(emb_input), 'LOCAL', str(emb_weight), 'REMOTE', str(emb_output), 'NONE', '0', 'NONE', 'hybrid'))

        # make transformer block
        block_res = []
        input_ln_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "input_layernorm")]
        in_ln_input, in_ln_weight, in_ln_output = calculateSizes(model, input_ln_matching_row["layer_name"].values[0], total_len)
        block_res.append(formatter(str(input_ln_matching_row["layer_name"].values[0]), str(input_ln_matching_row['latency(ns)'].values[0]), 'REMOTE', str(in_ln_input), 'LOCAL', str(in_ln_weight), 'REMOTE', str(in_ln_output), 'NONE', '0', 'NONE', 'hybrid'))

        qkv_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "attention/qkv")]
        qkv_input, qkv_weight, qkv_output = calculateSizes(model, qkv_matching_row["layer_name"].values[0], total_len)
        block_res.append(formatter(str(qkv_matching_row["layer_name"].values[0]), str(qkv_matching_row['latency(ns)'].values[0]), 'REMOTE', str(qkv_input), 'LOCAL', str(qkv_weight), 'REMOTE',  str(qkv_output), 'NONE', '0', 'NONE', 'hybrid'))
        # attention layer (Q*K=S & S*V)
        for i in range(len(attn)):
            gemv_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == 1) & (df['kv_cache'] == attn[i]) & (df['layer_name'] == "attention/wrapper")]
            block_res.append(f"ATTENTION {i}\n")
            gemv_input, gemv_weight, gemv_output = calculateSizes(model, gemv_matching_row["layer_name"].values[0], attn[i], init[i])
            block_res.append(formatter(str(gemv_matching_row["layer_name"].values[0]), str(gemv_matching_row['latency(ns)'].values[0]), 'REMOTE', str(gemv_input), 'LOCAL', str(gemv_weight), 'REMOTE', str(gemv_output), 'NONE', '0', 'NONE', 'hybrid'))
        block_res.append("ATTENTION END\n")

        attn_dns_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "attention/dense")]
        attn_input, attn_weight, attn_output = calculateSizes(model, attn_dns_matching_row["layer_name"].values[0], total_len)
        # tensor parallelism synchronization (ALLREDUCE)
        attn_comm_size = 0
        attn_comm_type = 'NONE' 
        if tp:
            attn_comm_size = attn_output
            attn_comm_type = 'ALLREDUCE'
        block_res.append(formatter(str(attn_dns_matching_row["layer_name"].values[0]), str(attn_dns_matching_row['latency(ns)'].values[0]), 'REMOTE', str(attn_input), 'LOCAL', str(attn_weight), 'REMOTE', str(attn_output), attn_comm_type, str(attn_comm_size), 'NONE', 'hybrid'))

        # add FFN layer
        fc_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "mlp/fc")]
        fc_input, fc_weight, fc_output = calculateSizes(model, fc_matching_row["layer_name"].values[0], total_len)
        block_res.append(formatter(str(fc_matching_row["layer_name"].values[0]), str(fc_matching_row['latency(ns)'].values[0]), 'REMOTE', str(fc_input), 'LOCAL', str(fc_weight), 'REMOTE', str(fc_output), 'NONE', '0', 'NONE', 'hybrid'))
        gelu_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "mlp/gelu")]
        gelu_input, gelu_weight, gelu_output = calculateSizes(model, gelu_matching_row["layer_name"].values[0], total_len)
        block_res.append(formatter(str(gelu_matching_row["layer_name"].values[0]), str(gelu_matching_row['latency(ns)'].values[0]), 'REMOTE', str(gelu_input), 'LOCAL', str(gelu_weight), 'REMOTE', str(gelu_output), 'NONE', '0', 'NONE', 'hybrid'))
        proj_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "mlp/proj")]
        proj_input, proj_weight, proj_output = calculateSizes(model, proj_matching_row["layer_name"].values[0], total_len)
        # tensor parallelism synchronization (ALLREDUCE)
        proj_comm_size = 0
        proj_comm_type = 'NONE' 
        if tp:
            proj_comm_size = proj_output
            proj_comm_type = 'ALLREDUCE'
        block_res.append(formatter(str(proj_matching_row["layer_name"].values[0]), str(proj_matching_row['latency(ns)'].values[0]), 'REMOTE', str(proj_input), 'LOCAL', str(proj_weight), 'REMOTE', str(proj_output), proj_comm_type, str(proj_comm_size), 'NONE', 'hybrid'))
        post_ln_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "post_layernorm")]
        post_ln_input, post_ln_weight, post_ln_output = calculateSizes(model, post_ln_matching_row["layer_name"].values[0], total_len)
        block_res.append(formatter(str(post_ln_matching_row["layer_name"].values[0]), str(post_ln_matching_row['latency(ns)'].values[0]), 'REMOTE', str(post_ln_input), 'LOCAL', str(post_ln_weight), 'REMOTE', str(post_ln_output), 'NONE', '0', 'NONE', 'hybrid'))
        
        for i in range(n_layer):
            f.writelines(block_res)

        # add lm_head layer
        ln_f_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "ln_f")]
        ln_f_input, ln_f_weight, ln_f_output = calculateSizes(model, ln_f_matching_row["layer_name"].values[0], total_len)
        f.write(formatter(str(ln_f_matching_row["layer_name"].values[0]), str(ln_f_matching_row['latency(ns)'].values[0]), 'REMOTE', str(ln_f_input), 'LOCAL', str(ln_f_weight), 'REMOTE', str(ln_f_output), 'NONE', '0', 'NONE', 'hybrid'))
        lm_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "lm_head")]
        lm_input, lm_weight, lm_output = calculateSizes(model, lm_matching_row["layer_name"].values[0], total_len)  
        f.write(formatter(str(lm_matching_row["layer_name"].values[0]), str(lm_matching_row['latency(ns)'].values[0]), 'REMOTE', str(lm_input), 'LOCAL', str(lm_weight), 'REMOTE', str(lm_output), 'NONE', '0', 'NONE', 'hybrid'))
        f.flush()


# generate event for first request arrival
def generateEvent(alarm):
    
    # make inputs for text file
    result = []
    fp = 2
    layer_name = f'event_{alarm}ns'
    comp_time = alarm
    input_loc = 'REMOTE'
    input_size = 0
    weight_loc = 'LOCAL'
    weight_size = 0
    output_loc = 'REMOTE'
    output_size = 0
    comm_type = 'NONE'
    comm_size = 0
    misc = 'NONE'
    result.append([layer_name, comp_time, input_loc, input_size, weight_loc, weight_size, output_loc, output_size, comm_type, comm_size, misc])

    # write to the text file
    output_path = f"inputs/custom_workload/event_handler.txt"
    with open(output_path, 'w') as f:
        f.write(f"EVENT\n")
        f.write(f'{len(result)}'+'\n') # length of the text is 1
        f.write(header())
        for i in result:
            f.write(formatter(*i, 'hybrid'))