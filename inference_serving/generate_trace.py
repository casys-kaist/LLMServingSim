import os
import subprocess
import re
from .request import *
from .utils import *
import pandas as pd
from .memory_model import calculate_sizes

def generate_trace(batch, hardware, npu_num, npu_group, fp=16):

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
        init.append(req.is_init)
    # orca = " ".join(attn)

    print(f"Trace: batch #{batch.batch_id}: model: {model}, num requests: {len(attn)}, total length: {input_len}, prompt/kv_cache length: {sum(attn)}")

    output_path = f"inputs/trace/{hardware}_{batch.model}_batch{batch.batch_id}.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # make trace
    synthsize_trace(hardware, model, input_len, attn, init, output_path, tp, fp)


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
def synthsize_trace(hardware, model, total_len, attn, init, output_path, tp, fp=2):
    file_path = f"../perf_model/{hardware}.csv"
    df = pd.read_csv(file_path, sep=',')
    config = get_config(model)

    with open(output_path, 'w') as f:
        # f.write(header())
        # write embedding
        embedding_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "embedding")]
        emb_input, emb_weight, emb_output = calculate_sizes(model, embedding_matching_row["layer_name"].values[0], total_len)
        f.write(formatter(str(embedding_matching_row["layer_name"].values[0]), str(embedding_matching_row['latency(ns)'].values[0]), 'REMOTE',
             str(emb_input), 'LOCAL', str(emb_weight), 'REMOTE', str(emb_output), 'NONE', '0', 'NONE', 'hybrid'))

        # make transformer block
        block_res = []
        input_ln_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "input_layernorm")]
        in_ln_input, in_ln_weight, in_ln_output = calculate_sizes(model, input_ln_matching_row["layer_name"].values[0], total_len)
        block_res.append(formatter(str(input_ln_matching_row["layer_name"].values[0]), str(input_ln_matching_row['latency(ns)'].values[0]), 'REMOTE', str(in_ln_input), 'LOCAL', str(in_ln_weight), 'REMOTE', str(in_ln_output), 'NONE', '0', 'NONE', 'hybrid'))

        # q, k ,v 
        q_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "q_proj")]
        q_input, q_weight, q_output = calculate_sizes(model, q_matching_row["layer_name"].values[0], total_len)
        block_res.append(formatter(str(q_matching_row["layer_name"].values[0]), str(q_matching_row['latency(ns)'].values[0]), 'REMOTE', str(q_input), 'LOCAL', str(q_weight), 'REMOTE',  str(q_output), 'NONE', '0', 'NONE', 'hybrid'))
        k_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "k_proj")]
        k_input, k_weight, k_output = calculate_sizes(model, k_matching_row["layer_name"].values[0], total_len)
        block_res.append(formatter(str(k_matching_row["layer_name"].values[0]), str(k_matching_row['latency(ns)'].values[0]), 'REMOTE', str(k_input), 'LOCAL', str(k_weight), 'REMOTE',  str(k_output), 'NONE', '0', 'NONE', 'hybrid'))
        v_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "v_proj")]
        v_input, v_weight, v_output = calculate_sizes(model, v_matching_row["layer_name"].values[0], total_len)
        block_res.append(formatter(str(v_matching_row["layer_name"].values[0]), str(v_matching_row['latency(ns)'].values[0]), 'REMOTE', str(v_input), 'LOCAL', str(v_weight), 'REMOTE',  str(v_output), 'NONE', '0', 'NONE', 'hybrid'))

        
        # attention layer (Q*K=S & S*V)
        for i in range(len(attn)):
            block_res.append(f"ATTENTION {i}\n")

            if 'llama' in model.lower():
                # RoPE
                if init[i]:
                    rope_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == attn[i]) & (df['kv_cache'] == 0) & (df['layer_name'] == "rope")]
                else:
                    rope_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == 1) & (df['kv_cache'] == attn[i]) & (df['layer_name'] == "rope")]
                rope_input, rope_weight, rope_output = calculate_sizes(model, rope_matching_row["layer_name"].values[0], attn[i], init[i])
                block_res.append(formatter(str(rope_matching_row["layer_name"].values[0]), str(rope_matching_row['latency(ns)'].values[0]), 'REMOTE', str(rope_input), 'LOCAL', str(rope_weight), 'REMOTE',  str(rope_output), 'NONE', '0', 'NONE', 'hybrid'))
                # Attention
                if init[i]:
                    attn_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == attn[i]) & (df['kv_cache'] == 0) & (df['layer_name'] == "attn")]
                else:
                    attn_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == 1) & (df['kv_cache'] == attn[i]) & (df['layer_name'] == "attn")]
                attn_input, attn_weight, attn_output = calculate_sizes(model, attn_matching_row["layer_name"].values[0], attn[i], init[i])
                block_res.append(formatter(str(attn_matching_row["layer_name"].values[0]), str(attn_matching_row['latency(ns)'].values[0]), 'REMOTE', str(attn_input), 'LOCAL', str(attn_weight), 'REMOTE',  str(attn_output), 'NONE', '0', 'NONE', 'hybrid'))
            else:
                # QK matmul
                if init[i]:
                    qk_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == attn[i]) & (df['kv_cache'] == 0) & (df['layer_name'] == "qk_matmul")]
                else:
                    qk_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == 1) & (df['kv_cache'] == attn[i]) & (df['layer_name'] == "qk_matmul")]
                qk_input, qk_weight, qk_output = calculate_sizes(model, qk_matching_row["layer_name"].values[0], attn[i], init[i])
                block_res.append(formatter(str(qk_matching_row["layer_name"].values[0]), str(qk_matching_row['latency(ns)'].values[0]), 'REMOTE', str(qk_input), 'LOCAL', str(qk_weight), 'REMOTE',  str(qk_output), 'NONE', '0', 'NONE', 'hybrid'))
                # softmax
                if init[i]:
                    softmax_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == attn[i]) & (df['kv_cache'] == 0) & (df['layer_name'] == "softmax")]
                else:
                    softmax_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == 1) & (df['kv_cache'] == attn[i]) & (df['layer_name'] == "softmax")]
                softmax_input, softmax_weight, softmax_output = calculate_sizes(model, softmax_matching_row["layer_name"].values[0], attn[i], init[i])
                block_res.append(formatter(str(softmax_matching_row["layer_name"].values[0]), str(softmax_matching_row['latency(ns)'].values[0]), 'REMOTE', str(softmax_input), 'LOCAL', str(softmax_weight), 'REMOTE',  str(softmax_output), 'NONE', '0', 'NONE', 'hybrid'))
                # SV matmul
                if init[i]:
                    sv_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == attn[i]) & (df['kv_cache'] == 0) & (df['layer_name'] == "sv_matmul")]
                else:
                    sv_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == 1) & (df['kv_cache'] == attn[i]) & (df['layer_name'] == "sv_matmul")]
                sv_input, sv_weight, sv_output = calculate_sizes(model, sv_matching_row["layer_name"].values[0], attn[i], init[i])
                block_res.append(formatter(str(sv_matching_row["layer_name"].values[0]), str(sv_matching_row['latency(ns)'].values[0]), 'REMOTE', str(sv_input), 'LOCAL', str(sv_weight), 'REMOTE',  str(sv_output), 'NONE', '0', 'NONE', 'hybrid'))
            
        block_res.append("ATTENTION END\n")

        # attention projection
        attn_dns_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "o_proj")]
        attn_input, attn_weight, attn_output = calculate_sizes(model, attn_dns_matching_row["layer_name"].values[0], total_len)
        # tensor parallelism synchronization (ALLREDUCE)
        attn_comm_size = 0
        attn_comm_type = 'NONE' 
        if tp:
            attn_comm_size = attn_output
            attn_comm_type = 'ALLREDUCE'
        block_res.append(formatter(str(attn_dns_matching_row["layer_name"].values[0]), str(attn_dns_matching_row['latency(ns)'].values[0]), 'REMOTE', str(attn_input), 'LOCAL', str(attn_weight), 'REMOTE', str(attn_output), attn_comm_type, str(attn_comm_size), 'NONE', 'hybrid'))

        # layer norm2
        layer_norm2_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "post_layernorm")]
        layer_norm2_input, layer_norm2_weight, layer_norm2_output = calculate_sizes(model, layer_norm2_matching_row["layer_name"].values[0], total_len)
        block_res.append(formatter(str(layer_norm2_matching_row["layer_name"].values[0]), str(layer_norm2_matching_row['latency(ns)'].values[0]), 'REMOTE', str(layer_norm2_input), 'LOCAL', str(layer_norm2_weight), 'REMOTE', str(layer_norm2_output), 'NONE', '0', 'NONE', 'hybrid'))

        if 'llama' in model.lower():
            ffn1_name = "gate_proj"
        else:
            ffn1_name = "fc1"

        ffn1_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == ffn1_name)]
        ffn1_input, ffn1_weight, ffn1_output = calculate_sizes(model, ffn1_matching_row["layer_name"].values[0], total_len)
        block_res.append(formatter(str(ffn1_matching_row["layer_name"].values[0]), str(ffn1_matching_row['latency(ns)'].values[0]), 'REMOTE', str(ffn1_input), 'LOCAL', str(ffn1_weight), 'REMOTE', str(ffn1_output), 'NONE', '0', 'NONE', 'hybrid'))
        
        if 'llama' in model.lower():
            ffn2_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "up_proj")]
            ffn2_input, ffn2_weight, ffn2_output = calculate_sizes(model, ffn2_matching_row["layer_name"].values[0], total_len)
            block_res.append(formatter(str(ffn2_matching_row["layer_name"].values[0]), str(ffn2_matching_row['latency(ns)'].values[0]), 'REMOTE', str(ffn2_input), 'LOCAL', str(ffn2_weight), 'REMOTE', str(ffn2_output), 'NONE', '0', 'NONE', 'hybrid'))

        act_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "act_fn")]
        act_input, act_weight, act_output = calculate_sizes(model, act_matching_row["layer_name"].values[0], total_len)
        block_res.append(formatter(str(act_matching_row["layer_name"].values[0]), str(act_matching_row['latency(ns)'].values[0]), 'REMOTE', str(act_input), 'LOCAL', str(act_weight), 'REMOTE', str(act_output), 'NONE', '0', 'NONE', 'hybrid'))

        if 'llama' in model.lower():
            proj_name = "down_proj"
        else:
            proj_name = "fc2"
        proj_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == proj_name)]
        proj_input, proj_weight, proj_output = calculate_sizes(model, proj_matching_row["layer_name"].values[0], total_len)
        # tensor parallelism synchronization (ALLREDUCE)
        proj_comm_size = 0
        proj_comm_type = 'NONE' 
        if tp:
            proj_comm_size = proj_output
            proj_comm_type = 'ALLREDUCE'
        block_res.append(formatter(str(proj_matching_row["layer_name"].values[0]), str(proj_matching_row['latency(ns)'].values[0]), 'REMOTE', str(proj_input), 'LOCAL', str(proj_weight), 'REMOTE', str(proj_output), proj_comm_type, str(proj_comm_size), 'NONE', 'hybrid'))
   
        for i in range(config['num_hidden_layers']):
            f.writelines(block_res)

        # add final layer norm
        final_ln_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "final_layernorm")]
        final_ln_input, final_ln_weight, final_ln_output = calculate_sizes(model, final_ln_matching_row["layer_name"].values[0], total_len)
        f.write(formatter(str(final_ln_matching_row["layer_name"].values[0]), str(final_ln_matching_row['latency(ns)'].values[0]), 'REMOTE', str(final_ln_input), 'LOCAL', str(final_ln_weight), 'REMOTE', str(final_ln_output), 'NONE', '0', 'NONE', 'hybrid'))

        # add lm_head layer
        lm_matching_row = df[(df['model'] == model) & (df['hardware'] == hardware) & (df['input'] == total_len) & (df['kv_cache'] == 0) & (df['layer_name'] == "lm_head")]
        lm_input, lm_weight, lm_output = calculate_sizes(model, lm_matching_row["layer_name"].values[0], total_len)  
        f.write(formatter(str(lm_matching_row["layer_name"].values[0]), str(lm_matching_row['latency(ns)'].values[0]), 'REMOTE', str(lm_input), 'LOCAL', str(lm_weight), 'REMOTE', str(lm_output), 'NONE', '0', 'NONE', 'hybrid'))
        f.flush()


# generate event for first request arrival
def generate_event(alarm):
    
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
    output_path = f"inputs/trace/event_handler.txt"
    with open(output_path, 'w') as f:
        f.write(f"EVENT\n")
        f.write(f'{len(result)}'+'\n') # length of the text is 1
        f.write(header())
        for i in result:
            f.write(formatter(*i, 'hybrid'))