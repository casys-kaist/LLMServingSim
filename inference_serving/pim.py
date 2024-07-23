import math
from functools import reduce
import re
from time import time
from .request import *
from .utils import *
import pandas as pd


# memory spec parameters
_dram_page_size = 512
_dram_banks_per_ch = 32
_gwrite_latency = 100
_gemv_latency = 184
# clock = 1GHz (same as LLM-Sim)

# model spec
E = 4096
n_tp = 4
_nh = 32
_dk = E/_nh


def estimate_mha_latency(batch):
    E, _, _nh = get_config(batch.model)
    _dk = E/_nh
    pim_times = []
    for req in batch.requests:
        if not req.isInit:
            seq_len = req.input

            _dk = E/_nh
            _effective_e = E / n_tp
            # calculate MHA latency with sequence length
            kq_latency = 0
            lv_latency = 0

            # key * query
            chunks = math.ceil(_effective_e / _dram_page_size)
            tiles = math.ceil(seq_len / _dram_banks_per_ch)
            kq_latency += chunks * _gwrite_latency
            kq_latency += chunks * tiles * _gemv_latency

            # logit * value
            chunks = math.ceil(seq_len / _dram_page_size) * _nh
            tiles = math.ceil(_dk / _dram_banks_per_ch)
            lv_latency += chunks * _gwrite_latency
            lv_latency += chunks * tiles * _gemv_latency

            pim_times.append(kq_latency)
            pim_times.append(lv_latency)

    return pim_times

def sum_load(seqlens):
    load = reduce(lambda acc, seq_len: acc + estimate_mha_latency(seq_len), seqlens, 0)
    return load

# distribute requests takes (1)new requests, (2)previous channel distributions, (3)total channel size
def distribute_requests(new_seq_lens, channels_seqlen, k):
    # Create a list to store the sum of values in each channel
    channels_load = [sum_load(seqlens) for seqlens in channels_seqlen]
    
    for element in sorted(new_seq_lens, reverse=True):
        min_sum_index = min(range(k), key=lambda i: channels_load[i])
        channels_seqlen[min_sum_index].append(element)
        channels_load[min_sum_index] += estimate_mha_latency(element)
        
    return channels_seqlen 

# # Example usage:
# request_lengths = [5, 8, 3, 2, 7]
# channels_seqlen = [[4, 6], [8, 1], [3, 9]]
# k = 3

# result = distribute_requests(request_lengths, channels_seqlen, k)
# print(result)
# print([sum_load(seqlens) for seqlens in channels_seqlen])


def addPIMtime(batch, npu_group, pim_times, pim_type):
    model = batch.model
    batch_size = batch.batch_size
    npu_group = str(npu_group)
    input_len = batch.input
    init_cnt = batch.output
    parallel = 'hybrid'
    output_path = f"inputs/custom_workload/{model}_b{batch_size}_s{input_len}_orca_n{npu_group}.txt"

    with open(output_path, 'r') as f:
        result = []
        for line in f.readlines():
            split = re.findall(r'\S+', line)
            result.append(split)
        
    with open(output_path, 'w') as f:
        if pim_type == 'local':
            f.write(f"ORCA\t\tmodel_parallel_NPU_group: {npu_group}\n")
        else:
            f.write(f"PIM_POOL\t\tmodel_parallel_NPU_group: {npu_group}\n")
        f.write(result[1][0]+'\n')
        f.write(header())

        # write body
        tp_cnt = 0
        mm_cnt = 0
        init = False

        for i in range(3, len(result)):
            if "ATTENTION" not in result[i][0]:
                if "tensor_transpose4d" in result[i][0]:
                    tp_cnt += 1
                if tp_cnt == 4:
                    init = True
                if not init and 'matmul4d' in result[i][0]:
                    f.write(formatter(f'pim_{mm_cnt}_{i-3}', str(pim_times[mm_cnt]), *result[i][2:], parallel))
                    mm_cnt += 1
                else:
                    f.write(formatter(*result[i], parallel))
            else:
                f.write(formatter(' '.join(result[i]),'','','','','','','','','','', parallel))
                tp_cnt = 0
                init = False
                if "END" in result[i][1]:
                    mm_cnt = 0

def subbatchInt(batch):
    if len(batch.requests) == 1:
        return [batch]
    if int(batch.output) == len(batch.requests):
        # no possible overlapping
        return [batch]
    # split batch in half
    reqs = batch.requests[:]
    reqs = sorted(reqs, reverse=True, key=lambda x: x.input)
    req1 = []
    req2 = []
    for i, req in enumerate(reqs):
        if i%2 == 0:
            req1.append(req)
        else:
            req2.append(req)
    req1 = sorted(req1, key=lambda x: x.arrival)
    req2 = sorted(req2, key=lambda x: x.arrival)
    total_len = 0
    init_cnt = 0
    for req in req1:
        if req.isInit:
            total_len += req.input
            init_cnt += 1
        else:
            total_len += 1
    # making batch for just hand out values to gnerateText(). so need only usable values (others 0)
    batch1 = Batch(0, batch.model, str(total_len), str(init_cnt), '1', 0, 0, batch.evict, batch.load, True)
    batch1.requests.extend(req1)
    total_len = 0
    init_cnt = 0
    for req in req2:
        if req.isInit:
            total_len += req.input
            init_cnt += 1
        else:
            total_len += 1
    # KV cache is just handled once
    batch2 = Batch(0, batch.model, str(total_len), str(init_cnt), '1', 0, 0, 0, 0, True)
    batch2.requests.extend(req2)
    return [batch1, batch2]

def mergeText(batch, subbatches, num_npus, npu_group):
    # if there is only init phase (no need of subbatching)
    if len(subbatches) == 1:
        return

    npus_per_group = num_npus // npu_group
    # open text
    model = batch.model
    batch_size = batch.batch_size # 1
    npu_group = str(npu_group)
    parallel = 'hybrid'

    input_len1 = int(subbatches[0].input)
    init_cnt1 = int(subbatches[0].output)
    output_path1 = f"inputs/custom_workload/{model}_b{batch_size}_s{input_len1}_orca_n{npu_group}.txt"
    input_len2 = int(subbatches[1].input)
    init_cnt2 = int(subbatches[1].output)
    output_path2 = f"inputs/custom_workload/{model}_b{batch_size}_s{input_len2}_orca_n{npu_group}.txt"
    input_len3 = int(batch.input)
    init_cnt3 = int(batch.output)
    output_path3 = f"inputs/custom_workload/{model}_b{batch_size}_s{input_len3}_orca_n{npu_group}.txt"

    with open(output_path1, 'r') as f1:
        b1 = []
        for line in f1.readlines():
            split = re.findall(r'\S+', line)
            b1.append(split)

    with open(output_path2, 'r') as f2:
        b2 = []
        for line in f2.readlines():
            split = re.findall(r'\S+', line)
            b2.append(split)
    # result dict
    b = []
    # add header
    b.extend(b1[:3])
    # remove header
    b1 = b1[3:]
    b2 = b2[3:]
    # add vllm
    i = 0
    while 'vllm' in b1[i][0]:
        b.append(b1[i])
        i+=1
    b1 = b1[i:]
    # extract each layer
    embd1, ln1, qkv1, attn1, proj1, res1, ffn11, gelu1, ffn21 = extractLayer(b1)
    embd2, ln2, qkv2, attn2, proj2, res2, ffn12, gelu2, ffn22 = extractLayer(b1)

    # schedule the layer
    b.extend(embd1)
    b.extend(ln1)
    b.append(qkv1)

    b.extend(embd2)
    b.extend(ln2)
    # count attention
    attn_npu1 = {}
    attn_init1 = {}

    attn_npu2 = {}
    attn_init2 = {}

    for i, attn in enumerate(attn1):
        if i < init_cnt1:
            if i%npus_per_group not in attn_init1:
                attn_init1[i%npus_per_group] = [attn]
            else:
                attn_init1[i%npus_per_group].extend(attn)
        if i%npus_per_group not in attn_npu1:
            attn_npu1[i%npus_per_group] = sum([int(j[1]) for j in attn])
        else:
            attn_npu1[i%npus_per_group] += sum([int(j[1]) for j in attn])

    for i, attn in enumerate(attn2):
        if i < init_cnt2:
            if i%npus_per_group not in attn_init2:
                attn_init2[i%npus_per_group] = [attn]
            else:
                attn_init2[i%npus_per_group].extend(attn)
        if i%npus_per_group not in attn_npu2:
            attn_npu2[i%npus_per_group] = sum([int(j[1]) for j in attn])
        else:
            attn_npu2[i%npus_per_group] += sum([int(j[1]) for j in attn])

    # add only left over attn comp time with overlapping with qkv2
    for i in range(npus_per_group):
        b.append([f'ATTENTION {i}','','','','','','','','','',''])
        if i in attn_init1:
            b.extend(attn_init1[i])
        if i in attn_npu1 and attn_npu1[i] > int(qkv2[1])//npus_per_group:
            b.append(['attn_overlap_1',f'{attn_npu1[i]-int(qkv2[1])//npus_per_group}','LOCAL','0','REMOTE','0','REMOTE','0','NONE','0','NONE'])
    b.append([f'ATTENTION END','','','','','','','','','',''])

    # make blocks
    block1 = [proj1, res1, *ln1, ffn11, *gelu1, ffn12, res1, *ln1, qkv1]
    block1_gemm = int(proj1[1]) + int(ffn11[1]) + int(ffn12[1]) + int(qkv1[1])
    block2 = [proj2, res2, *ln2, ffn21, *gelu2, ffn22, res2, *ln2, qkv2]
    block2_gemm = int(proj2[1]) + int(ffn21[1]) + int(ffn22[1]) + int(qkv2[1])

    # add only left over attn comp time with overlapping with gemms
    for i in range(npus_per_group):
        block1.append([f'ATTENTION {i}','','','','','','','','','',''])
        block2.append([f'ATTENTION {i}','','','','','','','','','',''])
        if i in attn_init1:
            block2.extend(attn_init1[i])
        if i in attn_npu1 and attn_npu1[i] > block2_gemm//npus_per_group:
            block2.append(['attn_overlap_1',f'{attn_npu1[i]-block2_gemm//npus_per_group}','LOCAL','0','REMOTE','0','REMOTE','0','NONE','0','NONE'])
        if i in attn_init2:
            block1.extend(attn_init2[i])
        if i in attn_npu2 and attn_npu2[i] > block1_gemm//npus_per_group:
            block1.append(['attn_overlap_1',f'{attn_npu2[i]-block1_gemm//npus_per_group}','LOCAL','0','REMOTE','0','REMOTE','0','NONE','0','NONE'])
    block1.append([f'ATTENTION END','','','','','','','','','',''])
    block2.append([f'ATTENTION END','','','','','','','','','',''])
    _, n_layer, _ = get_config(model)

    # repeat N-1 times
    for _ in range(n_layer-1):
        b.extend(block1)
        b.extend(block2)

    # ending two steps
    end1 = [proj1, res1, *ln1, ffn11, *gelu1, ffn12, res1, *ln1]
    end1_gemm = int(proj1[1]) + int(ffn11[1]) + int(ffn12[1])
    # add only left over attn comp time with overlapping with gemms
    for i in range(npus_per_group):
        end1.append([f'ATTENTION {i}','','','','','','','','','',''])
        if i in attn_init2:
            end1.extend(attn_init2[i])
        if i in attn_npu2 and attn_npu2[i] > end1_gemm//npus_per_group:
            end1.append(['attn_overlap_1',f'{attn_npu2[i]-end1_gemm//npus_per_group}','LOCAL','0','REMOTE','0','REMOTE','0','NONE','0','NONE'])
    end1.append([f'ATTENTION END','','','','','','','','','',''])

    end2 = [proj2, res2, *ln2, ffn21, *gelu2, ffn22, res2, *ln2]
    b.extend(end1)
    b.extend(end2)

    layer_num = len(b) - 3
    b[1][0] = str(layer_num)

    # store b in merged output txt
    with open(output_path3, 'w') as f:
        f.write(f"ORCA\t\tmodel_parallel_NPU_group: {npu_group}\n")
        f.write(b[1][0]+'\n')
        f.write(header())

        # write body
        for i in range(3, len(b)):
            if "ATTENTION" not in b[i][0]:
                name = '_'.join(b[i][0].split('_')[:-1])
                new_string = f'{name}_{i-3}'
                # check input, output
                if i < len(b) - 1 and "ATTENTION" not in b[i+1][0]:
                    cur_output = int(b[i][7])
                    next_input = int(b[i+1][3])
                    if next_input > cur_output:
                        b[i][7] = str(next_input)
                    else:
                        b[i+1][3] = str(cur_output)
                f.write(formatter(new_string, *b[i][1:], parallel))
            else:
                f.write(formatter(*b[i], parallel))


def extractLayer(b1):
    # check embd
    embd = b1[:2]
    # check layer norm
    i = 2
    ln = []
    while 'gemm' not in b1[i][0]:
        ln.append(b1[i])
        i+=1
    # check qkv
    qkv = b1[i]
    i+=1
    # check Attentions
    attn = []
    while 'END' not in b1[i][1]:
        temp = []
        i+=1
        while 'ATTENTION' not in b1[i][0]:
            temp.append(b1[i])
            i+=1
        attn.append(temp)
    i+=1
    # check proj
    proj = b1[i]
    i+=1
    # residual
    res = b1[i]
    i+=1
    # check ffn1
    while 'gemm' not in b1[i][0]:
        i+=1
    ffn1 = b1[i]
    i+=1
    # check gelu
    gelu = []
    while 'gemm' not in b1[i][0]:
        gelu.append(b1[i])
        i+=1
    # check ffn2
    ffn2 = b1[i]

    return embd, ln, qkv, attn, proj, res, ffn1, gelu, ffn2

def dataset_converter(input_file_path, output_file_path):
    alpaca_data = pd.read_csv(input_file_path)

    # Transform the data
    alpaca_data_transformed = pd.DataFrame({
        'input_toks': alpaca_data['seq_len'],
        'output_toks': 1,
        'arrival_time_ns': 0
    })

    alpaca_data_transformed.to_csv(output_file_path, sep='\t', index=False)
    return
