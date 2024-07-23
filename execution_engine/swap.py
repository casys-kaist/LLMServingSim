import re
import copy
from wrapper.util import header, formatter

def read_attn(model, batch, cache_len):
    r = ''
    if 'llama' in model:
        r = 'r'
    with open(f"simulator_result/{model}/b{batch}_gen/{model}-{r}gen-opt_b{batch}_s{cache_len}_gen.txt",'r') as f:
        swap_dict = dict()
        for line in f.readlines():
            split = re.findall(r'\S+', line)
            # print(split)
            if "matmul4d6" in split[0] or "matmul4d10" in split[0]:
                swap_dict["matmul1"] = split

            elif "elem_div_const" in split[0]:
                swap_dict["div"] = split
            
            elif "softmax" in split[0]:
                swap_dict["softmax"] = split

            elif "matmul4d9" in split[0] or "matmul4d13" in split[0]:
                swap_dict["matmul2"] = split
    return swap_dict

def read_attn_orca(model, batch, ORCA, init_cnt, overhead='none'):
    r = ''
    if 'llama' in model:
        r = 'r'
    swaps = []
    models = []
    if init_cnt > 0 :
        init_ORCA = ORCA[len(ORCA) - init_cnt:]
        for i in init_ORCA:
            models += [f"simulator_result/{model}-orca/attn-init/{model}-{r}init-{i}-opt_b{batch}_s{i}_init.txt"]
    if len(ORCA) - init_cnt > 0:
        gen_ORCA = ORCA[:len(ORCA) - init_cnt]
        for i in gen_ORCA:
            models += [f"simulator_result/{model}-orca/attn-gen/{model}-{r}gen-{i}-opt_b{batch}_s{i}_gen.txt"]
    # overhead modeling
    init_overhead = 1
    gen_overhead = 1
    
    if overhead == 'gpu':
        if '6.7b' in model or '7b' in model:
            init_overhead=2
            gen_overhead=2
        elif '30b' in model:
            init_overhead=8
            gen_overhead=1

    for model in models:
        if init_cnt > 0:
            isInit = True
            init_cnt -= 1
        else:
            isInit = False
        with open(model,'r') as f:
            swap_list = []
            for i, line in enumerate(f.readlines()):
                if i > 2:
                    split = re.findall(r'\S+', line)
                    split[0] = '_'.join(split[0].split('_')[1:])+f'_{i-2}'
                    if isInit:
                        split[1] = str(int(int(split[1])*init_overhead))
                    else:
                        if ('elem_div_const' in split[0] or 'softmax' in split[0]) and (overhead == 'neupims' or (overhead == 'gpu' and 'gpt3-30b' in model)):
                            continue
                        split[1] = str(int(int(split[1])*gen_overhead))
                    swap_list.append(split)
        swaps.append(swap_list)
    return swaps

def swap_in(swap_dict, model, batch, cache_len, parallel, node_num):
    with open(f'simulator_result/{model}/{model}_b{batch}_s1_{parallel}_n{node_num}_init.txt', 'r') as f:
        base = [] # list for right sequence
        swap_start = False
        KVload = 1
        KVsize = 0
        for line in f.readlines():
            split = re.findall(r'\S+', line)
            
            if "matmul4d" in split[0] and not swap_start:
                swap_start = True
                # add K loading
                K_load = copy.deepcopy(swap_dict["matmul1"])
                K_load[0] = f"load_k_{KVload}"     # name
                K_load[1] = '0'                         # comp
                trans_out = int(base[-1][7]) // 2
                base[-1][7] = str(trans_out)            # transpose output
                K_load[7] = K_load[3]                   # output
                KVsize = abs(int(K_load[3]) - trans_out)
                K_load[5] = str(KVsize)                 # weight (K)
                K_load[3] = str(trans_out)              # input
                base.append(K_load)

                to_base = copy.deepcopy(swap_dict["matmul1"])
                to_base[0] = split[0]
                base.append(to_base)
            
            elif "elem_div_const" in split[0] and swap_start:
                to_base = copy.deepcopy(swap_dict["div"])
                to_base[0] = split[0]
                base.append(to_base)

            elif "softmax" in split[0] and swap_start:
                to_base = copy.deepcopy(swap_dict["softmax"])
                to_base[0] = split[0]
                base.append(to_base)
            
            elif "matmul4d" in split[0] and swap_start:
                # add V loading
                V_load = copy.deepcopy(swap_dict["matmul2"])
                V_load[0] = f"load_v_{KVload}"      # name
                V_load[1] = '0'                     # comp

                mat_out = int(base[-1][7])
                V_load[7] = V_load[3]               # output
                V_load[5] = str(KVsize)             # weight (V)
                V_load[3] = str(abs(mat_out - KVsize))   # input
                base[-1][7] = V_load[3]
                base.append(V_load)

                to_base = copy.deepcopy(swap_dict["matmul2"])
                to_base[0] = split[0]
                base.append(to_base)
                swap_start = False

            else:
                base.append(split)

        layer_num = len(base) - 3
        base[1][0] = str(layer_num)

    return base

def swap_in_orca(swaps, model, batch, cache_len, node_num):
    with open(f'simulator_result/{model}-orca/{model}_b{batch}_s{cache_len}_orca_n{node_num}_init.txt', 'r') as f:
        base = [] # list for right sequence
        attn_id = 0
        for line in f.readlines():
            split = re.findall(r'\S+', line)
            if "ATTENTION" in split[0]: # attention start
                base.append([' '.join(split)])
                if "END" == split[1]: # end of attention
                    attn_id = 0
                    continue
                else:
                    attn_id = int(split[1])
                    swap_list = swaps[attn_id]
                    base += swap_list
                    continue
            else:
                base.append(split)

        layer_num = len(base) - 3
        base[1][0] = str(layer_num)

    return base


def store(model, batch, cache_len, parallel, nodes, result, ORCA=None):
    if ORCA == None:
        model_path = f"simulator_result/{model}/{model}_b{batch}_s{cache_len}_{parallel}_n{nodes}_gen.txt"
    else:
        # direct to astra-sim
        model_path = f"../astra-sim/inputs/custom_workload/{model}_b{batch}_s{cache_len}_orca_n{nodes}.txt"

    with open(model_path ,"w") as f:
        # write header
        if ORCA != None:
            f.write(f"ORCA\t\tmodel_parallel_NPU_group: {nodes}\n")
        else:
            if parallel == 'hybrid':
                f.write(f"HYBRID_TENSOR_PIPELINE\tmodel_parallel_NPU_group: {nodes}\n") # 1 for temporary
            elif parallel == 'pipeline':
                f.write("PIPELINE\n")
            else:
                f.write("TENSOR\n")
        f.write(result[1][0]+'\n')
        f.write(header())

        # write body
        for i in range(3, len(result)):
            if "ATTENTION" not in result[i][0]:
                name = '_'.join(result[i][0].split('_')[:-1])
                new_string = f'{name}_{i-3}'
                result[i][1] = str(int(int(result[i][1])))
                f.write(formatter(new_string, *result[i][1:], parallel))
            else:
                f.write(formatter(*result[i], *['','','','','','','','','',''], parallel))

def swap_model(model, batch, seq, parallel, node_num, ORCA=None, init_cnt=None, overhead='none'):
    # print("Making generation phase with the results")
    if ORCA == None:
        result = swap_in(read_attn(model, batch, seq), model, batch, seq, parallel, node_num)
    else:
        result = swap_in_orca(read_attn_orca(model, batch, ORCA, init_cnt, overhead), model, batch, seq, node_num)
    store(model, batch, seq, parallel, node_num, result, ORCA)


