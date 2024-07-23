import re
import copy
from wrapper.util import header, formatter
import sys

# models = ['embd', 'ln', 'attn', 'proj', 'linear1', 'linear2']

def read_file(model, layer, batch, seq, init, skip, ORCA=None):
    if ORCA == None:
        model_type = model
    else:
        model_type = f"{model}-orca"
    with open(f'simulator_result/{model_type}/b{batch}_s{seq}_{init}/{model}-{layer}-opt_b{batch}_s{seq}_{init}.txt','r') as f:
        dic = []
        for line in f.readlines():
            split = re.findall(r'\S+', line)
            dic.append(split)
        if skip: # skip header
            return dic[3:]
        else:
            return dic
    
def make_block(model, batch, seq, init, ORCA=None):
    if ORCA == None:
        if 'llama' in model:
            layers = ['rms', 'rattn', 'proj', 'rms', 'linswi']
        else:
            layers = ['ln', 'attn', 'proj', 'ln', 'linear1', 'linear2']
    else:
        total_attn = len(ORCA)
        if 'llama' in model:
            layers = ['rms', 'qkv']
            layers += ['rattn'] * total_attn
            layers += ['proj', 'rms', 'linswi']
        else:
            layers = ['ln', 'qkv']
            layers += ['rattn'] * total_attn
            layers += ['proj', 'ln', 'linear1', 'linear2']
        attn_cnt = 0
    block = []
    for layer in layers:
        if ORCA != None and 'attn' in layer: # add attention flag
            block += [[f'ATTENTION {attn_cnt}','','','','','','','','','','']]
            attn_cnt += 1
        if ORCA == None or 'attn' not in layer:
            block += read_file(model, layer, batch, seq, init, True, ORCA)
        if ORCA != None and attn_cnt == total_attn:
            # if chakra checks attention end add ALL_REDUCE
            block += [[f'ATTENTION END','','','','','','','','','','']]
            attn_cnt = 0

    return block

def make_model(model, num_layers, batch, seq, init, ORCA=None):
    block = make_block(model, batch, seq, init, ORCA)
    full = []
    full += read_file(model, 'embd', batch, seq, init, False, ORCA)

    for i in range(num_layers):
        full += block

    if 'llama' in model:
        full += read_file(model, 'rms', batch, seq, init, True, ORCA)
    else:
        full += read_file(model, 'ln', batch, seq, init, True, ORCA)

    layer_num = len(full) - 3
    full[1][0] = str(layer_num)

    return full

def store(model, batch, seq, init, parallel, nodes, result, ORCA=None):
    if ORCA == None:
        model_type = model
    else:
        model_type = f"{model}-orca"
        parallel = "orca"
    with open(f"simulator_result/{model_type}/{model}_b{batch}_s{seq}_{parallel}_n{nodes}_{init}.txt", "w") as f:
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
                name = '_'.join(result[i][0].split('_')[1:])
                new_string = f'{name}_{i-3}'
                if i < len(result) - 1 and "ATTENTION" not in result[i+1][0]:
                    cur_output = int(result[i][7])
                    next_input = int(result[i+1][3])
                    if next_input > cur_output:
                        result[i][7] = str(next_input)
                    else:
                        result[i+1][3] = str(cur_output)
                f.write(formatter(new_string, *result[i][1:], parallel))
            else:
                f.write(formatter(*result[i], parallel))

def full_model(model, batch_size, seq_len, init_or_gen, parallel, nodes, ORCA=None):
    if model == 'gpt2':
        num_layers = 12
    elif model == 'gpt3-125m':
        num_layers = 12
    elif model == 'gpt3-350m':
        num_layers = 24
    elif model == 'gpt3-760m':
        num_layers = 24
    elif model == 'gpt3-1.3b':
        num_layers = 24
    elif model == 'gpt3-2.7b':
        num_layers = 32
    elif model == 'gpt3-6.7b':
        num_layers = 32
    elif model == 'gpt3-13b':
        num_layers = 40
    elif model == 'gpt3-30b':
        num_layers = 48
    elif model == 'gpt3-175b':
        num_layers = 96
    elif model == 'opt-125m':
        num_layers = 12
    elif model == 'opt-350m':
        num_layers = 24
    elif model == 'opt-1.3b':
        num_layers = 24
    elif model == 'opt-2.7b':
        num_layers = 32
    elif model == 'opt-6.7b':
        num_layers = 32
    elif model == 'opt-13b':
        num_layers = 40
    elif model == 'opt-30b':
        num_layers = 48
    elif model == 'opt-66b':
        num_layers = 64
    elif model == 'opt-175b':
        num_layers = 96
    elif model == 'llama-7b':
        num_layers = 32
    elif model == 'llama-13b':
        num_layers = 40
    elif model == 'llama-30b':
        num_layers = 60
    elif model == 'llama-70b':
        num_layers = 80

    # print("Making full model with the results")
    if ORCA == None:
        store(model, batch_size, seq_len, init_or_gen, parallel, nodes, make_model(model, num_layers, batch_size, seq_len, init_or_gen))
    else:
        store(model, batch_size, seq_len, init_or_gen, parallel, nodes, make_model(model, num_layers, batch_size, seq_len, init_or_gen, ORCA), ORCA)

def read_full_file(model, batch_size, seq_len, init, parallel, nodes, skip=True):
    with open(f'simulator_result/{model}/{model}_b{batch}_s{seq_len}_{parallel}_n{nodes}_{init}.txt','r') as f:
        dic = []
        for line in f.readlines():
            split = re.findall(r'\S+', line)
            dic.append(split)
        if skip:
            return dic[3:]
        else:
            return dic

def store_request(model, batch_size, seq_len, end, parallel, nodes, result):
    with open(f"../astra-sim/inputs/custom_workload/{model}_b{batch}_s{seq_len}-{end}_{parallel}_n{nodes}.txt", "w") as f:
        # write header
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
            f.write(formatter(*result[i], parallel))

def full_request(model, batch_size, seq_len, end, parallel, nodes):
    full = []
    # read init
    full += read_full_file(model, batch_size, seq_len, 'init', parallel, nodes, False)
    # read gen
    for s in range(seq_len+1, end):
        gen = read_full_file(model, batch_size, s, 'gen', parallel, nodes)
        # match outputs (always output is bigger due to embedding)
        gen[0][3] = full[-1][7]
        full += gen
    # change layer length
    layer_num = len(full) - 3
    full[1][0] = str(layer_num)
    # store
    # print("Making full request with the results")
    store_request(model, batch_size, seq_len, end, parallel, nodes, full)

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print(f"Usage: {sys.argv[0]} [model_name] [batch] [seq] [end] [parallelism (tensor/pipeline/hybrid)] [num_node_group] (optional, defualt=1, only used in Hybrid)]")
        exit(0)

    model_name = sys.argv[1]
    batch = int(sys.argv[2])
    seq = int(sys.argv[3])
    end = int(sys.argv[4])

    parallel = sys.argv[5]
    if parallel not in ['hybrid', 'pipeline', 'tensor']:
        print(f"Error: No supported parallelism: {parallel}")
        parallel = 'hybrid'

    if parallel == 'hybrid':
        if len(sys.argv) > 6:
            node_num = int(sys.argv[6])
        else:
            node_num = 1
    else:
        node_num = 1

    full_request(model_name, batch, seq, end, parallel, node_num)