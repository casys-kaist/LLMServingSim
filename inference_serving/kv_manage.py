import os
import re
from .utils import *

def get_weight(model, npu_num):
    cwd = os.getcwd()
    weight = 0
    with open(cwd+f'/../inference_serving/model_reference/{model}.txt', 'r') as f:
        dic = []
        for line in f.readlines():
            split = re.findall(r'\S+', line)
            dic.append(split)
        # skip header
        dic = dic[3:]
        # add weight
        for r in dic:
            weight += int(r[5])
    return weight // npu_num

def get_kv(model, seq, npu_num):
    # shape of kv cache
    # (n_head, batch_size, n_embd//n_head, seq_len) per layer
    # return batch_size = 1 to caclulate max batch_size in scheduler
    fp = 2 # 2 byte if fp16, 4 byte if fp32
    n_embd, n_layer, n_head = get_config(model)
    # K & V multiply 2 
    return 2 * n_embd * seq * n_layer * fp // npu_num