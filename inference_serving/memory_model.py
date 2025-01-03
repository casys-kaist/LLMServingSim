import os
import re
from .utils import getConfig

class MemoryModel():
    def __init__(self, model, npu_num, npu_mem, block_size, fp, verbose=False):
        self.model = model
        self.npu_num = npu_num
        self.npu_mem = npu_mem
        self.block_size = block_size
        self.fp = fp // 8 # bit -> byte of floating point
        self.verbose = verbose

        self.n_embd, self.n_layer, self.n_head, self.vocab_size = getConfig(model)

        # assume NPUS use identical memory
        self.npu_mem = npu_mem * 1000000000

        # Memory model
        self.weight = self.getWeight() # assume weight is loaded
        self.kv_npu = self.npu_num # npus that store KV cache
        self.used_mem = self.weight


    # get weight of the model 
    def getWeight(self):
        cwd = os.getcwd()
        weight = 0

        # embedding
        _, embedding, _ = calculateSizes(self.model, 'vocab_embedding', 1)
        weight += embedding

        # block
        block_weight = 0
        # input layernorm
        _, input_ln, _ = calculateSizes(self.model, 'input_layernorm', 1)
        block_weight += input_ln
        # qkv
        _, qkv, _ = calculateSizes(self.model, 'attention/qkv', 1)
        block_weight += qkv
        # attention dense
        _, attn_dns, _ = calculateSizes(self.model, 'attention/dense', 1)
        block_weight += attn_dns
        # mlp fc
        _, mlp_fc, _ = calculateSizes(self.model, 'mlp/fc', 1)
        block_weight += mlp_fc
        # mlp proj
        _, mlp_proj, _ = calculateSizes(self.model, 'mlp/proj', 1)
        block_weight += mlp_proj
        # post layernorm
        _, post_ln, _ = calculateSizes(self.model, 'post_layernorm', 1)
        block_weight += post_ln

        weight += block_weight * self.n_layer

        # ln_f
        _, ln_f, _ = calculateSizes(self.model, 'ln_f', 1)
        weight += ln_f
        # lm_head
        _, lm_head, _ = calculateSizes(self.model, 'lm_head', 1)
        weight += lm_head

        if self.verbose:
            print(f"Memory: model weight {weight//1024//1024}MB loaded")

        return weight // self.npu_num


    def getKV(self, seq):
        # shape of kv cache
        # (n_head, batch_size, n_embd//n_head, seq_len) per layer
        # return batch_size = 1 to caclulate max batch_size in scheduler

        # K & V multiply 2 
        return 2 * self.n_embd * seq * self.n_layer * self.fp // self.kv_npu
    
    # used when batching. in case of vllm, it is only used in init phase
    def getBatchKV(self, batch_req, batch_len):
        batch_kv_size = 0
        for i in range(batch_len):
            num_blocks = batch_req[i].input // self.block_size + 1 # it includes kv_cache that will be generated in current iteration
            batch_kv_size += self.getKV(num_blocks * self.block_size)

        return batch_kv_size

    # get size of kv block that should be added. used in vllm gen phase
    # also checks evicted request and include its kv cache
    def getBlockKV(self, batch_req, batch_len):
        block_kv_size = 0
        for i in range(batch_len):
            if batch_req[i].evict or batch_req[i].isInit:
                num_blocks = batch_req[i].input // self.block_size + 1 # it includes kv_cache that will be generated in current iteration
                block_kv_size += self.getKV(num_blocks * self.block_size)
            else:
                num_before = (batch_req[i].input - 1) // self.block_size + 1
                num_after = batch_req[i].input // self.block_size + 1 # it includes kv_cache that will be generated in current iteration
                if num_after > num_before: # difference of the block is maximum one block
                    block_kv_size += self.getKV(self.block_size)
        
        return block_kv_size
    
    # get size of kv cache that should be evicted
    def getEvictKV(self, req):
        evict_size = 0
        # input + 1 is not loaded now
        num_blocks = (req.input-1) // self.block_size + 1
        evict_size += self.getKV(num_blocks * self.block_size)
        return evict_size
    
    def memLoad(self, size):
        if self.used_mem + size > self.npu_mem:
            print("ERROR: memLoad: no memory to load")
        if self.verbose:
            print(f"Memory: used: {self.used_mem} load: {size}", end=' ')
        self.used_mem += size
        if self.verbose:
            print(f"after: {self.used_mem}")

    def memStore(self, size):
        if self.used_mem - size < self.weight:
            print("ERROR: memStore: no memory to unload")
        if self.verbose:
            print(f"Memory: used: {self.used_mem} remove: {size}", end=' ')
        self.used_mem -= size
        if self.verbose:
            print(f"after: {self.used_mem}")

    def memAvail(self, size):
        if self.npu_mem - self.used_mem >= size:
            return True
        else:
            return False 

# calculate the input, weight, output size of each layer
# this function follows gpt model architecture, change it as needed
def calculateSizes(model, layer_name, length, init=False, fp=2):
    n_embd, n_layer, n_head, vocab_size = getConfig(model)
    if layer_name == "vocab_embedding":
        input_size = length * fp
        weight_size = vocab_size * n_embd * fp
        output_size = length * n_embd * fp
    elif layer_name in ["input_layernorm", "post_layernorm", "ln_f"]:
        input_size = length * n_embd * fp
        weight_size = 2 * n_embd * fp  # scale + bias
        output_size = length * n_embd * fp
    elif layer_name == "attention/qkv":
        input_size = length * n_embd * fp
        weight_size = n_embd * (3 * n_embd) * fp  # q, k, v
        output_size = length * (3 * n_embd) * fp
    elif layer_name == "attention/dense":
        input_size = length * n_embd * fp
        weight_size = n_embd * n_embd * fp
        output_size = length * n_embd * fp
    elif layer_name == "attention/wrapper":
        if init:
            input_size = 3 * length * n_embd * fp # Q (input) + K (input) + V (input)
            weight_size = 0
            output_size = length * n_embd * fp
        else:
            input_size = (2 * length + 1) * n_embd * fp # Q (1) + K (kv len) + V (kv len)
            weight_size = 0
            output_size = 1 * n_embd * fp
    elif layer_name == "mlp/fc":
        input_size = length * n_embd * fp
        weight_size = n_embd * (4 * n_embd) * fp
        output_size = length * (4 * n_embd) * fp
    elif layer_name == "mlp/gelu":
        input_size = length * (4 * n_embd) * fp
        weight_size = 0
        output_size = length * (4 * n_embd) * fp
    elif layer_name == "mlp/proj":
        input_size = length * (4 * n_embd) * fp
        weight_size = (4 * n_embd) * n_embd * fp
        output_size = length * n_embd * fp
    elif layer_name == "lm_head":
        input_size = length * n_embd * fp
        weight_size = n_embd * vocab_size * fp
        output_size = length * vocab_size * fp
    else:
        print("ERROR: calculateSizes: No matching layer name")
        input_size = 0
        weight_size = 0
        output_size = 0
    return input_size, weight_size, output_size