import os
from .utils import get_config

class MemoryModel():
    def __init__(self, model, npu_num, npu_mem, block_size, fp, verbose=False):
        self.model = model
        self.npu_num = npu_num
        self.npu_mem = npu_mem
        self.block_size = block_size
        self.fp = fp // 8 # bit -> byte of floating point
        self.verbose = verbose

        self.config = get_config(model)
        self.n_embd = self.config['hidden_size']
        self.n_layer = self.config['num_hidden_layers']
        self.n_head = self.config['num_attention_heads']
        self.vocab_size = self.config['vocab_size']

        # assume NPUS use identical memory
        self.npu_mem = npu_mem * 1000000000

        # Memory model
        self.weight = self.get_weight() # assume weight is loaded
        self.kv_npu = self.npu_num # npus that store KV cache
        self.used_mem = self.weight


    # get weight of the model 
    def get_weight(self):
        cwd = os.getcwd()
        weight = 0

        # embedding
        _, embedding, _ = calculate_sizes(self.model, 'embedding', 1)
        weight += embedding

        # block
        block_weight = 0
        # input layernorm
        _, input_ln, _ = calculate_sizes(self.model, 'input_layernorm', 1)
        block_weight += input_ln
        # qkv
        _, q, _ = calculate_sizes(self.model, 'q_proj', 1)
        block_weight += q
        _, k, _ = calculate_sizes(self.model, 'k_proj', 1)
        block_weight += k
        _, v, _ = calculate_sizes(self.model, 'v_proj', 1)
        block_weight += v
        # attention dense
        _, attn_dns, _ = calculate_sizes(self.model, 'o_proj', 1)
        block_weight += attn_dns
        if 'llama' in self.model.lower():
            _, ffn1, _ = calculate_sizes(self.model, 'gate_proj', 1)
            block_weight += ffn1
            _, ffn2, _ = calculate_sizes(self.model, 'up_proj', 1)
            block_weight += ffn2
            _, ffn3, _ = calculate_sizes(self.model, 'down_proj', 1)
            block_weight += ffn3
        else:
        # mlp fc
            _, ffn1, _ = calculate_sizes(self.model, 'fc1', 1)
            block_weight += ffn1
            # mlp proj
            _, ffn2, _ = calculate_sizes(self.model, 'fc2', 1)
            block_weight += ffn2
   
        # post layernorm
        _, post_ln, _ = calculate_sizes(self.model, 'post_layernorm', 1)
        block_weight += post_ln

        weight += block_weight * self.n_layer

        # ln_f
        _, ln_f, _ = calculate_sizes(self.model, 'final_layernorm', 1)
        weight += ln_f
        # lm_head
        _, lm_head, _ = calculate_sizes(self.model, 'lm_head', 1)
        weight += lm_head

        if self.verbose:
            print(f"Memory: model weight {weight//1024//1024}MB loaded")

        return weight // self.npu_num


    def get_kv(self, seq):
        # shape of kv cache
        # (n_head, batch_size, n_embd//n_head, seq_len) per layer
        # return batch_size = 1 to caclulate max batch_size in scheduler

        # K & V multiply 2 
        return 2 * self.n_embd * seq * self.n_layer * self.fp // self.kv_npu
    
    # used when batching. in case of vllm, it is only used in init phase
    def get_batch_kv(self, batch_req, batch_len):
        batch_kv_size = 0
        for i in range(batch_len):
            num_blocks = batch_req[i].input // self.block_size + 1 # it includes kv_cache that will be generated in current iteration
            batch_kv_size += self.get_kv(num_blocks * self.block_size)

        return batch_kv_size

    # get size of kv block that should be added. used in vllm gen phase
    # also checks evicted request and include its kv cache
    def get_block_kv(self, batch_req, batch_len):
        block_kv_size = 0
        for i in range(batch_len):
            if batch_req[i].evict or batch_req[i].is_init:
                num_blocks = batch_req[i].input // self.block_size + 1 # it includes kv_cache that will be generated in current iteration
                block_kv_size += self.get_kv(num_blocks * self.block_size)
            else:
                num_before = (batch_req[i].input - 1) // self.block_size + 1
                num_after = batch_req[i].input // self.block_size + 1 # it includes kv_cache that will be generated in current iteration
                if num_after > num_before: # difference of the block is maximum one block
                    block_kv_size += self.get_kv(self.block_size)
        
        return block_kv_size
    
    # get size of kv cache that should be evicted
    def get_evict_kv(self, req):
        evict_size = 0
        # input + 1 is not loaded now
        num_blocks = (req.input-1) // self.block_size + 1
        evict_size += self.get_kv(num_blocks * self.block_size)
        return evict_size
    
    def mem_load(self, size):
        if self.used_mem + size > self.npu_mem:
            print("ERROR: memLoad: no memory to load")
        if self.verbose:
            print(f"Memory: used: {self.used_mem} load: {size}", end=' ')
        self.used_mem += size
        if self.verbose:
            print(f"after: {self.used_mem}")

    def mem_store(self, size):
        if self.used_mem - size < self.weight:
            print("ERROR: memStore: no memory to unload")
        if self.verbose:
            print(f"Memory: used: {self.used_mem} remove: {size}", end=' ')
        self.used_mem -= size
        if self.verbose:
            print(f"after: {self.used_mem}")

    def mem_avail(self, size):
        if self.npu_mem - self.used_mem >= size:
            return True
        else:
            return False 

# calculate the input, weight, output size of each layer
def calculate_sizes(model, layer_name, length, init=False, fp=2):
    config = get_config(model)
    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    head_dim = n_embd // n_head
    vocab_size = config['vocab_size']
    kv_head = config.get("num_key_value_heads", n_head)  # fallback to n_head if not defined
    group = n_head // kv_head  # group size
    kv_dim = n_embd // group   # equivalent to: kv_head * (n_embd // n_head)
    ffn_dim = config.get("intermediate_size", config.get("ffn_dim")) # config conatins ffn_dim or intermediate_size
    
    if layer_name == "embedding":
        input_size = length * fp
        weight_size = vocab_size * n_embd * fp
        output_size = length * n_embd * fp
    elif layer_name in ["input_layernorm", "post_layernorm", "final_layernorm"]:
        input_size = length * n_embd * fp
        if "llama" in model.lower(): # llama use RMSNorm
            weight_size = 1 * n_embd * fp  # only scale
        else:
            weight_size = 2 * n_embd * fp  # scale + bias
        output_size = length * n_embd * fp
    elif layer_name == "q_proj":
        input_size = length * n_embd * fp
        weight_size = n_embd * n_embd * fp
        output_size = length * n_embd * fp
    elif layer_name == "k_proj":
        input_size = length * n_embd * fp
        weight_size = n_embd * kv_dim * fp # kv_dim if GQA is used
        output_size = length * n_embd * fp # orignal "length * kv_dim * fp" but to match V_Projection input
    elif layer_name == "v_proj":
        input_size = length * n_embd * fp
        weight_size = n_embd * kv_dim * fp # kv_dim if GQA is used
        output_size = length * n_embd * fp
    elif layer_name == "rope": # only for llama
        input_size =  (n_head + kv_head) * length * head_dim * fp # input q, k
        weight_size = 0
        output_size = (n_head + kv_head) * length * head_dim * fp # output q, k
    elif layer_name == "attn": # only for llama
        if init:
            input_size = n_head * length * head_dim * fp * 3 # q (input) + k (input) + v (input)
            weight_size = 0
            output_size = n_head * length * head_dim * fp
        else:
            input_size = n_head * 1 * head_dim * fp + n_head * head_dim * length * fp * 2 # q (input) + k (input) + v (input)
            weight_size = 0
            output_size = n_head * 1 * head_dim * fp
    elif layer_name == "qk_matmul":
        if init:
            input_size = n_head * length * head_dim * fp + n_head * head_dim * length * fp # q (input) + k (input)
            weight_size = 0
            output_size = n_head * length * length * fp
        else:
            input_size = n_head * 1 * head_dim * fp + n_head * head_dim * length * fp # q (input) + k (input)
            weight_size = 0
            output_size = n_head * 1 * length * fp
    elif layer_name == "softmax":
        if init:
            input_size = n_head * length * length * fp # output of QK_Matmul
            weight_size = 0
            output_size = n_head * length * length * fp
        else:
            input_size = n_head * 1 * length * fp
            weight_size = 0
            output_size = n_head * 1 * length * fp
    elif layer_name == "sv_matmul":
        if init:
            input_size = n_head * length * length * fp + n_head * length * head_dim * fp # s (output of Softmax) + v (input)
            weight_size = 0
            output_size = n_head * length * head_dim * fp
        else:
            input_size = n_head * 1 * length * fp + n_head * length * head_dim * fp
            weight_size = 0
            output_size = n_head * 1 * head_dim * fp
    elif layer_name == "o_proj":
        input_size = length * n_embd * fp
        weight_size = n_embd * n_embd * fp
        output_size = length * n_embd * fp
    elif layer_name == "gate_proj": # only for llama
        input_size = length * n_embd * fp
        weight_size = n_embd * ffn_dim * fp
        output_size = length * ffn_dim * fp
    elif layer_name == "up_proj": # only for llama
        input_size = length * ffn_dim * fp # original "length * n_embd * fp" but to match gate_proj output
        weight_size = n_embd * ffn_dim * fp
        output_size = length * ffn_dim * fp
    elif layer_name == "fc1":
        input_size = length * n_embd * fp
        weight_size = n_embd * ffn_dim * fp
        output_size = length * ffn_dim * fp
    elif layer_name == "act_fn":
        input_size = length * ffn_dim * fp
        weight_size = 0
        output_size = length * ffn_dim * fp
    elif layer_name == "down_proj": # only for llama
        input_size = length * ffn_dim * fp
        weight_size = ffn_dim * n_embd * fp
        output_size = length * n_embd * fp
    elif layer_name == "fc2":
        input_size = length * ffn_dim * fp
        weight_size = ffn_dim * n_embd * fp
        output_size = length * n_embd * fp
    elif layer_name == "lm_head":
        input_size = length * n_embd * fp
        weight_size = n_embd * vocab_size * fp
        output_size = length * vocab_size * fp
    else:
        print(f"ERROR: calculate_sizes: No matching layer name {layer_name} found for model {model}")
        input_size = 0
        weight_size = 0
        output_size = 0
    return input_size, weight_size, output_size