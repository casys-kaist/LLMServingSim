import os, threading
from .utils import get_config
from .radix_tree import *
import logging
from enum import Enum

GB_TO_BYTE = 1024 * 1024 * 1024
MB_TO_BYTE = 1024 * 1024
KB_TO_BYTE = 1024

class Device(Enum):
    NPU = 1
    CPU = 2
    CXL = 3

class MemoryModel():
    def __init__(self, model, instance_id, node_id, npu_num, npu_group, npu_mem, cpu_mem, block_size, fp, enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage, cxl_mem=0):
        self.model = model
        self.node_id = node_id
        self.instance_id = instance_id
        self.npu_num = npu_num
        self.npu_group = npu_group
        self.npus_per_group = npu_num // npu_group
        self.npu_mem = npu_mem * GB_TO_BYTE # GB -> Byte
        self.cpu_mem = cpu_mem * GB_TO_BYTE # GB -> Byte
        self.cxl_mem = cxl_mem * GB_TO_BYTE 
        self.block_size = block_size
        self.fp = fp // 8 # bit -> byte of floating point
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_prefix_sharing = enable_prefix_sharing
        self.prefix_storage = prefix_storage

        self.config = get_config(model)
        self.n_embd = self.config['hidden_size']
        self.n_layer = self.config['num_hidden_layers']
        self.n_head = self.config['num_attention_heads']
        self.head_dim = self.n_embd // self.n_head
        self.kv_head = self.config.get("num_key_value_heads", self.n_head)  # fallback to n_head if not defined
        self.group = self.n_head // self.kv_head  # group size
        self.kv_dim = self.n_embd // self.group   # equivalent to: kv_head * (n_embd // n_head)
        self.vocab_size = self.config['vocab_size']
        self.is_moe = True if 'num_local_experts' in self.config else False

        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)

        # Memory model
        self.weight = self.get_weight() # assume weight is loaded
        self.npu_used = self.weight
        self.cpu_used = 0
        if self.weight > self.npu_mem:
            raise RuntimeError(f"[MemoryModel] [node={self.node_id},inst={self.instance_id}]: Model size {self.weight*self.npu_num//GB_TO_BYTE}GB exceeds total NPU memory {self.npu_mem*self.npu_num//GB_TO_BYTE}GB")

        if enable_prefix_caching:
            one_token_kv_size = self.get_kv(1)
            self.mem_for_kv = self.npu_mem - self.weight
            self.npu_prefix_cache = RadixCache(device='NPU', 
                                               node_id=self.node_id,
                                               instance_id=self.instance_id,
                                               page_size=self.block_size,
                                               capacity=self.mem_for_kv,
                                               kv_size=one_token_kv_size,
                                               enable_kv_cache_events=True,
                                                )
            if prefix_storage is not None:
                if enable_prefix_sharing and prefix_pool is not None:
                    self.second_tier_prefix_cache = prefix_pool
                else:
                    prefix_cache_capacity = 0
                    if prefix_storage == Device.CPU:
                        device = "CPU"
                        prefix_cache_capacity = self.cpu_mem
                    elif prefix_storage == Device.CXL:
                        device = "CXL"
                        prefix_cache_capacity = self.cxl_mem
                    else:
                        raise RuntimeError(f"Device {prefix_storage} is currently not supported as a second tier prefix cache storage")

                    self.second_tier_prefix_cache = RadixCache(device=device, 
                                                    node_id=self.node_id,
                                                    instance_id=self.instance_id,
                                                    page_size=1,
                                                    capacity=prefix_cache_capacity,
                                                    kv_size=(one_token_kv_size * self.npu_num),
                                                    enable_kv_cache_events=True,
                                                    )
                
        # Hash id -> token length for corresponding prefix cache block
        self._npu_cache_hashtolen = {}
        self._cpu_cache_hashtolen = {}
        self._bytes_per_token = self.get_kv(1)  # bytes per token for kv cache
    # get weight of the model 
    def get_weight(self):
        cwd = os.getcwd()
        weight = 0

        # embedding
        _, embedding, _ = calculate_sizes(self.model, 'embedding', 1, tp=self.npus_per_group, fp=self.fp)
        weight += embedding

        # block
        block_weight = 0
        # input layernorm
        _, input_ln, _ = calculate_sizes(self.model, 'input_layernorm', 1, tp=self.npus_per_group, fp=self.fp)
        block_weight += input_ln
        # qkv
        _, q, _ = calculate_sizes(self.model, 'q_proj', 1, tp=self.npus_per_group, fp=self.fp)
        block_weight += q
        _, k, _ = calculate_sizes(self.model, 'k_proj', 1, tp=self.npus_per_group, fp=self.fp)
        block_weight += k
        _, v, _ = calculate_sizes(self.model, 'v_proj', 1, tp=self.npus_per_group, fp=self.fp)
        block_weight += v
        # attention dense
        _, attn_dns, _ = calculate_sizes(self.model, 'o_proj', 1, tp=self.npus_per_group, fp=self.fp)
        block_weight += attn_dns
        if self.is_moe:
            # gate function for MoE
            _, gate, _ = calculate_sizes(self.model, 'gate', 1, tp=self.npus_per_group, fp=self.fp)
            block_weight += gate
            # MoE experts
            _, w1, _ = calculate_sizes(self.model, 'expert.w1', 1, tp=self.npus_per_group, fp=self.fp)
            block_weight += w1 
            _, w2, _ = calculate_sizes(self.model, 'expert.w2', 1, tp=self.npus_per_group, fp=self.fp)
            block_weight += w2
            _, w3, _ = calculate_sizes(self.model, 'expert.w3', 1, tp=self.npus_per_group, fp=self.fp)
            block_weight += w3

        else:
            _, ffn1, _ = calculate_sizes(self.model, 'gate_proj', 1, tp=self.npus_per_group, fp=self.fp)
            block_weight += ffn1
            _, ffn2, _ = calculate_sizes(self.model, 'up_proj', 1, tp=self.npus_per_group, fp=self.fp)
            block_weight += ffn2
            _, ffn3, _ = calculate_sizes(self.model, 'down_proj', 1, tp=self.npus_per_group, fp=self.fp)
            block_weight += ffn3
   
        # post layernorm
        _, post_ln, _ = calculate_sizes(self.model, 'post_layernorm', 1, tp=self.npus_per_group, fp=self.fp)
        block_weight += post_ln

        weight += block_weight * self.n_layer

        # ln_f
        _, ln_f, _ = calculate_sizes(self.model, 'final_layernorm', 1, tp=self.npus_per_group, fp=self.fp)
        weight += ln_f
        # lm_head
        _, lm_head, _ = calculate_sizes(self.model, 'lm_head', 1, tp=self.npus_per_group, fp=self.fp)
        weight += lm_head

        self.logger.info(
            "NPU: model weight %dMB loaded",
            weight * self.npus_per_group // MB_TO_BYTE,
        )

        return weight


    def get_kv(self, seq):
        # shape of kv cache
        # (kv_head, batch_size, n_embd//n_head, seq_len) per layer
        # return batch_size = 1 to caclulate max batch_size in scheduler

        # K & V multiply 2 
        return 2 * self.kv_dim * seq * self.n_layer * self.fp // self.npu_num
    
    # get the total size of current kv cache for the request
    # used when adding prefilled request to decode instance.
    def get_total_kv(self, req):
        num_blocks = (req.input - 1) // self.block_size + 1 # decode instance will add new block if needed
        return self.get_kv(num_blocks * self.block_size)

    # get size of kv block that should be 'added'. including new init requests
    # also checks evicted request and include its kv cache
    def get_block_kv(self, batch_req, batch_len):
        block_kv_size = 0
        for i in range(batch_len):
            if batch_req[i].evict or batch_req[i].is_init:
                hit = getattr(batch_req[i], 'npu_cache_hit', 0) if self.enable_prefix_caching else 0
                needed = max(0, batch_req[i].input - hit)
                num_blocks = needed // self.block_size + 1 # it includes kv_cache that will be generated in current iteration
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
        hit = getattr(req, 'npu_cache_hit', 0) if self.enable_prefix_caching else 0
        needed = max(0, req.input - hit)
        num_blocks = (needed-1) // self.block_size + 1
        evict_size += self.get_kv(num_blocks * self.block_size)
        return evict_size

    def free_weight(self):
        if self.npu_used - self.weight < 0:
            raise RuntimeError(
                f"[MemoryModel] [node={self.node_id}, inst={self.instance_id}] NPU: tried to free model weight {self.weight / MB_TO_BYTE:.2f}MB "
                f"but only {self.npu_used / MB_TO_BYTE:.2f}MB is used."
            )
        self.logger.info(
            "NPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
            self.npu_used / MB_TO_BYTE,
            self.weight / MB_TO_BYTE,
            (self.npu_used - self.weight) / MB_TO_BYTE,
        )
        self.npu_used -= self.weight

    def is_free(self):
        return self.npu_used == 0 and self.cpu_used == 0

    # -------------------- Memory Management --------------------
    
    def allocate(self, size, device):
        if device == Device.NPU:
            if self.npu_used + size > self.npu_mem:
                raise RuntimeError(
                    f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] NPU: tried to load {size / MB_TO_BYTE:.2f}MB but only {(self.npu_mem - self.npu_used) / MB_TO_BYTE:.2f}MB is available."
                )
            self.logger.info(
                "NPU: used: %.2fMB load: %.2fMB after: %.2fMB",
                self.npu_used / MB_TO_BYTE,
                size / MB_TO_BYTE,
                (self.npu_used + size) / MB_TO_BYTE,
            )
            self.npu_used += size
        elif device == Device.CPU:
            if self.prefix_storage == Device.CPU and self.enable_prefix_sharing:
                self.second_tier_prefix_cache.allocate(size)
            else:
                if self.cpu_used + size > self.cpu_mem:
                    raise RuntimeError(
                        f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] CPU: tried to load {size / MB_TO_BYTE:.2f}MB "
                        f"but only {(self.cpu_mem - self.cpu_used) / MB_TO_BYTE:.2f}MB is available."
                    )
                self.logger.info(
                    "CPU: used: %.2fMB load: %.2fMB after: %.2fMB",
                    self.cpu_used / MB_TO_BYTE,
                    size / MB_TO_BYTE,
                    (self.cpu_used + size) / MB_TO_BYTE,
                )
                self.cpu_used += size
        elif device == Device.CXL:
            self.second_tier_prefix_cache.allocate(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to allocate KV cache in unsupported device {device}")
    
    def free(self, size, device):
        if device == Device.NPU:
            if self.npu_used - size < self.weight:
                raise RuntimeError(
                    f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] NPU: tried to free {size / MB_TO_BYTE:.2f}MB but only {(self.npu_used - self.weight) / MB_TO_BYTE:.2f}MB is used."
                )
            self.logger.info(
                "NPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
                self.npu_used / MB_TO_BYTE,
                size / MB_TO_BYTE,
                (self.npu_used - size) / MB_TO_BYTE,
            )
            self.npu_used -= size

        elif device == Device.CPU:
            if self.prefix_storage == Device.CPU and self.enable_prefix_sharing:
                self.second_tier_prefix_cache.free(size)
            else:
                if self.cpu_used - size < 0:
                    raise RuntimeError(
                        f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] CPU: tried to free {size / MB_TO_BYTE:.2f}MB "
                        f"but only {self.cpu_used / MB_TO_BYTE:.2f}MB is used."
                    )
                self.logger.info(
                    "CPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
                    self.cpu_used / MB_TO_BYTE,
                    size / MB_TO_BYTE,
                    (self.cpu_used - size) / MB_TO_BYTE,
                )
                self.cpu_used -= size
        elif device == Device.CXL:
            self.second_tier_prefix_cache.free(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to free KV cache in unsupported device {device}")
    
    def is_avail(self, size, device):
        if device == Device.NPU:
            if self.npu_mem - self.npu_used >= size:
                return True
            else:
                return False 
        elif device == Device.CPU:
            if self.enable_prefix_sharing:
                return self.second_tier_prefix_cache.is_avail(size)
            else:
                if self.cpu_mem - self.cpu_used >= size:
                    return True
                else:
                    return False 
        elif device == Device.CXL:
            return self.second_tier_prefix_cache.is_avail(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to check available size of unsupported device {device}")
    
    def need_size(self, size, device):
        if device == Device.NPU:
            needed = (size - (self.npu_mem - self.npu_used))
            if needed > 0:
                return needed
            else:
                return 0
        elif device == Device.CPU:
            if self.enable_prefix_sharing:
                return self.second_tier_prefix_cache.need_size(size)
            else:
                needed = (size - (self.cpu_mem - self.cpu_used))
                if needed > 0:
                    return needed
                else:
                    return 0
        elif device == Device.CXL:
            return self.second_tier_prefix_cache.need_size(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to check available size of unsupported device {device}")

    def avail_size(self, device):
        if not self.enable_prefix_caching:
            return 0
        
        if device == Device.NPU:
            return self.npu_prefix_cache.avail_size() * self._bytes_per_token
        elif device == Device.CPU or device == Device.CXL:
            return self.second_tier_prefix_cache.avail_size() * self._bytes_per_token
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to get available size of prefix cache in unsupported device {device}")
    
    # -------------------- Prefix Cache Management --------------------

    def storage_cache_evicted_req(self, req):
        if self.enable_prefix_caching:
            new_last_node = self.second_tier_prefix_cache.cache_unfinished_req(req, update=False) # do not update hit counts
            # should lock evicted kv cache in cpu
            self.npu_prefix_cache.inc_lock_ref(new_last_node)
            req.cpu_last_node = new_last_node
            self.apply_kv_cache_events()

    def evictable_size(self, device):
        if not self.enable_prefix_caching:
            return 0
        
        if device == Device.NPU:
            return self.npu_prefix_cache.evictable_size() * self._bytes_per_token
        elif device == Device.CPU or device == Device.CXL:
            return self.second_tier_prefix_cache.evictable_size() * self._bytes_per_token
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to get evictable size of prefix cache in unsupported device {device}")


    def lock_prefix(self, req, device):
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU and getattr(req, "npu_last_node", None) is not None:
            self.npu_prefix_cache.inc_lock_ref(req.npu_last_node)
        elif (device == Device.CPU or device == Device.CXL) and getattr(req, "cpu_last_node", None) is not None:
            self.second_tier_prefix_cache.inc_lock_ref(req.cpu_last_node)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to lock prefix cache in unsupported device {device}")
    
    def unlock_prefix(self, req, device):
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU and getattr(req, "npu_last_node", None) is not None:
            self.npu_prefix_cache.dec_lock_ref(req.npu_last_node)
            req.npu_last_node = None
        elif device == Device.CPU and getattr(req, "cpu_last_node", None) is not None:
            self.second_tier_prefix_cache.dec_lock_ref(req.cpu_last_node)
            req.cpu_last_node = None
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to unlock prefix cache in unsupported device {device}")
    
    def cache_unfinished_req(self, req, device):
        if not self.enable_prefix_caching:
            return

        if device == Device.NPU:
            new_last_node = self.npu_prefix_cache.cache_unfinished_req(req)
            self.npu_prefix_cache.dec_lock_ref(req.npu_last_node)
            self.npu_prefix_cache.inc_lock_ref(new_last_node)
            req.npu_last_node = new_last_node
            if self.logger.isEnabledFor(logging.DEBUG):
                print(f"cache_unfinished_req of req {req.id}")
                print(f"===============NPU PREFIX CAHCE of Instance[{self.instance_id}]=================")
                self.npu_prefix_cache.pretty_print()
        elif device == Device.CPU or device == Device.CXL:
            self.second_tier_prefix_cache.cache_unfinished_req(req)
            if self.logger.isEnabledFor(logging.DEBUG):
                print(f"cache_unfinished_req of req {req.id}")
                print(f"===============AFTER INSERT: {self.second_tier_prefix_cache.device} PREFIX CAHCE at pid={os.getpid()} tid={threading.get_ident()} pool_id={id(self.second_tier_prefix_cache)}, size={self.second_tier_prefix_cache.total_size()}=================")
                self.second_tier_prefix_cache.pretty_print()
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to cache prefix cache of unfinished request to unsupported device {device}")
        
        self.apply_kv_cache_events()

    def cache_finished_req(self, req, device):
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU:
            self.npu_prefix_cache.cache_finished_req(req)
            self.npu_prefix_cache.dec_lock_ref(req.npu_last_node)
            if self.logger.isEnabledFor(logging.DEBUG):
                print(f"cache_finished_req of req {req.id}")
                print(f"===============NPU PREFIX CAHCE of Instance[{self.instance_id}]=================")
                self.npu_prefix_cache.pretty_print()
        elif device == Device.CPU or device == Device.CXL:
            self.second_tier_prefix_cache.cache_finished_req(req)
            if self.logger.isEnabledFor(logging.DEBUG):
                print(f"cache_finished_req of req {req.id}")
                print(f"===============AFTER INSERT: {self.second_tier_prefix_cache.device} PREFIX CAHCE at pid={os.getpid()} tid={threading.get_ident()} pool_id={id(self.second_tier_prefix_cache)}, size={self.second_tier_prefix_cache.total_size()}=================")
                self.second_tier_prefix_cache.pretty_print()
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to cache prefix cache of finished request to unsupported device {device}")
        
        self.apply_kv_cache_events()

    def evict_prefix_cache(self, bytes, device):
        if not self.enable_prefix_caching and bytes <= 0:
            return
        
        space_needed = (bytes + self._bytes_per_token - 1) // self._bytes_per_token

        if device == Device.NPU:
            self.npu_prefix_cache.evict(space_needed)
        elif device == Device.CPU:
            self.second_tier_prefix_cache.evict(space_needed)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to evict prefix cache to unsupported device {device}")

        self.apply_kv_cache_events()

    # -------------------- Prefix Cache Helpers --------------------

    def prefix_match(self, req):
        if not self.enable_prefix_caching:
            return
        
        tokens = getattr(req, 'input_hash_ids', None)
        if tokens is None:
            return
        res = self.npu_prefix_cache.match_prefix(tokens[:req.input])
        req.npu_cache_hit = res.hit_length
        req.npu_last_node = res.last_device_node

        if self.prefix_storage is not None:
            res_storage = self.second_tier_prefix_cache.match_prefix(tokens[:req.input])
            req.storage_cache_hit = res_storage.hit_length
            req.storage_last_node = res_storage.last_device_node
        else:
            req.storage_cache_hit = 0
            req.storage_last_node = None
        
        req.prefix_cache_hit = max(req.npu_cache_hit, req.storage_cache_hit)

        # for debugging
        self.logger.info(f"Request[{req.id}] prefix cache hit: {req.prefix_cache_hit} tokens (NPU: {req.npu_cache_hit}, {self.prefix_storage}: {req.storage_cache_hit})")
        # print(f"===============NPU PREFIX CAHCE of Instance[{self.instance_id}]=================")
        # self.npu_prefix_cache.pretty_print()
        # print("===============CPU PREFIX CAHCE=================")
        # self.second_tier_prefix_cache.pretty_print()
    
    def erase_prefix_info(self, req):
        if not self.enable_prefix_caching:
            return
        
        req.prefix_cache_hit = 0
        req.npu_cache_hit = 0
        req.storage_cache_hit = 0
        req.npu_last_node = None
        req.storage_last_node = None

    def free_prefix_cache(self):
        if not self.enable_prefix_caching:
            return
        # free evictable prefix cache, if evictable_size != total_size there is locked prefix cache
        self.free(self.npu_prefix_cache.evictable_size() * self._bytes_per_token, Device.NPU)
        if not self.enable_prefix_sharing and self.prefix_storage is not None:
            self.free(self.second_tier_prefix_cache.evictable_size() * self._bytes_per_token * self.npu_num, self.prefix_storage)
    
    # Count load/unload events from prefix cache and update memory usage
    def apply_kv_cache_events(self):
        # if not self.enable_prefix_caching:
        #     return
        npu_byte_alloc = 0
        npu_byte_free = 0
        cpu_byte_alloc = 0
        cpu_byte_free = 0
        for ev in self.npu_prefix_cache.take_events():
            if isinstance(ev, BlockStored):
                tlen = len(ev.token_ids)
                for h in ev.block_hashes:
                    self._npu_cache_hashtolen[h] = tlen
                npu_byte_alloc += self.get_kv(tlen)
            elif isinstance(ev, BlockRemoved):
                for h in ev.block_hashes:
                    tlen = self._npu_cache_hashtolen.pop(h, 0)
                    if tlen == 0:
                        self.logger.warning("NPU prefix cache remove unknown block hash {h}")
                    npu_byte_free += self.get_kv(tlen)
        
        if npu_byte_alloc > 0:
            self.allocate(npu_byte_alloc, Device.NPU)
        if npu_byte_free > 0:
            self.free(npu_byte_free, Device.NPU)

        if not self.enable_prefix_sharing and self.prefix_storage is Device.CPU:
            for ev in self.second_tier_prefix_cache.take_events():
                if isinstance(ev, BlockStored):
                    tlen = len(ev.token_ids)
                    for h in ev.block_hashes:
                        self._cpu_cache_hashtolen[h] = tlen
                    cpu_byte_alloc += self.get_kv(tlen) * self.npu_num
                elif isinstance(ev, BlockRemoved):
                    for h in ev.block_hashes:
                        tlen = self._cpu_cache_hashtolen.pop(h, 0)
                        cpu_byte_free += self.get_kv(tlen) * self.npu_num

            if cpu_byte_alloc > 0:
                self.allocate(cpu_byte_alloc, Device.CPU)
            if cpu_byte_free > 0:
                self.free(cpu_byte_free, Device.CPU)

    def return_prefix_info(self):
        if not self.enable_prefix_caching:
            return (0, 0, 0, 0)
        if self.prefix_storage is None:
            return (self.npu_prefix_cache.return_prefix_info(), (0, 0))
        return (self.npu_prefix_cache.return_prefix_info(), self.second_tier_prefix_cache.return_prefix_info())

        
# calculate the per-rank input, weight, output size of each layer
def calculate_sizes(model, layer_name, length, kv_len=None, pim=False, tp=1, fp=2):
    config = get_config(model)
    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    head_dim = n_embd // n_head
    vocab_size = config['vocab_size']
    kv_head = config.get("num_key_value_heads", n_head)  # fallback to n_head if not defined
    group = n_head // kv_head  # group size for GQA
    kv_dim = n_embd // group   # = kv_head * head_dim
    ffn_dim = config.get("intermediate_size", config.get("ffn_dim"))  # ffn_dim or intermediate_size
    num_local_experts = config.get("num_local_experts", 1)

    # Tensor parallel degree (also used as EP degree for experts)
    tp = max(int(tp), 1)

    # NOTE (vLLM-style assumptions):
    # - Embedding / LM head: vocab-parallel  → split vocab_size across TP ranks.
    # - Q/K/V: ColumnParallelLinear          → split output dim across TP ranks.
    # - o_proj: RowParallelLinear           → split input dim across TP ranks.
    # - LayerNorm weights: replicated (NOT TP-sharded).
    # - Activations: always "per-rank" sizes:
    #       * if tensor is sharded on a dim → that dim // tp
    #       * if tensor is replicated      → that dim as-is.
    # - MoE experts: EP degree == TP degree (ep = tp),
    #       so each rank holds num_local_experts // tp experts.

    # ----------------- Embedding & Norms -----------------
    if layer_name == "embedding":
        # VocabParallelEmbedding:
        #   per-rank weight: [vocab_size // tp, n_embd]
        input_size = length * fp * 2  # token_ids are int32 or int64
        weight_size = (vocab_size // tp) * n_embd * fp
        # embedding output is NOT TP-sharded over hidden dim → full n_embd per rank
        output_size = length * n_embd * fp

    elif layer_name in ["input_layernorm", "post_layernorm", "final_layernorm"]:
        # LayerNorm / RMSNorm weights are replicated across TP ranks.
        # Activations are also treated as full hidden per rank (no sharding here).
        input_size = length * n_embd * fp
        weight_size = 1 * n_embd * fp  # scale only
        output_size = length * n_embd * fp

    # ----------------- Q/K/V Projections -----------------
    elif layer_name == "q_proj":
        # ColumnParallelLinear on output dim:
        #   input  : [L, n_embd]          (replicated)
        #   weight : [n_embd, n_embd // tp]
        #   output : [L, n_embd // tp]    (sharded)
        input_size = length * n_embd * fp
        weight_size = n_embd * (n_embd // tp) * fp
        # output_size = length * (n_embd // tp) * fp
        output_size = length * n_embd * fp # keep same size with k_proj input

    elif layer_name == "k_proj":
        # ColumnParallelLinear with GQA:
        #   input  : [L, n_embd]
        #   weight : [n_embd, kv_dim // tp]
        #   output : [L, kv_dim // tp]
        input_size = length * n_embd * fp
        weight_size = n_embd * (kv_dim // tp) * fp  # kv_dim is split over TP
        # output_size = length * (kv_dim // tp) * fp
        output_size = length * n_embd * fp # keep same size with v_proj input

    elif layer_name == "v_proj":
        # Same scheme as k_proj
        input_size = length * n_embd * fp
        weight_size = n_embd * (kv_dim // tp) * fp
        output_size = length * (kv_dim // tp) * fp

    # ----------------- RoPE & Attention Core -----------------
    elif layer_name == "rope":  # only for LLaMA
        # RoPE is weight-free and applied per rank on local heads:
        #   Q: (n_head // tp), K: (kv_head // tp)
        input_size = ((n_head // tp) + (kv_head // tp)) * length * head_dim * fp
        weight_size = 0
        output_size = ((n_head // tp) + (kv_head // tp)) * length * head_dim * fp

    elif layer_name == "attn":  # only for LLaMA, attention internal buffers
        # All attention internals are per-rank on local heads.
        if not pim:
            # prefill:
            #   q: [(n_head // tp), L, head_dim]
            #   k: [(kv_head // tp), L, head_dim]
            #   v: [(kv_head // tp), L, head_dim]
            input_size = (
                (n_head // tp) * length * head_dim * fp +
                (kv_head // tp) * kv_len * head_dim * fp * 2
            )
            weight_size = 0
            # output: [(n_head // tp), L, head_dim]
            output_size = (n_head // tp) * length * head_dim * fp
        else:
            # decode + PIM: q(1), small KV window(1) per rank
            input_size = (
                (n_head // tp) * 1 * head_dim * fp +
                (kv_head // tp) * 1 * head_dim * fp * 2
            )
            weight_size = 0
            output_size = (n_head // tp) * 1 * head_dim * fp

    # ----------------- Output Projection -----------------
    elif layer_name == "o_proj":
        # RowParallelLinear:
        #   per rank input : [L, n_embd // tp]
        #   per rank weight: [n_embd // tp, n_embd]
        #   per rank output: [L, n_embd] (after all-reduce)
        input_size = length * (n_embd // tp) * fp
        weight_size = (n_embd // tp) * n_embd * fp
        output_size = length * n_embd * fp

# ----------------- Dense FFN (non-MoE) -----------------
    elif layer_name == "gate_proj":
        # LLaMA-style SwiGLU gate projection (dense FFN, non-MoE):
        # ColumnParallelLinear:
        #   input  : [L, n_embd]
        #   weight : [n_embd, ffn_dim // tp]
        #   output : [L, ffn_dim // tp]
        input_size = length * n_embd * fp
        weight_size = n_embd * (ffn_dim // tp) * fp
        # output_size = length * (ffn_dim // tp) * fp 
        output_size = length * n_embd * fp # keep same size with up_proj input

    elif layer_name == "up_proj":
        # LLaMA up_proj (dense FFN, non-MoE), same scheme as gate_proj.
        input_size = length * n_embd * fp
        weight_size = n_embd * (ffn_dim // tp) * fp
        output_size = length * (ffn_dim // tp) * fp

    elif layer_name == "fc1":
        # Generic FFN first linear (dense):
        #   input  : [L, n_embd]
        #   weight : [n_embd, ffn_dim // tp]
        #   output : [L, ffn_dim // tp]
        input_size = length * n_embd * fp
        weight_size = n_embd * (ffn_dim // tp) * fp
        output_size = length * (ffn_dim // tp) * fp

    elif layer_name == "act_fn":
        # Activation is element-wise, no weights.
        input_size = length * (ffn_dim // tp) * fp
        weight_size = 0
        output_size = length * (ffn_dim // tp) * fp

    elif layer_name == "down_proj":
        # LLaMA down_proj (dense FFN, non-MoE):
        # RowParallelLinear:
        #   per rank input : [L, ffn_dim // tp]
        #   per rank weight: [ffn_dim // tp, n_embd]
        #   per rank output: [L, n_embd] (after all-reduce)
        input_size = length * (ffn_dim // tp) * fp
        weight_size = (ffn_dim // tp) * n_embd * fp
        output_size = length * n_embd * fp

    elif layer_name == "fc2":
        # Generic FFN second linear (dense):
        # RowParallelLinear:
        #   per rank input : [L, ffn_dim // tp]
        #   per rank weight: [ffn_dim // tp, n_embd]
        #   per rank output: [L, n_embd]
        input_size = length * (ffn_dim // tp) * fp
        weight_size = (ffn_dim // tp) * n_embd * fp
        output_size = length * n_embd * fp

    # ----------------- MoE Gate -----------------
    elif layer_name == "gate":  # gate function for MoE
        # Assume gate is replicated across TP ranks.
        #   per rank input : [L, n_embd]
        #   per rank weight: [n_embd, num_local_experts]
        input_size = length * n_embd * fp
        weight_size = n_embd * num_local_experts * fp
        # gate logits are over all experts (replicated per rank)
        # output_size = length * num_local_experts * fp
        output_size = length * n_embd* fp  # keep same size with expert input
    # ----------------- MoE Experts (EP degree = TP degree) -----------------
    # Only expert.w1 / expert.w2 / expert.w3 are MoE.
    elif layer_name == "expert.w1":
        # Expert W1: [n_embd, ffn_dim] per expert.
        # Per rank, we assume (num_local_experts // tp) experts.
        input_size = length * n_embd * fp
        weight_size = n_embd * ffn_dim * fp * (num_local_experts // tp)
        # output of W1 for routed tokens: model with full ffn_dim (no TP inside experts)
        output_size = length * ffn_dim * fp

    elif layer_name == "expert.w2":
        # Second linear inside expert, same (n_embd -> ffn_dim) shape per expert.
        input_size = length * n_embd * fp
        weight_size = n_embd * ffn_dim * fp * (num_local_experts // tp)
        output_size = length * ffn_dim * fp

    elif layer_name == "expert.w3":
        # Final projection back to hidden dim:
        #   [ffn_dim, n_embd] per expert
        input_size = length * ffn_dim * fp
        weight_size = ffn_dim * n_embd * fp * (num_local_experts // tp)
        output_size = length * n_embd * fp

    # ----------------- LM Head -----------------
    elif layer_name == "lm_head":
        # ParallelLMHead (vocab-parallel):
        #   per-rank weight: [n_embd, vocab_size // tp]
        #   per-rank logits: [L, vocab_size // tp]
        input_size = length * n_embd * fp
        weight_size = n_embd * (vocab_size // tp) * fp
        output_size = length * (vocab_size // tp) * fp

    else:
        raise ValueError(f"No matching layer name {layer_name} found for model {model}.")

    return input_size, weight_size, output_size