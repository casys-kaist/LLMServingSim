import os
from .utils import get_config
from .block_pool import Device, BlockPool, PrefixCacheStats
from .kv_cache_manager import TieredKVCacheManager, request_block_hashes
from .logger import get_logger

GB_TO_BYTE = 1024 * 1024 * 1024
MB_TO_BYTE = 1024 * 1024
KB_TO_BYTE = 1024

# LMCache's default chunk size. A host offload tier stores whole chunks of this
# many tokens, which must be a multiple of the NPU block size -- the tier keys on
# every (chunk // block_size)-th hash of the same chain. vLLM's own
# OffloadingConnector defaults to a factor of 1; 256 is kept because it matches
# LMCache and preserves the shared pool's existing granularity.
LOWER_TIER_CHUNK_TOKENS = 256


class MemoryModel():
    """Per-instance memory accounting, over one block pool per tier.

    Static sizing math (weights, per-layer tensor shapes, KV bytes per token)
    stays here because ``trace_generator`` and ``__main__`` import it. Everything
    about *which* blocks exist and *where* they live belongs to
    :class:`TieredKVCacheManager`, so there is exactly one ledger per tier and
    an allocation that cannot be satisfied says so in the call that asks.
    """

    def __init__(self, model, instance_id, node_id, num_npus, tp_size, npu_mem, cpu_mem, block_size, fp, enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage, cxl_mem=0, ep_size=1, pp_size=1, kv_cache_dtype='auto', npu_memory_utilization=1.0):
        self.model = model
        self.node_id = node_id
        self.instance_id = instance_id
        self.num_npus = num_npus
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.ep_size = ep_size
        self.npu_mem = npu_mem * GB_TO_BYTE # GB -> Byte
        self.cpu_mem = cpu_mem * GB_TO_BYTE # GB -> Byte
        self.cxl_mem = cxl_mem * GB_TO_BYTE
        self.block_size = block_size
        self.fp = fp // 8 # bit -> byte of floating point
        self.kv_fp = 1 if kv_cache_dtype == 'fp8' else self.fp  # KV cache bytes per element
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_prefix_sharing = enable_prefix_sharing
        self.prefix_storage = prefix_storage
        self.npu_memory_utilization = npu_memory_utilization

        self.config = get_config(model)
        self.n_embd = self.config['hidden_size']
        self.n_layer = self.config['num_hidden_layers']
        self.n_head = self.config['num_attention_heads']
        self.head_dim = self.config.get('head_dim', self.n_embd // self.n_head)
        self.kv_head = self.config.get("num_key_value_heads", self.n_head)  # fallback to n_head if not defined
        self.q_dim = self.n_head * self.head_dim       # total Q projection output dim
        self.kv_dim = self.kv_head * self.head_dim     # total KV projection output dim
        self.vocab_size = self.config['vocab_size']
        # Accept either the Mistral-style ``num_local_experts`` or the
        # HF/Qwen-style ``num_experts`` key — profiler configs track
        # upstream HF naming which varies per family.
        self.is_moe = 'num_local_experts' in self.config or 'num_experts' in self.config

        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)

        self.weight = self.get_weight() # assume weight is loaded
        if self.weight > self.npu_mem:
            raise RuntimeError(f"[MemoryModel] [node={self.node_id},inst={self.instance_id}]: Model size {self.weight*self.num_npus//GB_TO_BYTE}GB exceeds total NPU memory {self.npu_mem*self.num_npus//GB_TO_BYTE}GB")

        # Non-KV bytes the instance holds outside the pools: model weights, plus
        # anything --enable-local-offloading or the PIM model loads explicitly.
        self._npu_reserved = self.weight
        self._cpu_reserved = 0

        self._bytes_per_token = self.get_kv(1)              # per rank
        self._npu_bytes_per_block = self._bytes_per_token * block_size
        self._cluster_bytes_per_token = self._bytes_per_token * self.num_npus

        # vLLM: requested = total_gpu_memory * gpu_memory_utilization, then
        # subtract the non-KV memory (weights, and the activation peak plus CUDA
        # context, which the simulator cannot profile and does not model -- so
        # this capacity is an upper bound on vLLM's at the same utilization).
        requested = int(self.npu_mem * self.npu_memory_utilization)
        kv_bytes = requested - self.weight
        if kv_bytes < self._npu_bytes_per_block:
            raise RuntimeError(
                f"[MemoryModel] [node={self.node_id},inst={self.instance_id}]: "
                f"npu_memory_utilization={self.npu_memory_utilization} leaves "
                f"{kv_bytes / MB_TO_BYTE:.2f}MB for the KV cache "
                f"({requested / MB_TO_BYTE:.2f}MB requested of "
                f"{self.npu_mem / MB_TO_BYTE:.2f}MB, minus "
                f"{self.weight / MB_TO_BYTE:.2f}MB of weights), which is less "
                f"than one {block_size}-token block "
                f"({self._npu_bytes_per_block / MB_TO_BYTE:.2f}MB)"
            )
        npu_blocks = kv_bytes // self._npu_bytes_per_block

        self.npu_pool = BlockPool(
            Device.NPU, int(npu_blocks), block_size, self._npu_bytes_per_block,
            enable_caching=enable_prefix_caching,
            node_id=node_id, instance_id=instance_id,
        )
        self.logger.info(
            "NPU: KV cache %d blocks (%d tokens, %.2fMB) at utilization %.2f",
            self.npu_pool.num_blocks,
            self.npu_pool.num_blocks * block_size,
            self.npu_pool.num_blocks * self._npu_bytes_per_block / MB_TO_BYTE,
            self.npu_memory_utilization,
        )

        # Victim tiers, in lookup order. Present only with --prefix-storage:
        # without it there is no host KV tier, exactly as in plain vLLM, so a
        # preempted request recovers whatever is still resident on the NPU and
        # recomputes the rest.
        self.lower_pools = []
        self.storage_pool = None
        if enable_prefix_caching and prefix_storage is not None:
            self.storage_pool = self._build_storage_pool(prefix_pool, prefix_storage)
            self.lower_pools.append(self.storage_pool)

        self.kv = TieredKVCacheManager(
            block_size, self.npu_pool, self.lower_pools,
            enable_caching=enable_prefix_caching,
        )

    def _build_storage_pool(self, prefix_pool, prefix_storage):
        """The second-tier pool: shared across instances, or private.

        Its blocks hold ``LOWER_TIER_CHUNK_TOKENS`` tokens and are sized in
        full-cluster bytes, because a host copy holds every rank's shard.
        """
        if self.enable_prefix_sharing and prefix_pool is not None:
            if prefix_pool.block_size % self.block_size != 0:
                raise RuntimeError(
                    f"[MemoryModel] [node={self.node_id},inst={self.instance_id}]: "
                    f"shared prefix pool block_size {prefix_pool.block_size} is not "
                    f"a multiple of this instance's block_size {self.block_size}"
                )
            expected = self._cluster_bytes_per_token * prefix_pool.block_size
            if prefix_pool.bytes_per_block != expected:
                raise RuntimeError(
                    f"[MemoryModel] [node={self.node_id},inst={self.instance_id}]: "
                    f"shared prefix pool bytes_per_block {prefix_pool.bytes_per_block} "
                    f"disagrees with this instance's {expected} — instances sharing a "
                    f"pool must share model, dtype and kv_cache_dtype"
                )
            return prefix_pool

        if prefix_storage == Device.CPU:
            capacity = self.cpu_mem
        elif prefix_storage == Device.CXL:
            capacity = self.cxl_mem
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}]: Device {prefix_storage} is currently not supported as a second tier prefix cache storage")
        return build_prefix_pool(prefix_storage, capacity, self.block_size,
                                 self._cluster_bytes_per_token,
                                 node_id=self.node_id, instance_id=self.instance_id)

    def get_weight(self):
        """Per-GPU model weight in bytes.

        Conservative upper bound across PP ranks: assumes a single rank
        holds embedding + final_layernorm + lm_head along with its share
        of transformer blocks (n_layer // pp_size). In real PP these
        non-block weights live on the first/last rank only, so middle
        ranks are lighter — but using the heaviest-rank value here keeps
        the `weight > npu_mem` check safe.
        """
        tp = self.tp_size
        pp = max(self.pp_size, 1)
        ep = self.ep_size
        fp = self.fp
        weight = 0

        _, embedding, _ = calculate_sizes(self.model, 'embedding', 1, parallel=tp, fp=fp)
        weight += embedding
        weight += self._get_weight_per_block(tp, ep, fp) * (self.n_layer // pp)
        _, ln_f, _ = calculate_sizes(self.model, 'final_layernorm', 1, parallel=tp, fp=fp)
        weight += ln_f
        _, lm_head, _ = calculate_sizes(self.model, 'lm_head', 1, parallel=tp, fp=fp)
        weight += lm_head

        self.logger.info(
            "NPU: model weight %dMB loaded",
            weight * tp // MB_TO_BYTE,
        )
        return weight

    def _get_weight_per_block(self, tp, ep, fp):
        """Per-block weight: dense layers use TP, MoE experts use EP."""
        block_weight = 0
        _, ln_w, _ = calculate_sizes(self.model, 'layernorm', 1, parallel=tp, fp=fp)
        block_weight += ln_w  # input layernorm
        _, qkv_w, _ = calculate_sizes(self.model, 'qkv_proj', 1, parallel=tp, fp=fp)
        block_weight += qkv_w
        _, o_w, _ = calculate_sizes(self.model, 'o_proj', 1, parallel=tp, fp=fp)
        block_weight += o_w
        block_weight += ln_w  # post layernorm (same weight size)
        if self.is_moe:
            _, moe_w, _ = calculate_sizes(self.model, 'moe', 1, parallel=ep, fp=fp)
            block_weight += moe_w
        else:
            _, ffn1_w, _ = calculate_sizes(self.model, 'gate_up_proj', 1, parallel=tp, fp=fp)
            block_weight += ffn1_w
            _, ffn2_w, _ = calculate_sizes(self.model, 'down_proj', 1, parallel=tp, fp=fp)
            block_weight += ffn2_w
        return block_weight

    # -------------------- KV sizing math --------------------

    def get_kv(self, seq):
        # shape of kv cache
        # (kv_head, batch_size, n_embd//n_head, seq_len) per layer
        # return batch_size = 1 to caclulate max batch_size in scheduler

        # K & V multiply 2
        return 2 * self.kv_dim * seq * self.n_layer * self.kv_fp // self.num_npus

    def get_total_kv(self, req):
        """Bytes of KV a request's whole computed context occupies, per rank.

        Only used when handing a prefilled request to a decode instance, where
        the decode side allocates the full context in one go.
        """
        num_blocks = (req.num_computed_tokens + self.block_size - 1) // self.block_size
        return self.get_kv(num_blocks * self.block_size)

    def pd_kv_bytes(self, num_tokens):
        """KV bytes to ship to a paired decode instance for ``num_tokens``.

        Per rank, and K+V only -- deliberately *not* the ``qkv_proj`` output,
        which also carries Q and so overstated the P/D transfer by
        (q_dim + 2*kv_dim) / (2*kv_dim): 3x for Llama-3.1-8B, 1.5x for MHA,
        more at wider GQA ratios. It also honours ``kv_cache_dtype``, which the
        activation size did not.
        """
        return 2 * self.kv_dim * num_tokens * self.n_layer * self.kv_fp // self.tp_size

    def free_weight(self):
        if self._npu_reserved - self.weight < 0:
            raise RuntimeError(
                f"[MemoryModel] [node={self.node_id}, inst={self.instance_id}] NPU: tried to free model weight {self.weight / MB_TO_BYTE:.2f}MB "
                f"but only {self._npu_reserved / MB_TO_BYTE:.2f}MB is used."
            )
        self.logger.info(
            "NPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
            self.npu_used / MB_TO_BYTE,
            self.weight / MB_TO_BYTE,
            (self.npu_used - self.weight) / MB_TO_BYTE,
        )
        self._npu_reserved -= self.weight

    # -------------------- byte-level view over the pools --------------------

    @property
    def npu_used(self):
        """Bytes held on the NPU: weights and other reservations, plus KV.

        A property, not a counter: there is nothing to keep in sync with the
        pool, which is the whole point. The previous two-ledger arrangement
        (npu_used alongside RadixCache.capacity/total_memory_usage) is what let
        PR #59's mismatch through.
        """
        return self._npu_reserved + self.npu_pool.used_bytes()

    @property
    def cpu_used(self):
        cpu = self._cpu_reserved
        if self.storage_pool is not None and self.storage_pool.tier == Device.CPU:
            cpu += self.storage_pool.used_bytes()
        return cpu

    def allocate(self, size, device):
        """Reserve non-KV bytes: weights, PIM buffers, local weight offloading.

        KV never comes through here -- it is allocated in blocks by the pool,
        which is the only thing that can report failure in the same call.
        """
        if size <= 0:
            return
        if device == Device.NPU:
            if self.npu_used + size > self.npu_mem:
                raise RuntimeError(
                    f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] NPU: "
                    f"tried to load {size / MB_TO_BYTE:.2f}MB but only "
                    f"{(self.npu_mem - self.npu_used) / MB_TO_BYTE:.2f}MB is available "
                    f"(KV cache holds {self.npu_pool.used_blocks} of "
                    f"{self.npu_pool.num_blocks} blocks)."
                )
            self._npu_reserved += size
        elif device == Device.CPU:
            if self.cpu_used + size > self.cpu_mem:
                raise RuntimeError(
                    f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] CPU: "
                    f"tried to load {size / MB_TO_BYTE:.2f}MB but only "
                    f"{(self.cpu_mem - self.cpu_used) / MB_TO_BYTE:.2f}MB is available."
                )
            self._cpu_reserved += size
        elif device == Device.CXL:
            self._cpu_reserved += size
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to allocate in unsupported device {device}")

    def free(self, size, device):
        if size <= 0:
            return
        if device == Device.NPU:
            if self._npu_reserved - size < 0:
                raise RuntimeError(
                    f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] NPU: tried to free {size / MB_TO_BYTE:.2f}MB but only {self._npu_reserved / MB_TO_BYTE:.2f}MB is reserved."
                )
            self._npu_reserved -= size
        elif device in (Device.CPU, Device.CXL):
            if self._cpu_reserved - size < 0:
                raise RuntimeError(
                    f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] CPU: tried to free {size / MB_TO_BYTE:.2f}MB but only {self._cpu_reserved / MB_TO_BYTE:.2f}MB is reserved."
                )
            self._cpu_reserved -= size
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to free in unsupported device {device}")

    def is_avail(self, size, device):
        if device == Device.NPU:
            return self.npu_mem - self.npu_used >= size
        elif device == Device.CPU:
            return self.cpu_mem - self.cpu_used >= size
        elif device == Device.CXL:
            return self.cxl_mem - self.cpu_used >= size
        raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to check available size of unsupported device {device}")

    # -------------------- prefix cache statistics --------------------

    def record_prefix_stats(self, req):
        """Count a request's prompt tokens and hits, once per tier.

        Replaces the bookkeeping that used to ride along inside
        ``RadixCache.cache_unfinished_req``.
        """
        if not self.enable_prefix_caching:
            return
        if not req._prefix_npu_stats_counted:
            self.npu_pool.stats.record(req.original_input, req.npu_cache_hit)
            req._prefix_npu_stats_counted = True
        if self.storage_pool is not None and not req._prefix_storage_stats_counted:
            self.storage_pool.stats.record(
                req.original_input, max(0, req.storage_cache_hit - req.npu_cache_hit))
            req._prefix_storage_stats_counted = True

    def return_prefix_info(self):
        if not self.enable_prefix_caching:
            return (0, 0, 0, 0)
        npu = self.npu_pool.stats.return_prefix_info()
        if self.storage_pool is None:
            return (npu, (0, 0))
        return (npu, self.storage_pool.stats.return_prefix_info())

    def format_prefix_info(self):
        if not self.enable_prefix_caching:
            return ""
        return self.npu_pool.stats.format_prefix_info()

    # -------------------- teardown --------------------

    def free_prefix_cache(self):
        """Drop every cached block at end of run, so is_free() can be exact."""
        if not self.enable_prefix_caching:
            return
        self.npu_pool.reset()
        if self.storage_pool is not None and not self.enable_prefix_sharing:
            self.storage_pool.reset()

    def is_free(self):
        leaked = []
        if self._npu_reserved != 0:
            leaked.append(f"NPU reserved {self._npu_reserved / MB_TO_BYTE:.2f}MB")
        if self._cpu_reserved != 0:
            leaked.append(f"CPU reserved {self._cpu_reserved / MB_TO_BYTE:.2f}MB")
        if not self.npu_pool.is_free():
            leaked.append(f"NPU pool {self.npu_pool.used_blocks}/{self.npu_pool.num_blocks} blocks")
        if leaked:
            self.logger.error("Memory leak detected: %s", ", ".join(leaked))
        return not leaked


def build_prefix_pool(tier, capacity_bytes, npu_block_size, cluster_bytes_per_token,
                      node_id=None, instance_id=None):
    """A victim-tier :class:`BlockPool` sized from a byte capacity.

    Also used by ``__main__`` to build pools shared across instances, before any
    ``MemoryModel`` exists. Chunk size is ``LOWER_TIER_CHUNK_TOKENS`` rounded up
    to a multiple of the NPU block size, so the tier can key on every Nth hash of
    the same chain. Bytes are full-cluster: a host copy holds every rank's shard.
    """
    chunk = max(npu_block_size,
                (LOWER_TIER_CHUNK_TOKENS // npu_block_size) * npu_block_size)
    bytes_per_block = cluster_bytes_per_token * chunk
    num_blocks = int(capacity_bytes // bytes_per_block)
    if num_blocks < 1:
        raise RuntimeError(
            f"[build_prefix_pool] {tier}: {capacity_bytes / GB_TO_BYTE:.2f}GB is not "
            f"enough for one {chunk}-token chunk "
            f"({bytes_per_block / MB_TO_BYTE:.2f}MB)"
        )
    return BlockPool(tier, num_blocks, chunk, bytes_per_block,
                     enable_caching=True, node_id=node_id, instance_id=instance_id)


def full_cluster_kv_bytes_per_token(model, fp, kv_cache_dtype='auto'):
    """Bytes of KV cache per token aggregated over the full TP cluster.

    Mirrors MemoryModel.get_kv(1) * num_npus but computes directly, avoiding
    the per-rank floor-division roundoff. ``fp`` is the model weight dtype
    in bits (16, 32, ...). ``kv_cache_dtype='fp8'`` forces 1 byte per element
    for the KV cache regardless of weight dtype.
    """
    config = get_config(model)
    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    head_dim = config.get('head_dim', n_embd // n_head)
    kv_head = config.get('num_key_value_heads', n_head)
    kv_dim = kv_head * head_dim
    n_layer = config['num_hidden_layers']
    kv_fp = 1 if kv_cache_dtype == 'fp8' else fp // 8
    # 2 (K + V) * kv_dim * n_layer * bytes_per_elem
    return 2 * kv_dim * n_layer * kv_fp


# calculate the per-rank input, weight, output size of each layer
def calculate_sizes(model, layer_name, length, kv_len=None, pim=False, parallel=1, fp=2):
    """Calculate input, weight, and output tensor sizes for a given layer.

    Args:
        parallel: parallelism degree for weight/activation sharding.
            For dense layers this is TP; for MoE experts this is EP.
    """
    config = get_config(model)
    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    head_dim = config.get('head_dim', n_embd // n_head)
    vocab_size = config['vocab_size']
    kv_head = config.get("num_key_value_heads", n_head)  # fallback to n_head if not defined
    q_dim = n_head * head_dim       # total Q projection output dim
    kv_dim = kv_head * head_dim     # total KV projection output dim
    ffn_dim = config.get("intermediate_size", config.get("ffn_dim"))  # dense FFN dim
    moe_ffn_dim = config.get("moe_intermediate_size", ffn_dim)  # per-expert FFN dim (may differ from dense)
    # Same both-name fallback as MemoryModel.__init__ — HF / Qwen use
    # ``num_experts`` while Mistral uses ``num_local_experts``.
    num_local_experts = config.get(
        "num_local_experts", config.get("num_experts", 1)
    )

    p = max(int(parallel), 1)

    # NOTE (vLLM-style assumptions):
    # NOTE (vLLM-style assumptions):
    # - Embedding / LM head: vocab-parallel → split vocab_size across ranks.
    # - Q/K/V: ColumnParallelLinear         → split output dim across ranks.
    # - o_proj: RowParallelLinear           → split input dim across ranks.
    # - LayerNorm weights: replicated (NOT sharded).
    # - MoE experts: parallel = EP degree, each rank holds num_local_experts // p experts.

    # ----------------- Embedding & Norms -----------------
    if layer_name == "embedding":
        input_size = length * fp * 2  # token_ids are int32 or int64
        weight_size = (vocab_size // p) * n_embd * fp
        output_size = length * n_embd * fp

    elif layer_name in ["input_layernorm", "post_layernorm", "final_layernorm", "layernorm"]:
        input_size = length * n_embd * fp
        weight_size = 1 * n_embd * fp  # scale only
        output_size = length * n_embd * fp

    elif layer_name == "qk_norm":
        input_size = length * (q_dim + kv_dim) // p * fp
        weight_size = 2 * head_dim * fp
        output_size = length * (q_dim + kv_dim) // p * fp

    # ----------------- RoPE & Attention Core -----------------
    elif layer_name == "rotary_emb":
        input_size = ((n_head // p) + (kv_head // p)) * length * head_dim * fp
        weight_size = 0
        output_size = ((n_head // p) + (kv_head // p)) * length * head_dim * fp

    elif layer_name == "attention":
        if not pim:
            input_size = (
                (n_head // p) * length * head_dim * fp +
                (kv_head // p) * kv_len * head_dim * fp * 2
            )
            weight_size = 0
            output_size = (n_head // p) * length * head_dim * fp
        else:
            input_size = (
                (n_head // p) * 1 * head_dim * fp +
                (kv_head // p) * 1 * head_dim * fp * 2
            )
            weight_size = 0
            output_size = (n_head // p) * 1 * head_dim * fp

    # ----------------- QKV Projection (fused) -----------------
    elif layer_name == "qkv_proj":
        input_size = length * n_embd * fp
        weight_size = n_embd * ((q_dim + 2 * kv_dim) // p) * fp
        output_size = length * ((q_dim + 2 * kv_dim) // p) * fp

    elif layer_name == "o_proj":
        input_size = length * (q_dim // p) * fp
        weight_size = (q_dim // p) * n_embd * fp
        output_size = length * n_embd * fp

    elif layer_name == "gate_up_proj":
        input_size = length * n_embd * fp
        weight_size = n_embd * 2 * (ffn_dim // p) * fp
        output_size = length * 2 * (ffn_dim // p) * fp

    elif layer_name == "act_fn":
        input_size = length * 2 * (ffn_dim // p) * fp
        weight_size = 0
        output_size = length * (ffn_dim // p) * fp

    elif layer_name == "down_proj":
        input_size = length * (ffn_dim // p) * fp
        weight_size = (ffn_dim // p) * n_embd * fp
        output_size = length * n_embd * fp

    elif layer_name == "sampler":
        input_size = length * (vocab_size // p) * fp
        weight_size = 0
        output_size = length * 4  # int32 token IDs

    elif layer_name == "moe":
        experts_per_rank = num_local_experts // p
        input_size = length * n_embd * fp
        weight_size = (n_embd * num_local_experts * fp  # gate (replicated)
                     + experts_per_rank * 3 * n_embd * moe_ffn_dim * fp)  # local experts
        output_size = length * n_embd * fp

    # ----------------- LM Head -----------------
    elif layer_name == "lm_head":
        input_size = length * n_embd * fp
        weight_size = n_embd * (vocab_size // p) * fp
        output_size = length * (vocab_size // p) * fp

    else:
        raise ValueError(f"No matching layer name {layer_name} found for model {model}.")

    return input_size, weight_size, output_size
