import os
from .utils import (
    get_architecture, get_config, get_layer_stack, is_moe as _is_moe,
    num_experts as _num_experts, num_mtp_layers,
)
from profiler.core.stack import LayerSpec
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

    def __init__(self, model, instance_id, node_id, num_npus, tp_size, npu_mem, cpu_mem, block_size, fp, enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage, cxl_mem=0, ep_size=1, pp_size=1, kv_cache_dtype='auto', npu_memory_utilization=1.0, num_speculative_tokens=0):
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
        self.num_speculative_tokens = int(num_speculative_tokens or 0)

        self.config = get_config(model)
        self.n_embd = self.config['hidden_size']
        self.n_layer = self.config['num_hidden_layers']
        self.n_head = self.config['num_attention_heads']
        self.head_dim = self.config.get('head_dim', self.n_embd // self.n_head)
        self.kv_head = self.config.get("num_key_value_heads", self.n_head)  # fallback to n_head if not defined
        self.q_dim = self.n_head * self.head_dim       # total Q projection output dim
        self.kv_dim = self.kv_head * self.head_dim     # total KV projection output dim
        self.vocab_size = self.config['vocab_size']
        self.is_moe = _is_moe(self.config)

        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)

        # Filled on first get_kv(); the layer walk is per model, not per call.
        self._kv_bytes_per_token_per_rank = None

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
            state_blocks_per_request=self._state_blocks_per_request(),
        )

    def _state_blocks_per_request(self):
        """Pool blocks one request holds for its per-sequence layer state.

        0 for every model whose layers all cache per token, which is every
        family here except the gated-DeltaNet hybrids. For those the state does
        not grow with the sequence but is held for its whole life, so it bounds
        **concurrency** the way a KV cache bounds context.

        Counted in **pages**, not in bytes, because that is what vLLM
        allocates. It picks the attention block size so that one attention page
        is the smallest aligned multiple that covers one mamba page
        (``platforms/interface.py``: ``attn_block_size = alignment *
        cdiv(mamba_page_size, alignment * attn_page_size_1_token)``), then sets
        ``mamba_page_size_padded = attn_page_size`` -- so a layer's whole state
        occupies exactly one page and the padding is really allocated. On
        Qwen3.8-27B the mamba page is 3,207,168 bytes against an attention page
        of 3,211,264 at ``block_size 784``, which is how 784 gets chosen.

        How many pages per layer is ``MambaSpec.max_memory_usage_bytes``:

        ============  =====================  ===================================
        cache mode    pages per mamba layer  when
        ============  =====================  ===================================
        ``none``      ``1 + N``              prefix caching off
        ``align``     ``2 + N``              prefix caching on -- **the default**
        ``all``       ``cdiv(max_len, bs)``  opt-in, not modelled
        ============  =====================  ===================================

        ``align`` holds two because one page carries the state being written
        this step and the other the last checkpoint committed at a block
        boundary, which is what a later prefix hit resumes from. ``N`` is
        ``num_speculative_tokens`` (``MambaSpec.num_speculative_blocks``), one
        extra page per draft token.

        Charging one page per layer -- the ``none`` figure -- understated a
        prefix-caching run by exactly 2x.
        """
        stack = get_layer_stack(self.model)
        if not stack:
            return 0
        mamba_layers = sum(
            1 for spec in stack
            if state_bytes_per_sequence(
                self.config, self.fp, spec, self.num_speculative_tokens)
        )
        if not mamba_layers:
            return 0
        if self._npu_bytes_per_block <= 0:
            raise RuntimeError(
                f"[MemoryModel] {self.model} has per-sequence layer state but "
                f"no per-token KV at all, so there is no block size to charge "
                f"the state against. A model with only linear-attention layers "
                f"needs its own pool unit."
            )
        # One pool block is one block-worth of KV for **every** layer this rank
        # holds, so it is that many attention pages -- and a mamba page is one
        # of them.
        attn_layers = len(stack) - mamba_layers
        pages_per_block = max(attn_layers, 1)
        pages_per_layer = (2 if self.enable_prefix_caching else 1) \
            + self.num_speculative_tokens
        pages = mamba_layers * pages_per_layer
        blocks = -(-pages // (pages_per_block * max(self.pp_size, 1)))
        self.logger.info(
            "NPU: %d mamba layer(s) x %d page(s) = %d page(s) per request "
            "(%s mode) -> %d block(s)",
            mamba_layers, pages_per_layer, pages,
            "align" if self.enable_prefix_caching else "none", blocks,
        )
        return blocks

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

        Walks the architecture yaml's blocks and the checkpoint's own per-layer
        composition, so a heterogeneous stack is weighed layer by layer: on
        DeepSeek-V3.2 the first three layers carry a dense MLP and the other 58
        carry 256 experts, and on Qwen3.8 gated-DeltaNet layers weigh nothing
        like the full-attention ones. It used to sum a hardcoded
        ``layernorm + qkv_proj + o_proj + layernorm + mlp``, which is right for
        exactly the families whose blocks look like Llama's and silently wrong
        for the rest -- and unrunnable for MLA, whose blocks have no
        ``qkv_proj`` at all.

        Conservative upper bound across PP ranks: assumes a single rank
        holds embedding + final_layernorm + lm_head along with its share
        of transformer blocks (n_layer // pp_size). In real PP these
        non-block weights live on the first/last rank only, so middle
        ranks are lighter — but using the heaviest-rank value here keeps
        the `weight > npu_mem` check safe.
        """
        pp = max(self.pp_size, 1)
        arch = get_architecture(self.model)
        shared = arch.get("shared") or {}

        weight = sum(
            self._layer_weight(name)
            for name in (shared.get("prologue") or []) + (shared.get("head") or [])
        )

        # One built weight per distinct block shape, then the **heaviest**
        # contiguous run of this rank's layers. Taking the first
        # ``n_layer // pp`` would understate a heterogeneous stack: DeepSeek's
        # first three layers are the cheap dense ones, so on pp=2 the first
        # window is the light half and the check this feeds would pass a
        # configuration that does not fit.
        stack = get_layer_stack(self.model)
        per_shape: dict = {}
        for spec in set(stack):
            per_shape[spec] = sum(
                self._layer_weight(name)
                for name in self._block_layer_names(arch, spec)
            )
        per_layer = [per_shape[spec] for spec in stack]
        per_stage = -(-len(per_layer) // pp)  # ceil, the fullest stage
        weight += max(
            sum(per_layer[i:i + per_stage])
            for i in range(0, max(1, len(per_layer) - per_stage + 1))
        )

        self.logger.info(
            "NPU: model weight %dMB loaded",
            weight * self.tp_size // MB_TO_BYTE,
        )
        return weight

    @staticmethod
    def _block_layer_names(arch, spec):
        """Every canonical layer one decoder layer of shape ``spec`` emits."""
        blocks = arch.get("blocks") or {}
        group = None
        if spec.sparse:
            group = (blocks.get("sparse_attn") or {}).get(spec.attn)
        if group is None:
            group = (blocks.get("attn") or {}).get(spec.attn)
        if group is None:
            raise KeyError(
                f"Architecture for {arch.get('model_types') or 'this model'} "
                f"declares no 'blocks.attn.{spec.attn}'."
            )
        return (list(group.get("pre_attn") or [])
                + list(group.get("post_attn") or [])
                + list((blocks.get("mlp") or {}).get(spec.mlp) or []))

    def _layer_weight(self, name):
        """One canonical layer's per-rank parameter bytes.

        ``moe`` shards by EP, everything else by TP — the same split
        ``calculate_sizes`` documents.
        """
        parallel = self.ep_size if name == "moe" else self.tp_size
        _, weight, _ = calculate_sizes(
            self.model, name, 1, kv_len=1, parallel=parallel, fp=self.fp,
        )
        return weight

    # -------------------- KV sizing math --------------------

    def get_kv(self, seq):
        """Per-rank KV cache bytes for ``seq`` tokens, summed over layers.

        Walks the checkpoint's own layer composition, because the per-layer
        answer differs by block: an MLA layer caches a replicated latent, a
        sparse one adds the indexer's own cache beside it, and a
        gated-DeltaNet layer caches nothing per token at all.

        Sharding differs with it. GQA splits K and V across ranks; MLA does
        not (``num_kv_heads = 1``), so its bytes are the same on every rank and
        must not be divided.
        """
        if self._kv_bytes_per_token_per_rank is None:
            config, kv_fp = self.config, self.kv_fp
            stack = get_layer_stack(self.model) or [None] * self.n_layer
            parts = [
                kv_bytes_per_token_per_layer(config, kv_fp, spec)
                for spec in stack
            ]
            parts += self._drafter_kv_parts(stack)
            # Sharded caches divide across every NPU of the instance; a
            # replicated one is the same bytes on each, so it must not.
            sharded = sum(a for a, _ in parts)
            replicated = sum(b for _, b in parts)
            self._kv_bytes_per_token_per_rank = (
                sharded // self.num_npus + replicated
            )
        return self._kv_bytes_per_token_per_rank * seq

    def _drafter_kv_parts(self, stack):
        """KV the model's own drafter layers cache, one entry per layer.

        An MTP module wraps a real decoder layer -- ``DeepseekV2DecoderLayer``,
        ``Glm4DecoderLayer``, ``MiniMaxM3DecoderLayer``, ``Qwen3_5DecoderLayer``
        are the four -- so it publishes a KV cache spec of its own and vLLM
        allocates for it. Ignoring that undercounts by
        ``num_mtp_layers / num_hidden_layers``: 1.6% on DeepSeek-V3.2's one
        module, 11.7% on MiniMax-M3's seven.

        The drafter's attention is **full attention** whatever the target's
        layer is -- Qwen3.5's MTP builds its block with
        ``layer_type="full_attention"`` explicitly, which is what stops a
        hybrid's drafter from carrying a recurrent state. Everything else about
        the block follows the target's last layer, which is the only layer
        whose composition the MTP module is documented to mirror.

        Empty unless speculative decoding is on: with no drafts to propose the
        module is never built.
        """
        if not self.num_speculative_tokens:
            return []
        mtp_layers = num_mtp_layers(self.config)
        if not mtp_layers or not stack:
            return []
        base = stack[-1]
        drafter = (
            LayerSpec(attn='full_attention', mlp=base.mlp, sparse=base.sparse)
            if base is not None else None
        )
        return [
            kv_bytes_per_token_per_layer(self.config, self.kv_fp, drafter)
        ] * mtp_layers

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


def kv_bytes_per_token_per_layer(config, kv_fp, spec=None):
    """KV bytes one token occupies in one decoder layer, as ``(sharded, replicated)``.

    Two numbers rather than one because a layer can hold both, and they divide
    differently: a cache with ``num_kv_heads > 1`` splits across TP ranks while
    one with ``num_kv_heads = 1`` is the same bytes on every rank. MiniMax-M3's
    sparse layers hold one of each.

    The shapes, all read off the ``get_kv_cache_spec`` each layer publishes:

    * **Grouped-query attention** — ``2 * kv_head * head_dim`` elements, K and
      V, sharded. What every family here had until now.
    * **MLA** (DeepSeek-V3.2, GLM-5) — one latent of
      ``kv_lora_rank + qk_rope_head_dim``, no separate V
      (``MLAAttentionSpec.head_size_v = 0``), ``num_kv_heads = 1``, so
      replicated. 1152 bytes/token/layer on DeepSeek-V3.2 in bf16 against
      28672 for the GQA formula read off the same config.
    * **DeepSeek's sparse indexer** — a second cache beside the latent:
      ``index_head_dim`` fp8 keys plus one fp32 scale per 128 elements, stored
      as uint8, so literal bytes independent of ``kv_fp``. Replicated.
    * **MiniMax-M3's block-sparse indexer** — a third shape again: key-only,
      one ``sparse_index_dim`` vector per token at the indexer's own dtype
      (bf16 by default, *not* the main cache's), replicated, and **beside** an
      ordinary GQA cache rather than instead of one. Its top-k caps what
      attention *reads*, not what is *stored*.
    * **Linear attention** (gated DeltaNet) — nothing per token; the state is
      fixed per sequence.

    ``spec`` is the layer's :class:`LayerSpec`; ``None`` answers for a uniform
    stack.
    """
    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    head_dim = config.get('head_dim', n_embd // n_head)
    kv_head = config.get('num_key_value_heads', n_head)

    kv_lora_rank = config.get('kv_lora_rank')
    if kv_lora_rank is not None:
        # MLA: one replicated latent, and no GQA cache at all.
        replicated = (kv_lora_rank + (config.get('qk_rope_head_dim') or 0)) * kv_fp
        idx_head_dim = config.get('index_head_dim') or 0
        sparse = True if spec is None else spec.sparse
        if sparse and idx_head_dim and 'index_topk' in config:
            # fp8 keys + one fp32 scale per 128-element block, as uint8. The
            # scales are whole fp32s regardless of the key dtype.
            idx_b = cache_dtype_bytes(config, kv_fp, kv_fp)["indexer"]
            replicated += idx_head_dim * idx_b + (idx_head_dim // 128) * 4
        return 0, replicated

    if spec is not None and spec.attn == 'linear_attention':
        # Gated DeltaNet caches nothing per token: a fixed conv state plus a
        # fixed recurrent state **per sequence**, which is what makes its cost
        # independent of context length. See ``state_bytes_per_sequence``.
        return 0, 0

    sharded = 2 * kv_head * head_dim * kv_fp
    sparse_cfg = config.get('sparse_attention_config') or {}
    idx_dim = sparse_cfg.get('sparse_index_dim') or 0
    sparse = bool(sparse_cfg.get('use_sparse_attention')) if spec is None else spec.sparse
    if sparse and idx_dim:
        # Key-only side cache at the **indexer's own** dtype, which does not
        # follow --kv-cache-dtype: vLLM's M3 indexer asks for
        # ``resolve_indexer_kv_dtype("bf16")``, so an fp8 KV run leaves it bf16.
        return sharded, idx_dim * cache_dtype_bytes(config, kv_fp, kv_fp)["indexer"]
    return sharded, 0


_STATE_DTYPE_BYTES = {
    "float32": 4, "float": 4, "fp32": 4,
    "bfloat16": 2, "float16": 2, "half": 2, "bf16": 2, "fp16": 2,
    "float8_e4m3fn": 1, "fp8": 1, "int8": 1, "uint8": 1,
}


def _dtype_bytes(name, default):
    """Bytes per element for a dtype name, or ``default`` for auto/unknown."""
    if isinstance(name, str) and name.lower() not in ("auto", ""):
        return _STATE_DTYPE_BYTES.get(name.lower(), default)
    return default


def cache_dtype_bytes(config, fp, kv_fp):
    """Every cache dtype a model uses, in bytes per element.

    There is no single "the dtype" any more, and the five do not come from the
    same place. Reading one CLI value for all of them is how the recurrent
    state came out half size.

    ==================  =============================================
    cache               source (all five are the checkpoint's; none
                        is an input to this simulator)
    ==================  =============================================
    weight              ``quantization_config.quant_method``, then
                        ``torch_dtype`` / ``dtype``
    main KV             ``quantization_config.kv_cache_scheme`` or
                        ``kv_cache_quant_algo`` -> fp8, else the
                        weight dtype
    mamba conv state    ``mamba_cache_dtype``, "auto" -> the weight
                        dtype
    mamba SSM state     ``mamba_ssm_dtype``, "auto" -> the **conv**
                        dtype, not the weight dtype
    indexer side cache  fixed by the model, following neither
    ==================  =============================================

    vLLM does expose a flag for the first two, and this simulator
    deliberately does not: five dtypes decided in four places make a flag
    per dtype both unusable and unfaithful, and the checkpoint already
    says what it is. To simulate another precision, profile it.

    The last row is the one that looks like a serving knob and is not.
    DeepSeek's ``DeepseekV32IndexerCache`` is constructed with
    ``dtype=torch.uint8`` outright -- fp8 keys plus fp32 scales, packed -- and
    MiniMax-M3's indexer asks vLLM for ``resolve_indexer_kv_dtype("bf16")``,
    which returns bf16 unless a *separate* attention-config knob says
    otherwise. Neither follows ``--kv-cache-dtype``, so an fp8 KV run must not
    shrink them.
    """
    conv = _dtype_bytes(config.get("mamba_cache_dtype"), fp)
    ssm = _dtype_bytes(config.get("mamba_ssm_dtype"), conv)
    # Indexer side cache, per family, both fixed by the model:
    #   M3            MiniMaxM3IndexerCache(indexer_kv_dtype="bf16") -> bf16
    #   DeepSeek/GLM  DeepseekV32IndexerCache(dtype=torch.uint8)     -> 1 byte
    # Neither is --kv-cache-dtype. Both models do expose a separate knob for it
    # (M3's --attention-config indexer_kv_dtype), which this simulator does not,
    # so the model default is the only value either can take here.
    if config.get("sparse_attention_config"):
        indexer = _dtype_bytes("bfloat16", 2)
    else:
        indexer = _dtype_bytes("uint8", 1)
    return {"weight": fp, "kv": kv_fp, "mamba_conv": conv,
            "mamba_ssm": ssm, "indexer": indexer}


def _ssm_state_bytes(config, fp):
    """Bytes per element of the recurrent (SSM) state. See
    :func:`cache_dtype_bytes` for where each cache's dtype comes from."""
    return cache_dtype_bytes(config, fp, fp)["mamba_ssm"]


def state_bytes_per_sequence(config, fp, spec=None, num_spec_tokens=0):
    """Fixed per-**sequence** state bytes for one decoder layer, cluster-wide.

    Linear attention keeps no per-token cache. Gated DeltaNet holds a rolling
    convolution state and a recurrent state whose sizes do not depend on the
    sequence length at all, which is exactly why its cost does not either --
    but they are held for as long as the sequence is alive, so they bound
    concurrency the way a KV cache bounds context.

    Shapes from vLLM's ``MambaStateShapeCalculator.gated_delta_net_state_shape``::

        conv_dim   = 2 * key_head_dim * num_key_heads + value_head_dim * num_value_heads
        conv_state = (conv_dim, conv_kernel_dim - 1 + num_spec)
        ssm_state  = (num_value_heads, value_head_dim, key_head_dim)

    ``num_spec`` widens the **conv** state only -- a verification step needs
    the last ``conv_kernel - 1 + N`` inputs to replay any accepted prefix --
    while the recurrent state stays one checkpoint. It is the smaller half, so
    this moves ~2% at N=4, but it is vLLM's shape and costs one term.

    Both shard across TP ranks, and they do **not** share a dtype -- see
    :func:`_ssm_state_bytes`. On Qwen3.8-27B that is 3.06 MB per sequence per
    layer (60 KB of conv state at bf16 plus 3 MB of recurrent state at fp32),
    and 48 of its 64 layers are gated DeltaNet -- roughly 154 MB per concurrent
    sequence, so it is not a rounding error on capacity.

    Returns 0 for any layer that is not linear attention.
    """
    if spec is not None and spec.attn != 'linear_attention':
        return 0
    k_heads = config.get('linear_num_key_heads') or 0
    v_heads = config.get('linear_num_value_heads') or 0
    k_dim = config.get('linear_key_head_dim') or 0
    v_dim = config.get('linear_value_head_dim') or 0
    conv_kernel = config.get('linear_conv_kernel_dim') or 0
    if not (k_heads and v_heads):
        return 0
    conv_dim = 2 * k_heads * k_dim + v_heads * v_dim
    conv_state = conv_dim * max(conv_kernel - 1 + int(num_spec_tokens), 0)
    ssm_state = v_heads * v_dim * k_dim
    # Two dtypes, not one: the conv state follows the model dtype and the
    # recurrent state follows ``mamba_ssm_dtype``. Charging both at the model
    # dtype halved Qwen3.8's state, and the recurrent half is 96% of it.
    dt = cache_dtype_bytes(config, fp, fp)
    return conv_state * dt["mamba_conv"] + ssm_state * dt["mamba_ssm"]


def full_cluster_state_bytes_per_sequence(model, fp):
    """Per-sequence state bytes over the whole model, cluster-wide."""
    config = get_config(model)
    stack = get_layer_stack(model)
    if not stack:
        return 0
    fp_bytes = fp // 8
    return sum(state_bytes_per_sequence(config, fp_bytes, spec) for spec in stack)


def full_cluster_kv_bytes_per_token(model, fp, kv_cache_dtype='auto'):
    """Bytes of KV cache per token aggregated over the full TP cluster.

    Mirrors MemoryModel.get_kv(1) * num_npus but computes directly, avoiding
    the per-rank floor-division roundoff. ``fp`` is the model weight dtype
    in bits (16, 32, ...). ``kv_cache_dtype='fp8'`` forces 1 byte per element
    for the KV cache regardless of weight dtype.
    """
    config = get_config(model)
    kv_fp = 1 if kv_cache_dtype == 'fp8' else fp // 8
    stack = get_layer_stack(model) or [None] * config['num_hidden_layers']
    return sum(
        sum(kv_bytes_per_token_per_layer(config, kv_fp, spec)) for spec in stack
    )


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
    # Dense FFN width. MiniMax-M3 inverts the usual convention -- its
    # ``intermediate_size`` is the *per-expert* width (3072) and the dense
    # layers' is ``dense_intermediate_size`` (12288) -- so read that first
    # where it exists, or the three dense layers come out 4x narrow.
    ffn_dim = config.get(
        "dense_intermediate_size",
        config.get("intermediate_size", config.get("ffn_dim")),
    )
    # Per-expert FFN width, which may differ from the dense one.
    moe_ffn_dim = config.get(
        "moe_intermediate_size", config.get("intermediate_size", ffn_dim)
    )
    # ``utils.num_experts`` knows every spelling; missing one made a DeepSeek
    # MoE block weigh a single expert instead of 256.
    num_local_experts = _num_experts(config) or 1
    # Shared experts run on every token beside the routed ones, so their
    # weights are resident like any other expert.
    num_shared_experts = int(config.get("n_shared_experts") or 0)

    # --- MLA (DeepSeek-V3.2, GLM-5) ---------------------------------------
    # Present only on an MLA checkpoint; ``kv_lora_rank`` is the discriminator
    # because it is the one field the latent cache cannot do without.
    kv_lora_rank = config.get("kv_lora_rank")
    is_mla = kv_lora_rank is not None
    q_lora_rank = config.get("q_lora_rank") or 0
    qk_nope_dim = config.get("qk_nope_head_dim") or 0
    qk_rope_dim = config.get("qk_rope_head_dim") or 0
    v_head_dim = config.get("v_head_dim") or head_dim
    qk_head_dim = qk_nope_dim + qk_rope_dim
    # What one rank's o_proj reads, and what the rope actually rotates. On a
    # non-MLA model these fall back to the plain GQA values.
    o_in_dim = (n_head * v_head_dim) if is_mla else q_dim
    rope_dim = qk_rope_dim if is_mla else head_dim

    # --- DeepSeek sparse-attention indexer --------------------------------
    idx_heads = config.get("index_n_heads") or 0
    idx_head_dim = config.get("index_head_dim") or 0
    idx_topk = config.get("index_topk") or 0

    # --- MiniMax-M3 block-sparse attention --------------------------------
    # A different sparse shape from DeepSeek's: block-level top-k over plain
    # GQA rather than token-level top-k over MLA, and the index branch rides
    # the KV-head sharding path (vLLM asserts
    # ``total_num_index_heads == total_num_kv_heads``).
    sparse_cfg = config.get("sparse_attention_config") or {}
    m3_idx_dim = sparse_cfg.get("sparse_index_dim") or 0
    m3_idx_heads = sparse_cfg.get("sparse_num_index_heads") or 0
    m3_topk_blocks = sparse_cfg.get("sparse_topk_blocks") or 0
    m3_block = sparse_cfg.get("sparse_block_size") or 0
    # ``rotary_dim`` = head_dim * partial_rotary_factor; only that slice is
    # rotated.
    m3_rope_dim = config.get("rotary_dim") or head_dim

    # --- Gated DeltaNet (Qwen3.5 / 3.6 / 3.8) -----------------------------
    # No KV cache: a fixed conv state plus a fixed recurrent state **per
    # sequence**. Field names from vLLM's ``gated_delta_net_state_shape``.
    gdn_k_heads = config.get("linear_num_key_heads") or 0
    gdn_v_heads = config.get("linear_num_value_heads") or 0
    gdn_k_dim = config.get("linear_key_head_dim") or 0
    gdn_v_dim = config.get("linear_value_head_dim") or 0
    gdn_conv_kernel = config.get("linear_conv_kernel_dim") or 0
    gdn_key_dim = gdn_k_heads * gdn_k_dim
    gdn_value_dim = gdn_v_heads * gdn_v_dim
    gdn_conv_dim = 2 * gdn_key_dim + gdn_value_dim
    # Qwen3-Next / Qwen3.5 fuse an output gate into the Q half of the QKV
    # projection, doubling its Q width
    # (``total_num_heads * (1 + attn_output_gate)``). vLLM's class defaults the
    # flag to True, but that default only reaches checkpoints on that code
    # path -- which is exactly the ones with a linear-attention stack -- so it
    # is defaulted here the same way rather than globally. Plain Qwen3 and
    # MiniMax-M3 have no gate.
    attn_output_gate = bool(config.get("attn_output_gate", bool(gdn_k_heads)))
    q_out_dim = q_dim * (2 if attn_output_gate else 1)
    # The shared expert has its own width on M3.
    shared_ffn_dim = config.get("shared_intermediate_size", moe_ffn_dim)

    p = max(int(parallel), 1)

    # NOTE (vLLM-style assumptions):
    # NOTE (vLLM-style assumptions):
    # - Embedding / LM head: vocab-parallel → split vocab_size across ranks.
    # - Q/K/V: ColumnParallelLinear         → split output dim across ranks.
    # - o_proj: RowParallelLinear           → split input dim across ranks.
    # - LayerNorm weights: replicated (NOT sharded).
    # - MoE experts: parallel = EP degree, each rank holds num_local_experts // p experts.

    # ----------------- Embedding & Norms -----------------
    if layer_name in ("embedding", "mtp_embed"):
        # ``mtp_embed`` is the drafter's own embed_tokens -- the same shape as
        # the target's, and a separate table: vLLM's MTP wrappers build one
        # rather than sharing.
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
        # MLA rotates only the rope half of Q and the single shared K rope
        # vector (the rest of K lives compressed in the latent), so it is
        # ``qk_rope_head_dim`` wide over ``n_head + 1`` rows rather than
        # ``head_dim`` over ``n_head + kv_head``.
        if is_mla:
            input_size = ((n_head // p) + 1) * length * rope_dim * fp
            output_size = input_size
        else:
            input_size = ((n_head // p) + (kv_head // p)) * length * head_dim * fp
            output_size = input_size
        weight_size = 0

    elif layer_name == "attention":
        if is_mla and not pim:
            # MLA reads a single latent per cached token rather than K and V
            # per KV head, and that latent is replicated across ranks, so the
            # KV term does not divide by ``p``. Q is ``qk_head_dim`` wide
            # (nope + rope) and the output is ``v_head_dim``, which is a
            # different width again -- 192 in, 128 out on DeepSeek-V3.2.
            latent = kv_lora_rank + qk_rope_dim
            input_size = (
                (n_head // p) * length * qk_head_dim * fp
                + (kv_len or 0) * latent * fp
            )
            weight_size = 0
            output_size = (n_head // p) * length * v_head_dim * fp
        elif not pim:
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
        weight_size = n_embd * ((q_out_dim + 2 * kv_dim) // p) * fp
        output_size = length * ((q_out_dim + 2 * kv_dim) // p) * fp

    elif layer_name == "o_proj":
        # RowParallelLinear: sharded on the input dim. That dim is
        # ``n_head * v_head_dim`` under MLA, where V is a different width from
        # Q -- 128 x 128 on DeepSeek-V3.2, 64 x 256 on GLM-5 -- and
        # ``n_head * head_dim`` everywhere else.
        input_size = length * (o_in_dim // p) * fp
        weight_size = (o_in_dim // p) * n_embd * fp
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

    # ----------------- MLA (DeepSeek-V3.2, GLM-5) -----------------
    # Shapes read off vLLM's ``deepseek_v2.py``, not inferred. Checked against
    # the published parameter count: summing these over the checkpoint's own
    # layer composition gives 671.9B for DeepSeek-V3.2-Exp against a published
    # 671B.
    elif layer_name == "mla_qkv_a_proj":
        # ``DeepSeekV2FusedQkvAProjLinear``, one GEMM producing the q latent
        # and the kv latent + K rope. It subclasses MergedColumnParallelLinear
        # but passes ``disable_tp=True``, so it is **replicated** -- every rank
        # needs the whole latent. (vLLM asserts the fused weight is
        # 2112 x 7168 on V3.2, i.e. (1536 + 512 + 64) x hidden.)
        latent_out = q_lora_rank + kv_lora_rank + qk_rope_dim
        input_size = length * n_embd * fp
        weight_size = n_embd * latent_out * fp
        output_size = length * latent_out * fp

    elif layer_name == "mla_a_layernorm":
        # q_a_layernorm + kv_a_layernorm. Both are ``RMSNorm`` (scale only, no
        # bias) inside ``DeepseekV2MLAAttention``, so the profiler reports them
        # as one merged node and this entry carries their **sum**.
        input_size = length * (q_lora_rank + kv_lora_rank) * fp
        weight_size = (q_lora_rank + kv_lora_rank) * fp
        output_size = length * (q_lora_rank + kv_lora_rank) * fp

    elif layer_name == "mla_b_proj":
        # q_b_proj + kv_b_proj, both ColumnParallelLinear and both reported as
        # one merged node, so again the sum. Sharded on the output dim.
        q_b_out = n_head * qk_head_dim
        kv_b_out = n_head * (qk_nope_dim + v_head_dim)
        input_size = length * (q_lora_rank + kv_lora_rank) * fp
        weight_size = (q_lora_rank * (q_b_out // p)
                       + kv_lora_rank * (kv_b_out // p)) * fp
        output_size = length * ((q_b_out + kv_b_out) // p) * fp

    # ----------------- DeepSeek sparse-attention indexer -----------------
    elif layer_name == "indexer_wq_b":
        # ``ReplicatedLinear`` by construction -- the comment in vLLM is
        # literally "no tensor parallel, just replicated".
        idx_q = idx_heads * idx_head_dim
        input_size = length * q_lora_rank * fp
        weight_size = q_lora_rank * idx_q * fp
        output_size = length * idx_q * fp

    elif layer_name == "indexer_wk_proj":
        # ``wk_weights_proj``: one MergedColumnParallelLinear with
        # ``disable_tp=True``, producing [index_head_dim | index_n_heads] --
        # the index key and the per-head weights -- in a single GEMM.
        idx_kw = idx_head_dim + idx_heads
        input_size = length * n_embd * fp
        weight_size = n_embd * idx_kw * fp
        output_size = length * idx_kw * fp

    elif layer_name == "indexer_k_norm":
        # A bare ``nn.LayerNorm``, not RMSNorm, so it carries weight **and**
        # bias.
        input_size = length * idx_head_dim * fp
        weight_size = 2 * idx_head_dim * fp
        output_size = length * idx_head_dim * fp

    elif layer_name == "indexer_rope_emb":
        # The indexer's own rope: the rope slice of every index-query head,
        # plus the single MQA index key. Shared instance across layers, so no
        # weight of its own beyond the cos/sin cache, which is not per-layer.
        input_size = (idx_heads + 1) * length * qk_rope_dim * fp
        weight_size = 0
        output_size = input_size

    elif layer_name == "indexer_q_rope_quant":
        # ``fused_indexer_q_rope_quant``: rope on the index queries plus the
        # fp8 quantization they are scored in. Output is 1 byte per element
        # regardless of ``fp``, plus one fp32 scale per 128-element block --
        # the ``quant_block_size`` vLLM hardcodes -- and the fp32 weights.
        idx_q = idx_heads * idx_head_dim
        scales = idx_heads * max(1, idx_head_dim // 128)
        input_size = length * (idx_q + idx_heads) * fp
        weight_size = 0
        output_size = length * (idx_q + scales * 4 + idx_heads * 4)

    elif layer_name == "indexer":
        # ``SparseAttnIndexer``: scores every cached index key against this
        # step's fp8 queries and returns the top-k token ids. The read over
        # the KV is what makes it an attention-category layer, and it is the
        # indexer's **own** cache -- fp8 values plus one fp32 scale per
        # 128-element block -- not the MLA latent cache.
        idx_q = idx_heads * idx_head_dim
        idx_cache_per_tok = idx_head_dim + (idx_head_dim // 128) * 4
        kv = kv_len if kv_len is not None else length
        input_size = length * idx_q + kv * idx_cache_per_tok
        weight_size = 0
        output_size = length * idx_topk * 4  # int32 token indices

    elif layer_name == "indexer_glue":
        # Not one tensor: the bucket of eager reshape/copy/index_put work the
        # indexer does around its kernels, bound as a group in the catalog so
        # its measured cost is not silently dropped. Sized as one pass over
        # the index queries, which is what those ops move; the *latency* is
        # measured, this is a stand-in for the byte counts.
        idx_q = idx_heads * idx_head_dim
        input_size = length * idx_q * fp
        weight_size = 0
        output_size = length * idx_q * fp

    # ----------------- MiniMax-M3 block-sparse attention -----------------
    # Shapes read off vLLM's ``models/minimax_m3/nvidia/model.py`` and
    # ``MinimaxM3QKVParallelLinearWithIndexer``.
    elif layer_name == "qk_norm_rope":
        # One fused kernel doing per-head QK-norm, rope, the output gate where
        # the family has one, and the KV-cache write. Serves MiniMax-M3's
        # non-sparse layers and Qwen3.5's full-attention ones. The q_norm and
        # k_norm modules launch no kernel of their own, so this entry is where
        # their parameters live -- ``head_dim`` each. The gate has none: it is
        # fused into the Q half of ``qkv_proj``.
        input_size = length * ((q_out_dim + 2 * kv_dim) // p) * fp
        weight_size = 2 * head_dim * fp
        output_size = input_size

    elif layer_name == "sparse_qkv_proj":
        # A single column-parallel GEMM emitting [q | k | v | index_q |
        # index_k]. ``index_q`` has the KV head count and shares ``head_dim``,
        # so it shards exactly like K and V; ``index_k`` is one shared head,
        # **replicated** to every rank.
        fused = ((n_head // p) * head_dim
                 + 2 * (kv_head // p) * head_dim
                 + (kv_head // p) * m3_idx_dim
                 + m3_idx_dim)
        input_size = length * n_embd * fp
        weight_size = n_embd * fused * fp
        output_size = length * fused * fp

    elif layer_name == "sparse_qk_norm_rope":
        # The same fused kernel on a sparse layer, where it norms four things
        # rather than two: q, k, index_q and index_k. All four are
        # ``MiniMAXGemmaRMSNorm`` over ``head_dim`` (``sparse_index_dim`` is
        # equal to it, which the fused projection asserts), which is why the
        # profiler reports them as one node.
        fused = ((n_head // p) * head_dim
                 + 2 * (kv_head // p) * head_dim
                 + (kv_head // p) * m3_idx_dim
                 + m3_idx_dim)
        input_size = length * fused * fp
        weight_size = (2 * head_dim + 2 * m3_idx_dim) * fp
        output_size = input_size

    elif layer_name == "sparse_o_proj":
        # Same shape as ``o_proj``; a separate canonical name only because the
        # profiler must not merge the sparse layers' timing with the others'.
        input_size = length * (q_dim // p) * fp
        weight_size = (q_dim // p) * n_embd * fp
        output_size = length * n_embd * fp

    elif layer_name == "indexer":
        # M3's lightning indexer scores **blocks**, not tokens: it reads its
        # own key-only side cache over the whole KV and returns the top
        # ``sparse_topk_blocks`` block ids. The side cache is one vector per
        # token (``MLAAttentionSpec`` with ``num_kv_heads = 1``), so it is
        # replicated across ranks rather than sharded.
        kv = kv_len if kv_len is not None else length
        input_size = (length * (kv_head // p) * m3_idx_dim * fp
                      + kv * m3_idx_dim * fp)
        weight_size = 0
        output_size = length * m3_topk_blocks * 4  # int32 block ids

    elif layer_name == "sparse_attention":
        # Attention over the selected blocks only, which is what caps the work:
        # at most ``sparse_topk_blocks * sparse_block_size`` keys are read
        # however long the sequence is. The KV *capacity* is unchanged -- every
        # token is still cached -- so this bound belongs to the read, not to
        # ``kv_bytes_per_token_per_layer``.
        kv = kv_len if kv_len is not None else length
        read = min(kv, m3_topk_blocks * m3_block) if m3_block else kv
        input_size = ((n_head // p) * length * head_dim * fp
                      + (kv_head // p) * read * head_dim * fp * 2)
        weight_size = 0
        output_size = (n_head // p) * length * head_dim * fp

    # ----------------- Gated DeltaNet (Qwen3.5 / 3.6 / 3.8) -----------------
    # Shapes from vLLM's ``layers/mamba/gdn/qwen_gdn_linear_attn.py``. Each
    # parameter is attributed to exactly one canonical name, because
    # ``get_weight`` sums every name a block emits: the conv weights go to the
    # prefill conv entry and the recurrence's ``dt_bias`` / ``A_log`` to the
    # prefill recurrence, with the decode entries carrying 0 -- they read the
    # same weights, and counting them twice would inflate the block.
    elif layer_name == "gdn_in_proj":
        # ``in_proj_qkvz`` (hidden -> 2*key_dim + 2*value_dim) and
        # ``in_proj_ba`` (hidden -> 2*num_v_heads), reported by the profiler as
        # one merged node, so this entry carries their **sum**. On Qwen3.8-27B
        # that is 5120->16384 beside 5120->96: a large GEMM and a tiny one,
        # which is why the merged node's mean describes neither and the sum is
        # what the catalog models.
        qkvz = 2 * gdn_key_dim + 2 * gdn_value_dim
        ba = 2 * gdn_v_heads
        input_size = length * n_embd * fp
        weight_size = n_embd * ((qkvz + ba) // p) * fp
        output_size = length * ((qkvz + ba) // p) * fp

    elif layer_name == "gdn_conv_prefill":
        # Depthwise causal conv over the whole chunk. vLLM stores the weight as
        # ``ColumnParallelLinear(conv_kernel_size -> conv_dim)``.
        input_size = length * (gdn_conv_dim // p) * fp
        weight_size = gdn_conv_kernel * (gdn_conv_dim // p) * fp
        output_size = length * (gdn_conv_dim // p) * fp

    elif layer_name == "gdn_conv_decode":
        # Same weights as the prefill conv, so no weight of its own here. One
        # step against the rolling conv state.
        input_size = length * (gdn_conv_dim // p) * fp
        weight_size = 0
        output_size = length * (gdn_conv_dim // p) * fp

    elif layer_name in ("gdn_post_conv", "gdn_glue"):
        # ``gdn_post_conv`` is a fused kernel splitting the conv output into
        # q/k/v and applying the gates; ``gdn_glue`` is the block's eager
        # reshape/copy work, bound as a group so its measured cost is not
        # dropped. Neither has parameters; both move one conv-width activation.
        input_size = length * (gdn_conv_dim // p) * fp
        weight_size = 0
        output_size = length * (gdn_conv_dim // p) * fp

    elif layer_name == "gdn_norm":
        # ``RMSNormGated(head_v_dim)`` -- one scale shared across value heads,
        # applied to the recurrence output.
        input_size = length * (gdn_value_dim // p) * fp
        weight_size = gdn_v_dim * fp
        output_size = length * (gdn_value_dim // p) * fp

    elif layer_name == "gdn_out_proj":
        input_size = length * (gdn_value_dim // p) * fp
        weight_size = (gdn_value_dim // p) * n_embd * fp
        output_size = length * n_embd * fp

    elif layer_name == "gdn_prefill":
        # ``ChunkGatedDeltaRule``: the chunked scan over the whole prefill.
        # Carries the recurrence's own parameters -- ``dt_bias`` (one per value
        # head) and ``A_log`` (one per value head, fp32) -- which the decode
        # kernels read but must not re-count.
        state = (gdn_v_heads // p) * gdn_v_dim * gdn_k_dim * fp
        input_size = length * (gdn_conv_dim // p) * fp + state
        weight_size = (gdn_v_heads // p) * fp + (gdn_v_heads // p) * 4
        output_size = length * (gdn_value_dim // p) * fp + state

    elif layer_name in ("gdn_decode", "gdn_decode_mixed"):
        # One recurrent step per sequence. Cost is independent of sequence
        # length -- the state is fixed size, which is the whole point of a
        # linear-attention layer -- so ``length`` here is the decode count.
        state = (gdn_v_heads // p) * gdn_v_dim * gdn_k_dim * fp
        input_size = length * ((gdn_conv_dim // p) * fp + state)
        weight_size = 0
        output_size = length * ((gdn_value_dim // p) * fp + state)

    elif layer_name in ("attn_glue",):
        # The full-attention block's eager glue on a hybrid stack.
        input_size = length * ((q_out_dim + 2 * kv_dim) // p) * fp
        weight_size = 0
        output_size = input_size

    elif layer_name == "sampler":
        input_size = length * (vocab_size // p) * fp
        weight_size = 0
        output_size = length * 4  # int32 token IDs

    elif layer_name == "moe":
        # The whole block, matching what the catalog binds: gate, this rank's
        # routed experts, and the shared expert(s). A shared expert runs on
        # every token beside the routed ones and is replicated, not sharded
        # (DeepSeek and GLM ship one; families without the field ship none).
        experts_per_rank = num_local_experts // p
        input_size = length * n_embd * fp
        weight_size = (n_embd * num_local_experts * fp  # gate (replicated)
                     + experts_per_rank * 3 * n_embd * moe_ffn_dim * fp
                     + num_shared_experts * 3 * n_embd * shared_ffn_dim * fp)
        output_size = length * n_embd * fp

    # ----------------- LM Head -----------------
    elif layer_name == "lm_head":
        input_size = length * n_embd * fp
        weight_size = n_embd * (vocab_size // p) * fp
        output_size = length * (vocab_size // p) * fp

    # ----------------- MTP (the model's own drafter) -----------------
    # One module per declared MTP layer, run N times per speculative step.
    # Shapes read off live module dumps of all three families -- they agree on
    # the arithmetic and differ only in how the combine projection shards.
    elif layer_name in ("mtp_eh_proj", "mtp_fc"):
        # Concatenates the token embedding with the target's hidden state and
        # projects 2h -> h. ``mtp_eh_proj`` is DeepSeek/GLM's plain nn.Linear
        # and MiniMax-M3's ReplicatedLinear, both **unsharded**; ``mtp_fc`` is
        # Qwen's ColumnParallelLinear, which is sharded. Same math, different
        # divisor -- which is why they are separate names rather than one.
        shard = p if layer_name == "mtp_fc" else 1
        input_size = length * 2 * n_embd * fp
        weight_size = (2 * n_embd) * (n_embd // shard) * fp
        output_size = length * (n_embd // shard) * fp

    elif layer_name in ("mtp_enorm_hnorm", "mtp_norms"):
        # The norms in front of that projection, reported as one merged node:
        # DeepSeek/GLM's enorm + hnorm (2), Qwen's norm + pre_fc_norm_hidden +
        # pre_fc_norm_embedding (3). Scale-only weights, so the count barely
        # matters, but it is the honest shape.
        count = 2 if layer_name == "mtp_enorm_hnorm" else 3
        input_size = length * n_embd * fp
        weight_size = count * n_embd * fp
        output_size = length * n_embd * fp

    elif layer_name == "mtp_shared_head":
        # DeepSeek/GLM's SharedHead: a norm plus the LM head. Reported under
        # the wrapper, so its size is the pair's.
        input_size = length * n_embd * fp
        weight_size = (n_embd + n_embd * (vocab_size // p)) * fp
        output_size = length * (vocab_size // p) * fp

    elif layer_name == "mtp_glue":
        # Eager elementwise work around the projection -- reshapes and copies,
        # no parameters of its own.
        input_size = length * n_embd * fp
        weight_size = 0
        output_size = length * n_embd * fp

    else:
        raise ValueError(f"No matching layer name {layer_name} found for model {model}.")

    return input_size, weight_size, output_size
