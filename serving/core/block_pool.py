"""Per-tier KV cache block pool.

Port of vLLM v0.19.0's ``vllm/v1/core/block_pool.py`` and the block/queue
primitives in ``vllm/v1/core/kv_cache_utils.py``, adapted to a tiered memory
hierarchy (NPU / CPU / CXL).

The pool is the single authority for one tier: it owns the free list, the
prefix-cache index, and the reference counts, so ``num_free_blocks`` is exact
and an allocation either succeeds or reports failure in the same call. That is
the property the previous radix-tree allocator could not provide -- there,
``evictable_size_`` counted every unlocked token while ``evict()`` could only
drop unlocked *leaves*, so the scheduler could believe in space that did not
exist.

Deliberate differences from vLLM, each noted at its site:

* No ``null_block``. Sliding-window and Mamba layouts are out of scope, so
  block 0 is a normal block and ``usage()`` does not subtract one.
* ``block_size`` and ``bytes_per_block`` are pool attributes, so one class
  serves every tier at its own granularity (the NPU uses ``--block-size``,
  a host offload tier uses a multiple of it -- LMCache's chunk is 256) and its
  own byte scale (per-rank on the NPU, full-cluster on CPU/CXL).
* ``raise`` rather than ``assert`` on over-allocation, so a bug surfaces in a
  released run instead of vanishing under ``-O``.

There is no swap or demotion hook. Eviction is a silent side effect of
allocation: dropping a cached block's hash costs nothing in any mode, and the
only critical-path KV transfer is a *recall* from a lower tier, which the tier
manager accounts for.
"""

from enum import Enum

from .logger import get_logger

MB_TO_BYTE = 1024 * 1024


class Device(Enum):
    NPU = 1
    CPU = 2
    CXL = 3


# Seed of the chained block-hash. vLLM draws this from ``os.urandom`` unless
# PYTHONHASHSEED is set; the simulator must be reproducible run to run, so it is
# a fixed constant.
NONE_HASH = 0x9E3779B97F4A7C15


class KVCacheBlock:
    """Metadata for one KV cache block.

    ``prev_free_block`` / ``next_free_block`` are owned exclusively by
    :class:`FreeKVCacheBlockQueue`; nothing else may touch them.
    """

    __slots__ = ("block_id", "ref_cnt", "block_hash",
                 "prev_free_block", "next_free_block")

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_cnt = 0
        # Set only while the block is full and indexed for prefix caching.
        self.block_hash = None
        self.prev_free_block = None
        self.next_free_block = None

    def reset_hash(self):
        self.block_hash = None

    def __repr__(self):
        prev_id = self.prev_free_block.block_id if self.prev_free_block else None
        next_id = self.next_free_block.block_id if self.next_free_block else None
        return (f"KVCacheBlock(block_id={self.block_id}, ref_cnt={self.ref_cnt}, "
                f"block_hash={self.block_hash}, prev={prev_id}, next={next_id})")


class FreeKVCacheBlockQueue:
    """Doubly linked list of free blocks, ordered by eviction priority.

    A plain deque cannot remove from the middle in O(1), which ``touch()``
    needs when a request hits a block that is sitting in the free list. The
    links live on the blocks themselves, so manipulating the list allocates
    nothing.

    Ordering: blocks start in block-id order. A freed block is appended to the
    **tail**, so it is reused last. That is what lets a just-preempted
    request's blocks survive a small reclaim and be found again on resume.
    Callers free in reverse order (tail block first) so that when the pool does
    come under pressure the tail is taken first and the head -- the
    recoverable prefix -- survives longest.
    """

    def __init__(self, blocks):
        self.num_free_blocks = len(blocks)

        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # Fake head and tail remove the branching around the list ends. They
        # are never popped.
        self.fake_head = KVCacheBlock(block_id=-1)
        self.fake_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            self.fake_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_head
            self.fake_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_tail
        else:
            self.fake_head.next_free_block = self.fake_tail
            self.fake_tail.prev_free_block = self.fake_head

    def popleft(self):
        blocks = self.popleft_n(1)
        if not blocks:
            raise RuntimeError("[FreeKVCacheBlockQueue] no free blocks available")
        return blocks[0]

    def popleft_n(self, n):
        """Pop the ``n`` least recently freed blocks."""
        if n == 0:
            return []
        if self.num_free_blocks < n:
            raise RuntimeError(
                f"[FreeKVCacheBlockQueue] asked for {n} blocks but only "
                f"{self.num_free_blocks} are free"
            )
        self.num_free_blocks -= n

        curr = self.fake_head.next_free_block
        ret = []
        for _ in range(n):
            ret.append(curr)
            last = curr
            curr = curr.next_free_block
            last.prev_free_block = None
            last.next_free_block = None

        self.fake_head.next_free_block = curr
        curr.prev_free_block = self.fake_head
        return ret

    def remove(self, block):
        """Take ``block`` out of the middle of the list."""
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"[FreeKVCacheBlockQueue] remove() on a block that "
                               f"is not in the free list: {block}")
        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block):
        self.append_n([block])

    def append_n(self, blocks):
        """Put blocks back at the tail, so they are reused last."""
        if not blocks:
            return
        last = self.fake_tail.prev_free_block
        for block in blocks:
            block.prev_free_block = last
            last.next_free_block = block
            last = block
        last.next_free_block = self.fake_tail
        self.fake_tail.prev_free_block = last
        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self):
        """Free blocks in eviction order. For tests and diagnostics."""
        ret = []
        curr = self.fake_head.next_free_block
        while curr is not self.fake_tail:
            ret.append(curr)
            curr = curr.next_free_block
        return ret


class PrefixCacheStats:
    """Requested / hit prompt-token counters for one tier.

    Replaces ``RadixCache.total_requested_tokens`` and ``total_hit_tokens``,
    keeping the two accessors the run summary and the per-iteration status line
    already call.
    """

    def __init__(self):
        self.total_requested_tokens = 0
        self.total_hit_tokens = 0

    def record(self, num_tokens, num_hits):
        self.total_requested_tokens += num_tokens
        self.total_hit_tokens += num_hits

    def return_prefix_info(self):
        return self.total_requested_tokens, self.total_hit_tokens

    def format_prefix_info(self):
        if self.total_requested_tokens == 0:
            return ""
        ratio = (self.total_hit_tokens / self.total_requested_tokens) * 100
        return (f", Prefix Cache Hit ratio {ratio:.2f} %, "
                f"({self.total_hit_tokens} / {self.total_requested_tokens})")


class BlockPool:
    """Blocks of one memory tier, with the prefix-cache index over them.

    Args:
        tier: which :class:`Device` these blocks live on.
        num_blocks: pool size. For the NPU this comes from
            ``npu_mem.mem_size * npu_mem.mem_util - weight``, mirroring vLLM's
            ``requested_memory - non_kv_cache_memory``.
        block_size: tokens per block. The NPU uses ``--block-size``; a lower
            tier uses a multiple of it.
        bytes_per_block: KV bytes one block occupies *on this tier*. Per-rank
            for the NPU, full-cluster for CPU/CXL.
        enable_caching: when false, blocks are allocated and freed through the
            same free list but never hashed or indexed -- exactly what
            ``--no-enable-prefix-caching`` means in vLLM.
    """

    def __init__(self, tier, num_blocks, block_size, bytes_per_block,
                 enable_caching=True, node_id=None, instance_id=None):
        if not isinstance(num_blocks, int) or num_blocks <= 0:
            raise ValueError(
                f"[BlockPool] {tier}: num_blocks must be a positive int, got {num_blocks}"
            )
        self.tier = tier
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.bytes_per_block = bytes_per_block
        self.enable_caching = enable_caching

        self.blocks = [KVCacheBlock(i) for i in range(num_blocks)]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        # block_hash -> KVCacheBlock. vLLM keeps a hash -> {block_id: block}
        # map because its block ids must stay stable for append-only block
        # tables on the worker; the simulator has no worker, so a single block
        # per hash is enough.
        self.cached_block_hash_to_block = {}

        self.stats = PrefixCacheStats()
        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)

    # -------------------- prefix cache index --------------------

    def get_cached_block(self, block_hash):
        """The block indexed under ``block_hash``, or None."""
        if not self.enable_caching:
            return None
        return self.cached_block_hash_to_block.get(block_hash)

    def cache_full_blocks(self, block_hashes, blocks, num_cached_blocks, num_full_blocks):
        """Index ``blocks[num_cached_blocks:num_full_blocks]`` under their hashes.

        Only full blocks are indexed, which is why a request can never recover
        the tail of its sequence past the last block boundary.
        """
        if not self.enable_caching or num_cached_blocks >= num_full_blocks:
            return
        if len(block_hashes) < num_full_blocks:
            raise RuntimeError(
                f"[BlockPool] {self.tier}: need {num_full_blocks} block hashes "
                f"but only {len(block_hashes)} were computed"
            )
        for i in range(num_cached_blocks, num_full_blocks):
            block = blocks[i]
            if block.block_hash is not None:
                # Already indexed; nothing to do. vLLM asserts here, but a
                # re-cache after a resume is legitimate in the simulator.
                continue
            block_hash = block_hashes[i]
            block.block_hash = block_hash
            self.cached_block_hash_to_block[block_hash] = block

    def cache_copy(self, block_hash):
        """Make ``block_hash`` resident on this tier as an unpinned copy.

        Used for an inclusive lower tier: a block completed on the NPU is
        written down here, then immediately unpinned so it sits in the free
        list -- findable by hash, and the first thing dropped when the tier
        fills. That is the whole victim-cache behaviour, including
        "overflow drops the least recently written".

        Returns True when a new copy was placed, False when the hash was
        already resident (so no traffic should be charged).
        """
        if not self.enable_caching:
            return False
        if block_hash in self.cached_block_hash_to_block:
            return False
        block = self.get_new_blocks(1)[0]
        block.block_hash = block_hash
        self.cached_block_hash_to_block[block_hash] = block
        self.free_blocks([block])
        return True

    # -------------------- allocation --------------------

    def get_new_blocks(self, num_blocks):
        """Take ``num_blocks`` from the free list and pin them.

        Eviction happens here and nowhere else: if a popped block still carries
        a hash, that hash is dropped. Silently -- no traffic is charged, in any
        mode. The block's data either was never needed again (a finished
        request's cache) or already has a copy on a lower tier, written
        off the critical path.
        """
        if num_blocks > self.get_num_free_blocks():
            raise RuntimeError(
                f"[BlockPool] {self.tier}: cannot get {num_blocks} free blocks, "
                f"only {self.get_num_free_blocks()} of {self.num_blocks} are free"
            )
        blocks = self.free_block_queue.popleft_n(num_blocks)
        for block in blocks:
            if self.enable_caching:
                self._maybe_evict_cached_block(block)
            if block.ref_cnt != 0:
                raise RuntimeError(
                    f"[BlockPool] {self.tier}: popped block {block.block_id} "
                    f"has ref_cnt={block.ref_cnt}, expected 0"
                )
            block.ref_cnt += 1
        return blocks

    def _maybe_evict_cached_block(self, block):
        """Drop ``block``'s index entry, if it has one."""
        block_hash = block.block_hash
        if block_hash is None:
            return False
        if self.cached_block_hash_to_block.get(block_hash) is not block:
            # A newer block took over this hash; leave the index alone.
            block.reset_hash()
            return False
        del self.cached_block_hash_to_block[block_hash]
        block.reset_hash()
        return True

    def touch(self, blocks):
        """Claim already-cached blocks for one more request.

        A block with ``ref_cnt == 0`` is an eviction candidate sitting in the
        free list, so it has to come out of the list before it is pinned.
        """
        for block in blocks:
            if block.ref_cnt == 0:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1

    def free_blocks(self, ordered_blocks):
        """Release blocks, keeping their hashes so they stay findable.

        ``ordered_blocks`` must be ordered by eviction priority -- callers pass
        a request's blocks in reverse, so the tail is reused first.
        """
        blocks = list(ordered_blocks)
        for block in blocks:
            if block.ref_cnt <= 0:
                raise RuntimeError(
                    f"[BlockPool] {self.tier}: freeing block {block.block_id} "
                    f"with ref_cnt={block.ref_cnt}"
                )
            block.ref_cnt -= 1
        self.free_block_queue.append_n([b for b in blocks if b.ref_cnt == 0])

    # -------------------- accounting --------------------

    def get_num_free_blocks(self):
        return self.free_block_queue.num_free_blocks

    @property
    def used_blocks(self):
        return self.num_blocks - self.get_num_free_blocks()

    def used_bytes(self):
        """Bytes held by blocks that are pinned or still indexed.

        A block in the free list that keeps its hash is still holding data --
        it is reusable, not empty -- so it counts as used. This is the tier's
        one ledger; there is no second counter to keep in sync.
        """
        pinned = self.used_blocks
        cached_and_free = sum(
            1 for b in self.cached_block_hash_to_block.values() if b.ref_cnt == 0
        )
        return (pinned + cached_and_free) * self.bytes_per_block

    def usage(self):
        """Fraction of the pool that is pinned, matching vLLM's kv_cache_usage."""
        if self.num_blocks == 0:
            return 0.0
        return self.used_blocks / self.num_blocks

    def is_free(self):
        """True when every block is back in the free list. End-of-run leak check."""
        return self.get_num_free_blocks() == self.num_blocks

    def reset(self):
        """Drop the whole index and unpin everything."""
        self.cached_block_hash_to_block = {}
        for block in self.blocks:
            block.ref_cnt = 0
            block.reset_hash()
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

    def __repr__(self):
        return (f"BlockPool(tier={self.tier}, num_blocks={self.num_blocks}, "
                f"free={self.get_num_free_blocks()}, "
                f"cached={len(self.cached_block_hash_to_block)}, "
                f"block_size={self.block_size})")


def _selftest():
    """Runs with a plain ``python3 -m serving.core.block_pool``."""
    KB = 1024

    def new_pool(n=8, caching=True):
        return BlockPool(Device.NPU, n, block_size=16,
                         bytes_per_block=16 * 128 * KB, enable_caching=caching)

    # allocate / free round-trip restores the free count
    p = new_pool()
    blocks = p.get_new_blocks(3)
    assert p.get_num_free_blocks() == 5, p
    p.free_blocks(reversed(blocks))
    assert p.is_free(), p

    # a freed block is reused LAST, so a just-preempted request survives a
    # small reclaim
    p = new_pool()
    first = p.get_new_blocks(2)
    p.free_blocks(reversed(first))
    order = [b.block_id for b in p.free_block_queue.get_all_free_blocks()]
    assert order == [2, 3, 4, 5, 6, 7, 1, 0], order

    # an indexed block popped for reuse loses its hash; a live one is untouched
    p = new_pool(n=3)
    blocks = p.get_new_blocks(2)
    p.cache_full_blocks([111, 222], blocks, 0, 2)
    assert p.get_cached_block(111) is blocks[0]
    p.free_blocks(reversed(blocks))
    assert p.get_cached_block(222) is blocks[1], "hash must survive a free"
    p.get_new_blocks(3)  # forces both cached blocks out
    assert p.get_cached_block(111) is None and p.get_cached_block(222) is None
    assert len(p.cached_block_hash_to_block) == 0

    # touch() pulls a cached, unpinned block back out of the free list
    p = new_pool()
    blocks = p.get_new_blocks(1)
    p.cache_full_blocks([777], blocks, 0, 1)
    p.free_blocks(blocks)
    assert p.get_num_free_blocks() == 8
    hit = p.get_cached_block(777)
    p.touch([hit])
    assert hit.ref_cnt == 1 and p.get_num_free_blocks() == 7

    # shared prefix: two requests on one block, freed once each
    p = new_pool()
    shared = p.get_new_blocks(1)
    p.cache_full_blocks([42], shared, 0, 1)
    p.touch(shared)
    assert shared[0].ref_cnt == 2
    p.free_blocks(shared)
    assert shared[0].ref_cnt == 1 and p.get_num_free_blocks() == 7
    p.free_blocks(shared)
    assert p.is_free()

    # used_bytes counts a cached-but-unpinned block as occupied
    p = new_pool()
    blocks = p.get_new_blocks(2)
    p.cache_full_blocks([1, 2], blocks, 0, 2)
    assert p.used_bytes() == 2 * p.bytes_per_block
    p.free_blocks(reversed(blocks))
    assert p.used_bytes() == 2 * p.bytes_per_block, "still holding data"

    # caching disabled: allocation is identical, the index stays empty
    p = new_pool(caching=False)
    blocks = p.get_new_blocks(4)
    p.cache_full_blocks([1, 2, 3, 4], blocks, 0, 4)
    assert len(p.cached_block_hash_to_block) == 0
    assert p.get_cached_block(1) is None
    p.free_blocks(reversed(blocks))
    assert p.is_free()

    # over-allocation raises rather than silently wrapping
    p = new_pool(n=2)
    p.get_new_blocks(2)
    try:
        p.get_new_blocks(1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("over-allocation must raise")

    # a lower tier at a coarser granularity is the same class
    cpu = BlockPool(Device.CPU, 4, block_size=256,
                    bytes_per_block=256 * 128 * KB, enable_caching=True)
    assert cpu.block_size == 16 * 16
    assert cpu.bytes_per_block == 16 * p.bytes_per_block

    # cache_copy: inclusive victim cache, unpinned, LRU-dropped on overflow
    assert cpu.cache_copy(1001) is True
    assert cpu.cache_copy(1001) is False, "already resident, must not re-charge"
    assert cpu.get_cached_block(1001).ref_cnt == 0, "copies stay unpinned"
    assert cpu.is_free(), "a copy occupies the index, not the free list"
    for h in (1002, 1003, 1004):
        assert cpu.cache_copy(h) is True
    assert len(cpu.cached_block_hash_to_block) == 4
    assert cpu.cache_copy(1005) is True                  # forces an eviction
    assert cpu.get_cached_block(1001) is None, "oldest copy dropped first"
    assert len(cpu.cached_block_hash_to_block) == 4

    print("block_pool self-test: all checks passed")


if __name__ == "__main__":
    _selftest()
