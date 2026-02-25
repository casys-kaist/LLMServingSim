from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import threading
import heapq
import time
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, List, Optional, NamedTuple, Any
try:
    import msgspec
except ImportError:
    msgspec = None
from .logger import get_logger

GB_TO_BYTE = 1024 * 1024 * 1024
MB_TO_BYTE = 1024 * 1024
KB_TO_BYTE = 1024

class KVCacheEvent(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,
):
    """Base class for all KV cache-related events"""


class BlockStored(KVCacheEvent):
    block_hashes: list[int]
    parent_block_hash: Optional[int]
    token_ids: list[int]
    block_size: int
    lora_id: Optional[int]


class BlockRemoved(KVCacheEvent):
    block_hashes: list[int]


class AllBlocksCleared(KVCacheEvent):
    pass

class MatchResult(NamedTuple):
    """Result of a prefix match operation.

    Attributes:
        last_device_node:   The last TreeNode on the device that was matched.
        hit_length :   Length of the KV cache hit
    """
    last_device_node: Any
    hit_length: int = 0


class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None, device: Optional[str] = None):
        # self.children = defaultdict(TreeNode)
        self.children = dict()
        self.parent: TreeNode = None
        self.key: List[int] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None

        self.id = TreeNode.counter if id is None else id
        TreeNode.counter += 1

    @property
    def evicted(self):
        return self.value is None
    
    def get_last_hash_value(self) -> Optional[str]:
        """Returns the hash value of the last page in this node."""
        if self.hash_value is None or len(self.hash_value) == 0:
            return None
        return self.hash_value[-1]

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _key_match_page_size1(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: List, key1: List, page_size: int):
    min_len = min(len(key0), len(key1))

    i = 0
    while i + page_size <= min_len:
        if key0[i : i + page_size] != key1[i : i + page_size]:
            break
        i += page_size
    
    rem = min_len - i
    if rem and (len(key0) == len(key1)) and (key0[i:i+rem] == key1[i:i+rem]):
        i += rem

    return i


class RadixCache():
    def __init__(
        self,
        node_id,
        device: str,
        page_size: int,
        capacity: int,
        kv_size: int,
        instance_id: Optional[int] = None,
        enable_kv_cache_events: bool = False
    ):
        self.node_id = node_id
        self.device = device
        self.page_size = page_size
        # capacity of radix cache is handled outside
        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue = []

        self.total_requested_tokens = 0
        self.total_hit_tokens = 0

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = lambda key: key[0]
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=page_size)
            self.get_child_key_fn = lambda key: tuple(key[:page_size])
        self.reset()
        
        self.capacity = capacity
        self.kv_size = kv_size
        self.kv_stored = 0 # Size of inference kv caches from xPU

        #TODO(hmchoi): Currently saving unfulled chunk is not supported. TBA
        self.save_unfull_chunk = False
        self._lock = threading.RLock()

        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)

    ##### CPU Memory Size related API #####
    def total_memory_usage(self):
        # Add (Size of inference KV cache) + (Size of Prefix Cache(Radix Tree)) 
        return self.kv_stored + (self.total_size() * self.kv_size)
    
    def allocate(self, size):
        total_usage = self.total_memory_usage()
        if total_usage + size > self.capacity:
            raise RuntimeError(
                f"[RadixCache] [node_id={self.node_id}] CPU: tried to load {size / MB_TO_BYTE:.2f}MB "
                f"but only {total_usage / MB_TO_BYTE:.2f}MB is available."
            )
        before = total_usage
        self.kv_stored += size
        self.logger.info(
            "CPU: used: %.2fMB load: %.2fMB after: %.2fMB",
            before / MB_TO_BYTE,
            size / MB_TO_BYTE,
            self.total_memory_usage() / MB_TO_BYTE,
        )
        
    
    def free(self, size):
        total_usage = self.total_memory_usage()
        if total_usage - size < 0:
            raise RuntimeError(
                f"[RadixCache] [node_id={self.node_id}] CPU: tried to free {size / MB_TO_BYTE:.2f}MB "
                f"but only {total_usage / MB_TO_BYTE:.2f}MB is used."
            )
        before = total_usage
        self.kv_stored -= size
        self.logger.info(
            "CPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
            before / MB_TO_BYTE,
            size / MB_TO_BYTE,
            self.total_memory_usage() / MB_TO_BYTE,
        )
    
    def is_avail(self, size):
        if self.capacity - self.total_memory_usage() >= size:
            return True
        else:
            return False
    
    def need_size(self, size):
        needed = (size - (self.capacity - self.total_memory_usage()))
        if needed > 0:
            return needed
        else:
            return 0

    def avail_size(self):
        return max(self.capacity - self.total_memory_usage(), 0)
        
    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self._record_all_cleared_event()

    def match_prefix(self, key: List[int], **kwargs) -> MatchResult:
        """Find the matching prefix from the radix tree.
        Args:
            key: A list of token IDs to find a matching prefix.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """
        with self._lock:
            if len(key) == 0:
                return MatchResult(
                    last_device_node=self.root_node,
                    hit_length=0,
                )

            hit_length, last_node = self._match_prefix_helper(self.root_node, key)
            
            if self.page_size != 1:
                hit_length = hit_length // self.page_size * self.page_size

            return MatchResult(
                last_device_node=last_node,
                hit_length=hit_length,
            )

    def insert(self, key: List, value=None):
        return self._insert_helper(self.root_node, key)

    def cache_finished_req(self, req):
        """Cache request when it finishes."""
        with self._lock:
            # I don't know why we should exclude last one token...? Because is it EOS token? Need Discussion
            token_ids = (req.input_hash_ids + req.output_hash_ids)[:-1] 
            # token_ids = (req.input_hash_ids + req.output_hash_ids) 

            if self.page_size != 1 and not self.save_unfull_chunk:
                page_aligned_len = len(token_ids) // self.page_size * self.page_size
            else:
                page_aligned_len = len(token_ids)

            # Radix Cache takes one ref in memory pool
            new_prefix_len = self.insert(token_ids[:page_aligned_len])

    def cache_unfinished_req(self, req, update=True):
        """Cache request when it is unfinished."""
        with self._lock:
            token_ids = (req.input_hash_ids + req.output_hash_ids)[:req.input]
            
            if self.page_size != 1 and not self.save_unfull_chunk:
                page_aligned_len = len(token_ids) // self.page_size * self.page_size
            else:
                page_aligned_len = len(token_ids)
                    
            insert_token_ids = token_ids[:page_aligned_len]

            # Radix Cache takes one ref in memory pool
            new_prefix_len = self.insert(insert_token_ids)

            if req.is_init and update:
                self.total_requested_tokens += len(token_ids)
                if self.device == 'NPU':
                    self.total_hit_tokens += req.npu_cache_hit
                elif self.device == 'CPU' or self.device == 'CXL':
                    self.total_hit_tokens += (req.storage_cache_hit - req.npu_cache_hit) if (req.storage_cache_hit - req.npu_cache_hit) > 0 else 0
                else:
                    raise RuntimeError(f"[RadixCache] [node_id={self.node_id}]: Unknown device type {self.device} for prefix caching")

            # The prefix indices could be updated, reuse it
            result = self.match_prefix(insert_token_ids)
            new_last_node = result.last_device_node

            return new_last_node

    def print_prefix_info(self):
        # print(f"# of Total requested tokens {self.total_requested_tokens}, # of Hit tokens {self.total_hit_tokens}")
        if self.total_requested_tokens != 0:
            print(f", Prefix Cache Hit ratio {(self.total_hit_tokens / self.total_requested_tokens)*100:.2f} %, ({self.total_hit_tokens} / {self.total_requested_tokens})", end='')

    def return_prefix_info(self):
        return self.total_requested_tokens, self.total_hit_tokens

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def evict(self, num_tokens: int):

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            num_evicted += len(x.key)
            self._delete_leaf(x)
            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

            self._record_remove_event(x)

    def inc_lock_ref(self, node: TreeNode):
        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                k = len(node.key)
                self.evictable_size_ -= k
                self.protected_size_ += k
                delta -= k
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                k = len(node.key)
                self.evictable_size_ += k
                self.protected_size_ -= k
                delta += k
            node.lock_ref -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        # evictable size refers to the size of the cache that is not locked and can be evicted
        return self.evictable_size_ # token count not Byte size

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_ # token count not Byte size

    def total_size(self):
        # total size refers to the size of the entire cache, including both evictable and protected cache
        return self._total_size_helper() # token count not Byte size

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            cur = stack.pop()
            total_size += len(cur.key)
            for child in cur.children.values():
                stack.append(child)
        return total_size
    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: TreeNode, key: List) -> tuple[int, TreeNode]:
        node.last_access_time = time.monotonic()
        
        if len(key) == 0:
            return 0, node

        child_key = self.get_child_key_fn(key)
        matched = 0
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                matched += prefix_len
                new_node = self._split_node(child.key, child, prefix_len)
                node = new_node
                break
            else:
                matched += prefix_len
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)

        return matched, node

    def _split_node(self, key, child: TreeNode, split_len: int):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]

        child.parent = new_node
        child.key = child.key[split_len:]
        
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        return new_node

    def _insert_helper(self, node: TreeNode, key: List):
        node.last_access_time = time.monotonic()
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = time.monotonic()
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            self._record_store_event(new_node)
        return total_prefix_length
    
    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]

        def gather_chain(node):
            cur = node
            toks = []
            if getattr(cur, "key", None):
                toks.extend(cur.key)
            while len(cur.children) == 1:
                child = next(iter(cur.children.values()))
                if getattr(child, "key", None):
                    toks.extend(child.key)
                cur = child
        
            return cur, toks
        
        def fmt_tokens(toks):
            return "[" + ", ".join(str(t) for t in toks) + "]"

        while stack:
            node, ind = stack.pop()

            tail, toks = gather_chain(node)
            length = len(toks)
            r = getattr(tail, "lock_ref", 0)

            print(" " * ind + f"{length} {fmt_tokens(toks)} r={r}")

            children = list(tail.children.values())
            for child in reversed(children):
                stack.append((child, ind + 2))

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _collect_leaves(self):
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list

    def _record_store_event(self, node: TreeNode):
        # One BlockStored per ``page_size`` chunk.
        if self.enable_kv_cache_events:
            # First chunk links to the last page of the parent node (if any).
            if node.parent is None or node != self.root_node:
                parent_block_hash = None
            else:
                last_page_start = (
                    (len(node.parent.key) - 1) // self.page_size
                ) * self.page_size
                parent_parent_tokens = node.parent.key[last_page_start:]
                parent_block_hash = hash(tuple(parent_parent_tokens))

            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key[start : start + self.page_size]
                if not page_tokens:
                    continue

                block_hash = hash(tuple(page_tokens))
                self.kv_event_queue.append(
                    BlockStored(
                        block_hashes=[block_hash],
                        parent_block_hash=parent_block_hash,
                        token_ids=page_tokens,
                        block_size=len(page_tokens),
                        lora_id=None,
                    )
                )

                # Chain next chunk to this one.
                parent_block_hash = block_hash

    def _record_remove_event(self, node: TreeNode):
        # One BlockRemoved per chunk.
        if self.enable_kv_cache_events:
            for start in range(0, len(node.key), self.page_size):
                page_tokens = node.key[start : start + self.page_size]
                if not page_tokens:
                    continue
                block_hash = hash(tuple(page_tokens))
                self.kv_event_queue.append(BlockRemoved(block_hashes=[block_hash]))

    def _record_all_cleared_event(self):
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events


if __name__ == "__main__":
    tree = RadixCache(page_size=1)

    tree.insert([1,2,3])
    tree.insert([1,2,3])
    tree.insert([1,2,99,100])
    tree.pretty_print()
    # tree.insert("Hello_world! Happy")
    # tree.insert("I love you!")
    # tree.pretty_print()

    # print(tree.match_prefix("I love you! aha"))

    # def evict_callback(x):
    #    print("evict", x)
    #    return len(x)

    # tree.evict(5, evict_callback)
    # tree.evict(10, evict_callback)
    # tree.pretty_print()