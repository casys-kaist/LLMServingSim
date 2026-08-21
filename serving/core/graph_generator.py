import glob
import hashlib
import os
import subprocess
from collections import OrderedDict
from time import time
from .request import *
from .logger import get_logger
from .run_paths import input_path
from .trace_generator import write_trace

logger = get_logger("GraphGenerator")

# ----------------------------------------------------------------------
# Content-addressed cache of converted graphs.
#
# A DP group whose members are unevenly loaded spends half its waves on
# dummy batches: an idle member emits a 1-token placeholder so the round's
# ALLTOALL still has a partner, and _pad_batch_to_max then inflates it to
# the group's max, so its trace is the same shape and cost as a real one.
# On the swe-bench MoE DP+EP example that is 4,405 of 8,810 batches -- and
# only 22 of those 4,405 traces are distinct, with one accounting for
# 4,382. Converting that same graph 4,382 times cost ~12 s of a 63 s run.
#
# Keyed on the trace bytes plus every other input the converter reads
# (num_npus, npu_offset, local_offloading -- its whole CLI surface besides
# the paths), so a hit is byte-identical to a miss by construction. The
# PREFILL path writes two files per rank and both land next to each other,
# so the cache stores whatever `llm.*.et` a conversion produced rather
# than assuming a count.
#
# Held as bytes rather than paths because each DP wave needs its own copy:
# ASTRA-Sim is handed one folder per wave and every member reads its own
# `llm.<npu>.et` out of it.
# ----------------------------------------------------------------------
_ET_CACHE = OrderedDict()          # key -> [(basename, bytes), ...]
_ET_CACHE_BYTES = 0
_ET_CACHE_MAX_BYTES = 64 * 1024 * 1024
_ET_CACHE_STATS = {"hit": 0, "miss": 0, "skipped": 0}

# Traces seen exactly once. A graph is only worth holding after it repeats:
# most *real* batches produce a unique trace (3,482 distinct out of 4,405 on
# the swe-bench MoE DP+EP example), and caching those on first sight filled
# 64 MB with entries that were never read again, evicting the dummy-wave
# graph that actually repeats. Storing on second sight costs one extra
# conversion per distinct trace and keeps the cache to what earns its keep.
_ET_SEEN = OrderedDict()
_ET_SEEN_MAX = 200_000


def graph_cache_stats():
    """Hit/miss counts for the converted-graph cache."""
    return dict(_ET_CACHE_STATS, entries=len(_ET_CACHE), bytes=_ET_CACHE_BYTES)


def _rows_digest(trace):
    """Digest a synthesized trace without formatting it.

    Keyed on the same content the formatter would emit -- the header line and
    every field of every row, in order -- but joined with separators instead
    of padded into columns, which is far cheaper than the real format and
    just as discriminating. Cryptographic rather than ``hash()`` because a
    collision here would silently hand back the wrong graph.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(trace.header_line.encode())
    h.update(b"\n")
    h.update("\n".join("\t".join(row) for row in trace.rows).encode())
    return h.hexdigest()


def _et_names(workload_dir):
    return set(glob.glob(os.path.join(workload_dir, "llm.*.et")))


def _cache_store(key, paths):
    """Read the freshly converted files into the cache, evicting LRU.

    Only caches a trace that has been seen before; see _ET_SEEN.
    """
    global _ET_CACHE_BYTES
    if key not in _ET_SEEN:
        _ET_SEEN[key] = None
        while len(_ET_SEEN) > _ET_SEEN_MAX:
            _ET_SEEN.popitem(last=False)
        return
    entry = []
    total = 0
    for path in sorted(paths):
        with open(path, "rb") as f:
            blob = f.read()
        entry.append((os.path.basename(path), blob))
        total += len(blob)
    if not entry or total > _ET_CACHE_MAX_BYTES:
        _ET_CACHE_STATS["skipped"] += 1
        return
    _ET_CACHE[key] = entry
    _ET_CACHE_BYTES += total
    while _ET_CACHE_BYTES > _ET_CACHE_MAX_BYTES and len(_ET_CACHE) > 1:
        _, evicted = _ET_CACHE.popitem(last=False)
        _ET_CACHE_BYTES -= sum(len(b) for _, b in evicted)

# Chakra's LLMConverter, imported once and reused for the whole run.
#
# This used to be `python -m chakra.src.converter.converter LLM ...` in a
# fresh subprocess per batch. Measured on the sim container, that cost
# ~56 ms per call, of which ~52 ms was interpreter startup plus the
# protobuf import and ~1.3 ms was the actual conversion — which made
# graph generation 73-85% of simulator wall-clock across every config
# profiled (1 NPU, 8 NPUs, and MoE DP+EP alike).
#
# Calling the converter in-process is safe because it keeps *all* of its
# mutable state on the instance (`next_node_id`, `next_comm_tag`,
# `comm_tag_dict`) — there is no module-level state to leak between
# batches, so a fresh LLMConverter per batch is equivalent to a fresh
# process. Verified byte-identical over 146 `.et` files.
#
# The import resolves to the *installed* chakra in site-packages, exactly
# as the subprocess did (there is no `chakra/` package inside the chakra
# repo root, so cwd never mattered). Editing the tree still requires
# `pip3 install .` to take effect — see AGENTS.md.
_LLMConverter = None


def _get_llm_converter():
    """Import LLMConverter on first use and cache the class.

    Deferred rather than imported at module scope so a broken or
    not-yet-installed chakra fails at the same point in the run as it did
    with the subprocess, instead of at simulator import time.
    """
    global _LLMConverter
    if _LLMConverter is None:
        from chakra.src.converter.llm_converter import LLMConverter
        _LLMConverter = LLMConverter
    return _LLMConverter


def generate_graph(batch, hardware, num_npus, node_id=0, instance_id=0, npu_offset=0, enable_local_offloading=False, event=False, workload_name=None, inputs_root=None, cleanup_trace=True, in_process=True, reuse_graphs=True, trace=None):

    cwd = os.getcwd()
    chakra = os.path.join(cwd, "extern/graph_frontend/chakra")
    if inputs_root is None:
        inputs_root = os.path.join(cwd, "inputs")

    if event:
        file_name = 'event_handler'
    else:
        file_name = f'{hardware}/{batch.model}/instance{instance_id}_batch{batch.batch_id}'

    # For DP groups, all instances write .et files to a shared workload folder
    output_name = workload_name if workload_name else file_name

    trace_path = input_path(inputs_root, "trace", f"{file_name}.txt")
    output_path = input_path(inputs_root, "workload", output_name, "llm")
    workload_dir = os.path.dirname(output_path)
    os.makedirs(workload_dir, exist_ok=True)

    cache_key = None
    if reuse_graphs:
        if trace is not None:
            digest = _rows_digest(trace)
        else:
            # The event-handler trace is written straight to disk by
            # generate_event, so there are no rows to hash.
            with open(trace_path, "rb") as f:
                digest = hashlib.blake2b(f.read(), digest_size=16).hexdigest()
        cache_key = (digest, num_npus, npu_offset, enable_local_offloading)
        cached = _ET_CACHE.get(cache_key)
        if cached is not None:
            _ET_CACHE.move_to_end(cache_key)
            _ET_CACHE_STATS["hit"] += 1
            logger.debug("Graph cache hit for %s", trace_path,
                         extra={"node_id": node_id, "instance_id": instance_id})
            for name, blob in cached:
                with open(os.path.join(workload_dir, name), "wb") as g:
                    g.write(blob)
            # A hit never needs the text: nothing is going to parse it. Write
            # it only when the caller is keeping the intermediate artifacts
            # for inspection.
            if trace is not None:
                if not cleanup_trace:
                    write_trace(trace)
            elif cleanup_trace:
                try:
                    os.remove(trace_path)
                except FileNotFoundError:
                    pass
            return
        _ET_CACHE_STATS["miss"] += 1

    # The converter reads the trace as text, so it has to exist from here on.
    if trace is not None:
        write_trace(trace)

    before = _et_names(workload_dir) if cache_key is not None else None

    if in_process:
        logger.debug("Converting graph in-process: %s -> %s", trace_path, output_path,
                     extra={"node_id": node_id, "instance_id": instance_id})
        converter = _get_llm_converter()(
            trace_path, output_path, num_npus, npu_offset, enable_local_offloading,
        )
        converter.convert()
    else:
        cmd = [
            'python', '-m', 'chakra.src.converter.converter', 'LLM',
            '--input', trace_path,
            '--output', output_path,
            '--num-npus', str(num_npus),
            '--npu-offset', str(npu_offset),
        ]

        if enable_local_offloading:
            cmd.append('--local-offloading')

        logger.debug("Generating graph with command: %s", " ".join(cmd), extra={"node_id": node_id, "instance_id": instance_id})

        subprocess.run(cmd, cwd=chakra, text=True, check=True)

    if cache_key is not None:
        _cache_store(cache_key, _et_names(workload_dir) - before)

    if cleanup_trace:
        try:
            os.remove(trace_path)
        except FileNotFoundError:
            pass
    return
