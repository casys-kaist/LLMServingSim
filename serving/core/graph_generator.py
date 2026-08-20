import os
import subprocess
from time import time
from .request import *
from .logger import get_logger
from .run_paths import input_path

logger = get_logger("GraphGenerator")

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


def generate_graph(batch, hardware, num_npus, node_id=0, instance_id=0, npu_offset=0, enable_local_offloading=False, event=False, workload_name=None, inputs_root=None, cleanup_trace=True, in_process=True):

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

    if cleanup_trace:
        try:
            os.remove(trace_path)
        except FileNotFoundError:
            pass
    return
