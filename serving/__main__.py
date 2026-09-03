"""Simulation entry point: ``python -m serving --cluster-config <...> [...]``.

Parses CLI args, generates ASTRA-Sim input files via ``serving.core.config_builder``,
spawns the ASTRA-Sim subprocess, and runs the iteration loop:
``router.route -> scheduler.schedule -> trace_generator -> graph -> ASTRA-Sim
-> scheduler.add_done`` until every request completes.
"""

import os
import subprocess
import argparse
import json
import shutil
from time import time
from collections import defaultdict, deque

from serving.core.scheduler import *
from serving.core.request import *
from serving.core.utils import *
from serving.core.utils import (config_weight_dtype, config_kv_cache_dtype,
                                num_mtp_layers, get_architecture)
from serving.core.spec_decode import AcceptanceModel, published_defaults
from serving.core.controller import *
from serving.core.memory_model import *
from serving.core.graph_generator import *
from serving.core.trace_generator import *
from serving.core.pim_model import *
from serving.core.config_builder import *
from serving.core.router import *
from serving.core.power_model import *
from serving.core.logger import *
from serving.core.run_paths import build_run_paths, resolve_run_id
import sys as flush

from pyinstrument import Profiler


def _pad_batch_to_max(batch, max_len):
    """Pad a batch up to ``max_len`` for DP-sync.

    Mirrors vLLM's CUDA-graph DP padding: every DP rank's forward runs at
    ``max(num_tokens_across_dp)``. We bump the high-level counters so
    dense layers, lm_head, and the MoE compute path all reflect the
    padded shape — but we deliberately leave ``decode_k_list`` /
    prefill lists untouched so attention continues to see only the real
    decodes. FlashAttention's varlen kernel gives padded ``seq_len=0``
    entries zero compute in real vLLM, and extending ``decode_k_list``
    with ``kv=1`` dummies would instead collapse ``kv_decode_mean``
    toward 1 and push the attention lookup far outside the profiled
    sweep.

    The MoE AG/RS comm size is derived from the padded length in the
    iteration loop, summed over the group -- vLLM gathers every rank's
    own tokens, so the gathered buffer is the sum and not the max.

    Request-completion accounting (`scheduler.add_done`) reads
    ``batch.requests`` and ``batch.end``, not these mutated token-list
    fields, so it is unaffected.
    """
    pad = max_len - batch.total_len
    if pad <= 0:
        return
    batch.total_len = max_len
    batch.kv_len += pad                  # each dummy contributes kv=1
    batch.num_decode += pad              # counted for lm_head / dense shape


def _pass_response(router, current, state_changed=False):
    """The "pass" answer, carrying the next known arrival when there is one.

    ASTRA-Sim stops re-asking an NPU that passed until either some NPU
    reports an iteration the frontend has not processed yet, or this
    deadline is reached. Those are the only two things that can change what
    ``schedule()`` returns, so suppressing the re-asks in between skips no
    decision. Without the deadline an idle instance would stay suppressed
    past an arrival it should have admitted, in the case where every other
    instance is still mid-batch and so no report is coming.

``state_changed=True`` sends ``pass -1``: this pass altered scheduler
    state, so it is not idempotent and re-asking is not a wasted question.
    The three DP-barrier passes do that -- joining a round with a dummy,
    joining it with a real batch, or handing a batch claim back -- and none
    of them is preceded by a report, so nothing else would lift the
    suppression. ASTRA-Sim treats it like a workload assignment: this NPU
    stays askable and every other one is re-opened too.
    """
    if state_changed:
        return "pass -1"
    nxt = router.get_next_pending_arrival()
    if nxt is None or nxt <= current:
        return "pass"
    return f"pass {int(nxt)}"


def _runtime_limit(value):
    return float('inf') if value == 0 else value


def _cluster_config_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join("..", path)


def _load_cluster_config_for_overrides(path):
    with open(_cluster_config_path(path), "r") as f:
        return json.load(f)


def _resolve_output_file(path, run_id):
    if path is None:
        return None
    return path.replace("{run_id}", run_id)


def _cleanup_inputs_root(run_paths, logger):
    """Remove generated ASTRA-Sim inputs after a completed simulation."""
    runs_root = os.path.abspath(os.path.join("inputs", "runs"))
    inputs_root = os.path.abspath(run_paths.inputs_root)
    if inputs_root in (os.path.abspath("inputs"), runs_root):
        raise RuntimeError(f"Refusing to remove broad inputs root: {inputs_root}")
    if not inputs_root.startswith(runs_root + os.sep):
        logger.warning(
            "Skipping ASTRA-Sim inputs cleanup because inputs_root is outside %s: %s",
            runs_root, inputs_root,
        )
        return
    shutil.rmtree(inputs_root, ignore_errors=True)
    logger.info("Removed ASTRA-Sim inputs root: %s", inputs_root)


def _prepare_ns3_config(astra_sim, run_paths):
    template = os.path.join(astra_sim, "extern/network_backend/ns-3/scratch/config/config.txt")
    output_dir = os.path.join(run_paths.inputs_root, "ns3", "output")
    config_path = os.path.join(run_paths.inputs_root, "ns3", "config.txt")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    replacements = {
        "FLOW_FILE": os.path.join(output_dir, "flow.txt"),
        "TRACE_FILE": os.path.join(output_dir, "trace.txt"),
        "TRACE_OUTPUT_FILE": os.path.join(output_dir, "mix.tr"),
        "FCT_OUTPUT_FILE": os.path.join(output_dir, "fct.txt"),
        "PFC_OUTPUT_FILE": os.path.join(output_dir, "pfc.txt"),
        "QLEN_MON_FILE": os.path.join(output_dir, "qlen.txt"),
    }

    for path in (replacements["FLOW_FILE"], replacements["TRACE_FILE"]):
        open(path, "w").close()

    with open(template, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(config_path, "w", encoding="utf-8") as f:
        for line in lines:
            parts = line.split(maxsplit=1)
            if parts and parts[0] in replacements:
                f.write(f"{parts[0]} {replacements[parts[0]]}\n")
            else:
                f.write(line)
    return config_path


def _iter_raw_instances(cluster_config):
    for node in cluster_config.get("nodes", []):
        for instance in node.get("instances", []):
            yield instance


def _resolve_instance_dtypes(instance, dtype_to_bits):
    """This instance's ``(weight_dtype, kv_cache_dtype)``, both from the model
    config.

    Neither is an input any more. A dtype is a property of the checkpoint, and
    once a model carries five of them -- weights, KV cache, mamba conv state,
    mamba recurrent state, sparse-indexer side cache -- a flag per dtype is
    both unusable and unfaithful: the checkpoint already says what it is, and
    saying otherwise describes a model nobody can serve. The three cache dtypes
    were never flags; these two used to be, and are now read the same way. See
    ``memory_model.cache_dtype_bytes`` for the whole table.

    Both rules are the profiler's and vLLM's, not ours. ``config_weight_dtype``
    is what decides which ``perf/.../<variant>/`` folder the profiler *wrote*,
    so the simulator has to derive it identically or it reads a folder that
    does not exist -- and it prefers ``quantization_config`` over the dtype
    fields, which on a quantized checkpoint describe the activations rather
    than the weights (DeepSeek-V3.2-Exp is FP8 with ``torch_dtype: bfloat16``).
    ``config_kv_cache_dtype`` follows vLLM's own promotion at
    ``attention.py:281``.
    """
    config = get_config(instance["model_name"])
    declared = config_weight_dtype(config)
    dtype = declared if declared in dtype_to_bits else "bfloat16"
    return dtype, config_kv_cache_dtype(config)


_DEFAULT_BLOCK_SIZE = 16


def _resolve_block_size(instance, args):
    """This instance's KV block size: explicit, else the profiled one, else 16.

    vLLM takes a block size as a **floor and an alignment unit**, not as the
    answer. ``platforms/interface.py`` computes
    ``alignment = max(min(backend.get_supported_kernel_block_sizes()),
    cache_config.block_size)``, derives the smallest multiple of it whose
    attention page covers one mamba page, and raises ``block_size`` to that if
    it is larger -- never lowering it. So the resolved value is a function of
    what you asked for, and on Qwen3.8-27B asking for 16 gives 784 while asking
    for 64 gives 832.

    That is why this reads the number back out of the profile bundle rather
    than recomputing it: the bundle records what the engine settled on for the
    block size the *profiler* was run with, which is the configuration the
    latencies were measured under. An explicit value that disagrees is not
    wrong to allow -- studying a hypothetical block size is a legitimate thing
    to simulate -- but it no longer matches the measurement, so it is said out
    loud.
    """
    explicit = instance.get("block_size", args.block_size)
    variant = resolve_variant(get_config(instance["model_name"]))
    # The bundle records one resolved block size per TP degree, because both
    # the mamba page and the attention page scale with the rank's shard.
    tp = int(instance.get("tp_size") or 1)
    profiled = profiled_block_size(
        instance["hardware"], instance["model_name"], variant, tp)
    if explicit is None:
        return profiled or _DEFAULT_BLOCK_SIZE
    if profiled is not None and explicit != profiled:
        get_logger("main").warning(
            "--block-size %d for %s, but the profile bundle was measured at %d -- "
            "the value vLLM raised it to, so that one attention page covers one "
            "mamba page. Latency lookups will use measurements taken at a "
            "different block size.",
            explicit, instance["model_name"], profiled,
        )
    return explicit


def _resolve_mem_util(instance, cli_default):
    """Per-instance NPU memory utilization, from ``npu_mem.mem_util``.

    Lives inside ``npu_mem`` because its only job is to scale ``mem_size``, and
    it follows that block's ``mem_*`` naming. Falls back to the CLI default.
    """
    util = instance.get("npu_mem", {}).get("mem_util", cli_default)
    try:
        util = float(util)
    except (TypeError, ValueError):
        raise ValueError(
            f"npu_mem.mem_util for instance {instance.get('instance_id')} must be a "
            f"number in (0, 1]; got {util!r}"
        ) from None
    if not 0 < util <= 1:
        raise ValueError(
            f"npu_mem.mem_util for instance {instance.get('instance_id')} must be in "
            f"(0, 1]; got {util}. It is a fraction of npu_mem.mem_size, so 0.9 rather "
            f"than 90"
        )
    return util


def _build_instance_runtime_configs(instances, args, dtype_to_bits):
    runtime_configs = []
    for instance_id, instance in enumerate(instances):
        dtype, kv_cache_dtype = _resolve_instance_dtypes(instance, dtype_to_bits)

        enable_attn_offloading = instance.get("enable_attn_offloading", args.enable_attn_offloading)
        enable_sub_batch_interleaving = instance.get(
            "enable_sub_batch_interleaving", args.enable_sub_batch_interleaving)
        if enable_sub_batch_interleaving and not enable_attn_offloading:
            raise RuntimeError(
                f"Instance {instance_id} enables sub-batch interleaving without attention offloading")
        if enable_sub_batch_interleaving and instance.get("pp_size", 1) > 1:
            raise RuntimeError(
                f"Instance {instance_id} enables sub-batch interleaving with pp_size "
                f"{instance['pp_size']}: an interleaved trace leaves both sub-batches "
                f"mid-block at every group edge, so a pipeline stage has no single "
                f"hidden state to pass on")

        runtime_configs.append({
            "max_num_seqs": _runtime_limit(instance.get("max_num_seqs", args.max_num_seqs)),
            "max_num_batched_tokens": _runtime_limit(
                instance.get("max_num_batched_tokens", args.max_num_batched_tokens)),
            "long_prefill_token_threshold": instance.get(
                "long_prefill_token_threshold", args.long_prefill_token_threshold),
            "block_size": _resolve_block_size(instance, args),
            "dtype": dtype,
            "fp": dtype_to_bits[dtype],
            "kv_cache_dtype": kv_cache_dtype,
            "enable_chunked_prefill": instance.get(
                "enable_chunked_prefill", args.enable_chunked_prefill),
            "enable_prefix_caching": instance.get(
                "enable_prefix_caching", args.enable_prefix_caching),
            "npu_memory_utilization": _resolve_mem_util(
                instance, args.npu_memory_utilization),
            "reserve_full_isl": instance.get("reserve_full_isl", args.reserve_full_isl),
            "enable_local_offloading": instance.get(
                "enable_local_offloading", args.enable_local_offloading),
            "enable_attn_offloading": enable_attn_offloading,
            "enable_sub_batch_interleaving": enable_sub_batch_interleaving,
            "enable_block_copy": instance.get("enable_block_copy", args.enable_block_copy),
            "num_speculative_tokens": instance.get(
                "num_speculative_tokens", args.num_speculative_tokens),
            "spec_acceptance_rate": instance.get(
                "spec_acceptance_rate", args.spec_acceptance_rate),
            "spec_acceptance_policy": instance.get(
                "spec_acceptance_policy", args.spec_acceptance_policy),
        })
    return runtime_configs


def _require_drafter_cost(model_name, n, node_id, instance_id):
    """Refuse a speculative run whose drafter time cannot be charged.

    Acceptance is only half of speculative decoding. The other half is what the
    drafts cost to produce, and vLLM runs the drafter **N times per step** --
    once before the loop and ``num_speculative_tokens - 1`` inside it
    (``llm_base_proposer.py``). The first pass reuses the target's own token
    layout; the loop pins ``max_query_len = 1``, so the rest are pure decode.
    A model that drafts with itself runs an MTP module for each: two norms, an
    ``eh_proj``, **one full decoder layer of its own family**, then (DeepSeek
    and GLM) a norm and ``lm_head``. The decoder layer is the dominant term by
    two orders of magnitude over the wrapper, and charging zero for any of it
    would report a speedup no engine can deliver.

    So a model with MTP modules needs an ``mtp:`` section in its architecture
    catalog naming both halves -- the wrapper layers and which block the
    drafter's decoder layer is -- and the wrapper has to be written from a live
    profile dump like every other block, since the module tree and the profile
    tree differ both ways and writing one from vLLM's source binds names that
    measure nothing. Until that profiling happens the honest answer is to
    refuse, exactly as ``calculate_sizes`` refuses a layer name it has no
    formula for.

    A model with **no** MTP modules drafts with a separate model or with
    n-gram. That is a serving choice rather than a property of the checkpoint,
    so it is warned about rather than refused: the simulator has no second
    model to charge.
    """
    logger = get_logger("main", node_id=node_id, instance_id=instance_id)
    mtp = num_mtp_layers(get_config(model_name))
    if not mtp:
        logger.warning(
            "Speculative decoding with N=%d on %s, which declares no MTP "
            "modules -- it drafts with a separate model or with n-gram, and "
            "the simulator has no second model to charge. Draft *time* is not "
            "counted; acceptance still is, so the reported speedup is an upper "
            "bound.",
            n, model_name,
        )
        return
    section = (get_architecture(model_name) or {}).get("mtp") or {}
    missing = [
        key for key in ("prologue", "decoder_block") if not section.get(key)
    ]
    if missing:
        raise NotImplementedError(
            f"speculative decoding on {model_name!r} needs the cost of its "
            f"{mtp} MTP module(s), which run {n} time(s) per step, and "
            f"profiler/models/<model_type>.yaml's 'mtp:' section is missing "
            f"{', '.join(missing)}. Charging zero would report a speedup no "
            f"engine can deliver -- and 'decoder_block' is the dominant term, "
            f"since one drafter pass wraps a whole decoder layer. Add them "
            f"from a live profile dump (python -m profiler coverage "
            f"--profile-mtp) and profile the model with --profile-mtp, or "
            f"drop --num-speculative-tokens."
        )


def _build_acceptance_model(model_name, inst_cfg, node_id, instance_id):
    """The instance's speculative-decoding acceptance model, or None.

    Defaults come from the model's own published measurement
    (``configs/spec_decode.json``), which is why a model with no published
    figure has to be given a rate rather than being handed a plausible one:
    acceptance varies from 0.39 to 0.78 across the four modern families, so
    there is no defensible generic default.
    """
    n = inst_cfg["num_speculative_tokens"]
    rate = inst_cfg["spec_acceptance_rate"]
    if not n and rate is None:
        return None

    published = published_defaults(model_name) or {}
    if n in (None, 0, -1):
        n = published.get("num_speculative_tokens", 0)
    if rate is None:
        rate = published.get("acceptance_rate")
    if not n:
        return None
    _require_drafter_cost(model_name, n, node_id, instance_id)
    if rate is None:
        raise ValueError(
            f"speculative decoding requested for {model_name!r}, which has no "
            f"published acceptance rate in configs/spec_decode.json. Pass "
            f"--spec-acceptance-rate (accepted/drafted) rather than letting the "
            f"simulator invent one."
        )
    model = AcceptanceModel(
        num_speculative_tokens=n,
        acceptance_rate=rate,
        position_acceptance=published.get("position_acceptance"),
        policy=inst_cfg["spec_acceptance_policy"],
        node_id=node_id,
        instance_id=instance_id,
    )
    model.logger.info(
        "Speculative decoding: N=%d, acceptance %.3f (%s), mean accept length %.2f%s",
        model.N, model.rate, model.policy, model.mean_accept_length(),
        "" if inst_cfg["spec_acceptance_rate"] is not None else " [published]",
    )
    return model


def main():
    # ----------------------------------------------------------------------------------------------
    # LLMServingSim runs in astra-sim directory for easy path configuration
    # your relative path should start from astra-sim directory
    cwd = os.getcwd()
    astra_sim = os.path.join(cwd, "astra-sim")
    os.chdir(astra_sim)

    # -------------------------------------- Argument parsing --------------------------------------
    parser = argparse.ArgumentParser(prog='python -m serving',
                                     description='LLMServingSim') 
    
    parser.add_argument('--cluster-config', type=str, default='configs/cluster/single_node_single_instance.json',
                        help='path to cluster config JSON defining node topology, instance layout, hardware, and memory hierarchy')
    parser.add_argument('--max-num-seqs', type=int, default=128,
                        help='maximum number of sequences in a batch (0 = unlimited)')
    parser.add_argument('--max-num-batched-tokens', type=int, default=2048,
                        help='maximum number of tokens processed per iteration across all requests (the total token budget). '
                        'With chunked prefill, long inputs are split across iterations; '
                        'without chunked prefill, this effectively caps max input length')
    parser.add_argument('--long-prefill-token-threshold', type=int, default=0,
                        help='per-request token cap per step for chunked prefill (0 = disabled). '
                        'Limits how many tokens a single prefill request consumes per iteration, '
                        'preventing long prompts from monopolizing the token budget. '
                        'When 0, a single prefill can consume the entire budget')
    parser.add_argument('--num-speculative-tokens', type=int, default=0,
                        dest='num_speculative_tokens',
                        help='speculative decoding draft length N (vLLM\'s num_speculative_tokens). '
                        '0 (default) disables it. Omit --spec-acceptance-rate to take the model\'s '
                        'own published N and acceptance from configs/spec_decode.json; pass -1 to '
                        'use that N explicitly. Override per instance with "num_speculative_tokens"')
    parser.add_argument('--spec-acceptance-rate', type=float, default=None,
                        dest='spec_acceptance_rate',
                        help='fraction of drafted tokens the target model accepts, so the mean '
                        'accepted length is 1 + rate * N. This is the marginal rate every '
                        'published source reports, not Leviathan\'s conditional alpha -- see '
                        'serving/core/spec_decode.py. Defaults to the model\'s published value; '
                        'a model with no published value must be given one')
    parser.add_argument('--spec-acceptance-policy', type=str,
                        choices=['FIXED', 'DECAY', 'CUSTOM'], default='FIXED',
                        dest='spec_acceptance_policy',
                        help='how the accepted count is drawn: FIXED (default, every draft '
                        'position at the pooled rate), DECAY (per-position rates, which fall with '
                        'draft position -- same mean, different spread), CUSTOM (user-defined)')
    parser.add_argument('--request-routing-policy', type=str, choices=['LOAD', 'RR', 'RAND', 'CUSTOM'], default='LOAD',
                        help='request routing policy across instances: LOAD (vLLM-style weighted least-loaded, default), '
                        'RR (round-robin), RAND (random), CUSTOM (user-defined)')
    parser.add_argument('--expert-routing-policy', type=str,
                        choices=['BALANCED', 'RR', 'RAND', 'CUSTOM'],
                        default='BALANCED',
                        help='expert token routing policy for MoE models: '
                        'BALANCED (default; analytical pigeonhole approximation of '
                        'a trained load-balanced learned gate), '
                        'RR (round-robin), RAND (uniform random per token), '
                        'CUSTOM (user-defined)')
    parser.add_argument('--enable-block-copy', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Replay one transformer block\'s trace across every '
                        'layer instead of re-computing the routing per layer — '
                        'cuts trace-generation time roughly num_hidden_layers× '
                        'on MoE models. Safe with BALANCED (deterministic); '
                        'RR/RAND get a small per-layer variance averaged out. '
                        'Disable only for CUSTOM policies that need faithful '
                        'per-layer variance.')
    parser.add_argument('--enable-prefix-caching', action=argparse.BooleanOptionalAction, default=True,
                        help='enable prefix caching to reuse KV cache blocks across requests '
                        'with shared prefixes (default: enabled). Use --no-enable-prefix-caching to disable')
    parser.add_argument('--enable-chunked-prefill', action=argparse.BooleanOptionalAction, default=True,
                        help='enable chunked prefill to split long prefill requests across multiple iterations, '
                        'matching vLLM v1 behavior (default: enabled). Use --no-enable-chunked-prefill to disable')
    parser.add_argument('--enable-prefix-sharing', action='store_true', default=False,
                        help='enable second-tier prefix cache pooling across instances within a node')
    parser.add_argument('--prefix-storage', type=str, choices=['None', 'CPU', 'CXL'], default='None',
                        help='storage medium for the second-tier prefix cache pool: None (NPU only), CPU, or CXL')
    parser.add_argument('--enable-local-offloading', action='store_true', default=False,
                        help='enable weight offloading to local (NPU) memory. '
                        'Recommended to disable unless weight memory access is not counted in profiling')
    parser.add_argument('--enable-attn-offloading', action='store_true', default=False,
                        help='enable attention computation offloading to PIM (Processing-In-Memory) devices')
    parser.add_argument('--enable-sub-batch-interleaving', action='store_true', default=False,
                        help='enable sub-batch interleaving to overlap XPU and PIM computation. '
                        'Requires --enable-attn-offloading')
    parser.add_argument('--reserve-full-isl', action=argparse.BooleanOptionalAction, default=True,
                        help='admit a request only if its whole sequence fits in the KV cache, '
                        'not merely its first chunk. Mirrors vLLM\'s scheduler_reserve_full_isl '
                        '(True there too); without it chunked prefill over-admits and thrashes '
                        'the KV cache. Override per instance with "reserve_full_isl"')
    parser.add_argument('--npu-memory-utilization', type=float, default=0.9,
                        help='fraction of NPU memory an instance may use for weights plus '
                        "KV cache. Corresponds to vLLM's --gpu-memory-utilization, renamed "
                        'because every other memory surface here is NPU-terminology; '
                        'override per instance with "npu_mem": {"mem_util": ...}. KV capacity is '
                        '(npu_mem * this - model weight); the activation peak and CUDA '
                        'context that vLLM also subtracts are not modelled, so the '
                        'resulting capacity is an upper bound on vLLM\'s at the same value')
    parser.add_argument('--block-size', type=int, default=None,
                        help='KV cache block size in tokens. When omitted, taken from the '
                        'profile bundle\'s recorded engine_resolved.block_size -- vLLM derives '
                        'this from the backend rather than accepting what it is given, so a '
                        'MiniMax-M3 run is 128 and a Qwen3.8 hybrid is 784 whatever you ask for. '
                        'An explicit value that disagrees is simulating a configuration vLLM '
                        'cannot serve, and says so. Falls back to 16 when the bundle predates '
                        'the field. Override per instance with "block_size"')
    parser.add_argument('--dataset', type=str, default=None,
                        help='path to .jsonl dataset file with request traces. '
                        'If None, requests must be added manually in serving/__main__.py')
    parser.add_argument('--output', type=str, default=None,
                        help='path for per-request CSV output with latency metrics (TTFT, TPOT, ITL). '
                        'If None, results are printed to stdout only. Supports {run_id} placeholder')
    parser.add_argument('--run-id', type=str, default=None,
                        help='unique id for this simulation run. Intermediate ASTRA-Sim inputs are written under '
                        'astra-sim/inputs/runs/<run-id>. If omitted, a process-unique id is generated')
    parser.add_argument('--inputs-root', type=str, default=None,
                        help='override the root directory for generated ASTRA-Sim inputs. Defaults to '
                        'astra-sim/inputs/runs/<run-id>')
    parser.add_argument('--save-trace-text', action=argparse.BooleanOptionalAction, default=False,
                        help='write each batch\'s trace as text, for inspection (default: '
                        'disabled). Nothing in the pipeline reads it -- the Chakra converter takes '
                        'the trace rows directly -- so it is produced only on request, and it is '
                        'the only human-readable form of what the simulator emitted. Implies '
                        '--keep-inputs, since the text is written into the run directory. Can '
                        'leave gigabytes behind on a long run')
    parser.add_argument('--keep-inputs', action=argparse.BooleanOptionalAction, default=False,
                        help='keep the generated ASTRA-Sim inputs under '
                        'astra-sim/inputs/runs/<run-id> after a successful simulation (default: '
                        'disabled). Preserves the Chakra .et workloads and the generated network, '
                        'system and memory configs, so a run can be replayed through ASTRA-Sim by '
                        'hand. Replaces --cleanup-inputs, whose polarity was inverted')
    parser.add_argument('--skip-prefill', action='store_true', default=False,
                        help='skip the prefill phase, running decode only')
    parser.add_argument('--num-reqs', type=int, default=0,
                        help='number of entries (requests or sessions) to load from the dataset. '
                        'For agentic datasets, each entry is a session with multiple sub-requests. '
                        '0 = load all entries')
    parser.add_argument('--log-interval', type=float, default=1.0,
                        help='interval in seconds between throughput/memory usage log messages')
    parser.add_argument('--log-level', type=str, choices=['WARNING', 'INFO', 'DEBUG'], default='WARNING',
                        help='logging verbosity: WARNING (minimal), INFO (per-iteration details), DEBUG (per-layer memory)')
    parser.add_argument('--network-backend', type=str, choices=['analytical', 'ns3'], default='analytical',
                        help='network simulation backend: analytical (fast, default) or ns3 (detailed, WIP)')

    args = parser.parse_args()
    
    args.run_id = resolve_run_id(args.run_id)
    run_paths = build_run_paths(astra_sim, args.run_id, args.inputs_root)
    args.inputs_root = run_paths.inputs_root
    args.output = _resolve_output_file(args.output, args.run_id)

    configure_logger(level=args.log_level)
    logger = get_logger("Main")
    print_banner()
    print_input_config(args=args)
    flush.stdout.flush()
    
    _dtype_to_bits = {'float16': 16, 'bfloat16': 16, 'float32': 32, 'fp8': 8, 'int8': 8}
    request_routing_policy=args.request_routing_policy
    expert_routing_policy=args.expert_routing_policy
    enable_prefix_sharing=args.enable_prefix_sharing
    prefix_storage=args.prefix_storage
    dataset=args.dataset
    output_file=args.output
    is_init = not args.skip_prefill
    num_req=args.num_reqs
    log_interval=args.log_interval
    network_backend = args.network_backend
    raw_cluster_config = _load_cluster_config_for_overrides(args.cluster_config)
    raw_instances = list(_iter_raw_instances(raw_cluster_config))
    build_enable_local_offloading = args.enable_local_offloading or any(
        inst.get("enable_local_offloading", False) for inst in raw_instances)
    build_enable_attn_offloading = args.enable_attn_offloading or any(
        inst.get("enable_attn_offloading", False) for inst in raw_instances)
    # ---------------------------------- Extract cluster config -----------------------------------
    cluster = build_cluster_config(
        astra_sim, args.cluster_config, build_enable_local_offloading, build_enable_attn_offloading,
        inputs_root=run_paths.inputs_root)
    num_nodes = cluster["num_nodes"]
    num_instances = cluster["num_instances"]
    instances = cluster["instances"]
    inst2node_mapping = cluster["inst2node_mapping"]
    inst2npu_mapping = cluster["inst2npu_mapping"]
    npu2inst_mapping = cluster["npu2inst_mapping"]
    prefill_instance = cluster["prefill_instance"]
    decode_instance = cluster["decode_instance"]
    start_npu_ids = cluster["start_npu_ids"]
    end_npu_ids = cluster["end_npu_ids"]
    placement = cluster["placement"]
    block_mode_on = cluster["block_mode_on"]
    total_npu = cluster["total_npu"]
    cpu_mem_size = cluster["cpu_mem_size"]
    power_modeling = cluster["power_modeling"]
    power_configs = cluster["power_configs"]
    pim_models = cluster["pim_models"]
    instance_runtime_configs = _build_instance_runtime_configs(instances, args, _dtype_to_bits)
    any_prefix_caching = any(cfg["enable_prefix_caching"] for cfg in instance_runtime_configs)
    # ----------------------------------------- Set config -----------------------------------------
    # Automatic network, memory configuration
    # If you want to set more specific information such as latency, look at config.py and each json file
    if network_backend == 'analytical':
        network=run_paths.network_config
        binary=os.path.join(astra_sim, "build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra")
    elif network_backend == 'ns3':
        network=_prepare_ns3_config(astra_sim, run_paths)
        binary=os.path.join(astra_sim, "extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default")
    else:
        raise NotImplementedError("Only analytical and ns3 network backend are supported")
    memory=run_paths.memory_config
    system=run_paths.system_config
    # ------------------------------------- Prepare simulation -------------------------------------
    # Need to extract each instance's memory accessability 
    node2inst_mapping = defaultdict(list)
    for inst_id, node_id in inst2node_mapping.items():
        node2inst_mapping[node_id].append(inst_id)
    node2inst_mapping = dict(node2inst_mapping)

    prefix_pool_inst_mapping = {}
    for i in range(num_instances):
        prefix_pool_inst_mapping[i] = None

    pool_device = None

    if prefix_storage == "CPU":
        pool_device = Device.CPU
    elif prefix_storage == "CXL":
        pool_device = Device.CXL

    if any_prefix_caching and enable_prefix_sharing and prefix_storage != 'None':
        num_prefix_pool = num_nodes
        # make prefix pool objects based on num_prefix_pool
        prefix_pools = []

        def _pool_kv_bytes_per_token(inst_ids):
            """KV bytes per token for a shared pool."""
            kv_shapes = {
                (
                    instances[i]["model_name"],
                    instance_runtime_configs[i]["fp"],
                    instance_runtime_configs[i]["kv_cache_dtype"],
                )
                for i in inst_ids
            }
            if len(kv_shapes) > 1:
                raise RuntimeError(
                    "Shared prefix pool requires instances to share model, "
                    f"dtype, and kv_cache_dtype; got {kv_shapes}"
                )
            model = instances[inst_ids[0]]['model_name']
            cfg = instance_runtime_configs[inst_ids[0]]
            return full_cluster_kv_bytes_per_token(model, cfg["fp"], cfg["kv_cache_dtype"])

        def _pool_block_size(inst_ids):
            sizes = {instance_runtime_configs[i]["block_size"] for i in inst_ids}
            if len(sizes) > 1:
                raise RuntimeError(
                    f"Shared prefix pool requires instances to share block_size; got {sizes}")
            return sizes.pop()

        if prefix_storage == 'CPU':
            for i in range(num_prefix_pool):
                if cpu_mem_size[i] <= 0:
                    raise RuntimeError(f"Memory size for prefix storage type {prefix_storage} is invalid")
                inst_ids = node2inst_mapping[i]
                prefix_pools.append(build_prefix_pool(
                    pool_device, cpu_mem_size[i] * GB_TO_BYTE,
                    _pool_block_size(inst_ids), _pool_kv_bytes_per_token(inst_ids),
                    node_id=i))
            # This means one node shares one prefix pool
            prefix_pool_inst_mapping = inst2node_mapping

        elif prefix_storage == 'CXL':
            if cluster["cxl_mem_size"] <= 0:
                raise RuntimeError(f"Memory size for prefix storage type {prefix_storage} is invalid")
            inst_ids = list(range(num_instances))
            prefix_pools.append(build_prefix_pool(
                pool_device, cluster["cxl_mem_size"] * GB_TO_BYTE,
                _pool_block_size(inst_ids), _pool_kv_bytes_per_token(inst_ids)))
            # This means every instance shares the same universal prefix pool (maybe fixed later)
            prefix_pool_inst_mapping = [0 for _ in range(num_instances)]
        else:
            raise NotImplementedError(f"Prefix storage type {prefix_storage} is not supported or memory size is invalid")

    schedulers = []
    for instance_id, instance in enumerate(instances):
        prefix_pool_index = prefix_pool_inst_mapping[instance_id]
        prefix_pool = None
        if prefix_pool_index != None:
            prefix_pool = prefix_pools[prefix_pool_index]
        cxl_mem = 0
        if cluster["cxl_mem_size"] > 0:
            cxl_mem = cluster["cxl_mem_size"]        
        
        # Make scheduler for each instance

        inst_cfg = instance_runtime_configs[instance_id]

        schedulers.append(Scheduler(
            instance["model_name"], instance["node_id"], instance_id,
            inst_cfg["max_num_seqs"], inst_cfg["max_num_batched_tokens"],
            instance["num_npus"], instance["tp_size"], instance["pp_size"],
            instance["npu_mem"]["mem_size"], cpu_mem_size[instance["node_id"]],
            inst2npu_mapping[instance_id], instance["pd_type"],
            inst_cfg["fp"], inst_cfg["block_size"], num_req,
            inst_cfg["enable_prefix_caching"],
            enable_prefix_sharing, prefix_pool, pool_device, inst_cfg["enable_chunked_prefill"],
            inst_cfg["long_prefill_token_threshold"],
            cxl_mem,
            ep_size=instance.get("ep_total", 1),
            kv_cache_dtype=inst_cfg["kv_cache_dtype"],
            npu_memory_utilization=inst_cfg["npu_memory_utilization"],
            reserve_full_isl=inst_cfg["reserve_full_isl"],
            acceptance_model=_build_acceptance_model(
                instance["model_name"], inst_cfg, instance["node_id"], instance_id),
        ))

    # The derived KV capacity, not the utilization fraction, is what decides
    # memory pressure. It is per instance and only known once the schedulers
    # exist, so it gets its own section rather than a row in the input-config
    # block, which is printed before any of this is resolved.
    print_heading("KV Cache Initialization")
    print_markup("")
    # Pad only as far as the widest label, so the line stays inside the rule.
    pad = max(len(f"Instance [{i}]") for i in range(len(schedulers)))
    for inst_id, sched in enumerate(schedulers):
        pool = sched.memory.npu_pool
        label = f"Instance \\[{inst_id}]"
        print_markup(
            f"  \u2022 [cyan]{label:<{pad + 1}}[/cyan] : "
            f"{pool.num_blocks * pool.block_size} tokens / {pool.num_blocks} blocks "
            f"({pool.num_blocks * pool.bytes_per_block / GB_TO_BYTE:.2f} GiB/rank "
            f"at util {sched.memory.npu_memory_utilization:.2f})"
        )
    print_rule()

    # Controller for astra-sim process communication
    controller = Controller(total_npu)
    # Global Request Router
    router = Router(num_instances, schedulers, num_req, request_routing_policy)
    # Power Modeling if enabled
    if power_modeling:
        power_model = PowerModel(power_configs)
    else:
        power_model = None
    # Load requests into router (routed in real-time during simulation)
    if dataset != None:
        router.load_requests(dataset, enable_prefix_caching=any_prefix_caching, is_init=is_init)
    else:
        # Manually adding request (legacy: route all upfront)
        for i in range(16):
            for sched in schedulers:
                sched.add_request([i, sched.model, 64, 128, 0, i % num_instances])

    # Simulator start
    current = 0 # current tick of the system
    sys = 0 # current system id (NPU id)
    id = 0 # id of the request
    is_prefill_done = False # flag to check if prefill is done
    done_instance = [] # list of done instances
    done_inst_npus = [[] for _ in range(num_instances)]
    start_time = time()
    last_end_time = [0 for _ in range(num_instances)]
    last_calc_time = [0 for _ in range(num_instances)]
    waiting_request = [False for _ in range(num_instances)]

    # Calculating Simulator's Throughput
    throughput = []
    prompt_th = 0    # Avg Prompt Throguhput per Sec
    gen_th = 0       # Avg Generation Throughput per Sec
    last_log = 0    # last logged time
    FREQ = 1000_000_000 # 1 GHz (1e9 Hz)
    INTERVAL = log_interval*FREQ
    # Per-interval token counts -> tokens/s, so 1/log_interval. Floor division
    # collapsed to 0 for any --log-interval > 1, which zeroed every logged
    # throughput and made the summary line divide by zero.
    RATIO = FREQ/INTERVAL
    total_prompt = 0
    total_gen = 0
    total_latency = 0
    req_cnt = 0

    # Set Event Handler that loop with INTERVAL time until first request arrive (for all instances)
    first_arival_time = router.get_first_arrival_time()
    if INTERVAL > first_arival_time:
        event_time = first_arival_time
    else:
        event_time = INTERVAL
    event_trace = generate_event(int(event_time), inputs_root=run_paths.inputs_root)
    # Make Chakra Grapth
    generate_graph(None, None, total_npu, event=True, inputs_root=run_paths.inputs_root,
                   save_trace_text=args.save_trace_text, trace=event_trace)
    # set first workload file
    workload = get_workload(None, None, event=True, inputs_root=run_paths.inputs_root)
    # run subprocess
    astra_args = [binary, "--workload-configuration="+workload, "--system-configuration="+system, "--network-configuration="+network, "--memory-configuration="+memory]
    if start_npu_ids != "":
        astra_args.append("--start-npu-ids="+start_npu_ids)
    if end_npu_ids != "":
        astra_args.append("--end-npu-ids="+end_npu_ids)
    if network_backend == 'ns3':
        astra_args.append("--logical-topology-configuration="+astra_sim+"/inputs/logical_topology/logical_8nodes_1D.json")
    p = subprocess.Popen(astra_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    # DP group synchronization: defer trace generation until all members have scheduled
    # dp_groups maps dp_group_name -> list of instance_ids
    dp_groups = {}
    for inst in instances:
        dg = inst.get("dp_group")
        if dg is not None:
            dp_groups.setdefault(dg, []).append(inst["instance_id"])
    # Reverse lookup: instance_id -> dp_group_name
    inst_dp_group = {}
    for dg, members in dp_groups.items():
        for inst_id in members:
            inst_dp_group[inst_id] = dg
    # Batches waiting at the DP barrier: per group, per member, a FIFO. With
    # pp_size > 1 a member can have up to pp_size batches awaiting their round,
    # and vLLM pairs the members' forwards in order -- rank A's j-th forward
    # joins the same collective as rank B's j-th. A single slot per member
    # silently dropped the earlier batch, which then never got a workload_name,
    # so the instance's other NPUs retried joining it forever.
    dp_pending = {dg: defaultdict(deque) for dg in dp_groups}  # dg -> {inst: deque[(batch, node_id)]}
    # Workloads pre-generated by a DP round, keyed by the NPU that opened that
    # member's batch, i.e. the one that owes ASTRA-Sim its graph. A FIFO because
    # with pp_size > 1 an NPU can open a second round before it has been handed
    # the first one's graph, and a single slot silently dropped the first.
    dp_ready_workloads = defaultdict(deque)  # npu_id -> deque[workload_path]

    # ----------------------------------- Start simulation loop ------------------------------------
    print_markup("[sim.heading]▶ Starting simulation...[/]\n")
    flush.stdout.flush()

    # Starting simulation, one while loop processes one iteration
    while True:
        
        out = controller.read_wait(p)
        out_dict = controller.parse_output(out[-2])
        
        if out_dict != None:
            sys = out_dict['sys']
            id = out_dict['id']
            current = out_dict['cycle']

        # Route newly arrived requests to instances based on current load
        if dataset is not None:
            router.route_arrived_requests(current)

        instance_id = npu2inst_mapping[sys]  # get instance id from NPU id
        node_id = inst2node_mapping[instance_id] # get node id from instance id

        # add stanby energy consumption for power modeling
        if power_modeling and sys == inst2npu_mapping[instance_id] and waiting_request[instance_id]:
            power_model.add_npu_standby_energy_consumption(instances[instance_id]["hardware"], node_id, current,
                        last_end_time[instance_id], last_calc_time[instance_id], num_npus=instances[instance_id]["num_npus"])
            last_calc_time[instance_id] = current

        # mark latest end time of the first NPU in the instance
        # An instance can span multiple NPUs. Only update end-time when sys is the first NPU of the instance.
        # waiting_request[instance_id] = True means the instance has no batch to run (idle).
        if sys == inst2npu_mapping[instance_id] and not waiting_request[instance_id]:
            last_end_time[instance_id] = current
            waiting_request[instance_id] = True

        # check request is done
        prompt_t, gen_t, finished_reqs = schedulers[instance_id].add_done(id, sys, current)
        # add tokens in throughput
        prompt_th += prompt_t
        total_prompt += prompt_t
        gen_th += gen_t
        total_gen += gen_t
        # count only finished requests
        req_cnt += len(finished_reqs) if instances[instance_id]["pd_type"] != "prefill" else 0

        # Notify router of completed requests for dependency chain release
        if instances[instance_id]["pd_type"] != "prefill":
            for req in finished_reqs:
                router.notify_request_completed(req.id, current)

        # Add prefill ended requests to decode instance
        if instances[instance_id]["pd_type"] == "prefill" and len(finished_reqs) > 0:
            router.transfer_prefill_request(finished_reqs)

        # An NPU that opened a DP round owes ASTRA-Sim that round's graph, and it
        # has to be handed over before the scheduler may open anything new. vLLM
        # schedules and dispatches in one step (``schedule()`` then
        # ``execute_model()`` inside ``step_with_batch_queue``), so a scheduled
        # batch is never left un-dispatched. A DP batch has to break that up --
        # its graph cannot be emitted until every member has joined the barrier
        # and the padded ``max_total_len`` is known -- so the invariant to keep is
        # that the dispatch still lands before the next schedule for this NPU.
        # Without it, at pp_size > 1 the NPU built its next microbatch on the very
        # poll that should have handed over the previous one, and the round after
        # that overwrote the entry: the first graph never ran, and the other
        # pipeline stage blocked forever on a RECV that never came.
        pending = dp_ready_workloads.get(sys)
        new_req = None if pending else schedulers[instance_id].schedule(current, sys, id)
        responded = False  # track whether we already sent a response to ASTRA-Sim

        # Hand over a workload pre-generated by a DP round this NPU opened.
        if pending:
            controller.write_flush(p, pending.popleft())
            if not pending:
                del dp_ready_workloads[sys]
            responded = True
        # DP group: truly idle instance (no inflight batch) — create dummy batch so ALLTOALL syncs
        # An idle DP member has to keep pace with a busy one. vLLM requires every
        # rank of a DP group to run the same number of forwards, and with PP a
        # rank has pp_size microbatches in flight at once -- so the gate here is
        # schedule()'s own pipeline-depth rule, not "nothing in flight". Gating on
        # == 0 lets the member holding the real request run ahead by up to
        # pp_size batches, and the dp_pending barrier then waits for a round the
        # idle members can never join.
        #
        # Any NPU of the instance may open the round, not just its first. Which
        # NPU ASTRA-Sim asks about is not ours to choose, and an instance whose
        # start NPU is busy or starved would otherwise never contribute a dummy
        # -- the barrier then waits on a member that cannot answer. An empty
        # ``dp_pending[dg][instance_id]`` keeps it to one dummy per member per
        # round -- the member has nothing queued, so it has not yet joined the
        # round being assembled -- which is what the start-NPU test used to be
        # standing in for.
        elif (new_req is None and instance_id in inst_dp_group
              and not dp_pending[inst_dp_group[instance_id]][instance_id]
              and len(schedulers[instance_id].inflight) < schedulers[instance_id].pp_size):
            dg = inst_dp_group[instance_id]
            if any(dp_pending[dg][i] for i in dp_groups[dg]):
                # Emit a 1-token dummy; the uniform pad-to-max pass below
                # brings it (and any undersized real peers) up to the
                # group's max_total_len, matching vLLM's CUDA-graph DP padding.
                logger.debug(f"Instance {instance_id} is idle but DP group {dg} has pending batches. Creating dummy batch for synchronization.")
                dummy = Batch(schedulers[instance_id].get_batch_id(), instances[instance_id]["model_name"],
                              1, 1, [1], [], 0, 1, [], [], [1], current, 0)
                dummy.fired.append(sys)
                # Register it the way scheduler._build_batch registers a real
                # batch. Without this the instance's other NPUs get nothing:
                # schedule() routes them to _schedule_existing, which searches
                # inflight, finds no dummy, and they fall through to "pass" --
                # so their .et never runs and the group's EP collective blocks
                # forever. Invisible at tp=pp=1, where the start NPU is the only
                # NPU an instance owns.
                schedulers[instance_id].inflight.append(dummy)
                dp_pending[dg][instance_id].append((dummy, inst2node_mapping[instance_id]))

                if all(dp_pending[dg][i] for i in dp_groups[dg]):
                    # Every DP member has a batch queued — take one from each,
                    # oldest first, and pad them to the group's max (vLLM
                    # CUDA-graph DP padding) before generating.
                    round_batches = {i: dp_pending[dg][i].popleft() for i in dp_groups[dg]}
                    own_workload = None
                    config = get_config(instances[instance_id]["model_name"])
                    max_total_len = max(b.total_len for b, _ in round_batches.values())
                    for b, _ in round_batches.values():
                        _pad_batch_to_max(b, max_total_len)
                    # The gathered size, which is what both EP collectives are
                    # sized from. vLLM's ``AgRsAll2AllManager.dispatch_router_logits``
                    # all-gathers ``[hidden_states, router_logits]`` with
                    # ``sizes[rank] == hidden_states.shape[0]``, so every rank
                    # contributes *all* of its own tokens and the result is the
                    # concatenation over the group; ``combine`` reduce-scatters a
                    # buffer of that same gathered size. The trace generator turns
                    # this into ASTRA-Sim's two conventions by dividing by
                    # ``ep_total`` for the AllGather's per-rank chunk and passing it
                    # whole for the ReduceScatter's pre-scatter total -- both of
                    # which are right only if this really is the sum.
                    #
                    # It used to be set to ``max_total_len``, i.e. ep_total times
                    # too small, on the grounds that ASTRA-Sim's Ring model is "2x
                    # over real AG/RS". It is not: ``Ring.cc`` gives AllGather
                    # ``(N-1) x chunk``, ReduceScatter ``(N-1) x total/N`` and
                    # AllReduce ``2(N-1) x total/N`` per rank, which is exactly
                    # NCCL's ring algorithm for all three. What that halving really
                    # compensated for was the MoE *compute* being 1.33x over from
                    # the sub-top_k clamp; with the compute measured per EP slice,
                    # the halving is a 2x under-count of the collective and shows up
                    # as -23% TTFT on the dp+ep bench example while the TP-only one
                    # (which never takes this path) sits at +1%.
                    sum_total_len = max_total_len * len(dp_groups[dg])

                    # Shared workload folder for all DP members
                    first_inst_id = dp_groups[dg][0]
                    first_batch = round_batches[first_inst_id][0]
                    dp_workload_name = f'{instances[first_inst_id]["hardware"]}/{instances[first_inst_id]["model_name"]}/dp_{dg}_batch{first_batch.batch_id}'

                    for inst_id in dp_groups[dg]:
                        batch, nid = round_batches[inst_id]
                        batch.workload_name = dp_workload_name
                        inst = instances[inst_id]
                        inst_cfg = instance_runtime_configs[inst_id]
                        trace_data = generate_trace(batch, inst["hardware"], inst["tp_size"], inst["pp_size"],
                                       inst["local_ep"], inst["ep_total"], inst["pd_type"],
                                       nid, inst_id,
                                       inst_cfg["max_num_batched_tokens"], inst_cfg["max_num_seqs"],
                                       placement[inst_id], block_mode_on[inst_id],
                                       expert_routing_policy, inst_cfg["enable_prefix_caching"],
                                       inst_cfg["enable_attn_offloading"],
                                       power_model, pim_models[nid],
                                       inst_cfg["enable_sub_batch_interleaving"], inst_cfg["fp"],
                                       dtype=inst_cfg["dtype"], kv_cache_dtype=inst_cfg["kv_cache_dtype"],
                                       tp_dim=inst.get("tp_dim"), ep_dim=inst.get("ep_dim"),
                                       dp_sum_total_len=sum_total_len,
                                       enable_block_copy=inst_cfg["enable_block_copy"],
                                       inputs_root=run_paths.inputs_root,
                                   num_speculative_tokens=(
                                       schedulers[instance_id].spec.N
                                       if schedulers[instance_id].spec else 0))
                        generate_graph(batch, inst["hardware"], inst["num_npus"], nid,
                                       inst_id, inst2npu_mapping[inst_id],
                                       inst_cfg["enable_local_offloading"],
                                       workload_name=dp_workload_name,
                                       inputs_root=run_paths.inputs_root,
                                       save_trace_text=args.save_trace_text,
                                       trace=trace_data)
                        # ``fired[0]`` is the NPU that opened this member's
                        # round -- the one that owes ASTRA-Sim its graph. That is
                        # normally this very poll, and then it is answered
                        # directly. With pp_size > 1 the round can instead pop a
                        # batch that another NPU of the instance opened, or an
                        # older one this NPU opened, so queue it for that NPU.
                        ready = get_workload(batch, inst["hardware"], inst_id,
                                             workload_name=dp_workload_name,
                                             inputs_root=run_paths.inputs_root)
                        if batch.fired[0] == sys:
                            own_workload = ready
                        else:
                            dp_ready_workloads[batch.fired[0]].append(ready)

                    if own_workload is not None:
                        controller.write_flush(p, own_workload)
                    else:
                        controller.write_flush(p, _pass_response(router, current, state_changed=True))
                    responded = True
                else:
                    # Joined the round with a dummy; the round is not complete.
                    controller.write_flush(p, _pass_response(router, current, state_changed=True))
                    responded = True
        # runnable batch exists
        elif new_req is not None:
            # ``_build_batch`` returns a batch fired only by the NPU that built
            # it, so a longer ``fired`` means this poll joined a batch through
            # ``_schedule_existing``. With DP groups any NPU of an instance may
            # open a round (the idle-member dummy in particular), so the start
            # NPU can arrive here holding a batch it did not build -- it has to
            # be served like any other joiner, not registered into the round a
            # second time.
            built_here = len(new_req.fired) == 1  # implies sys is the start NPU
            if built_here:  # first NPU of the instance, opening a new batch
                waiting_request[instance_id] = False
                instance = instances[instance_id]
                dg = inst_dp_group.get(instance_id)

                if dg is not None:
                    # DP group: defer trace generation until all members scheduled
                    dp_pending[dg][instance_id].append((new_req, node_id))

                    if all(dp_pending[dg][i] for i in dp_groups[dg]):
                        # Every DP member has a batch queued — take one from
                        # each, oldest first, and pad them to the group's max
                        # (vLLM CUDA-graph DP padding) so smaller batches gain
                        # dummy decodes that all layers still compute over.
                        round_batches = {i: dp_pending[dg][i].popleft() for i in dp_groups[dg]}
                        own_workload = None
                        config = get_config(instance["model_name"])
                        max_total_len = max(b.total_len for b, _ in round_batches.values())
                        for b, _ in round_batches.values():
                            _pad_batch_to_max(b, max_total_len)
                        # See the twin block above for why this is the sum.
                        sum_total_len = max_total_len * len(dp_groups[dg])

                        # Shared workload folder for all DP members
                        first_inst_id = dp_groups[dg][0]
                        first_batch = round_batches[first_inst_id][0]
                        dp_workload_name = f'{instances[first_inst_id]["hardware"]}/{instances[first_inst_id]["model_name"]}/dp_{dg}_batch{first_batch.batch_id}'

                        for inst_id in dp_groups[dg]:
                            batch, nid = round_batches[inst_id]
                            batch.workload_name = dp_workload_name
                            inst = instances[inst_id]
                            inst_cfg = instance_runtime_configs[inst_id]
                            trace_data = generate_trace(batch, inst["hardware"], inst["tp_size"], inst["pp_size"],
                                           inst["local_ep"], inst["ep_total"], inst["pd_type"],
                                           nid, inst_id,
                                           inst_cfg["max_num_batched_tokens"], inst_cfg["max_num_seqs"],
                                           placement[inst_id], block_mode_on[inst_id],
                                           expert_routing_policy, inst_cfg["enable_prefix_caching"],
                                           inst_cfg["enable_attn_offloading"],
                                           power_model, pim_models[nid],
                                           inst_cfg["enable_sub_batch_interleaving"], inst_cfg["fp"],
                                           dtype=inst_cfg["dtype"], kv_cache_dtype=inst_cfg["kv_cache_dtype"],
                                           tp_dim=inst.get("tp_dim"), ep_dim=inst.get("ep_dim"),
                                           dp_sum_total_len=sum_total_len,
                                           enable_block_copy=inst_cfg["enable_block_copy"],
                                           inputs_root=run_paths.inputs_root,
                                   num_speculative_tokens=(
                                       schedulers[instance_id].spec.N
                                       if schedulers[instance_id].spec else 0))
                            generate_graph(batch, inst["hardware"], inst["num_npus"], nid,
                                           inst_id, inst2npu_mapping[inst_id],
                                           inst_cfg["enable_local_offloading"],
                                           workload_name=dp_workload_name,
                                           inputs_root=run_paths.inputs_root,
                                           save_trace_text=args.save_trace_text,
                                           trace=trace_data)
                            # See the twin block above: the NPU that opened a
                            # member's round owes its graph, and that is normally
                            # this poll.
                            ready = get_workload(batch, inst["hardware"], inst_id,
                                                 workload_name=dp_workload_name,
                                                 inputs_root=run_paths.inputs_root)
                            if batch.fired[0] == sys:
                                own_workload = ready
                            else:
                                dp_ready_workloads[batch.fired[0]].append(ready)

                        if own_workload is not None:
                            controller.write_flush(p, own_workload)
                        else:
                            controller.write_flush(p, _pass_response(router, current, state_changed=True))
                        responded = True
                    else:
                        # Waiting for other DP members — send pass
                        controller.write_flush(p, _pass_response(router, current, state_changed=True))
                        responded = True
                else:
                    # Independent instance: generate trace immediately
                    inst_cfg = instance_runtime_configs[instance_id]
                    trace_data = generate_trace(new_req, instance["hardware"], instance["tp_size"], instance["pp_size"],
                                   instance["local_ep"], instance["ep_total"],
                                   instance["pd_type"],
                                   node_id, instance_id,
                                   inst_cfg["max_num_batched_tokens"], inst_cfg["max_num_seqs"],
                                   placement[instance_id], block_mode_on[instance_id],
                                   expert_routing_policy, inst_cfg["enable_prefix_caching"],
                                   inst_cfg["enable_attn_offloading"], power_model, pim_models[node_id],
                                   inst_cfg["enable_sub_batch_interleaving"], inst_cfg["fp"],
                                   dtype=inst_cfg["dtype"], kv_cache_dtype=inst_cfg["kv_cache_dtype"],
                                   tp_dim=instance["tp_dim"], ep_dim=instance["ep_dim"],
                                   enable_block_copy=inst_cfg["enable_block_copy"],
                                   inputs_root=run_paths.inputs_root,
                                   num_speculative_tokens=(
                                       schedulers[instance_id].spec.N
                                       if schedulers[instance_id].spec else 0))
                    generate_graph(new_req, instance["hardware"], instance["num_npus"], node_id,
                                   instance_id, inst2npu_mapping[instance_id],
                                   inst_cfg["enable_local_offloading"],
                                   inputs_root=run_paths.inputs_root,
                                   save_trace_text=args.save_trace_text,
                                   trace=trace_data)
                    workload = get_workload(new_req, instance["hardware"], instance_id,
                                            inputs_root=run_paths.inputs_root)
                    controller.write_flush(p, workload)
            else:
                # Joined an existing batch: pick up its workload. workload_name
                # matters for a DP batch, whose graph lives in the group's shared
                # folder -- deriving the default instance<id>_batch<id> path here
                # points at a directory that was never written, and ASTRA-Sim
                # stalls on the missing .et instead of failing.
                #
                # A DP batch is in ``inflight`` from the moment its own instance
                # schedules it, but it is only stamped with the shared folder
                # when the *last* member of the group joins the barrier. In that
                # window ``_schedule_existing`` will hand it to this NPU with no
                # name yet, so wait instead of guessing a path: hand the claim
                # back and pass, and the batch is re-offered on a later poll once
                # the round is assembled. The batch is necessarily queued in
                # ``dp_pending`` already (it exists because an NPU of this
                # instance opened it), so passing here cannot stall the barrier.
                if sys == inst2npu_mapping[instance_id]:
                    waiting_request[instance_id] = False
                if instance_id in inst_dp_group and new_req.workload_name is None:
                    new_req.fired.remove(sys)
                    controller.write_flush(p, _pass_response(router, current, state_changed=True))
                    responded = True
                else:
                    workload = get_workload(new_req, instances[instance_id]["hardware"], instance_id,
                                            workload_name=new_req.workload_name,
                                            inputs_root=run_paths.inputs_root)
                    controller.write_flush(p, workload)

        # check time to store throughput (only print on start NPU to avoid transient states)
        if current > last_log + INTERVAL and sys == inst2npu_mapping[instance_id]:
            # store the prompt
            throughput.append((prompt_th*RATIO, gen_th*RATIO))
            last_log += INTERVAL
            log_time_str = f"[{last_log / FREQ:.1f}s]"
            log_time_len = len(log_time_str)
            log_indent = ' ' * log_time_len + '  '
            tree_indent = '├─'
            # Heartbeat timestamp stays in the terminal's default
            # colour — bright enough to scan, not so dim that it
            # disappears. (The per-log-record [HH:MM:SS.mmm] stays
            # dim via sim.time because it appears every other line.)
            print_markup(
                f"{log_time_str} "
                f"[blue]Avg prompt throughput: {prompt_th * RATIO:.1f} tokens/s,[/] "
                f"[blue]Avg generation throughput: {gen_th * RATIO:.1f} tokens/s[/]"
            )
            prompt_th = 0
            gen_th = 0

            ######### Per Instance Metrics #########

            for inst_id in range(num_instances):
                # len(running), not the size of the in-flight batch: the persistent
                # running set is the exact analogue of vLLM's num_running_reqs, which
                # is what bench compares this column against. The batch is only the
                # subset that fit in this step's token budget.
                running_reqs = len(schedulers[inst_id].running)
                waiting_reqs = len([req for req in schedulers[inst_id].waiting if req.arrival <= current])

                mem = schedulers[inst_id].memory
                npu_used_mb = mem.npu_used / MB_TO_BYTE
                npu_util = (mem.npu_used / mem.npu_mem * 100.0) if mem.npu_mem else 0.0

                line = (
                    f"{log_indent+tree_indent}Running Instance\\[{inst_id}]: "
                    f"{running_reqs} reqs, Waiting: {waiting_reqs} reqs, "
                    f"Total # {schedulers[inst_id].num_npus} NPUs, "
                    f"Each NPU Memory Usage {npu_used_mb:.2f} MB "
                    f"({npu_util:.3f} % Used)"
                )
                if schedulers[inst_id].enable_prefix_caching:
                    line += schedulers[inst_id].memory.format_prefix_info()
                print_markup(line)

            ######### Per Node Metrics #########
            if node2inst_mapping:
                num_nodes = len(node2inst_mapping)
                for i, (node_id, inst_ids) in enumerate(node2inst_mapping.items()):
                    node_cpu_usage = 0
                    inst_usage = []
                    if any_prefix_caching and enable_prefix_sharing and prefix_storage == "CPU":
                        node_cpu_usage = prefix_pools[node_id].used_bytes()
                    else:
                        for inst_id in inst_ids:
                            inst_cpu_usage = schedulers[inst_id].memory.cpu_used
                            node_cpu_usage += inst_cpu_usage
                            inst_usage.append(inst_cpu_usage)

                    cpu_util = (node_cpu_usage / (cpu_mem_size[node_id]*GB_TO_BYTE)) * 100
                    if prefix_storage != "CXL" and not power_modeling and i == num_nodes - 1:
                        tree_indent = '└─'
                    line = (
                        f"{log_indent+tree_indent}Node\\[{node_id}]: "
                        f"Total CPU Memory Usage {node_cpu_usage/MB_TO_BYTE:.2f} MB, "
                        f"{cpu_util:.3f} % Used "
                    )
                    if any_prefix_caching and enable_prefix_sharing and prefix_storage == "CPU":
                        line += prefix_pools[node_id].stats.format_prefix_info()

                    if (any_prefix_caching and enable_prefix_sharing and prefix_storage == "CPU") or (len(inst_ids) == 1):
                        print_markup(line)
                    else:
                        parts = []
                        for j, inst_cpu_usage in enumerate(inst_usage):
                            inst_cpu_util = (inst_cpu_usage / node_cpu_usage)*100 if node_cpu_usage else 0
                            parts.append(f"Instance\\[{inst_ids[j]}]: {inst_cpu_util:.2f} %")
                        print_markup(line + "(" + ", ".join(parts) + ")")

            ######### Per CXL Metrics #########
            if any_prefix_caching and prefix_storage == "CXL":
                if enable_prefix_sharing:
                    num_prefix_pool = len(prefix_pools)
                    for cxl_id, cxl_pool in enumerate(prefix_pools):
                        cxl_usage = cxl_pool.used_bytes()
                        cxl_util = cxl_pool.usage()
                        if not power_modeling and cxl_id == num_prefix_pool - 1:
                            tree_indent = '└─'
                        print_markup(
                            f"{log_indent+tree_indent}CXL\\[{cxl_id}]: "
                            f"Total CXL Device Memory Usage "
                            f"{cxl_usage/MB_TO_BYTE:.2f}MB, {cxl_util:.3f} % Used"
                        )
                else:
                    enabled_inst_ids = [
                        inst_id for inst_id, sched in enumerate(schedulers)
                        if sched.enable_prefix_caching
                    ]
                    for pos, inst_id in enumerate(enabled_inst_ids):
                        second_tier = schedulers[inst_id].memory.storage_pool
                        if second_tier is None:
                            continue
                        cxl_usage = second_tier.used_bytes()
                        cxl_util = second_tier.usage()
                        if not power_modeling and pos == len(enabled_inst_ids) - 1:
                            tree_indent = '└─'
                        print_markup(
                            f"{log_indent+tree_indent}CXL\\[0]/Instance\\[{inst_id}]: "
                            f"Total CXL Device Memory Usage {cxl_usage / MB_TO_BYTE:.2f} MB, "
                            f"{cxl_util:.3f} % Used"
                        )

            ######### Power Modeling #########
            if power_modeling:
                tree_indent = '└─'
                print_markup(
                    f"{log_indent+tree_indent}"
                    f"Avg power consumption: {power_model.get_current_power(current)} W"
                )
        # check if all requests are done for current instance#
        # NOTE: 'instance_id' could occur in duplicate, because 'npu2inst_mapping[sys]' is not one-to-one mapping
        if (instance_id not in decode_instance or is_prefill_done) and instance_id not in done_instance and schedulers[instance_id].is_request_empty() and not router.has_pending_requests() and not router.has_deferred_sessions():
            # For DP groups: only mark done when ALL members of the group are empty
            dg = inst_dp_group.get(instance_id)
            if dg is not None:
                all_dp_empty = all(
                    schedulers[inst_id].is_request_empty() and len(schedulers[inst_id].inflight) == 0
                    for inst_id in dp_groups[dg]
                )
                if not all_dp_empty:
                    # Other DP members still have work — keep this instance alive for dummy waves
                    if not responded:
                        controller.write_flush(p, _pass_response(router, current))
                    flush.stdout.flush()
                    continue

            if sys not in done_inst_npus[instance_id]:
                done_inst_npus[instance_id].append(sys)
            if len(done_inst_npus[instance_id]) == (1 if instances[instance_id]["num_npus"] == 1 else 2):
                done_instance.append(instance_id)

            # check if all prefill instances are done
            if len(done_instance) == len(prefill_instance):
                is_prefill_done = True

            # check if all instances are done
            if len(done_instance) == num_instances:
                for inst_idx in range(num_instances):
                    schedulers[inst_idx].memory.free_prefix_cache()
                    schedulers[inst_idx].memory.free_weight()
                
                # check memory leak before exit
                schedulers[inst_idx].memory.is_free()

                print_rule()
                print_markup("[sim.heading]▶ Exiting simulation...[/]\n")
                controller.write_flush(p, "exit")
                break
            controller.write_flush(p, "done") # make done instances to sleep
        elif new_req == None and not responded:
            # If all instances are idle but deferred sessions have pending
            # requests with future arrival times (tool calls still running),
            # advance current time so the next iteration can pick them up.
            # Built before the jump below: _pass_response compares against
            # the clock ASTRA-Sim is actually at, not the one we skip to.
            pass_msg = _pass_response(router, current)
            if router.has_deferred_sessions() or router.has_pending_requests():
                next_arrival = router.get_next_pending_arrival()
                if next_arrival is not None and next_arrival > current:
                    current = next_arrival
            controller.write_flush(p, pass_msg)
        
        # flush
        flush.stdout.flush()

    # calculate simulation time
    end_time = time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # check all scheduled requests in astra-sim are well done
    controller.check_end(p)

    # calcuate prefix caching metrics
    total_requested_tokens = 0
    total_npu_hit_tokens = 0
    total_cpu_hit_tokens = 0
    if any_prefix_caching:
        for i in range(num_instances):
            if not schedulers[i].enable_prefix_caching:
                continue
            (temp_npu_a, temp_npu_b), (temp_cpu_a, temp_cpu_b) = schedulers[i].memory.return_prefix_info()
            if (not enable_prefix_sharing) and (prefix_storage != "None") and (temp_npu_a != temp_cpu_a):
                raise RuntimeError(f"Instance[{i}] prefix caching requested tokens mismatch between NPU ({temp_npu_a}) and CPU ({temp_cpu_a})")
            total_requested_tokens += temp_npu_a
            total_npu_hit_tokens += temp_npu_b
            if not enable_prefix_sharing:
                total_cpu_hit_tokens += temp_cpu_b
        
        if enable_prefix_sharing:
            for pool in prefix_pools:
                _, temp_cpu_b = pool.stats.return_prefix_info()
                total_cpu_hit_tokens += temp_cpu_b
    
    # This is total system's throughput
    total_latency = current/FREQ
    print_rule()
    print_markup("[sim.heading]▶ Simulation results...[/]\n")
    print_markup(f"Total simulation time: {int(hours)}h {int(minutes)}m {seconds:.3f}s")
    print_rule("[sim.tagline]Throughput Results[/]")
    print_markup(f"Total requests:                                                     {req_cnt}")
    print_markup(f"Total clocks (ns):                                                  {current}")
    print_markup(f"Total latency (s):                                                  {total_latency:.3f}")
    # total_prompt is the vLLM prompt-throughput gauge: it counts every token
    # pushed through prefill, including prefix-cache hits and anything recomputed
    # after a preemption. Report the dataset input from the requests themselves
    # rather than by subtracting the recompute counter -- a request preempted
    # again mid-recompute is charged its full remaining work each time it is
    # re-admitted, so the two are not each other's complement.
    total_recompute = sum(s.recompute_tokens for s in schedulers)
    total_preempt = sum(s.num_preemptions for s in schedulers)
    total_input = sum(req.original_input for s in schedulers for req in s.done)
    print_markup(f"Total input tokens:                                                 {total_input}")
    if total_preempt:
        print_markup(f"Preemptions:                                                        {total_preempt}")
    if total_recompute:
        print_markup(f"Recomputed prompt tokens (preemption):                               {total_recompute}")
    print_markup(f"Total generated tokens:                                             {total_gen}")
    print_markup(f"Request throughput (req/s):                                         {req_cnt/total_latency:.2f}")
    print_markup(f"Average prompt throughput (tok/s):                                  {total_prompt/total_latency:.2f}")
    print_markup(f"Average generation throughput (tok/s):                              {total_gen/total_latency:.2f}")
    print_markup(f"Total token throughput (tok/s):                                     {(total_prompt + total_gen)/total_latency:.2f}")
    print_markup(f"Throughput per {log_interval:g} sec (\\[prompt_throughput], \\[gen_throughput]): {throughput}")
    print_rule()
    if any_prefix_caching:
        print_rule("[sim.tagline]Prefix Caching Results[/]")
        print_markup(f"Total requested prompt tokens:                                      {total_requested_tokens}")
        print_markup(f"NPU prefix hit prompt tokens:                                       {total_npu_hit_tokens}")
        if total_requested_tokens > 0:
            print_markup(f"NPU prefix hit ratio (%):                                           {(total_npu_hit_tokens/total_requested_tokens)*100:.2f}")
            if prefix_storage != "None":
                print_markup(f"{prefix_storage} prefix hit prompt tokens:                                       {total_cpu_hit_tokens}")
                print_markup(f"{prefix_storage} prefix hit ratio (%):                                           {(total_cpu_hit_tokens/total_requested_tokens)*100:.2f}")
            print_markup(f"Total prefix hit ratio (%):                                         {((total_npu_hit_tokens+total_cpu_hit_tokens)/total_requested_tokens)*100:.2f}")
        else:
            print_markup("NPU prefix hit ratio (%):                                           N/A (no requests tracked)")
        print_rule()
    if power_modeling:
        print_rule("[sim.tagline]Power Modeling Results[/]")
        total_energy = power_model.get_final_energy(current)
        print_markup(f"Total energy consumption (kJ):                                      {total_energy/1000:.2f}")
        # Each node results
        power_model.print_power_summary()
        print_markup(f"Power per {log_interval:g} sec (W): {power_model.power_time_series}")
        print_rule()
    # Each instacne results
    for i in range(num_instances):
        print_rule(f"[sim.tagline]Instance \\[{i}][/]")
        schedulers[i].print_result()
        print_rule()
    
    # Important informations about metrics
    # The TTFT (Time to First Token) in our simulator differs from vllm. 
    # While vllm measures TTFT as the time when the client receives the first token,
    # Our simulator measures it as the time when the computation of the first token is completed.
    # Therefore, vllm gets much more higher TTFT.
    # (Ref: https://docs.vllm.ai/en/latest/design/metrics.html?utm_source=chatgpt.com#interval-calculations-vs-preemptions)

    if output_file != None:
        print(f"Saving each request's information to output file: {output_file}")
        for i in range(num_instances):
            schedulers[i].save_output(output_file, is_append=False if i == 0 else True)

    # --save-trace-text writes the text into the run directory, so keeping it
    # is implied: producing the text and then deleting it would be pointless.
    if not (args.keep_inputs or args.save_trace_text):
        _cleanup_inputs_root(run_paths, logger)
    

if __name__ == "__main__": 
    # For simulation time breakdown
    # profiler = Profiler()
    # profiler.start()
    main()
    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))
