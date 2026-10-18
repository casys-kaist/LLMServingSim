"""vLLM engine lifecycle management.

Responsibilities:
  * Merge profiler defaults + CLI-provided engine kwargs into a final
    argument dict for vllm.LLM().
  * Shrink the model config (via hf_overrides) for the current TP
    degree so profiling is cheap.
  * Spin an LLM up (with vLLM stdout captured), read back the runtime
    limits it ended up with, and spin it down cleanly.

Separation of concerns: everything here is pure-host orchestration
(no CUDA, no vLLM internals). The vLLM-internal integration lives in
``profiler/core/hooks/``.
"""

from __future__ import annotations

import copy
import gc
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from vllm import LLM

from profiler.core import logger as log
from profiler.core.config import (
    HOST_ENGINE_DEFAULTS,
    SHARD_FIELDS,
    ProfileArgs,
    probe_linear_attn_chunk,
    probe_moe_params,
)
from profiler.core.stack import probe_key_saturation
from profiler.core.stack import describe as describe_stack
from profiler.core.stack import ALL_AXES, minimal_layer_count_for


# ---------------------------------------------------------------------------
# Runtime limits
# ---------------------------------------------------------------------------

@dataclass
class RuntimeLimits:
    """Snapshot of what the live engine actually supports.

    Grid generators use these to cap the sweep.

    Attributes:
        max_num_batched_tokens: vLLM's scheduler-side token budget.
        max_num_seqs: max concurrent sequences.
        num_cache_tokens: total KV slots allocated, group-aware (a
            hybrid request occupies several KV cache groups at once).
        block_size: KV block size in tokens, as the engine settled on it.
            Not necessarily what we asked for: on a hybrid stack vLLM
            enlarges the attention block until an attention page costs at
            least as many bytes as a mamba state page, then pads the mamba
            page to match, so one uniform pool covers both. Measured at 784
            on Qwen3.8-27B against the 16 we requested.
        max_model_len: longest single sequence the engine accepts.
        linear_attn_chunk: chunk length the linear-attention prefill scan
            works in, or None when the model has no linear-attention layer.
            A grid-placement quantity, not an engine one: measured cost
            tracks the chunk count, so the sweep has to sample boundaries and
            the points just past them.
        num_experts / top_k: Global MoE parameters from the live HF config.
            None for non-MoE models. Native component targets resolve their
            expert placement separately without shrinking these fields.
        moe_ep: Ordinary engine's logical EP degree, always 1. Native
            components record their deployment in their own contracts.
        key_saturation: how many key tokens a query can attend to, when the
            checkpoint bounds it (``index_topk``, or M3's selected-block
            count times its block size); None on a dense model, whose queries
            read their whole causal window. The attention grid stops its key
            axis there, because past it every shot measures the same thing --
            DeepSeek-V3.2's current grid spends 15 kv points out to 163,830
            and leaves only 9 below the bound, where all the variation is.
    """

    max_num_batched_tokens: int
    max_num_seqs: int
    num_cache_tokens: int
    max_model_len: int
    block_size: int = 16
    linear_attn_chunk: int | None = None
    num_experts: int | None = None
    top_k: int | None = None
    moe_ep: int = 1
    key_saturation: int | None = None
    # Native text configuration of this engine, not the wrapper or backend.
    # Persisted for authoritative main engines so consumed overrides survive
    # deletion of the temporary local config.
    model_text_config: dict[str, Any] | None = None
    # Initialization and postload checks, not a forward-execution certificate.
    dummy_weight_preparation: dict[str, Any] | None = None
    # Read-only acquisition runtime facts; neither an eager graph mode nor
    # proof that the full serving model completed graph capture.
    sampling_runtime: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Kwarg merging
# ---------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` on top of ``base``.

    For dict-valued keys we recurse; for everything else (lists,
    scalars), override wins outright.
    """
    out = copy.deepcopy(base)
    for k, v in override.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _profile_engine_overrides(args: ProfileArgs) -> dict[str, Any]:
    """Turn CLI-provided engine fields into a dict to merge under defaults.

    Only non-None values are included so they don't stomp the defaults.
    """
    out: dict[str, Any] = {}
    if args.dtype is not None:
        out["dtype"] = args.dtype
    if args.kv_cache_dtype is not None:
        out["kv_cache_dtype"] = args.kv_cache_dtype
    if args.max_num_batched_tokens is not None:
        out["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.max_num_seqs is not None:
        out["max_num_seqs"] = args.max_num_seqs
    if args.block_size is not None:
        out["block_size"] = args.block_size
    if args.gpu_memory_utilization is not None:
        out["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_model_len is not None:
        out["max_model_len"] = args.max_model_len
    if args.hf_overrides is not None:
        out["hf_overrides"] = args.hf_overrides
    return out


def fuse_engine_kwargs(
    args: ProfileArgs, tp: int, stack_axes: tuple[str, ...] = ALL_AXES,
    *, native_profile=None,
) -> dict[str, Any]:
    """Produce the final ``**kwargs`` to pass to ``vllm.LLM()``.

    Design: profile every TP degree on a **single GPU** by keeping
    the engine's ``tensor_parallel_size=1`` and shrinking the model
    config via ``SHARD_FIELDS`` so per-rank kernel shapes match what
    a real ``tp=N`` deployment would see on one of its ranks.
    Collective timing is handled analytically by ASTRA-Sim in
    LLMServingSim, so the profiler does not need to measure ALLREDUCE.

    The model's full config lives in ``args.model_config`` (the
    ``configs/model/<path>.json`` content). The engine writes that to
    a temporary directory at spin-up and points vLLM there, so HF
    hub access is never required at profile time. This function only
    computes the ``hf_overrides`` payload that vLLM applies on top of
    the written config:

        HOST_ENGINE_DEFAULTS["hf_overrides"]   num_hidden_layers=1 (profiling)
        stack.minimal_layer_count               smallest stack reaching every
                                                distinct block type
        args.num_hidden_layers                  --num-hidden-layers
        args.hf_overrides                       --hf-override KEY=VALUE
        sharded_overrides                       per-tp divide of SHARD_FIELDS

    Later entries win, so an explicit ``--hf-override num_hidden_layers=...``
    beats ``--num-hidden-layers``, and TP sharding beats both.

    MNBT bump: the engine is booted with ``max_num_batched_tokens``
    set to ``logical_mnbt + logical_msq`` so scheduler-bypass fires
    (attention + skew grids) can exceed the logical MNBT up to
    ``MNBT + MSQ`` without overflowing vLLM's input_batch buffer.
    ``probe_limits`` undoes the bump so downstream grid generators
    and feasibility filters still see the user-intended values.

    ``model`` is left blank here; ``spin_up`` fills it in with the
    path to the temp dir it creates.
    """
    # 1. Defaults + engine scalars (dtype, kv_cache_dtype, ...).
    kwargs = _deep_merge(HOST_ENGINE_DEFAULTS, _profile_engine_overrides(args))

    # 1b. Apply the MNBT bump. After the merge, kwargs["max_num_*"]
    # reflect the logical (user-intended) values; the engine must be
    # booted with room for skew/attention boundary shots that submit
    # up to ``MNBT + MSQ`` tokens in a single scheduler-bypass fire.
    logical_mnbt = int(kwargs["max_num_batched_tokens"])
    logical_msq = int(kwargs["max_num_seqs"])
    kwargs["max_num_batched_tokens"] = logical_mnbt + logical_msq

    # 2. Single-GPU emulation. Actual model= path is set by spin_up.
    kwargs["tensor_parallel_size"] = 1

    # 3. Compose hf_overrides:
    #       defaults (num_hidden_layers=1) → CLI → sharded.
    hf_overrides: dict[str, Any] = dict(
        HOST_ENGINE_DEFAULTS.get("hf_overrides", {})
    )

    # How far to shrink the stack. The default of 1 is right only when every
    # block is identical; a hybrid needs the smallest prefix that instantiates
    # every distinct block type, or the catalog never sees the others and
    # their layers read as free. Resolved from the checkpoint's own config so
    # nobody has to know that Qwen3.8-27B's answer is 4, and logged because it
    # decides whether a block type gets measured at all.
    resolved_layers = minimal_layer_count_for(
        args.model_config or {}, stack_axes)
    hf_overrides["num_hidden_layers"] = resolved_layers
    log.info("%s", describe_stack(args.model_config or {}))
    if stack_axes != ALL_AXES:
        # describe_stack answers for every axis; say what this engine is
        # actually being shrunk to, or the two lines contradict each other.
        log.info(
            "shrinking to %d layer(s) for this engine: it measures %s only, "
            "and the stack is uniform on %s",
            resolved_layers, "/".join(stack_axes),
            "those axes" if resolved_layers == 1 else "part of them",
        )

    if args.num_hidden_layers is not None:
        hf_overrides["num_hidden_layers"] = args.num_hidden_layers
        if args.num_hidden_layers < resolved_layers:
            log.warning(
                "--num-hidden-layers %d is below the %d needed to reach every "
                "block type; some layers will have no profiled data and will "
                "look free to the simulator",
                args.num_hidden_layers, resolved_layers,
            )
    if args.hf_overrides:
        hf_overrides = _deep_merge(hf_overrides, args.hf_overrides)

    # 4. Resolve native config defaults and user overrides before TP sharding.
    # Changing the head count must not change a dimension derived from it.
    if args.model_config is None:
        raise ValueError(
            "ProfileArgs.model_config is required — build it via "
            "config.read_model_config(path) in the CLI layer."
        )

    from profiler.core.hooks.tp_shape import resolved_tp_overrides
    sharded_overrides = resolved_tp_overrides(
        args.model_config, hf_overrides, tp, SHARD_FIELDS, _materialize_config)

    # 5. Sharding wins.
    kwargs["hf_overrides"] = _deep_merge(hf_overrides, sharded_overrides)

    if native_profile is not None:
        kwargs = _deep_merge(kwargs, native_profile.engine_overrides())

    # 6. Wire the worker extension.
    kwargs["worker_extension_cls"] = "profiler.core.hooks.extension.Extension"


    return kwargs


# ---------------------------------------------------------------------------
# Spin up / spin down
# ---------------------------------------------------------------------------

def _materialize_config(
    model_config: dict[str, Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """The config to write to disk, with the overrides mirrored where the
    model's config class will actually read them.

    ``hf_overrides`` reaches the **top level** of a config. That is enough for a
    flat one, and useless for a wrapper that builds its backbone from a
    ``text_config`` key: MiniMax-M3's class takes ``text_config`` and sends
    everything else to ``**kwargs``, so ``num_hidden_layers`` and the TP shard
    fields land nowhere and the backbone is constructed from its own defaults.
    That fails loudly at 60 layers of experts (out of memory) and would fail
    *silently* on a smaller model -- profiling a shape nobody asked for.

    For a nested config, backbone overrides are written into ``text_config``
    and consumed from the subsequently forwarded ``hf_overrides``. Reapplying
    them to the wrapper can hit a read-only proxy property or create a second,
    conflicting value. Unrelated or independently declared wrapper overrides
    stay forwarded. Only keys the backbone already declares are touched,
    apart from the decoder depth and a scalar attention head-width override.
    """
    from .stack import materialize_prefix
    written = materialize_prefix(model_config, kwargs.get("hf_overrides") or {})
    inner = written.get("text_config")
    overrides = kwargs.get("hf_overrides") or {}
    # Native config construction reapplies top-level overrides. Forward the
    # already validated prefix schedule, never the original full-depth list.
    native_text = written.get("text_config", written)
    overrides = copy.deepcopy(overrides)
    for key in ("linear_attn_config", "layer_types"):
        if key in overrides and key in native_text:
            overrides[key] = copy.deepcopy(native_text[key])
    kwargs["hf_overrides"] = overrides
    if isinstance(inner, dict):
        for key, value in overrides.items():
            # A native scalar head width may be derived rather than declared
            # in the raw text config. Materialize the TP-preservation pin in
            # the backbone and consume it below like other text-only fields.
            if key in inner or key in ("num_hidden_layers", "head_dim"):
                inner[key] = copy.deepcopy(value)
    else:
        inner = written
    # Profiling a decoder prefix must retain that prefix's explicit layer
    # types. Newer config classes validate list length against the depth.
    depth = overrides.get("num_hidden_layers")
    layer_types = overrides.get("layer_types", inner.get("layer_types"))
    if depth is not None and isinstance(layer_types, list):
        if type(depth) is not int or depth < 1 or len(layer_types) < depth:
            raise ValueError("Profiling depth exceeds the explicit layer_types prefix")
        inner["layer_types"] = copy.deepcopy(layer_types[:depth])
        inner["num_hidden_layers"] = depth
        # An explicit override is applied again after reading config.json.
        if "layer_types" in overrides:
            overrides["layer_types"] = copy.deepcopy(layer_types[:depth])
    if isinstance(written.get("text_config"), dict):
        # Those values are already present in the local file read by the
        # native config constructor. Apply them there once, not again to the
        # wrapper's attributes (which may expose read-only text properties).
        kwargs["hf_overrides"] = {
            key: value for key, value in overrides.items()
            if key not in inner or key in written
        }
    return written


def _private_engine_overrides(value):
    """Bound isolated acquisition cache allocation without changing workloads."""
    if value is None:
        return {}
    if (type(value) is not dict or set(value) != {'kv_cache_memory_bytes'}
            or type(value['kv_cache_memory_bytes']) is not int
            or value['kv_cache_memory_bytes'] < 1):
        raise ValueError('Private engine overrides require positive integer kv_cache_memory_bytes only')
    return dict(value)


def spin_up(
    args: ProfileArgs, tp: int, stack_axes: tuple[str, ...] = ALL_AXES,
    *, native_profile=None, engine_overrides=None,
) -> tuple[LLM, dict[str, Any], Path]:
    """Construct a vLLM engine ready for profiling.

    Side effect: creates a temporary directory containing
    ``config.json`` (the contents of ``args.model_config``) and
    points vLLM at it. This lets the profiler work entirely offline
    and lets users profile custom-shape models by simply editing the
    local config JSON — no HF hub id needs to correspond to the
    shape they want.

    Returns:
        Tuple ``(llm, kwargs, tmpdir)``:
            * ``llm``: the live vLLM engine.
            * ``kwargs``: the full dict passed to ``vllm.LLM()``
              (recorded in meta.yaml).
            * ``tmpdir``: path to the temp config directory. The
              caller MUST pass this to ``spin_down`` so it gets
              cleaned up.
    """
    overrides = _private_engine_overrides(engine_overrides)
    kwargs = fuse_engine_kwargs(args, tp, stack_axes, native_profile=native_profile)
    kwargs.update(overrides)
    from .hooks.dummy_weights import validate_execution_scope
    validate_execution_scope(args.model_config, tp=tp, moe_ep=1,
                             profile_mtp=args.profile_mtp, kwargs=kwargs)

    # Materialize the model's config.json in a temp directory so vLLM
    # can load it directly from disk.
    assert args.model_config is not None, (
        "spin_up requires ProfileArgs.model_config; fuse_engine_kwargs "
        "should have caught this already."
    )
    materialized = _materialize_config(args.model_config, kwargs)
    validate_execution_scope(materialized, tp=tp, moe_ep=1,
                             profile_mtp=args.profile_mtp, kwargs=kwargs)
    if args.profile_mtp:
        materialized = _cap_mtp_modules(materialized)
    tmpdir = Path(tempfile.mkdtemp(prefix="profiler_model_"))
    config_path = tmpdir / "config.json"
    config_path.write_text(json.dumps(materialized, indent=2))
    log.debug("model config written to %s", config_path)

    kwargs["model"] = str(tmpdir)
    if args.profile_mtp:
        # Same directory for both: MTP drafts with the target's own
        # checkpoint, and vLLM rewrites the config into the MTP one itself
        # (SpeculativeConfig.hf_config_override).
        # Always 1, whatever N the user intends to simulate: the drafter runs
        # once per speculative token, so booting at N records N passes in one
        # shot and the simulator -- which emits N passes of its own -- would
        # multiply again. One pass is the unit.
        kwargs["speculative_config"] = {
            "model": str(tmpdir),
            "num_speculative_tokens": 1,
        }

    llm = None
    try:
        with log.capture_stdio():
            llm = LLM(**kwargs)
        receipts = llm.collective_rpc("dummy_weight_preparation", args=())
        if not isinstance(receipts, list) or len(receipts) != 1:
            raise ValueError("Expected one dummy-preparation worker receipt")
        receipt = receipts[0]
        from .hooks.dummy_weights import format_intent
        if format_intent(materialized) and receipt is None:
            raise ValueError("Native MXFP4 engine has no dummy-preparation receipt")
        if receipt is not None:
            from .hooks.dummy_weights import POLICY
            if (not isinstance(receipt, dict) or receipt.get("policy") != POLICY
                    or receipt.get("complete") is not True
                    or receipt.get("forward_verified") is not False
                    or receipt.get("postload", {}).get("structure_verified") is not True
                    or receipt.get("postload", {}).get("forward_verified") is not False):
                raise ValueError("Worker dummy-preparation receipt is incomplete or incompatible")
            receipt = copy.deepcopy(receipt)
            receipt["checkpoint_quantization"] = {
                key: copy.deepcopy(holder["quantization_config"])
                for key, holder in (("root", args.model_config),
                                    ("text", args.model_config.get("text_config")))
                if isinstance(holder, dict) and "quantization_config" in holder}
        llm._profiler_dummy_weight_preparation = receipt
    except BaseException:
        if llm is not None:
            spin_down(llm, tmpdir)
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    return llm, kwargs, tmpdir


def _cap_mtp_modules(config: dict[str, Any], cap: int = 1) -> dict[str, Any]:
    """Cap ``num_mtp_modules`` in the config **file**, for MTP profiling.

    Not an ``hf_overrides`` entry, because the drafter reads
    ``speculative_config.draft_model_config.hf_config`` -- built from the
    config on disk -- rather than the target's overridden one. Unlike
    ``num_hidden_layers`` the field has no cross-field validation, so writing
    it is safe.

    MiniMax-M3 declares 7 modules, each a full MoE decoder layer at ~14.8 GB,
    which is ~103 GB and does not fit one card. They are all the same class and
    the simulator multiplies by the declared count, so profiling one is enough.
    """
    out = dict(config)
    for holder in (out, out.get("text_config")):
        if isinstance(holder, dict) and "num_mtp_modules" in holder:
            if int(holder["num_mtp_modules"] or 0) > cap:
                log.info(
                    "capping num_mtp_modules %s -> %d for MTP profiling; the "
                    "modules are identical and the simulator multiplies by the "
                    "declared count",
                    holder["num_mtp_modules"], cap,
                )
                holder["num_mtp_modules"] = cap
    return out


def resolve_attention_max_kv(args: ProfileArgs, limits: RuntimeLimits) -> int:
    """The kv-axis cap for the attention and skew grids.

    Unset means **the model's own context window**, not a constant: a cap that
    does not follow the checkpoint stops the sweep at 16k on a model that
    serves 160k, and the simulator then extrapolates the rest. That is fine for
    a dense kernel, which is linear in kv, and wrong for a sparse one -- past
    ``index_topk`` the attention kernel is flat while the indexer keeps growing
    with the whole KV, so the two curves diverge exactly in the region no shot
    covers.

    The derived cap is ``max_model_len - max(decode_q_len) - 1`` rather than
    ``max_model_len``: a decode request occupies ``kv + q`` positions and the
    grid needs one more to be a decode rather than the whole window, so passing
    the context length verbatim gets the top point filtered out and the sweep
    silently stops one doubling short.

    An explicit value still wins, and is still clamped to what the engine
    accepted.
    """
    if args.attention_max_kv is not None:
        return min(args.attention_max_kv, limits.max_model_len)
    q_max = max((int(q) for q in args.attention_decode_q_lens), default=1)
    return max(1, limits.max_model_len - max(1, q_max) - 1)


def probe_limits(llm: LLM, args: ProfileArgs | None = None) -> RuntimeLimits:
    """Read back the runtime shapes the engine accepted.

    Undoes the MNBT bump applied in ``fuse_engine_kwargs`` so
    downstream consumers see the logical (user-intended)
    ``max_num_batched_tokens``. Since the bump amount equals MSQ
    (which is not itself bumped), the recovery is simply
    ``engine_mnbt - engine_msq``.
    """
    cfg = llm.llm_engine.vllm_config

    # ``kv_cache_size_tokens`` is vLLM's own group-aware capacity, added in
    # v0.28. Prefer it: for a hybrid model whose requests occupy several KV
    # cache groups at once (attention pages + mamba state), the older
    # ``num_gpu_blocks * block_size`` overstates what a request can actually
    # get, and every feasibility filter downstream would inherit that error.
    # Fall back to the product for engines that leave it unset.
    num_cache_tokens = getattr(cfg.cache_config, "kv_cache_size_tokens", None)
    if num_cache_tokens is None:
        num_cache_blocks = cfg.cache_config.num_gpu_blocks
        block_size = cfg.cache_config.block_size
        assert num_cache_blocks is not None, "vLLM did not report num_gpu_blocks"
        num_cache_tokens = num_cache_blocks * block_size

    # MoE params read from the live HF config (post-override).
    hf_cfg = getattr(cfg.model_config, "hf_text_config", None)
    if hf_cfg is None:
        hf_cfg = cfg.model_config.hf_config
    cfg_dict = (
        hf_cfg.to_dict() if hasattr(hf_cfg, "to_dict") else vars(hf_cfg)
    )
    moe_params = probe_moe_params(cfg_dict)
    num_experts, top_k = moe_params or (None, None)
    key_saturation = probe_key_saturation(cfg_dict)

    # CLI wins over the resolved value, so a user can widen or coarsen the
    # prefill grid without editing a model config.
    chunk = probe_linear_attn_chunk(cfg_dict)
    if args is not None and args.linear_attn_chunk is not None:
        chunk = int(args.linear_attn_chunk)

    engine_mnbt = cfg.scheduler_config.max_num_batched_tokens
    engine_msq = cfg.scheduler_config.max_num_seqs
    logical_mnbt = engine_mnbt - engine_msq
    sampling_runtime = llm.collective_rpc('sampling_runtime')[0]

    return RuntimeLimits(
        max_num_batched_tokens=logical_mnbt,
        max_num_seqs=engine_msq,
        num_cache_tokens=int(num_cache_tokens),
        block_size=int(cfg.cache_config.block_size),
        linear_attn_chunk=chunk,
        max_model_len=cfg.model_config.max_model_len,
        num_experts=num_experts,
        top_k=top_k,
        key_saturation=key_saturation,
        model_text_config=copy.deepcopy(cfg_dict),
        dummy_weight_preparation=copy.deepcopy(getattr(llm, "_profiler_dummy_weight_preparation", None)),
        sampling_runtime=sampling_runtime,
    )


def spin_down(llm: LLM, tmpdir: Path | None = None) -> None:
    """Release GPU memory held by the engine and clean the temp
    config directory created by ``spin_up``.

    vLLM v1 runs an ``EngineCore`` in a subprocess that holds GPU
    memory independently of the host-side ``LLM`` object. ``del llm``
    alone doesn't guarantee the subprocess is reaped in time for the
    next ``spin_up`` to see the GPU clean — which is how a TP=2 boot
    right after a TP=1 finish hits "Free memory … less than desired
    GPU memory utilization".

    To reliably free memory between TP steps we:
      1. Try every shutdown / close hook on engine_core and
         llm_engine (names have shifted between vLLM versions).
      2. Tear down the distributed state (model parallel + nccl).
      3. Drop references, gc, ``torch.cuda.empty_cache``.
      4. Remove the spin_up tmpdir.
    """
    # 1. Best-effort shutdown of the engine-core subprocess.
    try:
        engine = getattr(llm, "llm_engine", None)
        if engine is not None:
            core = getattr(engine, "engine_core", None)
            if core is not None:
                for attr in ("shutdown", "close"):
                    fn = getattr(core, attr, None)
                    if callable(fn):
                        try:
                            fn()
                            break
                        except Exception as e:
                            log.debug("engine_core.%s() raised: %s", attr, e)
            for attr in ("shutdown", "close"):
                fn = getattr(engine, attr, None)
                if callable(fn):
                    try:
                        fn()
                        break
                    except Exception as e:
                        log.debug("llm_engine.%s() raised: %s", attr, e)
    except Exception as e:
        log.debug("spin_down engine shutdown: %s", e)

    # 2. Distributed teardown (skip silently if not set up / already torn).
    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        log.debug("spin_down distributed teardown: %s", e)

    # 3. Drop Python references + clear CUDA cache.
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    # Second pass after teardown calls often frees the final holdouts.
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Clean up the temp config dir.
    if tmpdir is not None and tmpdir.exists():
        shutil.rmtree(tmpdir, ignore_errors=True)
