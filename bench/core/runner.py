"""vLLM benchmark runner — strict replay of an existing dataset.

The runner reads a LLMServingSim-format JSONL workload (the same format
``python -m workloads.generators sharegpt`` produces and ``python -m serving
--dataset`` consumes) and replays every request through vLLM with its
``input_tok_ids`` and ``output_toks`` pinned, so the run is bit-for-bit
comparable to the simulator's view of the same workload.

White-room implementation against ``../vllm``:
  * ``vllm.v1.engine.async_llm.AsyncLLM`` — async engine, ``generate()``
    yields ``RequestOutput`` per chunk, ``RequestOutput.metrics`` carries
    per-request ``RequestStateStats`` (arrival_time / queued_ts /
    scheduled_ts / first_token_ts / last_token_ts).
  * ``vllm.v1.metrics.loggers.StatLoggerBase`` — pluggable per-engine stat
    logger; we hook it via ``BenchStatLogger`` to capture per-iteration
    scheduler/iteration stats for ``timeseries.csv``.

Output: ``<output-dir>/{meta.json, requests.jsonl, timeseries.csv}``.
The dataset itself is not modified — generation lives in
``workloads/generators``.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import logging
from pathlib import Path

from bench.core import logger as log
from bench.core import recorder


def register_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="HF model id passed verbatim to vllm.AsyncLLM.")
    p.add_argument("--dataset", required=True,
                   help="Path to a LLMServingSim-format JSONL workload "
                        "(produced by `python -m workloads.generators`).")
    p.add_argument("--output-dir", required=True, dest="output_dir",
                   help="Output directory for this run "
                        "(meta.json/requests.jsonl/timeseries.csv).")
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                   dest="tensor_parallel_size",
                   help="vLLM tensor_parallel_size.")
    p.add_argument("--data-parallel-size", type=int, default=1,
                   dest="data_parallel_size",
                   help="vLLM data_parallel_size (DP across engines).")
    p.add_argument("--enable-expert-parallel", action="store_true",
                   dest="enable_expert_parallel", default=False,
                   help="vLLM enable_expert_parallel for MoE models.")
    p.add_argument("--max-num-seqs", type=int, default=128,
                   dest="max_num_seqs",
                   help="vLLM scheduler max_num_seqs (per-engine running cap).")
    p.add_argument("--max-num-batched-tokens", type=int, default=2048,
                   dest="max_num_batched_tokens",
                   help="vLLM scheduler max_num_batched_tokens.")
    p.add_argument("--max-model-len", type=int, default=None,
                   dest="max_model_len",
                   help="vLLM max_model_len (None = model's max).")
    p.add_argument("--dtype", default="bfloat16",
                   help="Model dtype.")
    p.add_argument("--kv-cache-dtype", default="auto",
                   dest="kv_cache_dtype",
                   help="vLLM kv_cache_dtype.")
    p.add_argument("--seed", type=int, default=42,
                   help="Sampling seed for vLLM.")
    p.add_argument("--tick-seconds", type=float, default=1.0,
                   dest="tick_seconds",
                   help="Stat logger downsample interval (timeseries.csv row spacing).")
    p.add_argument("--num-reqs", type=int, default=0,
                   dest="num_reqs",
                   help="Cap on number of requests from the dataset (0 = all).")
    p.add_argument("--log-level", default="INFO",
                   dest="log_level",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logger verbosity (default: INFO).")


def run(args: argparse.Namespace) -> int:
    from bench.core.stat_logger import BenchStatLogger

    log.configure(args.log_level)
    log.print_banner(
        "LLMServingSim Bench",
        f"vLLM end-to-end run -> {args.output_dir}",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requests = _load_dataset(Path(args.dataset), cap=args.num_reqs)
    if not requests:
        raise ValueError(f"No requests loaded from {args.dataset}")
    log.info("Loaded %d requests from %s", len(requests), args.dataset)

    BenchStatLogger.reset()
    asyncio.run(_drive(args, requests, output_dir))
    return 0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset(path: Path, cap: int = 0) -> list[dict]:
    """Read a LLMServingSim-format JSONL workload.

    Skips agentic-session rows (with ``sub_requests``) — bench currently
    handles only flat requests. Each row must carry ``input_tok_ids``;
    bench cannot tokenize on the fly because the dataset's tokenizer may
    differ from ``args.model``.
    """
    requests: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "sub_requests" in row:
                continue  # agentic sessions: not supported in bench yet
            if "input_tok_ids" not in row or not row["input_tok_ids"]:
                raise ValueError(
                    f"Row missing input_tok_ids in {path}; regenerate the "
                    f"dataset with `python -m workloads.generators`."
                )
            requests.append(row)
            if cap and len(requests) >= cap:
                break
    return requests


# ---------------------------------------------------------------------------
# Async driver
# ---------------------------------------------------------------------------

async def _drive(args: argparse.Namespace, requests: list[dict], output_dir: Path) -> None:
    # Imports deferred so `validate` / `--help` works without vLLM installed.
    from vllm import AsyncEngineArgs, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.v1.engine.async_llm import AsyncLLM

    from bench.core.stat_logger import BenchStatLogger

    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        seed=args.seed,
        disable_log_stats=False,
    )
    engine_kwargs_for_meta = _engine_kwargs_for_meta(engine_args)

    with log.stage("Booting AsyncLLM"):
        with log.capture_stdio():
            engine = AsyncLLM.from_engine_args(
                engine_args, stat_loggers=[BenchStatLogger]
            )
    started_at = datetime.datetime.utcnow().isoformat() + "Z"

    try:
        with log.stage(f"Submitting {len(requests)} requests"):
            records = await _submit_all(
                engine, requests, SamplingParams, TokensPrompt
            )
    finally:
        with log.stage("Shutting AsyncLLM down"):
            engine.shutdown()

    finished_at = datetime.datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------
    # Persist outputs.
    # ------------------------------------------------------------------
    # Collecting metadata must never be the thing that loses a completed run.
    try:
        resolved_config = _resolved_config(engine)
        kv_cache = _kv_cache_facts(engine)
    except Exception as exc:
        log.warning("could not snapshot the resolved vLLM config: %s", exc)
        resolved_config, kv_cache = {}, {}

    recorder.write_meta(
        output_dir,
        model=args.model,
        vllm_version=_vllm_version(),
        engine_kwargs=engine_kwargs_for_meta,
        dataset_path=str(args.dataset),
        dataset_hash=_hash_file(Path(args.dataset)),
        num_requests=len(records),
        started_at=started_at,
        finished_at=finished_at,
        tick_seconds=args.tick_seconds,
        # What vLLM resolved, not what we asked for: defaults filled in,
        # max_model_len inferred, num_gpu_blocks profiled. kv_cache is pulled
        # out of it because num_gpu_blocks is the number a simulator has to
        # match, and the only place vLLM's activation peak becomes visible.
        kv_cache=kv_cache,
        hardware=_hardware_facts(),
        resolved_config=resolved_config,
    )
    recorder.write_requests(output_dir, records)
    header, rows = BenchStatLogger.downsample_to_csv_rows(args.tick_seconds)
    recorder.write_timeseries(output_dir, header, rows)
    log.success(
        "%d requests, %d timeseries rows -> %s",
        len(records), len(rows), output_dir,
    )


async def _submit_all(engine, requests: list[dict], SamplingParams, TokensPrompt) -> list[dict]:
    """Schedule each request at its arrival offset, gather metrics."""
    loop = asyncio.get_event_loop()
    t0_loop = loop.time()
    completed = [0]  # boxed so the inner closure can mutate

    with log.progress("Requests", total=len(requests)) as bar:

        async def _one(idx: int, req: dict) -> dict:
            target = t0_loop + req["arrival_time_ns"] / 1e9
            delay = target - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)

            # Strict replay: pin output length to whatever the dataset
            # recorded. ``ignore_eos`` blocks early termination; ``min_tokens``
            # blocks vLLM's async-scheduling early-exit (see vllm/v1/engine/
            # async_llm.py:async-scheduling block) so n_out is exactly fixed.
            n_out = int(req["output_toks"])
            sp = SamplingParams(
                min_tokens=n_out,
                max_tokens=n_out,
                ignore_eos=True,
                temperature=0.0,
            )
            prompt = TokensPrompt(prompt_token_ids=list(req["input_tok_ids"]))
            request_id = f"bench-{idx}"

            last_metrics = None
            async for output in engine.generate(prompt, sp, request_id):
                if output.metrics is not None:
                    last_metrics = output.metrics

            completed[0] += 1
            bar.advance()
            return _record_from_metrics(idx, req, last_metrics)

        tasks = [asyncio.create_task(_one(i, r)) for i, r in enumerate(requests)]
        return await asyncio.gather(*tasks)


def _record_from_metrics(idx: int, req: dict, metrics) -> dict:
    """Project ``RequestStateStats`` onto our flat per-request schema."""
    if metrics is None:
        return {
            "request_id": f"bench-{idx}",
            "input_toks": int(req["input_toks"]),
            "output_toks": int(req["output_toks"]),
            "arrival_time": None,
            "queued_ts": None,
            "scheduled_ts": None,
            "first_token_ts": None,
            "last_token_ts": None,
        }
    return {
        "request_id": f"bench-{idx}",
        "input_toks": int(req["input_toks"]),
        "output_toks": int(req["output_toks"]),
        "arrival_time": getattr(metrics, "arrival_time", None),
        "queued_ts": getattr(metrics, "queued_ts", None),
        "scheduled_ts": getattr(metrics, "scheduled_ts", None),
        "first_token_ts": getattr(metrics, "first_token_ts", None),
        "last_token_ts": getattr(metrics, "last_token_ts", None),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine_kwargs_for_meta(engine_args) -> dict:
    fields = (
        "model", "tensor_parallel_size", "data_parallel_size",
        "enable_expert_parallel", "max_num_seqs", "max_num_batched_tokens",
        "max_model_len", "dtype", "kv_cache_dtype", "seed",
    )
    return {k: getattr(engine_args, k, None) for k in fields}


# Values longer than this are recorded as a type tag instead, so one HF config
# object cannot bury the rest of meta.json.
_MAX_REPR = 200


def _normalize(value, depth: int = 0):
    """Reduce a config value to something JSON can hold.

    Scalars pass through. Enums become their value. Short flat containers are
    kept element-wise. Anything else -- an HF config object, a torch dtype, a
    callable -- becomes a ``"<type>"`` tag: the field is still recorded as
    present and resolved, without dragging its whole object graph into the file.
    """
    import enum

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return _normalize(value.value, depth + 1)
    if isinstance(value, (list, tuple, set, frozenset)):
        seq = list(value)
        if len(seq) > 64:
            return f"<{type(value).__name__} len={len(seq)}>"
        return [_normalize(v, depth + 1) for v in seq]
    if isinstance(value, dict):
        if len(value) > 64 or depth > 2:
            return f"<dict len={len(value)}>"
        return {str(k): _normalize(v, depth + 1) for k, v in value.items()}
    text = repr(value)
    if len(text) <= _MAX_REPR:
        return text
    return f"<{type(value).__name__}>"


def _config_fields(obj) -> dict:
    """Field name -> normalized value for one config object.

    vLLM's config classes are pydantic-decorated dataclasses, so try the
    dataclass field list first, then pydantic's, then fall back to __dict__.
    Order does not matter: the dict is sorted on the way out.
    """
    import dataclasses

    names: list[str] = []
    if dataclasses.is_dataclass(obj):
        names = [f.name for f in dataclasses.fields(obj)]
    elif hasattr(type(obj), "model_fields"):
        names = list(type(obj).model_fields)
    elif hasattr(obj, "__dict__"):
        names = list(vars(obj))
    out = {}
    for name in sorted(names):
        if name.startswith("_"):
            continue
        try:
            out[name] = _normalize(getattr(obj, name))
        except Exception as exc:                     # a property may raise
            out[name] = f"<unreadable: {type(exc).__name__}>"
    return out


def _resolved_config(engine) -> dict:
    """Snapshot vLLM's *resolved* configuration, one section per sub-config.

    Reads the engine's own ``VllmConfig`` after boot, so these are the values
    vLLM actually ran with rather than the arguments we asked for -- defaults
    filled in, ``max_model_len`` inferred, ``num_gpu_blocks`` profiled. Walks
    ``VllmConfig``'s field list rather than a hand-kept list of interesting
    knobs, so a vLLM upgrade that adds one shows up here on its own.
    """
    cfg = getattr(engine, "vllm_config", None)
    if cfg is None:
        return {}
    out = {}
    for name, sub in _config_fields(cfg).items():
        try:
            value = getattr(cfg, name)
        except Exception:
            continue
        if value is None or isinstance(value, (bool, int, float, str)):
            out[name] = sub                      # a scalar on VllmConfig itself
        else:
            fields = _config_fields(value)
            out[name] = fields if fields else sub
    return out


def _kv_cache_facts(engine) -> dict:
    """The KV cache vLLM actually allocated.

    ``num_gpu_blocks`` is the number the simulator has to match, and it is the
    only place the activation peak and CUDA context that vLLM subtracts become
    visible: everything else in the budget is known up front.
    """
    cfg = getattr(engine, "vllm_config", None)
    if cfg is None:
        return {}
    cache = getattr(cfg, "cache_config", None)
    if cache is None:
        return {}
    blocks = getattr(cache, "num_gpu_blocks", None)
    block_size = getattr(cache, "block_size", None)
    facts = {
        "num_gpu_blocks": blocks,
        "num_cpu_blocks": getattr(cache, "num_cpu_blocks", None),
        "block_size": block_size,
        "gpu_memory_utilization": getattr(cache, "gpu_memory_utilization", None),
        "enable_prefix_caching": getattr(cache, "enable_prefix_caching", None),
        "kv_cache_memory_bytes": getattr(cache, "kv_cache_memory_bytes", None),
    }
    if isinstance(blocks, int) and isinstance(block_size, int):
        facts["num_kv_tokens"] = blocks * block_size
    return facts


def _hardware_facts() -> dict:
    """Which accelerator this ran on, for matching against profiler/perf/<hw>/."""
    facts = {}
    try:
        import torch
        facts["torch_version"] = torch.__version__
        facts["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            facts["device_count"] = torch.cuda.device_count()
            facts["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            facts["device_total_memory_bytes"] = props.total_memory
            facts["device_capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:
        facts["error"] = f"{type(exc).__name__}: {exc}"
    return facts


def _vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except Exception:
        return "unknown"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
