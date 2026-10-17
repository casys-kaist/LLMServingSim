"""CLI entry point: ``python -m profiler ...``.

Profiling subcommands:

    profile <model> --hardware <hw> [options]
        Full sweep: every TP × every category.

    slice <model> --hardware <hw> --tp-refresh N --group G [options]
        Refresh one (tp, category) pair.

    coverage <model> --hardware <hw> [options]
        Boot once and report how much of the model's CUDA time the
        architecture catalog binds. Writes nothing. Run this first when
        adding a model family: a catalog entry can name a real class and
        still measure nothing, because vLLM's profile tree only holds
        modules that launch a kernel of their own, and reading the module
        tree cannot tell you which those are.

Offline calibration:

    import-execution <model> --hardware <hw> --bench-dir DIR [--variant NAME]
        Save a matching benchmark's observed serving graph policy in meta.yaml.

    plan-skew <model> --hardware <hw> [options]
        Preview heterogeneous coverage using saved engine limits and references.

    refit-skew <model> --hardware <hw> [--tp N]
        Compile existing skew measurements against local attention references
        without starting vLLM or using a GPU.

Hardware acquisition:

    hardware --hardware <hw> [--npus N] [--out ROOT]
        Record device specifications and measure NCCL interconnect parameters.

Model resolution
----------------
For model acquisition commands, ``<model>`` is an HF-style ``<org>/<name>``
or an explicit ``.json`` config path. The profiler:

1. Resolves the local config, downloading a missing named config from the Hub.
2. Extracts ``model_type``.
3. Looks up ``profiler/models/<model_type>.yaml`` as the
   architecture. Errors out if none is found.
4. Materializes the config in a temporary model directory for vLLM, using
   dummy weights and no tokenizer initialization for layer acquisition.

To profile a model not in ``configs/model/`` yet:
* Download its ``config.json`` from HuggingFace.
* Place it at ``configs/model/<org>/<name>.json``.
* Ensure a matching ``profiler/models/<model_type>.yaml`` exists.
* Run the profiler.

Verbosity
---------
    (default)                                INFO — shows TP limits,
                                             stage timings, per-category
                                             progress.
    --silent                                 WARNING — warnings only.
    --verbose                                DEBUG + vLLM stdout.
    --log-level {DEBUG,INFO,WARNING,ERROR}   explicit override.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from profiler.core import logger as log
from profiler.core.config import (
    SQRT2,
    ProfileArgs,
    detect_model_type,
    read_model_config,
    resolve_architecture_by_model_type,
)


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
#
# Resolved relative to the profiler/ directory that houses this
# package so the CLI works regardless of cwd.

_PKG_ROOT = Path(__file__).resolve().parent         # .../profiler
_REPO_ROOT = _PKG_ROOT.parent                       # .../LLMServingSim

ARCH_DIR = _PKG_ROOT / "models"                     # architecture yamls
PERF_DIR = _PKG_ROOT / "perf"                       # output root
MODEL_CONFIG_DIR = _REPO_ROOT / "configs" / "model" # LLMServingSim's shared configs


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _grid_factor(value: str) -> float:
    try:
        factor = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be finite and greater than 1") from None
    if not math.isfinite(factor) or factor <= 1:
        raise argparse.ArgumentTypeError("must be finite and greater than 1")
    return factor


def _parse_cudagraph(value: str) -> dict:
    """Validate target graph options without resolving engine-dependent bounds."""
    try:
        options = json.loads(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("must be a JSON object") from None
    if not isinstance(options, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    unknown = set(options) - {"mode", "capture_sizes", "max_capture_size", "enable_sp", "runner"}
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown cudagraph fields: {sorted(unknown)}")
    modes = ("NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE")
    if "mode" in options and options["mode"] not in modes:
        raise argparse.ArgumentTypeError(f"cudagraph.mode must be one of {modes}")
    if "runner" in options and options["runner"] not in ("v1", "v2"):
        raise argparse.ArgumentTypeError("cudagraph.runner must be 'v1' or 'v2'")
    if "enable_sp" in options and not isinstance(options["enable_sp"], bool):
        raise argparse.ArgumentTypeError("cudagraph.enable_sp must be boolean")
    cap = options.get("max_capture_size")
    if cap is not None and (type(cap) is not int or cap <= 0):
        raise argparse.ArgumentTypeError("cudagraph.max_capture_size must be a positive integer")
    sizes = options.get("capture_sizes")
    if sizes is not None and (not isinstance(sizes, list) or not sizes
                              or any(type(size) is not int or size <= 0 for size in sizes)):
        raise argparse.ArgumentTypeError("cudagraph.capture_sizes must be a nonempty list of positive integers")
    return options


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    """Flags registered by profile, slice, coverage and plan-skew."""
    p.add_argument(
        "--hardware",
        required=True,
        help="Hardware identifier (e.g., H100, A6000). Becomes a "
             "folder name under perf/.",
    )
    p.add_argument(
        "--tp",
        default="1",
        help="Comma-separated TP degrees to sweep, e.g. '1,2,4'. "
             "Must include 1 except for plan-skew or profile --only-skew. "
             "Default: '1'.",
    )
    p.add_argument(
        "--variant",
        default=None,
        help="Output folder label. Default: checkpoint precision plus explicit "
             "activation/KV dtype suffixes (e.g. 'fp8-actfp16'). Required for "
             "HF precision or nested text_config overrides.",
    )

    # Engine kwargs.
    p.add_argument("--dtype", default=None,
                   help="Override the vLLM engine dtype. Default: resolved "
                        "from the checkpoint; this flag does not quantize weights.")
    p.add_argument("--kv-cache-dtype", default=None,
                   help="Override the KV cache dtype with a vLLM-supported "
                        "value. Default: auto.")
    p.add_argument("--max-num-batched-tokens", type=int, default=None,
                   dest="max_num_batched_tokens",
                   help="Per-step token budget. Matches vLLM's own "
                        "``--max-num-batched-tokens``. Default: 2048.")
    p.add_argument("--max-num-seqs", type=int, default=None,
                   dest="max_num_seqs",
                   help="Max concurrent sequences. Matches vLLM's own "
                        "``--max-num-seqs``. Default: 256.")
    p.add_argument("--block-size", type=int, default=None,
                   dest="block_size",
                   help="KV block size in tokens, vLLM's own "
                        "``--block-size``. Default: unset, so vLLM chooses "
                        "a backend-compatible value. Match the simulator "
                        "to that resolved value, "
                        "since a profile measured under one paging regime "
                        "does not describe another. On a hybrid stack vLLM "
                        "overrides this to unify attention and mamba page "
                        "sizes, and the run logs what it settled on.")
    p.add_argument("--gpu-memory-utilization", type=float, default=None,
                   dest="gpu_memory_utilization",
                   help="Fraction of GPU memory vLLM may use; the simulator "
                        "spells the same thing --npu-memory-utilization. "
                        "Default: 0.9. Sets the KV block count, which every "
                        "shot-feasibility filter is measured against, so it "
                        "changes which shots a sweep contains.")
    p.add_argument("--max-model-len", type=int, default=None,
                   dest="max_model_len",
                   help="Cap the engine's context length. Default: whatever "
                        "the model config declares. Bounds the kv axes and "
                        "the per-shot length checks, so lowering it cuts "
                        "profile time on a long-context model.")
    p.add_argument("--num-hidden-layers", type=int, default=None,
                   dest="num_hidden_layers",
                   help="Override the number of layers to instantiate. By "
                        "default the profiler resolves the smallest stack "
                        "covering the category's required block types from "
                        "the checkpoint, typically 1 for a uniform model.")
    p.add_argument("--profile-mtp", action="store_true",
                   dest="profile_mtp",
                   help="Boot with speculative decoding so vLLM also builds "
                        "the model's own MTP module, and its kernels land in "
                        "the profile tree. Needed because the MTP config "
                        "model_type (deepseek_mtp / qwen3_5_mtp / "
                        "minimax_m3_mtp) is produced by vLLM's "
                        "SpeculativeConfig and is unknown to HF Transformers, "
                        "so the module cannot be loaded on its own. Takes no "
                        "draft count: the engine boots at "
                        "num_speculative_tokens=1 so what is measured is "
                        "**one** drafter pass, which is the unit the "
                        "simulator multiplies by its own N. Booting at N "
                        "would record N passes and the simulator would then "
                        "multiply again. Only models declaring MTP modules "
                        "support this. Pair it with the coverage subcommand "
                        "first: the drafter's kernels show up as unbound "
                        "until a catalog binds them.")
    p.add_argument("--dp", default=None, dest="dp_degrees",
                   help="Target DP degrees for native MoE component profiling, "
                        "comma separated. Experts retain global top-k and IDs; "
                        "EP is TP*DP. DP=1 measures local modular components without DP collectives. "
                        "The profiler still uses one physical GPU. "
                        "Defaults to DP=1 for MoE profiling. Only profile or "
                        "slice --group moe accept this option.")
    p.add_argument("--ep", default=None, dest="ep_degrees",
                   help="Target EP degrees for native MoE components, comma separated. "
                        "Defaults to all TP*DP targets. When set, selects matching "
                        "targets without changing TP/DP or expert tensor sharding. "
                        "Every requested EP must match a selected TP*DP. Only profile "
                        "or slice --group moe accept this option; a slice uses --tp-refresh.")
    p.add_argument("--moe-rounds", type=int, default=3,
                   help="Independent measurement contexts per native MoE point.")
    p.add_argument("--moe-grid-factor", type=float, default=2.0,
                   help="Geometric spacing for native MoE token, active-expert "
                        "and assignment axes (finite and greater than 1; default 2).")
    p.add_argument("--hf-override", action="append", default=None,
                   dest="hf_override", metavar="KEY=VALUE",
                   help="Override one model-config field, repeatable. The "
                        "value is parsed as JSON when it parses and kept as a "
                        "string otherwise, so both "
                        "``--hf-override index_topk=1024`` and "
                        "``--hf-override 'layer_types=[\"a\",\"b\"]'`` work. "
                        "Applied on top of the config on disk, under TP "
                        "sharding. The generic escape hatch for sweeping a "
                        "shape without editing configs/model/.")
    p.add_argument("--linear-attn-chunk", type=int, default=None,
                   dest="linear_attn_chunk",
                   help="Chunk length the linear-attention prefill scan works "
                        "in, used to place grid points. Default: resolved "
                        "from the model config's chunk_size, else vLLM's "
                        "FLA_CHUNK_SIZE. Measured cost tracks the chunk "
                        "count, not the token count, so the grid samples "
                        "boundaries and the points just past them.")

    # Target sampling policy; acquisition remains eager.
    p.add_argument("--grid-factor", type=_grid_factor, default=2.0,
                   help="Geometric spacing for ordinary non-graph token/request "
                        "sampling (finite and greater than 1; default 2). "
                        "Attention, skew and native MoE retain their own factor options.")
    p.add_argument("--cudagraph", type=_parse_cudagraph, default=None, metavar="JSON",
                   help="Target serving CUDA graph sampling options as a JSON object: "
                        "mode, capture_sizes, max_capture_size, enable_sp, runner. "
                        "Defaults use the serving resolver. Acquisition remains eager.")

    # Attention grid.
    p.add_argument("--attention-max-kv", type=int, default=None,
                   help="History cap used to construct prefill/decode shots. "
                        "Default: the model's own "
                        "context, max_model_len - max(decode_q_len) - 1 -- a "
                        "decode occupies kv + q positions and needs one more "
                        "to be a decode at all, so passing max_model_len "
                        "verbatim gets the top point filtered and the sweep "
                        "stops a doubling short. Lower it (e.g. 16384) to "
                        "trade coverage for time; extrapolation past measured "
                        "coverage is not an accuracy guarantee, especially "
                        "when sparse attention and its indexer scale differently.")
    p.add_argument("--attention-decode-q-lens", type=str, default="1",
                   dest="attention_decode_q_lens",
                   help="Comma-separated query-token counts per decode sequence "
                        "to sweep, e.g. '1,2,4,6'. Default '1' is ordinary "
                        "decoding. A speculative-decoding verification step "
                        "submits 1 + num_speculative_tokens queries per "
                        "sequence against that sequence's own KV, which is "
                        "neither a prefill chunk of the same size nor that many "
                        "single-token decodes. Each extra value adds a separate "
                        "decode-containing grid; pure prefill is measured once at q=1.")
    p.add_argument("--attention-chunk-factor", type=float, default=2.0,
                   dest="attention_chunk_factor",
                   help="Geometric factor for the prefill-token axis. "
                        "Default 2.0 (doubling). Lower values "
                        "produce a denser grid.")
    p.add_argument("--attention-kv-factor", type=float, default=2.0,
                   dest="attention_kv_factor",
                   help="Geometric factor for the prefill-key and kv_decode "
                        "axes. Default 2.0 (doubling). Lower values add samples. "
                        "Resume still requires compatible "
                        "measurement identities and exact shot keys.")
    p.add_argument("--attention-n-factor", type=float, default=SQRT2,
                   dest="attention_n_factor",
                   help="Geometric factor for the n_decode axis. sqrt(2) "
                        "(default); 2.0 is doubling. Lower values add samples "
                        "between decode request counts, including mixed "
                        "batches. Validate accuracy with the intended workload.")
    p.add_argument("--measurement-iterations", type=int, default=3,
                   dest="measurement_iterations",
                   help="Timed forwards per ordinary shot, averaged per "
                        "invocation (default: 3). More repeats increase "
                        "acquisition cost without guaranteeing an error bound.")
    p.add_argument("--skip-skew", action="store_true", default=False,
                   dest="skip_skew",
                   help="Skip the per-TP skew profiling step (skew.csv). "
                        "Alpha formula fit relies on this data; only skip "
                        "for quick uniform-attention-only runs.")
    p.add_argument("--skew-n-factor", type=float, default=2.0,
                   dest="skew_n_factor",
                   help="Geometric factor for the skew n (total decodes) "
                        "axis. 2.0 (default) is doubling; higher coarsens "
                        "and speeds up the sweep.")
    p.add_argument("--skew-pc-factor", type=float, default=2.0,
                   dest="skew_pc_factor",
                   help="Geometric factor for the skew pc (prefill chunk) "
                        "axis. 2.0 (default) is doubling.")
    p.add_argument("--skew-kp-factor", type=float, default=2.0,
                   dest="skew_kp_factor",
                   help="Geometric factor for the skew kp (prefill history) "
                        "axis. 2.0 (default) is doubling.")
    p.add_argument("--skew-kvs-factor", type=float, default=2.0,
                   dest="skew_kvs_factor",
                   help="Geometric factor for the skew kvs (small-decode kv) "
                        "axis. 2.0 (default) is doubling.")
    p.add_argument("--skew-samples-per-cell", type=int, default=32,
                   help="Distribution draws per operating cell (minimum 8).")
    p.add_argument("--skew-rounds", type=int, default=3,
                   help="Independent profile contexts per heterogeneous batch.")
    p.add_argument("--skew-seed", type=int, default=0,
                   help="Deterministic workload-independent sampling seed.")
    p.add_argument("--only-skew", action="store_true", default=False,
                   dest="only_skew",
                   help="Skip the uniform attention/dense/per_seq/moe "
                        "categories and run ONLY the skew step. Use when "
                        "the uniform sweep is already done and you want "
                        "to add (or refresh) skew.csv without redoing "
                        "the rest.")
    p.add_argument("--force", action="store_true", default=False,
                   dest="force",
                   help="Wipe existing CSVs and re-profile from scratch. "
                        "Default is resume mode: existing rows are preserved "
                        "and only shots whose keys aren't already in the CSV "
                        "get fired. Applies to every category plus skew.")

    # Output root.
    p.add_argument(
        "--out-root",
        type=Path,
        default=PERF_DIR,
        help=f"Output root (default: {PERF_DIR}).",
    )

    # Model config root.
    p.add_argument(
        "--model-config-root",
        type=Path,
        default=MODEL_CONFIG_DIR,
        help=f"Directory holding ``<org>/<name>.json`` HF configs. "
             f"Default: {MODEL_CONFIG_DIR}.",
    )

    # Verbosity.
    #
    # Default is INFO — shows TP limits, stage timings, and per-category
    # progress bars.
    #
    #   --silent    → WARNING (warnings + errors only)
    #   --verbose   → DEBUG   (also un-silences vLLM's own stdout/stderr)
    #   --log-level → explicit override of any of the above.
    verbosity = p.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Explicit log level. Default: INFO.",
    )
    verbosity.add_argument(
        "--silent",
        dest="verbose_shortcut",
        action="store_const",
        const=logging.WARNING,
        help="Quiet mode: only WARNING and ERROR.",
    )
    verbosity.add_argument(
        "--verbose",
        dest="verbose_shortcut",
        action="store_const",
        const=logging.DEBUG,
        help="Debug mode: DEBUG level + vLLM stdout.",
    )


def _resolve_log_level(ns: argparse.Namespace) -> int:
    shortcut = getattr(ns, "verbose_shortcut", None)
    if shortcut is not None:
        return shortcut
    return logging.getLevelName(ns.log_level)


def _fetch_hf_config(hf_id: str, target: Path) -> Path:
    """Download ``config.json`` for ``hf_id`` from HuggingFace Hub
    and cache it at ``target``. Returns ``target``.

    Auth: picks up ``HF_TOKEN`` from the environment for gated
    models (Llama etc.). Raises a clear message if the download
    fails for any reason (missing network, wrong id, auth denied).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise FileNotFoundError(
            f"{target} not found locally, and huggingface_hub is not "
            f"installed — cannot auto-download. Install huggingface_hub "
            f"or place the config.json manually."
        ) from e

    token = os.environ.get("HF_TOKEN")
    log.info(
        "Model config not found locally; fetching %s/config.json from "
        "HuggingFace Hub …",
        hf_id,
    )
    try:
        src = hf_hub_download(repo_id=hf_id, filename="config.json", token=token)
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to download config.json for {hf_id!r} from "
            f"HuggingFace Hub: {e}\n"
            f"  • For gated models (Llama, Gemma, ...) set HF_TOKEN to "
            f"a token that has access granted.\n"
            f"  • For offline use, place the config.json at {target}."
        ) from e

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, target)
    log.success("Cached HF config → %s", target)
    return target


def _resolve_model(model: str, root: Path) -> tuple[Path, str]:
    """Locate a model's HF config.json and return (path, hf_id).

    ``model`` is either an HF-style ``<org>/<name>`` identifier
    (resolving to ``<root>/<org>/<name>.json``) or an explicit path
    ending in ``.json``.

    Fallback: when an HF-style id has no local config, the function
    downloads ``config.json`` from HuggingFace Hub and caches it at
    the expected path. Explicit .json paths are never auto-fetched
    (if the user named a specific file and it's missing, that's an
    error).
    """
    candidate = Path(model)
    if candidate.suffix == ".json":
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Explicit config path does not exist: {candidate}. "
                f"Auto-fetch is only available for HF-style ``<org>/<name>`` ids."
            )
        # Derive hf_id from the path's stem under root.
        try:
            relative = candidate.resolve().relative_to(root.resolve())
            hf_id = str(relative.with_suffix(""))
        except ValueError:
            hf_id = candidate.stem
        return candidate.resolve(), hf_id

    # Treat as <org>/<name> under root.
    resolved = (root / f"{model}.json").resolve()
    if not resolved.is_file():
        # Auto-fetch from HF hub.
        resolved = _fetch_hf_config(model, resolved)
    return resolved, model


def _parse_tp(tp_str: str, *, require_tp1: bool = True) -> list[int]:
    tps = [int(x.strip()) for x in tp_str.split(",") if x.strip()]
    if not tps:
        raise ValueError("--tp must contain at least one value")
    if any(tp < 1 for tp in tps):
        raise ValueError("--tp values must be positive integers")
    if require_tp1 and 1 not in tps:
        raise ValueError("--tp must include 1")
    return tps


def _build_profile_args(
    ns: argparse.Namespace,
    hf_id: str,
    architecture: str,
    model_config: dict,
) -> ProfileArgs:
    if not math.isfinite(ns.grid_factor) or ns.grid_factor <= 1:
        raise ValueError("--grid-factor must be finite and greater than 1")
    if not math.isfinite(ns.moe_grid_factor) or ns.moe_grid_factor <= 1:
        raise ValueError("--moe-grid-factor must be finite and greater than 1")
    degrees = {}
    for name in ("dp", "ep"):
        raw = getattr(ns, f"{name}_degrees", None)
        if raw is None:
            degrees[name] = None
            continue
        try:
            values = tuple(sorted({int(value.strip()) for value in raw.split(",")}))
        except ValueError:
            raise ValueError(f"--{name} requires comma-separated positive integers") from None
        if not values or min(values) < 1:
            raise ValueError(f"--{name} degrees must be positive integers")
        degrees[name] = values
    if any(value is not None for value in degrees.values()):
        if ns.cmd != "profile" and not (ns.cmd == "slice" and ns.group == "moe"):
            raise ValueError("--dp and --ep are only supported for profile or slice --group moe")
        if getattr(ns, "only_skew", False) or getattr(ns, "profile_mtp", False):
            raise ValueError("Native MoE profiling cannot be combined with skew-only or MTP")
        if ns.moe_rounds < 1 or ns.measurement_iterations < 1:
            raise ValueError("Native MoE profiling requires positive repeat counts")
    return ProfileArgs(
        architecture=architecture,
        model=hf_id,
        hardware=ns.hardware,
        tp_degrees=_parse_tp(ns.tp, require_tp1=not (
            getattr(ns, "cmd", None) == "plan-skew" or
            (getattr(ns, "cmd", None) == "profile" and getattr(ns, "only_skew", False)))),
        variant=ns.variant,
        dtype=ns.dtype,
        kv_cache_dtype=ns.kv_cache_dtype,
        max_num_batched_tokens=ns.max_num_batched_tokens,
        max_num_seqs=ns.max_num_seqs,
        block_size=getattr(ns, "block_size", None),
        gpu_memory_utilization=getattr(ns, "gpu_memory_utilization", None),
        max_model_len=getattr(ns, "max_model_len", None),
        num_hidden_layers=getattr(ns, "num_hidden_layers", None),
        profile_mtp=bool(getattr(ns, "profile_mtp", False)),
        dp_degrees=degrees["dp"],
        ep_degrees=degrees["ep"],
        moe_rounds=ns.moe_rounds,
        moe_grid_factor=ns.moe_grid_factor,
        grid_factor=ns.grid_factor,
        cudagraph=ns.cudagraph,
        linear_attn_chunk=getattr(ns, "linear_attn_chunk", None),
        attention_max_kv=ns.attention_max_kv,
        attention_decode_q_lens=tuple(
            int(v) for v in str(ns.attention_decode_q_lens).split(",") if v.strip()
        ) or (1,),
        attention_chunk_factor=ns.attention_chunk_factor,
        attention_kv_factor=ns.attention_kv_factor,
        attention_n_factor=getattr(ns, "attention_n_factor", SQRT2),
        measurement_iterations=ns.measurement_iterations,
        skip_skew=getattr(ns, "skip_skew", False),
        skew_n_factor=getattr(ns, "skew_n_factor", 2.0),
        skew_pc_factor=getattr(ns, "skew_pc_factor", 2.0),
        skew_kp_factor=getattr(ns, "skew_kp_factor", 2.0),
        skew_kvs_factor=getattr(ns, "skew_kvs_factor", 2.0),
        skew_samples_per_cell=getattr(ns, "skew_samples_per_cell", 32),
        skew_rounds=getattr(ns, "skew_rounds", 3),
        skew_seed=getattr(ns, "skew_seed", 0),
        only_skew=getattr(ns, "only_skew", False),
        force=getattr(ns, "force", False),
        hf_overrides=_parse_hf_overrides(getattr(ns, "hf_override", None)),
        model_config=model_config,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_hf_overrides(pairs: list[str] | None) -> dict[str, Any] | None:
    """Turn repeated ``KEY=VALUE`` arguments into a model-config override dict.

    Values go through ``json.loads`` first so numbers, booleans, nulls and
    lists arrive as the types a model config would hold, and fall back to the
    raw string when that fails — ``bfloat16`` is not valid JSON but is a
    perfectly good ``torch_dtype``.

    Dotted keys nest, so ``--hf-override text_config.num_experts=8`` reaches
    into a wrapped config the way the file itself is shaped.
    """
    if not pairs:
        return None
    out: dict[str, Any] = {}
    for raw in pairs:
        if "=" not in raw:
            raise SystemExit(
                f"--hf-override expects KEY=VALUE, got {raw!r}"
            )
        key, _, value = raw.partition("=")
        key = key.strip()
        if not key:
            raise SystemExit(f"--hf-override has an empty key: {raw!r}")
        try:
            parsed: Any = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        target = out
        *parents, leaf = key.split(".")
        for part in parents:
            nxt = target.setdefault(part, {})
            if not isinstance(nxt, dict):
                raise SystemExit(
                    f"--hf-override {key!r} conflicts with an earlier "
                    f"non-mapping value at {part!r}"
                )
            target = nxt
        target[leaf] = parsed
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m profiler",
        description="Layerwise profiler for LLMServingSim.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_execution = sub.add_parser('import-execution',
        help='Import observed serving graph settings from a benchmark; no GPU.')
    p_execution.add_argument('model', help='HF model ID with a local config and existing bundle.')
    p_execution.add_argument('--hardware', required=True)
    p_execution.add_argument('--variant', default=None)
    p_execution.add_argument('--bench-dir', required=True)
    p_execution.add_argument('--out', default='profiler/perf', dest='out_root')
    p_execution.add_argument('--log-level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    p_refit = sub.add_parser("refit-skew", help="Rebuild skew calibration from measured CSVs without a GPU.")
    p_refit.add_argument("model", help="HF model id with a local config and profile bundle.")
    p_refit.add_argument("--hardware", required=True)
    p_refit.add_argument("--variant", default=None)
    p_refit.add_argument("--tp", default=None, help="Comma-separated TP degrees; default: every measured TP.")
    p_refit.add_argument("--out", default="profiler/perf", dest="out_root")
    p_refit.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # ---- profile ----
    p_profile = sub.add_parser(
        "profile",
        help="Full sweep (every TP × every category).",
    )
    p_profile.add_argument(
        "model",
        help="HF model id (e.g. meta-llama/Llama-3.1-8B). Must match a "
             "config file under configs/model/.",
    )
    _add_common_flags(p_profile)

    p_plan = sub.add_parser("plan-skew", help="Preview skew coverage from saved engine limits; no GPU.")
    p_plan.add_argument("model")
    _add_common_flags(p_plan)

    # ---- slice ----
    p_slice = sub.add_parser(
        "slice",
        help="Refresh one (tp, category) pair.",
    )
    p_slice.add_argument(
        "model",
        help="HF model id.",
    )
    p_slice.add_argument(
        "--tp-refresh", type=int, required=True, dest="tp_refresh",
        help="TP degree to refresh.",
    )
    p_slice.add_argument(
        "--group",
        choices=["dense", "per_sequence", "attention", "linear_attention",
                 "moe", "mtp"],
        required=True,
        help="Which profile category to refresh; mtp requires --profile-mtp.",
    )
    _add_common_flags(p_slice)

    # ---- coverage ----
    p_coverage = sub.add_parser(
        "coverage",
        help="Report how much CUDA time the catalog binds (writes nothing).",
    )
    p_coverage.add_argument(
        "model",
        help="HF model id.",
    )
    _add_common_flags(p_coverage)

    p_hardware = sub.add_parser(
        "hardware",
        help="Measure this machine's interconnect and record the card's spec "
             "(writes profiler/perf/<hw>/hardware.yaml). Needs two GPUs for "
             "the link; exits non-zero when it cannot measure it.",
    )
    p_hardware.add_argument(
        "--hardware", required=True, dest="hardware",
        help="Hardware label -- the folder name under profiler/perf/.",
    )
    p_hardware.add_argument(
        "--npus", type=int, default=2, dest="hw_npus",
        help="How many GPUs to benchmark collectives across (default 2). Recorded, "
             "because a config asking for more is extrapolating: eight cards "
             "over NVLink are not two over PCIe.",
    )
    p_hardware.add_argument(
        "--out", default="profiler/perf", dest="out_root",
        help="Output root (default profiler/perf).",
    )
    p_hardware.add_argument(
        "--log-level", default="INFO", dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger verbosity (default: INFO).",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)

    log.configure(_resolve_log_level(ns))

    if ns.cmd == 'import-execution':
        from profiler.core.execution_metadata import import_execution
        from serving.core.trace_generator import resolve_variant
        from serving.core.utils import get_config
        config = json.loads((MODEL_CONFIG_DIR / (ns.model + '.json')).read_text())
        root = Path(ns.out_root) / ns.hardware / ns.model / (ns.variant or resolve_variant(get_config(ns.model)))
        record = import_execution(ns.bench_dir, root, config)
        log.info('Recorded serving graph policy: runner=%s, mode=%s, %d capture sizes',
                 record['graph']['runner'], record['graph']['mode'],
                 len(record['graph']['capture_sizes']))
        return 0

    if ns.cmd == "refit-skew":
        from profiler.core.skew_calibration import rebuild_bundle
        from serving.core.trace_generator import resolve_variant
        config = json.loads((Path(__file__).resolve().parents[1]/"configs/model"/
                             (ns.model+".json")).read_text())
        variant = ns.variant or resolve_variant(config)
        root = Path(ns.out_root)/ns.hardware/ns.model/variant
        with (root/"meta.yaml").open() as stream:
            import yaml
            meta = yaml.safe_load(stream)
        if any(meta.get(k) != v for k, v in
               (("model", ns.model), ("hardware", ns.hardware), ("variant", variant))):
            raise ValueError("Requested profile identity does not match meta.yaml")
        tps = None if ns.tp is None else [int(v) for v in ns.tp.split(",")]
        if tps is not None and (not tps or any(tp < 1 for tp in tps)):
            raise ValueError("TP degrees must be positive integers")
        fit = rebuild_bundle(root, config, tps)
        log.info("Rebuilt skew calibration for TP degrees %s", sorted(fit["per_tp"]))
        return 0

    # `hardware` characterises the machine, not a model, so it runs before any
    # of the model-config resolution below -- it takes no model argument.
    if ns.cmd == "hardware":
        from profiler.core.hardware import run_hardware

        _, measured = run_hardware(ns.hardware, Path(ns.out_root),
                                   npus=ns.hw_npus)
        return 0 if measured else 1

    # 1. Locate the model's HF config.json.
    model_config_path, hf_id = _resolve_model(ns.model, ns.model_config_root)

    # 2. Read the full model config — it becomes vLLM's config.json
    # via a temp directory at engine spin-up (no HF hub round-trip).
    model_config = read_model_config(model_config_path)
    model_type = detect_model_type(model_config_path)

    # 3. Resolve the matching architecture yaml.
    arch_path = resolve_architecture_by_model_type(model_type, ARCH_DIR)

    log.info(
        "Model config: %s (model_type=%s → architecture=%s)",
        model_config_path, model_type, arch_path.stem,
    )
    log.debug("Model config fields: %s", sorted(model_config.keys()))

    # 4. Build per-session ProfileArgs.
    profile_args = _build_profile_args(
        ns, hf_id,
        architecture=arch_path.stem,
        model_config=model_config,
    )

    # 5. Dispatch.
    if ns.cmd == "plan-skew":
        from profiler.core.skew_support import complete_plan
        from profiler.core.skew import _existing_keys, recorded_contract_for_skew
        from profiler.core.component_layouts import selected_catalog_for_tp
        from profiler.core.config import load_architecture
        from profiler.core.categories import _entry_dict
        from profiler.core.hooks.capture_scope_contract import contexts_from_architecture
        from types import SimpleNamespace
        from serving.core.trace_generator import resolve_variant
        import dataclasses
        import yaml
        variant = profile_args.variant or resolve_variant(model_config)
        root = Path(ns.out_root) / profile_args.hardware / hf_id / variant
        metadata = yaml.safe_load((root / "meta.yaml").read_text())
        effective = metadata["engine_effective"]
        resolved = metadata["engine_resolved"]["per_tp"]
        plans = {}
        architecture = load_architecture(arch_path)
        scope_contexts = contexts_from_architecture(architecture)
        for tp in profile_args.tp_degrees:
            saved = resolved.get(str(tp)) or resolved.get(tp)
            if not saved:
                raise ValueError(f"Missing saved engine limits for TP={tp}; profile first")
            limits = SimpleNamespace(**{"max_num_batched_tokens": effective["max_num_batched_tokens"],
                "max_num_seqs": effective["max_num_seqs"], **saved})
            planned = dataclasses.replace(profile_args,
                attention_max_kv=profile_args.attention_max_kv or
                                 metadata.get("attention_grid", {}).get("max_kv") or limits.max_model_len)
            from profiler.core.attention_lookup import semantic_for_table
            lookup_record, _ = semantic_for_table(root / f"tp{tp}" / "attention.csv",
                model_config, architecture.model_dump(exclude_none=True))
            attention_sha = lookup_record['semantic_sha256'] if lookup_record else None
            choices, entries = selected_catalog_for_tp(root / f"tp{tp}",
                architecture.model_dump(exclude_none=True), "attention", architecture.catalog.attention)
            catalog = _entry_dict(entries, architecture)
            layout_contract = recorded_contract_for_skew(root / f"tp{tp}", architecture,
                choices, planned, logical_tp=tp, logical_ep=1, block_size=limits.block_size)
            completed = set() if planned.force else _existing_keys(root / f"tp{tp}" / "skew.csv", catalog,
                                       planned.skew_rounds, planned.measurement_iterations,
                                       block_size=limits.block_size, scope_contexts=scope_contexts,
                                       attention_lookup_sha256=attention_sha,
                                       component_layout_contract=layout_contract)
            plans[tp], _ = complete_plan(planned, limits, root / f"tp{tp}" / "attention.csv",
                architecture, completed, existing_csv=None if planned.force else root / f"tp{tp}" / "skew.csv")
        print(json.dumps(plans, indent=2))
        return 0

    from profiler.core.runner import run_coverage, run_full, run_slice
    if ns.cmd == "profile":
        # --profile-mtp needs no restriction here: run_full boots the drafter
        # in a second engine after the main pass, with only the `mtp` category,
        # so the two never share a profile tree.
        run_full(arch_path, profile_args, ns.out_root)
    elif ns.cmd == "slice":
        if profile_args.profile_mtp and ns.group != "mtp":
            parser.error(
                f"--profile-mtp with --group {ns.group} would record the sum "
                f"of the target's layers and the drafter's replayed copy of "
                f"them. Only --group mtp is scoped to the drafter."
            )
        run_slice(
            arch_path,
            profile_args,
            tp=ns.tp_refresh,
            group=ns.group,
            out_root=ns.out_root,
        )
    elif ns.cmd == "coverage":
        reports = run_coverage(arch_path, profile_args)
        # Both are catalog defects: a gap is CUDA time the simulator will never
        # see, an over-match is a number that is the sum of two roles.
        if any(r.gaps or r.over_matches for r in reports.values()):
            return 1
    else:  # pragma: no cover
        parser.error(f"unknown command: {ns.cmd!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
