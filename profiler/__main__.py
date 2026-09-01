"""CLI entry point: ``python -m profiler ...``.

Three subcommands:

    profile <model> --hardware <hw> [options]
        Full sweep: every TP × every category.

    slice <model> --hardware <hw> --tp-refresh N --group G [options]
        Refresh one (tp, category) pair.

    coverage <model> [options]
        Boot once and report how much of the model's CUDA time the
        architecture catalog binds. Writes nothing. Run this first when
        adding a model family: a catalog entry can name a real class and
        still measure nothing, because vLLM's profile tree only holds
        modules that launch a kernel of their own, and reading the module
        tree cannot tell you which those are.

Model resolution
----------------
``<model>`` is a path (HF-style ``<org>/<name>``) under the project's
``configs/model/`` directory. The profiler:

1. Reads ``configs/model/<model>.json`` (a raw HuggingFace config.json).
2. Extracts ``model_type``.
3. Looks up ``profiler/models/<model_type>.yaml`` as the
   architecture. Errors out if none is found.
4. Uses ``<model>`` verbatim as the ``vllm.LLM(model=...)`` argument
   (HF id; vLLM downloads tokenizer + auxiliary files from the hub).

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
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from profiler.core import logger as log
from profiler.core.config import (
    ProfileArgs,
    detect_model_type,
    read_model_config,
    resolve_architecture_by_model_type,
)
from profiler.core.runner import run_coverage, run_full, run_slice


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

def _add_common_flags(p: argparse.ArgumentParser) -> None:
    """Flags shared between profile and slice subcommands."""
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
             "Must include 1. Default: '1'.",
    )
    p.add_argument(
        "--variant",
        default=None,
        help="Output folder label. Default: auto-derived from dtype "
             "and kv_cache_dtype (e.g. 'bfloat16' or 'bfloat16-kvfp8').",
    )

    # Engine kwargs.
    p.add_argument("--dtype", default=None,
                   help="Model weight dtype (bfloat16/float16/float32/fp8). "
                        "Default: vLLM default.")
    p.add_argument("--kv-cache-dtype", default=None,
                   help="KV cache dtype (auto/fp8/fp16/bf16). Default: auto.")
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
                        "``--block-size``. Default: 16, matching the "
                        "simulator's own --block-size; the two should agree, "
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
                   help="Layers to instantiate. Default: 1, which is right "
                        "for a uniform stack where every block is identical. "
                        "A hybrid stack needs the smallest count that reaches "
                        "every distinct block type (4 for Qwen3.8-27B, whose "
                        "layer_types runs gated-DeltaNet x3 then full "
                        "attention) or the catalog only ever sees one.")
    p.add_argument("--profile-mtp", type=int, default=None,
                   dest="profile_mtp", metavar="N",
                   help="Boot with speculative decoding at N draft tokens so "
                        "vLLM also builds the model's own MTP module, and its "
                        "kernels land in the profile tree. Needed because the "
                        "MTP config model_type (deepseek_mtp / qwen3_5_mtp / "
                        "minimax_m3_mtp) is produced by vLLM's "
                        "SpeculativeConfig and is unknown to HF Transformers, "
                        "so the module cannot be loaded on its own. Only "
                        "models declaring MTP modules support this. Pair it "
                        "with the coverage subcommand first: the drafter's "
                        "kernels show up as unbound until a catalog binds "
                        "them.")
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

    # Attention grid.
    p.add_argument("--attention-max-kv", type=int, default=16384,
                   help="Cap for the kv_prefill / kv_decode axes. The "
                        "grid grows geometrically from 512 up to "
                        "min(this, max_model_len). Default: 16384.")
    p.add_argument("--attention-decode-q-lens", type=str, default="1",
                   dest="attention_decode_q_lens",
                   help="Comma-separated query-token counts per decode sequence "
                        "to sweep, e.g. '1,2,4,6'. Default '1' is ordinary "
                        "decoding. A speculative-decoding verification step "
                        "submits 1 + num_speculative_tokens queries per "
                        "sequence against that sequence's own KV, which is "
                        "neither a prefill chunk of the same size nor that many "
                        "single-token decodes. Opt-in because it multiplies the "
                        "attention grid.")
    p.add_argument("--attention-chunk-factor", type=float, default=2.0,
                   dest="attention_chunk_factor",
                   help="Geometric factor for the prefill_chunk axis. "
                        "2.0 (default) is doubling. Lower for denser grid.")
    p.add_argument("--attention-kv-factor", type=float, default=2.0,
                   dest="attention_kv_factor",
                   help="Geometric factor for the kv_prefill / kv_decode "
                        "axes. 2.0 (default) is doubling.")
    p.add_argument("--measurement-iterations", type=int, default=3,
                   dest="measurement_iterations",
                   help="Timed forwards per shot (averaged). A single sample "
                        "can swing 15-25%% on large GEMMs due to DVFS / clock "
                        "jitter; N=3 (default) cuts that to ~5%% at ~3x "
                        "profile time.")
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


def _parse_tp(tp_str: str) -> list[int]:
    tps = [int(x.strip()) for x in tp_str.split(",") if x.strip()]
    if not tps:
        raise ValueError("--tp must contain at least one value")
    if 1 not in tps:
        raise ValueError("--tp must include 1")
    return tps


def _build_profile_args(
    ns: argparse.Namespace,
    hf_id: str,
    architecture: str,
    model_config: dict,
) -> ProfileArgs:
    return ProfileArgs(
        architecture=architecture,
        model=hf_id,
        hardware=ns.hardware,
        tp_degrees=_parse_tp(ns.tp),
        variant=ns.variant,
        dtype=ns.dtype,
        kv_cache_dtype=ns.kv_cache_dtype,
        max_num_batched_tokens=ns.max_num_batched_tokens,
        max_num_seqs=ns.max_num_seqs,
        block_size=getattr(ns, "block_size", None),
        gpu_memory_utilization=getattr(ns, "gpu_memory_utilization", None),
        max_model_len=getattr(ns, "max_model_len", None),
        num_hidden_layers=getattr(ns, "num_hidden_layers", None),
        profile_mtp=getattr(ns, "profile_mtp", None),
        linear_attn_chunk=getattr(ns, "linear_attn_chunk", None),
        attention_max_kv=ns.attention_max_kv,
        attention_decode_q_lens=tuple(
            int(v) for v in str(ns.attention_decode_q_lens).split(",") if v.strip()
        ) or (1,),
        attention_chunk_factor=ns.attention_chunk_factor,
        attention_kv_factor=ns.attention_kv_factor,
        measurement_iterations=ns.measurement_iterations,
        skip_skew=getattr(ns, "skip_skew", False),
        skew_n_factor=getattr(ns, "skew_n_factor", 2.0),
        skew_pc_factor=getattr(ns, "skew_pc_factor", 2.0),
        skew_kp_factor=getattr(ns, "skew_kp_factor", 2.0),
        skew_kvs_factor=getattr(ns, "skew_kvs_factor", 2.0),
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
        help="Which profile category to refresh.",
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

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)

    log.configure(_resolve_log_level(ns))

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
    if ns.cmd == "profile":
        run_full(arch_path, profile_args, ns.out_root)
    elif ns.cmd == "slice":
        run_slice(
            arch_path,
            profile_args,
            tp=ns.tp_refresh,
            group=ns.group,
            out_root=ns.out_root,
        )
    elif ns.cmd == "coverage":
        reports = run_coverage(arch_path, profile_args)
        if any(r.gaps for r in reports.values()):
            return 1
    else:  # pragma: no cover
        parser.error(f"unknown command: {ns.cmd!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
