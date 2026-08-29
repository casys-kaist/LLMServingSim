import os
import sys
from .request import *
from .utils import *
from .utils import (
    _load_architecture, _stack_module, get_architecture, get_layer_stack,
    num_experts as utils_num_experts,
)
import pandas as pd
import yaml
from .memory_model import calculate_sizes
from .gate_function import GateRouter
from .config_builder import get_device
from .power_model import PowerModel, total_ring_data
from .pim_model import PIMModel
from .logger import get_logger
from .run_paths import input_path
import bisect
from dataclasses import dataclass, field

# ----------------------------------------------------------------------
# Global in-memory cache for the profiler's per-category performance DB.
# key: (hardware, model, variant)
# value: dict with keys {meta, architecture, layer_stack, tables}
# ----------------------------------------------------------------------
_perf_db_cache = {}

logger = get_logger("TraceGenerator")


# ----------------------------------------------------------------------
# Profile-data paths + variant resolution (mirrors the profiler).
# ----------------------------------------------------------------------

_PROFILER_ROOT_REL = "../profiler"

_DTYPE_SHORT = {
    "bfloat16": "bf16", "bf16": "bf16",
    "float16": "fp16", "half": "fp16", "fp16": "fp16",
    "float32": "fp32", "float": "fp32", "fp32": "fp32",
    "fp8": "fp8", "fp8_e4m3": "fp8",
    "int8": "int8", "int4": "int4",
}

# TP collective hooks keyed on canonical layer name. Applied after the
# named layer when tp_size > 1. Names must match the profiler's catalog.
_TP_ALLREDUCE_AFTER = frozenset({"o_proj", "down_proj"})


def _short_dtype(d):
    if d is None:
        return None
    return _DTYPE_SHORT.get(str(d), str(d))


def resolve_variant(dtype, kv_cache_dtype, model_config=None):
    """Compute the profiler's variant folder name from runtime dtype
    choices. Matches ``ProfileArgs.effective_variant`` in the profiler.
    """
    weight = dtype
    if not weight and model_config is not None:
        # Must match the profiler's ``config.model_config_weight_dtype``
        # exactly, including the order, or we look in a variant folder the
        # profiler never wrote.
        #
        # A ``quantization_config`` wins: for a quantized checkpoint the dtype
        # fields describe the *activation* dtype, not the weights.
        # DeepSeek-V3.2-Exp is FP8 block-quantized with
        # ``torch_dtype: bfloat16``. And HuggingFace renamed the dtype field
        # itself -- ``torch_dtype`` is legacy, ``dtype`` current (Qwen3.8
        # carries only the latter) -- so both are accepted, legacy first.
        quant = model_config.get("quantization_config")
        weight = (quant or {}).get("quant_method") if isinstance(quant, dict) else None
        weight = weight or model_config.get("torch_dtype") or model_config.get("dtype")
    parts = [_short_dtype(weight) if weight else "default"]
    if kv_cache_dtype and kv_cache_dtype != "auto":
        parts.append(f"kv{_short_dtype(kv_cache_dtype)}")
    return "-".join(parts)


def _variant_root(hardware, model, variant):
    return f"{_PROFILER_ROOT_REL}/perf/{hardware}/{model}/{variant}"


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class TraceCtx:
    """Immutable context for an entire trace generation."""
    hardware: str
    model: str
    config: dict
    perf_db: dict
    node_id: int
    fp: int
    placement: dict
    gate: object  # GateRouter or None
    enable_attn_offloading: bool
    power_model: object  # PowerModel or None
    pim_model: object  # PIMModel or None
    pim_channels: int
    n_head: int
    kv_head: int
    head_dim: int
    is_moe: bool
    kv_fp: int    # bytes per KV element (1 for fp8, else fp) -- P/D transfer sizing
    pd_type: str  # 'prefill', 'decode', or None
    tp_size: int       # tensor parallel degree (for ALLREDUCE on attention/FFN)
    pp_size: int       # pipeline parallel degree
    local_ep: int      # expert parallel degree within this instance
    ep_total: int      # total EP degree across DP group
    tp_dim: list       # involved_dim for TP collectives (ALLREDUCE), None = all dims
    ep_dim: list       # involved_dim for EP collectives (ALLTOALL), None = all dims
    dp_sum_total_len: int  # sum of total_len across DP group (0 = DP inactive). Captures the post-AG gathered size for MoE compute; dummy batches are pre-padded to max by serving/__main__.py so the sum reflects vLLM's CUDA-graph padding.


@dataclass
class BatchCtx:
    """Per-batch state computed from a Batch object."""
    batch: object  # Batch
    total_len: int
    prefill_chunk: int  # sum(prefill_q_list): new prefill tokens this step
    kv_prefill: int     # sum(prefill_k_list): existing kv history for prefill reqs
    n_decode: int       # number of decode requests
    kv_decode_mean: int # mean decode kv length (4D grid carries one value)
    kv_decode_max: int  # max decode kv length (for skew correction)
    kv_decode_min: int  # min decode kv length (for skew_rate in skew correction)
    lm_head_len: int    # number of sequences
    decode_lens: list   # per-PIM-channel decode lengths (None if no PIM)
    channel_split: int  # PIM channel split factor


@dataclass
class PowerAccumulator:
    """Accumulates power data for a block, then flushes to power_model."""
    npu_latencies_ns: list
    pim_latencies_ns: list
    dram_weight_bytes: int
    link_data_bytes: int

    def flush(self, ctx, enable_attn_offloading=False):
        if ctx.power_model is None:
            return
        ctx.power_model.add_dram_energy_consumption(ctx.node_id, self.dram_weight_bytes)
        ctx.power_model.add_link_energy_consumption(ctx.node_id, self.link_data_bytes)
        for lat in self.npu_latencies_ns:
            ctx.power_model.add_npu_active_energy_consumption(ctx.hardware, ctx.node_id, lat, num_npus=ctx.tp_size)
        if enable_attn_offloading:
            for lat in self.pim_latencies_ns:
                ctx.power_model.add_pim_active_energy_consumption(ctx.node_id, lat)


# ======================================================================
# Perf DB loading and lookup (new per-category format)
# ======================================================================
#
# New layout under profiler/perf/<hw>/<model>/<variant>/:
#     meta.yaml                       profiler settings, effective engine kwargs
#     tp<N>/dense.csv                 layer, tokens, time_us
#     tp<N>/per_sequence.csv          layer, sequences, time_us
#     tp<N>/attention.csv             prefill_chunk, kv_prefill, n_decode, kv_decode, time_us
#     tp<N>/moe.csv                   tokens, activated_experts, time_us    (MoE only)
#
def _load_meta(variant_root):
    path = os.path.join(variant_root, "meta.yaml")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"meta.yaml missing at {path}. Re-run the profiler to produce it."
        )
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _hydrate_skew_fit_tables(meta, variant_root):
    """Load each TP's per-bucket alpha table from CSV into the meta dict.

    Newer profile runs move the (1k+ rows per TP) ``alpha_by_bucket``
    mapping out of meta.yaml into ``tp{N}/skew_fit.csv``. This helper
    reads those CSVs and materialises the dict in-place so
    ``_skew_alpha`` finds it where it used to be. Older meta.yamls
    that still inline the dict are left untouched.
    """
    fit = (meta or {}).get("skew_fit") if isinstance(meta, dict) else None
    if not fit or not fit.get("enabled"):
        return
    per_tp = fit.get("per_tp")
    if not isinstance(per_tp, dict):
        return
    for tp_key, entry in per_tp.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("alpha_by_bucket"):
            continue
        rel = entry.get("bucket_table")
        if not rel:
            continue
        csv_path = os.path.join(variant_root, rel)
        if not os.path.isfile(csv_path):
            logger.warning(
                "skew_fit: tp=%s bucket_table %s missing — falling back to "
                "alpha_default", tp_key, csv_path,
            )
            continue
        alphas, counts = _read_skew_fit_csv(csv_path)
        entry["alpha_by_bucket"] = alphas
        entry["n_by_bucket"] = counts


def _read_skew_fit_csv(path):
    """Return (alpha_by_bucket, n_by_bucket) keyed by the pipe-delimited
    bucket string used by ``_skew_alpha``.

    A ``layer`` column is prefixed onto the key, naming the attention kernel
    the alpha was fitted on. A CSV written before that column existed holds one
    kernel, and its keys stay unprefixed — which is what ``_skew_alpha`` falls
    back to for ``attention`` and only for ``attention``.
    """
    df = pd.read_csv(path)
    has_layer = "layer" in df.columns
    alphas: dict = {}
    counts: dict = {}
    for row in df.itertuples(index=False):
        raw = getattr(row, "raw_key", None)
        if isinstance(raw, str) and raw:
            key = raw
        else:
            key = (
                f"pc={int(row.pc)}|{row.n_label}|{row.skew_rate_label}"
                f"|{row.kv_big_label}|{row.kp_label}"
            )
            if has_layer:
                key = f"{row.layer}|{key}"
        alphas[key] = float(row.alpha)
        if hasattr(row, "n_samples"):
            try:
                counts[key] = int(row.n_samples)
            except (TypeError, ValueError):
                pass
    return alphas, counts


def _read_category_csv(path, key_cols):
    """Read a category CSV and return a dict per layer (for dense/per_sequence)
    or a list of rows (for attention/moe).

    key_cols: for dense/per_sequence this is ["tokens"] or ["sequences"];
              for attention/moe pass None to return raw rows.
    """
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, sep=",")
    # time_us -> latency_ns (int, min 1)
    df["latency_ns"] = (df["time_us"].astype(float) * 1_000.0).round().astype(int).clip(lower=1)
    return df


def _build_1d_table(df, layer_col, key_col):
    """Dense / per-sequence: per-layer sorted (keys, values) table.

    Built from column lists in one pass rather than through
    ``groupby``/``sort_values``/``drop_duplicates``. pandas spends roughly
    37 us per row on that path, which put the attention table alone at
    ~693 ms per ``tp<N>/`` folder; the same construction in plain Python
    is ~9 ms and yields an identical structure.

    Duplicate keys resolve last-wins. That is deterministic, where the
    pandas path was not: ``sort_values`` defaults to an unstable
    quicksort, so which of two rows sharing a key survived
    ``drop_duplicates`` was unspecified. No bundle under
    ``profiler/perf/`` currently carries a duplicate key in any category,
    so this changes no existing lookup; for a CSV that gained rows from a
    partial re-profile, last-wins takes the newer measurement.
    """
    grouped = {}
    for layer, key, val in zip(df[layer_col].tolist(),
                               df[key_col].astype(int).tolist(),
                               df["latency_ns"].astype(int).tolist()):
        grouped.setdefault(str(layer), {})[key] = val
    out = {}
    for layer, by_key in grouped.items():
        keys = sorted(by_key)
        out[layer] = {"keys": keys, "values": [by_key[k] for k in keys]}
    return out


def _build_attention_table(df):
    """4D attention table indexed by (prefill_chunk, n_decode) slices,
    each slice a 2D grid over (kv_prefill, kv_decode). The profiler
    sweeps all four axes on doubling grids, so the lookup interpolates
    in log-space on each axis (plus a zero-pinned fallback when the
    axis value is 0, which always comes from an exact sample).
    """
    pc_col = df["prefill_chunk"].astype(int).tolist()
    nd_col = df["n_decode"].astype(int).tolist()
    kp_col = df["kv_prefill"].astype(int).tolist()
    kd_col = df["kv_decode"].astype(int).tolist()
    lat_col = df["latency_ns"].astype(int).tolist()

    # (prefill_chunk, n_decode) -> kv_prefill -> kv_decode -> latency_ns.
    # One pass in plain Python; see _build_1d_table for why not groupby.
    grouped = {}
    for pc, nd, kp, kd, lat in zip(pc_col, nd_col, kp_col, kd_col, lat_col):
        grouped.setdefault((pc, nd), {}).setdefault(kp, {})[kd] = lat

    slices = {}
    for key, by_kp in grouped.items():
        kp_vals_s = sorted(by_kp)
        rows = []
        for kp in kp_vals_s:
            by_kd = by_kp[kp]
            kd_keys = sorted(by_kd)
            rows.append({"keys": kd_keys, "values": [by_kd[k] for k in kd_keys]})
        slices[key] = {"kv_prefill_vals": kp_vals_s, "rows": rows}

    return {
        "pc_vals": sorted(set(pc_col)), "nd_vals": sorted(set(nd_col)),
        "pc_nd_pairs": sorted(slices.keys()),
        "slices": slices,
    }


def _build_attention_tables_by_layer(df):
    """``{layer_name: attention_table}``, one per kernel in the profile.

    A bundle profiled before the attention CSV grew a ``layer`` column has
    exactly one kernel in it, so every row belongs to ``attention`` and the
    resulting table is identical to what ``_build_attention_table`` returned
    for the whole frame. That is the invariant this function has to hold:
    every committed bundle must come out byte-identical.

    Newer bundles name the kernel per row, because a sparse-attention model
    runs an indexer over the whole KV before its top-k selection. It keys on
    the same four axes as the attention kernel and runs on the same layers,
    but it is different work -- merging the two gave a value describing
    neither.
    """
    if "layer" not in df.columns:
        return {"attention": _build_attention_table(df)}
    out = {}
    for layer in df["layer"].astype(str).unique().tolist():
        out[layer] = _build_attention_table(df[df["layer"].astype(str) == layer])
    return out


def _build_moe_table(df):
    """MoE table: (tokens, activated_experts) → latency_ns."""
    grouped = {}
    for ae, tok, lat in zip(df["activated_experts"].astype(int).tolist(),
                            df["tokens"].astype(int).tolist(),
                            df["latency_ns"].astype(int).tolist()):
        grouped.setdefault(ae, {})[tok] = lat
    ae_vals = sorted(grouped)
    rows = []
    for ae in ae_vals:
        by_tok = grouped[ae]
        tok_keys = sorted(by_tok)
        rows.append({"keys": tok_keys, "values": [by_tok[k] for k in tok_keys]})
    return {"activated_experts_vals": ae_vals, "rows": rows}


def _load_perf_db(hardware, model, variant, tp_needed, model_type,
                  model_config=None):
    """Load the per-category perf DB for a (hardware, model, variant)
    tuple and cache it. ``tp_needed`` is a set of int TP degrees the
    simulator will query; each must have its own ``tp<N>/`` folder.

    ``model_config`` is resolved once into a per-layer block list
    (``layer_stack``) and cached with the rest, because it answers a
    per-model question -- which block each decoder layer runs -- that the
    emit path asks once per layer per iteration.
    """
    cache_key = (hardware, model, variant)
    if cache_key in _perf_db_cache:
        db = _perf_db_cache[cache_key]
        _check_tp_coverage(db, tp_needed, hardware, model, variant)
        return db

    root = _variant_root(hardware, model, variant)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Profile variant folder not found: {root}. Run the profiler "
            f"with matching --dtype / --kv-cache-dtype, or pick an existing "
            f"variant under {os.path.dirname(root)}."
        )

    meta = _load_meta(root)
    _hydrate_skew_fit_tables(meta, root)
    arch = _load_architecture(model_type)
    available_tps = []
    for entry in sorted(os.listdir(root)):
        if not entry.startswith("tp"):
            continue
        try:
            available_tps.append(int(entry[2:]))
        except ValueError:
            continue

    perf_db = {
        "meta": meta,
        "architecture": arch,
        "variant": variant,
        "hardware": hardware,
        "model": model,
        "root": root,
        "available_tps": sorted(available_tps),
        # Per-TP category tables, filled in by _tp_tables on first lookup.
        "tables": {},
        # One LayerSpec per decoder layer, from the checkpoint's own config.
        # Empty only when the config declares no num_hidden_layers, which the
        # simulator has already failed on by the time it gets here.
        "layer_stack": (_stack_module().resolve_stack(model_config)
                        if model_config else []),
    }
    _perf_db_cache[cache_key] = perf_db
    _check_tp_coverage(perf_db, tp_needed, hardware, model, variant)
    return perf_db


def _check_tp_coverage(perf_db, tp_needed, hardware, model, variant):
    missing = sorted(set(tp_needed) - set(perf_db["available_tps"]))
    if missing:
        raise FileNotFoundError(
            f"No profile data for tp={missing} under "
            f"perf/{hardware}/{model}/{variant}/. Re-run the profiler with "
            f"TP_DEGREES including {','.join(str(t) for t in missing)}."
        )


def warn_if_runtime_exceeds_profiled(perf_db, runtime_max_num_batched_tokens,
                                     runtime_max_num_seqs):
    """Emit logger warnings when runtime batch limits exceed the values
    the profiler swept. Lookups will extrapolate, which is less accurate.
    Invoked once per (hw, model, variant) cache-hit.
    """
    meta = perf_db.get("meta", {})
    eff = (meta or {}).get("engine_effective") or {}
    p_tok = eff.get("max_num_batched_tokens")
    p_seqs = eff.get("max_num_seqs")
    key = ("warned", perf_db["hardware"], perf_db["model"], perf_db["variant"])
    if _perf_db_cache.get(key):
        return
    _perf_db_cache[key] = True
    if p_tok and runtime_max_num_batched_tokens and \
            runtime_max_num_batched_tokens > p_tok:
        logger.warning(
            "max-num-batched-tokens=%s exceeds profiled %s for %s/%s/%s; "
            "attention/dense lookups will extrapolate",
            runtime_max_num_batched_tokens, p_tok,
            perf_db["hardware"], perf_db["model"], perf_db["variant"],
        )
    if p_seqs and runtime_max_num_seqs and runtime_max_num_seqs > p_seqs:
        logger.warning(
            "max-num-seqs=%s exceeds profiled %s for %s/%s/%s; "
            "per-sequence lookups will extrapolate",
            runtime_max_num_seqs, p_seqs,
            perf_db["hardware"], perf_db["model"], perf_db["variant"],
        )


def _linear_interpolate(x0, y0, x1, y1, query):
    """Linear interpolation (or extrapolation)."""
    if x1 == x0:
        return y0
    t = (query - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def _lookup_bounds(keys, query):
    """Binary search returning (lo_idx, hi_idx) bracket.

    If query is below min, returns (0, 0).
    If query is above max, returns (len-2, len-1) to allow extrapolation.
    Otherwise returns the bracketing pair.
    """
    idx = bisect.bisect_right(keys, query)
    if idx == 0:
        return 0, 0
    if idx >= len(keys):
        if len(keys) < 2:
            return 0, 0
        return len(keys) - 2, len(keys) - 1
    return idx - 1, idx


def _lookup_1d(keys, values, query):
    """1D interpolation on sorted (keys, values)."""
    if not keys:
        return 0
    if len(keys) == 1:
        return values[0]

    lo, hi = _lookup_bounds(keys, query)
    if lo == hi:
        # Clamped or exact
        return values[lo]
    return _linear_interpolate(keys[lo], values[lo], keys[hi], values[hi], query)


def _build_tp_tables(tp_dir):
    """Build every category table present in one ``tp<N>/`` folder."""
    tables = {}
    dense_df = _read_category_csv(os.path.join(tp_dir, "dense.csv"), None)
    if dense_df is not None:
        tables["dense"] = _build_1d_table(dense_df, "layer", "tokens")

    per_seq_df = _read_category_csv(os.path.join(tp_dir, "per_sequence.csv"), None)
    if per_seq_df is not None:
        tables["per_sequence"] = _build_1d_table(per_seq_df, "layer", "sequences")

    attn_df = _read_category_csv(os.path.join(tp_dir, "attention.csv"), None)
    if attn_df is not None:
        # A sparse-attention profile carries more than one kernel here -- the
        # attention kernel and an indexer that scores the whole KV before the
        # top-k selection -- keyed on the same four axes. Split by layer so
        # they cannot contaminate each other; the plain ``attention`` entry
        # stays exactly what it has always been.
        # Keyed by kernel, with no pooled alias: there used to be a
        # ``tables["attention"]`` shortcut and every lookup silently took it,
        # so a sparse model's indexer and sparse-attention layers were both
        # served the non-sparse kernel's latency (2.1x too high per sparse
        # layer on MiniMax-M3). One way in, and it needs a layer name.
        tables["attention_by_layer"] = _build_attention_tables_by_layer(attn_df)

    moe_df = _read_category_csv(os.path.join(tp_dir, "moe.csv"), None)
    if moe_df is not None:
        tables["moe"] = _build_moe_table(moe_df)
    return tables


def _tp_tables(perf_db, tp):
    """Fetch the category-table dict for a given TP degree, building it on
    first use, with a clear error if the TP wasn't profiled.

    Lazy because a bundle may carry many ``tp<N>/`` folders while one run
    touches at most two of them: its own TP degree, plus tp1 for the
    ``tp_stable`` layers and for MoE, which are profiled once at tp=1 (see
    _effective_tp and _lookup_moe). Building every folder up front cost
    ~700 ms per folder that the run never queried.

    Every caller reaches the tables through here, so a lazily-built folder
    is indistinguishable from an eagerly-built one. That matters for
    _layer_available in particular: it used to read perf_db["tables"]
    directly with a {} default, which under lazy building would have
    reported a present layer as missing and silently skipped emitting it.
    """
    tables = perf_db["tables"].get(tp)
    if tables is not None:
        return tables
    if tp not in perf_db["available_tps"]:
        raise KeyError(
            f"No profile data for tp={tp} on {perf_db['hardware']}/"
            f"{perf_db['model']}/{perf_db['variant']}; available: "
            f"{perf_db['available_tps']}"
        )
    tables = _build_tp_tables(os.path.join(perf_db["root"], f"tp{tp}"))
    perf_db["tables"][tp] = tables
    return tables


def _tp_stable(perf_db, category, name):
    """Return True if the catalog marks this layer as TP-stable — i.e.
    the same kernel cost at any TP and profiled once at tp=1.
    """
    section = perf_db["architecture"]["catalog"].get(category) or {}
    entry = section.get(name)
    if not entry:
        return False
    return bool(entry.get("tp_stable"))


def _effective_tp(perf_db, category, name, tp):
    """Layers marked ``tp_stable`` in the architecture yaml are profiled
    once at tp=1 and the writer replicates them across TP folders, so
    either lookup works. Using the current TP keeps things uniform.
    """
    if _tp_stable(perf_db, category, name) and 1 in perf_db["available_tps"]:
        return 1
    return tp


def _lookup_dense(perf_db, name, tp, tokens):
    tp_eff = _effective_tp(perf_db, "dense", name, tp)
    tbl = _tp_tables(perf_db, tp_eff).get("dense", {}).get(name)
    if tbl is None:
        raise KeyError(
            f"Missing dense profile for layer={name} on tp={tp_eff}. "
            f"Check that the architecture catalog and dense.csv agree."
        )
    return max(1, int(_lookup_1d(tbl["keys"], tbl["values"], max(int(tokens), 1))))


def _lookup_per_sequence(perf_db, name, tp, sequences):
    tp_eff = _effective_tp(perf_db, "per_sequence", name, tp)
    tbl = _tp_tables(perf_db, tp_eff).get("per_sequence", {}).get(name)
    if tbl is None:
        raise KeyError(
            f"Missing per-sequence profile for layer={name} on tp={tp_eff}."
        )
    return max(1, int(_lookup_1d(tbl["keys"], tbl["values"], max(int(sequences), 1))))


def _axis_bracket(values, query):
    """Return (lo_idx, hi_idx, t) for linear interpolation on ``values``
    (sorted, non-negative, may include 0). ``t`` is the fractional
    position: 0 → use values[lo_idx], 1 → values[hi_idx].

    Below the min we clamp (a 0-valued sample is pinned exact); above
    the max we extrapolate linearly from the top two samples.

    ``t`` is measured on a **linear** scale even though the profiler
    sweeps every axis geometrically. Those are separate choices: the
    grid spacing decides where the kernel is sampled, the blend decides
    how two samples are combined, and the kernel is linear in each
    axis. Profiled decode attention fits ``time_us = a + b * (n_decode
    * kv_decode)`` with R^2 = 1.0000 on the RTX 4090 Llama-3.1-8B grid,
    at an implied 953 GB/s — 95% of the card's spec, i.e. a pure
    KV-bandwidth read.

    Blending a per-axis-linear function in log space is convex-biased
    upward: up to +6.0% per axis on a doubling grid (worst at
    ``query/x0 = 1/ln2 = 1.443``), compounding across axes. Leave-one-out
    over every profiled attention row — predict a grid point from its
    two neighbours, compare against what the GPU actually reported —
    puts log-space at +11.6% to +14.4% mean error across all seven
    bundles in the repo, against +2.3% to +3.7% for linear, with linear
    ahead on all four axes (``n_decode`` worst: +18.4% log vs +4.3%).
    """
    n = len(values)
    if n == 0:
        raise KeyError("empty axis")
    if n == 1 or query <= values[0]:
        return 0, 0, 0.0
    idx = bisect.bisect_right(values, query)
    if idx >= n:
        lo, hi = n - 2, n - 1
    else:
        lo, hi = idx - 1, idx
    x0, x1 = values[lo], values[hi]
    if x1 == x0:
        return lo, hi, 0.0
    return lo, hi, (query - x0) / (x1 - x0)


def _attn_slice_lookup(tbl, pc, nd, kv_prefill, kv_decode):
    """Bilinear (linear on each axis) within a single (pc, nd) slice."""
    slice_tbl = tbl["slices"].get((pc, nd))
    if slice_tbl is None:
        return None
    kp_vals = slice_tbl["kv_prefill_vals"]
    rows = slice_tbl["rows"]
    if not kp_vals:
        return None
    lo_kp, hi_kp, t_kp = _axis_bracket(kp_vals, max(int(kv_prefill), 0))

    def _row_lookup(row):
        ks, vs = row["keys"], row["values"]
        if not ks:
            return None
        if len(ks) == 1:
            return vs[0]
        lo, hi, t = _axis_bracket(ks, max(int(kv_decode), 0))
        return vs[lo] + t * (vs[hi] - vs[lo])

    v_lo = _row_lookup(rows[lo_kp])
    if lo_kp == hi_kp or v_lo is None:
        return v_lo
    v_hi = _row_lookup(rows[hi_kp])
    if v_hi is None:
        return v_lo
    return v_lo + t_kp * (v_hi - v_lo)


# ---------------------------------------------------------------------------
# Skew correction
# ---------------------------------------------------------------------------
# When the runtime batch has heterogeneous decode kv lengths, the
# profiled 4D grid (which carries one kv_decode value per shot) can
# only tell us the uniform-batch latency. ``kv_decode_mean`` is the
# right coordinate to ask it for: decode attention cost tracks the
# total KV read Sigma_k, and a uniform batch at the arithmetic mean has
# ``n * mean(k) = Sigma_k`` exactly. The median would not — the runtime
# kv distribution is right-skewed (measured ``kv_max/kv_mean`` p50 =
# 2.61 on the ShareGPT replay), so a median anchor would understate the
# read volume. The profiler uses the same definition (``skew.py``:
# ``kv_mean = total_kv // n``).
#
# A truly skewed batch is slightly *slower* than that uniform anchor,
# because FlashAttention's varlen kernel pays tile padding and
# SM-imbalance costs the uniform measurement misses. The skew profile
# (profiler/.../tp<N>/skew.csv + the fitted ``skew_fit`` block in
# meta.yaml, with the bucket alpha table spilled to
# ``tp<N>/skew_fit.csv``) captures that as a 5-axis lookup table of
# alpha values where
#
#     t_skew = t_mean + alpha * (t_max - t_mean)
#
# Lookup is resolved per-batch via ``_skew_alpha``. The bin edges and
# labels come from ``meta.yaml::skew_fit.bucket_axes`` so the profiler
# can widen any axis (e.g. raise ``max_num_seqs`` above 128) without a
# coordinated code change here. The ``_DEFAULT_SKEW_AXES`` block below
# is used as a fallback only when the meta predates that field (which
# is why its shape still matches the original hard-coded scheme).
#
# The fallback, used when a bundle carries no skew profile at all, is
# **0**: apply no correction you have not measured. It used to be
# 0.093, a constant no bundle in the repo reproduces — the measured
# pooled value for Llama-3.1-8B on RTXPRO6000 is 0.0543, and resolving
# a saturated RTX 4090 run's own batches against that bucket table
# gives alpha p50 0.059. A scalar cannot serve this parameter anyway:
# the endpoint gap ``(t_max - t_mean) * num_layers`` is ~12.6 ms on a
# ~29 ms iteration, so each 0.1 of alpha is ~4.3% of iteration time and
# alpha would have to be known to +/-0.023 to keep attention within 1%.
# Profile skew if you need the correction; guessing it is worse than
# omitting it.
_ATTN_SKEW_ALPHA_FALLBACK: float = 0.0

_DEFAULT_SKEW_AXES: dict = {
    "n_bins": (0, 2, 4, 8, 16, 32, 64, 128, 1_000_000),
    "n_labels": (
        "n<=2", "n<=4", "n<=8", "n<=16", "n<=32", "n<=64", "n<=128", "n>128",
    ),
    "kv_big_bins": (0, 1024, 4096, 16384, 1_000_000_000),
    "kv_big_labels": ("kvB<=1k", "kvB<=4k", "kvB<=16k", "kvB>16k"),
    "skew_rate_bins": (-0.01, 0.05, 0.15, 0.40, 0.70, 1.01),
    "skew_rate_labels": ("sr<=5%", "sr<=15%", "sr<=40%", "sr<=70%", "sr>70%"),
    "kp_bins": (-1, 0, 2048, 1_000_000_000),
    "kp_labels": ("kp=0", "kp<=2k", "kp>2k"),
}


def _bucket_label(bins, labels, val) -> str:
    # Bucketing is (bins[i], bins[i+1]] — inclusive on the right so
    # the label matches its intuitive reading (``n<=8`` includes 8).
    for i in range(len(labels)):
        if val <= bins[i + 1]:
            return labels[i]
    return labels[-1]


def _resolve_skew_axes(fit_block, tp_entry):
    """Return the (bins, labels) axes used for key construction.

    Priority: per-TP entry > block top-level > module defaults. The
    per-TP override is primarily a transition path — the writer
    promotes ``bucket_axes`` to the top of the block when it's
    identical across TPs, which is the common case.
    """
    axes = None
    if isinstance(tp_entry, dict):
        axes = tp_entry.get("bucket_axes")
    if not axes and isinstance(fit_block, dict):
        axes = fit_block.get("bucket_axes")
    if not axes:
        return _DEFAULT_SKEW_AXES
    return axes


def _skew_alpha(
    perf_db,
    tp: int,
    pc: int,
    n: int,
    skew_rate: float,
    kv_big: int,
    kp: int,
    layer: str = "attention",
) -> float:
    """Resolve alpha for a specific batch from the profile's
    ``skew_fit`` meta block.

    Lookup order:
        1. meta.yaml::skew_fit.per_tp[tp].alpha_by_bucket[bucket_key]
           (hydrated from ``tp<N>/skew_fit.csv`` when the meta points
           at a CSV instead of inlining the mapping). The bucket_key is
           ``{layer}|pc={pc}|{n_label}|{sr_label}|{kvb_label}|{kp_label}``,
           built against ``skew_fit.bucket_axes`` if present — which
           lets the profiler widen axes (more n bins, finer kp bins)
           without a simulator-side code change.
        2. The unprefixed key, **only for ``attention``**: a bundle
           profiled before skew.csv had a ``layer`` column holds one
           kernel and it is that one.
        3. ``alpha_default_by_layer[layer]``, the pooled WLS constant for
           this kernel.
        4. ``alpha_default`` (pooled over every kernel), again only for
           ``attention``.
        5. Module-level fallback constant (``_ATTN_SKEW_ALPHA_FALLBACK``).

    Why ``layer`` is in the key: the sparse families run two or three
    kernels in this category and their alphas are not interchangeable.
    MiniMax-M3's block selection caps the work its sparse layers do, so
    past the block budget their cost stops tracking kv length and the
    endpoint gap collapses; its indexer scans the whole KV and is the most
    skew-sensitive thing in the model. Steps 2 and 4 deliberately refuse to
    answer for any other layer — handing a sparse kernel the dense one's
    alpha is the mistake this prefix exists to prevent, and 0 (no
    correction) is the documented behaviour when a kernel has no skew data.

    Returns the fallback constant when the meta block is disabled or
    missing.
    """
    meta = perf_db.get("meta") if isinstance(perf_db, dict) else None
    if not meta:
        return _ATTN_SKEW_ALPHA_FALLBACK
    fit_block = meta.get("skew_fit")
    if not fit_block or not fit_block.get("enabled"):
        return _ATTN_SKEW_ALPHA_FALLBACK
    per_tp = fit_block.get("per_tp") or {}
    entry = per_tp.get(tp) or per_tp.get(int(tp)) or per_tp.get(str(tp))
    if not entry:
        return float(fit_block.get("alpha_default", _ATTN_SKEW_ALPHA_FALLBACK))
    axes = _resolve_skew_axes(fit_block, entry)
    if layer is None:
        layer = "attention"
    sr = max(0.0, min(1.0, float(skew_rate)))
    n_label = _bucket_label(axes["n_bins"], axes["n_labels"], int(n))
    sr_label = _bucket_label(
        axes["skew_rate_bins"], axes["skew_rate_labels"], sr,
    )
    kvb_label = _bucket_label(
        axes["kv_big_bins"], axes["kv_big_labels"], int(kv_big),
    )
    kp_label = _bucket_label(axes["kp_bins"], axes["kp_labels"], int(kp))
    bucket = f"pc={int(pc)}|{n_label}|{sr_label}|{kvb_label}|{kp_label}"
    alphas = entry.get("alpha_by_bucket") or {}
    key = f"{layer}|{bucket}"
    if key in alphas:
        return float(alphas[key])
    if layer == "attention" and bucket in alphas:
        return float(alphas[bucket])
    per_layer = entry.get("alpha_default_by_layer") or {}
    if layer in per_layer:
        return float(per_layer[layer])
    if layer == "attention":
        return float(entry.get("alpha_default", _ATTN_SKEW_ALPHA_FALLBACK))
    return 0.0


def _lookup_attention_with_skew(
    perf_db, tp, prefill_chunk, kv_prefill,
    n_decode, kv_decode_mean, kv_decode_max, kv_decode_min,
    layer="attention",
):
    """Attention lookup with skew correction applied.

    Looks the batch up at kv_decode_mean (the canonical point the
    profiler measured) and, when a non-zero alpha applies, blends
    toward a second lookup at kv_decode_max (the per-batch longest
    decode sequence) using the bucket-specific alpha resolved from
    ``meta.yaml::skew_fit``. Returns t_mean directly -- one lookup --
    for a single decode, for a batch whose decodes are all the same
    length, or when alpha resolves to 0.

    Returns an integer nanosecond count. The interpolation in
    ``_lookup_attention`` produces a float, and the skew formula
    compounds that; the Chakra trace converter requires integer
    ``comp_time`` so we round here.
    """
    t_mean = _lookup_attention(
        perf_db, tp, prefill_chunk, kv_prefill, n_decode, kv_decode_mean,
        layer,
    )
    # No skew → no correction (also saves a redundant lookup).
    if n_decode <= 1 or kv_decode_max == kv_decode_mean:
        return max(1, int(round(t_mean)))
    # skew_rate ∈ [0, 1]; = nb / n exactly for a bimodal batch.
    # Fallback to 0.5 (balanced) when kv_max == kv_min (shouldn't
    # reach here due to the short-circuit above, but defensive).
    kv_gap = kv_decode_max - kv_decode_min
    skew_rate = (kv_decode_mean - kv_decode_min) / kv_gap if kv_gap > 0 else 0.5
    alpha = _skew_alpha(
        perf_db, tp, prefill_chunk, n_decode, skew_rate, kv_decode_max,
        kv_prefill, layer,
    )
    if alpha == 0.0:
        return max(1, int(round(t_mean)))
    t_max = _lookup_attention(
        perf_db, tp, prefill_chunk, kv_prefill, n_decode, kv_decode_max,
        layer,
    )
    # Guard against interpolation producing t_max < t_mean (can happen
    # at the axis boundary); in that case the formula would produce a
    # negative correction, which isn't physical.
    if t_max <= t_mean:
        return max(1, int(round(t_mean)))
    return max(1, int(round(t_mean + alpha * (t_max - t_mean))))


def _lookup_attention(perf_db, tp, prefill_chunk, kv_prefill, n_decode,
                      kv_decode, layer="attention"):
    """4D interpolation on (prefill_chunk, kv_prefill, n_decode, kv_decode).

    Each axis is bracketed by its two nearest profiled values and blended
    **linearly** -- not in log space, even though the profiler sweeps every
    axis geometrically. Grid spacing decides where the kernel is sampled; the
    blend decides how two samples combine; the kernel is linear in each axis.

    ``layer`` selects which kernel's table to read. A sparse-attention model
    has several in this category -- MiniMax-M3 profiles ``attention`` (the
    non-sparse layers), ``sparse_attention`` and ``indexer``, all keyed on the
    same four axes -- and they are not interchangeable: at a 4-decode/kv-256
    batch the non-sparse kernel costs 16.1 us against 7.5 for either sparse
    one, because block selection caps the work the sparse layers do. A bundle
    profiled before the CSV grew a ``layer`` column has exactly one kernel,
    filed under ``attention``, so the default keeps it byte-identical.
    """
    by_layer = _tp_tables(perf_db, tp).get("attention_by_layer") or {}
    tbl = by_layer.get(layer)
    if tbl is None:
        raise KeyError(
            f"Missing attention profile for layer={layer!r} at tp={tp}. "
            f"Profiled kernels: {sorted(by_layer) or 'none'}."
        )
    if not tbl["pc_nd_pairs"]:
        raise KeyError(f"Missing attention profile for tp={tp}.")

    pcq, ndq = max(int(prefill_chunk), 0), max(int(n_decode), 0)
    pc_vals, nd_vals = tbl["pc_vals"], tbl["nd_vals"]
    lo_pc, hi_pc, t_pc = _axis_bracket(pc_vals, pcq)
    lo_nd, hi_nd, t_nd = _axis_bracket(nd_vals, ndq)

    # Grab the four corners; missing corners fall back to the closest
    # available (pc, nd) pair.
    def _corner(pc, nd):
        v = _attn_slice_lookup(tbl, pc, nd, kv_prefill, kv_decode)
        if v is not None:
            return v
        nearest = min(tbl["pc_nd_pairs"],
                      key=lambda p: (p[0] - pc) ** 2 + (p[1] - nd) ** 2)
        return _attn_slice_lookup(tbl, nearest[0], nearest[1],
                                  kv_prefill, kv_decode) or 0.0

    c00 = _corner(pc_vals[lo_pc], nd_vals[lo_nd])
    c01 = _corner(pc_vals[lo_pc], nd_vals[hi_nd])
    c10 = _corner(pc_vals[hi_pc], nd_vals[lo_nd])
    c11 = _corner(pc_vals[hi_pc], nd_vals[hi_nd])

    v0 = c00 + t_nd * (c01 - c00)
    v1 = c10 + t_nd * (c11 - c10)
    out = v0 + t_pc * (v1 - v0)
    return max(1, int(out))


def _lookup_moe(perf_db, tokens, activated_experts):
    """MoE is profiled once at tp=1 (single-rank view); the simulator
    looks up per EP-rank token counts.
    """
    tp_eff = 1 if 1 in perf_db["available_tps"] else perf_db["available_tps"][0]
    tbl = _tp_tables(perf_db, tp_eff).get("moe")
    if tbl is None:
        raise KeyError(
            f"Missing moe profile. Check that moe.csv exists under "
            f"perf/{perf_db['hardware']}/{perf_db['model']}/{perf_db['variant']}/tp{tp_eff}/."
        )
    ae_vals = tbl["activated_experts_vals"]
    rows = tbl["rows"]
    aeq = max(int(activated_experts), 1)
    tokq = max(int(tokens), 1)
    lo, hi = _lookup_bounds(ae_vals, aeq)
    val_lo = _lookup_1d(rows[lo]["keys"], rows[lo]["values"], tokq)
    if lo == hi:
        return max(1, int(val_lo))
    val_hi = _lookup_1d(rows[hi]["keys"], rows[hi]["values"], tokq)
    out = _linear_interpolate(ae_vals[lo], val_lo, ae_vals[hi], val_hi, aeq)
    return max(1, int(out))


def _catalog_has(perf_db, category, name):
    section = perf_db["architecture"]["catalog"].get(category) or {}
    return name in section


# ======================================================================
# Context builders
# ======================================================================

def _build_trace_ctx(hardware, model, config, tp_size, pp_size, local_ep, ep_total, node_id, fp,
                     placement, gate, enable_attn_offloading, power_model, pim_model, pd_type,
                     variant, kv_cache_dtype='auto',
                     runtime_max_num_batched_tokens=None, runtime_max_num_seqs=None,
                     tp_dim=None, ep_dim=None, dp_sum_total_len=0):
    model_type = config.get('model_type')
    if not model_type:
        raise KeyError(
            f"Model config for {model!r} has no 'model_type'; cannot locate "
            f"profiler/models/<model_type>.yaml"
        )
    tp_needed = {max(int(tp_size), 1)}
    perf_db = _load_perf_db(hardware, model, variant, tp_needed, model_type,
                            model_config=config)
    warn_if_runtime_exceeds_profiled(
        perf_db, runtime_max_num_batched_tokens, runtime_max_num_seqs)

    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    kv_head = config.get('num_key_value_heads', n_head)
    head_dim = config.get('head_dim', n_embd // n_head)
    is_moe = gate is not None

    pim_channels = 0
    if enable_attn_offloading and pim_model is not None:
        pim_config = pim_model.get_config()
        pim_channels = int(pim_config["mem_size"] // pim_config["dimm_size"])

    return TraceCtx(
        hardware=hardware, model=model, config=config, perf_db=perf_db,
        node_id=node_id,
        fp=fp, placement=placement, gate=gate,
        enable_attn_offloading=enable_attn_offloading,
        power_model=power_model, pim_model=pim_model, pim_channels=pim_channels,
        n_head=n_head, kv_head=kv_head, head_dim=head_dim, is_moe=is_moe,
        kv_fp=(1 if kv_cache_dtype == 'fp8' else fp),
        pd_type=pd_type,
        tp_size=tp_size, pp_size=pp_size, local_ep=local_ep, ep_total=ep_total,
        tp_dim=tp_dim, ep_dim=ep_dim, dp_sum_total_len=dp_sum_total_len,
    )


def _build_batch_ctx(batch, ctx):
    # batch.total_len is the number of tokens actually computed this iteration:
    # the scheduler builds it from chunk_size = original_input - num_computed_tokens,
    # and num_computed_tokens already absorbs any prefix-cache hit, so no further
    # subtraction is needed even when prefix caching is on.
    total_len = batch.total_len
    # DP padding (see serving.__main__._pad_batch_to_max) adds dummy decodes without
    # touching batch.requests. vLLM keeps lm_head's output shape pinned to
    # num_tokens_after_padding for CUDA-graph replay, so each padded decode
    # also contributes a logit. Track it via num_prefill + num_decode.
    lm_head_len = max(len(batch.requests), batch.num_prefill + batch.num_decode)

    # 4D attention keys: profiler sweeps (prefill_chunk, kv_prefill,
    # n_decode, kv_decode). The kv_decode axis carries a single value
    # per shot, so we collapse multi-decode requests to their mean
    # AND capture the per-batch max/min for the skew correction below.
    prefill_chunk = sum(batch.prefill_q_list)
    kv_prefill = sum(batch.prefill_k_list)
    n_decode = len(batch.decode_k_list)
    kv_decode_mean = (sum(batch.decode_k_list) // n_decode) if n_decode > 0 else 0
    kv_decode_max = max(batch.decode_k_list) if n_decode > 0 else 0
    kv_decode_min = min(batch.decode_k_list) if n_decode > 0 else 0

    # PIM offloading: NPU sees only the prefill portion.
    decode_lens = None
    channel_split = 0
    if ctx.enable_attn_offloading and ctx.pim_model is not None:
        channel_split = min(ctx.pim_channels, ctx.kv_head)
        _, decode_lens = _attn_load_balancer(batch.requests, ctx.tp_size, ctx.pim_channels, channel_split)
        n_decode = 0
        kv_decode_mean = 0
        kv_decode_max = 0
        kv_decode_min = 0
        total_len = max(1, total_len)  # preserve for size calcs

    return BatchCtx(batch, total_len, prefill_chunk, kv_prefill, n_decode,
                    kv_decode_mean, kv_decode_max, kv_decode_min,
                    lm_head_len, decode_lens, channel_split)


# ======================================================================
# Layer emission helpers
# ======================================================================

def _layer_category(perf_db, layer_name):
    """Return which catalog category (dense/per_sequence/attention/moe)
    a canonical layer belongs to for this architecture, or None if the
    catalog doesn't include it.
    """
    for cat in ("per_sequence", "attention", "moe", "dense"):
        if _catalog_has(perf_db, cat, layer_name):
            return cat
    return None


def _emit_layer(ctx, bctx, layer_name, lines, power_acc, batch_tag='NONE', layer_num=None,
                comm_type='NONE', comm_size=0, input_loc='LOCAL', output_loc='LOCAL'):
    """Emit a single trace layer: lookup latency, compute sizes, format, track power."""
    category = _layer_category(ctx.perf_db, layer_name)
    if category is None:
        raise KeyError(
            f"Layer {layer_name!r} is not declared in the architecture yaml "
            f"catalog for {ctx.perf_db['variant']}. Add it to "
            f"profiler/models/<model_type>.yaml or remove it from the block order."
        )

    if category == "per_sequence":
        latency_ns = _lookup_per_sequence(ctx.perf_db, layer_name, ctx.tp_size, bctx.lm_head_len)
    elif category == "attention":
        latency_ns = _lookup_attention_with_skew(
            ctx.perf_db, ctx.tp_size,
            bctx.prefill_chunk, bctx.kv_prefill,
            bctx.n_decode, bctx.kv_decode_mean, bctx.kv_decode_max,
            bctx.kv_decode_min, layer_name,
        )
    else:  # dense
        latency_ns = _lookup_dense(ctx.perf_db, layer_name, ctx.tp_size, bctx.total_len)

    # Size calculation uses the same canonical layer names.
    if layer_name == 'attention':
        kv_len_for_sizes = bctx.kv_prefill + bctx.n_decode * bctx.kv_decode_mean
        inp, wt, out = calculate_sizes(ctx.model, layer_name, bctx.total_len,
                                       kv_len=kv_len_for_sizes,
                                       parallel=ctx.tp_size, fp=ctx.fp)
    else:
        inp, wt, out = calculate_sizes(ctx.model, layer_name, bctx.total_len,
                                       parallel=ctx.tp_size, fp=ctx.fp)

    wt_loc = get_device(ctx.placement, layer_num, layer_name, "weights")

    lines.append((layer_name, str(latency_ns), input_loc, str(inp), wt_loc,
                  str(wt), output_loc, str(out), comm_type, str(comm_size), batch_tag))

    if power_acc is not None:
        power_acc.npu_latencies_ns.append(latency_ns)
        if wt_loc != 'LOCAL':
            power_acc.dram_weight_bytes += wt
        if comm_size > 0:
            collective = comm_type.split(':', 1)[0].lower()
            if collective == 'none':
                # Point-to-point (the P/D KV send): the bytes cross the link once,
                # with no ring amplification.
                power_acc.link_data_bytes += comm_size
            else:
                power_acc.link_data_bytes += total_ring_data(comm_size, ctx.tp_size, collective=collective)

    return latency_ns


def _pd_kv_send_bytes(ctx, bctx):
    """Per-layer, per-rank KV bytes a prefill instance ships to its decode peer.

    ``batch.pd_kv_send_tokens`` counts this iteration's computed tokens plus, on a
    request's first step, its prefix-cache hit: the decode side needs that KV even
    though the prefill side read it from cache instead of computing it. Using the
    trace's ``total_len`` instead would silently drop exactly the hit.
    """
    tokens = getattr(bctx.batch, 'pd_kv_send_tokens', 0) or 0
    if tokens <= 0:
        return 0
    kv_dim = ctx.kv_head * ctx.head_dim
    return 2 * kv_dim * tokens * ctx.kv_fp // max(ctx.tp_size, 1)


def _tp_comm(ctx, layer_name, total_len, collective='ALLREDUCE'):
    """Compute TP communication size. Returns (comm_size, comm_type)."""
    if ctx.tp_size <= 1:
        return 0, 'NONE'
    _, _, out = calculate_sizes(ctx.model, layer_name, total_len, parallel=ctx.tp_size, fp=ctx.fp)
    return out, collective


def _with_dim(comm_type, involved_dim):
    """Encode involved_dim into comm_type string: 'ALLREDUCE' + [T,F] -> 'ALLREDUCE:1,0'."""
    if involved_dim is None or comm_type == 'NONE':
        return comm_type
    dim_str = ','.join('1' if d else '0' for d in involved_dim)
    return f"{comm_type}:{dim_str}"


def _emit_pim_attention(ctx, bctx, lines, power_acc, layer_num, batch_tag='NONE'):
    """Emit PIM attention for decode requests across PIM channels."""
    for ch in range(ctx.pim_channels):
        lines.append((f"PIM {ch}",))
        for L in bctx.decode_lens[ch]:
            inp, _, out = calculate_sizes(ctx.model, "attention", L, pim=True, parallel=ctx.tp_size, fp=ctx.fp)
            inp //= bctx.channel_split
            out //= bctx.channel_split
            pim_lat = int(ctx.pim_model.get_pim_latency(ctx.n_head, ctx.kv_head, ctx.head_dim, L, bctx.channel_split))
            lines.append(("attention", str(pim_lat),
                f'REMOTE:{ctx.node_id}.{ch}', str(inp),
                get_device(ctx.placement, layer_num, "attention", "weights"), '0',
                f'REMOTE:{ctx.node_id}.{ch}', str(out),
                'NONE', '0', batch_tag))
            if power_acc is not None and pim_lat > 0:
                power_acc.pim_latencies_ns.append(pim_lat)
                power_acc.dram_weight_bytes += inp + out
    lines.append(("PIM END",))


def _emit_npu_attention(ctx, bctx, lines, power_acc, layer_num, batch_tag='NONE'):
    """Emit NPU attention (unified prefill+decode lookup)."""
    if bctx.prefill_chunk == 0 and bctx.n_decode == 0:
        return
    _emit_layer(ctx, bctx, "attention", lines, power_acc, batch_tag, layer_num)


def _emit_moe_block(ctx, bctx, lines, power_acc, layer_num, batch_id_str, batch_tag='NONE'):
    """Emit MoE block: dispatch ALLTOALL + per-EP-rank expert compute + combine ALLTOALL.

    Each EP rank receives a different number of tokens based on expert routing.
    Per-rank latency is looked up independently from profiled data at tp=1.
    ALLTOALL is handled by ASTRA-Sim with involved_dim scoping for DP groups,
    or as a simple ALLTOALL for local EP groups.
    """
    ep_total = ctx.ep_total

    # MoE compute uses ``bctx.total_len`` (= per-rank padded count after
    # ``_pad_batch_to_max``), matching how the real vLLM kernel runs on the
    # full padded forward shape. Routing / ``_lookup_moe`` therefore see
    # the same per-rank padded value as every other dense layer.
    effective_total_len_compute = bctx.total_len
    routing = ctx.gate.route_ep(layer_num, batch_id_str, effective_total_len_compute, ep_total)

    # AG/RS comm sizes are anchored to ``dp_sum_total_len``, which
    # ``serving/__main__.py`` sets to ``max_total_len`` (NOT ``max × dp_group_size``)
    # for DP groups; this calibrates the AG/RS bandwidth model against the same
    # ``link_bw`` that already matches AllReduce. Falls back to this rank's own
    # ``total_len`` when DP is inactive.
    effective_total_len_comm = ctx.dp_sum_total_len if ctx.dp_sum_total_len > 0 else bctx.total_len

    # vLLM default ``allgather_reducescatter`` backend: dispatch = AllGather
    # (hidden + router_logits), combine = ReduceScatter (hidden only).
    # ASTRA-Sim AG ``data_size`` is per-rank local chunk (sum / ep_total);
    # RS ``data_size`` is the pre-scatter total buffer.
    n_embd = ctx.config['hidden_size']
    num_experts = utils_num_experts(ctx.config)
    dispatch_per_token = (n_embd + num_experts) * ctx.fp
    combine_per_token = n_embd * ctx.fp
    ag_per_rank_tokens = max(1, effective_total_len_comm // max(ep_total, 1))
    dispatch_comm_size = ag_per_rank_tokens * dispatch_per_token
    combine_comm_size = effective_total_len_comm * combine_per_token

    if ep_total > 1:
        dispatch_comm_type = _with_dim('ALLGATHER', ctx.ep_dim)
        combine_comm_type = _with_dim('REDUCESCATTER', ctx.ep_dim)
    else:
        dispatch_comm_type = 'NONE'
        combine_comm_type = 'NONE'
        dispatch_comm_size = 0
        combine_comm_size = 0

    wt_loc = get_device(ctx.placement, layer_num, "moe", "weights")

    # Each local GPU handles exactly one EP rank. The routing result already
    # accounts for cross-instance token redistribution (ALLTOALL), so
    # local_tokens[rank] reflects the post-dispatch workload for that rank.
    emit_ep = max(ctx.local_ep, 1)
    max_rank_latency_ns = 0

    # Pre-expert AllGather power (dispatch)
    if power_acc is not None and ep_total > 1:
        power_acc.link_data_bytes += total_ring_data(dispatch_comm_size, ep_total, collective="allgather")

    for i in range(emit_ep):
        if i == 0:
            lines.append((f"EXPERT {i} {dispatch_comm_type} {dispatch_comm_size}",))
        else:
            lines.append((f"EXPERT {i} NONE 0",))

        # ``local_tokens`` here is the per-rank workload after dispatch
        # — already scaled to this rank's real tokens (no DP-padding sum).
        # We feed it straight into the MoE profile lookup.
        local_tokens = routing.local_tokens[i]
        activated_experts = routing.activated_experts[i]

        if local_tokens > 0:
            rank_latency_ns = _lookup_moe(ctx.perf_db, local_tokens, max(activated_experts, 1))
            rank_inp, rank_wt, rank_out = calculate_sizes(
                ctx.model, "moe", local_tokens, parallel=ep_total, fp=ctx.fp)
            max_rank_latency_ns = max(max_rank_latency_ns, rank_latency_ns)

            lines.append(("expert", str(rank_latency_ns), 'LOCAL', str(rank_inp),
                wt_loc, str(rank_wt), 'LOCAL', str(rank_out), 'NONE', '0', batch_tag))

            if power_acc is not None and wt_loc != 'LOCAL':
                power_acc.dram_weight_bytes += rank_wt

    # Power: all local GPUs are active for the duration of the slowest rank
    if power_acc is not None and max_rank_latency_ns > 0:
        power_acc.npu_latencies_ns.append(max_rank_latency_ns)

    lines.append((f"EXPERT END {combine_comm_type} {combine_comm_size}",))

    # Post-expert ReduceScatter power (combine)
    if power_acc is not None and ep_total > 1:
        power_acc.link_data_bytes += total_ring_data(combine_comm_size, ep_total, collective="reducescatter")


# ======================================================================
# Block builders (split for interleaving)
# ======================================================================

def _shared_layers(perf_db, section):
    """``shared.prologue`` or ``shared.head`` -- the layers outside every block.

    Once per iteration, not once per decoder layer, which is the whole reason
    they live outside ``blocks``.
    """
    shared = perf_db["architecture"].get("shared") or {}
    return list(shared.get(section) or [])


def _layer_spec(perf_db, layer_num):
    """The block composition of one decoder layer, from the checkpoint.

    Which block a layer runs is a property of the *checkpoint*, never of the
    yaml: ``layer_types`` decides the attention, ``first_k_dense_replace`` /
    ``decoder_sparse_step`` / ``moe_layer_freq`` the MLP, and
    ``sparse_attention_freq`` / ``index_topk_pattern`` the sparse flag. See
    ``profiler/core/stack.py``, which the profiler reads for the same answer.
    """
    stack = perf_db.get("layer_stack") or []
    if not stack:
        mod = _stack_module()
        return mod.LayerSpec(attn=mod.FULL_ATTENTION, mlp=mod.MLP_DENSE)
    return stack[int(layer_num or 0) % len(stack)]


def _block_layers(perf_db, layer_num, part):
    """The canonical layer names decoder layer ``layer_num`` emits for ``part``
    (``pre_attn`` / ``post_attn`` / ``mlp``).

    This is where a heterogeneous stack stops being uniform. ``blocks`` is
    keyed by **axis** -- ``attn.<layer_types value>``, ``mlp.dense|moe``, and
    ``sparse_attn.<layer_types value>`` as an overlay when the layer's sparse
    flag is set -- because a layer's identity is a tuple and naming every
    combination explodes: Qwen3.5 varies the attention, DeepSeek and GLM the
    MLP, MiniMax-M3 both plus sparsity.

    ``sparse_attn`` is consulted first and falls through to ``attn`` when it
    has no entry for this attention type, which is what DeepSeek and GLM
    need: every one of their layers is sparse, so there is nothing to tell
    apart and one ``attn`` block serves them all.
    """
    blocks = perf_db["architecture"].get("blocks") or {}
    spec = _layer_spec(perf_db, layer_num)

    if part == "mlp":
        by_type = blocks.get("mlp") or {}
        if spec.mlp not in by_type:
            raise KeyError(
                f"Architecture {perf_db['variant']} declares no "
                f"'blocks.mlp.{spec.mlp}', but layer {layer_num} of "
                f"{perf_db['model']} runs a {spec.mlp} MLP. Declared: "
                f"{sorted(by_type) or 'none'}."
            )
        return list(by_type.get(spec.mlp) or [])

    group = None
    if spec.sparse:
        group = (blocks.get("sparse_attn") or {}).get(spec.attn)
    if group is None:
        group = (blocks.get("attn") or {}).get(spec.attn)
    if group is None:
        declared = sorted((blocks.get("attn") or {}))
        raise KeyError(
            f"Architecture {perf_db['variant']} declares no "
            f"'blocks.attn.{spec.attn}', but layer {layer_num} of "
            f"{perf_db['model']} runs {spec.attn}. Declared: "
            f"{declared or 'none'}."
        )
    return list(group.get(part) or [])


_skipped_layer_warned = set()


def _layer_available(perf_db, tp, layer_name):
    """Return True when the CSV-backed table has data for this layer at
    the TP the simulator is about to query. Attention/MoE are always
    present when their category CSVs exist.
    """
    category = _layer_category(perf_db, layer_name)
    if category is None:
        return False
    tp_eff = _effective_tp(perf_db, category, layer_name, tp)
    tables = _tp_tables(perf_db, tp_eff)
    if category == "dense":
        return layer_name in tables.get("dense", {})
    if category == "per_sequence":
        return layer_name in tables.get("per_sequence", {})
    if category == "attention":
        # Ask for the requested kernel, not just "is there attention data".
        # The group can hold more than one now -- a sparse-attention profile
        # carries an indexer beside the attention kernel -- and answering yes
        # for a layer the profile has no rows for would emit a trace node
        # backed by nothing. For a bundle profiled before the CSV gained a
        # layer column, ``attention_by_layer`` is ``{"attention": ...}``, so a
        # catalog declaring ``attention`` still answers exactly as before.
        return layer_name in (tables.get("attention_by_layer") or {})
    if category == "moe":
        return bool(tables.get("moe"))
    return False


def _emit_sequence(ctx, bctx, layer_num, layers, lines, power_acc, batch_tag):
    """Walk a flat list of canonical layer names from the architecture
    yaml, emitting each. ``attention`` triggers PIM attention before the
    NPU kernel when attn offloading is enabled; layers in
    ``_TP_ALLREDUCE_AFTER`` get an ALLREDUCE attached. When a layer is
    declared in a block but the profile CSV lacks data for it
    (e.g., an older profile run that predates a yaml addition), the
    emission is skipped with a single warning per (variant, layer).
    """
    for layer_name in layers:
        if layer_name == "attention":
            if ctx.enable_attn_offloading:
                _emit_pim_attention(ctx, bctx, lines, power_acc, layer_num, batch_tag)
            _emit_npu_attention(ctx, bctx, lines, power_acc, layer_num, batch_tag)
            continue
        if layer_name == "rotary_emb" and "TPU" in ctx.hardware:
            continue
        if not _layer_available(ctx.perf_db, ctx.tp_size, layer_name):
            key = (ctx.perf_db["variant"], ctx.perf_db["model"], layer_name)
            if key not in _skipped_layer_warned:
                _skipped_layer_warned.add(key)
                logger.warning(
                    "Layer %r is in the architecture yaml block order but missing from "
                    "the profile CSVs for %s/%s/%s — skipping. Re-profile to include it.",
                    layer_name, ctx.perf_db["hardware"],
                    ctx.perf_db["model"], ctx.perf_db["variant"],
                )
            continue
        if layer_name in _TP_ALLREDUCE_AFTER:
            comm_size, comm_type = _tp_comm(ctx, layer_name, bctx.total_len)
            _emit_layer(ctx, bctx, layer_name, lines, power_acc, batch_tag, layer_num,
                        comm_type=_with_dim(comm_type, ctx.tp_dim), comm_size=comm_size)
        elif layer_name == 'qkv_proj' and ctx.pd_type == 'prefill':
            # P/D disaggregation: this layer's KV has to reach the paired decode
            # instance. The Chakra converter's PREFILL path emits a point-to-point
            # send after every *v_proj layer and takes the byte count from the
            # trace's comm_size column, so put the KV size there.
            #
            # Deliberately not the layer's output_size, which the converter used
            # to read: that is the whole QKV activation, so it shipped Q as well
            # and overstated the transfer by (q_dim + 2*kv_dim) / (2*kv_dim) --
            # 3x for Llama-3.1-8B, 1.5x for MHA, more at wider GQA ratios -- and
            # it ignored kv_cache_dtype. comm_type stays NONE: the converter
            # builds a SEND/RECV pair, for which ASTRA-Sim reads comm_size,
            # comm_src, comm_dst and comm_tag.
            _emit_layer(ctx, bctx, layer_name, lines, power_acc, batch_tag, layer_num,
                        comm_size=_pd_kv_send_bytes(ctx, bctx))
        else:
            _emit_layer(ctx, bctx, layer_name, lines, power_acc, batch_tag, layer_num)


def _block_copy_key(ctx, block_mode_on, *layer_nums):
    """Cache key for a built transformer block, or None if it must be rebuilt.

    A block's rows carry no layer index -- ``_emit_layer`` writes the canonical
    name and the writer numbers the lines -- so two layers can share one built
    block whenever they are the *same* block. That is what makes trace
    generation O(1) in depth instead of O(num_hidden_layers).

    Two things break the equivalence. ``block_mode_on`` emits each layer
    separately by definition, and a MoE router that is not deterministic
    carries per-layer variance (``gate.block_copy`` opts into swallowing it).
    Beyond those, the key is the layers' own :class:`LayerSpec`s: a
    heterogeneous stack has genuinely different blocks, and replaying layer 0's
    would emit gated DeltaNet for all 64 of Qwen3.8's layers, or a dense MLP
    for all 61 of DeepSeek-V3.2's. Several layer numbers for the interleaved
    path, whose block straddles a boundary.
    """
    if block_mode_on:
        return None
    if ctx.is_moe and not ctx.gate.block_copy:
        return None
    return tuple(_layer_spec(ctx.perf_db, n) for n in layer_nums)


def _emit_pre_attn_layers(ctx, bctx, layer_num, lines, power_acc, batch_tag='NONE'):
    _emit_sequence(ctx, bctx, layer_num,
                   _block_layers(ctx.perf_db, layer_num, "pre_attn"),
                   lines, power_acc, batch_tag)


def _emit_post_attn_layers(ctx, bctx, layer_num, lines, power_acc, batch_id_str, batch_tag='NONE'):
    # Attention post-processing, from this layer's own block.
    _emit_sequence(ctx, bctx, layer_num,
                   _block_layers(ctx.perf_db, layer_num, "post_attn"),
                   lines, power_acc, batch_tag)
    # MLP: whichever this layer runs. Resolved per layer, not per model --
    # DeepSeek-V3.2 and GLM-5 run a dense MLP for their first
    # ``first_k_dense_replace`` layers and MoE for the rest, and a model-level
    # flag emitted MoE for all of them.
    for layer_name in _block_layers(ctx.perf_db, layer_num, "mlp"):
        if layer_name == "moe":
            _emit_moe_block(ctx, bctx, lines, power_acc, layer_num,
                            batch_id_str, batch_tag)
        else:
            _emit_sequence(ctx, bctx, layer_num, [layer_name],
                           lines, power_acc, batch_tag)


def _build_transformer_block(ctx, bctx, layer_num, batch_tag, batch_id_str):
    """Build a complete transformer block. Returns (lines, PowerAccumulator)."""
    lines = []
    power_acc = PowerAccumulator([], [], 0, 0)
    _emit_pre_attn_layers(ctx, bctx, layer_num, lines, power_acc, batch_tag)
    _emit_post_attn_layers(ctx, bctx, layer_num, lines, power_acc, batch_id_str, batch_tag)
    return lines, power_acc


# ======================================================================
# Final layers and power helpers
# ======================================================================

def _layer_latency_for_power(ctx, bctx, layer_name):
    """Per-layer latency lookup used purely for power accounting; the
    trace writes its own values via _emit_layer and _emit_moe_block.
    """
    category = _layer_category(ctx.perf_db, layer_name)
    if category == "per_sequence":
        return _lookup_per_sequence(ctx.perf_db, layer_name, ctx.tp_size, bctx.lm_head_len)
    if category == "attention":
        return _lookup_attention_with_skew(
            ctx.perf_db, ctx.tp_size,
            bctx.prefill_chunk, bctx.kv_prefill,
            bctx.n_decode, bctx.kv_decode_mean, bctx.kv_decode_max,
            bctx.kv_decode_min, layer_name,
        )
    return _lookup_dense(ctx.perf_db, layer_name, ctx.tp_size, bctx.total_len)


def _emit_final_layers(ctx, bctx, rows, batch_tag='NONE'):
    """Emit the architecture's head layers (final_layernorm, lm_head,
    sampler — ordered per the yaml) and feed them into the power model.
    The last emitted layer routes its output to REMOTE so the Chakra
    converter places a MEM_STORE node back to CPU.
    """
    head_layers = _shared_layers(ctx.perf_db, "head")
    for i, layer_name in enumerate(head_layers):
        output_loc = f'REMOTE:{ctx.node_id}' if i == len(head_layers) - 1 else 'LOCAL'
        _emit_layer(ctx, bctx, layer_name, rows, None, batch_tag, output_loc=output_loc)

    if ctx.power_model is not None:
        for layer_name in head_layers:
            lat = _layer_latency_for_power(ctx, bctx, layer_name)
            ctx.power_model.add_npu_active_energy_consumption(ctx.hardware, ctx.node_id, lat, num_npus=ctx.tp_size)
            if get_device(ctx.placement, None, layer_name, "weights") != 'LOCAL':
                _, wt, _ = calculate_sizes(ctx.model, layer_name, bctx.total_len, parallel=ctx.tp_size, fp=ctx.fp)
                ctx.power_model.add_dram_energy_consumption(ctx.node_id, wt)


def _emit_pp_pd_power(ctx, bctx):
    """Emit pipeline parallelism and P/D sync power."""
    if ctx.power_model is None:
        return
    if ctx.pp_size > 1:
        pp_comm = bctx.total_len * ctx.config['hidden_size'] * (ctx.pp_size - 1)
        ctx.power_model.add_link_energy_consumption(ctx.node_id, pp_comm)
    if ctx.pd_type == 'prefill':
        kv_comm = bctx.total_len * ctx.config['hidden_size'] * ctx.fp
        out_size = bctx.lm_head_len * ctx.config['hidden_size'] * ctx.fp
        ctx.power_model.add_link_energy_consumption(ctx.node_id, kv_comm + out_size)


# ======================================================================
# _synthesize_trace (non-interleaved)
# ======================================================================

def _emit_prologue(ctx, bctx, rows, batch_tag='NONE'):
    """Emit prologue layers (typically just embedding). The first layer's
    input is routed from REMOTE to match the Chakra converter's
    MEM_LOAD node placement.
    """
    prologue_layers = _shared_layers(ctx.perf_db, "prologue")
    if not prologue_layers:
        return 0
    before = len(rows)
    for i, layer_name in enumerate(prologue_layers):
        input_loc = f'REMOTE:{ctx.node_id}' if i == 0 else 'LOCAL'
        _emit_layer(ctx, bctx, layer_name, rows, None, batch_tag, input_loc=input_loc)
    if ctx.power_model:
        for layer_name in prologue_layers:
            lat = _layer_latency_for_power(ctx, bctx, layer_name)
            ctx.power_model.add_npu_active_energy_consumption(
                ctx.hardware, ctx.node_id, lat, num_npus=ctx.tp_size)
            if get_device(ctx.placement, None, layer_name, "weights") != 'LOCAL':
                _, wt, _ = calculate_sizes(ctx.model, layer_name, bctx.total_len, fp=ctx.fp)
                ctx.power_model.add_dram_energy_consumption(ctx.node_id, wt)
    return len(rows) - before


def _synthesize_trace(hardware, model, config, tp_size, pp_size, local_ep, ep_total, pd_type, node_id, instance_id,
                      batch, max_len, placement, block_mode_on, gate,
                      enable_attn_offloading, power_model, pim_model, fp,
                      variant, kv_cache_dtype='auto',
                      runtime_max_num_batched_tokens=None, runtime_max_num_seqs=None,
                      tp_dim=None, ep_dim=None, dp_sum_total_len=0):
    ctx = _build_trace_ctx(hardware, model, config, tp_size, pp_size, local_ep, ep_total, node_id, fp,
                           placement, gate, enable_attn_offloading, power_model, pim_model, pd_type,
                           variant=variant, kv_cache_dtype=kv_cache_dtype,
                           runtime_max_num_batched_tokens=runtime_max_num_batched_tokens,
                           runtime_max_num_seqs=runtime_max_num_seqs,
                           tp_dim=tp_dim, ep_dim=ep_dim, dp_sum_total_len=dp_sum_total_len)
    bctx = _build_batch_ctx(batch, ctx)

    logger.info(
        "Batch #%d: model=%s num_reqs=%d total_len=%d req_ids=%s",
        batch.batch_id, model, len(batch.requests), batch.total_len,
        [r.id for r in batch.requests],
        extra={"node_id": node_id, "instance_id": instance_id},
    )

    # Line index at which each transformer block starts, used to cut
    # pipeline stages on block boundaries (see _pp_stage_boundaries).
    block_starts = []

    rows = []
    written = _emit_prologue(ctx, bctx, rows)

    # Transformer blocks
    num_layers = config['num_hidden_layers']

    # Build one block per distinct block *shape* and replay it for every layer
    # that shares it. A uniform stack builds once and replays N times, exactly
    # as before; a heterogeneous one builds once per shape, which is what makes
    # the replay correct rather than merely fast.
    built = {}
    for layer_num in range(num_layers):
        key = _block_copy_key(ctx, block_mode_on, layer_num)
        cached = built.get(key) if key is not None else None
        if cached is None:
            cached = _build_transformer_block(
                ctx, bctx, layer_num, 'NONE', str(batch.batch_id))
            if key is not None:
                built[key] = cached
        block_lines, block_power = cached
        block_starts.append(written)
        rows.extend(block_lines)
        written += len(block_lines)
        block_power.flush(ctx, enable_attn_offloading)

    # Final layers
    _emit_final_layers(ctx, bctx, rows)
    _emit_pp_pd_power(ctx, bctx)

    return rows, block_starts


# ======================================================================
# _synthesize_interleaved_trace (two sub-batches)
# ======================================================================

def _synthesize_interleaved_trace(hardware, model, config, tp_size, pp_size, local_ep, ep_total, pd_type, node_id, instance_id,
                                  batches, max_len, placement, block_mode_on, gate,
                                  enable_attn_offloading, power_model, pim_model, fp,
                                  variant, kv_cache_dtype='auto',
                                  runtime_max_num_batched_tokens=None, runtime_max_num_seqs=None,
                                  tp_dim=None, ep_dim=None, dp_sum_total_len=0):
    ctx = _build_trace_ctx(hardware, model, config, tp_size, pp_size, local_ep, ep_total, node_id, fp,
                           placement, gate, enable_attn_offloading, power_model, pim_model, pd_type,
                           variant=variant, kv_cache_dtype=kv_cache_dtype,
                           runtime_max_num_batched_tokens=runtime_max_num_batched_tokens,
                           runtime_max_num_seqs=runtime_max_num_seqs,
                           tp_dim=tp_dim, ep_dim=ep_dim, dp_sum_total_len=dp_sum_total_len)
    bctx1 = _build_batch_ctx(batches[0], ctx)
    bctx2 = _build_batch_ctx(batches[1], ctx)

    logger.info(
        "Sub-batch #%s: model=%s num_reqs=%d total_len=%d req_ids=%s",
        f"{batches[0].batch_id}.0", model, len(batches[0].requests), batches[0].total_len,
        [r.id for r in batches[0].requests],
        extra={"node_id": node_id, "instance_id": instance_id},
    )
    logger.info(
        "Sub-batch #%s: model=%s num_reqs=%d total_len=%d req_ids=%s",
        f"{batches[1].batch_id}.1", model, len(batches[1].requests), batches[1].total_len,
        [r.id for r in batches[1].requests],
        extra={"node_id": node_id, "instance_id": instance_id},
    )

    num_layers = config['num_hidden_layers']

    rows = []

    # PROLOGUE: Batch1 prologue + first pre-attn
    _emit_prologue(ctx, bctx1, rows, 'BATCH_1')

    pre_attn1_power = PowerAccumulator([], [], 0, 0)
    _emit_pre_attn_layers(ctx, bctx1, 0, rows, pre_attn1_power, 'BATCH_1')
    pre_attn1_power.flush(ctx, enable_attn_offloading)

    # Batch2 prologue + first pre-attn
    _emit_prologue(ctx, bctx2, rows, 'BATCH_2')

    pre_attn2_power = PowerAccumulator([], [], 0, 0)
    _emit_pre_attn_layers(ctx, bctx2, 0, rows, pre_attn2_power, 'BATCH_2')
    pre_attn2_power.flush(ctx, enable_attn_offloading)

    # MIDDLE LAYERS: interleaved post_attn + pre_attn
    middle_layers = num_layers - 1
    # Each interleaved block straddles a layer boundary -- this layer's
    # post_attn followed by the next layer's pre_attn -- so its shape depends
    # on both layers, and the cache key carries both.
    built = {}
    for layer_num in range(middle_layers):
        key = _block_copy_key(ctx, block_mode_on, layer_num, layer_num + 1)
        cached = built.get(key) if key is not None else None
        if cached is None:
            block_lines = []
            block_power = PowerAccumulator([], [], 0, 0)

            # Batch1: post_attn(current) + pre_attn(next)
            _emit_post_attn_layers(ctx, bctx1, layer_num, block_lines, block_power, f"{batches[0].batch_id}.0", 'BATCH_1')
            _emit_pre_attn_layers(ctx, bctx1, layer_num + 1, block_lines, block_power, 'BATCH_1')

            # Batch2: post_attn(current) + pre_attn(next)
            _emit_post_attn_layers(ctx, bctx2, layer_num, block_lines, block_power, f"{batches[1].batch_id}.1", 'BATCH_2')
            _emit_pre_attn_layers(ctx, bctx2, layer_num + 1, block_lines, block_power, 'BATCH_2')

            cached = (block_lines, block_power)
            if key is not None:
                built[key] = cached
        block_lines, block_power = cached
        rows.extend(block_lines)
        block_power.flush(ctx, enable_attn_offloading)

    # EPILOGUE: last layer post_attn + final layers
    last_power = PowerAccumulator([], [], 0, 0)
    _emit_post_attn_layers(ctx, bctx1, num_layers - 1, rows, last_power, f"{batches[0].batch_id}.0", 'BATCH_1')
    last_power.flush(ctx, enable_attn_offloading)
    _emit_final_layers(ctx, bctx1, rows, 'BATCH_1')

    last_power2 = PowerAccumulator([], [], 0, 0)
    _emit_post_attn_layers(ctx, bctx2, num_layers - 1, rows, last_power2, f"{batches[1].batch_id}.1", 'BATCH_2')
    last_power2.flush(ctx, enable_attn_offloading)
    _emit_final_layers(ctx, bctx2, rows, 'BATCH_2')

    _emit_pp_pd_power(ctx, bctx1)

    # Sub-batch interleaving leaves both sub-batches mid-block at every
    # group edge, so there is no single tensor to hand to the next stage.
    # generate_trace refuses the combination before we get here.
    return rows, []


# ======================================================================
# Pipeline-stage partitioning
# ======================================================================

def _pp_stage_boundaries(block_starts, pp_size):
    """Trace-line indices at which each pipeline stage after the first begins.

    Mirrors vLLM's ``get_pp_indices``: the transformer blocks are split
    evenly and any remainder goes to the stages *before* the last one,
    which also carries ``final_layernorm`` / ``lm_head`` / ``sampler``.

    Cutting only on block boundaries is load-bearing, not cosmetic. The
    Chakra converter sizes the stage-to-stage SEND from the last layer's
    ``output_size`` and the RECV from the first layer's ``input_size``,
    and ASTRA-Sim's callback tracker keys on chunk size — so the two must
    agree or the receiving NPU waits forever. A block boundary is the only
    place where they do: a block starts at ``layernorm`` and ends at
    ``down_proj`` / ``moe``, both of which carry the hidden state
    (``total_len * hidden_size * fp``). Inside a block they diverge, e.g.
    ``qkv_proj`` emits Q+K+V while ``rotary_emb`` declares only Q+K.

    Indices are relative to the list the converter partitions, i.e. after
    any leading ``kv_load`` / ``kv_evict`` lines have been stripped.
    """
    if pp_size <= 1:
        return []
    n_blocks = len(block_starts)
    if pp_size > n_blocks:
        raise ValueError(
            f"pp_size ({pp_size}) exceeds the model's transformer block count "
            f"({n_blocks}); a pipeline stage cannot be empty"
        )
    per_stage = n_blocks // pp_size
    partitions = [per_stage] * pp_size
    for i in range(2, n_blocks % pp_size + 2):
        partitions[-i] += 1
    boundaries = []
    acc = 0
    for count in partitions[:-1]:
        acc += count
        boundaries.append(block_starts[acc])
    return boundaries


# ======================================================================
# generate_trace() — public entry point
# ======================================================================

# Wrapper function that creates trace for an instance



def generate_trace(batch, hardware, tp_size, pp_size, local_ep, ep_total, pd_type=None, node_id=0, instance_id=0,
                   max_num_batched_tokens=2048, max_num_seqs=None,
                   placement={}, block_mode_on=False, expert_routing_policy="BALANCED",
                   enable_prefix_caching=False, enable_attn_offloading=False, power_model=None, pim_model=None,
                   enable_sub_batch_interleaving=False, fp=16, dtype=None, kv_cache_dtype='auto',
                   tp_dim=None, ep_dim=None, dp_sum_total_len=0, enable_block_copy=True, inputs_root=None):

    model = batch.model
    config = get_config(model)
    fp = fp // 8  # bit -> byte of floating point
    max_len = min(max_num_batched_tokens, config['max_position_embeddings'])
    variant = resolve_variant(dtype, kv_cache_dtype, config)

    # vllm: add load or eviction in the txt file
    load_size = batch.load
    evict_size = batch.evict

    if inputs_root is None:
        inputs_root = os.path.join(os.getcwd(), "inputs")
    output_path = input_path(
        inputs_root, "trace", hardware, batch.model,
        f"instance{instance_id}_batch{batch.batch_id}.txt",
    )

    # make trace — ``utils.num_experts`` knows every spelling the families
    # use, so a MoE checkpoint resolves to a live GateRouter whichever key it
    # declares. DeepSeek and GLM write ``n_routed_experts``, which this site
    # used to miss, leaving ctx.gate None and the MoE block unemittable.
    num_experts_cfg = utils_num_experts(config)
    if num_experts_cfg:
        gate = GateRouter(
            node_id, instance_id, num_experts_cfg,
            num_experts_per_tok=config.get('num_experts_per_tok', 1),
            routing_policy=expert_routing_policy,
            seed=42,
            block_copy=enable_block_copy,
        )
    else:
        gate = None

    # reset power model logs
    if power_model is not None:
        power_model.reset_log()

    # make trace
    synth_args = (hardware, model, config, tp_size, pp_size, local_ep, ep_total, pd_type, node_id, instance_id)
    # enable_prefix_caching is intentionally not forwarded: with chunked-prefill
    # semantics, the scheduler already encodes prefix hits via num_computed_tokens,
    # so trace synthesis no longer needs the flag.
    del enable_prefix_caching
    synth_kwargs = dict(placement=placement, block_mode_on=block_mode_on, gate=gate,
                        enable_attn_offloading=enable_attn_offloading,
                        power_model=power_model, pim_model=pim_model, fp=fp,
                        variant=variant, kv_cache_dtype=kv_cache_dtype,
                        runtime_max_num_batched_tokens=max_num_batched_tokens,
                        runtime_max_num_seqs=max_num_seqs,
                        tp_dim=tp_dim, ep_dim=ep_dim, dp_sum_total_len=dp_sum_total_len)
    if not enable_sub_batch_interleaving:
        rows, block_starts = _synthesize_trace(*synth_args, batch, max_len, **synth_kwargs)
    else:
        batches = _make_sub_batch(batch)
        if len(batches) < 2 or len(batches[0].requests) == 0 or len(batches[1].requests) == 0:
            rows, block_starts = _synthesize_trace(*synth_args, batch, max_len, **synth_kwargs)
        else:
            if pp_size > 1:
                raise ValueError(
                    "--enable-sub-batch-interleaving is not supported with pp_size > 1: "
                    "an interleaved trace leaves both sub-batches mid-block at every "
                    "group edge, so a pipeline stage has no single hidden state to pass on"
                )
            rows, block_starts = _synthesize_interleaved_trace(*synth_args, batches, max_len, **synth_kwargs)

    stage_boundaries = _pp_stage_boundaries(block_starts, pp_size)

    # vllm: prepend the load / evict rows
    mem = []
    if load_size != 0:
        load = ["kv_load", '0', 'LOCAL', '0', get_device(placement, None, None, 'kv_evict_loc'), str(load_size), 'LOCAL', '0', 'NONE', '0', 'NONE']
        mem.append(load)
        if power_model is not None:
            power_model.add_dram_energy_consumption(node_id, load_size)
    if evict_size != 0:
        evict = ["kv_evict", '0', 'LOCAL', '0', get_device(placement, None, None, 'kv_evict_loc'), str(evict_size), 'LOCAL', '0', 'NONE', '0', 'NONE']
        mem.append(evict)
        if power_model is not None:
            power_model.add_dram_energy_consumption(node_id, evict_size)

    if power_model is not None:
        power_model.print_log(node_id)

    # instance type
    if pd_type == None:
        instance_type = 'COLOCATED'
    elif pd_type == 'prefill':
        instance_type = 'PREFILL'
    elif pd_type == 'decode':
        instance_type = 'DECODE'
    else:
        raise ValueError(f"Unknown instance type {pd_type}.")

    header_line = f"{instance_type}\t\tmodel_parallel_NPU_group: {pp_size}"
    if stage_boundaries:
        header_line += "\t\tpp_stage_boundaries: " + ",".join(str(b) for b in stage_boundaries)

    # The rows go to generate_graph rather than to disk. It hashes them for
    # the converted-graph cache, and only a cache miss needs the text file at
    # all -- so on a hit nothing is formatted and nothing is written. Writing
    # it here unconditionally cost 0.616 ms per batch, 8,810 times on the
    # swe-bench MoE DP+EP example, for a file that was then read straight
    # back in the same process.
    return TraceData(header_line=header_line, rows=mem + rows, path=output_path)


# ======================================================================
# Trace file writer
# ======================================================================

@dataclass
class TraceData:
    """A synthesized trace, before it is turned into text.

    ``rows`` are field tuples straight from the emitters: eleven fields for a
    layer, one for an EXPERT/PIM marker. ``path`` is where the ``.txt`` goes
    if anything asks for it.
    """
    header_line: str
    rows: list
    path: str


_TRACE_ROW_FIELDS = 11

# One-shot guard, see _write_trace.
_row_format_checked = False


def indexed_cols(rows):
    """The field lists the Chakra converter sees after parsing the trace text.

    Mirrors _write_trace's two cases. A layer row gets its final row index
    appended to the name and keeps its ten other fields. A marker row becomes
    the tokens of its text -- which is how it comes back today, once the
    formatter has padded it into the name column and the reader has split the
    line on whitespace.

    Kept beside _write_trace because the two have to agree exactly: the index
    is positional and depends on the kv_load/kv_evict rows already sitting at
    the front. Their agreement is not assumed -- the graphs built from each
    path are byte-compared.
    """
    out = []
    for i, row in enumerate(rows):
        if len(row) == _TRACE_ROW_FIELDS:
            out.append([f'{row[0]}_{i}', *row[1:]])
        elif len(row) == 1:
            out.append(row[0].split())
        else:
            raise ValueError(
                f"trace row {i} has {len(row)} fields; expected "
                f"{_TRACE_ROW_FIELDS} for a layer row or 1 for a marker. "
                f"Row: {row!r}"
            )
    return out


def write_trace(trace):
    """Materialise a TraceData as text, for inspection.

    Nothing in the pipeline reads it any more -- the converter takes the rows
    -- so this only runs when --save-trace-text asks for it. Creates the
    directory because it is now the only thing that writes there.
    """
    os.makedirs(os.path.dirname(trace.path), exist_ok=True)
    _write_trace(trace.path, trace.header_line, trace.rows)


def _write_trace(output_path, header_line, rows):
    """Write the trace file in a single pass.

    Rows arrive as field tuples straight from the emitters: eleven fields
    for a layer, one for an ``EXPERT``/``PIM`` marker whose text occupies
    the name column with the rest blank. Layer names get their final row
    index appended here, and here only, because that index depends on how
    many ``kv_load``/``kv_evict`` rows were prepended -- which is not known
    until every row exists.

    This replaces a write -> read back -> ``re.findall`` per line -> rewrite
    round trip. The round trip existed purely to renumber, and it cost a
    regex split per row on top of writing the file, reading it, and writing
    it again. A 145-iteration run spent 42,341 ``re.findall`` calls on it.

    Marker rows are told apart by field count rather than by the old
    ``"EXPERT" not in name and "PIM" not in name`` substring test, which
    would have mislabelled any future layer whose canonical name contained
    either word.
    """
    global _row_format_checked

    lines = []
    for i, row in enumerate(rows):
        if len(row) == _TRACE_ROW_FIELDS:
            lines.append(formatter(f'{row[0]}_{i}', *row[1:]))
        elif len(row) == 1:
            lines.append(formatter(row[0], *([''] * 10)))
        else:
            raise ValueError(
                f"trace row {i} has {len(row)} fields; expected "
                f"{_TRACE_ROW_FIELDS} for a layer row or 1 for a marker. "
                f"Row: {row!r}"
            )

    # Dropping the read-back also drops the place where a formatted row
    # that had swallowed its own column separator used to surface -- as a
    # short field list, which then raised a TypeError on the rewrite. That
    # is how the 15-character ``ALLREDUCE:1,0,0`` case was caught. The
    # explicit separators in utils._FMT now make a merge unrepresentable,
    # so this is a regression guard on _FMT rather than the primary
    # defence, and it runs on the *formatted* text because the field tuple
    # is never the thing at fault: a correct 11-element row is exactly what
    # formatting merged. Checking every row of every trace would cost a
    # split per row and give back the speed this change bought, so it
    # validates the first trace written in the process, which is enough to
    # fail immediately on any run.
    if not _row_format_checked:
        _row_format_checked = True
        for i, (row, line) in enumerate(zip(rows, lines)):
            if len(row) != _TRACE_ROW_FIELDS:
                continue
            got = len(line.split())
            if got != _TRACE_ROW_FIELDS:
                raise ValueError(
                    f"trace row {i} formatted to {got} whitespace-separated "
                    f"fields instead of {_TRACE_ROW_FIELDS}: a value filled "
                    f"its column and merged with the next. Widen the column "
                    f"in utils._FMT. Row: {row!r}"
                )

    with open(output_path, 'w') as f:
        f.write(header_line + "\n")
        f.write(str(len(rows)) + '\n')
        f.write(header())
        f.writelines(lines)


# ======================================================================
# generate_event() — preserved exactly
# ======================================================================


# generate event for first request arrival
def generate_event(alarm, inputs_root=None):
    """The one-layer trace that idles every NPU until ``alarm``.

    Returns a TraceData like generate_trace, so generate_graph converts it
    from rows like any other batch. It used to write its file directly, and
    being the one trace that still arrived as text kept a whole second path
    alive in generate_graph -- file hashing, ``convert()``, and a per-batch
    unlink -- for a call that happens once per run.

    Fields are strings for the same reason the emitters produce strings: the
    row digest joins them, and the text writer formats them.
    """
    if inputs_root is None:
        inputs_root = os.path.join(os.getcwd(), "inputs")
    row = (f'event_{alarm}ns', str(alarm), 'REMOTE', '0', 'LOCAL', '0',
           'REMOTE', '0', 'NONE', '0', 'NONE')
    return TraceData(
        header_line="EVENT",
        rows=[row],
        path=input_path(inputs_root, "trace", "event_handler.txt"),
    )


# ======================================================================
# Helper Functions for PIM Scheduling
# ======================================================================

# Greedy Min-Load Bin Packing Algorithm for PIM Attention Load Balancing
def _attn_load_balancer(requests, tp_size, pim_channels=0, channel_split=1):

    # Sort all requests by input length in descending order (longest first)
    requests = sorted(requests, key=lambda r: r.input, reverse=True)
    prefill_len = 0
    decode_lens = [[] for _ in range(pim_channels)]
    decode_loads = [0 for _ in range(pim_channels)]

    # Greedy load balancing with separate prefill / decode loads
    for req in requests:

        if req.is_init:
            # For prefill, just accumulate total length
            prefill_len += req.input
        else:
            # For decode with attn offloading, choose the PIM channel with the smallest decode load
            for channel in range(channel_split): # one channel can handle multiple heads if load is still small
                pim_id = min(range(pim_channels), key=lambda i: decode_loads[i])
                decode_lens[pim_id].append(req.input)
                decode_loads[pim_id] += req.input

    return prefill_len, decode_lens


# ======================================================================
# _make_sub_batch() — chunked-prefill aware sub-batch split
# ======================================================================

# spliting one batch into sub-batches to do sub-batch interleaving while using PIM
def _make_sub_batch(batch):
    if len(batch.requests) == 1:
        return [batch]

    # Read the split off the parent batch rather than off the request objects.
    # The scheduler advances num_computed_tokens when it forms the batch (as vLLM
    # does), so a request's own counter already reflects the state *after* this
    # iteration; q_list / k_list carry the values this iteration actually ran
    # with, aligned with batch.requests.
    per_req = {
        req.id: (q, k)
        for req, q, k in zip(batch.requests, batch.q_list, batch.k_list)
    }

    def compute_tokens(req):
        return per_req[req.id][0]

    # Greedy split: longest per-iteration compute first, assign to lighter side.
    reqs = sorted(batch.requests, key=compute_tokens, reverse=True)
    req_groups = [[], []]
    loads = [0, 0]
    for req in reqs:
        target = 0 if loads[0] <= loads[1] else 1
        loads[target] += compute_tokens(req)
        req_groups[target].append(req)

    sub_batches = []
    for i, sub_reqs in enumerate(req_groups):
        sub_reqs.sort(key=lambda r: r.arrival)

        total_len = 0
        kv_len = 0
        num_prefill = 0
        num_decode = 0
        q_list = []
        k_list = []
        prefill_q_list = []
        prefill_k_list = []
        decode_k_list = []

        for req in sub_reqs:
            chunk, kv_before = per_req[req.id]
            total_len += chunk
            q_list.append(chunk)
            k_list.append(kv_before)
            # More than one token is a prefill chunk, exactly one is a decode --
            # the same classification the parent batch used, and the one the
            # attention profile axes expect.
            if chunk > 1:
                prefill_q_list.append(chunk)
                prefill_k_list.append(kv_before)
                num_prefill += 1
            else:
                kv_len += kv_before
                decode_k_list.append(kv_before)
                num_decode += 1

        # evict/load are counted once for the original batch; attach to sub-batch 0 only.
        evict, load = (batch.evict, batch.load) if i == 0 else (0, 0)
        sub = Batch(
            batch.batch_id, batch.model,
            total_len, kv_len,
            q_list, k_list, num_prefill,
            num_decode, prefill_q_list,
            prefill_k_list, decode_k_list,
            0, 0, evict, load,
        )
        sub.requests.extend(sub_reqs)
        sub_batches.append(sub)

    return sub_batches
