"""CSV + meta.yaml writing.

Every Point produced by a Category ultimately flows through a
``DedupSink`` — an in-memory accumulator that averages duplicate
measurements keyed on everything except ``microseconds``. On
``flush()`` the sink writes a deterministic, sorted CSV.

Also here:
  * ``persist_meta``: writes the per-variant ``meta.yaml`` (one file
    per variant folder, not per tp).
  * ``replicate_tp_stable``: post-pass that copies tp_stable layers
    from ``tp1/*.csv`` into every other ``tp{N}/*.csv`` so the
    simulator doesn't need special-case logic at lookup time.
"""

from __future__ import annotations

import csv
import datetime
import os
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from profiler import __version__ as profiler_version
from profiler.core import logger as log
from profiler.core.categories import (
    AttentionPoint,
    Category,
    DensePoint,
    ExpertPoint,
    Point,
    SequencePoint,
)
from profiler.core.config import (
    Architecture,
    ProfileArgs,
    architecture_hash,
)


# ---------------------------------------------------------------------------
# DedupSink — the CSV-producing accumulator
# ---------------------------------------------------------------------------

class DedupSink:
    """Accumulates Points, averages duplicates, flushes a sorted CSV.

    Uses dataclass field names to derive the CSV schema, so adding a
    new axis to a Point type is a zero-writer-code change.

    Duplicate detection key = every field except ``microseconds``.
    """

    def __init__(self, out_path: Path, key_fields: list[str]) -> None:
        """
        Args:
            out_path: full CSV path, e.g. ``perf/.../tp1/dense.csv``.
            key_fields: the non-time field names to index by. Must be
                a subset of the Point dataclass's field names; the
                remaining field must be ``microseconds``.
        """
        self.out_path = out_path
        self.key_fields = key_fields
        # key tuple -> (running_sum, count)
        self._bucket: dict[tuple, tuple[float, int]] = {}
        # Track fieldnames in the order we first see them so the CSV
        # column order is deterministic (matches insertion in the
        # writer).
        self._fieldnames: list[str] | None = None

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def coalesce(self, point: Point) -> None:
        """Accept one Point; average on key collision."""
        d = asdict(point)
        if self._fieldnames is None:
            # Preserve dataclass field declaration order; put time_us
            # last regardless of where it appears in the dataclass.
            ordered = [f for f in d.keys() if f != "microseconds"]
            ordered.append("microseconds")
            self._fieldnames = ordered

        key = tuple(d[f] for f in self.key_fields)
        us = float(d["microseconds"])
        prev = self._bucket.get(key)
        if prev is None:
            self._bucket[key] = (us, 1)
        else:
            self._bucket[key] = (prev[0] + us, prev[1] + 1)

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    def preload(self) -> int:
        """Seed the in-memory bucket from any existing CSV at
        ``out_path``. Returns the number of rows ingested (0 when
        the file is missing or empty).

        Used for resume mode: after preload, ``prior_shot_keys()``
        reports which shot identities are already covered so the
        firing loop can skip them. The flush at the end of the run
        will then contain both preserved and newly-measured rows.
        """
        if not self.out_path.exists():
            return 0
        count = 0
        with self.out_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            field_order = reader.fieldnames or []
            if not field_order:
                return 0
            # A prior file that predates one of this sink's key fields cannot
            # be preloaded at all -- every row would fail the key build below --
            # and adopting its column order would pin ``flush`` to a schema too
            # narrow for the rows about to be written. That is not
            # hypothetical: adding the ``ep`` key to moe.csv made flush raise
            # ``dict contains fields not in fieldnames: 'ep'`` *after* the whole
            # sweep had run, and the file had already been truncated.
            missing = [k for k in self.key_fields if k not in field_order]
            if missing:
                log.info(
                    "%s predates the %s column(s); its rows cannot be keyed, "
                    "so the sweep starts clean rather than resuming",
                    self.out_path.name, ", ".join(missing),
                )
                return 0
            # Establish ordering now so flush preserves the original
            # schema even when no new rows come in.
            if self._fieldnames is None:
                ordered = [f for f in field_order if f != "time_us"]
                ordered.append("microseconds")
                self._fieldnames = ordered
            for row in reader:
                try:
                    us = float(row["time_us"])
                except (KeyError, ValueError):
                    continue
                key_parts: list[Any] = []
                bad = False
                for kf in self.key_fields:
                    v = row.get(kf)
                    if v is None:
                        bad = True
                        break
                    # Preserve numeric types so subsequent shot-key
                    # matching compares like-with-like.
                    try:
                        if "." in v or "e" in v or "E" in v:
                            key_parts.append(float(v))
                        else:
                            key_parts.append(int(v))
                    except (TypeError, ValueError):
                        key_parts.append(v)
                if bad:
                    continue
                key = tuple(key_parts)
                # Single-sample bucket entry: preserve the exact value.
                self._bucket[key] = (us, 1)
                count += 1
        return count

    def prior_shot_keys(self, layer_column: str = "layer") -> set[tuple]:
        """Set of shot-level identity keys already present in the
        bucket (i.e., after preload). The shot identity is the row
        key with the ``layer`` field removed — multiple rows of the
        same shot (one per layer) collapse to one entry.
        """
        if layer_column in self.key_fields:
            idx = self.key_fields.index(layer_column)
            return {
                tuple(v for i, v in enumerate(k) if i != idx)
                for k in self._bucket.keys()
            }
        return set(self._bucket.keys())

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Write the accumulated rows to ``out_path`` and clear state.

        CSV conventions:
          * Rows sorted lexicographically by key fields (deterministic
            diffs; shape-friendly for human skim).
          * ``microseconds`` column renamed to ``time_us`` on write.
          * Floats emitted with 6 sig figs (``%.6g``) to keep files
            readable while preserving resolution.
        """
        if not self._bucket:
            log.warning("nothing to write to %s", self.out_path)
            return

        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        # Produce rows in sort order.
        rows: list[dict[str, Any]] = []
        for key in sorted(self._bucket.keys()):
            total_us, count = self._bucket[key]
            avg_us = total_us / count
            row = {f: v for f, v in zip(self.key_fields, key)}
            row["time_us"] = _format_time_us(avg_us)
            rows.append(row)

        assert self._fieldnames is not None
        header = self._fieldnames.copy()
        # Swap 'microseconds' → 'time_us' for the CSV header.
        header = [
            "time_us" if f == "microseconds" else f
            for f in header
        ]

        # Write beside the target and rename. Opening ``out_path`` directly
        # truncates it before the first row is validated, so one bad row
        # destroys a bundle that took hours -- which is exactly what happened
        # when the moe key grew an ``ep`` field.
        tmp = self.out_path.with_suffix(self.out_path.suffix + ".tmp")
        try:
            with tmp.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                writer.writerows(rows)
            tmp.replace(self.out_path)
        finally:
            tmp.unlink(missing_ok=True)

        log.debug("wrote %d rows → %s", len(rows), self.out_path)
        self._bucket.clear()

    # ------------------------------------------------------------------
    # Convenience: attach a human-friendly 'layer' prefix
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self.out_path


def _format_time_us(v: float) -> str:
    """6 sig figs, no scientific notation for typical values."""
    # %.6g drops trailing zeros naturally and keeps things readable.
    s = f"{v:.6g}"
    return s


# ---------------------------------------------------------------------------
# Sink factory per category
# ---------------------------------------------------------------------------

def sink_for(category: Category, out_dir: Path) -> DedupSink:
    """Build a DedupSink pre-configured for a given category's schema."""
    csv_path = out_dir / category.sink_filename
    key_fields = _KEY_FIELDS_BY_CATEGORY[category.name]
    return DedupSink(out_path=csv_path, key_fields=key_fields)


# The only place where category→key-field mapping is specified. Adding
# a new Point type means adding a line here.
_KEY_FIELDS_BY_CATEGORY: dict[str, list[str]] = {
    "dense": ["layer", "tokens"],
    "per_sequence": ["layer", "sequences"],
    "attention": ["layer", "prefill_chunk", "kv_prefill", "n_decode",
                  "kv_decode", "decode_q_len"],
    "linear_attention": ["layer", "prefill_tokens", "n_decode"],
    "moe": ["ep", "tokens", "activated_experts"],
    "mtp": ["layer", "sequences"],
}


# ---------------------------------------------------------------------------
# meta.yaml — per-variant session metadata
# ---------------------------------------------------------------------------

# CSV filename under tp{N}/ that holds the per-bucket alpha table.
# The meta.yaml skew_fit.per_tp[tp] block points at this file instead
# of inlining the (usually 1-2k) bucket rows as a YAML mapping.
_SKEW_FIT_CSV_NAME = "skew_fit.csv"


def _geometric_spec(values) -> Any:
    """Compact string for a geometric (doubling) sequence.

    Returns ``"<start>-<end> x<factor>"`` when ``values`` is a clean
    geometric progression; ``"0, <start>-<end> x<factor>"`` when it
    starts with a 0 sentinel followed by a geometric tail; and a list
    copy otherwise (for irregular sequences the spec string would be
    misleading, so the literal values are preserved).

    Tolerates a small amount of round-off in the factor (up to 2%) and
    a final non-geometric value clamped to a user cap.
    """
    vals = list(values)
    if not vals:
        return []
    # Handle leading 0 sentinel (common in pc / kp / attention_grid axes).
    prefix = None
    if vals[0] == 0 and len(vals) > 1 and vals[1] != 0:
        prefix = 0
        tail = vals[1:]
    else:
        tail = vals
    if len(tail) < 2:
        return list(vals)
    if any(v <= 0 for v in tail):
        return list(vals)
    # Derive factor from the first step and verify the rest.
    r0 = tail[1] / tail[0]
    if r0 <= 1.0:
        return list(vals)
    for a, b in zip(tail[:-2], tail[1:-1]):
        if abs((b / a) - r0) / r0 > 0.02:
            return list(vals)
    # Allow the last step to fall short (clamp-to-max behaviour in
    # both _doubling and _geometric_grid) — that's still a clean spec.
    last_ratio = tail[-1] / tail[-2]
    if last_ratio > r0 * 1.02:
        return list(vals)
    factor = f"x{int(round(r0))}" if abs(r0 - round(r0)) < 1e-3 \
        else f"x{r0:.3g}"
    core = f"{tail[0]}-{tail[-1]} {factor}"
    return core if prefix is None else f"{prefix}, {core}"


def _skew_fit_block(variant_root: Path, tp_degrees: list[int]) -> dict:
    """Fit alpha per TP from each tp{N}/skew.csv and return a meta
    block. Empty / missing skew.csv → ``{"enabled": False}``.

    The per-bucket alpha table (potentially 1k+ rows per TP) is written
    to ``tp{N}/skew_fit.csv``; the returned dict keeps only a per-TP
    summary (method, n_samples, alpha_default, self-eval errors, and a
    pointer to the CSV).

    ``bucket_axes`` — derived from the skew.csv data by
    ``profiler.fit_alpha`` so the bins adapt to whatever axis coverage
    the profile actually contains — is promoted to the block top level
    when all TPs agree (the common case, since all TPs share the same
    profile grid). If TPs disagree, each entry keeps its own axes; the
    simulator handles both shapes.
    """
    from profiler.core.fit_alpha import fit_alpha_per_tp
    fit = fit_alpha_per_tp(variant_root, tp_degrees)
    if not fit.get("enabled"):
        return fit

    per_tp_in = fit.get("per_tp", {})
    per_tp_out: dict[int, dict[str, Any]] = {}
    axes_seen: list[Any] = []

    for tp, entry in per_tp_in.items():
        axes_seen.append(entry.get("bucket_axes"))
        tp_dir = variant_root / f"tp{int(tp)}"
        csv_path = tp_dir / _SKEW_FIT_CSV_NAME
        _write_skew_fit_csv(csv_path, entry)
        summary = {
            "method": entry.get("method"),
            "n_samples": entry.get("n_samples"),
            "alpha_default": entry.get("alpha_default"),
            # Per-kernel fallback. An unfitted bucket on a sparse-attention
            # layer must not inherit the dense kernel's alpha.
            "alpha_default_by_layer": entry.get("alpha_default_by_layer"),
            "bucket_table": f"tp{int(tp)}/{_SKEW_FIT_CSV_NAME}",
        }
        for k in ("rel_err_p50", "rel_err_p90", "rel_err_p99", "signed_mean"):
            if k in entry:
                summary[k] = entry[k]
        per_tp_out[int(tp)] = summary

    out: dict[str, Any] = {"enabled": True}
    if axes_seen and all(a == axes_seen[0] for a in axes_seen) and axes_seen[0]:
        # All TPs derived the same axes — promote once to avoid
        # duplicating the block in meta.yaml.
        out["bucket_axes"] = axes_seen[0]
    else:
        # TPs disagree (e.g. one TP was profiled at a different sweep
        # width); keep per-TP axes so the simulator resolves each one
        # against its own entry.
        for tp, axes in zip(per_tp_in.keys(), axes_seen):
            if axes is not None:
                per_tp_out[int(tp)]["bucket_axes"] = axes
    out["per_tp"] = per_tp_out
    return out


def _write_skew_fit_csv(csv_path: Path, fit_entry: dict) -> None:
    """Write one TP's per-bucket alpha table to CSV.

    Bucket keys in the fit dict are pipe-delimited strings produced by
    ``profiler.fit_alpha._bucket_key``. We split them back into their
    components for analyst-friendly columns. The simulator reassembles
    the key from these columns.

    Six parts means the key carries the attention kernel's name as a prefix,
    which is how a sparse-attention model keeps its indexer's alpha off its
    attention kernel; five means a fit that predates the prefix, all of it the
    ``attention`` kernel.
    """
    alphas = fit_entry.get("alpha_by_bucket") or {}
    counts = fit_entry.get("n_by_bucket") or {}
    if not alphas:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for key, alpha in alphas.items():
        parts = key.split("|")
        layer = "attention"
        if len(parts) == 6:
            layer, *parts = parts
        if len(parts) != 5 or not parts[0].startswith("pc="):
            # Don't silently drop malformed rows — keep the raw key.
            rows.append({
                "layer": layer,
                "pc": "", "n_label": "", "skew_rate_label": "",
                "kv_big_label": "", "kp_label": "",
                "alpha": float(alpha),
                "n_samples": int(counts.get(key, 0)),
                "raw_key": key,
            })
            continue
        pc_token, n_label, sr_label, kvb_label, kp_label = parts
        try:
            pc = int(pc_token.split("=", 1)[1])
        except (IndexError, ValueError):
            pc = pc_token
        rows.append({
            "layer": layer,
            "pc": pc,
            "n_label": n_label,
            "skew_rate_label": sr_label,
            "kv_big_label": kvb_label,
            "kp_label": kp_label,
            "alpha": float(alpha),
            "n_samples": int(counts.get(key, 0)),
        })

    rows.sort(key=lambda r: (
        r["layer"],
        r["pc"] if isinstance(r["pc"], int) else 1 << 30,
        r["n_label"], r["skew_rate_label"],
        r["kv_big_label"], r["kp_label"],
    ))
    fieldnames = [
        "layer", "pc", "n_label", "skew_rate_label", "kv_big_label",
        "kp_label", "alpha", "n_samples",
    ]
    # Preserve the optional raw_key column if any row needed it.
    if any("raw_key" in r for r in rows):
        fieldnames.append("raw_key")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _skew_meta_block(args) -> dict:
    """Record the skew grid configuration actually used.

    Delegates to profiler.skew._build_grid for the dynamic axes and
    emits each doubling axis as a compact ``"<start>-<end> x<factor>"``
    spec string. Irregular axes (``ratio``) stay as literal lists.
    Raw data is in tp<N>/skew.csv; this block tells the simulator
    which grid density produced them.
    """
    if args.skip_skew:
        return {"enabled": False}
    # Import lazily to avoid a circular profiler import at module load.
    from profiler.core.skew import _build_grid, _SKEW_REP

    class _FakeLimits:
        max_num_batched_tokens = args.max_num_batched_tokens or 2048
        max_num_seqs = args.max_num_seqs or 256
    grid = _build_grid(args, _FakeLimits())
    return {
        "enabled": True,
        # Geometric factors actually used. 2.0 (doubling) is the
        # default; higher values coarsen the sweep and speed it up.
        "factors": {
            "n": args.skew_n_factor,
            "pc": args.skew_pc_factor,
            "kp": args.skew_kp_factor,
            "kvs": args.skew_kvs_factor,
        },
        "grid": {
            "n": _geometric_spec(grid["n"]),
            "ratio": list(grid["ratio"]),
            "pc": _geometric_spec(grid["pc"]),
            "kp": _geometric_spec(grid["kp"]),
            "kvs": _geometric_spec(grid["kvs"]),
            "skew_rep": _SKEW_REP,
        },
    }


def _attention_grid_spec(args, effective_mnbt: int, effective_msq: int) -> dict:
    """Compact spec of the attention sweep axes actually fired.

    The knobs (max_kv / chunk_factor / kv_factor) are retained so the
    old field names still carry information; the three spec strings
    give a human-readable view of the values ``AttentionCategory``
    walked during profiling.
    """
    from profiler.core.categories import (
        _geometric_grid, _ATTN_CHUNK_START,
        _ATTN_N_DECODE_START, _ATTN_KV_START,
    )
    chunks = _geometric_grid(
        effective_mnbt, _ATTN_CHUNK_START,
        factor=args.attention_chunk_factor,
    )
    n_dec = _geometric_grid(
        effective_msq, _ATTN_N_DECODE_START,
    )
    kv = _geometric_grid(
        args.attention_max_kv, _ATTN_KV_START,
        factor=args.attention_kv_factor,
    )
    return {
        "max_kv": args.attention_max_kv,
        "chunk_factor": args.attention_chunk_factor,
        "kv_factor": args.attention_kv_factor,
        "chunks": _geometric_spec(chunks),
        "n_decode": _geometric_spec(n_dec),
        "kv": _geometric_spec(kv),
        # The fifth axis. Not derivable from the other four -- a q > 1 sweep
        # yields no pure-prefill shot, so the row count alone cannot tell you
        # which query lengths were fired -- and a bundle whose CSV holds five
        # of them while the meta records none reads as a q=1 sweep.
        "decode_q_lens": sorted({max(1, int(q))
                                 for q in args.attention_decode_q_lens}),
    }


def _category_provenance(
    prior: dict[str, Any],
    measured: tuple[str, ...],
    version: str,
    stamp: str,
) -> dict[str, Any] | None:
    """``{category: {vllm_version, profiled_at}}``, accumulated across refreshes.

    A bundle is not necessarily one measurement session: a slice refresh
    rewrites one category and leaves the rest alone, and those rest may have
    been measured under a different vLLM. Recording it per category is the only
    way the file can say so.
    """
    out: dict[str, Any] = {}
    prior_block = prior.get("category_provenance")
    if isinstance(prior_block, dict):
        out.update({str(k): dict(v) for k, v in prior_block.items()
                    if isinstance(v, dict)})
    elif prior:
        # First time: everything already in the file belongs to the run that
        # wrote it, whose version and timestamp are the prior top-level ones.
        seed = {"vllm_version": prior.get("vllm_version"),
                "profiled_at": prior.get("profiled_at")}
        if seed["vllm_version"]:
            for name in _KEY_FIELDS_BY_CATEGORY:
                out[name] = dict(seed)
    for name in measured:
        out[str(name)] = {"vllm_version": version, "profiled_at": stamp}
    return out or None


def persist_meta(
    args: ProfileArgs,
    arch_path: Path,
    engine_kwargs_used: dict[str, Any] | None,
    variant_root: Path,
    limits: Any = None,
    *,
    records_engine: bool = True,
    records_attention_grid: bool = True,
    measured_categories: tuple[str, ...] = (),
) -> None:
    """Write ``variant_root/meta.yaml`` describing the profile session.

    Written once per variant; a ``slice`` refresh rewrites it in place. What a
    refresh may rewrite is the point of the two flags, because the file
    describes more than any one refresh measures:

    ``records_engine`` -- ``engine_effective`` and ``engine_resolved`` describe
    the **deepest main engine**, the one whose shapes the simulator will run
    (``run_full`` picks it deliberately: "the deepest engine is the one whose
    shapes describe the stack"). An MTP refresh boots a different engine
    entirely -- one extra full-attention layer and a conv state widened by
    ``num_speculative_tokens`` -- so its numbers describe a model nobody
    simulates. Letting it write them put ``block_size: 800`` and half the KV
    cache into Qwen3.8-27B's bundle where the main engine resolves 784 and
    19,839,182, and the simulator reads that block size whenever
    ``--block-size`` is omitted. A shrunk single-category engine is not
    authoritative either, for the same reason.

    ``records_attention_grid`` -- only a run that swept attention knows which
    axes were fired. A ``per_sequence`` refresh regenerating the block from its
    own defaults would have reported DeepSeek-V3.2's five query lengths as one
    and its 163,830 kv reach as 163,838.

    ``measured_categories`` names what this run actually fired, which is what
    ``category_provenance`` records. The top-level ``vllm_version`` and
    ``profiled_at`` describe the *most recent* refresh and so cannot describe a
    bundle a refresh only partly rewrote -- adding the EP axis to
    Qwen3-30B-A3B's MoE stamped the whole file 0.28.0 while its dense,
    attention and per_sequence rows stayed 0.19.0 measurements. Which matters:
    vLLM 0.28 restructured MoE substantially, and the two versions agree to
    within noise on decode-sized batches but differ by 16-26% at 2048 tokens.

    Anything not recorded is carried over from the file, not dropped.
    """
    prior = _prior_meta(variant_root)
    # ``engine_kwargs_used`` carries the BUMPED ``max_num_batched_tokens``
    # (see engine.fuse_engine_kwargs). Record the LOGICAL value in
    # meta.yaml so the simulator's runtime-vs-profiled bound comparison
    # and any human inspection see the user-intended cap.
    engine_effective = dict(engine_kwargs_used or {})
    try:
        engine_effective["max_num_batched_tokens"] = (
            int(engine_effective["max_num_batched_tokens"])
            - int(engine_effective["max_num_seqs"])
        )
    except (KeyError, TypeError, ValueError):
        pass

    # Effective sweep caps for attention: engine_effective holds the
    # logical (un-bumped) MNBT; MSQ comes from the same block or falls
    # back to the profiler's default.
    try:
        eff_mnbt = int(engine_effective.get("max_num_batched_tokens") or 2048)
    except (TypeError, ValueError):
        eff_mnbt = 2048
    try:
        eff_msq = int(engine_effective.get("max_num_seqs") or 256)
    except (TypeError, ValueError):
        eff_msq = 256

    meta = {
        "profiler_version": profiler_version,
        "vllm_version": _vllm_version(),
        "cuda_version": _cuda_version(),
        "gpu": _gpu_name(),
        "hardware": args.hardware,
        "profiled_at": _utcnow_iso(),
        "architecture": args.architecture,
        "architecture_sha256": architecture_hash(arch_path),
        "model": args.model,
        "variant": args.effective_variant,
        "tp_degrees": args.tp_degrees,
        # Per-category provenance. Seeded from the prior file's top-level
        # version/timestamp for every category this run did not measure, so the
        # first partial refresh of an older bundle labels its untouched
        # categories correctly rather than inheriting the new run's version.
        "category_provenance": _category_provenance(
            prior, measured_categories, _vllm_version(), _utcnow_iso()),
        "engine_effective": (
            _stringify(engine_effective) if records_engine
            else prior.get("engine_effective") or _stringify(engine_effective)
        ),
        # What the engine actually SETTLED ON, which is not always what was
        # asked for. vLLM derives the KV block size from the backend's
        # ``get_supported_kernel_block_sizes`` and from hybrid page
        # unification: MiniMax-M3's sparse backend accepts only 128, and a
        # gated-DeltaNet stack enlarges the attention block until an attention
        # page costs at least as many bytes as a mamba state page (784 on
        # Qwen3.8-27B, against the 16 requested). The simulator reads this
        # rather than reimplementing vLLM's backend selection, and a run whose
        # --block-size disagrees is simulating a configuration vLLM cannot
        # serve.
        # Keyed by TP under ``per_tp``: on a hybrid stack the resolved block
        # size is per-rank, so it is a per-TP fact, and a single value would be
        # whichever TP happened to run last.
        "engine_resolved": _engine_resolved_block(
            variant_root, limits if records_engine else None),
        # Attention-grid shape knobs + compact spec of the values the
        # sweep actually visited. Simulator uses the knobs to recognise
        # which density produced the CSVs; humans get the axes too.
        "attention_grid": (
            _attention_grid_spec(args, eff_mnbt, eff_msq)
            if records_attention_grid
            else prior.get("attention_grid")
            or _attention_grid_spec(args, eff_mnbt, eff_msq)
        ),
        "measurement_iterations": args.measurement_iterations,
        "skew_profile": _skew_meta_block(args),
        "skew_fit": _skew_fit_block(variant_root, args.tp_degrees),
    }
    variant_root.mkdir(parents=True, exist_ok=True)
    out = variant_root / "meta.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.dump(meta, f, Dumper=_CompactDumper, sort_keys=False)
    log.debug("wrote meta.yaml → %s", out)


# ---------------------------------------------------------------------------
# YAML dumper that keeps short scalar lists on one line
# ---------------------------------------------------------------------------


class _CompactDumper(yaml.SafeDumper):
    """SafeDumper that emits lists of primitives in flow style.

    Block-style lists balloon meta.yaml (every bin / label / tp_degree
    on its own line). Flow style collapses them to a single line while
    still being valid YAML and round-trippable by safe_load.
    """


def _represent_list(dumper, data):
    flow = all(
        isinstance(x, (int, float, str, bool, type(None))) for x in data
    )
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", data, flow_style=flow,
    )


_CompactDumper.add_representer(list, _represent_list)
_CompactDumper.add_representer(tuple, _represent_list)


# ---------------------------------------------------------------------------
# TP-stable replication pass
# ---------------------------------------------------------------------------

def replicate_tp_stable(
    variant_root: Path,
    arch: Architecture,
    tp_degrees: list[int],
) -> None:
    """For layers marked ``tp_stable``, copy their rows from tp1/ into
    every other tp{N}/.

    Attention and MoE are never marked tp_stable (their kernel cost
    genuinely varies with TP), so only dense / per_sequence apply.
    """
    tp1_dir = variant_root / "tp1"
    if not tp1_dir.is_dir():
        log.warning("tp1/ missing; skipping tp_stable replication")
        return

    stable_dense = {
        name for name, e in arch.catalog.dense.items() if e.tp_stable
    }
    stable_seq = {
        name for name, e in arch.catalog.per_sequence.items() if e.tp_stable
    }

    for tp in tp_degrees:
        if tp == 1:
            continue
        dst_dir = variant_root / f"tp{tp}"
        dst_dir.mkdir(parents=True, exist_ok=True)

        if stable_dense:
            _replicate_layer_file(
                src=tp1_dir / "dense.csv",
                dst=dst_dir / "dense.csv",
                key_fields=["layer", "tokens"],
                layer_whitelist=stable_dense,
            )
        if stable_seq:
            _replicate_layer_file(
                src=tp1_dir / "per_sequence.csv",
                dst=dst_dir / "per_sequence.csv",
                key_fields=["layer", "sequences"],
                layer_whitelist=stable_seq,
            )


def _replicate_layer_file(
    src: Path,
    dst: Path,
    key_fields: list[str],
    layer_whitelist: set[str],
) -> None:
    """Copy rows for ``layer`` ∈ whitelist from ``src`` into ``dst``.

    Handles three cases:
      1. dst doesn't exist → create it with just the replicated rows.
      2. dst exists → merge (rows not in whitelist preserved, rows in
         whitelist overwritten by src values).
    """
    if not src.exists():
        log.warning("%s missing; cannot replicate tp_stable layers", src)
        return

    # Read src rows, keep only whitelisted layers.
    src_rows = _read_csv_rows(src)
    src_stable = [r for r in src_rows if r["layer"] in layer_whitelist]

    if dst.exists():
        dst_rows = _read_csv_rows(dst)
        # Drop any existing dst rows whose layer is in whitelist —
        # those slots are owned by the tp1 canonical values.
        dst_rows = [r for r in dst_rows if r["layer"] not in layer_whitelist]
        merged = dst_rows + src_stable
    else:
        merged = src_stable

    # Re-sort for deterministic output.
    merged.sort(key=lambda r: tuple(
        int(r[k]) if k != "layer" else r[k]
        for k in key_fields
    ))

    _write_csv_rows(dst, merged)
    log.debug(
        "replicated %d tp_stable rows into %s", len(src_stable), dst
    )


# ---------------------------------------------------------------------------
# Tiny CSV helpers (we don't want pandas here just for read/write)
# ---------------------------------------------------------------------------

def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Environment probes for meta.yaml
# ---------------------------------------------------------------------------

def _vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except ImportError:
        return "unknown"


def _cuda_version() -> str:
    # torch.version.cuda is the runtime CUDA version linked into torch.
    try:
        import torch
        return torch.version.cuda or "unknown"
    except Exception:
        return "unknown"


def _gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


def _utcnow_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds"
    )


def _prior_meta(variant_root: Path) -> dict[str, Any]:
    """The meta.yaml already in this variant root, or ``{}``.

    A refresh boots one engine and sweeps one category, so most of the file
    describes work it did not do. Reading the prior file is how those parts
    survive; see ``persist_meta``'s ``records_*`` flags for which.
    """
    existing = variant_root / "meta.yaml"
    if not existing.is_file():
        return {}
    try:
        with existing.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _engine_resolved_block(
    variant_root: Path,
    limits_by_tp: Any,
) -> dict[str, Any] | None:
    """``{"per_tp": {tp: {block_size, max_model_len, num_cache_tokens}}}``.

    What the engine actually **settled on**, which is not always what was asked
    for: vLLM derives the KV block size from the backend's
    ``get_supported_kernel_block_sizes`` and from hybrid page unification --
    MiniMax-M3's sparse backend accepts only 128, and a gated-DeltaNet stack
    raises the attention block until an attention page costs at least as many
    bytes as a mamba state page (784 on Qwen3.8-27B, against the 16
    requested). The simulator reads this rather than reimplementing vLLM's
    backend selection, and a run whose ``--block-size`` disagrees is
    simulating a configuration vLLM cannot serve.

    Keyed by TP because the mamba and attention pages both scale with the
    rank's shard, so the resolved size is a per-rank fact. Entries already in
    the file are **kept**: a ``slice`` refresh boots one TP and must not erase
    what the other TPs resolved.
    """
    prior = _prior_meta(variant_root)
    prior_block = prior.get("engine_resolved")
    if not limits_by_tp:
        # Nothing new to say. Returning None here erased the block instead of
        # leaving it alone, which is how an MTP-only refresh could delete what
        # the main engine resolved.
        return prior_block or None

    merged: dict[str, Any] = {}
    prior_per_tp = ((prior_block or {}).get("per_tp") or {})
    if isinstance(prior_per_tp, dict):
        merged.update({str(k): v for k, v in prior_per_tp.items()})

    for tp, limits in limits_by_tp.items():
        if limits is None:
            continue
        merged[str(tp)] = _stringify({
            "block_size": getattr(limits, "block_size", None),
            "max_model_len": getattr(limits, "max_model_len", None),
            "num_cache_tokens": getattr(limits, "num_cache_tokens", None),
        })
    return {"per_tp": dict(sorted(merged.items(), key=lambda kv: int(kv[0])))} if merged else None


def _stringify(obj: Any) -> Any:
    """Best-effort coercion to YAML-friendly primitives.

    vLLM kwargs include enum values, Path objects, and occasionally
    tensors. We coerce them to plain strings/dicts so yaml.safe_dump
    can handle the whole structure.
    """
    if isinstance(obj, dict):
        return {str(k): _stringify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stringify(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
