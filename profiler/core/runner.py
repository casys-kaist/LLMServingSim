"""Top-level orchestration.

Entry points:
    run_full(arch_path, args, out_root)
    run_slice(arch_path, args, tp, group, out_root)
    run_coverage(arch_path, args)

The first two are the profiling runs. They differ only in which categories and
TPs are iterated; everything else — engine spin-up, catalog slicing, shot
firing, sink coalescing, tp_stable replication — is shared.

``run_coverage`` writes nothing: it boots the same engine and asks how much of
the model's CUDA time the catalog binds, which is the acceptance test for a
newly written one. See ``hooks/timings.CoverageReport`` for why reading the
module tree is not enough.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import dataclasses

from profiler.core import logger as log
from profiler.core.categories import (
    CATEGORY_BY_NAME,
    Category,
    _entry_dict,
    categories_for,
)
from profiler.core.config import Architecture, ProfileArgs, load_architecture
from profiler.core.engine import (
    probe_limits,
    resolve_attention_max_kv,
    spin_down,
    spin_up,
)
from profiler.core.hooks.batch import Shot
from profiler.core.hooks.timings import CoverageReport, TimingSample
from profiler.core.stack import ALL_AXES, minimal_layer_count_for
from profiler.core.writer import (
    persist_meta,
    replicate_tp_stable,
    sink_for,
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _variant_root(out_root: Path, args: ProfileArgs) -> Path:
    """Build ``<out_root>/<hardware>/<model_path>/<variant>/``.

    Model path preserves the HuggingFace ``org/model`` layout so the
    simulator's loader (which already expects this shape) doesn't need
    to change. Local paths are normalized to their directory name.
    """
    # If `args.model` is a local path (contains "/" and exists on disk),
    # use its final component as the output subfolder; otherwise treat
    # as HF id verbatim.
    model_as_path = Path(args.model)
    if model_as_path.exists() and model_as_path.is_dir():
        model_subpath = model_as_path.name
    else:
        model_subpath = args.model
    return out_root / args.hardware / model_subpath / args.effective_variant


# ---------------------------------------------------------------------------
# Shot firing + point ingestion (shared between run_full and run_slice)
# ---------------------------------------------------------------------------

def _fire_one_category(
    llm,
    category: Category,
    arch: Architecture,
    args: ProfileArgs,
    limits,
    tp: int,
    out_dir: Path,
) -> None:
    """Sweep all of this category's shots, write the resulting CSV.

    Resume behaviour: unless ``args.force`` is set, an existing CSV is
    preloaded into the sink and shots whose key is already covered are
    skipped. The sink's flush at the end writes both preserved and
    newly-measured rows. ``--force`` restores wipe-and-rewrite.
    """
    sink = sink_for(category, out_dir)
    catalog_slice = category.catalog_slice(arch)

    prior_keys: set[tuple] = set()
    if not args.force:
        preloaded = sink.preload()
        if preloaded:
            prior_keys = sink.prior_shot_keys()
            log.info(
                "%s resume: %d prior rows preloaded, "
                "%d prior shot keys recognized",
                category.label, preloaded, len(prior_keys),
            )

    # Materialize shots up-front so the progress bar has a total.
    # Grids are small enough (<a few thousand shots) that holding them
    # in memory is fine.
    all_shots = list(category.compose_shots(arch, args, limits, tp))
    if not all_shots:
        log.warning(
            "category %s produced no shots for tp=%d; skipping",
            category.label, tp,
        )
        return

    if prior_keys:
        shots = [s for s in all_shots if category.shot_key(s) not in prior_keys]
        skipped = len(all_shots) - len(shots)
        if skipped:
            log.info(
                "%s: skipping %d already-measured shots, firing %d new",
                category.label, skipped, len(shots),
            )
    else:
        shots = all_shots

    if not shots:
        # Nothing to fire; still flush so the CSV is rewritten with
        # preloaded rows (a no-op schema repair if anything changed).
        sink.flush()
        log.info("%s: nothing to do (all shots already measured)", category.label)
        return

    label = f"TP={tp}  {category.label}"
    with log.progress(label, total=len(shots)) as bar:
        for shot in shots:
            raw = llm.collective_rpc(
                "fire",
                args=(shot.as_dict(), catalog_slice, category.name,
                      args.measurement_iterations),
            )
            # collective_rpc returns one result per worker (one per
            # TP rank). The timings are identical across ranks; take
            # rank 0's.
            timings_dicts = raw[0]
            timings = [
                TimingSample(
                    layer=d["layer"],
                    microseconds=float(d["microseconds"]),
                )
                for d in timings_dicts
            ]
            for point in category.extract_points(shot, timings, arch, tp):
                sink.coalesce(point)
            bar.advance(1)

    sink.flush()
    log.success("%s → %s", category.label, sink.path)


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

def run_full(
    arch_path: Path,
    args: ProfileArgs,
    out_root: Path,
) -> None:
    """Profile every (tp, category) pair for this architecture × model."""
    arch = load_architecture(arch_path)
    variant_root = _variant_root(out_root, args)

    log.banner(args, variant_root)

    last_engine_kwargs: dict[str, Any] | None = None
    # Keyed by TP, not a single "last": the engine resolves a different
    # block size per TP degree on a hybrid stack, because both the mamba page
    # and the attention page scale with the rank's shard. Qwen3.8-27B comes
    # out 784 at tp=1 and 784 at tp=2 -- but a *single* recorded value is
    # whichever TP ran last, and the simulator would read it for every TP.
    limits_by_tp: dict[int, Any] = {}

    for tp in args.tp_degrees:
        # Skip TPs with nothing non-tp_stable to do. The post-pass
        # replicate_tp_stable will populate their CSVs from tp1.
        if not arch.has_tp_dependent_work(tp):
            log.info("TP=%d has only tp_stable work; deferring to replication",
                     tp)
            continue

        # The main pass always boots **without** the drafter. Booting with it
        # would put a second copy of every class in the decoder block into the
        # profile tree -- the drafter replays the target's own block -- so each
        # dense / attention / moe entry would match twice and record the sum.
        # The drafter gets its own engine at the end of this TP instead.
        main_args = dataclasses.replace(args, profile_mtp=False)

        # Group the categories by how deep a stack they actually need. Every
        # layer costs the profiler its whole op count on every shot, and the
        # tree merges same-class siblings -- so a second layer of a type
        # already present adds no information, only time. DeepSeek-V3.2 needs
        # 4 layers for its MLP axis and 1 for attention, and the attention
        # sweep is the expensive one.
        wanted = categories_for(arch, tp)
        moe_eps = tuple(args.moe_ep_degrees or (1,))
        by_depth: dict[int, list] = {}
        for category in wanted:
            if category.name == "mtp":
                continue  # second pass, with the drafter built
            if category.name == "moe" and 1 not in moe_eps:
                continue  # every requested EP slice needs its own engine
            depth = minimal_layer_count_for(
                args.model_config or {}, category.stack_axes)
            by_depth.setdefault(depth, []).append(category)
        # Deepest first: its engine also answers probe_limits for the meta and
        # for `--attention-max-kv`, which has to describe the stack the
        # simulator will run rather than a shrunk one.
        depths = sorted(by_depth, reverse=True)
        if len(depths) > 1:
            log.info(
                "TP=%d: %s", tp, "; ".join(
                    f"{d} layer(s) for " + "/".join(c.label for c in by_depth[d])
                    for d in depths))

        tp_root = variant_root / f"tp{tp}"
        tp_root.mkdir(parents=True, exist_ok=True)

        for depth in depths:
            axes = by_depth[depth][0].stack_axes
            stage = (f"TP={tp}  booting vLLM engine"
                     + (f" ({depth} layer(s))" if len(depths) > 1 else ""))
            with log.stage(stage):
                llm, engine_kwargs, tmpdir = spin_up(main_args, tp, axes)
                limits = probe_limits(llm, main_args)
            if depth == depths[0]:
                # The deepest engine is the one whose shapes describe the stack
                # the simulator will run, so it is the one the meta records and
                # the one --attention-max-kv resolves against.
                last_engine_kwargs = engine_kwargs
                limits_by_tp[tp] = limits
                args = dataclasses.replace(
                    args,
                    attention_max_kv=resolve_attention_max_kv(args, limits),
                )
                # Visibility: what the live engine actually allocated for this
                # (tp, shrunk) configuration. Drives every feasibility filter
                # downstream.
                log.info(
                    "TP=%d limits: num_cache_tokens=%d max_model_len=%d "
                    "max_num_batched_tokens=%d max_num_seqs=%d "
                    "block_size=%d%s%s",
                    tp,
                    limits.num_cache_tokens,
                    limits.max_model_len,
                    limits.max_num_batched_tokens,
                    limits.max_num_seqs,
                    limits.block_size,
                    (f" linear_attn_chunk={limits.linear_attn_chunk}"
                     if limits.linear_attn_chunk else ""),
                    (f" num_experts={limits.num_experts} top_k={limits.top_k}"
                     if limits.num_experts else ""),
                )

            try:
                if not args.only_skew:
                    for category in by_depth[depth]:
                        _fire_one_category(
                            llm, category, arch, args, limits, tp, tp_root,
                        )
                else:
                    log.info("only_skew mode: skipping dense / per_seq / "
                             "attention / moe categories")
                # Skew rides on the attention kernel slice, so it belongs to
                # the engine that measured attention -- same shrink, same
                # limits, same feasibility bounds.
                if not args.skip_skew and any(
                        c.name == "attention" for c in by_depth[depth]):
                    from profiler.core.skew import sample_skew
                    sample_skew(llm, arch, args, limits, tp, tp_root)
            finally:
                spin_down(llm, tmpdir)

        # Per-EP pass for the MoE block. ep=1 rode the main engine above (it
        # *is* the main engine's expert count), so only the other degrees need
        # a boot -- each one a different model: E/ep experts, k/ep assignments
        # per token. That is what makes the rank-local operating point a
        # measured grid point instead of a clamp against the ep=1 floor.
        moe_category = next(
            (c for c in categories_for(arch, tp) if c.name == "moe"), None)
        if moe_category is not None and not args.only_skew:
            for ep in (e for e in moe_eps if e != 1):
                with log.stage(f"TP={tp}  booting vLLM engine at EP={ep} "
                               f"for the MoE slice"):
                    llm, _, tmpdir = spin_up(args, tp, moe_category.stack_axes,
                                             moe_ep=ep)
                    ep_limits = dataclasses.replace(
                        probe_limits(llm, args), moe_ep=ep)
                log.info(
                    "TP=%d EP=%d MoE slice: %s local experts, top_k=%s",
                    tp, ep, ep_limits.num_experts, ep_limits.top_k,
                )
                try:
                    _fire_one_category(
                        llm, moe_category, arch, args, ep_limits, tp, tp_root,
                    )
                finally:
                    spin_down(llm, tmpdir)

        # Second pass: the drafter, on its own engine and its own category.
        # Only the `mtp` catalog group is in the slice handed to the matcher,
        # and its entries are pinned to the drafter's wrapper, so nothing can
        # double here. Skipped when the model declares no MTP modules -- the
        # category's slice would be empty and _fire_one_category says so.
        if args.profile_mtp and not args.only_skew:
            mtp_category = next(
                (c for c in categories_for(arch, tp) if c.name == "mtp"), None)
            if mtp_category is None:
                log.info("TP=%d: --profile-mtp set but this architecture "
                         "declares no 'mtp' catalog group; nothing to do", tp)
            else:
                with log.stage(f"TP={tp}  booting vLLM engine with the drafter"):
                    llm, _, tmpdir = spin_up(args, tp)
                    mtp_limits = probe_limits(llm, args)
                try:
                    _fire_one_category(
                        llm, mtp_category, arch, args, mtp_limits, tp, tp_root,
                    )
                finally:
                    spin_down(llm, tmpdir)

    # After every tp has run, copy tp_stable rows from tp1 into the rest.
    # Skip when only_skew=True (nothing new to replicate).
    if not args.only_skew:
        with log.stage("replicating tp_stable layers across TP folders"):
            replicate_tp_stable(variant_root, arch, args.tp_degrees)

    if last_engine_kwargs is None:
        last_engine_kwargs = {}
    persist_meta(args, arch_path, last_engine_kwargs, variant_root, limits_by_tp,
                 measured_categories=tuple(
                     c.name for c in categories_for(arch, args.tp_degrees[0])))

    log.done(variant_root)


# ---------------------------------------------------------------------------
# Slice refresh
# ---------------------------------------------------------------------------

def run_slice(
    arch_path: Path,
    args: ProfileArgs,
    tp: int,
    group: str,
    out_root: Path,
) -> None:
    """Re-profile one (tp, category) pair without redoing everything."""
    arch = load_architecture(arch_path)
    variant_root = _variant_root(out_root, args)

    if group not in CATEGORY_BY_NAME:
        raise ValueError(
            f"unknown group {group!r}; must be one of "
            f"{sorted(CATEGORY_BY_NAME)}"
        )
    if tp not in args.tp_degrees:
        raise ValueError(
            f"tp={tp} is not in the session's tp_degrees ({args.tp_degrees})"
        )

    category_cls = CATEGORY_BY_NAME[group]
    category = category_cls()

    if not category.catalog_slice(arch):
        raise ValueError(
            f"architecture has no entries in catalog.{group}; "
            f"nothing to profile"
        )

    log.banner(args, variant_root)
    log.info("Slice refresh: tp=%d group=%s", tp, group)

    # Shrink to what this category measures, not to what every category
    # together would need: the attention sweep does not care which MLP a layer
    # runs, and each extra layer costs the profiler its whole op count on every
    # shot. DeepSeek-V3.2 and GLM-5 go from 4 layers to 1 here.
    depth = minimal_layer_count_for(args.model_config or {},
                                    category.stack_axes)
    tp_root = variant_root / f"tp{tp}"
    tp_root.mkdir(parents=True, exist_ok=True)

    # The MoE block is the one category whose engine is a *slice* of the model
    # rather than the whole of it, so a refresh may need several boots. Every
    # other group is one engine and ``moe_ep`` stays 1.
    eps = (tuple(args.moe_ep_degrees or (1,)) if group == "moe" else (1,))
    engine_kwargs: dict = {}
    limits = None
    for ep in eps:
        stage = f"TP={tp}  booting vLLM engine ({depth} layer(s))"
        if ep != 1:
            stage += f" at EP={ep}"
        with log.stage(stage):
            llm, engine_kwargs, tmpdir = spin_up(
                args, tp, category.stack_axes, moe_ep=ep)
            limits = dataclasses.replace(probe_limits(llm, args), moe_ep=ep)
            args = dataclasses.replace(
                args, attention_max_kv=resolve_attention_max_kv(args, limits),
            )
        if ep != 1:
            log.info("TP=%d EP=%d MoE slice: %s local experts, top_k=%s",
                     tp, ep, limits.num_experts, limits.top_k)
        try:
            _fire_one_category(
                llm, category, arch, args, limits, tp, tp_root,
            )
        finally:
            spin_down(llm, tmpdir)

    # A slice refresh at tp=1 may invalidate prior replication; redo it.
    if tp == 1:
        with log.stage("replicating tp_stable layers"):
            replicate_tp_stable(variant_root, arch, args.tp_degrees)

    # What this refresh is entitled to rewrite. ``engine_effective`` and
    # ``engine_resolved`` describe the deepest main engine; this one booted a
    # stack shrunk to its own category's axes, and with ``--profile-mtp`` it
    # booted the drafter alongside -- a different model either way. And only a
    # run that swept attention knows which axes the grid covers.
    deepest = minimal_layer_count_for(args.model_config or {}, ALL_AXES)
    records_engine = not args.profile_mtp and depth == deepest
    persist_meta(
        args, arch_path,
        engine_kwargs if records_engine else None,
        variant_root,
        {tp: limits} if records_engine else None,
        records_engine=records_engine,
        records_attention_grid=(group == "attention"),
        measured_categories=(group,),
    )
    if not records_engine:
        log.info(
            "meta: keeping the recorded engine_effective/engine_resolved -- "
            "this refresh booted %d layer(s)%s, not the %d-layer main engine "
            "the file describes",
            depth, " with the drafter" if args.profile_mtp else "", deepest,
        )

    log.done(variant_root)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

# One shot per attention regime. Which kernels a model launches is not a
# property of the model alone: MiniMax-M3's sparse attention runs a different
# Triton kernel on a prefill-only batch, on a decode-only batch, and on a mixed
# one, so a single shot would have declared two thirds of it missing (or, worse,
# declared full coverage while two kernels went unbound). Small and fixed:
# coverage asks which nodes appear, not how much they cost.
_COVERAGE_REGIMES: tuple[tuple[str, dict[str, int]], ...] = (
    ("prefill", {"prefill_chunk": 64, "kv_prefill": 0,
                 "n_decode": 0, "kv_decode": 0}),
    ("decode", {"prefill_chunk": 0, "kv_prefill": 0,
                "n_decode": 4, "kv_decode": 256}),
    ("mixed", {"prefill_chunk": 64, "kv_prefill": 0,
               "n_decode": 4, "kv_decode": 256}),
)


def run_coverage(arch_path: Path, args: ProfileArgs) -> dict[str, CoverageReport]:
    """Report how much of the model's CUDA time the catalog binds.

    Boots at tp=1 only. Coverage is about which nodes exist, and TP changes
    tensor shapes rather than the module graph -- profiling every TP would cost
    minutes to re-answer the same question.

    Returns the per-regime reports and logs them; writes no CSV. A non-empty
    ``gaps`` list is the actionable output: each line is a node the catalog
    binds nothing to, at the shallowest level where that is true.
    """
    arch = load_architecture(arch_path)

    # The whole catalog as one slice. Coverage is a property of the catalog,
    # not of a category, and a layer filed under the wrong category still
    # binds its node -- that is a different (and much more visible) bug.
    # Groups come from the Catalog model, not a literal list: a hardcoded one
    # silently omits any group added later, and coverage reporting a node as
    # unbound when the catalog does bind it is worse than useless -- it sends
    # you looking for a binding that is already there.
    whole_catalog: dict[str, dict[str, Any]] = {}
    for group in type(arch.catalog).model_fields:
        entries = getattr(arch.catalog, group, None)
        if isinstance(entries, dict):
            whole_catalog.update(_entry_dict(entries, arch))

    log.info(
        "Coverage check: %s (%d catalog entries, %d regimes)",
        arch_path.stem, len(whole_catalog), len(_COVERAGE_REGIMES),
    )

    with log.stage("TP=1  booting vLLM engine"):
        llm, _engine_kwargs, tmpdir = spin_up(args, 1)

    reports: dict[str, CoverageReport] = {}
    try:
        for label, spec in _COVERAGE_REGIMES:
            shot = Shot.attention(**spec)
            raw = llm.collective_rpc(
                "coverage",
                args=(shot.as_dict(), whole_catalog, 1),
            )[0]
            reports[label] = CoverageReport.hydrate(raw)
    finally:
        spin_down(llm, tmpdir)

    for label, report in reports.items():
        log.info(
            "%-8s %8.1f us total, %8.1f us bound (%.1f%%), %d unbound node(s)",
            label, report.total_us, report.bound_us,
            100.0 * report.fraction, len(report.gaps),
        )
        for cls, us, ancestors in report.gaps[:12]:
            log.warning(
                "  unbound %8.1f us  %s   under: %s", us, cls[:64], ancestors,
            )
        # With the drafter booted, every class in the decoder block appears
        # twice -- once under the target, once under the drafter's replayed
        # copy -- so a non-mtp entry matching both is expected rather than a
        # defect. That configuration cannot reach a CSV: --profile-mtp is
        # rejected for a full run and for any --group but mtp, and the mtp
        # group's slice holds only mtp entries. Report the count, not a line
        # each, and do not fail on it.
        mtp_names = set((arch.catalog.mtp or {}))
        real = [(c, ch) for c, ch in report.over_matches
                if not args.profile_mtp or c in mtp_names]
        expected = len(report.over_matches) - len(real)
        for canonical, chains in real:
            log.warning(
                "  OVER-MATCH %s claims nodes in %d unrelated places: %s",
                canonical, len(chains), " | ".join(chains),
            )
        if expected:
            log.info(
                "  (%d non-mtp entr%s also match the drafter's replayed "
                "block, which is why --profile-mtp is restricted to "
                "--group mtp)", expected, "y" if expected == 1 else "ies",
            )
        report.over_matches = real

    missed = {l: r for l, r in reports.items() if r.gaps}
    over = {l: r for l, r in reports.items() if r.over_matches}
    if over:
        log.warning(
            "One entry is measuring more than one thing in %s. Its number is "
            "the sum of two roles, and the coverage percentage above cannot "
            "show it -- over-matching leaves nothing unbound. Pin the entry "
            "with 'within' (or exclude the other role with 'not_within') and "
            "give the second role an entry of its own. Qwen3.5's 'mtp_norms' "
            "read 1287 us for two RMSNorms this way, and its 'embedding' "
            "summed the target's table with the drafter's own.",
            ", ".join(over),
        )
    if missed:
        log.warning(
            "Catalog does not account for every kernel in %s. Each line above "
            "is CUDA time the simulator will never see. Bind it, or record in "
            "the catalog why it is deliberately omitted.",
            ", ".join(missed),
        )
    else:
        log.info(
            "Catalog binds every measured kernel, in all %d regimes.",
            len(reports),
        )
    return reports
