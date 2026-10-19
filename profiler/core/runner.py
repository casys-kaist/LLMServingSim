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
import time

from profiler.core import logger as log
from profiler.core.categories import (
    CATEGORY_BY_NAME,
    Category,
    _entry_dict,
    categories_for,
)
from profiler.core.config import Architecture, ProfileArgs, declares_moe, load_architecture
from profiler.core.engine import (
    probe_limits,
    resolve_attention_max_kv,
    spin_down,
    spin_up,
)
from profiler.core.hooks.batch import Shot
from profiler.core.hooks.capture_scope_contract import contexts_from_architecture
from profiler.core.hooks.timings import CoverageReport, TimingSample
from profiler.core.stack import ALL_AXES, minimal_layer_count_for
from profiler.core.writer import (
    persist_meta,
    replicate_tp_stable,
    sink_for,
)
from profiler.core.callable_profile import profile_for, digest
from profiler.core.context_tables import CategoryRecord


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

def _validate_layout_invocations(llm, architecture, choices, reports_by_rank, catalog):
    """Apply an opt-in physical-call gate to every discovery rank and regime."""
    from profiler.core import component_layouts as layouts
    if not any(group["variants"][choices[name]].get("invocation_contract") is not None
               for name, group in (architecture.get("component_layouts") or {}).items()):
        return
    config = layouts.resolved_text_config(llm.llm_engine.vllm_config)
    expected = layouts.expected_scope_calls(architecture, choices, config)
    for reports in reports_by_rank:
        for regime, report in reports.items():
            layouts.validate_invocation_report(architecture, choices, report, expected, 1, catalog, regime)


def _component_layout_context(llm, category, arch, scope_contexts, args=None, tp=1, logical_ep=1):
    """Discover representations on the live engine before writing category rows."""
    from profiler.core import component_layouts as layouts
    architecture = arch.model_dump(exclude_none=True)
    groups = architecture.get("component_layouts") or {}
    affected = {name for group in groups.values() for name in group["categories"]}
    runtime_possible = any(variant.get('runtime_contract') is not None
                           for group in groups.values() for variant in group['variants'].values())
    if category.name not in affected and not runtime_possible:
        return None
    whole_catalog = {}
    for group in type(arch.catalog).model_fields:
        whole_catalog.update(_entry_dict(getattr(arch.catalog, group), arch))
    probes = layouts.probe_catalog(architecture, whole_catalog)
    # Do not cache across engine boots or TP degrees. Per-shot probes below
    # additionally reject shape-dependent changes after these three checks.
    reports_by_rank = None
    for regime, shot in _coverage_shots(llm.llm_engine.vllm_config):
        reports = llm.collective_rpc(
            "coverage", args=(shot.as_dict(), probes, 1, scope_contexts))
        if reports_by_rank is None:
            reports_by_rank = [{} for _ in reports]
        if not reports or len(reports) != len(reports_by_rank):
            raise ValueError("Component-layout discovery changed its worker set")
        for values, report in zip(reports_by_rank, reports):
            values[regime] = report
    selections = [layouts.select_runtime_layouts(architecture, reports)[0]
                  for reports in reports_by_rank]
    choices = selections[0]
    if any(selection != choices for selection in selections):
        raise ValueError("Component layouts differ across TP workers")
    _validate_layout_invocations(llm, architecture, choices, reports_by_rank, probes)
    runtime_schema = layouts.selected_runtime_contract(architecture, choices)
    if category.name not in affected and runtime_schema is None:
        return None
    runtime = None
    if runtime_schema is not None:
        if args is None:
            raise ValueError('Runtime admission requires host source configuration and logical degrees')
        # Keep the original source depth. --num-hidden-layers and per-category
        # reductions are acquisition prefixes, not new source model identities.
        from profiler.core.hooks.runtime_admission import profile_descriptor, admit_runtime
        descriptor = profile_descriptor(runtime_schema, args.model_config, args.hf_overrides,
                                        logical_tp=tp, logical_ep=logical_ep)
        results = llm.collective_rpc('component_runtime_admission', args=(descriptor,))
        if len(results) != len(reports_by_rank) or not results:
            raise ValueError('Runtime admission changed the discovery worker set')
        for result in results:
            if admit_runtime(descriptor, result['engine_facts']) != result:
                raise ValueError('Worker runtime admission result is inconsistent')
        runtime = results[0]
        if any(item['runtime_sha256'] != runtime['runtime_sha256'] for item in results):
            raise ValueError('Workers disagree on component runtime modes')
    return architecture, choices, probes, runtime


def _fire_one_category(
    llm,
    category: Category,
    arch: Architecture,
    args: ProfileArgs,
    limits,
    tp: int,
    out_dir: Path,
    *,
    wipe: bool | None = None,
) -> bool:
    """Sweep all of this category's shots, write the resulting CSV.

    Resume behaviour: unless ``args.force`` is set, an existing CSV is
    preloaded into the sink and shots whose key is already covered are
    skipped. Periodic atomic checkpoints include both preserved and newly
    measured rows; only completed shots reach them. ``--force`` replaces
    the selected category rather than appending to its prior acquisition.

    ``wipe`` overrides the session's replacement policy for this category.
    """
    if wipe is None:
        wipe = args.force
    catalog_slice = category.catalog_slice(arch)
    native_scope = None
    profile = profile_for(arch)
    if profile is not None:
        native_scope = profile.native_scope(llm, args, tp, limits)
        # Full-depth residual roles are isolated native calls, not the pooled
        # prefix-engine calls. All other entries retain ordinary category axes.
        catalog_slice = profile.ordinary_catalog(catalog_slice)
    from profiler.core import component_layouts as layouts
    scope_contexts = contexts_from_architecture(arch)
    layout_context = _component_layout_context(llm, category, arch, scope_contexts,
                                               args, tp, limits.moe_ep)
    layout_sha256 = None
    component_layout_contract = None
    runtime = None
    runtime_sha256 = None
    capture_catalog = catalog_slice
    if layout_context is not None:
        architecture, choices, probes, runtime = layout_context
        catalog_slice = layouts.filter_catalog_slice(architecture, choices, catalog_slice)
        capture_catalog = {**catalog_slice, **probes}
        layout_sha256 = layouts.layout_identity(architecture, choices)
        if any(group["variants"][choices[name]].get("invocation_contract") is not None
               for name, group in architecture["component_layouts"].items()):
            component_layout_contract = dict(architecture=architecture, selections=choices,
                                              layout_sha256=layout_sha256)
        if runtime is not None:
            runtime_sha256 = runtime['runtime_sha256']
            component_layout_contract['runtime'] = dict(
                descriptor=runtime['invariant_facts']['descriptor'], runtime_sha256=runtime_sha256)
    from .measurement import layerwise_identity, layerwise_measurement, measurement_token, native_scope_token
    from . import attention_lookup as lookup_contract
    attention_semantic = (lookup_contract.acquire(llm, args.model_config,
                          arch.model_dump(exclude_none=True), tp) if category.name == 'attention' else None)
    attention_sha = lookup_contract.digest(attention_semantic) if attention_semantic is not None else None
    identity = layerwise_identity(capture_catalog, args.measurement_iterations,
        scope_contexts=scope_contexts, component_layout_sha256=layout_sha256,
        attention_lookup_sha256=attention_sha, runtime_sha256=runtime_sha256,
        native_profile_sha256=native_scope_token(native_scope) if native_scope is not None else None)
    measurement = measurement_token(identity)
    from .sampling import resolve_sampling, record_category
    from .bundle_metadata import read_bundle
    limits.sampling_plan = resolve_sampling(args, limits, tp, meta=read_bundle(out_dir.parent),
                                            include_stable=category.name == 'dense')
    all_shots = list(category.compose_shots(arch, args, limits, tp))
    if not all_shots:
        log.warning('category %s produced no shots for tp=%d; skipping', category.label, tp)
        return False
    record_category(out_dir, category, args, limits.sampling_plan, all_shots)
    native_record = (CategoryRecord(out_dir, category, arch, all_shots, catalog_slice,
                      measurement, native_scope, replace=wipe, identity=identity, tp=tp)
                     if native_scope is not None else None)
    if attention_semantic is not None:
        lookup_contract.begin(out_dir, attention_semantic, measurement, replace=wipe)
    if layout_context is not None:
        # The shared probe catalog provides one version/source/repetition
        # identity across coupled categories; each CSV still stamps its
        # actual capture catalog independently.
        acquisition = layerwise_measurement(probes, args.measurement_iterations,
            scope_contexts=scope_contexts, component_layout_sha256=layout_sha256,
            runtime_sha256=runtime_sha256)
        layouts.begin_category(out_dir, architecture, choices, category.name,
                               measurement, acquisition, replace=wipe, runtime=runtime)
    sink = sink_for(category, out_dir, measurement=measurement)

    prior_keys: set[tuple] = set()
    # Copied stable rows retain their TP1 acquisition, including when the
    # selected representation differs at this degree. They cannot certify
    # a measured shot or be replaced by new TP2 samples.
    replicated = set()
    if tp > 1:
        replicated = {name for name, entry in catalog_slice.items()
                      if entry.get("tp_stable")
                      and (layout_context is None or name not in probes)}
    if not wipe or replicated:
        preloaded = sink.preload(preserve_layers=replicated, preserve_only=wipe)
        if preloaded:
            prior_keys = sink.prior_shot_keys()
            log.info(
                "%s resume: %d prior rows preloaded, "
                "%d prior shot keys recognized",
                category.label, preloaded, len(prior_keys),
            )
    if native_record is not None:
        prior_keys = native_record.resume_keys(sink)

    # Retain the planned shots for progress totals and resume filtering.
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
        if native_record is not None:
            native_record.complete(sink)
            if category.name == 'dense':
                return profile.acquire_contexts(llm, args, all_shots, out_dir, replace=wipe)
        if attention_semantic is not None:
            lookup_contract.complete(out_dir)
        if layout_context is not None:
            layouts.complete_category(out_dir, architecture, category.name)
        log.info("%s: nothing to do (all shots already measured)", category.label)
        return False

    label = f"TP={tp}  {category.label}"
    # Checkpoint between complete shots, never while processing a worker
    # result. Interrupted writes retain the previous atomically replaced CSV.
    heartbeat = max(1, len(shots) // 100)
    t_start = time.monotonic()
    checkpoint_at = t_start + 60
    with log.progress(label, total=len(shots)) as bar:
        for done, shot in enumerate(shots, start=1):
            rpc_args = (shot.as_dict(), capture_catalog, category.name,
                        args.measurement_iterations, scope_contexts, layout_sha256, attention_semantic)
            if component_layout_contract is not None:
                rpc_args += (component_layout_contract,)
            if native_scope is not None:
                if component_layout_contract is not None:
                    raise ValueError('Local native profiles require their own admitted category contract')
                rpc_args += (None, native_scope)
            raw = llm.collective_rpc(
                "fire", args=rpc_args,
            )
            # collective_rpc returns one result per worker (one per
            # TP rank). The timings are identical across ranks; take
            # rank 0's.
            if not raw:
                raise ValueError('No worker timings returned')
            timings_dicts = raw[0]
            if attention_semantic is not None and any(not worker_rows or any(
                    any(row.get(k) != v for k, v in measurement.items()) for row in worker_rows)
                    for worker_rows in raw):
                raise ValueError('Worker attention lookup acquisition identity differs')
            if not timings_dicts or any(
                    any(d.get(k) != v for k, v in measurement.items())
                    for d in timings_dicts):
                raise ValueError('Worker timings do not match the requested acquisition method')
            if layout_context is not None:
                # These two categories carry q=1 decodes only. Reuse the
                # existing linear-category coordinate partition, including
                # explicit one-token prefill roles.
                from profiler.core.categories import _split_linear_attention_shot
                prefill, decode = _split_linear_attention_shot(shot)
                regime = "mixed" if prefill and decode else "prefill" if prefill else "decode"
                for worker_rows in raw:
                    if any(any(item.get(k) != v for k, v in measurement.items())
                           for item in worker_rows):
                        raise ValueError("Worker component-layout acquisition identity differs")
                    layouts.validate_runtime_shot(architecture, choices, regime,
                                                   {item["layer"] for item in worker_rows})
            timings = [
                TimingSample(
                    layer=d["layer"],
                    microseconds=float(d["microseconds"]),
                )
                for d in timings_dicts
                if d["layer"] in catalog_slice and d["layer"] not in replicated
            ]
            for point in category.extract_points(shot, timings, arch, tp):
                sink.coalesce(point)
            if done == 1:
                sink.stage_measurement(identity, replace=wipe)
            bar.advance(1)
            if done == 1 or done % 100 == 0 or time.monotonic() >= checkpoint_at:
                # Retain the complete accumulator, including duplicate counts:
                # flushing only the newest rows would erase earlier checkpoints.
                sink.flush(clear=False)
                checkpoint_at = time.monotonic() + 60
            if done % heartbeat == 0 or done == len(shots):
                rate = done / max(1e-9, time.monotonic() - t_start)
                log.info(
                    "%s: %d/%d shots (%.1f%%), %.2f shot/s, eta %.1f min",
                    category.label, done, len(shots), 100.0 * done / len(shots),
                    rate, (len(shots) - done) / rate / 60.0,
                )

    sink.flush(clear=native_record is None)
    if native_record is not None:
        native_record.complete(sink)
        if category.name == 'dense':
            profile.acquire_contexts(llm, args, all_shots, out_dir, replace=wipe)
    if attention_semantic is not None:
        lookup_contract.complete(out_dir)
    if layout_context is not None:
        layouts.complete_category(out_dir, architecture, category.name)
    log.success("%s → %s", category.label, sink.path)
    return True


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
    profile = profile_for(arch)
    if profile is not None:
        profile.validate_request(args, full=True)
    variant_root = _variant_root(out_root, args)

    if args.only_skew:
        for tp in args.tp_degrees:
            if not (variant_root / f"tp{tp}" / "attention.csv").exists():
                raise FileNotFoundError("--only-skew requires existing per-TP attention.csv files")

    log.banner(args, variant_root)

    component_args = None
    if not args.only_skew and (args.dp_degrees is not None or args.ep_degrees is not None
                              or any(declares_moe(holder) for holder in (
                                  args.model_config or {},
                                  (args.model_config or {}).get("text_config") or {}))):
        from .moe_profile import resolve_targets
        component_args = dataclasses.replace(
            args, dp_degrees=(1,) if args.dp_degrees is None else args.dp_degrees,
            profile_mtp=False)
        resolve_targets(arch, component_args, args.tp_degrees)

    last_engine_kwargs: dict[str, Any] | None = None
    # Keyed by TP, not a single "last": the engine resolves a different
    # block size per TP degree on a hybrid stack, because both the mamba page
    # and the attention page scale with the rank's shard. Qwen3.8-27B comes
    # out 784 at tp=1 and 784 at tp=2 -- but a *single* recorded value is
    # whichever TP ran last, and the simulator would read it for every TP.
    limits_by_tp: dict[int, Any] = {}
    # Whether the skew sweep actually fired. It is conditional twice over
    # (--skip-skew, and on the stack having an attention category at this
    # depth), and it is what the provenance stamp has to record -- not the
    # flag, the firing.
    skew_measured = False
    acquired_categories = set()

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
        by_depth: dict[int, list] = {}
        for category in wanted:
            if category.name == "mtp":
                continue  # second pass, with the drafter built
            depth = minimal_layer_count_for(
                args.model_config or {}, profile.stack_axes(category) if profile else category.stack_axes)
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
            axes = profile.stack_axes(by_depth[depth][0]) if profile else by_depth[depth][0].stack_axes
            stage = (f"TP={tp}  booting vLLM engine"
                     + (f" ({depth} layer(s))" if len(depths) > 1 else ""))
            with log.stage(stage):
                llm, engine_kwargs, tmpdir = spin_up(main_args, tp, axes, native_profile=profile)
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
                        if _fire_one_category(
                            llm, category, arch, args, limits, tp, tp_root,
                        ):
                            acquired_categories.add(category.name)
                else:
                    log.info("only_skew mode: skipping dense / per_seq / "
                             "attention / moe categories")
                # Skew rides on the attention kernel slice, so it belongs to
                # the engine that measured attention -- same shrink, same
                # limits, same feasibility bounds.
                if not args.skip_skew and any(
                        c.name == "attention" for c in by_depth[depth]):
                    from profiler.core.skew import sample_skew
                    if sample_skew(llm, arch, args, limits, tp, tp_root) is not None:
                        skew_measured = True
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
                    llm, _, tmpdir = spin_up(args, tp, native_profile=profile)
                    mtp_limits = probe_limits(llm, args)
                try:
                    if _fire_one_category(
                        llm, mtp_category, arch, args, mtp_limits, tp, tp_root,
                    ):
                        acquired_categories.add(mtp_category.name)
                finally:
                    spin_down(llm, tmpdir)

    if component_args is not None:
        from .moe_profile import run_components
        run_components(arch_path, component_args, args.tp_degrees, variant_root)

    # After every tp has run, copy tp_stable rows from tp1 into the rest.
    # Skip when only_skew=True (nothing new to replicate).
    if not args.only_skew:
        with log.stage("replicating tp_stable layers across TP folders"):
            replicate_tp_stable(variant_root, arch, args.tp_degrees)

    if last_engine_kwargs is None:
        last_engine_kwargs = {}
    # What this run is entitled to stamp. ``--only-skew`` fires no category at
    # all, so claiming them would relabel dense / attention / per_sequence as
    # freshly measured on a run that only touched skew.csv -- the exact failure
    # the per-artifact block exists to prevent.
    measured = tuple(sorted(acquired_categories))
    if skew_measured:
        measured += ("skew",)
    # And what it is entitled to *describe*. A skew-only run sweeps no
    # attention grid, so regenerating that block from its own defaults would
    # replace the recorded axes with a spec no CSV in the bundle matches --
    # it would have overwritten a x1.5, 37,962-shot grid with a x2 one. It
    # also boots only the attention-depth engine, which on a stack whose
    # categories need different depths is not the deepest, so it is not
    # authoritative for engine_effective / engine_resolved either.
    persist_meta(args, arch_path, last_engine_kwargs, variant_root, limits_by_tp,
                 records_engine=not args.only_skew,
                 records_attention_grid=not args.only_skew,
                 measured_categories=measured)

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
    profile = profile_for(arch)
    if profile is not None:
        profile.validate_request(args)
    variant_root = _variant_root(out_root, args)

    if group not in CATEGORY_BY_NAME and group != "moe":
        raise ValueError(
            f"unknown group {group!r}; must be one of "
            f"{sorted((*CATEGORY_BY_NAME, 'moe'))}"
        )
    if tp not in args.tp_degrees:
        raise ValueError(
            f"tp={tp} is not in the session's tp_degrees ({args.tp_degrees})"
        )

    if group == "moe":
        from .moe_profile import run_components
        component_args = dataclasses.replace(
            args, dp_degrees=(1,) if args.dp_degrees is None else args.dp_degrees)
        run_components(arch_path, component_args, [tp], variant_root)
        return
    if args.dp_degrees is not None or args.ep_degrees is not None:
        raise ValueError("Explicit DP/EP applies only to the MoE slice")


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
    axes = profile.stack_axes(category) if profile else category.stack_axes
    depth = minimal_layer_count_for(args.model_config or {}, axes)
    tp_root = variant_root / f"tp{tp}"
    tp_root.mkdir(parents=True, exist_ok=True)

    with log.stage(f"TP={tp}  booting vLLM engine ({depth} layer(s))"):
        llm, engine_kwargs, tmpdir = spin_up(args, tp, axes, native_profile=profile)
        limits = probe_limits(llm, args)
        args = dataclasses.replace(
            args, attention_max_kv=resolve_attention_max_kv(args, limits),
        )
    try:
        acquired = _fire_one_category(llm, category, arch, args, limits, tp, tp_root)
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
        # A category slice sweeps no skew either, and ``_skew_meta_block``
        # derives its grid from this run's args, which are the defaults
        # unless the caller happened to repeat every SKEW_* factor.
        records_skew=False,
        measured_categories=(group,) if acquired else (),
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


def _coverage_shots(config):
    """Check every discovery shape before the first direct worker call."""
    from .hooks.batch import validate_shot_limits
    shots = [(name, Shot.attention(**spec)) for name, spec in _COVERAGE_REGIMES]
    for name, shot in shots:
        try:
            validate_shot_limits(shot, config)
        except ValueError as exc:
            raise ValueError(f'{name} layout/coverage probe does not fit the acquisition '
                             f'limits: {exc}. Use max_num_seqs >= 5, max_model_len >= 258 '
                             'and an acquisition token buffer >= 68 for these probes.') from exc
    return shots


def run_coverage(arch_path: Path, args: ProfileArgs) -> dict[str, CoverageReport]:
    """Report how much of the model's CUDA time the catalog binds.

    Boots at physical tp=1 only: a bounded structural smoke, not certification
    of other TP degrees' fused paths, collective placement or complete sweeps.

    Returns the per-regime reports and logs them; writes no CSV. A non-empty
    ``gaps`` list reports unbound native kernel/copy/memset work by activity
    name and dynamic ancestry, including explicitly unscoped runner work.
    """
    arch = load_architecture(arch_path)
    profile = profile_for(arch)
    if profile is not None:
        profile.validate_request(args)

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
        llm, _engine_kwargs, tmpdir = spin_up(args, 1, native_profile=profile)

    reports: dict[str, CoverageReport] = {}
    scope_contexts = contexts_from_architecture(arch)
    try:
        reports_by_rank = None
        for label, shot in _coverage_shots(llm.llm_engine.vllm_config):
            raw = llm.collective_rpc(
                "coverage",
                args=(shot.as_dict(), whole_catalog, 1, scope_contexts),
            )
            if reports_by_rank is None:
                reports_by_rank = [{} for _ in raw]
            if not raw or len(raw) != len(reports_by_rank):
                raise ValueError("Coverage changed its worker set")
            for values, report in zip(reports_by_rank, raw):
                values[label] = report
            reports[label] = CoverageReport.hydrate(raw[0])
        architecture = arch.model_dump(exclude_none=True)
        if any(variant.get("invocation_contract") is not None
               for group in (architecture.get("component_layouts") or {}).values()
               for variant in group["variants"].values()):
            from profiler.core import component_layouts as layouts
            selections = [layouts.select_runtime_layouts(architecture, values)[0]
                          for values in reports_by_rank]
            if any(choice != selections[0] for choice in selections):
                raise ValueError("Component layouts differ across coverage workers")
            _validate_layout_invocations(llm, architecture, selections[0], reports_by_rank, whole_catalog)
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
                if (not args.profile_mtp or c in mtp_names
                    or any(chain.startswith("shared CUDA activity:") for chain in ch))]
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
