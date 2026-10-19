"""Dynamic heterogeneous acquisition with bounded, resumable raw storage.

The actual batch is measured; attention.csv provides both lookup references.
"""
import json
import hashlib
import math
import os
from pathlib import Path
import statistics
import tempfile
import time
from itertools import chain

import pandas as pd

from . import logger as log
from .skew_calibration import measurement_shape
from .bundle_metadata import read_tp, write_tp
from .skew_plan import iter_cases
from .skew_support import complete_plan

PROTOCOL = "pytorch-kineto-device-active-union-skew"


def layout_contract_for_skew(arch, choices, *, runtime=None):
    """Describe a selected nonhistorical attention layout without GPU imports.

    Native attention requires both an invocation declaration and a recorded
    runtime envelope; live acquisition separately re-admits its actual engine.
    """
    from . import component_layouts as layouts
    architecture = arch.model_dump(exclude_none=True)
    groups = architecture.get("component_layouts") or {}
    if not any("attention" in group["categories"]
               and choices[name] != group["historical"]
               for name, group in groups.items()):
        return None
    contract = dict(architecture=architecture, selections=choices,
                    layout_sha256=layouts.layout_identity(architecture, choices))
    schema = layouts.selected_runtime_contract(architecture, choices)
    if schema is None:
        raise NotImplementedError("Nonhistorical skew requires an explicit runtime admission contract; use --skip-skew")
    if not isinstance(runtime, dict) or set(runtime) != {'descriptor', 'runtime_sha256'}:
        raise ValueError('Native skew requires paired runtime admission metadata')
    from .hooks.runtime_admission import validate_descriptor
    descriptor = validate_descriptor(runtime['descriptor'])
    digest = runtime['runtime_sha256']
    if (descriptor['schema'] != schema
            or not isinstance(digest, str) or len(digest) != 64
            or any(char not in '0123456789abcdef' for char in digest)):
        raise ValueError('Invalid native skew runtime identity')
    contract['runtime'] = dict(descriptor=descriptor, runtime_sha256=digest)
    return contract


def recorded_contract_for_skew(tp_root, arch, choices, args, *, logical_tp, logical_ep=1,
                               block_size=None):
    """Project validated paired runtime evidence for host and CPU resume identity."""
    from . import component_layouts as layouts
    architecture = arch.model_dump(exclude_none=True)
    groups = architecture.get('component_layouts') or {}
    if not any('attention' in group['categories']
               and choices[name] != group['historical']
               for name, group in groups.items()):
        return None
    root = Path(tp_root)
    if root.name != f'tp{logical_tp}':
        raise ValueError('Skew TP directory differs from requested logical degree')
    # The caller has read_for_trace's validated paired CSVs through the common
    # selected-catalog reader. Never infer admission from row names or timings.
    block = layouts.metadata_records(root.parent, architecture)
    record = (block or {}).get('per_tp', {}).get(logical_tp)
    if (record is None or record['selections'] != choices
            or any(value['state'] != 'complete' for value in record['categories'].values())):
        raise ValueError('Skew requires complete matching component-layout references')
    saved = record.get('runtime')
    if saved is None:
        raise ValueError('Skew references lack native runtime admission')
    if block_size is not None and saved['invariant_facts']['manager_block_size'] != block_size:
        raise ValueError('Skew planning block size differs from paired runtime admission')
    from .hooks.runtime_admission import profile_descriptor
    descriptor = profile_descriptor(layouts.selected_runtime_contract(architecture, choices),
        args.model_config, args.hf_overrides, logical_tp=logical_tp, logical_ep=logical_ep)
    if descriptor != saved['invariant_facts']['descriptor']:
        raise ValueError('Requested skew source configuration differs from paired references')
    return layout_contract_for_skew(arch, choices, runtime=dict(
        descriptor=descriptor, runtime_sha256=saved['runtime_sha256']))


def _admit_skew_runtime(llm, contract, query_lengths):
    """Read and validate every live worker before initialization or output mutation."""
    if contract is None:
        return None
    from .hooks.runtime_admission import admit_runtime
    requested = contract['runtime']
    descriptor = requested['descriptor']
    results = llm.collective_rpc('component_runtime_admission', args=(descriptor,))
    if not results:
        raise ValueError('Skew runtime admission returned no worker evidence')
    for result in results:
        if (admit_runtime(descriptor, result['engine_facts']) != result
                or result['runtime_sha256'] != requested['runtime_sha256']):
            raise ValueError('Live skew runtime differs from paired attention references')
        # Preflight requested query axes through the same raw-shot admission
        # rule, not a second model/mode policy in the skew planner.
        for query in query_lengths:
            admit_runtime(descriptor, result['engine_facts'], shot=dict(
                requests=[[query, 1]], n_prefill=0, decode_q_len=query))
    return dict(schema=results[0]['schema'], runtime_sha256=requested['runtime_sha256'],
                invariant_facts=results[0]['invariant_facts'],
                worker_engines=[result['engine_facts'] for result in results])


def capture_catalog_for_skew(catalog, contract):
    """Use the same coupled probes as ordinary acquisition, never another shot."""
    if contract is None:
        return dict(catalog)
    from . import component_layouts as layouts
    from .config import Architecture
    from .categories import _entry_dict
    if not isinstance(contract, dict) or set(contract) != {"architecture", "selections", "layout_sha256", "runtime"}:
        raise ValueError("Malformed skew component-layout contract")
    arch = Architecture.model_validate(contract["architecture"])
    architecture = arch.model_dump(exclude_none=True)
    choices = contract["selections"]
    rebuilt = layout_contract_for_skew(arch, choices, runtime=contract['runtime'])
    if rebuilt != contract:
        raise ValueError("Skew layout descriptor differs from its selected architecture")
    unsupported = [name for name, group in architecture["component_layouts"].items()
                   if "attention" in group["categories"]
                   and choices[name] != group["historical"]
                   and group["variants"][choices[name]].get("invocation_contract") != "site-scope-calls"]
    if unsupported:
        raise NotImplementedError("Skew acquisition requires a same-shot layout/invocation contract "
                                  f"({', '.join(unsupported)}); use --skip-skew")
    selected = layouts.filter_catalog_slice(architecture, choices, _entry_dict(arch.catalog.attention, arch))
    if catalog != selected:
        raise ValueError("Skew timing catalog differs from the selected attention entries")
    whole = {}
    for category in type(arch.catalog).model_fields:
        whole.update(_entry_dict(getattr(arch.catalog, category), arch))
    return {**catalog, **layouts.probe_catalog(architecture, whole)}


def measurement_fingerprint(scope_contexts=None, *, catalog, attention_lookup_sha256=None,
                            component_layout_contract=None):
    """Bind raw timings to effective selectors, normalization and runtime stack."""
    from .hooks.capture_scope_contract import normalize_contexts
    if not isinstance(catalog, dict) or not catalog:
        raise ValueError("Skew acquisition requires a nonempty serialized attention catalog")
    contexts = normalize_contexts(scope_contexts, catalog)
    from .attention_lookup import protocol
    identity = dict(protocol=protocol(PROTOCOL, attention_lookup_sha256), catalog=catalog, scope_contexts=contexts,
                    attention_lookup_sha256=attention_lookup_sha256,
                    aggregation='median of per-forward device-active unions, then median across rounds')
    if component_layout_contract is not None:
        identity["component_layout_contract"] = component_layout_contract
    from .measurement import semantic_identity
    return hashlib.sha256(json.dumps(semantic_identity(identity), sort_keys=True,
                           separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _measurement_metadata(path):
    current = read_tp(Path(path).parent, 'skew', {})
    if not isinstance(current, dict):
        raise ValueError('Expected per-TP skew metadata mapping')
    return current

def _flush_rows(csv_path: Path, new_rows: list[dict]) -> pd.DataFrame:
    """Merge ``new_rows`` with any existing CSV and rewrite atomically.

    Returns the combined DataFrame. De-duplicates on the case key so
    a re-run of the same case overwrites rather than duplicates.

    ``layer`` is part of that key: one case now yields one row per
    attention-category kernel, and without it the last layer measured would
    evict every other layer's row for the same case. Absent from a CSV written
    before the column existed, where every row is the ``attention`` kernel.
    """
    frames: list[pd.DataFrame] = []
    previous = csv_path.stat() if csv_path.exists() else None
    if csv_path.exists():
        frames.append(pd.read_csv(csv_path, low_memory=False))
    if new_rows:
        frames.append(pd.DataFrame(new_rows))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Rows from a CSV written before the layer column existed come out of the
    # concat with a missing layer. They are all the ``attention`` kernel, and
    # naming them so is what keeps them in the fit -- left blank they become a
    # phantom "nan" kernel with its own alpha table.
    if "layer" in df.columns:
        df["layer"] = df["layer"].fillna("attention").replace("", "attention")
    else:
        df["layer"] = "attention"
    # Put it first, so the file reads layer-major like skew_fit.csv.
    df = df[["layer"] + [c for c in df.columns if c != "layer"]]
    # Keep the latest measurement for any repeated key.
    if "case_id" not in df.columns:
        df["case_id"] = ""
    missing = df["case_id"].isna() | (df["case_id"] == "")
    df.loc[missing, "case_id"] = [measurement_shape(row)[4]
        for row in df.loc[missing].fillna("").to_dict("records")]
    df = df.drop_duplicates(subset=["layer", "case_id"], keep="last").reset_index(drop=True)
    # Keep the previous checkpoint intact if serialization is interrupted.
    with tempfile.NamedTemporaryFile(dir=csv_path.parent,
                                     prefix=csv_path.name + ".", suffix=".tmp",
                                     delete=False) as stream:
        temporary = Path(stream.name)
    try:
        df.to_csv(temporary, index=False)
        # NamedTemporaryFile starts at 0600. Preserve a refreshed file's
        # permissions (and ownership when running as container root), and
        # keep a new profile readable from the host bind mount.
        if previous is not None and os.geteuid() == 0:
            os.chown(temporary, previous.st_uid, previous.st_gid)
        temporary.chmod(previous.st_mode & 0o777 if previous else 0o644)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(csv_path)
    finally:
        temporary.unlink(missing_ok=True)
    return df



def _existing_keys(path, layers, rounds, iterations, *, block_size=None, scope_contexts=None,
                   attention_lookup_sha256=None, component_layout_contract=None):
    """A partial kernel set or an under-repeated shot is still unfinished."""
    if not path.exists():
        return set()
    if attention_lookup_sha256 is not None:
        from .attention_lookup import check_skew
        check_skew(path.parent, attention_lookup_sha256, allow_pending=True)
    grouped = {}
    from .attention_lookup import protocol
    expected_protocol = protocol(PROTOCOL, attention_lookup_sha256)
    fingerprint = measurement_fingerprint(scope_contexts, catalog=layers,
                                          attention_lookup_sha256=attention_lookup_sha256,
                                          component_layout_contract=component_layout_contract)
    metadata = _measurement_metadata(path)
    for row in pd.read_csv(path, low_memory=False).fillna("").to_dict("records"):
        key = measurement_shape(row)[4]
        if row.get("case_id") and row["case_id"] != key:
            raise ValueError("Stored skew identity disagrees with its request geometry")
        if (row.get("measurement_protocol", metadata.get('measurement_protocol')) != expected_protocol
                or row.get("measurement_sha256", metadata.get('measurement_sha256')) != fingerprint
                or (block_size is not None and int(row.get("block_size", metadata.get('block_size')) or 0) != block_size)
                or int(row.get("rounds", metadata.get('rounds')) or 0) < rounds
                or int(row.get("timed_forwards", metadata.get('timed_forwards_per_round')) or 0) != iterations):
            continue
        samples = json.loads(row.get("round_timings_us_json") or "[]")
        if (len(samples) < rounds or any(len(values) != iterations for values in samples)
                or any(not math.isfinite(v) or v <= 0 for values in samples for v in values)):
            continue
        expected = statistics.median(statistics.median(values) for values in samples)
        if not math.isclose(expected, float(row["t_skew_us"]), abs_tol=0.00051):
            raise ValueError("Stored skew target does not match its raw repetitions")
        grouped.setdefault(key, set()).add(row.get("layer") or "attention")
    return {key for key, present in grouped.items() if present >= set(layers)}


def sample_skew(llm, arch, args, limits, tp, tp_root):
    """Return its CSV path, or None for an empty plan without changing artifacts.

    Failures preserve completed cases and raise.
    """
    if args.skew_rounds < 1 or args.measurement_iterations < 1:
        raise ValueError("Positive skew repetition counts required")
    out = Path(tp_root) / "skew.csv"
    attention = Path(tp_root) / "attention.csv"
    if not attention.exists():
        raise FileNotFoundError("Skew calibration needs attention.csv; profile attention first")
    from .hooks.capture_scope_contract import contexts_from_architecture
    scope_contexts = contexts_from_architecture(arch)
    from .categories import _entry_dict
    from . import attention_lookup as lookup_contract
    architecture = arch.model_dump(exclude_none=True)
    record, _ = lookup_contract.semantic_for_table(attention, args.model_config, architecture)
    live = lookup_contract.acquire(llm, args.model_config, architecture, tp)
    if live != (record['semantic'] if record else None):
        raise ValueError('Skew runtime differs from the attention acquisition contract')
    attention_sha = record['semantic_sha256'] if record else None
    measured_protocol = lookup_contract.protocol(PROTOCOL, attention_sha)
    from .component_layouts import selected_catalog_for_tp
    choices, entries = selected_catalog_for_tp(
        tp_root, architecture, "attention", arch.catalog.attention)
    catalog = _entry_dict(entries, arch)
    component_layout_contract = recorded_contract_for_skew(tp_root, arch, choices, args,
        logical_tp=tp, logical_ep=getattr(limits, 'moe_ep', None), block_size=limits.block_size)
    # Fail before force/resume mutation, initialization or any worker request.
    capture_catalog_for_skew(catalog, component_layout_contract)
    fingerprint = measurement_fingerprint(scope_contexts, catalog=catalog,
                                          attention_lookup_sha256=attention_sha,
                                          component_layout_contract=component_layout_contract)
    from serving.core import trace_generator as tg
    table = tg._build_attention_tables_by_layer(tg._read_category_csv(str(attention), None))
    for name in catalog:
        if not set(args.attention_decode_q_lens) <= set(table.get(name, {})):
            raise ValueError(f"Missing exact attention query slices for {name}")
    runtime = _admit_skew_runtime(llm, component_layout_contract, args.attention_decode_q_lens)
    if out.exists() and not args.force:
        recorded = pd.read_csv(out, low_memory=False).fillna('')
        metadata = _measurement_metadata(out)
        expected = dict(measurement_protocol=measured_protocol,
                        measurement_sha256=fingerprint,
                        block_size=limits.block_size)
        if any((not (recorded[k] == v).all()) if k in recorded else metadata.get(k) != v
               for k, v in expected.items()):
            raise ValueError(
                'Existing skew uses a different acquisition method or block size; '
                'use a separate output root or --force to remeasure')
    prior = set() if args.force else _existing_keys(
        out, catalog, args.skew_rounds, args.measurement_iterations, block_size=limits.block_size,
        scope_contexts=scope_contexts, attention_lookup_sha256=attention_sha,
        component_layout_contract=component_layout_contract)
    log.info("TP=%d skew: checking reference-cell support before acquisition", tp)
    plan, extra = complete_plan(args, limits, attention, arch, prior,
                               existing_csv=None if args.force else out)
    if not plan['cases']:
        log.info("TP=%d skew: empty plan; preserving existing artifacts", tp)
        return None
    if (plan['reusable_cases'] == plan['cases'] and out.exists()
            and _measurement_metadata(out).get('completed') is True):
        log.info('TP=%d skew: all planned cases already measured; preserving metadata', tp)
        return None
    # A crash can leave every validated row durable but its completion record
    # pending. Run the normal skip-only publication path to seal that record;
    # it performs no measurements and adds no acquisition provenance.
    plan.update(enabled=True, tp=tp, measurement_protocol=measured_protocol,
                measurement_sha256=fingerprint,
                attention_lookup_sha256=attention_sha,
                scope_contexts=scope_contexts, completed=False, measured_cases=0)
    from .measurement import method_metadata
    from importlib import metadata as distributions
    versions = {}
    for package in ('torch', 'vllm'):
        try:
            versions[package] = distributions.version(package)
        except distributions.PackageNotFoundError:
            versions[package] = None
    facts = dict(method=method_metadata(args.measurement_iterations), runtime_versions=versions)
    previous = _measurement_metadata(out) if out.exists() and not args.force else {}
    acquisitions = list(previous.get('acquisitions', []))
    plan['acquisitions'] = acquisitions
    plan['case_families'] = dict(previous.get('case_families', {}))
    if component_layout_contract is not None:
        plan["component_layout_sha256"] = component_layout_contract["layout_sha256"]
        plan['runtime'] = runtime
    Path(tp_root).mkdir(parents=True, exist_ok=True)
    if args.force and out.exists():
        out.unlink()  # Explicit --force discards this category's old acquisition.
    total = plan["cases"]
    log.info("TP=%d skew: %d planned cases, %d reusable; %d rounds x %d forwards",
             tp, total, plan["reusable_cases"], args.skew_rounds, args.measurement_iterations)
    support = plan["support_completion"]
    log.info("TP=%d skew: %d reference-selected additions; %d cells remain below support floor",
             tp, len(extra), len(support["remaining_deficits"]))
    write_tp(tp_root, 'skew', plan)
    rows, fired, skipped = [], 0, 0
    started, initialized = time.monotonic(), False
    try:
        for shot, family, key in chain(iter_cases(args, limits), extra):
            if key in prior:
                skipped += 1
                continue
            if not initialized:
                llm.collective_rpc("skew_initialize")
                initialized = True
            by_layer = {name: [] for name in catalog}
            for _ in range(args.skew_rounds):
                rpc_args = (shot.as_dict(), catalog, args.measurement_iterations, scope_contexts,
                            live)
                if component_layout_contract is not None:
                    rpc_args += (component_layout_contract,)
                results = llm.collective_rpc("skew_measure", args=rpc_args)
                if not results:
                    raise ValueError('No worker skew timings returned')
                for result in results:
                    if (result.get('measurement_protocol') != measured_protocol
                            or result.get('measurement_sha256') != plan['measurement_sha256']):
                        raise ValueError('Worker skew timings do not match the requested acquisition method')
                    if component_layout_contract is not None and result.get('component_layout_sha256') != component_layout_contract['layout_sha256']:
                        raise ValueError('Worker skew layout validation is missing or differs')
                    if runtime is not None and result.get('runtime_sha256') != runtime['runtime_sha256']:
                        raise ValueError('Worker skew runtime validation is missing or differs')
                    forwards = result["per_forward_us"]
                    if len(forwards) != args.measurement_iterations:
                        raise ValueError("Incomplete skew timed-forward set")
                    if any(set(row) != set(catalog) or any(not math.isfinite(value) or value <= 0
                           for value in row.values()) for row in forwards):
                        raise ValueError("Invalid or incomplete skew kernel time")
                result = results[0]
                forwards = result["per_forward_us"]
                for name in catalog:
                    values = [float(row[name]) for row in forwards]
                    by_layer[name].append(values)
            for name, samples in by_layer.items():
                target = statistics.median(statistics.median(values) for values in samples)
                rows.append(dict(layer=name, case_id=key,
                    requests_json=json.dumps(shot.requests, separators=(",", ":")),
                    n_prefill=shot.n_prefill, decode_q_len=shot.decode_q_len,
                    round_timings_us_json=json.dumps(samples, separators=(",", ":")),
                    t_skew_us=round(target, 3)))
            if facts not in acquisitions:
                acquisitions.append(facts)
            plan['case_families'][key] = family
            fired += 1
            if fired % 20 == 0:
                _flush_rows(out, rows)
                rows.clear()
                elapsed = max(time.monotonic() - started, 1e-9)
                remaining = max(0, total - fired - plan["reusable_cases"])
                log.info("TP=%d skew: %d/%d cases, %.2f case/s, eta %.1f min",
                         tp, fired + skipped, total, fired / elapsed,
                         remaining * elapsed / fired / 60)
                plan.update(measured_cases=fired, reused_cases=skipped, remaining_cases=remaining)
                write_tp(tp_root, 'skew', plan)
    finally:
        if rows:
            _flush_rows(out, rows)
    plan.update(completed=True, measured_cases=fired, reused_cases=skipped,
                remaining_cases=0, elapsed_seconds=time.monotonic() - started)
    if attention_sha is not None:
        plan['skew_sha256'] = lookup_contract.checksum(out)
    write_tp(tp_root, 'skew', plan)
    log.success("skew -> %s", out)
    return out
