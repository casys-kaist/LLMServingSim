"""Explicit runtime-selected component representations and acquisition state.

This module has no model-family, Torch, or GPU dependencies. Architecture
declarations define alternatives. Positive runtime coverage selects one; a
per-TP metadata record couples categories that share the representation.
Unannotated historical bundles retain their original block decomposition.
``meta.yaml.component_layouts`` is the single authoritative state. Tables
carry their existing acquisition stamps; no journal or alternate lookup is used.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import yaml

SCHEMA = "component-layouts-v1"
REGIMES = ("prefill", "decode", "mixed")
MEASUREMENT_FIELDS = ("measurement_protocol", "measurement_sha256")


def _hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _mapping(value, keys, required, where):
    if not isinstance(value, dict) or set(value) - set(keys) or set(required) - set(value):
        raise ValueError(f"Invalid {where}: expected {sorted(required)}, allowed {sorted(keys)}")


def _names(value, where, *, nonempty=True):
    if (not isinstance(value, list) or (nonempty and not value)
            or any(not isinstance(x, str) or not x for x in value)
            or len(value) != len(set(value))):
        raise ValueError(f"Invalid distinct names in {where}")
    return set(value)


def _catalog(architecture):
    result = {}
    for category, entries in architecture["catalog"].items():
        for name, entry in (entries or {}).items():
            if name in result:
                raise ValueError(f"Repeated canonical layer: {name}")
            result[name] = (category, entry)
    return result


def _sequences(architecture):
    blocks = architecture.get("blocks") or {}
    for axis in ("attn", "sparse_attn"):
        for block in (blocks.get(axis) or {}).values():
            yield [*(block.get("pre_attn") or []), *(block.get("post_attn") or [])]
    yield from (blocks.get("mlp") or {}).values()
    for section in ("shared", "mtp"):
        value = architecture.get(section) or {}
        yield [*(value.get("prologue") or []), *(value.get("head") or [])]


def _members(group):
    return set().union(*(set(names) for variant in group["variants"].values()
                         for names in variant["observed"].values()))


def _active(group, selection):
    return set().union(*(set(names) for names in
                         group["variants"][selection]["observed"].values()))


def validate_definitions(architecture):
    """Validate canonical references and disjoint, nonrecursive rewrites.

    The historical variant is exactly the original block order. A variant's
    observed component union must equal the historical union after its one-
    pass substitutions. Categories are explicit because unchanged canonical
    names (for example, glue outside a newly fused core) can also change scope.
    """
    groups = architecture.get("component_layouts") or {}
    if not isinstance(groups, dict):
        raise ValueError("component_layouts must be a mapping")
    if not groups:
        return groups
    catalog = _catalog(architecture)
    claimed = set()
    for name, group in groups.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid component-layout name")
        _mapping(group, {"historical", "categories", "variants"},
                 {"historical", "categories", "variants"}, name)
        categories = _names(group["categories"], f"{name}.categories")
        if not categories <= set(architecture["catalog"]):
            raise ValueError(f"Unknown category in {name}")
        if not categories <= {"dense", "linear_attention"}:
            raise ValueError(f"Component layouts currently require dense/linear_attention row domains: {name}")
        variants = group["variants"]
        if not isinstance(variants, dict) or group["historical"] not in variants:
            raise ValueError(f"Missing historical component layout in {name}")
        if len(variants) < 2:
            raise ValueError(f"{name} needs at least two explicit alternatives")
        for label, variant in variants.items():
            if not isinstance(label, str) or not label:
                raise ValueError(f"Invalid variant in {name}")
            _mapping(variant, {"observed", "replace"}, {"observed"}, f"{name}.{label}")
            observations = variant["observed"]
            if not isinstance(observations, dict) or set(observations) != set(REGIMES):
                raise ValueError(f"{name}.{label} must declare all three regimes")
            for regime, names in observations.items():
                if not _names(names, f"{name}.{label}.{regime}") <= set(catalog):
                    raise ValueError(f"Unknown component in {name}.{label}")
            replacements = variant.get("replace") or {}
            if not isinstance(replacements, dict):
                raise ValueError(f"Invalid replacements in {name}.{label}")
            for anchor, names in replacements.items():
                if anchor not in catalog or not _names(names, f"replacement {anchor}", nonempty=False) <= set(catalog):
                    raise ValueError(f"Unknown replacement component in {name}.{label}")
        historical = group["historical"]
        if variants[historical].get("replace"):
            raise ValueError(f"Historical layout {name} cannot rewrite existing blocks")
        baseline = _active(group, historical)
        members = _members(group)
        if members & claimed:
            raise ValueError(f"Component layouts have overlapping members: {members & claimed}")
        claimed |= members
        if not {catalog[layer][0] for layer in members} <= categories:
            raise ValueError(f"{name} omits an affected category")
        declared = set().union(*(set(sequence) for sequence in _sequences(architecture)))
        if not baseline <= declared or (members - baseline) & declared:
            raise ValueError(f"{name} does not preserve the historical block decomposition")
        for label, variant in variants.items():
            replacements = variant.get("replace") or {}
            if not set(replacements) <= baseline:
                raise ValueError(f"{name}.{label} may rewrite historical members only")
            transformed = {item for old in baseline for item in replacements.get(old, [old])}
            if transformed != _active(group, label):
                raise ValueError(f"{name}.{label} replacement and observed members disagree")
            # A consumed historical member without its replacement anchor in
            # the same block would delete work from an unrelated block.
            for sequence in _sequences(architecture):
                if not set(sequence) & baseline:
                    continue
                rewritten = [item for old in sequence for item in replacements.get(old, [old])]
                if set(rewritten) & members != transformed:
                    raise ValueError(f"{name}.{label} does not form a complete block")
                if any(rewritten.count(layer) != 1 for layer in transformed):
                    raise ValueError(f"Repeated component-layout members need explicit normalization: {name}.{label}")
    return groups


def select_runtime_layouts(architecture, reports):
    """Choose only from positive launch-correlated runtime coverage.

    Reports have CoverageReport's native representation. CPU event existence
    alone, vLLM version, raw kernel spelling and model names select nothing.
    Unrelated coverage gaps remain visible to the caller's broader audit.
    """
    groups = validate_definitions(architecture)
    if not groups:
        return {}, {}
    if set(reports) != set(REGIMES):
        raise ValueError("Component-layout discovery requires prefill, decode and mixed coverage")
    observed = {}
    for regime, report in reports.items():
        if report.get("over_matches"):
            raise ValueError(f"Ambiguous activity ownership during {regime} layout discovery")
        values = report["bound_us_by_layer"]
        if any(not math.isfinite(value) or value <= 0 for value in values.values()):
            raise ValueError(f"Nonpositive/nonfinite observed coverage in {regime}")
        observed[regime] = set(values)
    choices, evidence = {}, {}
    for name, group in groups.items():
        members = _members(group)
        seen = {regime: observed[regime] & members for regime in REGIMES}
        candidates = [label for label, variant in group["variants"].items()
                      if all(seen[regime] == set(variant["observed"][regime])
                             for regime in REGIMES)]
        if len(candidates) != 1:
            raise ValueError(f"No unique measured component layout for {name}: {seen}")
        choices[name] = candidates[0]
        evidence[name] = {regime: sorted(seen[regime]) for regime in REGIMES}
    return choices, evidence


def layout_identity(architecture, choices):
    groups = validate_definitions(architecture)
    if set(choices) != set(groups):
        raise ValueError("A component layout must select every declared group")
    for name, label in choices.items():
        if label not in groups[name]["variants"]:
            raise ValueError(f"Unknown selection {name}={label}")
    categories = set().union(*(set(group["categories"]) for group in groups.values())) if groups else set()
    # Include affected-category bindings: an unchanged canonical glue name can
    # have a different exclusion boundary in a fused representation.
    definitions = {
        name: dict(historical=group["historical"], categories=sorted(group["categories"]),
                   variants={label: dict(observed={regime: sorted(names)
                                                    for regime, names in variant["observed"].items()},
                                         replace=variant.get("replace") or {})
                             for label, variant in group["variants"].items()})
        for name, group in groups.items()}
    defaults = {"tp_stable": False, "key_saturates": False, "normalization": "parent"}
    catalog = {
        category: {name: {key: value for key, value in entry.items()
                           if value is not None and not (key in defaults and value == defaults[key])}
                   for name, entry in architecture["catalog"][category].items()}
        for category in sorted(categories)}
    return _hash(dict(schema=SCHEMA, definitions=definitions, choices=choices, catalog=catalog))


def expand_sequence(architecture, choices, sequence):
    """Resolve one sequence without mutating the shared architecture or TP cache."""
    groups = architecture.get("component_layouts") or {}
    choices = choices if choices is not None else {n: g["historical"] for n, g in groups.items()}
    if set(choices) != set(groups):
        raise ValueError("Incomplete component-layout selection")
    replacements = {}
    for name, group in groups.items():
        replacements.update(group["variants"][choices[name]].get("replace") or {})
    return [item for layer in sequence for item in replacements.get(layer, [layer])]


def filter_catalog_slice(architecture, choices, catalog_slice):
    """Remove alternative-only components; preserve all unrelated bindings."""
    excluded = set()
    for name, group in (architecture.get("component_layouts") or {}).items():
        excluded |= _members(group) - _active(group, choices[name])
    return {name: entry for name, entry in catalog_slice.items() if name not in excluded}


def probe_catalog(architecture, whole_catalog):
    """All alternatives, used for discovery and observed per-shot guards.

    The same capture supplies the current category's data and these small
    ownership probes; no extra timed forward is needed for per-shot guards.
    """
    members = set().union(*(_members(group) for group in
                            (architecture.get("component_layouts") or {}).values()))
    if not members <= set(whole_catalog):
        raise ValueError("Serialized catalog lacks declared component probes")
    return {name: whole_catalog[name] for name in sorted(members)}


def validate_runtime_shot(architecture, choices, regime, measured_layers):
    """Also check another category's probes: an empty dense core set is not evidence."""
    if regime not in REGIMES:
        raise ValueError(f"Unknown layout acquisition regime: {regime}")
    for name, group in (architecture.get("component_layouts") or {}).items():
        expected = set(group["variants"][choices[name]]["observed"][regime])
        found = set(measured_layers) & _members(group)
        if found != expected:
            raise ValueError(f"Runtime layout changed or is incomplete for {name}/{regime}: "
                             f"expected {sorted(expected)}, found {sorted(found)}")


def expected_components(architecture, choices, category, regime):
    catalog = _catalog(architecture)
    result, owned = set(), set()
    for name, group in (architecture.get("component_layouts") or {}).items():
        owned |= {layer for layer in _members(group) if catalog[layer][0] == category}
        result |= {layer for layer in group["variants"][choices[name]]["observed"][regime]
                   if catalog[layer][0] == category}
    return result, owned


def validate_shot_components(architecture, choices, category, regime, measured_layers):
    expected, owned = expected_components(architecture, choices, category, regime)
    if set(measured_layers) & owned != expected:
        raise ValueError(f"Incomplete/mixed {category}/{regime} component layout: "
                         f"expected {sorted(expected)}, found {sorted(set(measured_layers) & owned)}")


def _read_meta(root):
    """An invalid metadata file must not be treated as an old, unmarked bundle."""
    path = Path(root) / "meta.yaml"
    if not path.exists():
        return {}
    class UniqueLoader(yaml.SafeLoader):
        pass
    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if key in result:
                raise ValueError(f"Repeated metadata key {key!r} in {path}")
            result[key] = loader.construct_object(value_node)
        return result
    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    value = yaml.load(path.read_text(), Loader=UniqueLoader)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid metadata mapping: {path}")
    return value


def _publish(root, metadata):
    from profiler.core.skew_calibration import atomic_yaml
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_yaml(root / "meta.yaml", metadata)
    _sync_directory(root)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _clear_category(path, *, preserve_layers=()):
    """Clear only after pending is durable; an interrupted reset stays pending."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header, preserved = [], []
    if preserve_layers and path.is_file() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames or []
            preserved = [row for row in reader if row.get("layer") in preserve_layers]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            if preserved:
                writer = csv.DictWriter(stream, fieldnames=header)
                writer.writeheader()
                writer.writerows(preserved)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _measurement(value):
    if (not isinstance(value, dict) or set(value) != set(MEASUREMENT_FIELDS)
            or any(not isinstance(item, str) or not item for item in value.values())):
        raise ValueError("Missing measurement identity")


def _validate_record(architecture, value):
    _mapping(value, {"layout_sha256", "selections", "acquisition", "categories"},
             {"layout_sha256", "selections", "acquisition", "categories"}, "layout record")
    if value["layout_sha256"] != layout_identity(architecture, value["selections"]):
        raise ValueError("Stale/invalid component-layout acquisition contract; use a fresh output bundle")
    _measurement(value["acquisition"])
    groups = architecture.get("component_layouts") or {}
    required = set().union(*(set(group["categories"]) for group in groups.values()))
    if set(value["categories"]) != required:
        raise ValueError("Incomplete component-layout acquisition contract")
    for category, record in value["categories"].items():
        _mapping(record, {"state", "measurement"}, {"state"}, category)
        if record["state"] not in ("pending", "writing", "complete"):
            raise ValueError(f"Invalid acquisition state for {category}")
        if record["state"] != "pending":
            _measurement(record.get("measurement"))
        elif "measurement" in record:
            raise ValueError(f"Pending category {category} cannot certify existing rows")
    return value


def metadata_records(variant_root, architecture):
    """Read/validate the sole record; metadata writers preserve it verbatim."""
    value = _read_meta(variant_root).get("component_layouts")
    if value is None:
        return None
    _mapping(value, {"schema", "per_tp"}, {"schema", "per_tp"}, "component layouts")
    if value["schema"] != SCHEMA or not isinstance(value["per_tp"], dict):
        raise ValueError("Invalid component-layout metadata schema")
    normalized = {}
    for tp, record in value["per_tp"].items():
        if (type(tp) not in (int, str) or not str(tp).isascii()
                or not str(tp).isdigit() or int(tp) < 1 or int(tp) in normalized):
            raise ValueError("Invalid/repeated component-layout TP degree")
        normalized[int(tp)] = _validate_record(architecture, record)
    return dict(schema=SCHEMA, per_tp=normalized)


def begin_category(tp_root, architecture, choices, category, measurement,
                   acquisition, *, replace=False):
    """Durably invalidate before replacement; resume only certified whole shots.

    A changed acquisition (versions, source or repetitions) resets *all* coupled
    categories under explicit force. Refreshing one category with the same
    acquisition preserves its independently validated siblings. A different
    representation always requires a fresh bundle.
    """
    root = Path(tp_root)
    groups = validate_definitions(architecture)
    categories = set().union(*(set(group["categories"]) for group in groups.values())) if groups else set()
    if category not in categories:
        return None
    _measurement(measurement)
    _measurement(acquisition)
    metadata = _read_meta(root.parent)
    block = metadata_records(root.parent, architecture) or dict(schema=SCHEMA, per_tp={})
    metadata["component_layouts"] = block
    tp = int(root.name.removeprefix("tp"))
    identity = layout_identity(architecture, choices)
    value = block["per_tp"].get(tp)
    if value is not None:
        if value["layout_sha256"] != identity:
            raise ValueError("Mixed component layouts across categories; use a fresh output bundle")
        if value["acquisition"] != acquisition:
            if not replace:
                raise ValueError("Changed coupled acquisition cannot resume; force all affected categories")
            value["acquisition"] = dict(acquisition)
            value["categories"] = {name: {"state": "pending"} for name in sorted(categories)}
    else:
        if not replace and any((root / f"{name}.csv").exists() for name in categories):
            raise ValueError("Historical category CSVs need coordinated --force refresh or a fresh output bundle")
        value = dict(layout_sha256=identity, selections=dict(choices), acquisition=dict(acquisition),
                     categories={name: {"state": "pending"} for name in sorted(categories)})
        block["per_tp"][tp] = value
    record = value["categories"][category]
    if record["state"] != "pending" and record.get("measurement") != measurement and not replace:
        raise ValueError(f"Changed {category} measurement identity cannot resume")
    # A request to repair this category does not excuse a corrupt sibling.
    for sibling, state in value["categories"].items():
        if sibling == category and replace:
            continue
        if state["state"] != "pending":
            _category_content(root, architecture, value, sibling,
                              allow_partial=state["state"] == "writing")
    if replace or record["state"] == "pending":
        value["categories"][category] = {"state": "pending"}
        _publish(root.parent, metadata)
        # TP-stable rows belong to their source degree, not this category's
        # acquisition. A reset must not erase or recertify those replicas.
        owned = set().union(*(_members(group) for group in groups.values()))
        replicated = {name for name, entry in architecture["catalog"][category].items()
                      if tp > 1 and entry.get("tp_stable") and name not in owned}
        _clear_category(root / f"{category}.csv", preserve_layers=replicated)
    value["categories"][category] = dict(state="writing", measurement=dict(measurement))
    _validate_record(architecture, value)
    _publish(root.parent, metadata)
    return value


def _category_content(root, architecture, record, category, *, allow_partial=False):
    path = root / f"{category}.csv"
    if not path.is_file():
        if allow_partial:
            return
        raise ValueError(f"Missing acquired category: {path}")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not fields and allow_partial:
            return
        if len(fields) != len(set(fields)) or not {"layer", "time_us", *MEASUREMENT_FIELDS} <= set(fields):
            raise ValueError(f"Invalid acquired category header: {path}")
        rows = list(reader)
    catalog = architecture["catalog"].get(category) or {}
    # Validate every stored row before excluding source-owned replicas from
    # this TP's acquisition coverage. Their different TP1 identity is allowed;
    # malformed rows, invalid times and duplicate coordinates are not.
    members = set().union(*(_members(group) for group in architecture["component_layouts"].values()))
    tp = int(root.name.removeprefix("tp"))
    measurement = record["categories"][category]["measurement"]
    shots = {}
    seen = set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed acquired category row: {path}")
        if row["layer"] not in catalog:
            raise ValueError(f"Unknown canonical component in {path}: {row['layer']}")
        _measurement({field: row[field] for field in MEASUREMENT_FIELDS})
        latency = float(row["time_us"])
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"Invalid acquired latency in {path}")
        coords = tuple((key, row[key]) for key in fields if key not in
                       {"layer", "time_us", *MEASUREMENT_FIELDS})
        if category == "linear_attention":
            prefill, decode = int(row["prefill_tokens"]), int(row["n_decode"])
            if prefill < 0 or decode < 0 or not prefill + decode:
                raise ValueError(f"Invalid linear-attention point in {path}")
            regime = "mixed" if prefill and decode else "prefill" if prefill else "decode"
        else:
            # Token-linear categories are acquired with one fresh request.
            # Its single-token case enters the native decode fast path.
            # No sequence-domain category can currently hold these layouts.
            if category != "dense":
                raise ValueError(f"No acquired-row regime adapter for {category}")
            tokens = int(row["tokens"])
            if tokens < 1:
                raise ValueError(f"Invalid dense point in {path}")
            regime = "prefill" if tokens > 1 else "decode"
        key = (regime, coords)
        row_key = (row["layer"], key)
        if row_key in seen:
            raise ValueError(f"Duplicate acquired component row in {path}")
        seen.add(row_key)
        replicated = (tp > 1 and row["layer"] not in members
                      and catalog[row["layer"]].get("tp_stable", False))
        if replicated:
            continue
        if any(row[field] != value for field, value in measurement.items()):
            raise ValueError(f"Mixed measurement identity in {path}")
        present = shots.setdefault(key, set())
        present.add(row["layer"])
    if not shots:
        if allow_partial:
            return
        raise ValueError(f"No nonstable acquisition rows in {path}")
    for (regime, _), present in shots.items():
        validate_shot_components(architecture, record["selections"], category, regime, present)
    if not allow_partial and category == "linear_attention" and {key[0] for key in shots} != set(REGIMES):
        raise ValueError(f"Acquired linear-attention layout lacks an entire regime: {path}")


def complete_category(tp_root, architecture, category):
    root = Path(tp_root)
    metadata = _read_meta(root.parent)
    block = metadata_records(root.parent, architecture)
    if block is None:
        raise ValueError("Missing component-layout acquisition record")
    metadata["component_layouts"] = block
    value = block["per_tp"][int(root.name.removeprefix("tp"))]
    record = value["categories"][category]
    if record["state"] not in ("writing", "complete"):
        raise ValueError(f"Category {category} was never started")
    _category_content(root, architecture, value, category)
    record["state"] = "complete"
    _publish(root.parent, metadata)
    return value


def read_for_trace(tp_root, architecture):
    """Require every coupled category complete; never infer layout from row names."""
    root = Path(tp_root)
    groups = validate_definitions(architecture)
    if not groups:
        return None
    block = metadata_records(root.parent, architecture)
    record = (block or {}).get("per_tp", {}).get(int(root.name.removeprefix("tp")))
    if record is None:
        choices = {name: group["historical"] for name, group in groups.items()}
        forbidden = set().union(*(_members(group) - _active(group, group["historical"])
                                  for group in groups.values()))
        for category in set().union(*(set(g["categories"]) for g in groups.values())):
            csv_path = root / f"{category}.csv"
            if csv_path.exists():
                with csv_path.open(newline="") as stream:
                    if any(row.get("layer") in forbidden for row in csv.DictReader(stream)):
                        raise ValueError("New-layout rows lack an acquisition contract")
        return choices
    for category, state in record["categories"].items():
        if state["state"] != "complete":
            raise ValueError(f"Component-layout acquisition is incomplete: {category}={state['state']}")
        _category_content(root, architecture, record, category)
    return dict(record["selections"])
