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
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

SCHEMA = "component-layouts"
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


def sequence_sites(architecture):
    """Existing block/shared/MTP sequence paths, without inventing missing sites."""
    result = {}
    blocks = architecture.get("blocks") or {}
    for axis in ("attn", "sparse_attn"):
        for block, parts in (blocks.get(axis) or {}).items():
            for part in ("pre_attn", "post_attn"):
                if part in parts:
                    result[("blocks", axis, block, part)] = tuple(parts[part] or [])
    for block, sequence in (blocks.get("mlp") or {}).items():
        result[("blocks", "mlp", block)] = tuple(sequence)
    for section in ("shared", "mtp"):
        value = architecture.get(section) or {}
        for part in ("prologue", "head"):
            if part in value:
                result[(section, part)] = tuple(value[part] or [])
    return result


def _site_path(value, where):
    if (not isinstance(value, (tuple, list)) or not value
            or any(not isinstance(item, str) or not item for item in value)):
        raise ValueError(f"Invalid exact sequence site: {where}")
    return tuple(value)


def _site_footprint(group, all_sites, where):
    value = group.get("sites")
    if not isinstance(value, list) or not value:
        raise ValueError(f"Explicit component sites must be nonempty: {where}")
    paths = [_site_path(path, where) for path in value]
    if (len(paths) != len(set(paths)) or not set(paths) <= set(all_sites)
            or any(path[0] not in ("blocks", "shared") for path in paths)):
        raise ValueError(f"Duplicate, unknown or unsupported component site: {where}")
    return set(paths)


def _rewrite(sequence, replacements):
    return [item for old in sequence for item in replacements.get(old, [old])]


def _validate_site_variant(name, label, group, catalog, all_sites, guarded_paths):
    footprint = _site_footprint(group, all_sites, name)
    baseline = _active(group, group["historical"])
    members = _members(group)
    declared = {item for path in footprint for item in all_sites[path]}
    if not baseline <= declared or (members - baseline) & declared:
        raise ValueError(f"Historical site membership changed: {name}")
    if any(set(sequence) & baseline for path, sequence in all_sites.items() if path not in footprint):
        raise ValueError(f"Component layout escapes its declared sites: {name}")
    variant = group["variants"][label]
    replacements = variant.get("replace") or {}
    active = _active(group, label)
    rewritten = {path: _rewrite(all_sites[path], replacements) for path in footprint}
    if label != group["historical"]:
        counts = Counter(item for sequence in rewritten.values() for item in sequence if item in active)
        if any(count > 1 and catalog[item][1].get("normalization") != "invocations"
               for item, count in counts.items()):
            raise ValueError(f"Repeated site components require invocation normalization: {name}.{label}")
    rules = variant.get("layer0") or []
    if not isinstance(rules, list):
        raise ValueError(f"layer0 requires an explicit rule list: {name}.{label}")
    for rule in rules:
        _mapping(rule, {"path", "expected_prefix", "replacement_prefix", "consumed_by"},
                 {"path", "expected_prefix", "replacement_prefix", "consumed_by"}, "layer0 rule")
        path = _site_path(rule["path"], "layer0")
        if (path not in footprint or len(path) != 4
                or path[:2] not in (("blocks", "attn"), ("blocks", "sparse_attn"))):
            raise ValueError("layer0 must name a declared decoder attention sequence")
        guard_key = (label, path)
        # Each variant has at most one guard at a site. Different groups may
        # not conditionally edit the same prefix in an order-dependent way.
        if guard_key in guarded_paths.get(name, set()) or any(
                path in {key[1] for key in keys} for owner, keys in guarded_paths.items() if owner != name):
            raise ValueError("Overlapping or duplicate target-layer-zero prefix rules")
        guarded_paths.setdefault(name, set()).add(guard_key)
        expected = rule["expected_prefix"]
        retained = rule["replacement_prefix"]
        _names(expected, "layer0.expected_prefix")
        _names(retained, "layer0.replacement_prefix", nonempty=False)
        if not set(expected) <= active or not set(retained) <= set(expected):
            raise ValueError("layer0 may consume only active members of its own group")
        # This contract consumes work into an observed prologue; it does not
        # authorize inventing or reordering a special first-layer operation.
        if len(retained) >= len(expected) or [item for item in expected if item in retained] != retained:
            raise ValueError("layer0 replacement must retain an ordered proper subset of its prefix")
        if rewritten[path][:len(expected)] != expected:
            raise ValueError("Declared layer0 prefix no longer matches")
        if any(catalog[item][1].get("normalization") != "invocations" for item in expected):
            raise ValueError("Consumed decoder components require invocation normalization")
        consumer = rule["consumed_by"]
        if not isinstance(consumer, str) or consumer not in active:
            raise ValueError("layer0 requires an active consuming prologue component")
        prologue = rewritten.get(("shared", "prologue"), [])
        if prologue.count(consumer) != 1:
            raise ValueError("Consuming component must appear once in the declared shared prologue")
        required = set(expected) | {consumer}
        if any(not required <= set(variant["observed"][regime]) for regime in REGIMES):
            raise ValueError("Layer-zero consumption requires positive observations in every regime")


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
    all_sites = sequence_sites(architecture)
    guarded_paths = {}
    for name, group in groups.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid component-layout name")
        _mapping(group, {"historical", "categories", "variants", "sites"},
                 {"historical", "categories", "variants"}, name)
        categories = _names(group["categories"], f"{name}.categories")
        if not categories <= set(architecture["catalog"]):
            raise ValueError(f"Unknown category in {name}")
        site_aware = group.get("sites") is not None
        domains = {"dense", "attention", "linear_attention"} if site_aware else {"dense", "linear_attention"}
        if not categories <= domains:
            raise ValueError(f"Component layouts currently require dense/linear_attention row domains: {name}")
        variants = group["variants"]
        if not isinstance(variants, dict) or group["historical"] not in variants:
            raise ValueError(f"Missing historical component layout in {name}")
        if len(variants) < 2:
            raise ValueError(f"{name} needs at least two explicit alternatives")
        for label, variant in variants.items():
            if not isinstance(label, str) or not label:
                raise ValueError(f"Invalid variant in {name}")
            _mapping(variant, {"observed", "replace", "layer0", "invocation_contract", "runtime_contract"},
                     {"observed"}, f"{name}.{label}")
            if variant.get("layer0") is not None and not site_aware:
                raise ValueError("Target-layer-zero rules require declared sequence sites")
            if variant.get("layer0") is not None and not isinstance(variant["layer0"], list):
                raise ValueError("layer0 requires an explicit rule list")
            observations = variant["observed"]
            if not isinstance(observations, dict) or set(observations) != set(REGIMES):
                raise ValueError(f"{name}.{label} must declare all three regimes")
            for regime, names in observations.items():
                if not _names(names, f"{name}.{label}.{regime}") <= set(catalog):
                    raise ValueError(f"Unknown component in {name}.{label}")
            contract = variant.get("invocation_contract")
            runtime = variant.get("runtime_contract")
            if runtime is not None and (runtime != "native-dsa-eager-tail"
                    or contract != "site-scope-calls"):
                raise ValueError("Runtime admission requires a known explicit scope-call contract")
            if contract is not None:
                if contract != "site-scope-calls" or not site_aware or label == group["historical"]:
                    raise ValueError("Scope-call contracts require an explicit nonhistorical site variant")
                first = set(observations[REGIMES[0]])
                if any(set(observations[regime]) != first for regime in REGIMES):
                    raise ValueError("Scope-call contracts require the same positive members in every regime")
            replacements = variant.get("replace") or {}
            if not isinstance(replacements, dict):
                raise ValueError(f"Invalid replacements in {name}.{label}")
            for anchor, names in replacements.items():
                if anchor not in catalog or not _names(names, f"replacement {anchor}", nonempty=False) <= set(catalog):
                    raise ValueError(f"Unknown replacement component in {name}.{label}")
        historical = group["historical"]
        if variants[historical].get("replace") or variants[historical].get("layer0"):
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
            if site_aware:
                _validate_site_variant(name, label, group, catalog, all_sites, guarded_paths)
                continue
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
    for name, group in groups.items():
        if group.get("sites") is not None:
            sites = sequence_sites(architecture)
            paths = sorted(_site_path(path, name) for path in group["sites"])
            definitions[name]["sites"] = [list(path) for path in paths]
            # Only site-aware definitions bind original order. Do not change
            # existing groups' identity by introducing an empty/default field.
            definitions[name]["site_sequences"] = [[list(path), list(sites[path])] for path in paths]
        for label, variant in group["variants"].items():
            if variant.get("invocation_contract") is not None:
                definitions[name]["variants"][label]["invocation_contract"] = variant["invocation_contract"]
            if variant.get("runtime_contract") is not None:
                definitions[name]["variants"][label]["runtime_contract"] = variant["runtime_contract"]
            if variant.get("layer0"):
                definitions[name]["variants"][label]["layer0"] = sorted(
                    variant["layer0"], key=lambda rule: tuple(rule["path"]))
    defaults = {"tp_stable": False, "key_saturates": False, "normalization": "parent"}
    catalog = {
        category: {name: {key: value for key, value in entry.items()
                           if value is not None and not (key in defaults and value == defaults[key])}
                   for name, entry in architecture["catalog"][category].items()}
        for category in sorted(categories)}
    return _hash(dict(schema=SCHEMA, definitions=definitions, choices=choices, catalog=catalog))


def spec_block_site(architecture, spec, part):
    """Resolve one target block site, including an explicitly declared sparse overlay.

    This is the shared path-selection rule for trace composition and physical
    scope-call expectations. Missing block types are not uniform fallbacks.
    """
    blocks = architecture.get("blocks") or {}
    if part == "mlp":
        if spec.mlp not in (blocks.get("mlp") or {}):
            raise ValueError(f"Architecture has no blocks.mlp.{spec.mlp}")
        return ("blocks", "mlp", spec.mlp)
    if part not in ("pre_attn", "post_attn"):
        raise ValueError(f"Unknown decoder part: {part}")
    axis = "attn"
    group = None
    if spec.sparse:
        group = (blocks.get("sparse_attn") or {}).get(spec.attn)
        if group is not None:
            axis = "sparse_attn"
    if group is None:
        group = (blocks.get("attn") or {}).get(spec.attn)
    if group is None:
        raise ValueError(f"Architecture has no blocks.attn.{spec.attn}")
    return ("blocks", axis, spec.attn, part)


def resolved_text_config(vllm_config):
    """Read this live engine's post-override text config without importing vLLM."""
    model = getattr(vllm_config, "model_config", None)
    hf = getattr(model, "hf_text_config", None)
    if hf is None:
        hf = getattr(model, "hf_config", None)
    if hf is None:
        raise ValueError("Scope-call contract requires the live resolved model config")
    value = hf if isinstance(hf, dict) else hf.to_dict() if hasattr(hf, "to_dict") else vars(hf)
    if not isinstance(value, dict):
        raise ValueError("Resolved text config must serialize to a mapping")
    return value


def expected_scope_calls(architecture, choices, resolved_config):
    """Derive physical call units from selected sites and the live reduced stack.

    Only an explicit ``site-scope-calls`` variant opts in. These are aggregate
    per-forward counts, not a decoder-layer bijection: compensating missing and
    duplicate calls on different layers are indistinguishable. Concrete depth
    is intentionally absent from the shared per-layer acquisition identity.
    """
    from profiler.core.stack import resolve_stack, text_config

    groups = validate_definitions(architecture)
    # Validate all choices even when no selected variant opts into this gate.
    layout_identity(architecture, choices)
    contracted = {name: group for name, group in groups.items()
                  if group["variants"][choices[name]].get("invocation_contract") is not None}
    if not contracted:
        return {}
    if not isinstance(resolved_config, dict):
        raise ValueError("Resolved scope-call config must be a mapping")
    cfg = text_config(resolved_config)
    depth = cfg.get("num_hidden_layers")
    if type(depth) is not int or depth < 1:
        raise ValueError("Scope-call contract requires a positive resolved layer count")
    sparse_config = cfg.get("sparse_attention_config")
    if sparse_config is not None and not isinstance(sparse_config, dict):
        raise ValueError("Malformed resolved sparse-attention config")
    vectors = {"layer_types": cfg.get("layer_types"),
               "moe_layer_freq": cfg.get("moe_layer_freq"),
               "index_topk_pattern": cfg.get("index_topk_pattern"),
               "sparse_attention_freq": (sparse_config or {}).get("sparse_attention_freq")}
    for name, value in vectors.items():
        if isinstance(value, tuple):
            raise ValueError(f"Malformed resolved per-layer vector: {name}")
        if isinstance(value, list) or (name == "index_topk_pattern" and isinstance(value, str)):
            if len(value) < depth:
                raise ValueError(f"Incomplete resolved per-layer vector: {name}")
        elif name in ("layer_types", "index_topk_pattern") and value is not None:
            raise ValueError(f"Malformed resolved per-layer vector: {name}")
    stack = resolve_stack(cfg)
    if len(stack) != depth:
        raise ValueError("Resolved stack length differs from the live layer count")
    catalog = _catalog(architecture)
    expected = {}
    for name, group in contracted.items():
        active = _active(group, choices[name])
        footprint = {tuple(path) for path in group["sites"]}
        counts = Counter()
        for path in sorted(footprint):
            if path[0] == "shared":
                counts.update(layer for layer in resolve_sequence_site(
                    architecture, choices, path, role="shared") if layer in active)
        for index, spec in enumerate(stack):
            for part in ("pre_attn", "post_attn", "mlp"):
                path = spec_block_site(architecture, spec, part)
                if path in footprint:
                    counts.update(layer for layer in resolve_sequence_site(
                        architecture, choices, path, role="target", layer_index=index)
                        if layer in active)
        if set(counts) != active or any(count < 1 for count in counts.values()):
            raise ValueError(f"Live stack does not instantiate the selected scope-call contract: {name}")
        for layer, count in counts.items():
            category, entry = catalog[layer]
            kind = ("callable" if entry.get("callable") is not None else
                    "operator" if entry.get("operator") is not None else "module")
            expected[layer] = dict(calls=count, scope_kind=kind,
                normalization=entry.get("normalization") or ("invocations" if category == "moe" else "parent"))
    return expected


def validate_scope_calls(report, expected, iterations, catalog):
    """Check every measured forward against physical units, never timing divisors."""
    if not expected:
        return
    audit = report.get("canonical_invocations")
    if (not isinstance(audit, dict) or type(audit.get("schema")) is not int
            or audit["schema"] != 1 or type(iterations) is not int or iterations < 1
            or type(audit.get("timed_forwards")) is not int
            or audit["timed_forwards"] != iterations):
        raise ValueError("Missing or invalid canonical invocation audit")
    forwards = audit.get("per_forward")
    if not isinstance(forwards, list) or any(not isinstance(row, dict) for row in forwards):
        raise ValueError("Malformed scope-call forward audit")
    indices = [row.get("forward") for row in forwards]
    if (len(indices) != iterations or any(type(index) is not int for index in indices)
            or sorted(indices) != list(range(iterations))):
        raise ValueError("Incomplete or duplicate scope-call forward indices")
    for forward in forwards:
        measured = forward.get("canonical")
        if not isinstance(measured, dict):
            raise ValueError("Missing per-forward canonical counts")
        for name, spec in expected.items():
            row = measured.get(name)
            binding = catalog.get(name)
            if not isinstance(row, dict) or not isinstance(binding, dict):
                raise ValueError(f"Missing expected canonical scope: {name}")
            count, kind = spec["calls"], spec["scope_kind"]
            if any(type(row.get(key)) is not int for key in
                   ("scope_calls", "positive_cuda_scope_calls", "raw_cuda_activities")):
                raise ValueError(f"Scope-call counts must be integers: {name}")
            for key in ("scope_calls_by_kind", "positive_cuda_scope_calls_by_kind"):
                value = row.get(key)
                if (not isinstance(value, dict) or set(value) != {kind}
                        or type(value[kind]) is not int or value[kind] != count):
                    raise ValueError(f"Physical scope kinds/counts differ: {name}")
            if (row["scope_calls"] != count or row["positive_cuda_scope_calls"] != count
                    or row["raw_cuda_activities"] != 0):
                raise ValueError(f"Physical scope calls differ from selected site occurrences: {name}")
            if (binding.get("normalization", "parent") != spec["normalization"]
                    or row.get("normalization") != spec["normalization"]
                    or type(row.get("occurrences")) is not int
                    or row["occurrences"] != binding.get("occurrences", 1)):
                raise ValueError(f"Scope-call normalization metadata differs: {name}")


def validate_invocation_report(architecture, choices, report, expected, iterations, catalog, regime):
    """Validate a selected layout and its native duration/count evidence together.

    Category/probe captures may leave unrelated model work and runner work
    unbound. Those gaps remain explicit and cannot replace a required positive
    scope; their native duration is conserved. This gate is not a full-model
    coverage proof. Inactive alternatives are allowed and are not zero-filled.
    """
    if not expected:
        return
    if report.get("over_matches") or report.get("overlapping_activities"):
        raise ValueError("Scope-call contract has ambiguous native activity ownership")
    bound, gaps, total = report.get("bound_us_by_layer"), report.get("gaps"), report.get("total_us")
    if not isinstance(bound, dict) or not isinstance(gaps, list):
        raise ValueError("Missing native duration ledger")
    values = list(bound.values())
    for gap in gaps:
        if not isinstance(gap, (list, tuple)) or len(gap) != 3:
            raise ValueError("Malformed native duration gap")
        values.append(gap[1])
    if (not isinstance(total, (int, float)) or isinstance(total, bool) or not math.isfinite(total) or total <= 0
            or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                   or not math.isfinite(value) or value <= 0 for value in values)
            or not math.isclose(sum(values), total, rel_tol=1e-10, abs_tol=1e-8)
            or type(report.get("unique_cuda_activities")) is not int
            or report["unique_cuda_activities"] < 1):
        raise ValueError("Scope-call report does not conserve native CUDA duration")
    validate_runtime_shot(architecture, choices, regime, set(bound))
    validate_scope_calls(report, expected, iterations, catalog)


def selected_runtime_contract(architecture, choices):
    """An explicit variant selects a source adapter, never a model-name guess."""
    layout_identity(architecture, choices)
    contracts = {group['variants'][choices[name]].get('runtime_contract')
                 for name, group in (architecture.get('component_layouts') or {}).items()}
    contracts.discard(None)
    if len(contracts) > 1:
        raise ValueError('Multiple component runtime contracts are not supported')
    return next(iter(contracts), None)


def prepare_invocation_contract(contract, layout_sha256, catalog, vllm_config, *, scope_contexts=None):
    """Validate an optional RPC descriptor before any preparation or warmup."""
    if contract is None:
        return None
    _mapping(contract, {"architecture", "selections", "layout_sha256", "runtime"},
             {"architecture", "selections", "layout_sha256"}, "component layout RPC contract")
    architecture, choices = contract["architecture"], contract["selections"]
    identity = layout_identity(architecture, choices)
    if contract["layout_sha256"] != identity or layout_sha256 != identity:
        raise ValueError("Component-layout RPC identity differs")
    runtime_schema = selected_runtime_contract(architecture, choices)
    runtime = contract.get('runtime')
    if runtime_schema is None and runtime is not None:
        raise ValueError('Unexpected component runtime descriptor')
    if runtime_schema is not None:
        _mapping(runtime, {'descriptor', 'runtime_sha256'}, {'descriptor', 'runtime_sha256'}, 'runtime RPC')
        from profiler.core.hooks.runtime_admission import validate_descriptor
        descriptor = validate_descriptor(runtime['descriptor'])
        if descriptor['schema'] != runtime_schema:
            raise ValueError('Runtime descriptor differs from selected declaration')
        digest = runtime['runtime_sha256']
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in '0123456789abcdef' for char in digest)):
            raise ValueError('Malformed component runtime identity')
    # Reuse the one authoritative RPC serializer, including occurrences and
    # default normalization; a same-name altered selector is not a contract.
    from profiler.core.config import Architecture
    from profiler.core.categories import _entry_dict
    from profiler.core.hooks.capture_scope_contract import contexts_from_architecture, normalize_contexts
    typed = Architecture.model_validate(architecture)
    if normalize_contexts(scope_contexts, catalog) != normalize_contexts(contexts_from_architecture(typed), catalog):
        raise ValueError("Capture scope contexts differ from the component-layout declarations")
    whole = {}
    for category in type(typed.catalog).model_fields:
        whole.update(_entry_dict(getattr(typed.catalog, category), typed))
    probes = probe_catalog(architecture, whole)
    if any(catalog.get(name) != entry for name, entry in probes.items()):
        raise ValueError("Capture catalog differs from the component-layout probes")
    expected = expected_scope_calls(architecture, choices, resolved_text_config(vllm_config))
    return architecture, choices, expected


def expand_sequence(architecture, choices, sequence):
    """Resolve one sequence without mutating the shared architecture or TP cache."""
    groups = architecture.get("component_layouts") or {}
    choices = choices if choices is not None else {n: g["historical"] for n, g in groups.items()}
    if set(choices) != set(groups):
        raise ValueError("Incomplete component-layout selection")
    replacements = {}
    for name, group in groups.items():
        if group.get("sites") is not None and choices[name] != group["historical"]:
            raise ValueError("An active site-aware layout requires resolve_sequence_site")
        replacements.update(group["variants"][choices[name]].get("replace") or {})
    return [item for layer in sequence for item in replacements.get(layer, [layer])]


def resolve_sequence_site(architecture, choices, path, *, role, layer_index=None):
    """Resolve a validated existing site with explicit runtime role and depth.

    No model-name inference, mutation or cache lives here. Consumers cache the
    returned immutable sequence and include it in repeated-block identities.
    """
    path = _site_path(path, "runtime sequence")
    sources = sequence_sites(architecture)
    if path not in sources:
        raise ValueError("Unknown architecture sequence site")
    if role not in ("target", "shared", "mtp"):
        raise ValueError("Unknown dynamic sequence role")
    if role == "target" and (path[0] != "blocks" or type(layer_index) is not int or layer_index < 0):
        raise ValueError("Target decoder sites require an absolute nonnegative layer index")
    if role == "shared" and (path[0] != "shared" or layer_index is not None):
        raise ValueError("Shared sites cannot impersonate a decoder layer")
    if role == "mtp" and (path[0] not in ("blocks", "mtp")
                           or (layer_index is not None and (type(layer_index) is not int or layer_index < 0))):
        raise ValueError("Invalid MTP sequence site or layer index")
    groups = architecture.get("component_layouts") or {}
    choices = choices if choices is not None else {name: group["historical"] for name, group in groups.items()}
    if set(choices) != set(groups):
        raise ValueError("Incomplete component-layout selection")
    replacements, rules = {}, []
    for name, group in groups.items():
        if choices[name] not in group["variants"]:
            raise ValueError("Unknown component-layout selection")
        selected = group["variants"][choices[name]]
        if group.get("sites") is not None:
            if role == "mtp" and choices[name] != group["historical"]:
                raise NotImplementedError("Active site-aware component layouts do not yet model MTP")
            if path not in {_site_path(site, name) for site in group["sites"]}:
                continue
        replacements.update(selected.get("replace") or {})
        rules.extend(selected.get("layer0") or [])
    result = _rewrite(sources[path], replacements)
    if role == "target" and layer_index == 0:
        for rule in rules:
            if path != tuple(rule["path"]):
                continue
            expected = rule["expected_prefix"]
            if result[:len(expected)] != expected:
                raise ValueError("Declared layer0 prefix no longer matches")
            result = list(rule["replacement_prefix"]) + result[len(expected):]
    return tuple(result)


def filter_catalog_slice(architecture, choices, catalog_slice):
    """Remove alternative-only components; preserve all unrelated bindings."""
    excluded = set()
    for name, group in (architecture.get("component_layouts") or {}).items():
        excluded |= _members(group) - _active(group, choices[name])
    return {name: entry for name, entry in catalog_slice.items() if name not in excluded}


def selected_catalog_for_tp(tp_root, architecture, category, entries=None):
    """Project one category from the sole validated per-TP layout record.

    ``entries`` may retain typed host entries; selection uses canonical names
    only. Keep the full architecture for capture contexts and site validation.
    Unmarked historical tables keep the historical projection without writes.
    """
    catalog = architecture.get("catalog") or {}
    if category not in catalog:
        raise ValueError(f"Unknown catalog category: {category}")
    declared = catalog[category] or {}
    source = declared if entries is None else entries
    if not isinstance(source, dict) or set(source) != set(declared):
        raise ValueError("Category projection requires its complete declared entries")
    if not architecture.get("component_layouts"):
        return None, dict(source)
    choices = read_for_trace(tp_root, architecture)
    return choices, filter_catalog_slice(architecture, choices, source)


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
    from .bundle_metadata import read_bundle
    return read_bundle(root)


def _publish(root, metadata, tp):
    from .bundle_metadata import update_bundle
    value = metadata['component_layouts']['per_tp'][tp]
    def publish(current):
        block = current.setdefault('component_layouts', dict(schema=SCHEMA, per_tp={}))
        if block.get('schema') != SCHEMA:
            raise ValueError('Unknown component layout metadata schema')
        block['per_tp'].pop(str(tp), None)
        block['per_tp'][tp] = value
    update_bundle(root, publish)


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


def _required_categories(architecture, choices):
    """Every CSV the selected contract must certify, not just rewritten nodes.

    The native duration contract consumes the head too. Native MoE components
    have their own deployment contract and complete-table publication; they
    must not leave this ordinary-category acquisition waiting for moe.csv.
    Historical layouts keep their previous coupled categories and identities.
    """
    groups = architecture.get("component_layouts") or {}
    required = set().union(*(set(group["categories"]) for group in groups.values()))
    if selected_runtime_contract(architecture, choices) is not None:
        # validate_definitions currently admits only native-dsa-eager-tail.
        required |= {"dense", "attention", "per_sequence"}
        if any(not architecture["catalog"].get(name) for name in required):
            raise ValueError("The selected runtime contract lacks a required category")
    return required


def _validate_record(architecture, value):
    _mapping(value, {"layout_sha256", "selections", "acquisition", "categories", "runtime"},
             {"layout_sha256", "selections", "acquisition", "categories"}, "layout record")
    if value["layout_sha256"] != layout_identity(architecture, value["selections"]):
        raise ValueError("Stale/invalid component-layout acquisition contract; use a fresh output bundle")
    _measurement(value["acquisition"])
    required = _required_categories(architecture, value["selections"])
    # Existing native whole-block acquisitions remain readable under their
    # original, fully checked publication record. New records omit that table.
    if (selected_runtime_contract(architecture, value['selections']) is not None
            and 'moe' in value['categories']):
        required.add('moe')
    if set(value["categories"]) != required:
        raise ValueError("Incomplete component-layout acquisition contract; use a fresh output bundle")
    runtime_schema = selected_runtime_contract(architecture, value['selections'])
    runtime = value.get('runtime')
    if runtime_schema is None:
        if runtime is not None:
            raise ValueError('Historical layout cannot claim a new runtime admission')
    else:
        _mapping(runtime, {'schema', 'runtime_sha256', 'invariant_facts', 'category_engines'},
                 {'schema', 'runtime_sha256', 'invariant_facts', 'category_engines'}, 'runtime metadata')
        if runtime['schema'] != runtime_schema or not isinstance(runtime['category_engines'], dict):
            raise ValueError('Invalid component runtime metadata schema')
        from profiler.core.hooks.runtime_admission import admit_runtime
        invariant = runtime['invariant_facts']
        if not isinstance(invariant, dict) or 'descriptor' not in invariant:
            raise ValueError('Missing admitted source configuration')
        if not runtime['category_engines']:
            raise ValueError('Runtime admission has no live category evidence')
        for category, facts in runtime['category_engines'].items():
            if category not in architecture['catalog']:
                raise ValueError('Unknown runtime evidence category')
            admitted = admit_runtime(invariant['descriptor'], facts)
            if (admitted['invariant_facts'] != invariant
                    or admitted['runtime_sha256'] != runtime['runtime_sha256']):
                raise ValueError('Stale or mixed runtime admission evidence')
    for category, record in value["categories"].items():
        _mapping(record, {"state", "measurement"}, {"state"}, category)
        if record["state"] not in ("pending", "writing", "complete"):
            raise ValueError(f"Invalid acquisition state for {category}")
        if record["state"] != "pending":
            _measurement(record.get("measurement"))
            if runtime is not None and category not in runtime['category_engines']:
                raise ValueError('Started category lacks runtime admission evidence')
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
        runtime = record.get('runtime')
        if runtime is not None and runtime['invariant_facts']['descriptor']['logical_tp'] != int(tp):
            raise ValueError('Runtime admission logical TP differs from its metadata degree')
    return dict(schema=SCHEMA, per_tp=normalized)


def begin_category(tp_root, architecture, choices, category, measurement,
                   acquisition, *, replace=False, runtime=None):
    """Durably invalidate before replacement; resume only certified whole shots.

    A changed acquisition (versions, source or repetitions) resets *all* coupled
    categories under explicit force. Refreshing one category with the same
    acquisition preserves its independently validated siblings. A different
    representation always requires a fresh bundle.
    """
    root = Path(tp_root)
    groups = validate_definitions(architecture)
    categories = _required_categories(architecture, choices)
    if category not in categories and runtime is None:
        return None
    if category not in architecture['catalog']:
        raise ValueError('Unknown component acquisition category')
    runtime_schema = selected_runtime_contract(architecture, choices)
    if (runtime is None) != (runtime_schema is None):
        raise ValueError('Selected runtime admission is missing or unexpected')
    if runtime is not None:
        _mapping(runtime, {'schema', 'runtime_sha256', 'invariant_facts', 'engine_facts'},
                 {'schema', 'runtime_sha256', 'invariant_facts', 'engine_facts'}, 'admitted runtime')
        from profiler.core.hooks.runtime_admission import admit_runtime
        checked = admit_runtime(runtime['invariant_facts']['descriptor'], runtime['engine_facts'])
        if checked != runtime or runtime['schema'] != runtime_schema:
            raise ValueError('Invalid live runtime admission result')
    _measurement(measurement)
    _measurement(acquisition)
    metadata = _read_meta(root.parent)
    block = metadata_records(root.parent, architecture) or dict(schema=SCHEMA, per_tp={})
    metadata["component_layouts"] = block
    tp = int(root.name.removeprefix("tp"))
    if runtime is not None and runtime['invariant_facts']['descriptor']['logical_tp'] != tp:
        raise ValueError('Runtime admission logical TP differs from its output directory')
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
            if runtime is not None:
                value.pop('runtime', None)
        if runtime is not None and value.get('runtime', {}).get('runtime_sha256') not in (
                None, runtime['runtime_sha256']):
            raise ValueError('Different runtime facts cannot share a coupled acquisition identity')
    else:
        if not replace and any((root / f"{name}.csv").exists() for name in categories):
            raise ValueError("Historical category CSVs need coordinated --force refresh or a fresh output bundle")
        value = dict(layout_sha256=identity, selections=dict(choices), acquisition=dict(acquisition),
                     categories={name: {"state": "pending"} for name in sorted(categories)})
        block["per_tp"][tp] = value
    if runtime is not None:
        if 'runtime' not in value:
            value['runtime'] = {key: item for key, item in runtime.items() if key != 'engine_facts'}
            value['runtime']['category_engines'] = {}
        value['runtime']['category_engines'][category] = runtime['engine_facts']
    if category not in categories:
        for sibling, state in value['categories'].items():
            if state['state'] != 'pending':
                _category_content(root, architecture, value, sibling,
                                  allow_partial=state['state'] == 'writing')
        _validate_record(architecture, value)
        _publish(root.parent, metadata, tp)
        return value
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
        _publish(root.parent, metadata, tp)
        # TP-stable rows belong to their source degree, not this category's
        # acquisition. A reset must not erase or recertify those replicas.
        owned = set().union(*(_members(group) for group in groups.values()))
        replicated = {name for name, entry in architecture["catalog"][category].items()
                      if tp > 1 and entry.get("tp_stable") and name not in owned}
        _clear_category(root / f"{category}.csv", preserve_layers=replicated)
    value["categories"][category] = dict(state="writing", measurement=dict(measurement))
    _validate_record(architecture, value)
    _publish(root.parent, metadata, tp)
    return value


def _attention_row_key(row, path):
    """Current attention writer coordinates, with its exact Python float key.

    Keep CSV bytes untouched. Numeric key equality must match DedupSink's
    resume keys, not distinguish different textual spellings of one number or
    round nearby representable prefill coordinates onto a coarser grid.
    """
    try:
        prefill = int(row["prefill_chunk"])
        key = float(row["prefill_key"])
        decode = int(row["n_decode"])
        history = int(row["kv_decode"])
        q = int(row["decode_q_len"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"Invalid attention coordinates in {path}") from error
    if (prefill < 0 or decode < 0 or history < 0 or q < 1
            or not math.isfinite(key) or key < 0 or not prefill + decode
            or (prefill == 0) != (key == 0) or (decode == 0 and history != 0)):
        raise ValueError(f"Invalid attention coordinates in {path}")
    regime = "mixed" if prefill and decode else "prefill" if prefill else "decode"
    return regime, (prefill, key, decode, history, q)


def _category_content(root, architecture, record, category, *, allow_partial=False):
    if record.get("runtime") is not None:
        return _runtime_category_content(root, architecture, record, category,
                                         allow_partial=allow_partial)
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
        stamps = set(MEASUREMENT_FIELDS).intersection(fields)
        if (len(fields) != len(set(fields)) or not {"layer", "time_us"} <= set(fields)
                or (stamps and stamps != set(MEASUREMENT_FIELDS))):
            raise ValueError(f"Invalid acquired category header: {path}")
        if category == "attention" and set(fields) - stamps != {
                "layer", "time_us", "prefill_chunk", "prefill_key",
                "n_decode", "kv_decode", "decode_q_len"}:
            raise ValueError(f"Acquired attention rows require the current coordinate schema: {path}")
        rows = list(reader)
    catalog = architecture["catalog"].get(category) or {}
    # Validate every stored row before excluding source-owned replicas from
    # this TP's acquisition coverage. Their different TP1 identity is allowed;
    # malformed rows, invalid times and duplicate coordinates are not.
    members = set().union(*(_members(group) for group in architecture["component_layouts"].values()))
    tp = int(root.name.removeprefix("tp"))
    measurement = record["categories"][category]["measurement"]
    if not stamps:
        from .writer import category_measurement
        saved = category_measurement(root, category)
        if saved is None or saved['measurement'] != measurement:
            raise ValueError(f"Category metadata does not match its selected layout: {path}")
    shots = {}
    seen = set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed acquired category row: {path}")
        if row["layer"] not in catalog:
            raise ValueError(f"Unknown canonical component in {path}: {row['layer']}")
        if stamps:
            _measurement({field: row[field] for field in MEASUREMENT_FIELDS})
        latency = float(row["time_us"])
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"Invalid acquired latency in {path}")
        coords = tuple((key, row[key]) for key in fields if key not in
                       {"layer", "time_us", *MEASUREMENT_FIELDS})
        if category == "attention":
            regime, coords = _attention_row_key(row, path)
        elif category == "linear_attention":
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
        if stamps and any(row[field] != value for field, value in measurement.items()):
            raise ValueError(f"Mixed measurement identity in {path}")
        present = shots.setdefault(key, set())
        present.add(row["layer"])
    if not shots:
        if allow_partial:
            return
        raise ValueError(f"No nonstable acquisition rows in {path}")
    for (regime, _), present in shots.items():
        validate_shot_components(architecture, record["selections"], category, regime, present)
    if (not allow_partial and category in {"attention", "linear_attention"}
            and {key[0] for key in shots} != set(REGIMES)):
        raise ValueError(f"Acquired {category} layout lacks an entire regime: {path}")


def _runtime_category_content(root, architecture, record, category, *, allow_partial):
    """Certify current native rows without promoting or relabelling old data.

    Dense/head shots retain a row for every selected canonical component;
    whole-block MoE has one implicit component and its own EP/token/expert key.
    Completion covers saved coordinates, not an unmeasured grid or accuracy.
    """
    coordinates = {
        "dense": {"layer", "tokens"},
        "per_sequence": {"layer", "sequences"},
        "attention": {"layer", "prefill_chunk", "prefill_key", "n_decode", "kv_decode", "decode_q_len"},
        "moe": {"ep", "tokens", "activated_experts"},
    }
    if category not in coordinates:
        raise ValueError(f"No admitted native CSV contract for {category}")
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
        stamps = set(MEASUREMENT_FIELDS).intersection(fields)
        if (len(fields) != len(set(fields)) or (stamps and stamps != set(MEASUREMENT_FIELDS))
                or set(fields) - stamps != coordinates[category] | {"time_us"}):
            raise ValueError(f"Invalid native {category} header: {path}")
        rows = list(reader)
    expected = set(filter_catalog_slice(architecture, record["selections"],
                                       architecture["catalog"][category]))
    if not expected or (category == "moe" and len(expected) != 1):
        raise ValueError(f"Invalid native component set for {category}")
    measurement = record["categories"][category]["measurement"]
    if not stamps:
        from .writer import category_measurement
        saved = category_measurement(root, category)
        if saved is None or saved['measurement'] != measurement:
            raise ValueError(f"Category metadata does not match its native layout: {path}")
    descriptor = record["runtime"]["invariant_facts"]["descriptor"]
    source = descriptor["source_config"]
    shots, seen = {}, set()

    def positive(row, name):
        value = row[name]
        if not value.isascii() or not value.isdigit() or int(value) < 1:
            raise ValueError(f"Invalid positive {name} coordinate in {path}")
        return int(value)

    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed native category row: {path}")
        if stamps:
            _measurement({field: row[field] for field in MEASUREMENT_FIELDS})
        if stamps and any(row[field] != value for field, value in measurement.items()):
            raise ValueError(f"Mixed measurement identity in {path}")
        try:
            latency = float(row["time_us"])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Invalid acquired latency in {path}") from error
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"Invalid acquired latency in {path}")
        name = next(iter(expected)) if category == "moe" else row["layer"]
        if name not in expected:
            raise ValueError(f"Unknown/inactive native component in {path}: {name}")
        if category == "attention":
            regime, key = _attention_row_key(row, path)
            if key[-1] != 1:
                raise ValueError(f"Native attention requires decode_q_len=1: {path}")
        elif category == "moe":
            ep, tokens, active = (positive(row, field) for field in
                                  ("ep", "tokens", "activated_experts"))
            topk, experts = source["num_experts_per_tok"], source["n_routed_experts"]
            if (ep != descriptor["logical_ep"] or active < topk
                    or active > min(experts, tokens * topk)):
                raise ValueError(f"MoE point is outside the admitted EP/expert geometry: {path}")
            regime, key = "moe", (ep, tokens, active)
        else:
            field = "tokens" if category == "dense" else "sequences"
            count = positive(row, field)
            regime, key = category, (count,)
        row_key = (name, regime, key)
        if row_key in seen:
            raise ValueError(f"Duplicate acquired component row in {path}")
        seen.add(row_key)
        shots.setdefault((regime, key), set()).add(name)
    if not shots and not allow_partial:
        raise ValueError(f"No acquired rows in {path}")
    if any(names != expected for names in shots.values()):
        raise ValueError(f"Incomplete native {category} shot in {path}")
    if (not allow_partial and category == "attention"
            and {regime for regime, _ in shots} != set(REGIMES)):
        raise ValueError(f"Acquired attention layout lacks an entire regime: {path}")


def complete_category(tp_root, architecture, category):
    root = Path(tp_root)
    metadata = _read_meta(root.parent)
    block = metadata_records(root.parent, architecture)
    if block is None:
        raise ValueError("Missing component-layout acquisition record")
    metadata["component_layouts"] = block
    tp = int(root.name.removeprefix("tp"))
    value = block["per_tp"][tp]
    if category not in value['categories']:
        return value
    record = value["categories"][category]
    if record["state"] not in ("writing", "complete"):
        raise ValueError(f"Category {category} was never started")
    _category_content(root, architecture, value, category)
    record["state"] = "complete"
    _publish(root.parent, metadata, tp)
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
