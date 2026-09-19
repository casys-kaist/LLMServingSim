"""Offline relative-latency calibration and bounded-cost bucket lookup.

Uniform attention references must come from the same table and lookup used
at execution. Actual latencies are profiling measurements, never benchmark
request latencies. N anchors are supported independently within each kernel,
query-length, prefill and lever partition. Prefill partitions scale with the
measured envelope; these summaries are not sufficient distribution descriptors.
"""

from collections import defaultdict
import csv
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import tempfile


SCHEMA = "runtime-skew-calibration-v1"
LEV_BINS = (0.0, 0.25, 0.75, 1.5, 3.0, 1_000_000_000.0)
LEV_LABELS = ("lev0", "lev1", "lev2", "lev3", "lev4")
MIN_ROWS = 20
ALPHA_CLIP = (-0.2, 1.0)


def checksum(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def lookup_fingerprint():
    """Identify the actual endpoint implementation, including shape reduction."""
    from serving.core import trace_generator as tg
    from .attention_shape import prefill_key, partition_q1_requests
    functions = (tg._lookup_attention, tg._attention_q_slice, tg._attn_slice_lookup,
                 tg._axis_bracket, tg._build_attention_table,
                 tg._build_attention_tables_by_layer, tg._build_attention_tables_by_q,
                 tg._read_category_csv, tg._build_batch_ctx, tg._key_saturates,
                 prefill_key, partition_q1_requests,
                 reference_rows, measurement_shape)
    return hashlib.sha256("\n".join(inspect.getsource(f) for f in functions).encode()).hexdigest()


def measurement_shape(raw):
    """Recover complete query/history geometry and a stable measurement key."""
    from .attention_shape import partition_q1_requests
    def integer(value):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError("Expected an integer shape coordinate")
        return int(number)
    q = integer(raw.get("decode_q_len") or 1)
    if q < 1:
        raise ValueError("Decode query length must be positive")
    encoded = raw.get("requests_json")
    if isinstance(encoded, str) and encoded:
        requests = json.loads(encoded)
    else:
        if raw.get("kv_big") in (None, ""):
            raw = dict(raw, kv_big=round(float(raw["kvs"])*float(raw["skew"])))
        pc, kp, n, nb, big, small = (integer(raw[k]) for k in
            ("pc", "kp", "n", "nb", "kv_big", "kvs"))
        if q != 1 or not 0 < nb < n or pc < 0:
            raise ValueError("Invalid legacy bimodal skew shape")
        requests = ([[pc, kp]] if pc else []) + [[1, big]]*nb + [[1, small]]*(n-nb)
    if (not isinstance(requests, list) or not requests or
            any(not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(type(v) is not int for v in pair) or pair[0] < 1 or pair[1] < 0
                for pair in requests)):
        raise ValueError("Expected positive queries and nonnegative integer histories")
    if q == 1:
        pf, dk = partition_q1_requests(requests)
        ordered = sorted(requests, key=lambda pair: pair[0])
        key_payload = [q, ordered]
    else:
        if raw.get("n_prefill") in (None, ""):
            raise ValueError("Multi-query skew rows require an explicit prefill boundary")
        count = integer(raw["n_prefill"])
        if not 0 <= count <= len(requests):
            raise ValueError("Invalid query roles")
        pf, decode = requests[:count], requests[count:]
        if any(query != q for query, _ in decode):
            raise ValueError("Decode query lengths disagree with the declared roles")
        dk = [history for _, history in decode]
        key_payload = [q, count, requests]
    case = hashlib.sha256(json.dumps(key_payload, separators=(",", ":")).encode()).hexdigest()
    return requests, pf, dk, q, case


def reference_rows(skew_csv, attention_csv, tp, architecture, key_saturation):
    """Recompute lookup references from raw shapes without changing raw times.

    Legacy bimodal rows are exactly reconstructible. General q=1 rows carry
    requests_json. Multi-query rows must explicitly carry n_prefill and
    decode_q_len; their role boundary cannot be inferred from query counts.
    """
    from serving.core import trace_generator as tg
    from .attention_shape import prefill_key
    frame = tg._read_category_csv(str(attention_csv), None)
    db = dict(tables={tp: {"attention_by_layer": tg._build_attention_tables_by_layer(frame)}})
    entries = architecture["catalog"].get("attention") or {}
    rows = []
    with Path(skew_csv).open(newline="") as stream:
        for raw in csv.DictReader(stream):
            layer = raw.get("layer") or "attention"
            if layer not in entries:
                raise ValueError(f"Skew row names unbound attention kernel {layer!r}")
            requests, pf, dk, q, case = measurement_shape(raw)
            if len(dk) < 2 or min(dk) == max(dk):
                continue
            by_q = db["tables"][tp]["attention_by_layer"].get(layer, {})
            if q not in by_q:
                raise ValueError(f"No exact attention reference for {layer!r}, q={q}, TP={tp}")
            saturates = bool(entries[layer].get("key_saturates"))
            cap = key_saturation if saturates else None
            key = prefill_key(pf, cap=cap)
            n, pc = len(dk), sum(query for query, _ in pf)
            mean = tg._lookup_attention(db, tp, pc, key, n, sum(dk)//n, layer, q)/1000.0
            maximum = tg._lookup_attention(db, tp, pc, key, n, max(dk), layer, q)/1000.0
            rows.append(dict(case_id=case, layer=layer, decode_q_len=q, n=n, pc=pc,
                reference_mean_us=mean, reference_max_us=maximum,
                measured_us=float(raw["t_skew_us"]), allow_negative_gap=saturates))
    return rows


def fit_csv(skew_csv, attention_csv, identity, architecture, key_saturation=None):
    rows = reference_rows(skew_csv, attention_csv, identity["tp"], architecture, key_saturation)
    reference = dict(attention_sha256=checksum(attention_csv), lookup_sha256=lookup_fingerprint(),
                     skew_sha256=checksum(skew_csv), key_saturation=key_saturation,
                     saturation_by_layer={k: bool(v.get("key_saturates"))
                         for k, v in (architecture["catalog"].get("attention") or {}).items()})
    return fit(rows, identity, reference)


def fit_bundle(variant_root, identity, model_config, tp_degrees=None):
    """Compile measured TP folders automatically, without starting an engine."""
    from serving.core.utils import _load_architecture
    from .stack import probe_key_saturation, text_config
    root = Path(variant_root)
    config = text_config(model_config)
    architecture = _load_architecture(config["model_type"])
    cap = probe_key_saturation(config)
    available = sorted(int(p.name[2:]) for p in root.glob("tp*")
                       if p.is_dir() and p.name[2:].isdigit() and (p/"skew.csv").exists())
    wanted = available if tp_degrees is None else sorted(set(tp_degrees))
    per_tp = {}
    for tp in wanted:
        folder = root / f"tp{tp}"
        status = folder / "skew.meta.yaml"
        if status.exists():
            import yaml
            if not (yaml.safe_load(status.read_text()) or {}).get("completed"):
                raise ValueError(f"Incomplete skew acquisition at {folder}; resume profiling first")
        if not (folder / "skew.csv").exists():
            raise FileNotFoundError(f"Missing skew measurements at {folder}")
        fit_identity = dict(identity, tp=tp)
        table = fit_csv(folder/"skew.csv", folder/"attention.csv", fit_identity, architecture, cap)
        output = folder / "skew_fit.csv"
        write_table(output, table)
        summary = {k: v for k, v in table.items() if k != "slices"}
        summary.update(bucket_table=f"tp{tp}/skew_fit.csv", bucket_table_sha256=checksum(output))
        per_tp[tp] = summary
    return dict(enabled=bool(per_tp), per_tp=per_tp)


def rebuild_bundle(variant_root, model_config, tp_degrees=None):
    """CPU-only refresh; preserve all unrelated metadata and profile files."""
    import yaml
    root = Path(variant_root)
    path = root / "meta.yaml"
    metadata = yaml.safe_load(path.read_text())
    identity = {key: metadata[key] for key in ("hardware", "model", "variant")}
    block = dict(metadata.get("skew_fit") or {})
    previous = dict(block.get("per_tp") or {})
    fitted = fit_bundle(root, identity, model_config, tp_degrees)
    if not fitted["per_tp"]:
        raise FileNotFoundError(f"No raw skew measurements to rebuild at {root}")
    for tp, entry in fitted["per_tp"].items():
        previous.pop(str(tp), None)
        previous[tp] = entry
    block.update(enabled=bool(previous), per_tp=previous)
    metadata["skew_fit"] = block
    atomic_yaml(path, metadata)
    return fitted


def atomic_yaml(path, value, **kwargs):
    """Publish complete metadata without truncating the previous checkpoint."""
    import yaml
    path = Path(path)
    previous = path.stat() if path.exists() else None
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+".",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("w") as stream:
            yaml.dump(value, stream, sort_keys=False, **kwargs)
            stream.flush()
            os.fsync(stream.fileno())
        if previous is not None and os.geteuid() == 0:
            os.chown(temporary, previous.st_uid, previous.st_gid)
        temporary.chmod(previous.st_mode & 0o777 if previous else 0o644)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _label(bins, labels, value):
    return next((label for upper, label in zip(bins[1:], labels)
                 if value <= upper), labels[-1])


def _kernel(layer, q):
    return f"{layer}|q={q}"


def _partition(layer, q, pc, lev, axes):
    return "|".join((_kernel(layer, q),
                     _label(axes["pc_bins"], axes["pc_labels"], pc),
                     _label(axes["lev_bins"], axes["lev_labels"], lev)))


def weighted_median(pairs):
    """A minimizer of sum(weight * abs(value - estimate))."""
    values = sorted(pairs)
    if not values or any(not math.isfinite(v) or not math.isfinite(w) or w <= 0
                         for v, w in values):
        raise ValueError("Finite values and positive finite weights required")
    half = math.fsum(w for _, w in values) / 2
    cumulative = 0.0
    for index, (value, weight) in enumerate(values):
        cumulative += weight
        if cumulative > half:
            return value
        if cumulative == half:
            return (value + values[min(index + 1, len(values) - 1)][0]) / 2
    return values[-1][0]


def _anchor_index(anchors, n):
    # Squared integer midpoints make the lower-anchor tie exact, including
    # N=20 between anchors 16 and 25, without floating log distances.
    lo, hi = 0, len(anchors) - 1
    square = n * n
    while lo < hi:
        mid = (lo + hi) // 2
        if square <= anchors[mid] * anchors[mid + 1]:
            hi = mid
        else:
            lo = mid + 1
    return lo


def fit(rows, identity, reference, *, min_rows=MIN_ROWS):
    """Fit one bundle/TP with one observation per ordered measurement case.

    Each row names case_id, layer, decode_q_len, n, pc, reference_mean_us,
    reference_max_us and measured_us. Duplicate identities are rejected;
    callers must summarize repetitions before invoking the estimator.
    """
    if (set(identity) != {"hardware", "model", "variant", "tp"}
            or any(not identity[k] for k in identity)
            or type(identity["tp"]) is not int or identity["tp"] < 1):
        raise ValueError("Explicit hardware/model/variant/TP identity required")
    if type(min_rows) is not int or min_rows < 1:
        raise ValueError("A positive integer support floor is required")
    if not reference or not reference.get("attention_sha256") or not reference.get("lookup_sha256"):
        raise ValueError("Attention data and lookup fingerprints are required")
    rows = list(rows)
    pc_cap = max((r["pc"] for r in rows), default=0)
    pc_edges = sorted({1, max(1, pc_cap // 8), max(1, pc_cap // 2)})
    # Token coordinates scale with the measured envelope; leverage is already
    # dimensionless. Preserve a separate zero/tiny-prefill partition.
    axes = dict(pc_bins=[-1, *pc_edges, max(1_000_000_000, pc_cap + 1)],
                pc_labels=["pc0"] + [f"pc{i}" for i in range(1, len(pc_edges) + 1)],
                lev_bins=list(LEV_BINS), lev_labels=list(LEV_LABELS))
    seen, dropped, partitions, kernels = set(), defaultdict(int), defaultdict(list), defaultdict(list)
    for source in rows:
        r = dict(source)
        n, pc, q = r["n"], r["pc"], r["decode_q_len"]
        if any(type(v) is not int for v in (n, pc, q)) or n < 2 or pc < 0 or q < 1:
            raise ValueError("Invalid batch coordinates")
        layer = r["layer"]
        if not isinstance(layer, str) or not layer or "|" in layer:
            raise ValueError("Invalid attention kernel name")
        key = (layer, q, r["case_id"])
        if not r["case_id"] or key in seen:
            raise ValueError("Duplicate or absent measured case identity")
        seen.add(key)
        mean, maximum, actual = (float(r[k]) for k in
            ("reference_mean_us", "reference_max_us", "measured_us"))
        if any(not math.isfinite(v) or v <= 0 for v in (mean, maximum, actual)):
            raise ValueError("All reference and measured times must be positive and finite")
        gap, delta = maximum - mean, actual - mean
        if gap == 0 or (gap < 0 and not r.get("allow_negative_gap", False)):
            dropped["unusable_reference_gap"] += 1
            continue
        alpha, weight = delta / gap, abs(gap) / actual
        if not math.isfinite(alpha) or not math.isfinite(weight) or weight <= 0:
            dropped["unusable_reference_gap"] += 1
            continue
        row = dict(n=n, gap=gap, delta=delta, alpha=alpha, weight=weight)
        partitions[_partition(layer, q, pc, gap / mean, axes)].append(row)
        kernels[_kernel(layer, q)].append(row)
    slices = {}
    for name, selected in sorted(partitions.items()):
        by_n = defaultdict(list)
        for row in selected:
            by_n[row["n"]].append(row)
        anchors = sorted(n for n, values in by_n.items() if len(values) >= min_rows)
        pooled = [[] for _ in anchors]
        for row in selected:
            if anchors:
                pooled[_anchor_index(anchors, row["n"])].append((row["alpha"], row["weight"]))
        slices[name] = dict(anchors=anchors,
            alpha=[round(max(ALPHA_CLIP[0], min(ALPHA_CLIP[1], weighted_median(values))), 4)
                   for values in pooled],
            direct_rows=[len(by_n[n]) for n in anchors], pooled_rows=[len(v) for v in pooled])
    defaults, ranges = {}, {}
    for name, selected in sorted(kernels.items()):
        denominator = math.fsum(r["gap"]**2 for r in selected)
        # A same-kernel/query pooled least-squares estimate supplies cells
        # with insufficient local support. It is never borrowed across slices.
        defaults[name] = round(math.fsum(r["gap"]*r["delta"] for r in selected) / denominator, 4)
        ranges[name] = [min(r["n"] for r in selected), max(r["n"] for r in selected)]
    return dict(schema=SCHEMA, identity=dict(identity), reference=dict(reference),
        estimator="supported_n_relative_latency_l1", min_rows=min_rows,
        n_samples=sum(map(len, kernels.values())), dropped_rows=dict(dropped),
        axes=axes, alpha_clip=list(ALPHA_CLIP), alpha_default_by_kernel=defaults,
        n_range_by_kernel=ranges, slices=slices)


def lookup(table, n, pc, lev, layer="attention", decode_q_len=1):
    """Select one compiled bucket; never fit, interpolate neighbors or read files."""
    kernel = _kernel(layer, decode_q_len)
    bounds = table["n_range_by_kernel"].get(kernel)
    if bounds is None:
        return 0.0  # No borrowing across kernels or query lengths.
    fallback = table["alpha_default_by_kernel"].get(kernel, 0.0)
    if not bounds[0] <= n <= bounds[1]:
        return fallback
    part = table["slices"].get(_partition(layer, decode_q_len, pc, lev, table["axes"]))
    if not part or not part["anchors"]:
        return fallback
    return part["alpha"][_anchor_index(part["anchors"], n)]


def write_table(path, table):
    """Write compact fitted cells, leaving axes, defaults and identity in meta."""
    rows = []
    for name, part in sorted(table["slices"].items()):
        layer, q, pc, lev = name.split("|")
        for index, anchor in enumerate(part["anchors"]):
            rows.append(dict(layer=layer, decode_q_len=int(q[2:]), pc_label=pc,
                lev_label=lev, n_anchor=anchor, alpha=part["alpha"][index],
                direct_rows=part["direct_rows"][index], pooled_rows=part["pooled_rows"][index]))
    fields = ("layer", "decode_q_len", "pc_label", "lev_label", "n_anchor",
              "alpha", "direct_rows", "pooled_rows")
    path = Path(path)
    previous = path.stat() if path.exists() else None
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+".",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        if previous is not None and os.geteuid() == 0:
            os.chown(temporary, previous.st_uid, previous.st_gid)
        temporary.chmod(previous.st_mode & 0o777 if previous else 0o644)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_table(path, summary):
    """Validate persisted cells once, outside the runtime lookup loop."""
    if summary.get("schema") != SCHEMA:
        raise ValueError("Unknown skew calibration schema")
    table = {k: v for k, v in summary.items() if k != "calibration"}
    table["slices"] = {}
    for axis in ("pc", "lev"):
        bins, labels = table["axes"][axis+"_bins"], table["axes"][axis+"_labels"]
        if (len(bins) != len(labels)+1 or not labels or len(set(labels)) != len(labels)
                or any(not isinstance(label, str) or not label or "|" in label for label in labels)
                or any(not math.isfinite(v) for v in bins)
                or any(a >= b for a, b in zip(bins, bins[1:]))):
            raise ValueError("Invalid skew partition axes")
    clip = table["alpha_clip"]
    if (len(clip) != 2 or any(not math.isfinite(v) for v in clip) or clip[0] > clip[1]
            or type(table["min_rows"]) is not int or table["min_rows"] < 1
            or set(table["n_range_by_kernel"]) != set(table["alpha_default_by_kernel"])):
        raise ValueError("Invalid skew support or fallback contract")
    for name, value in table["alpha_default_by_kernel"].items():
        bounds = table["n_range_by_kernel"].get(name)
        if (not math.isfinite(value) or not bounds or len(bounds) != 2
                or any(type(v) is not int for v in bounds) or not 2 <= bounds[0] <= bounds[1]):
            raise ValueError("Invalid fallback or measured range")
    with Path(path).open(newline="") as stream:
        for r in csv.DictReader(stream):
            layer, q = r["layer"], int(r["decode_q_len"])
            kernel = _kernel(layer, q)
            if kernel not in table["alpha_default_by_kernel"] or q < 1:
                raise ValueError("Cell has no matching kernel/query fallback")
            if (r["pc_label"] not in table["axes"]["pc_labels"]
                    or r["lev_label"] not in table["axes"]["lev_labels"]):
                raise ValueError("Cell has unknown axis labels")
            name = "|".join((kernel, r["pc_label"], r["lev_label"]))
            part = table["slices"].setdefault(name, dict(anchors=[], alpha=[], direct_rows=[], pooled_rows=[]))
            anchor, alpha = int(r["n_anchor"]), float(r["alpha"])
            direct, pooled = int(r["direct_rows"]), int(r["pooled_rows"])
            low, high = table["n_range_by_kernel"][kernel]
            if (not low <= anchor <= high or (part["anchors"] and anchor <= part["anchors"][-1])
                    or not math.isfinite(alpha) or not table["alpha_clip"][0] <= alpha <= table["alpha_clip"][1]
                    or direct < table["min_rows"] or pooled < direct):
                raise ValueError("Invalid or duplicated supported skew cell")
            for key, value in (("anchors", anchor),("alpha", alpha),("direct_rows", direct),("pooled_rows", pooled)):
                part[key].append(value)
    return table
