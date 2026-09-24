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
from .skew_calibration import atomic_yaml, measurement_shape
from .skew_plan import iter_cases
from .skew_support import complete_plan

PROTOCOL = "native-skew-per-forward-v1"


def measurement_fingerprint():
    """Changing timing attribution or batch construction requires remeasurement."""
    hooks = Path(__file__).parent / "hooks"
    value = hashlib.sha256()
    for name in ("skew_measurement.py", "timings.py", "batch.py", "sampler_shim.py"):
        value.update((hooks / name).read_bytes())
    return value.hexdigest()

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



def _existing_keys(path, layers, rounds, iterations, *, block_size=None):
    """A partial kernel set or an under-repeated shot is still unfinished."""
    if not path.exists():
        return set()
    grouped = {}
    fingerprint = measurement_fingerprint()
    for row in pd.read_csv(path, low_memory=False).fillna("").to_dict("records"):
        key = measurement_shape(row)[4]
        if row.get("case_id") and row["case_id"] != key:
            raise ValueError("Stored skew identity disagrees with its request geometry")
        if (row.get("measurement_protocol") != PROTOCOL
                or row.get("measurement_sha256") != fingerprint
                or (block_size is not None and int(row.get("block_size") or 0) != block_size)
                or int(row.get("rounds") or 0) < rounds
                or int(row.get("timed_forwards") or 0) != iterations):
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
    """Measure the dynamic plan; failures preserve completed cases and raise."""
    if args.skew_rounds < 1 or args.measurement_iterations < 1:
        raise ValueError("Positive skew repetition counts required")
    out = Path(tp_root) / "skew.csv"
    attention = Path(tp_root) / "attention.csv"
    if not attention.exists():
        raise FileNotFoundError("Skew calibration needs attention.csv; profile attention first")
    catalog = {name: dict(vllm=e.vllm, within=e.within, not_within=e.not_within,
                         tp_stable=e.tp_stable)
               for name, e in arch.catalog.attention.items()}
    from serving.core import trace_generator as tg
    table = tg._build_attention_tables_by_layer(tg._read_category_csv(str(attention), None))
    for name in catalog:
        if not set(args.attention_decode_q_lens) <= set(table.get(name, {})):
            raise ValueError(f"Missing exact attention query slices for {name}")
    prior = set() if args.force else _existing_keys(
        out, catalog, args.skew_rounds, args.measurement_iterations, block_size=limits.block_size)
    log.info("TP=%d skew: checking reference-cell support before acquisition", tp)
    plan, extra = complete_plan(args, limits, attention, arch, prior,
                               existing_csv=None if args.force else out)
    plan.update(enabled=True, tp=tp, measurement_protocol=PROTOCOL,
                measurement_sha256=measurement_fingerprint(),
                completed=False, measured_cases=0)
    status = Path(tp_root) / "skew.meta.yaml"
    Path(tp_root).mkdir(parents=True, exist_ok=True)
    if args.force and out.exists():
        out.unlink()  # Explicit --force discards this category's old acquisition.
    total = plan["cases"]
    log.info("TP=%d skew: %d planned cases, %d reusable; %d rounds x %d forwards",
             tp, total, plan["reusable_cases"], args.skew_rounds, args.measurement_iterations)
    support = plan["support_completion"]
    log.info("TP=%d skew: %d reference-selected additions; %d cells remain below support floor",
             tp, len(extra), len(support["remaining_deficits"]))
    atomic_yaml(status, plan)
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
                result = llm.collective_rpc("skew_measure", args=(
                    shot.as_dict(), catalog, args.measurement_iterations))[0]
                forwards = result["per_forward_us"]
                if len(forwards) != args.measurement_iterations:
                    raise ValueError("Incomplete skew timed-forward set")
                for name in catalog:
                    values = [float(row[name]) for row in forwards]
                    if any(not math.isfinite(value) or value <= 0 for value in values):
                        raise ValueError("Invalid skew kernel time")
                    by_layer[name].append(values)
            for name, samples in by_layer.items():
                target = statistics.median(statistics.median(values) for values in samples)
                rows.append(dict(layer=name, case_id=key,
                    requests_json=json.dumps(shot.requests, separators=(",", ":")),
                    n_prefill=shot.n_prefill, decode_q_len=shot.decode_q_len,
                    family=family, measurement_protocol=PROTOCOL,
                    measurement_sha256=plan["measurement_sha256"],
                    block_size=limits.block_size,
                    rounds=args.skew_rounds, timed_forwards=args.measurement_iterations,
                    round_timings_us_json=json.dumps(samples, separators=(",", ":")),
                    t_skew_us=round(target, 3)))
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
                atomic_yaml(status, plan)
    finally:
        if rows:
            _flush_rows(out, rows)
    plan.update(completed=True, measured_cases=fired, reused_cases=skipped,
                remaining_cases=0, elapsed_seconds=time.monotonic() - started)
    atomic_yaml(status, plan)
    log.success("skew -> %s", out)
    return out
