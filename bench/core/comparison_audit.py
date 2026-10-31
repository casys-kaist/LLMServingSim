"""Optional offline validation of declared native/simulator request correspondence."""

import csv
import hashlib
import json
import math
from pathlib import Path


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_comparison(bench_dir, sim_csv, workload):
    """Check declared request pairing; eligibility is not prediction accuracy."""
    bench_dir, sim_csv, workload = map(Path, (bench_dir, sim_csv, workload))
    errors, warnings, provenance = [], [], {}
    valid_rows = 0

    def issue(code, where, message):
        errors.append(dict(code=code, where=where, message=message))

    def read(path, kind):
        try:
            provenance[str(path)] = file_hash(path)
            if kind == 'json':
                return json.loads(path.read_text())
            if kind == 'jsonl':
                return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            with path.open() as f:
                return list(csv.DictReader(f))
        except (OSError, ValueError, csv.Error) as exc:
            issue('unreadable_input', str(path), str(exc))
            return {} if kind == 'json' else []

    meta = read(bench_dir/'meta.json', 'json')
    native = read(bench_dir/'requests.jsonl', 'jsonl')
    simulated = read(sim_csv, 'csv')
    planned = read(workload, 'jsonl')
    if not isinstance(meta, dict):
        issue('invalid_metadata', 'meta', 'Expected an object')
        meta = {}
    if not planned:
        issue('empty_workload', 'workload', 'A nonempty declared request plan is required')
    n = len(planned)
    if meta.get('num_requests') != n:
        issue('planned_count', 'meta.num_requests', f'Expected workload size {n}')
    if meta.get('dataset_hash') != provenance.get(str(workload)):
        issue('workload_hash', 'meta.dataset_hash', 'Does not bind to this workload file')
    if (bench_dir/'ABORTED.json').exists() or meta.get('resolve_only'):
        issue('ineligible_run', 'meta', 'Aborted or initialization-only acquisition')
    audits = meta.get('step_audit') or []
    if isinstance(audits, dict):
        audits = [audits]
    if not isinstance(audits, list) or any(not isinstance(a, dict) for a in audits):
        issue('invalid_step_audit', 'meta.step_audit', 'Expected records')
    else:
        for a in audits:
            if (a.get('kernel_trace_steps') or a.get('idle_ms') or a.get('error')
                    or a.get('end_to_end_control_eligible') is False):
                issue('instrumented_run', 'meta.step_audit', 'Not eligible as unpaced end-to-end truth')
    if not meta.get('model'):
        issue('model_missing', 'meta.model', 'Model identity is required')

    def by_id(rows, field, prefix, side):
        out, duplicates, invalid = {}, [], []
        for position, row in enumerate(rows):
            try:
                if not isinstance(row, dict):
                    raise ValueError('Row must be an object')
                value = str(row[field])
                if prefix and not value.startswith(prefix):
                    raise ValueError('Unexpected ID namespace')
                key = int(value[len(prefix):])
                if key in out:
                    duplicates.append(key)
                else:
                    out[key] = row
            except (ValueError, KeyError, TypeError):
                invalid.append(position)
        if duplicates:
            issue('duplicate_id', side, str(duplicates))
        if invalid:
            issue('invalid_id', side, str(invalid))
        missing, unexpected = sorted(set(range(n))-set(out)), sorted(set(out)-set(range(n)))
        if missing:
            issue('missing_id', side, str(missing))
        if unexpected:
            issue('unexpected_id', side, str(unexpected))
        if len(rows) != n:
            issue('row_count', side, f'{len(rows)} observed, {n} planned')
        return out, dict(rows=len(rows), unique_ids=len(out), duplicates=duplicates,
                         invalid_id_rows=invalid, missing_ids=missing, unexpected_ids=unexpected)

    nr, na = by_id(native, 'request_id', 'bench-', 'native')
    sr, sa = by_id(simulated, 'request id', '', 'simulator')

    def number(row, key, where, integer=False, csv_value=False):
        try:
            raw = row[key]
            # JSON timestamps are used directly by the native latency reader.
            # Accepting strings here would validate a temporary conversion while
            # leaving the original strings to fail during report generation.
            if not csv_value and type(raw) not in (int, float):
                raise ValueError()
            # The simulator reader uses int(text), not int(float(text)). Match
            # its grammar and retain exact integers rather than rounding via float.
            if integer:
                v = int(raw)
                if not csv_value and v != raw:
                    raise ValueError()
            else:
                v = float(raw)
            if not math.isfinite(v):
                raise ValueError()
            return v
        except (KeyError, TypeError, ValueError, OverflowError):
            issue('invalid_number', f'{where}.{key}', 'Required finite numeric value missing or invalid')
            return None

    for i in range(n):
        if i not in nr or i not in sr:
            continue
        r, s, w = nr[i], sr[i], planned[i]
        if not isinstance(w, dict):
            issue('invalid_plan', str(i), 'Expected workload object')
            continue
        row_start = len(errors)
        lengths = {}
        for field, sf in [('input_toks', 'input'), ('output_toks', 'output')]:
            values = [number(r, field, f'native[{i}]', True),
                      number(s, sf, f'simulator[{i}]', True, csv_value=True),
                      number(w, field, f'workload[{i}]', True)]
            if None not in values and (len(set(values)) != 1 or min(values) < 1):
                issue('token_count', f'request[{i}].{field}', str(values))
            lengths[field] = values[0]
        if s.get('model') != meta.get('model'):
            issue('model_mismatch', f'simulator[{i}].model',
                  f'{s.get("model")!r} versus {meta.get("model")!r}')
        times = {k:number(r,k,f'native[{i}]') for k in
                 ['arrival_time','queued_ts','scheduled_ts','first_token_ts','last_token_ts']}
        sim = {k:number(s,k,f'simulator[{i}]',csv_value=True) for k in
               ['arrival','end_time','latency','queuing_delay','TTFT','TPOT']}
        expected_arrival = number(w,'arrival_time_ns',f'workload[{i}]')
        if sim['arrival'] is not None and expected_arrival is not None:
            if abs(sim['arrival']-expected_arrival) > 1:
                issue('arrival_mismatch',f'simulator[{i}].arrival','Not the declared scheduled arrival')
        if all(v is not None for v in times.values()):
            order = [times[k] for k in ['queued_ts','scheduled_ts','first_token_ts','last_token_ts']]
            if order != sorted(order):
                issue('native_time_order',f'native[{i}]','Lifecycle timestamps are out of order')
        if all(v is not None for v in sim.values()):
            if min(sim.values()) < 0 or not 0 <= sim['queuing_delay'] <= sim['TTFT'] <= sim['latency']:
                issue('sim_time_order',f'simulator[{i}]','Negative or inconsistent lifecycle duration')
            if abs(sim['end_time']-sim['arrival']-sim['latency']) > 2:
                issue('sim_end_identity',f'simulator[{i}]','End minus arrival differs from latency')
            output = lengths['output_toks']
            if output and output > 1:
                expected_tpot=(sim['latency']-sim['TTFT'])/(output-1)
                if abs(expected_tpot-sim['TPOT']) > 2:
                    issue('sim_tpot_identity',f'simulator[{i}]','TPOT differs from decode duration / intervals')
        if len(errors) == row_start:
            valid_rows += 1

    if not errors:
        # Use precisely the conversion and units used to render the report,
        # including same-clock arrivals and the mixed-clock fallback. Do not
        # infer arrival ordering by directly comparing the two clock domains.
        from bench.core.validate import _bench_latencies

        try:
            metrics = _bench_latencies(native)
            for name, values in zip(('TTFT', 'TPOT', 'latency'), metrics):
                if any(not math.isfinite(value) or value < 0 for value in values):
                    issue('native_latency', name,
                          'Derived latency must be finite and nonnegative under the report conversion')
        except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
            issue('native_latency', 'native', f'Cannot compute report latency: {type(exc).__name__}')
    warnings.extend([
        'Latency calculation retains the existing upstream clock alignment; this audit does not calibrate clocks.',
        'The simulator output does not attest its input token IDs, model revision, hardware or execution settings.',
        'This upstream native recorder copies planned token counts; matching its reported counts does not verify actual returned token counts.',
        'Matching declared rows does not establish held-out calibration, native fault labels, or online feature availability.',
        'Simulator batch, cache and dispatch equivalence are not verified by this request-level adapter.'
    ])
    report=dict(schema_version=1,eligible=not errors,
                scope='offline declared-workload request-pair eligibility',
                planned=n,native=na,simulator=sa,valid_pair_rows=valid_rows,
                errors=errors,warnings=warnings,inputs=provenance,
                )
    return report
