"""Atomic category publications and exact-context dense acquisition tables."""

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from dataclasses import asdict

from .hooks.callable_context import expected_counts, validate_context_text
from .measurement import FIELDS, measurement_token, measurement_compatibility, semantic_identity
from .bundle_metadata import read_tp, update_tp


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


CATEGORY_SCHEMA = 'category-acquisition'


def read_contract(path):
    """Read canonical JSON; duplicate keys and non-finite values are refused."""
    raw = Path(path).read_text()
    value = json.loads(raw)
    if raw != canonical(value) + '\n':
        raise ValueError('Acquisition contract is not canonical JSON')
    return value


def category_record_path(directory, category):
    if (type(category) is not str or not category or
            any(char not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for char in category)):
        raise ValueError('Invalid acquisition category name')
    return Path(directory) / ('category_' + category + '.json')


def metadata_namespace(directory):
    directory = Path(directory)
    return (directory.parent, 'callables') if directory.name == 'callables' else (directory, 'ordinary')


def read_namespace(directory, section):
    tpdir, namespace = metadata_namespace(directory)
    records = read_tp(tpdir, section, {})
    if type(records) is not dict:
        raise ValueError('Invalid ' + section + ' metadata')
    if namespace not in records:
        return None
    if type(records[namespace]) is not dict:
        raise ValueError('Invalid ' + section + ' namespace')
    return records[namespace]


def update_namespace(directory, section, mutator):
    tpdir, namespace = metadata_namespace(directory)
    def update(records):
        records = {} if records is None else dict(records)
        records[namespace] = mutator(records.get(namespace))
        return records
    return update_tp(tpdir, section, update)


def read_category_record(directory, category):
    """Prefer the root publication; historical JSON is a read-only fallback."""
    path = category_record_path(directory, category)
    records = read_tp(directory, 'categories', {})
    if type(records) is not dict:
        raise ValueError('Invalid category publication metadata')
    if category in records:
        return records[category]
    return read_contract(path) if path.exists() else None


def read_context_record(directory):
    """Read root context state without creating files; retain old sidecar reads."""
    record = read_namespace(directory, 'contexts')
    if record is not None:
        if set(record) != {'contract', 'receipts'}:
            raise ValueError('Invalid context metadata record')
        return record
    directory = Path(directory)
    manifest = directory / 'dense_context_manifest.json'
    receipts = directory / 'dense_context_shots.json'
    if not manifest.exists():
        return None
    return dict(contract=read_contract(manifest),
                receipts=read_contract(receipts) if receipts.exists() else None)


class CategoryRecord:
    """Complete category publication and all-layer shot resume for any catalog.

    The category supplies the point schema and shot grouping. The caller supplies
    its complete acquisition identity, including any source-admitted runtime or
    attention semantics; this writer neither selects a model nor admits one.
    """

    def __init__(self, directory, category, arch, shots, catalog, measurement, scope, *,
                 replace, identity, tp=1):
        from .hooks.timings import TimingSample
        from .writer import sink_for
        self.directory = Path(directory)
        self.category = category
        self.path = category_record_path(directory, category.name)
        fields = sink_for(category, directory, measurement).key_fields
        self.groups = {}
        for shot in shots:
            keys = {tuple(asdict(point)[field] for field in fields)
                    for point in category.extract_points(shot,
                        [TimingSample(name, 1.0) for name in catalog], arch, tp)}
            if not keys:
                raise ValueError('Category shot has no planned coordinates')
            self.groups.setdefault(category.shot_key(shot), set()).update(keys)
        self.keys = set().union(*self.groups.values()) if self.groups else set()
        if not self.keys:
            raise ValueError('Category has no planned coordinates')
        expected = measurement_token(identity)
        if _identity(measurement) != expected:
            raise ValueError('Category identity differs from its acquisition contract')
        self.contract = dict(schema=CATEGORY_SCHEMA, category=category.name, scope=scope,
            identity=identity, measurement=measurement, key_fields=fields,
            expected_keys=sorted(self.keys))
        csv_path = self.directory / category.sink_filename
        old = read_category_record(directory, category.name)
        if old is not None and not replace:
            old_contract = old.get('contract', {})
            current = json.loads(canonical(self.contract))
            if semantic_identity(old_contract) != semantic_identity(current):
                raise ValueError('Category plan/runtime changed; use fresh output or explicit --force')
            self.contract = old_contract
            if old.get('status') not in ('writing', 'complete'):
                raise ValueError('Invalid category publication status')
            if old['status'] == 'complete' and old.get('csv_sha256') != hashlib.sha256(
                    csv_path.read_bytes()).hexdigest():
                raise ValueError('Completed category CSV changed after publication')
        elif csv_path.exists() and not replace:
            raise ValueError('Existing timings lack an acquisition contract')
        if csv_path.exists() and not replace:
            # Validate before the ordinary sink collapses duplicate keys or
            # ignores extra cells. A partial file may omit planned rows, but
            # malformed retained rows must never be republished as clean data.
            from .context_lookup import validate_category_rows
            validate_category_rows(csv_path, self.contract, require_complete=False)
        self.write('writing')

    def write(self, status, **fields):
        publication = dict(contract=self.contract, status=status, **fields)
        def update(records):
            records = {} if records is None else dict(records)
            records[self.category.name] = publication
            return records
        update_tp(self.directory, 'categories', update)

    def resume_keys(self, sink):
        actual = set(sink._bucket)
        if actual - self.keys:
            raise ValueError('Category table contains unplanned coordinates')
        complete = {shot for shot, keys in self.groups.items() if keys <= actual}
        for shot, keys in self.groups.items():
            if shot not in complete:
                for key in keys:
                    sink._bucket.pop(key, None)
                    sink._row_measurements.pop(key, None)
        return complete

    def complete(self, sink):
        if set(sink._bucket) != self.keys:
            raise ValueError('Planned category is incomplete')
        # Check disk as well as memory before publishing its digest. A stale
        # flush cannot turn an incomplete file into a complete acquisition.
        from .context_lookup import validate_category_rows
        validate_category_rows(sink.out_path, self.contract)
        self.write('complete', csv_sha256=hashlib.sha256(sink.out_path.read_bytes()).hexdigest())


def normalize_manifest(value):
    if (not isinstance(value, dict) or set(value) != {'schema', 'shots'}
            or type(value['schema']) is not int or value['schema'] != 1
            or not isinstance(value['shots'], list) or not value['shots']):
        raise ValueError('An independent nonempty schema-1 shot manifest is required')
    shots = {}
    for shot in value['shots']:
        if not isinstance(shot, dict) or set(shot) != {'tokens', 'ordinary_layers', 'contexts'}:
            raise ValueError('Invalid context shot manifest')
        tokens, ordinary = shot['tokens'], shot['ordinary_layers']
        if type(tokens) is not int or tokens < 1 or tokens in shots:
            raise ValueError('Shot token coordinates must be positive and unique')
        if (not isinstance(ordinary, list) or any(not isinstance(x, str) or not x for x in ordinary)
                or len(set(ordinary)) != len(ordinary)):
            raise ValueError('Ordinary layers must be distinct names')
        counts = expected_counts(shot['contexts'])
        if set(ordinary) & {key[0] for key in counts}:
            raise ValueError('A layer cannot be both ordinary and contextual')
        shots[tokens] = dict(tokens=tokens, ordinary_layers=sorted(ordinary), contexts=[
            dict(layer=k[0], context=k[1], context_tokens=k[2], invocations=n)
            for k, n in sorted(counts.items())])
    return dict(schema=1, shots=[shots[key] for key in sorted(shots)])


def _identity(value):
    if (not isinstance(value, dict) or set(value) != set(FIELDS)
            or any(not isinstance(x, str) or not x for x in value.values())):
        raise ValueError('Invalid acquisition identity')
    return dict(value)


def _atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _positive_time(value):
    if isinstance(value, bool):
        raise ValueError('Invalid context timing')
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError('Timing must be finite and positive')
    return result


def _integer(text):
    if not isinstance(text, str) or not text.isascii() or not text.isdecimal() or str(int(text)) != text:
        raise ValueError('CSV coordinates must be canonical positive integers')
    result = int(text)
    if result < 1:
        raise ValueError('CSV coordinates must be positive')
    return result


class ContextDenseTable:
    """Two-set resume with a fixed independent plan and per-shot method stamps.

    Incomplete disk checkpoints are safe: only shots with every ordinary row
    and every planned context row are complete. Reacquisition replaces the
    whole shot instead of averaging a partial previous acquisition into it.
    """

    columns = ['layer', 'tokens', 'context', 'context_tokens', 'time_us']

    def __init__(self, directory, manifest, ordinary_measurement, context_measurements):
        self.directory = Path(directory)
        self.manifest = normalize_manifest(manifest)
        self.shots = {shot['tokens']: shot for shot in self.manifest['shots']}
        self.sha256 = hashlib.sha256(canonical(self.manifest).encode()).hexdigest()
        self.ordinary_measurement = _identity(ordinary_measurement)
        if set(context_measurements) != set(self.shots):
            raise ValueError('Every planned shot needs a contextual acquisition identity')
        self.context_measurements = {key: _identity(value) for key, value in context_measurements.items()}
        self.contract = dict(manifest=self.manifest, context_manifest_sha256=self.sha256,
            ordinary_measurement=self.ordinary_measurement,
            context_measurements={str(k): v for k, v in sorted(self.context_measurements.items())})
        self.rows = {}
        self.ordinary_keys = set()
        self.ordinary_values = {}
        self.receipts = {}

    @property
    def path(self):
        return self.directory / 'dense_context.csv'

    def _shot_hash(self, tokens):
        values = dict(ordinary=[(name, format(self.ordinary_values[(name, tokens)], '.6g'))
                               for name in self.shots[tokens]['ordinary_layers']],
                      contexts=[(*key, format(self.rows[(tokens, *key)], '.6g'))
                                for key in sorted(self._planned(tokens))])
        return hashlib.sha256(canonical(values).encode()).hexdigest()

    def _planned(self, tokens):
        if tokens not in self.shots:
            raise ValueError('Unplanned shot coordinate')
        return set(expected_counts(self.shots[tokens]['contexts']))

    def preload(self, *, replace=False):
        """Validate existing files without writing; return complete shot keys."""
        if replace:
            return set()
        files_exist = self.path.exists() or (self.directory / 'dense.csv').exists()
        metadata = read_context_record(self.directory)
        if metadata is not None:
            if semantic_identity(metadata['contract']) != semantic_identity(self.contract):
                raise ValueError('Context manifest or acquisition identity changed')
        elif files_exist:
            raise ValueError('Context tables have no independent persisted manifest')
        if self.path.exists():
            with self.path.open(newline='') as stream:
                reader = csv.DictReader(stream)
                historical = reader.fieldnames == [*self.columns, *FIELDS, 'context_manifest_sha256']
                if reader.fieldnames != self.columns and not historical:
                    raise ValueError('Invalid contextual CSV columns')
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError('Malformed contextual CSV row')
                    tokens, actual = _integer(row['tokens']), _integer(row['context_tokens'])
                    validate_context_text(row['context'])
                    key = (tokens, row['layer'], row['context'], actual)
                    if key[1:] not in self._planned(tokens) or key in self.rows:
                        raise ValueError('Unexpected or duplicate contextual row')
                    if historical and ({f: row[f] for f in FIELDS} != self.context_measurements[tokens]
                            or row['context_manifest_sha256'] != self.sha256):
                        raise ValueError('Context row acquisition or manifest identity differs')
                    self.rows[key] = _positive_time(row['time_us'])
        ordinary_path = self.directory / 'dense.csv'
        if ordinary_path.exists():
            with ordinary_path.open(newline='') as stream:
                reader = csv.DictReader(stream)
                historical = reader.fieldnames == ['layer', 'tokens', 'time_us', *FIELDS]
                if reader.fieldnames != ['layer', 'tokens', 'time_us'] and not historical:
                    raise ValueError('Invalid ordinary CSV in context acquisition')
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError('Malformed ordinary row')
                    tokens = _integer(row['tokens'])
                    key = (row['layer'], tokens)
                    if (tokens not in self.shots or row['layer'] not in self.shots[tokens]['ordinary_layers']
                            or key in self.ordinary_keys):
                        raise ValueError('Unexpected or duplicate ordinary row')
                    if historical and {f: row[f] for f in FIELDS} != self.ordinary_measurement:
                        raise ValueError('Ordinary acquisition identity differs')
                    _positive_time(row['time_us'])
                    self.ordinary_keys.add(key)
                    self.ordinary_values[key] = float(row['time_us'])
        if metadata is not None and metadata['receipts'] is not None:
            receipt = metadata['receipts']
            if (not isinstance(receipt, dict) or set(receipt) != {'context_manifest_sha256', 'shots'}
                    or receipt['context_manifest_sha256'] != self.sha256
                    or not isinstance(receipt['shots'], dict)):
                raise ValueError('Invalid context completion receipt')
            for token_text, digest in receipt['shots'].items():
                token = _integer(token_text)
                if (token not in self.shots or not isinstance(digest, str) or len(digest) != 64
                        or any(c not in '0123456789abcdef' for c in digest)):
                    raise ValueError('Invalid context shot receipt')
                self.receipts[token] = digest
        return self.complete_keys()

    def complete_keys(self):
        return {(tokens,) for tokens, shot in self.shots.items()
                if {(name, tokens) for name in shot['ordinary_layers']} <= self.ordinary_keys
                and {(tokens, *key) for key in self._planned(tokens)} <= self.rows.keys()
                and self.receipts.get(tokens) == self._shot_hash(tokens)}

    def validate_rows(self, tokens, rows):
        """Check the whole rank result before either table is mutated."""
        planned = self._planned(tokens)
        ordinary, contextual = set(), set()
        if not isinstance(rows, list):
            raise ValueError('Expected worker timing rows')
        for row in rows:
            _positive_time(row['microseconds'])
            is_context = 'context' in row or 'context_tokens' in row
            identity = self.context_measurements[tokens] if is_context else self.ordinary_measurement
            if any(row.get(f) != identity[f] for f in FIELDS):
                raise ValueError('Worker acquisition identity differs')
            if is_context:
                validate_context_text(row.get('context'))
                actual = row.get('context_tokens')
                if type(actual) is not int or actual < 1:
                    raise ValueError('Actual context token dimension must be positive')
                key = (row['layer'], row['context'], actual)
                if key not in planned or key in contextual:
                    raise ValueError('Unexpected or repeated context')
                contextual.add(key)
            else:
                if row['layer'] not in self.shots[tokens]['ordinary_layers'] or row['layer'] in ordinary:
                    raise ValueError('Unexpected or repeated ordinary layer')
                ordinary.add(row['layer'])
        if contextual != planned or ordinary != set(self.shots[tokens]['ordinary_layers']):
            raise ValueError('Worker result is missing required ordinary/context rows')

    def accept_shot(self, tokens, points, ordinary_sink):
        """Commit validated points, replacing any partial previous shot."""
        points = list(points)
        rows = [dict(layer=p.layer, microseconds=p.microseconds,
                     **(dict(context=p.context, context_tokens=p.context_tokens)
                        if hasattr(p, 'context') else {}),
                     **(self.context_measurements[tokens] if hasattr(p, 'context')
                        else self.ordinary_measurement)) for p in points]
        if any(p.tokens != tokens for p in points):
            raise ValueError('Point shot coordinate differs')
        self.validate_rows(tokens, rows)
        for key in list(ordinary_sink._bucket):
            if key[1] == tokens:
                del ordinary_sink._bucket[key]
                ordinary_sink._row_measurements.pop(key, None)
        self.rows = {key: value for key, value in self.rows.items() if key[0] != tokens}
        self.ordinary_keys = {key for key in self.ordinary_keys if key[1] != tokens}
        for point in points:
            if hasattr(point, 'context'):
                self.rows[(tokens, point.layer, point.context, point.context_tokens)] = point.microseconds
            else:
                ordinary_sink.coalesce(point)
                self.ordinary_keys.add((point.layer, tokens))
                self.ordinary_values[(point.layer, tokens)] = point.microseconds
        self.receipts[tokens] = self._shot_hash(tokens)

    def flush(self, ordinary_sink):
        # Persist the independent contract first. A crash between table writes
        # leaves an incomplete shot, which resume reacquires in full.
        previous = read_context_record(self.directory)
        retained = (previous['receipts'] if previous is not None
                    and semantic_identity(previous['contract']) == semantic_identity(self.contract) else None)
        update_namespace(self.directory, 'contexts', lambda _current:
            dict(contract=self.contract, receipts=retained))
        if ordinary_sink._bucket:
            ordinary_sink.flush(clear=False)
        else:
            # A context-only catalog still replaces stale ordinary rows under
            # an explicit force acquisition, without changing legacy sinks.
            _atomic(self.directory / 'dense.csv', 'layer,tokens,time_us\r\n')
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=self.columns)
        writer.writeheader()
        for (tokens, layer, context, actual), value in sorted(self.rows.items()):
            writer.writerow(dict(layer=layer, tokens=tokens, context=context, context_tokens=actual,
                time_us=format(value, '.6g')))
        _atomic(self.path, stream.getvalue())
        receipt = dict(context_manifest_sha256=self.sha256,
            shots={str(key[0]): self.receipts[key[0]] for key in sorted(self.complete_keys())})
        update_namespace(self.directory, 'contexts', lambda _current:
            dict(contract=self.contract, receipts=receipt))

    def exact(self, *, shot_tokens, layer, context, context_tokens):
        """Read a planned exact coordinate only; no nearest-token fallback."""
        validate_context_text(context)
        if type(shot_tokens) is not int or type(context_tokens) is not int:
            raise ValueError('Exact lookup needs integer coordinates')
        key = (layer, context, context_tokens)
        if key not in self._planned(shot_tokens):
            raise KeyError('Context lookup is not in the independent plan')
        if (shot_tokens,) not in self.complete_keys():
            raise KeyError('Context shot has incomplete ordinary/context coverage')
        return self.rows[(shot_tokens, *key)]
