"""Validated category publications and immutable exact-context serving snapshots.

Never wrap a live writer's dictionaries. Load from independently planned,
fully validated disk artifacts and copy only complete exact coordinates.
Reload explicitly to observe a later acquisition; an existing snapshot remains
the same verified data even if files or writer state subsequently change.
"""

from dataclasses import dataclass
from bisect import bisect_left
import csv
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

from .context_tables import (CATEGORY_SCHEMA, ContextDenseTable, canonical,
                             read_category_record, _identity)
from .measurement import measurement_token, measurement_compatibility


# Shared by the acquisition writer and CPU-only readers. A saved contract is
# not authority for its own point schema; callers select this independently.
CATEGORY_KEY_FIELDS = {
    'dense': ['layer', 'tokens'],
    'per_sequence': ['layer', 'sequences'],
    'attention': ['layer', 'prefill_chunk', 'prefill_key', 'n_decode',
                  'kv_decode', 'decode_q_len'],
    'linear_attention': ['layer', 'prefill_tokens', 'n_decode'],
    'moe': ['ep', 'tokens', 'activated_experts'],
    'mtp': ['layer', 'sequences'],
}


def validate_category_coverage(contract, expected_key_fields, expected_catalog):
    """Require independently selected axes and all bindings at every point.

    This checks binding coverage, not whether an acquisition sampled every
    possible coordinate. Its recorded grid remains the measured grid. A point
    schema without a label axis can represent only one canonical binding.
    """
    if (type(expected_key_fields) not in (list, tuple) or not expected_key_fields
            or any(type(field) is not str or not field for field in expected_key_fields)
            or len(set(expected_key_fields)) != len(expected_key_fields)
            or contract['key_fields'] != list(expected_key_fields)):
        raise ValueError('Category coordinate schema differs from its independent declaration')
    if (type(expected_catalog) is not dict or not expected_catalog
            or any(type(name) is not str or not name for name in expected_catalog)):
        raise ValueError('Category needs independently selected canonical bindings')
    if 'layer' not in expected_key_fields:
        if len(expected_catalog) != 1:
            raise ValueError('An unlabeled category cannot identify multiple canonical bindings')
        return
    label_index = expected_key_fields.index('layer')
    groups = {}
    for key in contract['expected_keys']:
        # Coordinate types/arity/duplicates are validated before this helper.
        coordinate = tuple(key[:label_index]) + tuple(key[label_index + 1:])
        groups.setdefault(coordinate, set()).add(key[label_index])
    required = set(expected_catalog)
    if not groups or any(labels != required for labels in groups.values()):
        raise ValueError('Category plan omits or adds canonical bindings at a coordinate')


def validate_category_rows(path, contract, *, require_complete=True):
    """Check exact planned coverage and stamps without importing GPU code.

    Coordinate types come from the independent category point plan. This covers
    integer axes, fractional attention keys and string labels without a list of
    model/category exceptions. The publication reader separately admits the
    acquisition scope and reconstructs its measurement identity.
    """
    fields = contract['key_fields']
    keys = contract['expected_keys']
    if (type(fields) is not list or not fields or len(set(fields)) != len(fields)
            or any(type(field) is not str or not field for field in fields)
            or set(fields) & {'time_us', 'measurement_protocol', 'measurement_sha256'}
            or type(keys) is not list or not keys):
        raise ValueError('Invalid category coordinate plan')
    types = None
    planned = set()
    for key in keys:
        if type(key) not in (list, tuple) or len(key) != len(fields):
            raise ValueError('Invalid planned category coordinate')
        current = tuple(type(value) for value in key)
        if (any(kind not in (str, int, float) for kind in current)
                or any(type(value) is float and not math.isfinite(value) for value in key)):
            raise ValueError('Invalid category coordinate type')
        # An axis may use exact integers and fractional values in one plan.
        current = tuple(float if kind in (int, float) else str for kind in current)
        if types is not None and current != types:
            raise ValueError('Inconsistent category coordinate types')
        types = current
        if tuple(key) in planned:
            raise ValueError('Duplicate planned category coordinate')
        planned.add(tuple(key))
    stamp = _identity(contract['measurement'])
    actual = set()
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        historical = reader.fieldnames == [*fields, 'time_us', 'measurement_protocol', 'measurement_sha256']
        if reader.fieldnames != [*fields, 'time_us'] and not historical:
            raise ValueError('Category CSV schema differs')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Ragged category row')
            values = []
            for field, kind in zip(fields, types):
                value = row[field]
                if kind is str:
                    values.append(value)
                else:
                    # Preserve arbitrary-size integer coordinates; float(int)
                    # would alias distinct coordinates beyond 2**53.
                    try:
                        parsed = int(value)
                    except ValueError:
                        parsed = float(value)
                        if not math.isfinite(parsed):
                            raise ValueError('Non-finite category coordinate')
                    values.append(parsed)
            key = tuple(values)
            time = float(row['time_us'])
            if (key not in planned or key in actual or not math.isfinite(time) or time <= 0
                    or (historical and any(row[field] != value for field, value in stamp.items()))):
                raise ValueError('Invalid, repeated or incompatible category row')
            actual.add(key)
    if require_complete and actual != planned:
        raise ValueError('Category is missing planned rows')


def load_category_record(directory, category, *, expected_key_fields,
                         validate_scope, identity_for_scope):
    """Load a complete table under an independently admitted acquisition.

    ``validate_scope`` admits the recorded execution facts. ``identity_for_scope``
    takes that scope and the recorded torch/vLLM version strings and rebuilds the
    identity using the caller's independently selected bindings and contexts.
    Neither callback receives the recorded catalog as authority.
    ``expected_key_fields`` comes from the caller's category declaration, not
    from the persisted plan. Every coordinate must cover the independently
    reconstructed identity's complete catalog.
    """
    saved = read_category_record(directory, category)
    if (type(saved) is not dict or set(saved) != {'contract', 'status', 'csv_sha256'}
            or saved['status'] != 'complete'):
        raise ValueError('Category is not complete: ' + category)
    contract = saved['contract']
    if (type(contract) is not dict or set(contract) != {'schema', 'category', 'scope',
            'identity', 'measurement', 'key_fields', 'expected_keys'}
            or contract['schema'] != CATEGORY_SCHEMA or contract['category'] != category):
        raise ValueError('Category publication differs')
    validate_scope(contract['scope'])
    ident = contract['identity']
    if type(ident) is not dict:
        raise ValueError('Invalid category measurement identity')
    version = {key: ident.get(key) for key in ('torch', 'vllm')}
    if any(type(value) is not str or not value for value in version.values()):
        raise ValueError('Invalid recorded acquisition runtime versions')
    expected = identity_for_scope(contract['scope'], version)
    if canonical(measurement_compatibility(ident)) != canonical(measurement_compatibility(expected)):
        raise ValueError('Category measurement semantics/catalog/runtime changed')
    stamp = measurement_token(ident)
    if 'sources' in ident:
        stamp = dict(measurement_protocol=ident['protocol'],
                     measurement_sha256=hashlib.sha256(canonical(ident).encode()).hexdigest())
    if _identity(contract['measurement']) != stamp:
        raise ValueError('Category measurement stamp differs from its identity')
    path = Path(directory) / (category + '.csv')
    if hashlib.sha256(path.read_bytes()).hexdigest() != saved['csv_sha256']:
        raise ValueError('Category CSV changed after completion')
    validate_category_rows(path, contract)
    validate_category_coverage(contract, expected_key_fields, expected['catalog'])
    return contract


@dataclass(frozen=True, slots=True, init=False)
class FrozenContextLookup:
    _manifest_text: str
    _values: MappingProxyType
    _curves: MappingProxyType

    def __init__(self):
        raise TypeError('Use FrozenContextLookup.load for full disk validation')

    @classmethod
    def load(cls, directory, manifest, ordinary_measurement, context_measurements):
        """Validate the entire independent plan and acquisition before freezing.

        An incomplete shot is not accepted just because another shot is complete.
        Model-specific scope checks must precede this generic table load.
        """
        table = ContextDenseTable(directory, manifest, ordinary_measurement, context_measurements)
        completed = table.preload()
        expected = {(shot['tokens'],) for shot in table.manifest['shots']}
        if completed != expected:
            raise ValueError('Cannot freeze incomplete ordinary/context acquisition coverage')
        instance = object.__new__(cls)
        object.__setattr__(instance, '_manifest_text', canonical(table.manifest))
        # Fresh values dict: no alias into a writer or caller-owned object.
        object.__setattr__(instance, '_values', MappingProxyType(dict(table.rows)))
        curves = {}
        for (shot, layer, context, tokens), value in table.rows.items():
            if shot != tokens:
                continue
            curves.setdefault((layer, context), []).append((tokens, value))
        object.__setattr__(instance, '_curves', MappingProxyType({
            key: tuple(sorted(rows)) for key, rows in curves.items()}))
        return instance

    @property
    def manifest(self):
        """Detached configuration for one-time adapter compatibility checks."""
        return json.loads(self._manifest_text)

    def exact(self, *, shot_tokens, layer, context, context_tokens):
        """One exact dictionary lookup, no geometry parsing or integrity scans.

        The context string must equal a canonical planned key already validated
        at load. Bool/float numeric aliases are refused instead of matching ints.
        """
        if (type(shot_tokens) is not int or type(context_tokens) is not int
                or type(layer) is not str or type(context) is not str):
            raise ValueError('Exact context lookup requires integer and string coordinates')
        return self._values[(shot_tokens, layer, context, context_tokens)]

    def interpolate(self, *, tokens, layer, context):
        """Linear token interpolation inside one immutable structural context.

        There is no nearest-context fallback or extrapolation. Only tables
        whose shot and actual token dimensions agree form these curves.
        """
        if type(tokens) is not int or tokens < 1 or type(layer) is not str or type(context) is not str:
            raise ValueError('Context lookup requires positive integer tokens and exact string keys')
        rows = self._curves[(layer, context)]
        if not rows[0][0] <= tokens <= rows[-1][0]:
            raise ValueError('Context token count is outside the measured range')
        index = bisect_left(rows, (tokens,))
        high, b = rows[index]
        if high == tokens:
            return b
        low, a = rows[index - 1]
        return a + (b - a) * ((tokens - low) / (high - low))
