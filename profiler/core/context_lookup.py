"""Immutable exact-context serving snapshot, validated once before lookup.

Never wrap a live writer's dictionaries. Load from independently planned,
fully validated disk artifacts and copy only complete exact coordinates.
Reload explicitly to observe a later acquisition; an existing snapshot remains
the same verified data even if files or writer state subsequently change.
"""

from dataclasses import dataclass
from bisect import bisect_left
import json
from types import MappingProxyType

from .context_tables import ContextDenseTable, canonical


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
