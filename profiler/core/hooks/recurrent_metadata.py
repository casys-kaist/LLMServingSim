"""Fail-closed recurrent warmup validation; never used during timing.

Dispatch is by installed cache specs and an explicit metadata field contract,
not checkpoint names. Certifies ordinary non-spec, cache-mode-none GDN and the
native KDA metadata schema with their distinct prefill views.
Other recurrent schemas and
active speculative/checkpoint protocols require their own audited contract.
"""
from collections import Counter
from dataclasses import dataclass
import inspect
from numbers import Integral


@dataclass(frozen=True)
class RequestState:
    query: int
    history: int
    pages: tuple[int, ...]


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{label} must be an integer')
    return int(value)


def _array(value, ndim, label):
    if value is None or getattr(value, 'ndim', None) != ndim:
        raise ValueError(f'{label} requires a {ndim}-dimensional tensor')
    return value.detach().cpu().tolist()


def _starts(value, label):
    starts = [_integer(v, label) for v in _array(value, 1, label)]
    if not starts or starts[0] != 0 or any(b < a for a, b in zip(starts, starts[1:])):
        raise ValueError(f'Invalid {label} query boundaries')
    lengths = [b - a for a, b in zip(starts, starts[1:])]
    active = len(lengths)
    while active and lengths[active - 1] == 0:
        active -= 1
    if not active or any(length == 0 for length in lengths[:active]):
        raise ValueError(f'{label} has empty/interleaved active requests')
    return lengths, active


def common_geometry(common, expected):
    queries, active = _starts(common.query_start_loc, 'common query_start_loc')
    lengths = [_integer(v, 'common seq_lens') for v in _array(common.seq_lens, 1, 'common seq_lens')]
    if len(lengths) != len(queries):
        raise ValueError('Common query/sequence row count mismatch')
    rows = [(q, total - q) for q, total in zip(queries[:active], lengths[:active])]
    if any(history < 0 for _, history in rows):
        raise ValueError('Negative common request history')
    if Counter(rows) != Counter((r.query, r.history) for r in expected):
        raise ValueError('Common recurrent query/history geometry differs from the shot')
    if _integer(common.num_actual_tokens, 'num_actual_tokens') != sum(q for q, _ in rows):
        raise ValueError('Common recurrent actual-token count differs from the shot')
    return rows


def verify_block_rows(block_table, rows, expected):
    """Bind each runtime query/history pair to its own physical state pages."""
    table = _array(block_table, 2, 'state block_table_tensor')
    if len(table) < len(rows):
        raise ValueError('Missing state block-table rows')
    by_first = {r.pages[0]: r for r in expected if r.pages}
    if len(by_first) != len(expected):
        raise ValueError('Duplicate or missing expected state page')
    seen = set()
    for (query, history), actual in zip(rows, table):
        if not actual:
            raise ValueError('Empty state block-table row')
        first = _integer(actual[0], 'state page')
        record = by_first.get(first)
        if first in seen or record is None:
            raise ValueError('State block table aliases or addresses another cache group')
        seen.add(first)
        if (query, history) != (record.query, record.history):
            raise ValueError('State page is assigned to the wrong query/history pair')
        prefix = tuple(_integer(v, 'state page') for v in actual[:len(record.pages)])
        if prefix != record.pages:
            raise ValueError('State block-table page span differs from assigned pages')
    if seen != set(by_first):
        raise ValueError('State block table omits a request')


# Exact union of the 0.28/0.31 ordinary GDN dataclass field sets. A recurrent
# backend adding checkpoint/RecoverSSM fields must not silently inherit this
# contract just because it also exposes a query-start vector.
KNOWN_FIELDS = frozenset('''num_prefills num_prefill_tokens num_decodes num_decode_tokens
    num_spec_decodes num_spec_decode_tokens num_actual_tokens has_initial_state
    spec_query_start_loc non_spec_query_start_loc spec_state_indices_tensor
    non_spec_state_indices_tensor spec_sequence_masks spec_sequence_masks_cpu
    spec_token_indx non_spec_token_indx num_accepted_tokens uniform_spec_sequence_length
    chunk_indices chunk_offsets prefill_query_start_loc prefill_state_indices
    prefill_has_initial_state nums_dict batch_ptr token_chunk_offset_ptr'''.split())

_KDA_FIELDS = frozenset('''spec_token_start non_spec_token_start
    flashinfer_prefill_query_start_loc flashinfer_prefill_seq_order
    recoverssm_commit recoverssm_context checkpoint'''.split())
_KDA_UNUSED_GDN_FIELDS = ('chunk_indices', 'chunk_offsets', 'prefill_query_start_loc',
                          'prefill_state_indices', 'prefill_has_initial_state')


def _ordinary_kda(metadata):
    cls = type(metadata)
    if (cls.__module__, cls.__name__) != (
            'vllm.models.kimi_k3.nvidia.kda_metadata', 'KimiK3KDAMetadata'):
        return False
    # Native builder: no speculative partition offsets in ordinary mode;
    # FlashKDA prepares its own chunks and consumes the whole non-spec view
    # (including the decode prefix of a mixed batch). It never fills GDN's
    # prefill-only FLA view. Other optional execution paths need their own audit.
    if not _KDA_FIELDS.issubset(vars(metadata)):
        raise ValueError('Incomplete native KDA metadata schema')
    for field in (*_KDA_FIELDS, *_KDA_UNUSED_GDN_FIELDS):
        if getattr(metadata, field, None) is not None:
            raise ValueError('Active KDA metadata path is not certified: ' + field)
    return True


def verify_final_state(metadata, expected, geometry_rows):
    """Verify the actual metadata handed to the model, after group reuse."""
    fields = set(vars(metadata))
    is_kda = _ordinary_kda(metadata)
    known = KNOWN_FIELDS | _KDA_FIELDS if is_kda else KNOWN_FIELDS
    required = {'num_prefills', 'num_prefill_tokens', 'num_decodes', 'num_decode_tokens',
                'num_spec_decodes', 'num_spec_decode_tokens', 'num_actual_tokens',
                'non_spec_query_start_loc', 'non_spec_state_indices_tensor'}
    if not required.issubset(fields) or fields - known:
        raise ValueError(f'Unsupported recurrent metadata field contract: {type(metadata).__name__}; '
                         f'missing={sorted(required-fields)}, extra={sorted(fields-known)}')
    if (_integer(metadata.num_spec_decodes, 'num_spec_decodes') != 0 or
            _integer(metadata.num_spec_decode_tokens, 'num_spec_decode_tokens') != 0):
        raise ValueError('Active speculative recurrent metadata is not certified')
    for field in ('spec_query_start_loc', 'spec_state_indices_tensor', 'spec_sequence_masks',
                  'spec_sequence_masks_cpu', 'spec_token_indx', 'non_spec_token_indx',
                  'num_accepted_tokens', 'uniform_spec_sequence_length'):
        if getattr(metadata, field, None) is not None:
            raise ValueError(f'Active recurrent partition field is not certified: {field}')
    queries, active = _starts(metadata.non_spec_query_start_loc, 'non_spec_query_start_loc')
    ids = [_integer(v, 'state index') for v in
           _array(metadata.non_spec_state_indices_tensor, 1, 'non_spec_state_indices_tensor')]
    if len(ids) < active or active != len(geometry_rows):
        raise ValueError('Final recurrent state/query row count mismatch')
    if queries[:active] != [query for query, _ in geometry_rows]:
        raise ValueError('Final recurrent query order differs from verified common geometry')
    expected_triples = Counter((r.query, r.history, r.pages[0]) for r in expected)
    actual_triples = Counter((q, h, state) for (q, h), state in zip(geometry_rows, ids[:active]))
    if actual_triples != expected_triples:
        raise ValueError('Final recurrent state IDs do not match assigned query/history pairs')
    total = sum(queries[:active])
    n_prefill = _integer(metadata.num_prefills, 'num_prefills')
    n_decode = _integer(metadata.num_decodes, 'num_decodes')
    p_tokens = _integer(metadata.num_prefill_tokens, 'num_prefill_tokens')
    d_tokens = _integer(metadata.num_decode_tokens, 'num_decode_tokens')
    if (min(n_prefill, n_decode, p_tokens, d_tokens) < 0 or
            n_prefill + n_decode != active or p_tokens + d_tokens != total or
            d_tokens != sum(queries[:n_decode]) or
            _integer(metadata.num_actual_tokens, 'num_actual_tokens') != total):
        raise ValueError('Recurrent partition/actual-token counts do not conserve the query')
    initial = getattr(metadata, 'has_initial_state', None)
    if n_prefill and initial is None:
        raise ValueError('Prefill recurrent metadata lacks initial-state flags')
    if initial is not None:
        flags = _array(initial, 1, 'has_initial_state')
        if len(flags) < active or any(type(v) is not bool for v in flags):
            raise ValueError('Malformed recurrent initial-state flags')
        if flags[:active] != [history > 0 for _, history in geometry_rows]:
            raise ValueError('Recurrent initial-state flags differ from request histories')
    if n_prefill and not is_kda:
        p_queries, p_active = _starts(getattr(metadata, 'prefill_query_start_loc', None),
                                      'prefill_query_start_loc')
        p_ids = _array(getattr(metadata, 'prefill_state_indices', None), 1, 'prefill_state_indices')
        p_flags = _array(getattr(metadata, 'prefill_has_initial_state', None), 1,
                         'prefill_has_initial_state')
        if (p_active != n_prefill or p_queries[:p_active] != queries[n_decode:active] or
                p_ids[:p_active] != ids[n_decode:active] or len(p_ids) < p_active or
                len(p_flags) < p_active or any(type(v) is not bool for v in p_flags) or
                p_flags[:p_active] != [h > 0 for _, h in geometry_rows[n_decode:]]):
            raise ValueError('Prefill-only recurrent metadata differs from its actual request suffix')


class RecurrentMetadataGuard:
    """Untimed builder/update provenance and per-layer final metadata checks."""
    def __init__(self, runner, shot):
        from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs
        from .cache_pages import allocate_shot_blocks
        self.runner, self.shot = runner, shot
        self.layer_group, self.group_records = {}, {}
        self.provenance, self.checked, self.patched = {}, set(), []
        allocation = allocate_shot_blocks(shot, runner)
        for index, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            specs = (group.kv_cache_spec.kv_cache_specs if
                     isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) else
                     {name: group.kv_cache_spec for name in group.layer_names})
            for name, spec in specs.items():
                if isinstance(spec, MambaSpec):
                    self.layer_group[name] = index
                    self.group_records[index] = [RequestState(q, h, tuple(ids[index]))
                        for (q, h), ids in zip(shot.requests, allocation)]
        if self.layer_group and runner.vllm_config.cache_config.mamba_cache_mode != 'none':
            raise ValueError('Recurrent warmup verification currently certifies cache mode none only')

    def _remember(self, metadata, group, rows):
        verify_final_state(metadata, self.group_records[group], rows)
        # Keep a strong reference; id reuse cannot certify an unrelated object.
        self.provenance[id(metadata)] = (metadata, group, rows)

    def _patch(self, builder, name, group):
        original = getattr(builder, name)
        had_local, old_local = name in vars(builder), vars(builder).get(name)
        signature = inspect.signature(original)

        def checked(*args, **kwargs):
            bound = signature.bind(*args, **kwargs).arguments
            expected = self.group_records[group]
            if name == 'build':
                common = bound.get('common_attn_metadata')
                if common is None:
                    raise ValueError('Missing common metadata in recurrent builder')
                rows = common_geometry(common, expected)
                verify_block_rows(common.block_table_tensor, rows, expected)
            else:
                source = bound.get('metadata')
                entry = self.provenance.get(id(source))
                if entry is None or entry[0] is not source:
                    raise ValueError('Recurrent metadata update lacks verified common provenance')
                rows = entry[2]
                table = bound.get('blk_table', bound.get('block_table'))
                verify_block_rows(table, rows, expected)
            result = original(*args, **kwargs)
            self._remember(result, group, rows)
            return result

        setattr(builder, name, checked)
        self.patched.append((builder, name, had_local, old_local))

    def __enter__(self):
        seen = {}
        try:
            for group_index, groups in enumerate(getattr(self.runner, 'attn_groups', ())):
                if group_index not in self.group_records:
                    continue
                for group in groups:
                    if not set(group.layer_names).intersection(self.layer_group):
                        continue
                    for builder in group.metadata_builders:
                        previous = seen.get(id(builder))
                        if previous is not None:
                            if previous != group_index:
                                raise ValueError('One recurrent builder ambiguously serves multiple cache groups')
                            continue
                        seen[id(builder)] = group_index
                        self._patch(builder, 'build', group_index)
                        if callable(getattr(builder, 'update_block_table', None)):
                            self._patch(builder, 'update_block_table', group_index)
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def validate(self, name, metadata):
        group = self.layer_group[name]
        entry = self.provenance.get(id(metadata))
        if entry is None or entry[0] is not metadata or entry[1] != group:
            raise ValueError(f'No verified recurrent builder/update provenance for {name}')
        verify_final_state(metadata, self.group_records[group], entry[2])
        self.checked.add(name)

    def finish(self):
        missing = set(self.layer_group) - self.checked
        if missing:
            raise ValueError(f'Recurrent warmup omitted declared cache layers: {sorted(missing)}')

    def __exit__(self, *_exc):
        for builder, name, had_local, old_local in reversed(self.patched):
            if had_local:
                setattr(builder, name, old_local)
            else:
                delattr(builder, name)
        self.patched.clear()
