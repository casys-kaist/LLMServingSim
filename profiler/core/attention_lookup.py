"""CPU-only, per-TP ownership of newly acquired attention coordinates.

Unmarked historical tables retain stack.probe_key_saturation. Native records
describe measured lookup semantics in metadata and bind a completed table.
"""
import csv
import hashlib
import json
from pathlib import Path
import re

SCHEMA = 'native-attention-lookup'
RULE = 'minimax-triton-total-selected-blocks'
INDEXER = 'vllm.models.minimax_m3.common.indexer.MiniMaxM3IndexerTritonImpl'
ATTEND = 'vllm.models.minimax_m3.common.sparse_attention.MiniMaxM3SparseTritonImpl'


def semantic_facts(value):
    """Separate lookup geometry from acquisition provenance."""
    if not isinstance(value, dict) or 'key_saturation' not in value or 'source' not in value:
        return value
    return dict(value, source={key: item for key, item in value['source'].items()
                               if key not in ('vllm_version', 'sha256')})


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def digest(value):
    return _digest(semantic_facts(value))


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mapping(value, keys, optional=()):
    if (not isinstance(value, dict) or not set(keys) <= set(value)
            or set(value) - set(keys) - set(optional)):
        raise ValueError('Malformed attention lookup contract')


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Invalid attention lookup digest')
    return value


def protocol(base, semantic_sha=None):
    # Lookup geometry has its own metadata field, not an encoded method suffix.
    if semantic_sha is not None:
        _sha(semantic_sha)
    return base


def protocol_digest(value):
    if not isinstance(value, str):
        raise ValueError('Invalid attention row protocol')
    if 'native-attention-lookup' not in value:
        return None
    parts = value.split('|')
    if len(parts) != 2:
        raise ValueError('Unknown native attention row protocol')
    name, separator, recorded = parts[1].partition(':')
    if not separator or name != SCHEMA:
        raise ValueError('Unknown native attention row protocol')
    return _sha(recorded)


def saturation(architecture):
    return {name: bool(entry.get('key_saturates'))
            for name, entry in architecture['catalog'].get('attention', {}).items()}


def validate_semantic(value, config=None, architecture=None):
    _mapping(value, ('schema', 'rule', 'model_type', 'profile_tp', 'key_saturation',
                     'saturation_by_layer', 'source'))
    if (value['schema'] != SCHEMA or value['rule'] != RULE
            or value['model_type'] != 'minimax_m3_vl'):
        raise ValueError('Unsupported native attention lookup rule')
    if type(value['profile_tp']) is not int or value['profile_tp'] < 1:
        raise ValueError('Invalid attention profile TP')
    source = value['source']
    _mapping(source, ('indexer_class', 'attend_class',
                     'topk_blocks', 'block_size', 'init_blocks', 'local_blocks',
                     'topk_width', 'indexer_dtype', 'kv_dtype', 'score_type'),
             optional=('vllm_version', 'sha256'))
    if (source['indexer_class'] != INDEXER or source['attend_class'] != ATTEND
            or source['indexer_dtype'] != 'bf16' or source['kv_dtype'] != 'bfloat16'
            or source['score_type'] != 'max'):
        raise ValueError('Unsupported native attention implementation/layout')
    for name in ('topk_blocks', 'block_size', 'init_blocks', 'local_blocks', 'topk_width'):
        if type(source[name]) is not int or source[name] < (0 if name in ('init_blocks', 'local_blocks') else 1):
            raise ValueError('Invalid native attention dimensions')
    # The Triton implementation consumes the configured top-k. Sixteen is the
    # native SM100 MSA indexer selection condition, not a Triton layout rule
    # (common/indexer.py:495-535). Keep the selected implementation checks above
    # and the backend's supported block size (common/indexer.py:90-95).
    if (source['topk_width'] != source['topk_blocks'] or source['block_size'] != 128
            or source['init_blocks'] + source['local_blocks'] > source['topk_blocks']
            or type(value['key_saturation']) is not int
            or value['key_saturation'] != source['topk_blocks'] * source['block_size']):
        raise ValueError('Contradictory native attention dimensions')
    flags = value['saturation_by_layer']
    if (flags != {'attention': False, 'indexer': False, 'sparse_attention': True}
            or any(type(flag) is not bool for flag in flags.values())):
        raise ValueError('Unsupported native attention saturation mapping')
    if architecture is not None and saturation(architecture) != flags:
        raise ValueError('Attention saturation catalog changed')
    if config is not None:
        from .stack import text_config
        cfg = text_config(config)
        sparse = cfg.get('sparse_attention_config') or {}
        if config.get('model_type') not in ('minimax_m3_vl', 'minimax_m3'):
            raise ValueError('Attention contract model type changed')
        expected = {'topk_blocks': sparse.get('sparse_topk_blocks'),
                    'block_size': sparse.get('sparse_block_size'),
                    'init_blocks': sparse.get('sparse_init_block', 0),
                    'local_blocks': sparse.get('sparse_local_block', 0)}
        if (sparse.get('use_sparse_attention') is not True
                or any(type(v) is not int or source[k] != v for k, v in expected.items())
                or sparse.get('sparse_score_type', 'max') != source['score_type']):
            raise ValueError('Attention contract model dimensions changed')
    return value


def metadata(root):
    from .bundle_metadata import read_bundle
    return read_bundle(root)


def records(meta):
    if 'attention_lookup' not in meta:
        return {}
    block = meta['attention_lookup']
    _mapping(block, ('schema', 'per_tp'))
    if block['schema'] != SCHEMA or not isinstance(block['per_tp'], dict):
        raise ValueError('Unknown attention lookup schema')
    result = {}
    for key, record in block['per_tp'].items():
        if not ((type(key) is int and key > 0) or
                (type(key) is str and re.fullmatch('[1-9][0-9]*', key))):
            raise ValueError('Invalid attention lookup TP')
        tp = int(key)
        if tp in result:
            raise ValueError('Repeated attention lookup TP')
        _mapping(record, ('semantic', 'semantic_sha256', 'measurement', 'completed', 'attention_sha256'))
        validate_semantic(record['semantic'])
        if record['semantic']['profile_tp'] != tp:
            raise ValueError('Attention row identity belongs to another TP')
        if record['semantic_sha256'] not in (digest(record['semantic']), _digest(record['semantic'])):
            raise ValueError('Attention semantic digest changed')
        identity = record['measurement']
        _mapping(identity, ('measurement_protocol', 'measurement_sha256'))
        _sha(identity['measurement_sha256'])
        marker = protocol_digest(identity['measurement_protocol'])
        if marker is not None and marker != record['semantic_sha256']:
            raise ValueError('Attention measurement does not bind its semantics')
        if not identity['measurement_protocol']:
            raise ValueError('Missing attention measurement method')
        if type(record['completed']) is not bool:
            raise ValueError('Invalid attention completion state')
        if record['completed']:
            _sha(record['attention_sha256'])
        elif record['attention_sha256'] is not None:
            raise ValueError('Incomplete attention contract has a seal')
        result[tp] = record
    return result


def check_rows(path, semantic_sha=None, measurement=None):
    path = Path(path)
    if not path.exists():
        if semantic_sha is not None:
            raise ValueError('Missing contracted attention table')
        return
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError('Invalid attention table header')
        fields = {'measurement_protocol', 'measurement_sha256'}
        stamped = fields.intersection(reader.fieldnames)
        if stamped and stamped != fields:
            raise ValueError('Incomplete historical attention measurement identity')
        count = 0
        identities = set()
        for row in reader:
            count += 1
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Malformed attention table row')
            if not stamped:
                continue
            marker = protocol_digest(row.get('measurement_protocol') or '')
            if marker != semantic_sha:
                raise ValueError('Attention row semantic marker disagrees with metadata')
            if measurement is not None and any(row.get(k) != v for k, v in measurement.items()):
                raise ValueError('Attention row acquisition identity differs')
            if semantic_sha is not None:
                _sha(row.get('measurement_sha256'))
                identities.add((row['measurement_protocol'], row['measurement_sha256']))
        if semantic_sha is not None and count == 0:
            raise ValueError('Empty contracted attention table')
        if semantic_sha is not None and stamped and len(identities) != 1:
            raise ValueError('Mixed contracted row acquisition identities')


def check_skew(folder, semantic_sha, *, allow_pending=False):
    check_rows(Path(folder) / 'skew.csv', semantic_sha)
    if semantic_sha is not None:
        from .bundle_metadata import read_tp
        state = read_tp(folder, 'skew')
        if state is None:
            raise ValueError('Missing native skew completion record')
        if (not isinstance(state, dict) or type(state.get('completed')) is not bool
                or (state.get('attention_lookup_sha256') or
                    protocol_digest(state.get('measurement_protocol', ''))) != semantic_sha):
            raise ValueError('Incomplete or incompatible native skew record')
        if not state['completed'] and not allow_pending:
            raise ValueError('Incomplete native skew acquisition')
        check_rows(Path(folder) / 'skew.csv', semantic_sha,
                   {k: state.get(k) for k in ('measurement_protocol', 'measurement_sha256')})
        if state['completed'] and state.get('skew_sha256') != checksum(Path(folder) / 'skew.csv'):
            raise ValueError('Native skew table seal changed')


def resolve(root, config, architecture, *, meta=None):
    """Validate every present TP and return its sealed semantic record."""
    root = Path(root)
    meta = metadata(root) if meta is None else meta
    entries = records(meta)
    folders = {int(p.name[2:]): p for p in root.glob('tp*')
               if p.is_dir() and re.fullmatch('tp[1-9][0-9]*', p.name)}
    if set(entries) - set(folders):
        raise ValueError('Attention contract has no TP folder')
    for tp, folder in folders.items():
        record = entries.get(tp)
        if record is None:
            check_rows(folder / 'attention.csv')
            check_rows(folder / 'skew.csv')
            continue
        validate_semantic(record['semantic'], config, architecture)
        if not record['completed']:
            raise ValueError('Incomplete attention acquisition; resume before consuming')
        table = folder / 'attention.csv'
        check_rows(table, record['semantic_sha256'], record['measurement'])
        if checksum(table) != record['attention_sha256']:
            raise ValueError('Attention table seal changed')
    return entries


def cap_for(entries, tp, config):
    from .stack import probe_key_saturation, text_config
    return (entries[tp]['semantic']['key_saturation'] if tp in entries else
            probe_key_saturation(text_config(config or {})))


def semantic_for_table(path, config, architecture):
    path = Path(path)
    if not re.fullmatch('tp[1-9][0-9]*', path.parent.name):
        raise ValueError('Expected a per-TP attention table')
    entries = resolve(path.parent.parent, config, architecture)
    tp = int(path.parent.name[2:])
    return entries.get(tp), cap_for(entries, tp, config)


def _save(root, tp, record):
    from .bundle_metadata import update_bundle
    def publish(meta):
        entries = records(meta)
        entries[tp] = record
        meta['attention_lookup'] = dict(schema=SCHEMA, per_tp=entries)
    update_bundle(root, publish)


def begin(folder, semantic, measurement, *, replace=False):
    """Mark pending before accepting rows; never bless old timings on refresh."""
    folder = Path(folder)
    if not re.fullmatch('tp[1-9][0-9]*', folder.name):
        raise ValueError('Expected a per-TP acquisition folder')
    validate_semantic(semantic)
    tp, root = int(folder.name[2:]), folder.parent
    if semantic['profile_tp'] != tp:
        raise ValueError('Requested attention contract belongs to another TP')
    meta = metadata(root)
    entries = records(meta)
    previous = entries.get(tp)
    sha = digest(semantic)
    _mapping(measurement, ('measurement_protocol', 'measurement_sha256'))
    _sha(measurement['measurement_sha256'])
    marker = protocol_digest(measurement['measurement_protocol'])
    if marker is not None and marker != sha:
        raise ValueError('Requested attention measurement lacks semantic identity')
    if not replace:
        if previous is not None:
            if (semantic_facts(previous['semantic']) != semantic_facts(semantic)
                    or previous['measurement'] != measurement):
                raise ValueError('Attention resume geometry/measurement changed; remeasure')
            if previous['completed'] and checksum(folder / 'attention.csv') != previous['attention_sha256']:
                raise ValueError('Attention resume seal changed')
        if (folder / 'attention.csv').exists():
            if previous is None:
                raise ValueError('Existing attention is uncontracted; remeasure in a fresh root')
            check_rows(folder / 'attention.csv', sha, measurement)
    entries[tp] = dict(semantic=semantic, semantic_sha256=sha, measurement=measurement,
                       completed=False, attention_sha256=None)
    _save(root, tp, entries[tp])


def complete(folder):
    folder = Path(folder)
    root, tp = folder.parent, int(folder.name[2:])
    meta = metadata(root)
    entries = records(meta)
    record = entries[tp]
    path = folder / 'attention.csv'
    check_rows(path, record['semantic_sha256'], record['measurement'])
    record.update(completed=True, attention_sha256=checksum(path))
    _save(root, tp, record)


def acquire(llm, config, architecture, tp):
    """Collect only plain serialized facts, agree on every worker."""
    if config.get('model_type') not in ('minimax_m3_vl', 'minimax_m3'):
        return None
    values = llm.collective_rpc('attention_lookup_contract', args=(tp,))
    if not values or any(value != values[0] for value in values):
        raise ValueError('Worker attention lookup contracts disagree')
    value = validate_semantic(values[0], config, architecture)
    if value['profile_tp'] != tp:
        raise ValueError('Worker attention profile TP differs')
    return value
