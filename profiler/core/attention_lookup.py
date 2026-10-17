"""CPU-only, per-TP ownership of newly acquired attention coordinates.

Unmarked historical tables retain stack.probe_key_saturation. Native records
are authoritative only with matching row identities and a completed table seal.
"""
import csv
import hashlib
import json
from pathlib import Path
import re

import yaml

SCHEMA = 'native-attention-lookup-v1'
RULE = 'minimax-triton-total-selected-blocks-v1'
MARKER = '|native-attention-lookup-v1:'
INDEXER = 'vllm.models.minimax_m3.common.indexer.MiniMaxM3IndexerTritonImpl'
ATTEND = 'vllm.models.minimax_m3.common.sparse_attention.MiniMaxM3SparseTritonImpl'
SOURCES = {
    'nvidia/model.py': '2c241e8f8f1f983d6186ec8712910e7aa6b5181633ec835cb124a8337626bc6c',
    'common/indexer.py': '6ca7b3d10f7fdbe03830091064b8dbab8c60b149b5dea15d3de0fb804f820edc',
    'common/sparse_attention.py': '764eb3160b3d50541d739c4ce64db2520262b403c20fb37b3ffe8d0f1fd19c8a',
    'common/ops/index_topk.py': '1ca8557aed4260d2f4a742b9eda81d6f774e49d4b016ccc555dde40976c5a1a0',
    'common/ops/sparse_attn.py': '5db5d5e692044e54bb57418cbb7dcf5c9cc94f789eaccf128eec6fa8df86d967',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mapping(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError('Malformed attention lookup contract')


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Invalid attention lookup digest')
    return value


def protocol(base, semantic_sha=None):
    if semantic_sha is None:
        return base
    return base + MARKER + _sha(semantic_sha)


def _ordinary_protocols():
    from .measurement import PROTOCOL
    return (PROTOCOL, 'kineto-context-scoped-cuda-union-dummy-kv-query-v7')


def protocol_digest(value):
    if not isinstance(value, str):
        raise ValueError('Invalid attention row protocol')
    if 'native-attention-lookup' not in value:
        return None
    if value.count(MARKER) != 1:
        raise ValueError('Unknown native attention row protocol')
    base, recorded = value.split(MARKER)
    if base not in (*_ordinary_protocols(),
                    'kineto-context-scoped-dummy-kv-skew-per-forward-v9'):
        raise ValueError('Unknown native attention measurement method')
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
    _mapping(source, ('vllm_version', 'indexer_class', 'attend_class', 'sha256',
                     'topk_blocks', 'block_size', 'init_blocks', 'local_blocks',
                     'topk_width', 'indexer_dtype', 'kv_dtype', 'score_type'))
    if (source['vllm_version'] != '0.31.0' or source['indexer_class'] != INDEXER
            or source['attend_class'] != ATTEND or source['sha256'] != SOURCES
            or source['indexer_dtype'] != 'bf16' or source['kv_dtype'] != 'bfloat16'
            or source['score_type'] != 'max'):
        raise ValueError('Unreviewed native attention source/layout')
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
    path = Path(root) / 'meta.yaml'
    result = yaml.safe_load(path.read_text()) if path.exists() else {}
    if not isinstance(result, dict):
        raise ValueError('Invalid profile metadata')
    return result


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
        if record['semantic_sha256'] != digest(record['semantic']):
            raise ValueError('Attention semantic digest changed')
        identity = record['measurement']
        _mapping(identity, ('measurement_protocol', 'measurement_sha256'))
        _sha(identity['measurement_sha256'])
        if protocol_digest(identity['measurement_protocol']) != record['semantic_sha256']:
            raise ValueError('Attention measurement does not bind its semantics')
        if identity['measurement_protocol'] not in (
                protocol(base, record['semantic_sha256']) for base in _ordinary_protocols()):
            raise ValueError('Attention record uses another category protocol')
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
        count = 0
        identities = set()
        for row in reader:
            count += 1
            marker = protocol_digest(row.get('measurement_protocol') or '')
            if marker != semantic_sha:
                raise ValueError('Attention row semantic marker disagrees with metadata')
            if measurement is not None and any(row.get(k) != v for k, v in measurement.items()):
                raise ValueError('Attention row acquisition identity differs')
            if semantic_sha is not None:
                _sha(row.get('measurement_sha256'))
                identities.add((row['measurement_protocol'], row['measurement_sha256']))
                expected_bases = (('kineto-context-scoped-dummy-kv-skew-per-forward-v9',)
                                  if path.name == 'skew.csv' else _ordinary_protocols())
                if row['measurement_protocol'] not in (
                        protocol(base, semantic_sha) for base in expected_bases):
                    raise ValueError('Contracted row uses another category protocol')
        if semantic_sha is not None and count == 0:
            raise ValueError('Empty contracted attention table')
        if semantic_sha is not None and len(identities) != 1:
            raise ValueError('Mixed contracted row acquisition identities')


def check_skew(folder, semantic_sha, *, allow_pending=False):
    check_rows(Path(folder) / 'skew.csv', semantic_sha)
    if semantic_sha is not None:
        status = Path(folder) / 'skew.meta.yaml'
        if not status.exists():
            raise ValueError('Missing native skew completion record')
        state = yaml.safe_load(status.read_text())
        if (not isinstance(state, dict) or type(state.get('completed')) is not bool
                or protocol_digest(state.get('measurement_protocol', '')) != semantic_sha):
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


def _save(root, meta, entries):
    from .skew_calibration import atomic_yaml
    meta['attention_lookup'] = dict(schema=SCHEMA, per_tp=entries)
    Path(root).mkdir(parents=True, exist_ok=True)
    atomic_yaml(Path(root) / 'meta.yaml', meta)


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
    if protocol_digest(measurement['measurement_protocol']) != sha:
        raise ValueError('Requested attention measurement lacks semantic identity')
    if not replace:
        if previous is not None:
            if previous['semantic'] != semantic or previous['measurement'] != measurement:
                raise ValueError('Attention resume source/identity changed; remeasure')
            if previous['completed'] and checksum(folder / 'attention.csv') != previous['attention_sha256']:
                raise ValueError('Attention resume seal changed')
        if (folder / 'attention.csv').exists():
            if previous is None:
                raise ValueError('Existing attention is uncontracted; remeasure in a fresh root')
            check_rows(folder / 'attention.csv', sha, measurement)
    entries[tp] = dict(semantic=semantic, semantic_sha256=sha, measurement=measurement,
                       completed=False, attention_sha256=None)
    _save(root, meta, entries)


def complete(folder):
    folder = Path(folder)
    root, tp = folder.parent, int(folder.name[2:])
    meta = metadata(root)
    entries = records(meta)
    record = entries[tp]
    path = folder / 'attention.csv'
    check_rows(path, record['semantic_sha256'], record['measurement'])
    record.update(completed=True, attention_sha256=checksum(path))
    _save(root, meta, entries)


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
