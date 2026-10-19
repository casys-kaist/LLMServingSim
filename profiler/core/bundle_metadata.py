"""Atomic, locked access to the single metadata document of a perf variant."""

import fcntl
import os
from pathlib import Path
import re
import tempfile

import yaml


class _UniqueLoader(getattr(yaml, 'CSafeLoader', yaml.SafeLoader)):
    pass


def _unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f'Duplicate metadata key: {key!r}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


class _CompactDumper(getattr(yaml, 'CSafeDumper', yaml.SafeDumper)):
    pass


def _compact_list(dumper, values):
    flow = all(isinstance(value, (str, int, float, bool, type(None))) for value in values)
    return dumper.represent_sequence('tag:yaml.org,2002:seq', values, flow_style=flow)


_CompactDumper.add_representer(list, _compact_list)
_CompactDumper.add_representer(tuple, _compact_list)


def read_bundle(root):
    """Read without creating files or changing historical metadata."""
    path = Path(root) / 'meta.yaml'
    if not path.exists():
        return {}
    value = yaml.load(path.read_text(), Loader=_UniqueLoader)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f'Expected metadata mapping in {path}')
    records = value.get('per_tp', {})
    if not isinstance(records, dict):
        raise ValueError(f'Expected per_tp metadata mapping in {path}')
    seen = set()
    for tp, record in records.items():
        if (type(tp) not in (int, str) or not str(tp).isascii() or not str(tp).isdigit()
                or int(tp) < 1 or str(int(tp)) != str(tp) or not isinstance(record, dict)
                or str(tp) in seen):
            raise ValueError(f'Invalid or duplicate per_tp metadata entry in {path}: {tp!r}')
        seen.add(str(tp))
    return value


def _tp_location(directory):
    directory = Path(directory)
    match = re.fullmatch(r'tp([1-9][0-9]*)', directory.name)
    if match is None:
        raise ValueError(f'Expected a tp<N> profile directory: {directory}')
    return directory.parent, match[1]


def read_tp(directory, key, default=None):
    root, tp = _tp_location(directory)
    records = read_bundle(root).get('per_tp') or {}
    record = records.get(tp, records.get(int(tp), {})) or {}
    return record.get(key, default)


def _atomic_yaml(path, value):
    text = yaml.dump(value, Dumper=_CompactDumper, sort_keys=False)
    previous = path.stat() if path.exists() else None
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.meta-', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if os.geteuid() == 0:
            owner = previous or path.parent.stat()
            os.chown(temporary, owner.st_uid, owner.st_gid)
        temporary.chmod(previous.st_mode & 0o777 if previous else 0o644)
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def update_bundle(root, mutator):
    """Lock, read current state, mutate it in place, and atomically publish it.

    The empty lock file is synchronization only. Its writable descriptor also
    supports NFS implementations that emulate flock with POSIX record locks.
    Never unlink it: replacing its inode would split concurrent writers' locks.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.meta.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if os.geteuid() == 0:
                path = root / 'meta.yaml'
                owner = path.stat() if path.exists() else root.stat()
                os.fchown(lock.fileno(), owner.st_uid, owner.st_gid)
            value = read_bundle(root)
            result = mutator(value)
            _atomic_yaml(root / 'meta.yaml', value)
            return result
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def update_tp(directory, key, mutator):
    """Replace one TP owner's value with mutator(current_or_None), under lock."""
    root, tp = _tp_location(directory)
    def apply(bundle):
        records = bundle.setdefault('per_tp', {})
        if int(tp) in records:
            if tp in records:
                raise ValueError(f'Duplicate TP metadata key: {tp}')
            records[tp] = records.pop(int(tp))
        record = records.setdefault(tp, {})
        value = mutator(record.get(key))
        record[key] = value
        return value
    return update_bundle(root, apply)


def write_tp(directory, key, value):
    return update_tp(directory, key, lambda _previous: value)
