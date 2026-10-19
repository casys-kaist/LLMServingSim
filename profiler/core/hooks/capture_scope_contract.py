"""Serializable capture contexts, independent of category timing bindings.

The full architecture selects instrumentation contexts once. Every category
receives the same projection, so an exclusion naming an operator remains
effective when that operator's own timing belongs to another category.
"""

DEFAULT_METHODS = ('embed_input_ids', 'compute_logits')


def normalize_callable(declaration):
    """Validate one explicit loaded-function or live-object method boundary.

    Resolution never imports a module or substitutes an implementation. A
    declaration may be inactive on a different installed vLLM version; the
    collector reports that state instead of manufacturing a timing.
    """
    if not isinstance(declaration, dict):
        raise ValueError('Callable declaration must be an explicit mapping')
    kind = declaration.get('kind')
    fields = ({'module', 'attribute'} if kind == 'function' else
              {'owner', 'path'} if kind == 'method' else None)
    if fields is None:
        raise ValueError('Callable kind must be function or method')
    required = {'name', 'kind'} | fields
    if not required <= set(declaration) or set(declaration) - required - {'within', 'context'}:
        raise ValueError('Callable declaration has missing or incompatible target fields')
    result = dict(declaration)
    _names([result['name']])
    for field in fields:
        value = result[field]
        parts = value.split('.') if isinstance(value, str) else []
        if (not parts or any(not part.isidentifier() or part.startswith('__') for part in parts)
                or (field in ('owner', 'attribute') and len(parts) != 1)):
            raise ValueError('Callable target fields must be exact identifier paths')
    if 'within' in result:
        value = result['within']
        if value is None:
            del result['within']
        elif isinstance(value, str):
            _names([value], methods=True)
        else:
            result['within'] = _names(value, methods=True)
            if not result['within']:
                raise ValueError('Callable within alternatives cannot be empty')
    if 'context' in result:
        from .callable_context import normalize_context
        result['context'] = normalize_context(result['context'])
    return result


def _callables(values):
    if not isinstance(values, (list, tuple)):
        raise ValueError('Callable contexts must be an explicit list')
    result = {}
    for value in values:
        declaration = normalize_callable(value)
        name = declaration['name']
        if name in result:
            raise ValueError('Duplicate callable context name')
        result[name] = declaration
    return [result[name] for name in sorted(result)]


def _names(values, *, methods=False):
    if not isinstance(values, (list, tuple)):
        raise ValueError('Scope names must be an explicit list')
    if any(not isinstance(name, str) or not name or '*' in name or '(' in name
           or (methods and not name.isidentifier()) for name in values):
        raise ValueError('Scope names must be exact nonempty names')
    if len(values) != len(set(values)):
        raise ValueError('Duplicate capture scope name')
    return sorted(values)


def normalize_contexts(contexts=None, catalog=None):
    """Validate a full-architecture contract against the requested slice.

    Omitted contracts derive from the supplied catalog for standalone RPC
    callers. Normal host sweeps always pass the full architecture projection.
    No catalog binding is invented, filled with zero, or enabled by this API.
    """
    operators, callable_names, declarations = set(), set(), {}
    for spec in (catalog or {}).values():
        if len(set(spec) & {'vllm', 'operator', 'callable'}) != 1:
            raise ValueError('A binding requires exactly one vllm, operator or callable selector')
        if 'operator' in spec:
            operators.update(_names([spec['operator']]))
        if 'callable' in spec:
            value = spec['callable']
            if isinstance(value, dict):
                value = normalize_callable(value)
                name = value['name']
                if name in declarations and declarations[name] != value:
                    raise ValueError('Callable name has inconsistent target declarations')
                declarations[name] = value
            else:
                name = value
                _names([name])
            callable_names.add(name)
    if contexts is None:
        contexts = dict(schema=1, methods=list(DEFAULT_METHODS), operators=sorted(operators))
        if callable_names:
            if callable_names != set(declarations):
                raise ValueError('Callable bindings require full declared capture contexts')
            contexts.update(schema=3 if any('context' in value for value in declarations.values()) else 2,
                            callables=list(declarations.values()))
    if not isinstance(contexts, dict) or type(contexts.get('schema')) is not int:
        raise ValueError('Unsupported capture-context schema')
    schema = contexts['schema']
    fields = {'schema', 'methods', 'operators'} | ({'callables'} if schema in (2, 3) else set())
    if schema not in (1, 2, 3) or set(contexts) != fields:
        raise ValueError('Invalid capture-context schema fields')
    result = dict(schema=schema, methods=_names(contexts['methods'], methods=True),
                  operators=_names(contexts['operators']))
    available = {}
    if schema in (2, 3):
        result['callables'] = _callables(contexts['callables'])
        if not result['callables']:
            raise ValueError('Empty callable contexts must use schema 1')
        if any('context' in value for value in result['callables']) != (schema == 3):
            raise ValueError('Argument/layout contexts require capture schema 3 exactly')
        available = {value['name']: value for value in result['callables']}
        if set(available) & set(result['operators']):
            raise ValueError('Callable names collide with native operator contexts')
    if not callable_names <= set(available):
        raise ValueError('Full capture contexts omit a callable required by the category')
    if any(available[name] != value for name, value in declarations.items()):
        raise ValueError('Capture context target differs from its catalog declaration')
    if not operators <= set(result['operators']):
        raise ValueError('Full capture contexts omit an operator required by the category')
    return result


def contexts_from_architecture(arch):
    """Project every catalog group's declared execution boundary and alternatives.

    Contexts are independent of the category currently acquiring timings:
    exclusions still work when an operator's cost belongs to another group.
    """
    whole = {}
    for group in type(arch.catalog).model_fields:
        for name, entry in (getattr(arch.catalog, group) or {}).items():
            if name in whole:
                raise ValueError('Duplicate canonical name in full architecture')
            whole[name] = (entry.model_dump(exclude_none=True)
                           if hasattr(entry, 'model_dump') else dict(entry))
    return normalize_contexts(catalog=whole)
