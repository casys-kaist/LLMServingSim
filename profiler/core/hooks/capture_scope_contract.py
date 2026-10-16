"""Serializable capture contexts, independent of category timing bindings.

The full architecture selects instrumentation contexts once. Every category
receives the same projection, so an exclusion naming an operator remains
effective when that operator's own timing belongs to another category.
"""

DEFAULT_METHODS = ('embed_input_ids', 'compute_logits')


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
    operators = set()
    for spec in (catalog or {}).values():
        if 'operator' in spec:
            if 'vllm' in spec:
                raise ValueError('A binding cannot declare operator and vllm together')
            operators.update(_names([spec['operator']]))
    if contexts is None:
        contexts = dict(schema=1, methods=list(DEFAULT_METHODS), operators=sorted(operators))
    if not isinstance(contexts, dict) or set(contexts) != {'schema','methods','operators'}:
        raise ValueError('Capture contexts require exactly schema/methods/operators')
    if type(contexts['schema']) is not int or contexts['schema'] != 1:
        raise ValueError('Unsupported capture-context schema')
    result = dict(schema=1, methods=_names(contexts['methods'], methods=True),
                  operators=_names(contexts['operators']))
    if not operators <= set(result['operators']):
        raise ValueError('Full capture contexts omit an operator required by the category')
    return result


def contexts_from_architecture(arch):
    """Project every catalog group's declared operator, including alternatives.

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
