"""Declarative imported-function/plain-method runtime boundaries.

No family dispatch or replacement kernels. Wrappers preserve arguments,
outputs and the original callable, adding one record_function scope only.
Targets resolve from loaded module globals or live module-owned object paths.
All changes are temporary and restored, including inherited descriptors.
"""
from contextlib import contextmanager
from functools import wraps
import inspect
import sys
import threading

from .method_scopes import RuntimeMethodScopes, base
from .capture_scope_contract import normalize_callable

_MISSING = object()


def _read_attribute(obj, attribute):
    """Distinguish an absent alternative from a failing live descriptor."""
    descriptor = inspect.getattr_static(obj, attribute, _MISSING)
    try:
        return getattr(obj, attribute)
    except AttributeError as error:
        if descriptor is _MISSING:
            return _MISSING
        raise ValueError('Declared callable descriptor failed: ' + attribute) from error
    except Exception as error:
        raise ValueError('Declared callable attribute failed: ' + attribute) from error


def _identity(function):
    return (id(getattr(function, '__self__', None)), id(getattr(function, '__func__', function)))


def _has_class(module, name):
    return any(cls.__name__ == name for cls in type(module).__mro__)


class DeclaredCallableScopes(RuntimeMethodScopes):
    """Scope exact interfaces listed in ``declarations`` outside timed setup.

    Function example: {name: Core, kind: function, module: package.source,
    attribute: imported_function, within: AttentionClass}.

    Method example: {name: Core, kind: method, owner: AttentionBaseClass,
    path: impl.forward_mqa}. ``owner`` selects subclasses through actual MRO,
    not by a guessed model family. A shared implementation dispatches using
    the current runtime owner. Unavailable alternatives are reported, not
    counted as zero timings. Required positive measurements are enforced by
    the caller's selected catalog/coverage contract, not by every alternative.
    """
    def __init__(self, roots, *, declarations=(), **kwargs):
        super().__init__(roots, **kwargs)
        self.declarations = [normalize_callable(value) for value in declarations]
        if len({value['name'] for value in self.declarations}) != len(self.declarations):
            raise ValueError('Duplicate callable context name')
        self._declared_targets = []
        self._callable_patches = []
        self._callable_counts = {}
        self._resolved_declarations = set()
        self._resolution = []
        self._resolve_targets()

    def _resolve_targets(self):
        by_target = {}
        for index, declaration in enumerate(self.declarations):
            name = declaration.get('name')
            self._callable_counts[index] = 0
            candidates = []
            reasons = []
            if declaration.get('kind') == 'function':
                owner = sys.modules.get(declaration['module'])
                if owner is not None:
                    candidates.append((owner, declaration['attribute'], None))
                else:
                    reasons.append('module_not_loaded')
            elif declaration.get('kind') == 'method':
                owner_class, path = declaration.get('owner'), declaration.get('path')
                fields = path.split('.')
                owners = 0
                for module in self._modules.values():
                    if not _has_class(module, owner_class):
                        continue
                    owners += 1
                    obj = module
                    for field in fields[:-1]:
                        obj = _read_attribute(obj, field)
                        if obj is _MISSING or obj is None:
                            reasons.append('object_path_unavailable')
                            break
                    if obj is not _MISSING and obj is not None:
                        candidates.append((obj, fields[-1], id(module)))
                if not owners:
                    reasons.append('owner_not_present')
            for obj, attribute, owner_id in candidates:
                function = _read_attribute(obj, attribute)
                if function is _MISSING:
                    reasons.append('attribute_missing')
                    continue
                if not callable(function):
                    reasons.append('target_not_callable')
                    continue
                interfaces = (function, getattr(function, '__call__', None))
                if any(check(interface) for interface in interfaces for check in
                       (inspect.iscoroutinefunction, inspect.isgeneratorfunction, inspect.isasyncgenfunction)):
                    raise TypeError('Declared scopes require synchronous callables')
                if id(obj) in self._method_modules and attribute in self._method_names:
                    raise ValueError('A callable declaration conflicts with common method-context instrumentation')
                descriptor = inspect.getattr_static(obj, attribute, _MISSING)
                namespace = getattr(obj, '__dict__', None)
                if (not isinstance(namespace, dict)
                        or (descriptor is not _MISSING and inspect.isdatadescriptor(descriptor))):
                    raise TypeError('Declared targets need a mutable instance namespace without a data descriptor')
                key = (id(obj), attribute)
                target = by_target.get(key)
                if target is None:
                    target = dict(object=obj, attribute=attribute, original=function,
                                  identity=_identity(function), descriptor=descriptor,
                                  namespace=namespace, rules=[])
                    by_target[key] = target
                elif target['identity'] != _identity(function):
                    raise ValueError('Declared target identity changed during resolution')
                rule = (index, owner_id)
                if rule in target['rules']:
                    raise ValueError('Duplicate callable target declaration')
                target['rules'].append(rule)
                if 'context' in declaration:
                    from .callable_context import bind_signature
                    target.setdefault('context_signatures', {})[index] = bind_signature(
                        function, declaration['context'])
                self._resolved_declarations.add(index)
            self._resolution.append(dict(name=name, resolved=index in self._resolved_declarations,
                                         reasons=sorted(set(reasons))))
        self._declared_targets = list(by_target.values())

    def _select(self, target):
        stack = self._stack()
        selected = []
        for index, owner_id in target['rules']:
            declaration = self.declarations[index]
            depths = [depth for depth, (_, module_id, _) in enumerate(stack)
                      if module_id == owner_id] if owner_id is not None else [-1]
            within = declaration.get('within')
            if within is not None:
                names = [within] if isinstance(within, str) else list(within)
                contextual = [depth for depth, (_, module_id, _) in enumerate(stack)
                              if module_id in self._modules and
                              any(_has_class(self._modules[module_id], name) for name in names)]
                if not contextual:
                    continue
                depths = [max(depths + contextual)] if depths else []
            if depths:
                selected.append((max(depths), index, owner_id))
        if not selected:
            # A declared bound method must not invent its owning module's
            # ancestry if called independently of that module's invocation.
            if any(owner_id is not None for _, owner_id in target['rules']):
                raise ValueError('Declared method has no active runtime owner')
            return None
        deepest = max(value[0] for value in selected)
        best = [value for value in selected if value[0] == deepest]
        if len(best) != 1:
            raise ValueError('Ambiguous callable ownership declarations')
        return best[0][1:]

    @contextmanager
    def _semantic_scope(self, index, owner_id, context_data=None):
        from torch.profiler import record_function
        declaration = self.declarations[index]
        name = declaration['name']
        stack = self._stack()
        if any(self._calls[call_id].get('semantic_name') == name for call_id, _, _ in stack):
            raise ValueError('Nested calls claim the same semantic operation')
        with self._lock:
            call_id = len(self._calls)
            self._calls.append(dict(id=call_id, parent=stack[-1][0] if stack else None,
                forward=self._local.forward, module_id=-(call_id + 1), class_name=name,
                summary_name=name, thread_id=threading.get_native_id(), paths=[], completed=False,
                scope_kind='callable', semantic_name=name, declaration=index,
                runtime_owner_id=owner_id, normalization_context_only=True))
            if context_data is not None:
                self._calls[-1].update(context_data)
            self._callable_counts[index] += 1
        scope = record_function(base.CALL_PREFIX + str(call_id))
        scope.__enter__()
        stack.append((call_id, -(call_id + 1), scope))
        try:
            yield
        finally:
            if not stack or stack[-1][0] != call_id:
                raise ValueError('Unbalanced declared callable scope')
            stack.pop()
            scope.__exit__(None, None, None)
            self._calls[call_id]['completed'] = True

    def _invoke(self, target, args, kwargs):
        original = target['original']
        if getattr(self._local, 'forward', None) is None:
            return original(*args, **kwargs)
        chosen = self._select(target)
        if chosen is None:
            return original(*args, **kwargs)
        context_data = None
        declaration = self.declarations[chosen[0]]
        if 'context' in declaration:
            from .callable_context import extract_context
            context_data = extract_context(target['context_signatures'][chosen[0]],
                                           declaration['context'], args, kwargs)
        with self._semantic_scope(*chosen, context_data=context_data):
            return original(*args, **kwargs)

    def __enter__(self):
        super().__enter__()
        try:
            for target in self._declared_targets:
                obj, attribute = target['object'], target['attribute']
                if (_identity(getattr(obj, attribute)) != target['identity'] or
                        inspect.getattr_static(obj, attribute, _MISSING) is not target['descriptor']):
                    raise ValueError('Declared callable changed before capture')
                namespace = target['namespace']
                if getattr(obj, '__dict__', None) is not namespace:
                    raise ValueError('Declared target namespace changed before capture')
                present, saved = attribute in namespace, namespace.get(attribute)
                def wrapper(target):
                    @wraps(target['original'])
                    def invoke(*args, **kwargs):
                        return self._invoke(target, args, kwargs)
                    return invoke
                installed = wrapper(target)
                # Only ordinary instance/module attributes are supported.
                # Writing their verified namespace avoids user __setattr__
                # side effects and cannot mutate an inherited descriptor.
                self._callable_patches.append((obj, attribute, namespace, present, saved, installed))
                namespace[attribute] = installed
                if getattr(obj, attribute) is not installed:
                    raise ValueError('Declared target does not expose its temporary wrapper')
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        errors = []
        for obj, attribute, namespace, present, saved, installed in reversed(self._callable_patches):
            if getattr(obj, '__dict__', None) is not namespace or namespace.get(attribute) is not installed:
                errors.append('Declared callable was replaced during capture; foreign value preserved')
                continue
            try:
                if present:
                    namespace[attribute] = saved
                else:
                    del namespace[attribute]
            except BaseException as error:
                errors.append(str(error))
        self._callable_patches.clear()
        try:
            super().__exit__(exc_type, exc, tb)
        finally:
            if errors:
                raise ValueError('; '.join(errors))

    def metadata(self):
        result = super().metadata()
        result['callable_audit'] = dict(
            declarations=self.declarations, calls=dict(self._callable_counts),
            resolved=sorted(self._resolved_declarations),
            inactive=[index for index, count in self._callable_counts.items() if not count],
            resolution=self._resolution,
            targets=[dict(attribute=target['attribute'], object_id=id(target['object']),
                          callable_identity=target['identity'],
                          function_module=getattr(target['original'], '__module__', None),
                          function_qualname=getattr(target['original'], '__qualname__', None))
                     for target in self._declared_targets])
        return result
