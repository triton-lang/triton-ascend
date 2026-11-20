"""Cheap live dependency stamps; AST traversal happens only on source changes."""
import ast
import builtins
import types
from dataclasses import dataclass
from weakref import ref

from ..identity import digest
from ..operations import operation, resolve

_standard_capture_scope = None


def path(node):
    if isinstance(node, ast.Name):
        return (node.id, )
    if isinstance(node, ast.Attribute):
        base = path(node.value)
        return (*base, node.attr) if base else ()
    return ()


def lookup(fn, route):
    name = route[0]
    nonlocal_names = fn.fn.__code__.co_freevars
    if name in nonlocal_names and fn.fn.__closure__:
        value = fn.fn.__closure__[nonlocal_names.index(name)].cell_contents
    else:
        value = fn.__globals__.get(name, vars(builtins).get(name))
    for name in route[1:]:
        value = vars(value).get(name) if isinstance(value, types.ModuleType) else None
    return value


def capture_token(value):
    # Immutable scalars and module bindings need no JSON encoding/hash. Module
    # member calls are separately tracked by lookup records. Keeping identity
    # also detects replacement by another module with the same name.
    if value is None or type(value) in (bool, int, str):
        return (type(value).__name__, value)
    if isinstance(value, types.ModuleType):
        return ("module", id(value), value.__name__)
    from ..snapshot import encode, SourceNotSerializable

    try:
        return digest(encode(value))
    except SourceNotSerializable:
        return ("identity", id(value))


def capture_value(fn, name):
    method = fn.get_capture_scope
    if (_standard_capture_scope is not None and getattr(method, "__func__", None) is _standard_capture_scope):
        names = fn.fn.__code__.co_freevars
        if name in names and fn.fn.__closure__:
            return fn.fn.__closure__[names.index(name)].cell_contents
        return fn.__globals__.get(name)
    # Frozen/custom JIT capture scopes may intentionally differ from closure
    # cells. Preserve their method instead of assuming standard JIT semantics.
    return method().get(name)


@dataclass(frozen=True)
class WeakSourceGuard:
    records: tuple
    captures: tuple

    def is_current(self):
        for weak, source, references in self.records:
            fn = weak()
            if fn is None or fn.src != source or any(
                    id(lookup(fn, route)) != identity for route, identity in references):
                return False
        for weak, name, token in self.captures:
            fn = weak()
            if fn is None or capture_token(capture_value(fn, name)) != token:
                return False
        return True


@dataclass(frozen=True)
class SourceStamp:
    records: tuple
    token: str
    captures: tuple = ()

    def matches(self):
        return all(fn.src == source and all(lookup(fn, route) is value
                                            for route, value in references)
                   for fn, source, references in self.records) and all(
                       capture_token(capture_value(fn, name)) == token for fn, name, token in self.captures)

    def weak_guard(self):
        return WeakSourceGuard(
            tuple((ref(fn), source, tuple((route, id(value))
                                          for route, value in references))
                  for fn, source, references in self.records),
            tuple((ref(fn), name, token) for fn, name, token in self.captures))


def source_stamp(fn):
    from triton.runtime.jit import JITFunction

    global _standard_capture_scope
    _standard_capture_scope = getattr(JITFunction.__base__, "get_capture_scope", None)

    records, captures, visited = [], [], set()

    def visit(current):
        if id(current) in visited:
            return
        visited.add(id(current))
        scope = current.get_capture_scope()
        local_names = set(current.arg_names) | {
            n.id
            for n in ast.walk(current.parse())
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        for name in sorted(
            {n.id
             for n in ast.walk(current.parse())
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - local_names):
            if name in scope:
                captures.append((current, name, capture_token(scope[name])))
        references = []
        for node in ast.walk(current.parse()):
            if not isinstance(node, ast.Call):
                continue
            obj = resolve(node.func, scope)
            route = path(node.func)
            if route and obj is not None:
                references.append((route, obj))
            if isinstance(obj, JITFunction) and operation(obj) is None:
                visit(obj)
        records.append((current, current.src, tuple(references)))

    visit(fn)
    token = digest((tuple(
        (f.cache_key, src) for f, src, _ in records), tuple((name, token) for _, name, token in captures)))
    return SourceStamp(tuple(records), token, tuple(captures))
