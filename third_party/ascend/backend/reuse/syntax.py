"""Parse each JIT function once per analysis.

Analysis passes only read trees; `transform` deep-copies before rewriting.
Outside a scope this is exactly `fn.parse()`, so patched sources still apply.
"""
from contextlib import contextmanager
from contextvars import ContextVar

_trees = ContextVar("ascend_reuse_trees", default=None)


@contextmanager
def parse_scope():
    token = _trees.set({})
    try:
        yield
    finally:
        _trees.reset(token)


def parse(fn):
    trees = _trees.get()
    if trees is None:
        return fn.parse()
    entry = trees.get(id(fn))
    if entry is None or entry[0] is not fn:
        # Keep fn alive with its tree so id(fn) stays unique within the scope.
        entry = trees[id(fn)] = (fn, fn.parse())
    return entry[1]
