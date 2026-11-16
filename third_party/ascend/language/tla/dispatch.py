"""Dispatch handlers for TLA ``with tla.scope.*`` statements.

``register_with_dispatch()`` merges ``TLA_WITH_DISPATCH`` into the community
``WITH_DISPATCH`` registry; it runs inside ``setup_unified_builder`` (mounted
in ``CodeGenerator.__init__`` by the patch) before any kernel compiles.
``triton.compiler`` is imported lazily because ``tla`` loads during
``triton.language`` init, when a module-level import would deadlock.
"""

from .code_generator import handle_scope_with
from .scope import vector, simd, cube

__all__ = ["TLA_WITH_DISPATCH", "register_with_dispatch"]

# 'with' handlers keyed by scope class.
TLA_WITH_DISPATCH = {
    vector: handle_scope_with,
    simd: handle_scope_with,
    cube: handle_scope_with,
}


def register_with_dispatch():
    # Called by setup_unified_builder before any kernel compiles; the merge
    # replaces the old build-time patch of community code_generator.py.
    from triton.compiler import code_generator as _code_generator

    _code_generator.WITH_DISPATCH.update(TLA_WITH_DISPATCH)
