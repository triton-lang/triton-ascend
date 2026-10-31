"""Dispatch table for TLA ``with`` statement context managers.

The Triton compiler patch merges ``TLA_WITH_DISPATCH`` into
``triton.compiler.code_generator.WITH_DISPATCH`` so that
``with tla.scope.simd():`` is handled by ``handle_scope_with`` (inline
``scope.scope`` emission; bishengir ``outline-scope`` extracts the VF).
"""

from .code_generator import handle_scope_with
from .scope import vector, simd, cube

__all__ = ["TLA_WITH_DISPATCH"]

# Registry of 'with' statement handlers for the TLA namespace, keyed by the
# context-manager class itself.
TLA_WITH_DISPATCH = {
    vector: handle_scope_with,
    simd: handle_scope_with,
    cube: handle_scope_with,
}
