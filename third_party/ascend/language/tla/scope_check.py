# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Static scope check for TLA cube and vector regions.

A name assigned inside ``with tla.scope.cube()`` must not be read inside
``with tla.scope.simd()``, and the other way around. Names assigned outside
either region, including on-chip buffers, may be used in both. Reassigning a
name outside a region, or assigning it again in the region that reads it,
clears the previous core binding.
"""

import ast

__all__ = ["bind_name", "check_scope_body", "install_scope_tracking"]

_CORE_MODES = ("vector", "cube")


def bind_name(defs, mode, name):
    """Record ``name`` as born in ``mode``, or drop it when ``mode`` is outside a core."""
    if mode in _CORE_MODES:
        defs[name] = mode
    else:
        defs.pop(name, None)


def _use(name, mode, defs):
    if mode not in _CORE_MODES:
        return
    born = defs.get(name)
    if born in _CORE_MODES and born != mode:
        raise ValueError(f"scope failure: variable '{name}' is defined in {born} scope "
                         f"and cannot be used in {mode} scope")


def _store_names(target):
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names = []
        for elt in target.elts:
            names.extend(_store_names(elt))
        return names
    if isinstance(target, ast.Starred):
        return _store_names(target.value)
    return []


def _mode_of_with(stmt):
    """Return ``vector`` or ``cube`` when ``stmt`` is a TLA core-mode scope."""
    if not isinstance(stmt, ast.With) or len(stmt.items) != 1:
        return None
    call = stmt.items[0].context_expr
    if not isinstance(call, ast.Call):
        return None
    func = call.func
    if isinstance(func, ast.Attribute):
        if func.attr == "simd":
            return "vector"
        if func.attr == "cube":
            return "cube"
    return None


def _walk(node, mode, defs):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
        return
    if isinstance(node, ast.With):
        check_stmt(node, mode, defs)
        return
    if isinstance(node, ast.NamedExpr):
        _walk(node.value, mode, defs)
        for name in _store_names(node.target):
            bind_name(defs, mode, name)
        return
    if isinstance(node, ast.Name):
        if isinstance(node.ctx, ast.Load):
            _use(node.id, mode, defs)
        return
    for child in ast.iter_child_nodes(node):
        _walk(child, mode, defs)


def check_stmt(stmt, mode, defs):
    """Check one statement and record names it assigns in ``defs``."""
    nested = _mode_of_with(stmt)
    if nested is not None:
        item = stmt.items[0]
        _walk(item.context_expr, mode, defs)
        if item.optional_vars is not None:
            for name in _store_names(item.optional_vars):
                bind_name(defs, mode, name)
        check_scope_body(stmt.body, nested, defs)
        return
    if isinstance(stmt, ast.Assign):
        _walk(stmt.value, mode, defs)
        for target in stmt.targets:
            _walk(target, mode, defs)
            for name in _store_names(target):
                bind_name(defs, mode, name)
        return
    if isinstance(stmt, ast.AnnAssign):
        if stmt.value is not None:
            _walk(stmt.value, mode, defs)
        _walk(stmt.target, mode, defs)
        if stmt.annotation is not None:
            _walk(stmt.annotation, mode, defs)
        for name in _store_names(stmt.target):
            bind_name(defs, mode, name)
        return
    if isinstance(stmt, ast.AugAssign):
        for name in _store_names(stmt.target):
            _use(name, mode, defs)
        _walk(stmt.target, mode, defs)
        _walk(stmt.value, mode, defs)
        for name in _store_names(stmt.target):
            bind_name(defs, mode, name)
        return
    if isinstance(stmt, ast.For):
        _walk(stmt.iter, mode, defs)
        for name in _store_names(stmt.target):
            bind_name(defs, mode, name)
        check_scope_body(stmt.body, mode, defs)
        check_scope_body(stmt.orelse, mode, defs)
        return
    if isinstance(stmt, (ast.If, ast.While)):
        _walk(stmt.test, mode, defs)
        check_scope_body(stmt.body, mode, defs)
        check_scope_body(stmt.orelse, mode, defs)
        return
    if isinstance(stmt, ast.With):
        for item in stmt.items:
            _walk(item.context_expr, mode, defs)
            if item.optional_vars is not None:
                for name in _store_names(item.optional_vars):
                    bind_name(defs, mode, name)
        check_scope_body(stmt.body, mode, defs)
        return
    _walk(stmt, mode, defs)


def check_scope_body(stmts, mode, defs):
    """Check ``stmts`` as a ``mode`` region (``vector``, ``cube``, or ``None``)."""
    if stmts is None:
        return
    if not isinstance(stmts, list):
        stmts = [stmts]
    for stmt in stmts:
        check_stmt(stmt, mode, defs)


def install_scope_tracking(generator):
    """Keep ``defs`` aligned with assignments the code generator performs.

    The AST walk records names inside a core region, including ``static_range``
    induction variables, which never go through ``set_value``. Assignments
    outside a region go through ``set_value`` and drop that name's core binding.
    Reads go through ``dereference_name`` and are rejected when the current
    region is the other core.
    """
    if getattr(generator, "_tla_scope_installed", False):
        return
    generator._tla_scope_installed = True
    generator._tla_scope_stack = []
    generator._tla_scope_defs = {}
    original_set = generator.set_value
    original_deref = generator.dereference_name

    def set_value(name, value):
        stack = generator._tla_scope_stack
        bind_name(generator._tla_scope_defs, stack[-1] if stack else None, name)
        return original_set(name, value)

    def dereference_name(name):
        stack = generator._tla_scope_stack
        if stack:
            _use(name, stack[-1], generator._tla_scope_defs)
        return original_deref(name)

    generator.set_value = set_value
    generator.dereference_name = dereference_name
