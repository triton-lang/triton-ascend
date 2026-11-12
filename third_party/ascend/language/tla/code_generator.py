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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
Ascend-specific code generation handlers for 'with' statement context managers.
"""

__all__ = ["handle_scope_with"]


def handle_scope_with(generator, node):
    """Emit a core scope, including a named task on that core."""
    from triton.compiler.code_generator import enter_sub_region
    from triton.language.core import _unwrap_if_constexpr
    from .scope import _TaskScope

    item = node.items[0]
    if item.optional_vars is not None:
        raise ValueError("TLA scopes do not support 'as' bindings")
    spec = _unwrap_if_constexpr(generator.visit(item.context_expr))
    builder = generator.builder
    domain = getattr(generator, "_tla_pipeline", None)
    if domain is None and isinstance(spec, _TaskScope) and spec.stage is not None:
        raise ValueError("task stage requires an enclosing tla.pipeline loop")
    with enter_sub_region(generator):
        scope_op = builder.create_scope_op(spec.mlir_attrs(builder), [])
        if domain is not None:
            domain.add_scope(node, spec, scope_op, builder.get_insertion_block().id())
        body = builder.get_scope_entry_block(scope_op)
        builder.set_insertion_point_to_start(body)
        generator.visit_compound_statement(node.body)
        builder.set_insertion_point_to_end(body)
        if not body.has_terminator():
            builder.scope_return([])
