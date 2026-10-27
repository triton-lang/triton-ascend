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
    """
    Handle 'with tla.scope.simd()/tla.scope.cube():' by emitting an inline
    ``scope.scope`` carrying the attributes defined by the scope class
    (``mlir_attrs``); the bishengir HFusion pipeline's outline-scope extracts
    the marked SIMD scope into the ``func.func`` + ``func.call`` that the HIVM
    stage expects.

    The register ops stay inline (sharing the enclosing function's scope, so
    buffer handles are used directly, without function-argument remapping).
    """
    from triton.compiler.code_generator import enter_sub_region

    context_expr = node.items[0].context_expr
    withitem_class = generator.visit(context_expr.func)

    with enter_sub_region(generator):
        scope_op = generator.builder.create_scope_op(withitem_class.mlir_attrs(generator.builder), [])
        body = generator.builder.get_scope_entry_block(scope_op)
        generator.builder.set_insertion_point_to_start(body)
        generator.visit_compound_statement(node.body)
        generator.builder.set_insertion_point_to_end(body)
        if not body.has_terminator():
            generator.builder.scope_return([])
    return None
