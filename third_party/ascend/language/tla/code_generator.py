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

__all__ = ["handle_scope_with", "handle_pipeline_for"]


def handle_scope_with(generator, node):
    """Emit a core scope, or a task directly inside a pipeline loop."""
    from triton.compiler.code_generator import enter_sub_region
    from triton.language.core import _unwrap_if_constexpr
    from .scope import _TaskScope

    if node.items[0].optional_vars is not None:
        raise ValueError("TLA scopes do not support 'as' bindings")
    spec = _unwrap_if_constexpr(generator.visit(node.items[0].context_expr))
    domain = getattr(generator, "_tla_pipeline_domain", None)
    direct = domain is not None and id(node) in domain["scope_nodes"]
    if direct and not isinstance(spec, _TaskScope):
        raise ValueError("pipeline scopes require vector().task(name, stage=...) or cube().task(name, stage=...)")
    if isinstance(spec, _TaskScope):
        if domain is None:
            raise ValueError("task() requires an enclosing tla.pipeline loop")
        if not direct:
            raise ValueError("task scopes must be direct children of tla.pipeline")
        key = (node.lineno, node.col_offset)
        previous = domain["names"].get(spec.name)
        if previous is not None and previous != key:
            raise ValueError(f"duplicate task name in pipeline: {spec.name}")
        # Loop carry analysis visits the same source node more than once.
        domain["names"][spec.name] = key
        if spec.stage < domain["last_stage"]:
            raise ValueError("pipeline stages must be nondecreasing in source order")
        domain["last_stage"] = spec.stage
    old_depth = getattr(generator, "_tla_scope_depth", 0)
    generator._tla_scope_depth = old_depth + 1
    try:
        with enter_sub_region(generator):
            scope_op = generator.builder.create_scope_op(spec.mlir_attrs(generator.builder), [])
            body = generator.builder.get_scope_entry_block(scope_op)
            generator.builder.set_insertion_point_to_start(body)
            generator.visit_compound_statement(node.body)
            generator.builder.set_insertion_point_to_end(body)
            if not body.has_terminator():
                generator.builder.scope_return([])
    finally:
        generator._tla_scope_depth = old_depth
    return None


def handle_pipeline_for(generator, node):
    """Emit the restricted pipeline loop without modifying generic range codegen."""
    import ast
    import triton.language as tl
    from triton.compiler.code_generator import enter_sub_region

    if getattr(generator, "_tla_pipeline_domain", None) is not None:
        raise ValueError("nested tla.pipeline loops are not supported")
    if (any(not isinstance(parent, ast.For) for parent in generator.scf_stack)
            or getattr(generator, "_tla_scope_depth", 0)):
        raise ValueError("tla.pipeline cannot be nested in while loops or scopes")
    if len(node.iter.args) != 1 or node.iter.keywords:
        raise ValueError("tla.pipeline expects one num_iters argument (start=0, step=1)")
    if node.orelse or not isinstance(node.target, ast.Name):
        raise ValueError("tla.pipeline requires one induction variable and no for-else")
    if any(isinstance(stmt, (ast.Break, ast.Continue, ast.Return)) for stmt in ast.walk(node)):
        raise ValueError("tla.pipeline does not support break, continue or return")
    num_iters = generator.semantic.to_tensor(generator.visit(node.iter.args[0]))
    if num_iters.type.is_block() or num_iters.dtype not in (tl.int32, tl.int64):
        raise ValueError("tla.pipeline num_iters must be a scalar signed int32 or int64")

    def direct_scope_nodes(statements):
        # static_range emits its body directly into the enclosing IR block.
        # Do not descend through runtime loops, ifs, or core scopes: their
        # tasks would not be direct children of the pipeline's scf.for.
        from triton.language.core import _unwrap_if_constexpr
        for stmt in statements:
            if isinstance(stmt, ast.With):
                yield id(stmt)
            elif (isinstance(stmt, ast.For) and isinstance(stmt.iter, ast.Call)
                  and _unwrap_if_constexpr(generator.visit(stmt.iter.func)) is tl.static_range):
                yield from direct_scope_nodes(stmt.body)

    generator._tla_pipeline_domain = {
        "scope_nodes": set(direct_scope_nodes(node.body)),
        "names": {},
        "last_stage": -1,
    }
    try:
        with enter_sub_region(generator):
            # Bounds are emitted outside the domain. The domain itself contains
            # exactly one index-typed loop and a scope terminator, as NPU-IR
            # requires. Expose its IV as i64 to ordinary TLA scalar arithmetic.
            loop, iv = generator.builder.create_tla_pipeline(num_iters.handle)
            # Delegated builder calls restore the caller's insertion point.
            generator.builder.set_insertion_point_to_end(loop.get_body(0))
            generator.set_value(node.target.id, tl.tensor(iv, tl.int64))
            ip, loc = generator._get_insertion_point_and_loc()
            carries, _, _ = generator._find_carries(node, generator.lscope.copy(), ignore={node.target.id})
            if carries:
                raise ValueError("tla.pipeline does not support loop-carried values")
            generator._set_insertion_point_and_loc(ip, loc)
            generator._tla_pipeline_domain["last_stage"] = -1
            generator.scf_stack.append(node)
            try:
                generator.visit_compound_statement(node.body)
            finally:
                generator.scf_stack.pop()
            if not generator._tla_pipeline_domain["names"]:
                raise ValueError("tla.pipeline requires at least one direct vector/cube task")
            generator.builder.set_insertion_point_to_end(loop.get_body(0))
            generator.builder.create_yield_op([])
    finally:
        generator._tla_pipeline_domain = None
    return None
