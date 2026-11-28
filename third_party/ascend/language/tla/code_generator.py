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
"""Generate TLA core scopes and staged pipeline loops."""
import ast

import triton.language as tl
from triton.compiler.code_generator import CodeGenerator, enter_sub_region, flatten_values_to_ir
from triton.language.core import _unwrap_if_constexpr, base_value

from ._options import _EXTRA_OPTIONS, _PIPELINE_MODE, _PRELOAD_LOOP, preload_override
from .scope import cube, pipeline, vector

__all__ = ["TLACodeGenerator", "handle_scope_with"]


class _TLACallerContext:
    """Carry the effective core and VF modes into JIT helper specializations."""

    def __init__(self, core_mode, vf_mode):
        self.core_mode = core_mode
        self.vf_mode = vf_mode

    def mangle(self):
        suffix = f"_tla_{self.core_mode}"
        if self.vf_mode is not None:
            suffix += f"_{self.vf_mode}"
        return suffix

    def initialize_callee(self, function, builder):
        builder._tla_context = self


def handle_scope_with(generator, node):
    """Emit the core scope inline and inherit its mode in JIT helpers."""
    item, = node.items
    if item.optional_vars is not None:
        raise ValueError("TLA scopes do not support 'as' bindings")
    context_expr = item.context_expr
    scope_kind = generator.visit(context_expr.func)
    task = None
    if scope_kind in (vector.task, cube.task):
        task = _unwrap_if_constexpr(generator.visit(context_expr))
        scope_kind = task.kind
    tasks = getattr(generator, "_tla_pipeline", None)
    if tasks is None and task is not None and task.stage is not None:
        raise ValueError("task stage requires an enclosing tla.pipeline loop")

    previous_context = generator.caller_context
    previous_tla_context = getattr(generator.builder, "_tla_context", None)
    generator.caller_context = _TLACallerContext(scope_kind.core_mode, scope_kind.vf_mode)
    generator.builder._tla_context = generator.caller_context
    try:
        with enter_sub_region(generator):
            attrs = scope_kind.mlir_attrs(generator.builder)
            scope_op = generator.builder.create_scope_op(attrs, [])
            if tasks is not None:
                tasks.add_scope(node, task, scope_op, generator.builder.get_insertion_block().id())
            body = generator.builder.get_scope_entry_block(scope_op)
            generator.builder.set_insertion_point_to_start(body)
            generator.visit_compound_statement(node.body)
            generator.builder.set_insertion_point_to_end(body)
            if not body.has_terminator():
                generator.builder.scope_return([])
    finally:
        generator.caller_context = previous_context
        generator.builder._tla_context = previous_tla_context


class _PipelineTasks:

    def __init__(self, body):
        self.body_id = body.id()
        self.stages = {}
        self.names = {}

    def add_scope(self, node, spec, op, block_id):
        if block_id != self.body_id:
            if spec is not None:
                raise ValueError("task scopes must be direct children of tla.pipeline")
            return
        if spec is None or spec.stage is None:
            raise ValueError("pipeline scopes require vector().task(name, stage=...) or cube().task(name, stage=...)")
        # A static_range may emit the same source task multiple times.
        if self.names.setdefault(spec.name, node) is not node:
            raise ValueError(f"duplicate task name in pipeline: {spec.name}")
        if self.stages and spec.stage < next(reversed(self.stages)):
            raise ValueError("pipeline stages must be nondecreasing in source order")
        self.stages.setdefault(spec.stage, []).append(op)

    def mark_preload(self, builder, loop):
        depth = max(self.stages) + 1
        loop.set_attr(_PRELOAD_LOOP, builder.get_unit_attr())
        for stage, tasks in self.stages.items():
            op = builder.create_tla_stage(tasks)
            op.set_attr("hivm.preload_num", builder.get_int32_attr(depth - 1 - stage))
            op.set_attr("hivm.max_preload_num", builder.get_int32_attr(depth))
            op.set_attr("no_inline", builder.get_unit_attr())


class TLACodeGenerator(CodeGenerator):
    _tla_pipeline = None
    _scope_depth = 0

    def __init__(self, context, *args, **kwargs):
        from triton._C.libtriton.ascend import ir as ascend_ir

        from .builder import setup_unified_builder

        super().__init__(context, *args, **kwargs)
        arch = getattr(self.builder.options, "arch", None)
        tla_builder = ascend_ir.tla_builder(context, arch or "")
        tla_builder.set_loc(self.file_name, self.begin_line + 1, 0)
        setup_unified_builder(self.builder, tla_builder)

    def visit_With(self, node):
        self._scope_depth += 1
        try:
            return super().visit_With(node)
        finally:
            self._scope_depth -= 1

    def visit_For(self, node):
        if isinstance(node.iter, ast.Call) and _unwrap_if_constexpr(self.visit(node.iter.func)) is pipeline:
            return self._visit_pipeline(node)
        return super().visit_For(node)

    def _visit_pipeline(self, node):
        if self._tla_pipeline is not None:
            raise ValueError("nested tla.pipeline loops are not supported")
        if self._scope_depth or any(not isinstance(parent, ast.For) for parent in self.scf_stack):
            raise ValueError("tla.pipeline cannot be nested in while loops or scopes")
        if len(node.iter.args) != 1 or node.iter.keywords:
            raise ValueError("tla.pipeline expects one num_iters argument (start=0, step=1)")
        if node.orelse or not isinstance(node.target, ast.Name):
            raise ValueError("tla.pipeline requires one induction variable and no for-else")
        if any(isinstance(stmt, (ast.Break, ast.Continue, ast.Return)) for stmt in ast.walk(node)):
            raise ValueError("tla.pipeline does not support break, continue or return")
        stop = self.semantic.to_tensor(self.visit(node.iter).num_iters)
        if stop.type.is_block() or stop.dtype not in (tl.int32, tl.int64):
            raise ValueError("tla.pipeline num_iters must be a scalar signed int32 or int64")

        builder = self.builder
        with enter_sub_region(self) as (liveins, _):
            loop, iv, can_preload = builder.create_tla_pipeline(stop.handle)
            body = loop.get_body(0)
            builder.set_insertion_point_to_end(body)
            self.set_value(node.target.id, tl.tensor(iv, tl.int64))
            tasks = self._tla_pipeline = _PipelineTasks(body)
            self.scf_stack.append(node)
            try:
                self.visit_compound_statement(node.body)
            finally:
                self.scf_stack.pop()
                self._tla_pipeline = None
            self._check_pipeline_carries(liveins, node.target.id)
            if not tasks.stages:
                raise ValueError("tla.pipeline requires at least one direct vector/cube task")
            extra = getattr(builder.options, _EXTRA_OPTIONS, ())
            preload = can_preload and preload_override(extra) is not False
            if preload:
                tasks.mark_preload(builder, loop)
            # A kernel can contain several independent pipeline loops.
            if preload or self.module.get_int_attr(_PIPELINE_MODE) is None:
                self.module.set_attr(_PIPELINE_MODE, builder.get_int32_attr(int(preload)))
            builder.set_insertion_point_to_end(body)
            builder.create_yield_op([])

    def _check_pipeline_carries(self, liveins, induction_var):
        for name, before in liveins.items():
            if name == induction_var:
                continue
            if isinstance(before, base_value):
                after = self.lscope[name]
                self._verify_loop_carried_variable(name, after, before)
                if flatten_values_to_ir([before]) == flatten_values_to_ir([after]):
                    continue
            elif name not in self.local_defs:
                continue
            raise ValueError(f"tla.pipeline does not support loop-carried values: {name}")
