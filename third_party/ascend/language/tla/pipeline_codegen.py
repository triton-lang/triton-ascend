"""Lower TLA pipeline loops to index loops and native preload scopes."""
import ast

import triton.language as tl
from triton.compiler.code_generator import CodeGenerator, enter_sub_region, flatten_values_to_ir
from triton.language.core import _unwrap_if_constexpr, base_value

from .scope import _TaskScope, pipeline


class _PipelineTasks:

    def __init__(self, body):
        self.body_id = body.id()
        self.tasks = []
        self.names = {}

    def add_scope(self, node, spec, op, block_id):
        if block_id != self.body_id:
            if isinstance(spec, _TaskScope):
                raise ValueError("task scopes must be direct children of tla.pipeline")
            return
        if not isinstance(spec, _TaskScope) or spec.stage is None:
            raise ValueError("pipeline scopes require vector().task(name, stage=...) or cube().task(name, stage=...)")
        # A static_range may emit the same source task multiple times.
        if self.names.setdefault(spec.name, node) is not node:
            raise ValueError(f"duplicate task name in pipeline: {spec.name}")
        if self.tasks and spec.stage < self.tasks[-1][1]:
            raise ValueError("pipeline stages must be nondecreasing in source order")
        self.tasks.append((op, spec.stage))

    def mark_preload(self, builder, loop):
        depth = self.tasks[-1][1] + 1
        loop.set_attr("hivm.cv_pipelined_loop", builder.get_unit_attr())
        for op, stage in self.tasks:
            op.set_attr("hivm.preload_num", builder.get_int32_attr(depth - 1 - stage))
            op.set_attr("hivm.max_preload_num", builder.get_int32_attr(depth))
            op.set_attr("no_inline", builder.get_unit_attr())


class PipelineCodeGenerator(CodeGenerator):
    _tla_pipeline = None
    _scope_depth = 0

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
        stop = self.semantic.to_tensor(self.visit(node.iter.args[0]))
        if stop.type.is_block() or stop.dtype not in (tl.int32, tl.int64):
            raise ValueError("tla.pipeline num_iters must be a scalar signed int32 or int64")

        builder = self.builder
        with enter_sub_region(self) as (liveins, _):
            loop, iv = builder.create_tla_pipeline(stop.handle)
            body = loop.get_body(0)
            builder.set_insertion_point_to_end(body)
            self.set_value(node.target.id, tl.tensor(iv, tl.int64))
            domain = self._tla_pipeline = _PipelineTasks(body)
            self.scf_stack.append(node)
            try:
                self.visit_compound_statement(node.body)
            finally:
                self.scf_stack.pop()
                self._tla_pipeline = None
            self._check_pipeline_carries(liveins, node.target.id)
            if not domain.tasks:
                raise ValueError("tla.pipeline requires at least one direct vector/cube task")
            if getattr(builder.options, "enable_tla_cv_pipelining", None) is not False:
                domain.mark_preload(builder, loop)
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
