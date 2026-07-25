# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
Patches to ``triton.compiler.code_generator.CodeGenerator`` for Ascend NPU.

Applied by ``_apply_ascend_patch()`` before any compilation happens.
Each patch is a small, targeted override — no full-method duplication.
"""

import ast
import inspect

from triton import knobs, language
from triton._C.libtriton import buffer_ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton._utils import find_paths_if, get_iterable_path
from triton.compiler.code_generator import (CodeGenerator, enter_sub_region, _is_constexpr, _is_non_scalar_tensor,
                                            _is_triton_value, flatten_values_to_ir, unflatten_ir_values,
                                            _unwrap_if_constexpr, BoundJITMethod, _apply_to_tuple_values, ASTFunction)
from triton.compiler.errors import CompilationError
from triton.language import constexpr
from triton.language.core import _unwrap_if_constexpr as _unwrap
from triton.runtime.jit import (get_jit_fn_file_line, get_full_name, JITFunction, ConstexprFunction,
                                BoundConstexprFunction)
import triton.language.extra.cann.extension as extension
from triton.language.extra.cann.extension.builder import setup_unified_builder
from triton.language.extra.cann.extension.dispatch import ASCEND_WITH_DISPATCH
from triton.language.extra.extension.buffer.language.builder import (setup_unified_builder_with_buffer_builder)


def _check_fn_args(node, fn, args):
    from triton.compiler.code_generator import _check_fn_args as _cfa
    return _cfa(node, fn, args)


def _mangle_fn(fn, arg_tys, constants, caller_context):
    from triton.compiler.code_generator import mangle_fn
    return mangle_fn(fn, arg_tys, constants, caller_context)


# ---------------------------------------------------------------------------
# Patch 1: __init__ — add ascend_builder + buffer_builder after base init
# ---------------------------------------------------------------------------
_original_init = CodeGenerator.__init__


def _patched_init(self, context, prototype, gscope, function_name, jit_fn, *, options, codegen_fns, module_map,
                  is_gluon, module=None, is_kernel=False, function_types=None, noinline=False, caller_context=None,
                  file_name=None, begin_line=0):
    _original_init(self, context, prototype, gscope, function_name, jit_fn, options=options, codegen_fns=codegen_fns,
                   module_map=module_map, is_gluon=is_gluon, module=module, is_kernel=is_kernel,
                   function_types=function_types, noinline=noinline, caller_context=caller_context, file_name=file_name,
                   begin_line=begin_line)
    if not is_gluon:
        compile_mode = "simt" if (hasattr(options, "force_simt_only") and options.force_simt_only) else "simd"
        self.ascend_builder = ascend_ir.ascendnpu_ir_builder(context, getattr(options, "arch", ""),
                                                             compile_mode=compile_mode)
        self.ascend_builder.set_loc(file_name, begin_line, 0)
        setup_unified_builder(self.builder, self.ascend_builder)
        self.buffer_builder = buffer_ir.buffer_builder(context)
        self.buffer_builder.set_loc(file_name, begin_line, 0)
        setup_unified_builder_with_buffer_builder(self.builder, self.buffer_builder)


# ---------------------------------------------------------------------------
# Patch 2: _get_insertion_point_and_loc — accept optional builder
# ---------------------------------------------------------------------------
_original_get_ip = CodeGenerator._get_insertion_point_and_loc


def _patched_get_ip(self, builder=None):
    b = self.builder if builder is None else builder
    return b.get_insertion_point(), b.get_loc()


# ---------------------------------------------------------------------------
# Patch 3: _set_insertion_point_and_loc — accept optional builder
# ---------------------------------------------------------------------------
_original_set_ip = CodeGenerator._set_insertion_point_and_loc


def _patched_set_ip(self, ip, loc, builder=None):
    b = self.builder if builder is None else builder
    b.restore_insertion_point(ip)
    b.set_loc(loc)


# ---------------------------------------------------------------------------
# Patch 4: visit_With — dispatch Ascend context managers first, then fall back
# ---------------------------------------------------------------------------
_WITH_DISPATCH = dict(ASCEND_WITH_DISPATCH)
_original_visit_With = CodeGenerator.visit_With


def _patched_visit_With(self, node):
    if len(node.items) == 1:
        ctx = node.items[0].context_expr
        if isinstance(ctx, ast.Call):
            cls = self.visit(ctx.func)
            handler = _WITH_DISPATCH.get(cls)
            if handler is not None:
                return handler(self, node)
    return _original_visit_With(self, node)


# ---------------------------------------------------------------------------
# Patch 5: visit_For — add extension.parallel iterator support.
# The base method is replaced because the change (IteratorClass check) is
# embedded mid-method.  This is the full base logic with two additions:
#   (a) ``IteratorClass in [language.range, extension.parallel]``
#   (b) ``hivm.parallel_loop`` attr on for_op when iterator is parallel.
# ---------------------------------------------------------------------------
_original_visit_For = CodeGenerator.visit_For


def _patched_visit_For(self, node):
    IteratorClass = self.visit(node.iter.func)
    iter_args = [self.visit(arg) for arg in node.iter.args]
    iter_kwargs = dict(self.visit(keyword) for keyword in node.iter.keywords)

    if IteratorClass == language.static_range:
        iterator = IteratorClass(*iter_args, **iter_kwargs)
        static_range = range(iterator.start.value, iterator.end.value, iterator.step.value)
        for i in static_range:
            self.lscope[node.target.id] = constexpr(i)
            self.visit_compound_statement(node.body)
            for stmt in node.orelse:
                ast.NodeVisitor.generic_visit(self, stmt)
        return

    num_stages = None
    loop_unroll_factor = None
    disallow_acc_multi_buffer = False
    flatten = False
    warp_specialize = False
    disable_licm = False

    # --- Ascend change (a): allow extension.parallel ---
    if IteratorClass in [language.range, extension.parallel]:
        iterator = IteratorClass(*iter_args, **iter_kwargs)
        lb = iterator.start
        ub = iterator.end
        step = iterator.step
        num_stages = iterator.num_stages
        loop_unroll_factor = iterator.loop_unroll_factor
        disallow_acc_multi_buffer = iterator.disallow_acc_multi_buffer
        flatten = iterator.flatten
        warp_specialize = iterator.warp_specialize
        disable_licm = iterator.disable_licm
    elif IteratorClass is range:
        lb = iter_args[0] if len(iter_args) > 1 else self.visit(ast.Constant(0))
        ub = iter_args[1] if len(iter_args) > 1 else self.visit(node.iter.args[0])
        step = iter_args[2] if len(iter_args) > 2 else self.visit(ast.Constant(1))
    else:
        raise RuntimeError('Only `range` and `static_range` iterators are currently supported')

    # --- remainder is identical to base visit_For ---
    negative_step = False
    if _is_constexpr(step) and step.value < 0:
        step = constexpr(-step.value)
        negative_step = True
        lb, ub = ub, lb
    lb = self.semantic.to_tensor(lb)
    ub = self.semantic.to_tensor(ub)
    step = self.semantic.to_tensor(step)
    if not lb.dtype.is_int() or not ub.dtype.is_int() or not step.dtype.is_int():
        raise TypeError(f"For loop bounds and step must all be ints, are ({lb.dtype}, {ub.dtype}, {step.dtype})")
    if _is_non_scalar_tensor(lb):
        raise TypeError(f"For lower bound must be a scalar, got {lb.type}")
    if _is_non_scalar_tensor(ub):
        raise TypeError(f"For upper bound must be a scalar, got {ub.type}")
    if _is_non_scalar_tensor(step):
        raise TypeError(f"For step must be a scalar, got {step.type}")
    iv_type = self.semantic.integer_promote_impl(lb.dtype, ub.dtype)
    iv_type = self.semantic.integer_promote_impl(iv_type, step.dtype)
    iv_ir_type = iv_type.to_ir(self.builder)
    iv_is_signed = iv_type.int_signedness == language.core.dtype.SIGNEDNESS.SIGNED
    lb = lb.handle
    ub = ub.handle
    step = step.handle
    lb = self.builder.create_int_cast(lb, iv_ir_type, iv_is_signed)
    ub = self.builder.create_int_cast(ub, iv_ir_type, iv_is_signed)
    step = self.builder.create_int_cast(step, iv_ir_type, iv_is_signed)
    iv_placeholder = self.builder.create_poison(iv_ir_type)
    self.set_value(node.target.id, language.core.tensor(iv_placeholder, iv_type))

    with enter_sub_region(self) as sr:
        liveins, insert_block = sr
        ip, last_loc = self._get_insertion_point_and_loc()
        names, init_handles, init_tys = self._find_carries(node, liveins, ignore={node.target.id})
        self._set_insertion_point_and_loc(ip, last_loc)
        for_op = self.builder.create_for_op(lb, ub, step, init_handles)
        if _unwrap(num_stages) is not None:
            for_op.set_attr("tt.num_stages", self.builder.get_int32_attr(num_stages))
        if _unwrap(loop_unroll_factor) is not None:
            for_op.set_attr("tt.loop_unroll_factor", self.builder.get_int32_attr(loop_unroll_factor))
        if disallow_acc_multi_buffer:
            for_op.set_attr("tt.disallow_acc_multi_buffer", self.builder.get_unit_attr())
        if flatten:
            for_op.set_attr("tt.flatten", self.builder.get_unit_attr())
        if warp_specialize:
            for_op.set_attr("tt.warp_specialize", self.builder.get_unit_attr())
        if disable_licm:
            for_op.set_attr("tt.disable_licm", self.builder.get_unit_attr())
        # --- Ascend change (b): parallel loop attr ---
        if IteratorClass is extension.parallel:
            for_op.set_attr("hivm.parallel_loop", self.builder.get_unit_attr())

        self.scf_stack.append(node)
        for_op_body = for_op.get_body(0)
        self.builder.set_insertion_point_to_start(for_op_body)
        block_handles = [for_op_body.arg(i + 1) for i in range(len(init_handles))]
        block_args = unflatten_ir_values(block_handles, init_tys)
        for name, val in zip(names, block_args):
            self._maybe_set_loc_to_name(val, name)
            self.set_value(name, val)
        self.visit_compound_statement(node.body)
        self.scf_stack.pop()
        yield_handles = flatten_values_to_ir(self.lscope[name] for name in names)
        if len(yield_handles) > 0:
            self.builder.create_yield_op(yield_handles)
        for_op_region = for_op_body.get_parent()
        assert for_op_region.size() == 1, "We use SCF, so the loop body should only have one block"
        self.builder.set_insertion_point_to_start(for_op_body)
        iv = for_op.get_induction_var()
        if negative_step:
            iv = self.builder.create_sub(ub, iv)
            iv = self.builder.create_add(iv, lb)
        iv_placeholder.replace_all_uses_with(iv)
        self.set_value(node.target.id, language.core.tensor(iv, iv_type))
        self._maybe_set_loc_to_name(iv, node.target.id)

    result_handles = [for_op.get_result(i) for i in range(len(init_handles))]
    result_values = unflatten_ir_values(result_handles, init_tys)
    for name, val in zip(names, result_values):
        self.set_value(name, val)
        self._maybe_set_loc_to_name(val, name)
    for stmt in node.orelse:
        assert False, "Don't know what to do with else after for"
        ast.NodeVisitor.generic_visit(self, stmt)


# ---------------------------------------------------------------------------
# Patch 6: call_JitFunction — add PropagateNan to constexpr wrapping
# ---------------------------------------------------------------------------
_original_call_JitFunction = CodeGenerator.call_JitFunction


def _patched_call_JitFunction(self, fn, args, kwargs, caller_context=None):
    args = inspect.getcallargs(fn.fn, *args, **kwargs)
    args = [args[name] for name in fn.arg_names]
    for i, arg in enumerate(args):
        # --- Ascend: added language.PropagateNan ---
        if isinstance(arg, (language.dtype, float, int, bool, JITFunction, language.PropagateNan)):
            args[i] = language.core.constexpr(arg)
    args_cst = find_paths_if(args, lambda _, x: _is_constexpr(x))
    args_cst = {path: get_iterable_path(args, path) for path in args_cst}
    args_path = find_paths_if(args, lambda _, x: not _is_constexpr(x))
    args_val = [get_iterable_path(args, path) for path in args_path]
    caller_context = caller_context or self.caller_context
    fn_name = _mangle_fn(get_full_name(fn), [arg.type for arg in args_val], args_cst, caller_context)
    if not self.module.has_function(fn_name):
        file_name, begin_line = get_jit_fn_file_line(fn)
        arg_types = [
            language.core.constexpr if arg is None or isinstance(arg, (bool, int, language.core.dtype)) else arg.type
            for arg in args
        ]
        prototype = ASTFunction([], arg_types, args_cst, dict())
        generator = CodeGenerator(self.context, prototype, fn.get_capture_scope(), module=self.module, jit_fn=fn,
                                  function_name=fn_name, function_types=self.function_ret_types, noinline=fn.noinline,
                                  file_name=file_name, begin_line=begin_line, options=self.builder.options,
                                  codegen_fns=self.builder.codegen_fns, module_map=self.builder.module_map,
                                  caller_context=caller_context, is_gluon=self.is_gluon)
        try:
            generator.visit(fn.parse())
        except Exception as e:
            if knobs.compilation.front_end_debugging:
                raise
            raise CompilationError(self.jit_fn.src, self.cur_node, None) from e
        callee_ret_type = generator.ret_type
        self.function_ret_types[fn_name] = callee_ret_type
    else:
        callee_ret_type = self.function_ret_types[fn_name]
    symbol = self.module.get_function(fn_name)
    args_val = flatten_values_to_ir(args_val)
    call_op = self.builder.call(symbol, args_val)
    if callee_ret_type == language.void:
        return None
    handles = [call_op.get_result(i) for i in range(call_op.get_num_results())]
    return next(unflatten_ir_values(handles, [callee_ret_type]))


# ---------------------------------------------------------------------------
# Patch 7: call_Function — use ascend_builder for Ascend builtin ops
# ---------------------------------------------------------------------------
_original_call_Function = CodeGenerator.call_Function


def _patched_call_Function(self, node, fn, args, kws):
    if isinstance(fn, (BoundJITMethod, BoundConstexprFunction)):
        args.insert(0, fn.__self__)
        fn = fn.__func__
    if isinstance(fn, JITFunction):
        _check_fn_args(node, fn, args)
        return self.call_JitFunction(fn, args, kws)
    if (hasattr(fn, '__self__') and _is_triton_value(fn.__self__)) or language.core.is_builtin(fn) or isinstance(
            fn, ConstexprFunction):
        ip, last_loc = self._get_insertion_point_and_loc()
        # --- Ascend: use ascend_builder for extension builtins ---
        _builder = self.ascend_builder if extension.is_builtin(fn) else self.builder
        self._set_insertion_point_and_loc(ip, last_loc, _builder)
        extra_kwargs = dict()
        if isinstance(fn, ConstexprFunction):
            sig = inspect.signature(fn.__call__)
        else:
            sig = inspect.signature(fn)
        if '_semantic' in sig.parameters:
            extra_kwargs["_semantic"] = self.semantic
        if '_generator' in sig.parameters:
            extra_kwargs['_generator'] = self
        try:
            ret = fn(*args, **extra_kwargs, **kws)
            if isinstance(ret, tuple):
                ret = language.tuple(ret)
            ip, last_loc = self._get_insertion_point_and_loc(_builder)
            self._set_insertion_point_and_loc(ip, last_loc)
            return ret
        except Exception as e:
            if knobs.compilation.front_end_debugging:
                raise
            raise CompilationError(self.jit_fn.src, node, str(e)) from e
    # --- upstream fallback (unchanged) ---
    if fn in self.builtin_namespace.values() or (hasattr(fn, '__self__') and not _is_triton_value(fn.__self__)):
        args = map(_unwrap_if_constexpr, args)
    ret = fn(*args, **kws)

    def wrap_constexpr(x):
        if _is_triton_value(x):
            return x
        return constexpr(x)

    from builtins import tuple as builtin_tuple
    if isinstance(ret, (builtin_tuple, language.tuple)):
        return _apply_to_tuple_values(ret, wrap_constexpr)
    return wrap_constexpr(ret)


# ---------------------------------------------------------------------------
# Patch 7: statically_implemented_functions — register int64
# ---------------------------------------------------------------------------
CodeGenerator.statically_implemented_functions[extension.int64] = \
    CodeGenerator.static_executor(extension.int64)


# ---------------------------------------------------------------------------
# Apply all patches
# ---------------------------------------------------------------------------
def apply():
    """Apply all Ascend code-generator patches (idempotent)."""
    if getattr(apply, "_done", False):
        return

    CodeGenerator.__init__ = _patched_init
    CodeGenerator._get_insertion_point_and_loc = _patched_get_ip
    CodeGenerator._set_insertion_point_and_loc = _patched_set_ip
    CodeGenerator.visit_With = _patched_visit_With
    CodeGenerator.visit_For = _patched_visit_For
    CodeGenerator.call_JitFunction = _patched_call_JitFunction
    CodeGenerator.call_Function = _patched_call_Function

    apply._done = True
