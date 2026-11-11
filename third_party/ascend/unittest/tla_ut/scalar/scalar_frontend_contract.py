# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
"""Host-side API contracts and scalar IR legalization for scalar/ interfaces.

Own parameter validation, type promotion, literals, diagnostics, constant folding,
scalar-only adapter IR, and JIT helper composition. A TLA-enabled native build is
required; no NPU launch is performed.

Per-interface files own test_ir/test_compile/test_run for their writeback kernels,
including buffer/VF structure, target objects, and device accuracy.
"""
import re
import sys
import warnings
from types import SimpleNamespace

import pytest
import triton as tr
import triton.language as tl
from triton.language.extra import tla
from triton.compiler.errors import CompilationError
from triton.language.core import builtin
from triton.language.semantic import TritonSemantic
from triton._C.libtriton import ir

from utils.tla_test_utils import compile_offline, lower_offline, make_context, matrix


@tr.jit
def _select_expression(condition, x, y):
    tla.device_print("selected", tla.select(condition, x, y))


def _lower(fn, signature, constants=None):
    module, ctx = compile_offline(fn, signature, constants or {})
    return lower_offline(module, ctx)


def _ops(module, name):
    result = []
    module.walk(lambda op: result.append(op) if op.get_name() == name else None)
    return result


SCALAR_TYPES = ("i1", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "fp16", "bf16", "fp32", "fp64")


def _scalar_frontend(operation, operands, **kwargs):
    """Emit real scalar IR without requiring the TLA-specific native builder."""
    ctx = ir.context()
    ir.load_dialects(ctx)
    builder = ir.builder(ctx)
    builder.options = SimpleNamespace(sanitize_overflow=False, supported_fp8_dtypes=("fp8e5", ))
    builder.codegen_fns = {}
    module = builder.create_module()
    types = [value for value in operands if isinstance(value, tl.dtype)]
    fn = builder.get_or_insert_function(module, "scalar",
                                        builder.get_function_ty([dtype.to_ir(builder) for dtype in types], []),
                                        "public", False)
    module.push_back(fn)
    builder.set_insertion_point_to_start(fn.add_entry_block())
    args = []
    for value in operands:
        if isinstance(value, tl.dtype):
            index = sum(isinstance(v, tl.tensor) for v in args)
            value = tl.tensor(fn.args(index), value)
        args.append(value)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        try:
            result = operation(*args, _semantic=TritonSemantic(builder), **kwargs)
        except (TypeError, ValueError, AssertionError) as error:
            outcome = type(error), str(error)
        else:
            assert not result.type.is_block()
            builder.ret([])
            assert module.verify()
            outcome = result.dtype, str(module)
    return outcome, [(warning.category, str(warning.message)) for warning in emitted]


def _extremum_frontend(operation, operands, propagate_nan):
    return _scalar_frontend(operation, operands, propagate_nan=propagate_nan)


@pytest.mark.parametrize("dtype", SCALAR_TYPES)
@pytest.mark.parametrize("wrapped", [False, True])
def test_abs_matches_triton_frontend(dtype, wrapped):
    dtype = tl.str_to_ty(dtype, None)
    operation = (lambda x, _semantic: tla.abs(tl.constexpr(x), _semantic=_semantic)) if wrapped else tla.abs
    if dtype in (tl.bfloat16, tl.float64):
        outcome, _ = _scalar_frontend(operation, (dtype, ))
        assert outcome[0] is TypeError and ("bfloat16" if dtype == tl.bfloat16 else "fp64") in outcome[1]
    else:
        assert _scalar_frontend(operation, (dtype, )) == _scalar_frontend(tl.abs, (dtype, ))


@pytest.mark.parametrize("value", [
    False, True, 0, -3, -(1 << 31), -(1 << 63), (1 << 64) - 1, -2.5, -0.0, -float("inf"), -float("nan"),
    tl.constexpr(-3)
])
def test_abs_literals_match_triton(value):
    assert _scalar_frontend(tla.abs, (value, )) == _scalar_frontend(tl.abs, (value, ))


@pytest.mark.parametrize("dtype", [tl.float8e5, tl.float8e4nv, tl.float8e4b15, tl.float64])
@pytest.mark.parametrize("wrapped", [False, True])
def test_abs_rejects_unsupported_floats(dtype, wrapped):
    value = tl.tensor(None, dtype)
    with pytest.raises(TypeError, match="tla.abs does not support dtype"):
        tla.abs(tl.constexpr(value) if wrapped else value, _semantic=object())


@tr.jit
def _abs_expression(x):
    tla.device_print("absolute", tla.abs(x))


@pytest.mark.parametrize("dtype", SCALAR_TYPES)
def test_abs_scalar_ir(dtype):
    if dtype in ("bf16", "fp64"):
        message = "does not support bfloat16" if dtype == "bf16" else "does not support dtype fp64"
        with pytest.raises(CompilationError, match=message):
            _lower(_abs_expression, {"x": dtype})
        return
    module = _lower(_abs_expression, {"x": dtype})
    if dtype.startswith("u") or dtype == "i1":
        assert not _ops(module, "math.absi") and not _ops(module, "math.absf")
    else:
        opcode = "math.absf" if dtype.startswith("fp") else "math.absi"
        op, = _ops(module, opcode)
        assert op.get_operand(0).get_type() == op.get_result(0).get_type()
    debug, = _ops(module, "hivm.hir.debug")
    expected = {"fp16": "f16", "fp32": "f32", "fp64": "f64"}.get(dtype, "i" + dtype[1:])
    assert str(debug.get_operand(0).get_type()) == expected
    assert "tensor<" not in str(module) and not _ops(module, "memref.alloc")


@tr.jit
def _constant_abs(X: tl.constexpr, DTYPE: tl.constexpr):
    tla.device_print("absolute", tla.abs(tl.full((), X, DTYPE)))


@pytest.mark.parametrize("value,dtype,expected", [
    (-127, tl.int8, "127"),
    (-128, tl.int8, "-128"),
    (-32768, tl.int16, "-32768"),
    (-(1 << 31), tl.int32, str(-(1 << 31))),
    (-(1 << 63), tl.int64, str(-(1 << 63))),
    ((1 << 64) - 1, tl.uint64, "-1"),
    (True, tl.int1, "true"),
    (-0.0, tl.float32, "0.000000e+00"),
    (-1.5, tl.float32, "1.500000e+00"),
    (-float("inf"), tl.float32, "0x7F800000"),
    (-float("nan"), tl.float32, "0x7FC00000"),
])
def test_abs_constant_boundaries(value, dtype, expected):
    module = _lower(_constant_abs, {}, {"X": value, "DTYPE": dtype})
    assert re.search(rf"arith.constant {re.escape(expected)}(?:\s|$)", str(module))
    assert not _ops(module, "math.absi") and not _ops(module, "math.absf")


@tr.jit
def _abs_select_helper(condition, x, y):
    return tla.select(condition, tla.abs(x), y)


@tr.jit
def _abs_helper_caller(condition, x, y):
    tla.device_print("absolute", _abs_select_helper(condition, x, y))


def test_abs_composes_in_jit_helper():
    module = _lower(_abs_helper_caller, {"condition": "i1", "x": "i32", "y": "i32"})
    assert len(_ops(module, "math.absi")) == len(_ops(module, "arith.select")) == 1
    assert not _ops(module, "tt.call")


@pytest.mark.parametrize("condition", SCALAR_TYPES)
@pytest.mark.parametrize("left", SCALAR_TYPES + ("fp8e5", ))
@pytest.mark.parametrize("right", SCALAR_TYPES + ("fp8e5", ))
def test_select_matches_triton_frontend(condition, left, right):
    operands = tuple(tl.str_to_ty(dtype, None) for dtype in (condition, left, right))
    if "bf16" in (condition, left, right):
        outcome, _ = _scalar_frontend(tla.select, operands)
        assert outcome[0] is TypeError and "bfloat16" in outcome[1]
        return
    assert _scalar_frontend(tla.select, operands) == _scalar_frontend(tl.where, operands)


@pytest.mark.parametrize("condition", [True, False, 0, -2, 0.0, -0.5, float("nan"), tl.constexpr(True)])
@pytest.mark.parametrize("operands", [
    (tl.bfloat16, 257),
    (257, tl.bfloat16),
    (tl.bfloat16, tl.int32),
    (tl.float16, tl.bfloat16),
    (tl.float16, 1.5),
    (1.5, tl.float16),
    (tl.uint32, -1),
    (tl.int64, 2**63),
    (True, False),
    (2, 3.5),
    (tl.constexpr(257), tl.bfloat16),
    (tl.float32, float("nan")),
    (float("nan"), 1.5),
    (-0.0, 0.0),
])
def test_select_literals_match_triton(condition, operands):
    args = (condition, *operands)
    if any(dtype == tl.bfloat16 for dtype in operands if isinstance(dtype, tl.dtype)):
        outcome, _ = _scalar_frontend(tla.select, args)
        assert outcome[0] is TypeError and "bfloat16" in outcome[1]
        return
    assert _scalar_frontend(tla.select, args) == _scalar_frontend(tl.where, args)


@pytest.mark.parametrize("is_max", [True, False], ids=["maximum", "minimum"])
@pytest.mark.parametrize("propagate_nan", [tl.PropagateNan.NONE, tl.PropagateNan.ALL])
@pytest.mark.parametrize("left", SCALAR_TYPES + ("fp8e5", ))
@pytest.mark.parametrize("right", SCALAR_TYPES + ("fp8e5", ))
def test_minmax_matches_triton_frontend(left, right, propagate_nan, is_max):
    operands = (tl.str_to_ty(left, None), tl.str_to_ty(right, None))
    operation = tla.maximum if is_max else tla.minimum
    reference = tl.maximum if is_max else tl.minimum
    if "bf16" in (left, right):
        outcome, _ = _extremum_frontend(operation, operands, propagate_nan)
        assert outcome[0] is TypeError and "bfloat16" in outcome[1]
        return
    if "fp16" in (left, right):
        outcome, _ = _extremum_frontend(operation, operands, propagate_nan)
        assert outcome[0] is TypeError and "float16" in outcome[1]
        return
    assert _extremum_frontend(operation, operands,
                              propagate_nan) == _extremum_frontend(reference, operands, propagate_nan)


@pytest.mark.parametrize("is_max", [True, False], ids=["maximum", "minimum"])
@pytest.mark.parametrize("propagate_nan", [tl.PropagateNan.NONE, tl.PropagateNan.ALL])
@pytest.mark.parametrize("operands", [
    (tl.bfloat16, 257),
    (257, tl.bfloat16),
    (tl.float16, 1.5),
    (tl.uint32, -1),
    (tl.int64, 2**63),
    (True, False),
    (2, 3.5),
    (tl.constexpr(257), tl.bfloat16),
    (float("nan"), float("nan")),
    (-0.0, 0.0),
])
def test_minmax_literals_match_triton(operands, propagate_nan, is_max):
    operation = tla.maximum if is_max else tla.minimum
    reference = tl.maximum if is_max else tl.minimum
    if any(dtype == tl.bfloat16 for dtype in operands if isinstance(dtype, tl.dtype)):
        outcome, _ = _extremum_frontend(operation, operands, tl.constexpr(propagate_nan))
        assert outcome[0] is TypeError and "bfloat16" in outcome[1]
        return
    if any(dtype == tl.float16 for dtype in operands if isinstance(dtype, tl.dtype)):
        outcome, _ = _extremum_frontend(operation, operands, tl.constexpr(propagate_nan))
        assert outcome[0] is TypeError and "float16" in outcome[1]
        return
    assert _extremum_frontend(operation, operands,
                              tl.constexpr(propagate_nan)) == _extremum_frontend(reference, operands,
                                                                                 tl.constexpr(propagate_nan))


@pytest.mark.parametrize("is_max", [True, False], ids=["maximum", "minimum"])
@pytest.mark.parametrize("position,name", enumerate(("a", "b")))
@pytest.mark.parametrize("dtype,dtype_name", [(tl.float16, "float16"), (tl.bfloat16, "bfloat16")])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("propagate_nan", [tl.PropagateNan.NONE, tl.PropagateNan.ALL])
def test_minmax_rejects_16bit_float_inputs(is_max, position, name, dtype, dtype_name, wrapped, propagate_nan):
    operation = tla.maximum if is_max else tla.minimum
    operation_name = "maximum" if is_max else "minimum"
    operands = [tl.tensor(None, tl.float32), tl.tensor(None, tl.float32)]
    value = tl.tensor(None, dtype)
    operands[position] = tl.constexpr(value) if wrapped else value
    with pytest.raises(TypeError, match=f"tla.{operation_name} {name} does not support {dtype_name}"):
        operation(*operands, propagate_nan=propagate_nan, _semantic=object())


@pytest.mark.parametrize("dtype", SCALAR_TYPES)
def test_select_ir(dtype):
    if dtype == "bf16":
        with pytest.raises(Exception, match="does not support bfloat16"):
            _lower(_select_expression, {"condition": "i1", "x": dtype, "y": dtype})
        return
    module = _lower(_select_expression, {"condition": "i1", "x": dtype, "y": dtype})
    op, = _ops(module, "arith.select")
    assert str(op.get_operand(0).get_type()) == "i1"
    assert op.get_operand(1).get_type() == op.get_operand(2).get_type() == op.get_result(0).get_type()
    assert "tensor<" not in str(module)
    assert not _ops(module, "scf.if")


@tr.jit
def _constant_select(COND: tl.constexpr, X: tl.constexpr, Y: tl.constexpr):
    tla.device_print("selected", tla.select(COND, X, Y))


@pytest.mark.parametrize("condition", [True, False])
@pytest.mark.parametrize("x,y", [(4, -7), (2.5, -1.5), (True, False)])
def test_select_python_literals(condition, x, y):
    module = _lower(_constant_select, {}, {"COND": condition, "X": x, "Y": y})
    assert not _ops(module, "arith.select")
    expected = x if condition else y
    if type(expected) is bool:
        expected = str(expected).lower()
    elif type(expected) is float:
        expected = f"{expected:.6e}"
    assert f"arith.constant {expected}" in str(module)


@pytest.mark.parametrize("condition,x,y,expected", [
    (True, 1, 2.0, "1.000000e+00"),
    (False, True, 2, "2"),
])
def test_select_promotes_literals_with_constant_condition(condition, x, y, expected):
    module = _lower(_constant_select, {}, {"COND": condition, "X": x, "Y": y})
    assert f"arith.constant {expected}" in str(module)


@builtin
def _branch(value, _semantic=None):
    tla.device_print("branch", value, _semantic=_semantic)
    return value


@tr.jit
def _eager_select(condition, x, y):
    tla.device_print("selected", tla.select(condition, _branch(x), _branch(y)))


def test_select_evaluates_both_branches():
    module = _lower(_eager_select, {"condition": "i1", "x": "i32", "y": "i32"})
    prints = _ops(module, "hivm.hir.debug")
    assert len(prints) == 3
    assert [op.get_str_attr("prefix") for op in prints] == ["branch", "branch", "selected"]
    assert not _ops(module, "scf.if")


@tr.jit
def _scalar_helper(condition, x, y):
    return tla.select(condition, x, y)


@tr.jit
def _helper_caller(condition, x, y):
    tla.device_print("selected", _scalar_helper(condition, x, y))


def test_scalar_expressions_in_jit_helper():
    module = _lower(_helper_caller, {"condition": "i1", "x": "i32", "y": "i32"})
    assert len(_ops(module, "arith.select")) == 1
    assert not _ops(module, "tt.call")


@pytest.mark.parametrize("left,right,result_type", [
    ("fp32", "i32", "f32"),
    ("i32", "u32", "i32"),
    ("fp16", "fp32", "f32"),
])
def test_select_promotes_branch_types(left, right, result_type):
    module = _lower(_select_expression, {"condition": "i1", "x": left, "y": right})
    assert _ops(module, "arith.select")
    debug, = _ops(module, "hivm.hir.debug")
    assert str(debug.get_operand(0).get_type()) == result_type


@pytest.mark.parametrize("value", [
    tl.tensor(None, tl.block_type(tl.float32, [1])),
    tl.tensor(None, tl.block_type(tl.int1, [64])),
    tl.tensor(None, tl.pointer_type(tl.float32)),
    tla.vector(None, tla.vector_type(64, tl.float32)),
    tla.buffer(None, tla.buffer_type((64, ), tl.float32, "ub")),
    tla.buffer(None, tla.buffer_type((64, ), tl.float32, "ub"))[0],
    None,
    [1],
    "1",
])
def test_rejects_non_scalar_values(value):
    # Every argument is checked, including the unselected branch. These inputs
    # must be rejected before any IR emission or implicit memory access.
    scalar = tl.tensor(None, tl.int32)
    cond = tl.tensor(None, tl.int1)
    with pytest.raises(TypeError, match="tla.abs x.*scalar"):
        tla.abs(value, _semantic=object())
    for operation in (tla.maximum, tla.minimum):
        for operands in ((value, scalar), (scalar, value)):
            with pytest.raises(TypeError, match="scalar"):
                operation(*operands, _semantic=object())
    for operands in ((value, scalar, scalar), (cond, value, scalar), (cond, scalar, value), (True, scalar, value),
                     (False, value, scalar)):
        with pytest.raises(TypeError, match="scalar"):
            tla.select(*operands, _semantic=object())
    with pytest.raises(TypeError, match="tla.cast input.*scalar"):
        tla.cast(value, tl.int32, _semantic=object())


@tr.jit
def _minmax_expression(x, y, IS_MAX: tl.constexpr):
    if IS_MAX:
        result = tla.maximum(x, y)
    else:
        result = tla.minimum(x, y)
    tla.device_print("extremum", result)


def _extremum_opcode(dtype, is_max):
    kind = "max" if is_max else "min"
    if dtype.startswith(("fp", "bf")):
        return "arith." + kind + "numf"
    return "arith." + kind + ("ui" if dtype.startswith("u") or dtype == "i1" else "si")


@pytest.mark.parametrize("is_max", [True, False], ids=["maximum", "minimum"])
@pytest.mark.parametrize("dtype", SCALAR_TYPES)
def test_minmax_ir(dtype, is_max):
    if dtype in ("fp16", "bf16"):
        name = "float16" if dtype == "fp16" else "bfloat16"
        with pytest.raises(Exception, match=f"does not support {name}"):
            _lower(_minmax_expression, {"x": dtype, "y": dtype}, {"IS_MAX": is_max})
        return
    module = _lower(_minmax_expression, {"x": dtype, "y": dtype}, {"IS_MAX": is_max})
    op, = _ops(module, _extremum_opcode(dtype, is_max))
    assert op.get_operand(0).get_type() == op.get_operand(1).get_type() == op.get_result(0).get_type()
    assert "tensor<" not in str(module)


@pytest.mark.parametrize("is_max", [True, False], ids=["maximum", "minimum"])
@pytest.mark.parametrize("left,right,result", [
    ("i8", "i16", "i16"),
    ("u8", "u32", "u32"),
    ("i8", "u16", "u16"),
    ("u8", "i16", "i16"),
    ("i32", "u32", "u32"),
    ("u32", "i32", "u32"),
    ("i64", "u32", "i64"),
    ("i32", "u64", "u64"),
    ("i1", "i8", "i8"),
    ("fp16", "i32", "fp16"),
    ("fp32", "fp16", "fp32"),
    ("fp32", "fp64", "fp64"),
    ("fp64", "u64", "fp64"),
])
def test_minmax_type_matching(left, right, result, is_max):
    if "fp16" in (left, right):
        with pytest.raises(Exception, match="does not support float16"):
            _lower(_minmax_expression, {"x": left, "y": right}, {"IS_MAX": is_max})
        return
    module = _lower(_minmax_expression, {"x": left, "y": right}, {"IS_MAX": is_max})
    op, = _ops(module, _extremum_opcode(result, is_max))
    expected = {"fp16": "f16", "bf16": "bf16", "fp32": "f32", "fp64": "f64"}.get(result, "i" + result[1:])
    debug, = _ops(module, "hivm.hir.debug")
    assert str(debug.get_operand(0).get_type()) == expected
    assert "tensor<" not in str(module)


@pytest.mark.parametrize("is_max", [True, False], ids=["maximum", "minimum"])
def test_minmax_literal_uses_default_scalar_dtype(is_max):
    module = _lower(_minmax_expression, {"x": "i16"}, {"y": 1.5, "IS_MAX": is_max})
    op, = _ops(module, _extremum_opcode("fp32", is_max))
    assert str(op.get_result(0).get_type()) == "f32"
    assert _ops(module, "arith.sitofp")


@tr.jit
def _constant_minmax(X: tl.constexpr, Y: tl.constexpr, DTYPE: tl.constexpr, IS_MAX: tl.constexpr):
    x = tl.full((), X, DTYPE)
    y = tl.full((), Y, DTYPE)
    if IS_MAX:
        result = tla.maximum(x, y)
    else:
        result = tla.minimum(x, y)
    tla.device_print("extremum", result)


@pytest.mark.parametrize("is_max", [True, False], ids=["maximum", "minimum"])
@pytest.mark.parametrize("x,y,dtype,maximum,minimum", [
    (-2147483648, 2147483647, tl.int32, "2147483647", "-2147483648"),
    (0, 4294967295, tl.uint32, "-1", "0"),
    (False, True, tl.int1, "true", "false"),
    (-2.5, 1.5, tl.float32, "1.500000e+00", "-2.500000e+00"),
    (-float("inf"), float("inf"), tl.float32, "0x7F800000", "0xFF800000"),
    (float("nan"), float("nan"), tl.float32, "0x7FC00000", "0x7FC00000"),
])
def test_minmax_constant_boundaries(x, y, dtype, maximum, minimum, is_max):
    module = _lower(_constant_minmax, {}, {"X": x, "Y": y, "DTYPE": dtype, "IS_MAX": is_max})
    expected = maximum if is_max else minimum
    assert re.search(rf"arith.constant {re.escape(expected)}(?:\s|$)", str(module))
    assert not _ops(module, "arith.maxnumf") and not _ops(module, "arith.minnumf")


@tr.jit
def _xor_expression(x, y):
    tla.device_print("xor", x ^ y)


INTEGER_SCALAR_TYPES = tuple(dtype for dtype in SCALAR_TYPES if dtype.startswith(("i", "u")))


@pytest.mark.parametrize("left", INTEGER_SCALAR_TYPES)
@pytest.mark.parametrize("right", INTEGER_SCALAR_TYPES)
def test_scalar_xor_ir(left, right):
    module = _lower(_xor_expression, {"x": left, "y": right})
    op, = _ops(module, "arith.xori")
    assert str(op.get_result(0).get_type()) == "i" + str(max(int(left[1:]), int(right[1:])))
    assert op.get_operand(0).get_type() == op.get_operand(1).get_type() == op.get_result(0).get_type()
    assert not _ops(module, "memref.alloc")


@pytest.mark.parametrize("left,right", [("fp32", "i32"), ("i32", "fp32"), ("fp32", "fp32")])
def test_scalar_xor_rejects_floats(left, right):
    with pytest.raises(Exception, match="invalid operands"):
        _lower(_xor_expression, {"x": left, "y": right})


@tr.jit
def _xor_literal(x, MASK: tl.constexpr, REVERSE: tl.constexpr):
    if REVERSE:
        result = MASK ^ x
    else:
        result = x ^ MASK
    tla.device_print("xor", result)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype,mask", [("i1", True), ("i8", 85), ("u32", 2**31), ("u64", 2**63)])
def test_scalar_xor_literals(dtype, mask, reverse):
    module = _lower(_xor_literal, {"x": dtype}, {"MASK": mask, "REVERSE": reverse})
    op, = _ops(module, "arith.xori")
    assert str(op.get_result(0).get_type()) == "i" + dtype[1:]


@pytest.mark.parametrize("reverse", [False, True])
def test_scalar_xor_unsigned_negative_literal(reverse):
    with pytest.raises(Exception, match="unsigned tensor and a negative scalar"):
        _lower(_xor_literal, {"x": "u32"}, {"MASK": -1, "REVERSE": reverse})


@tr.jit
def _cast_expression(x, DTYPE: tl.constexpr, BITCAST: tl.constexpr, ROUND: tl.constexpr):
    tla.device_print("cast", tla.cast(x, DTYPE, bitcast=BITCAST, fp_downcast_rounding=ROUND))


@pytest.mark.parametrize("source", SCALAR_TYPES)
@pytest.mark.parametrize("destination", SCALAR_TYPES)
@pytest.mark.parametrize("bitcast", [False, True])
def test_scalar_cast_matches_triton_frontend(source, destination, bitcast):
    operands = (tl.str_to_ty(source, None), )
    kwargs = dict(dtype=tl.constexpr(tl.str_to_ty(destination, None)), bitcast=tl.constexpr(bitcast))
    if "bf16" in (source, destination):
        outcome, _ = _scalar_frontend(tla.cast, operands, **kwargs)
        assert outcome[0] is TypeError
        assert "does not support bfloat16" in outcome[1]
        return
    assert _scalar_frontend(tla.cast, operands, **kwargs) == _scalar_frontend(tl.cast, operands, **kwargs)


@pytest.mark.parametrize("value,dtype,rounding,bitcast", [
    (True, tl.int32, None, False),
    (-257, tl.uint8, None, False),
    (257, tl.float32, None, False),
    (tl.constexpr(257), tl.float32, None, False),
    (-3.75, tl.int32, None, False),
    (float("nan"), tl.int1, None, False),
    (1.5, tl.float16, "rtz", False),
    (1.5, tl.float16, "invalid", False),
    (0x3f800000, tl.float32, None, True),
    (-0.0, tl.int32, None, True),
    (float("inf"), tl.int32, None, True),
    (1.5, tl.int16, None, True),
])
def test_scalar_cast_literals_match_triton(value, dtype, rounding, bitcast):
    kwargs = dict(dtype=dtype, fp_downcast_rounding=tl.constexpr(rounding), bitcast=bitcast)
    assert _scalar_frontend(tla.cast, (value, ), **kwargs) == _scalar_frontend(tl.cast, (value, ), **kwargs)


@pytest.mark.parametrize("value", [True, 257, tl.constexpr(257), 1.5])
def test_scalar_cast_rejects_bfloat16_literals(value):
    outcome, _ = _scalar_frontend(tla.cast, (value, ), dtype=tl.bfloat16)
    assert outcome[0] is TypeError
    assert "does not support bfloat16" in outcome[1]


@pytest.mark.parametrize("source", SCALAR_TYPES)
@pytest.mark.parametrize("destination", SCALAR_TYPES)
def test_scalar_cast_numeric_ir(source, destination):
    if "bf16" in (source, destination):
        with pytest.raises(Exception, match="does not support bfloat16"):
            _lower(_cast_expression, {"x": source},
                   {"DTYPE": tl.str_to_ty(destination, None), "BITCAST": False, "ROUND": None})
        return
    module = _lower(_cast_expression, {"x": source},
                    {"DTYPE": tl.str_to_ty(destination, None), "BITCAST": False, "ROUND": None})
    debug, = _ops(module, "hivm.hir.debug")
    expected = {"fp16": "f16", "bf16": "bf16", "fp32": "f32", "fp64": "f64"}.get(destination, "i" + destination[1:])
    assert str(debug.get_operand(0).get_type()) == expected
    assert "tensor<" not in str(module) and "tt." not in str(module)
    assert not _ops(module, "memref.alloc")


@pytest.mark.parametrize("source,destination", [
    ("fp16", "i16"),
    ("i16", "fp16"),
    ("fp32", "i32"),
    ("u32", "fp32"),
    ("fp64", "i64"),
    ("u64", "fp64"),
])
def test_scalar_cast_bitcast_ir(source, destination):
    module = _lower(_cast_expression, {"x": source},
                    {"DTYPE": tl.str_to_ty(destination, None), "BITCAST": True, "ROUND": None})
    op, = _ops(module, "arith.bitcast")
    assert not _ops(module, "memref.alloc")
    assert not _ops(module, "tt.bitcast")
    debug, = _ops(module, "hivm.hir.debug")
    assert debug.get_operand(0).get_type() == op.get_result(0).get_type()


def test_scalar_cast_bitcast_requires_equal_widths():
    with pytest.raises(Exception, match="Cannot bitcast data-type of size 32 to data-type of size 16"):
        _lower(_cast_expression, {"x": "fp32"}, {"DTYPE": tl.int16, "BITCAST": True, "ROUND": None})


@pytest.mark.parametrize("destination", [tl.float16])
def test_scalar_cast_explicit_rtne(destination):
    module = _lower(_cast_expression, {"x": "fp32"}, {"DTYPE": destination, "BITCAST": False, "ROUND": "rtne"})
    assert not _ops(module, "tt.fp_to_fp")


@pytest.mark.parametrize("rounding", ["invalid", "rtne"])
def test_scalar_cast_invalid_rounding(rounding):
    with pytest.raises(Exception, match="Invalid rounding mode|only for truncating fp conversions"):
        _lower(_cast_expression, {"x": "i32"}, {"DTYPE": tl.float32, "BITCAST": False, "ROUND": rounding})


@tr.jit
def _xor_cast_helper(x, y):
    return tla.cast(tla.cast(x, tl.int32, bitcast=True) ^ y, tl.float32, bitcast=True)


@tr.jit
def _xor_cast_caller(x, y):
    tla.device_print("result", _xor_cast_helper(x, y))


def test_scalar_xor_and_cast_in_jit_helper():
    module = _lower(_xor_cast_caller, {"x": "fp32", "y": "i32"})
    assert len(_ops(module, "arith.bitcast")) == 2
    assert len(_ops(module, "arith.xori")) == 1
    assert not _ops(module, "tt.call") and not _ops(module, "memref.alloc")


@tr.jit
def _constant_xor(X: tl.constexpr, Y: tl.constexpr, DTYPE: tl.constexpr):
    x = tl.full((), X, DTYPE)
    y = tl.full((), Y, DTYPE)
    tla.device_print("xor", x ^ y)


@pytest.mark.parametrize("x,y,dtype,expected", [
    (False, True, tl.int1, "true"),
    (True, True, tl.int1, "false"),
    (-128, 127, tl.int8, "-1"),
    (-2147483648, -1, tl.int32, "2147483647"),
    (0xaaaaaaaa, 0x55555555, tl.uint32, "-1"),
    (1 << 63, (1 << 63) - 1, tl.uint64, "-1"),
])
def test_scalar_xor_constant_boundaries(x, y, dtype, expected):
    module = _lower(_constant_xor, {}, {"X": x, "Y": y, "DTYPE": dtype})
    assert re.search(rf"arith.constant {expected}(?:\s|$)", str(module))
    assert not _ops(module, "arith.xori")


@tr.jit
def _constant_cast(X: tl.constexpr, SRC: tl.constexpr, DST: tl.constexpr, BITCAST: tl.constexpr):
    source = tl.full((), X, SRC)
    result = tla.cast(source, DST, bitcast=BITCAST)
    if DST == tl.float32:
        result = tla.cast(result, tl.int32, bitcast=True)
    elif DST == tl.float16:
        result = tla.cast(result, tl.int16, bitcast=True)
    tla.device_print("cast bits", result)


@pytest.mark.parametrize("value,source,destination,bitcast,expected", [
    (-1, tl.int8, tl.int32, False, "-1"),
    (255, tl.uint8, tl.int32, False, "255"),
    (257, tl.int32, tl.uint8, False, "1"),
    ((1 << 40) + 7, tl.int64, tl.int32, False, "7"),
    (-1, tl.int32, tl.uint64, False, "-1"),
    (3.75, tl.float32, tl.int32, False, "3"),
    (-3.75, tl.float32, tl.int32, False, "-3"),
    (float("nan"), tl.float32, tl.int1, False, "true"),
    (-0.0, tl.float32, tl.int1, False, "false"),
    (65504.0, tl.float32, tl.float16, False, "31743"),
    (float("inf"), tl.float32, tl.int32, True, "2139095040"),
    (0x3f800000, tl.int32, tl.float32, True, "1065353216"),
])
def test_scalar_cast_constant_boundaries(value, source, destination, bitcast, expected):
    module = _lower(_constant_cast, {}, {"X": value, "SRC": source, "DST": destination, "BITCAST": bitcast})
    assert re.search(rf"arith.constant {expected}(?:\s|$)", str(module))
    assert not _ops(module, "memref.alloc")


# cdiv frontend and scalar IR contracts.


@tla.jit
def _cdiv_expression(x, div, USE_TLA: tl.constexpr):
    if USE_TLA:
        result = tla.cdiv(x, div)
    else:
        result = tl.cdiv(x, div)
    tla.device_print("cdiv", result)


def _cdiv_frontend(signature, constants, use_tla):
    try:
        module, _ = compile_offline(_cdiv_expression, signature, dict(constants, USE_TLA=use_tla))
    except CompilationError as error:
        while error.__cause__ is not None:
            error = error.__cause__
        # The two call sites have different locations and source excerpts.
        message = error.error_message if isinstance(error, CompilationError) else str(error)
        return type(error), message
    return "ok", module.str_nodebug()


CDIV_SCALAR_TYPES = ("i1", "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "fp16", "fp32", "fp64")


@pytest.mark.parametrize("left", CDIV_SCALAR_TYPES)
@pytest.mark.parametrize("right", CDIV_SCALAR_TYPES)
def test_cdiv_matches_triton_frontend(left, right):
    signature = {"x": left, "div": right}
    assert _cdiv_frontend(signature, {}, True) == _cdiv_frontend(signature, {}, False)


@pytest.mark.parametrize("signature,constants", [
    ({"x": "i32"}, {"div": 4}),
    ({"x": "u32"}, {"div": tl.constexpr(4)}),
    ({"x": "u32"}, {"div": -1}),
    ({"div": "i64"}, {"x": 2**40 + 7}),
    ({"div": "u32"}, {"x": tl.constexpr(17)}),
    ({"x": "i8"}, {"div": 256}),
    ({"x": "i32"}, {"div": 2.5}),
    ({}, {"x": 17, "div": 4}),
    ({}, {"x": 0, "div": 4}),
    ({}, {"x": -7, "div": 3}),
    ({}, {"x": 7, "div": -3}),
    ({}, {"x": True, "div": True}),
    ({}, {"x": 5.5, "div": 2.0}),
    ({}, {"x": tl.constexpr(17), "div": tl.constexpr(4)}),
    ({}, {"x": 7, "div": 0}),
])
def test_cdiv_literals_match_triton(signature, constants):
    assert _cdiv_frontend(signature, constants, True) == _cdiv_frontend(signature, constants, False)


@pytest.mark.parametrize("x,div,expected", [(0, 3, 0), (1, 3, 1), (15, 4, 4), (16, 4, 4), (17, 4, 5),
                                            (2**40 + 7, 8, 137438953473)])
def test_cdiv_constant_results(x, div, expected):
    module, ctx = compile_offline(_cdiv_expression, {}, dict(x=x, div=div, USE_TLA=True))
    text = str(lower_offline(module, ctx))
    assert re.search(rf"arith.constant {expected} : i(?:32|64)\b", text)
    assert "arith.div" not in text and "tt." not in text


@pytest.mark.parametrize("position", [0, 1], ids=["x", "div"])
@pytest.mark.parametrize("value", [
    None,
    "int32",
    tl.int32,
    tl.tensor(None, tl.pointer_type(tl.int32)),
    tl.tensor(None, tl.block_type(tl.int32, [1])),
    tla.buffer(None, tla.buffer_type((64, ), tl.int32)),
    tla.buffer(None, tla.buffer_type((64, ), tl.int32))[0],
    tla.vector(None, tla.vector_type(64, tl.int32)),
])
def test_cdiv_reject_non_scalar_inputs(value, position):
    operands = [17, 4]
    operands[position] = value
    with pytest.raises(TypeError, match="tla.cdiv .* must be a .*scalar"):
        tla.cdiv(*operands, _semantic=SimpleNamespace())


@pytest.mark.parametrize("position", [0, 1], ids=["x", "div"])
def test_cdiv_reject_bfloat16(position):
    operands = [17, 4]
    operands[position] = tl.tensor(None, tl.bfloat16)
    with pytest.raises(TypeError, match="does not support bfloat16"):
        tla.cdiv(*operands, _semantic=SimpleNamespace())


# clamp frontend and scalar IR contracts.


@pytest.mark.parametrize("x", [tl.float16, tl.bfloat16, tl.float32, tl.float64])
@pytest.mark.parametrize("lower", [tl.float16, tl.bfloat16, tl.float32, tl.float64])
@pytest.mark.parametrize("upper", [tl.float16, tl.bfloat16, tl.float32, tl.float64])
@pytest.mark.parametrize("propagate_nan", [tl.PropagateNan.NONE, tl.PropagateNan.ALL])
def test_clamp_matches_triton_frontend(x, lower, upper, propagate_nan):
    operands = (x, lower, upper)
    if tl.bfloat16 in operands:
        outcome, _ = _scalar_frontend(tla.clamp, operands, propagate_nan=propagate_nan)
        assert outcome[0] is TypeError and "bfloat16" in outcome[1]
        return
    if tl.float16 in operands:
        outcome, _ = _scalar_frontend(tla.clamp, operands, propagate_nan=propagate_nan)
        assert outcome[0] is TypeError and "float16" in outcome[1]
        return
    assert _scalar_frontend(tla.clamp, operands,
                            propagate_nan=propagate_nan) == _scalar_frontend(tl.clamp, operands,
                                                                             propagate_nan=propagate_nan)


@pytest.mark.parametrize("operands", [
    (tl.bfloat16, 257, 259),
    (tl.bfloat16, tl.int32, tl.int32),
    (tl.bfloat16, tl.uint32, tl.uint32),
    (tl.int32, tl.float16, tl.float32),
    (tl.float32, tl.int32, tl.float16),
    (tl.float16, tl.float32, tl.int32),
    (tl.float16, -1.5, 2.5),
    (tl.constexpr(1.5), tl.constexpr(-2.0), tl.constexpr(3.0)),
    (2, -3.5, 4.0),
    (True, False, 2.0),
    (float("inf"), -float("inf"), float("inf")),
    (-0.0, -1.0, 1.0),
])
@pytest.mark.parametrize("propagate_nan", [tl.PropagateNan.NONE, tl.PropagateNan.ALL])
def test_clamp_literals_match_triton(operands, propagate_nan):
    policy = tl.constexpr(propagate_nan)
    if any(dtype == tl.bfloat16 for dtype in operands if isinstance(dtype, tl.dtype)):
        outcome, _ = _scalar_frontend(tla.clamp, operands, propagate_nan=policy)
        assert outcome[0] is TypeError and "bfloat16" in outcome[1]
        return
    if any(dtype == tl.float16 for dtype in operands if isinstance(dtype, tl.dtype)):
        outcome, _ = _scalar_frontend(tla.clamp, operands, propagate_nan=policy)
        assert outcome[0] is TypeError and "float16" in outcome[1]
        return
    assert _scalar_frontend(tla.clamp, operands,
                            propagate_nan=policy) == _scalar_frontend(tl.clamp, operands, propagate_nan=policy)


def test_clamp_nan_literal_with_propagation():
    operands = (float("nan"), -2.0, 3.0)
    policy = tl.constexpr(tl.PropagateNan.ALL)
    assert _scalar_frontend(tla.clamp, operands,
                            propagate_nan=policy) == _scalar_frontend(tl.clamp, operands, propagate_nan=policy)


@pytest.mark.parametrize("dtype", ("i1", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"))
def test_clamp_rejects_integer_computation(dtype):
    operands = (tl.str_to_ty(dtype, None), ) * 3
    outcome = _scalar_frontend(tla.clamp, operands)
    assert outcome == _scalar_frontend(tl.clamp, operands)
    assert outcome[0][0] is TypeError
    assert "Only floating point clamp is supported" in outcome[0][1]


def test_clamp_default_policy_and_export():
    operands = (tl.float32, tl.float32, tl.float32)
    assert _scalar_frontend(tla.clamp, operands) == _scalar_frontend(tl.clamp, operands)
    from triton.language.extra.tla import scalar
    assert "clamp" in tla.__all__ and "clamp" in scalar.__all__
    assert tla.clamp is scalar.clamp


@tla.jit
def _clamp_expression(x, lo, hi, PROPAGATE_NAN: tl.constexpr):
    result = tla.clamp(x=x, min=lo, max=hi, propagate_nan=PROPAGATE_NAN)
    tla.device_print("clamped", result)


@pytest.mark.parametrize("propagate_nan", [tl.PropagateNan.NONE, tl.PropagateNan.ALL])
def test_clamp_scalar_lowering(propagate_nan):
    module = _lower(_clamp_expression, {"x": "fp32", "lo": "fp32", "hi": "fp32"}, {"PROPAGATE_NAN": propagate_nan})
    suffix = "numf" if propagate_nan == tl.PropagateNan.NONE else "imumf"
    minimum, = _ops(module, "arith.min" + suffix)
    maximum, = _ops(module, "arith.max" + suffix)
    assert maximum.get_operand(1).id() == minimum.get_result(0).id()
    assert str(maximum.get_result(0).get_type()) == "f32"
    debug, = _ops(module, "hivm.hir.debug")
    assert str(debug.get_operand(0).get_type()) == "f32"
    assert debug.get_operand(0).id() == maximum.get_result(0).id()
    assert not _ops(module, "arith.cmpf") and not _ops(module, "arith.select")
    text = str(module)
    assert "tt." not in text and "tensor<" not in text and "memref.alloc" not in text


def test_clamp_boundary_rejects_unconverted_op(tmp_path):
    from triton.language.extra.tla.compiler import check_ttadapter

    path = tmp_path / "unconverted_clamp.mlir"
    path.write_text("""
module {
  tt.func public @kernel(%x: f32, %lo: f32, %hi: f32) {
    %result = tt.clampf %x, %lo, %hi, propagateNan = none : f32
    tt.return
  }
}
""")
    ctx = make_context()
    module = ir.parse_mlir_module(str(path), ctx)
    with pytest.raises(ValueError, match="unconverted TLA adapter operations.*tt.clampf"):
        check_ttadapter(module)


@pytest.mark.parametrize("value", [
    tl.tensor(None, tl.block_type(tl.float32, [1])),
    tl.tensor(None, tl.block_type(tl.int1, [64])),
    tl.tensor(None, tl.pointer_type(tl.float32)),
    tla.vector(None, tla.vector_type(64, tl.float32)),
    tla.buffer(None, tla.buffer_type((64, ), tl.float32, "ub")),
    tla.buffer(None, tla.buffer_type((64, ), tl.float32, "ub"))[0],
    None,
    [1],
    "1",
])
@pytest.mark.parametrize("position,name", enumerate(("x", "min", "max")))
def test_clamp_rejects_non_scalar_values(value, position, name):
    scalar = tl.tensor(None, tl.float32)
    operands = [scalar, scalar, scalar]
    operands[position] = tl.constexpr(value)
    with pytest.raises(TypeError, match=f"tla.clamp {name}"):
        tla.clamp(*operands, _semantic=object())


@pytest.mark.parametrize("position,name", enumerate(("x", "min", "max")))
@pytest.mark.parametrize("dtype,dtype_name", [(tl.float16, "float16"), (tl.bfloat16, "bfloat16")])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("propagate_nan", [tl.PropagateNan.NONE, tl.PropagateNan.ALL])
def test_clamp_rejects_16bit_float_inputs(position, name, dtype, dtype_name, wrapped, propagate_nan):
    operands = [tl.tensor(None, tl.float32) for _ in range(3)]
    value = tl.tensor(None, dtype)
    operands[position] = tl.constexpr(value) if wrapped else value
    with pytest.raises(TypeError, match=f"tla.clamp {name} does not support {dtype_name}"):
        tla.clamp(*operands, propagate_nan=propagate_nan, _semantic=object())


# max_value frontend and scalar IR contracts.

MAX_VALUE_CASES = [
    dict(name=name, dtype=dtype, value=value) for name, dtype, value in (
        ("bool", tl.int1, 1),
        ("int8", tl.int8, 127),
        ("uint8", tl.uint8, 255),
        ("int16", tl.int16, 32767),
        ("uint16", tl.uint16, 65535),
        ("int32", tl.int32, 2147483647),
        ("uint32", tl.uint32, 4294967295),
        ("int64", tl.int64, 9223372036854775807),
        ("uint64", tl.uint64, 18446744073709551615),
        ("float16", tl.float16, 65504.0),
        ("bfloat16", tl.bfloat16, float.fromhex("0x1.fep127")),
        ("float32", tl.float32, float.fromhex("0x1.fffffep127")),
        ("float64", tl.float64, sys.float_info.max),
    )
]


@matrix(MAX_VALUE_CASES)
@pytest.mark.parametrize("wrapped", [False, True])
def test_max_value_frontend(case, wrapped):
    dtype = case["dtype"]
    argument = tl.constexpr(dtype) if wrapped else dtype
    ctx = ir.context()
    ir.load_dialects(ctx)
    builder = ir.builder(ctx)
    module = builder.create_module()
    fn = builder.get_or_insert_function(module, "constant", builder.get_function_ty([], []), "public", False)
    module.push_back(fn)
    builder.set_insertion_point_to_start(fn.add_entry_block())
    semantic = TritonSemantic(builder)
    result = tla.max_value(argument, _semantic=semantic)
    # Independent exact limits, materialized through the standard frontend.
    reference = tl.full((), case["value"], dtype, _semantic=semantic)
    assert result.dtype == reference.dtype == dtype
    assert not result.type.is_block()
    builder.ret([])
    assert module.verify()
    constants = re.findall(r"arith.constant (.*?)(?: loc\(|$)", str(module), re.MULTILINE)
    assert len(constants) == 2 and constants[0] == constants[1]
    assert "max_value" in tla.__all__


@pytest.mark.parametrize("dtype", [
    None, True, 32, 1.0,
    tl.tensor(None, tl.int32),
    tl.pointer_type(tl.float32),
    tl.block_type(tl.float32, [1]),
    tla.buffer_type((64, ), tl.float32),
    tla.vector_type(64, tl.float32)
])
def test_max_value_reject_non_scalar_dtype(dtype):
    with pytest.raises(TypeError, match="compile-time scalar dtype"):
        tla.max_value(dtype, _semantic=SimpleNamespace())


@pytest.mark.parametrize("dtype", ["float32", "fp32", "int32", "i32", "bfloat16", "bool", ""])
@pytest.mark.parametrize("wrapped", [False, True])
def test_max_value_reject_string_dtype(dtype, wrapped):
    if wrapped:
        dtype = tl.constexpr(dtype)
    with pytest.raises(TypeError, match="compile-time scalar dtype"):
        tla.max_value(dtype, _semantic=SimpleNamespace())


@pytest.mark.parametrize("dtype", [tl.void, tl.float8e5, tl.float8e4nv])
def test_max_value_reject_unsupported_dtype(dtype):
    with pytest.raises(ValueError, match="tla.max_value does not support dtype"):
        tla.max_value(dtype, _semantic=SimpleNamespace())


# infinity frontend and scalar IR contracts.


@pytest.mark.parametrize("dtype,ir_type,bits", [
    (tl.float16, "f16", "7C00"),
    (tl.bfloat16, "bf16", "7F80"),
    (tl.float32, "f32", "7F800000"),
])
@pytest.mark.parametrize("wrapped", [False, True])
def test_infinity_frontend(dtype, ir_type, bits, wrapped):
    ctx = ir.context()
    ir.load_dialects(ctx)
    builder = ir.builder(ctx)
    module = builder.create_module()
    fn = builder.get_or_insert_function(module, "constant", builder.get_function_ty([], []), "public", False)
    module.push_back(fn)
    builder.set_insertion_point_to_start(fn.add_entry_block())
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        result = tla.infinity(tl.constexpr(dtype) if wrapped else dtype, _semantic=TritonSemantic(builder))
    assert result.dtype == dtype
    assert not result.type.is_block()
    builder.ret([])
    assert module.verify()
    text = str(module)
    assert f"arith.constant 0x{bits} : {ir_type}" in text
    assert "tensor<" not in text and "tt.splat" not in text and "memref.alloc" not in text
    assert not emitted
    assert "infinity" in tla.__all__


@pytest.mark.parametrize("dtype", [
    tl.int1,
    tl.int8,
    tl.int16,
    tl.int32,
    tl.int64,
    tl.uint8,
    tl.uint16,
    tl.uint32,
    tl.uint64,
    tl.float64,
    tl.float8e5,
    tl.float8e4nv,
    tl.float8e4b8,
    tl.float8e4b15,
    tl.float8e5b16,
    tl.pointer_type(tl.float32),
    tl.block_type(tl.float32, [1]),
])
def test_infinity_reject_unsupported_dtype(dtype):
    with pytest.raises(ValueError, match="tla.infinity supports float16, bfloat16 and float32"):
        tla.infinity(tl.constexpr(dtype), _semantic=SimpleNamespace())


@pytest.mark.parametrize("value", [
    None,
    "float32",
    32,
    1.0,
    True,
    float,
    tl.tensor(None, tl.int32),
    tl.tensor(None, tl.block_type(tl.float32, [1])),
    tla.vector(None, tla.vector_type(64, tl.float32)),
    tla.buffer(None, tla.buffer_type((64, ), tl.float32, "ub")),
])
def test_infinity_reject_non_scalar_dtype(value):
    with pytest.raises(TypeError, match="tla.infinity expects a compile-time dtype"):
        tla.infinity(tl.constexpr(value), _semantic=SimpleNamespace())
