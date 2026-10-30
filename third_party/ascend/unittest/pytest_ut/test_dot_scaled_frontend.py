# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import NPUOptions
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import ASTSource
from triton.compiler.errors import CompilationError

pytestmark = pytest.mark.backend("none")


@triton.jit
def _scaled_dot_shapes(A_SHAPE: tl.constexpr, B_SHAPE: tl.constexpr, SA_SHAPE: tl.constexpr, SB_SHAPE: tl.constexpr,
                       SCALE_DTYPE: tl.constexpr):
    a = tl.full(A_SHAPE, 1, tl.bfloat16)
    b = tl.full(B_SHAPE, 1, tl.bfloat16)
    if SA_SHAPE is None:
        sa = None
    else:
        sa = tl.full(SA_SHAPE, 0x38, tl.uint8).to(SCALE_DTYPE, bitcast=True)
    if SB_SHAPE is None:
        sb = None
    else:
        sb = tl.full(SB_SHAPE, 0x38, tl.uint8).to(SCALE_DTYPE, bitcast=True)
    tl.dot_scaled(a, sa, "bf16", b, sb, "bf16")


def _compile_shapes(a_shape, b_shape, sa_shape, sb_shape, scale_dtype):
    source = ASTSource(_scaled_dot_shapes, {}, {
        "A_SHAPE": a_shape,
        "B_SHAPE": b_shape,
        "SA_SHAPE": sa_shape,
        "SB_SHAPE": sb_shape,
        "SCALE_DTYPE": scale_dtype,
    })
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    return ast_to_ttir(_scaled_dot_shapes, source, context, NPUOptions(arch="Ascend950PR_9579"), {}, {})


@pytest.mark.parametrize("batch", [None, 2])
@pytest.mark.parametrize("scale_dtype,group", [(tl.uint8, 32), (tl.float8e4nv, 16)])
@pytest.mark.parametrize("lhs_scale,rhs_scale", [(False, False), (True, False), (False, True), (True, True)])
def test_dot_scaled_scale_shapes(batch, scale_dtype, group, lhs_scale, rhs_scale):
    prefix = () if batch is None else (batch, )
    sa = prefix + (16, 64 // group) if lhs_scale else None
    sb = prefix + (32, 64 // group) if rhs_scale else None
    module = _compile_shapes(prefix + (16, 64), prefix + (64, 32), sa, sb, scale_dtype)
    assert module.verify()
    assert "tt.dot_scaled" in str(module)


@pytest.mark.parametrize("side", ["lhs", "rhs"])
@pytest.mark.parametrize("scale_dtype,group", [(tl.uint8, 32), (tl.float8e4nv, 16)])
@pytest.mark.parametrize("invalid", ["missing_batch", "wrong_batch", "extra_batch", "wrong_non_k", "wrong_k"])
def test_dot_scaled_rejects_invalid_batched_scale(side, scale_dtype, group, invalid):
    non_k = 16 if side == "lhs" else 32
    shapes = {
        "missing_batch": (non_k, 64 // group),
        "wrong_batch": (4, non_k, 64 // group),
        "extra_batch": (2, 2, non_k, 64 // group),
        "wrong_non_k": (2, non_k * 2, 64 // group),
        "wrong_k": (2, non_k, 128 // group),
    }
    sa, sb = (shapes[invalid], None) if side == "lhs" else (None, shapes[invalid])
    with pytest.raises(CompilationError, match=f"{side}_scale must be a tensor of shape"):
        _compile_shapes((2, 16, 64), (2, 64, 32), sa, sb, scale_dtype)


@pytest.mark.parametrize("side", ["lhs", "rhs"])
def test_dot_scaled_rejects_batched_scale_for_2d_inputs(side):
    scale = (2, 16 if side == "lhs" else 32, 2)
    sa, sb = (scale, None) if side == "lhs" else (None, scale)
    with pytest.raises(CompilationError, match=f"{side}_scale must be a tensor of shape"):
        _compile_shapes((16, 64), (64, 32), sa, sb, tl.uint8)


@pytest.mark.parametrize("lhs_scale,rhs_scale", [(False, False), (True, False), (False, True), (True, True)])
def test_dot_scaled_rejects_mismatched_operand_batches(lhs_scale, rhs_scale):
    sa = (2, 16, 2) if lhs_scale else None
    sb = (4, 32, 2) if rhs_scale else None
    with pytest.raises(CompilationError, match="Both inputs must have the same batch size"):
        _compile_shapes((2, 16, 64), (4, 64, 32), sa, sb, tl.uint8)
