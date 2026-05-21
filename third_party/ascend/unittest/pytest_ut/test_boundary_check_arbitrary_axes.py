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

import pytest
import torch
import torch_npu
import triton
import triton.language as tl


# =============================================================================
# boundary_check regression suite
#
# Broad layout classes:
# 1. Direct layout: the kernel directly builds the target block_ptr view.
# 2. Implicit permute layout: the block_ptr shape/stride mapping follows the
#    same reverse-axes style as the implicit-permute repro path.
#
# Test matrix:
#
#  | case group | layout class  | rank | boundary_check |
#  |------------|---------------|------|----------------|
#  | (A2)       | direct layout | 2-D  | partial axes   |
#  | (A3)       | direct layout | 3-D  | partial axes   |
#  | (B2)       | direct layout | 2-D  | all axes       |
#  | (B3)       | direct layout | 3-D  | all axes       |
#  | (C2)       | implicit permute layout | 2-D  | partial axes   |
#  | (C3)       | implicit permute layout | 3-D  | all axes       |
#  | (D2)       | direct layout + tl.advance | 2-D | partial axes |
#  | (E2)       | direct layout + loop-carried for | 2-D | partial axes |
#
# Notes:
# 1. A2/A3 exercise plain direct-layout make_block_ptr with boundary_check on a
#    later axis, proving the partial-axis issue is not limited to the implicit
#    implicit-permute-style path.
# 2. B2/B3 exercise direct-layout remapped views with boundary_check on all
#    axes, covering the make_tensor_ptr full-boundary lowering issue.
# 3. C2/C3 mirror the original implicit-permute reverse-axes repro style:
#    C2 for partial-axis checking, C3 for all-axis checking.
# 4. D2/E2 extend the suite to explicit tl.advance and runtime loop-carried
#    block_ptr flows, covering the newly added tensor-ptr tracing paths.
# =============================================================================


SHAPE_2D = (32, 50, 8, 8)
SHAPE_3D_PARTIAL = (8, 8, 11, 4, 4, 8)
SHAPE_3D_FULL = (5, 9, 11, 4, 4, 8)
SHAPE_2D_ADVANCE = (32, 50, 8, 8, 45)
SHAPE_2D_LOOP = (32, 50, 8, 8, 38, 6, 2)


def ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def make_input(shape: tuple[int, ...]) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= dim
    return torch.arange(numel, dtype=torch.float32, device="npu").reshape(shape)


def assert_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0.0, atol=0.0)


def make_partial_copy_expected(
    src: torch.Tensor,
    start_col: int,
    col_block: int,
    step_col: int = 0,
    num_steps: int = 1,
) -> torch.Tensor:
    expected = torch.full_like(src, -1.0)
    cols = src.shape[1]
    for step in range(num_steps):
        col_begin = start_col + step * step_col
        col_end = min(col_begin + col_block, cols)
        if col_begin < cols:
            expected[:, col_begin:col_end] = src[:, col_begin:col_end]
    return expected


@triton.jit
def bc_a2_direct_2d_partial(
    in_ptr,
    out_ptr,
    rows,
    cols,
    stride_row,
    stride_col,
    ROWBLOCK: tl.constexpr,
    COLBLOCK: tl.constexpr,
):
    row0 = tl.program_id(0) * ROWBLOCK
    col0 = tl.program_id(1) * COLBLOCK
    in_block_ptr = tl.make_block_ptr(
        base=in_ptr,
        shape=(rows, cols),
        strides=(stride_row, stride_col),
        offsets=(row0, col0),
        block_shape=(ROWBLOCK, COLBLOCK),
        order=(1, 0),
    )
    value = tl.load(in_block_ptr, boundary_check=(1,), padding_option="zero")

    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(rows, cols),
        strides=(stride_row, stride_col),
        offsets=(row0, col0),
        block_shape=(ROWBLOCK, COLBLOCK),
        order=(1, 0),
    )
    tl.store(out_block_ptr, value, boundary_check=(1,))


@triton.jit
def bc_a3_direct_3d_partial(
    in_ptr,
    out_ptr,
    d0,
    d1,
    d2,
    stride0,
    stride1,
    stride2,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
):
    i0 = tl.program_id(0) * B0
    i1 = tl.program_id(1) * B1
    i2 = tl.program_id(2) * B2
    in_block_ptr = tl.make_block_ptr(
        base=in_ptr,
        shape=(d0, d1, d2),
        strides=(stride0, stride1, stride2),
        offsets=(i0, i1, i2),
        block_shape=(B0, B1, B2),
        order=(2, 1, 0),
    )
    value = tl.load(in_block_ptr, boundary_check=(2,), padding_option="zero")

    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(d0, d1, d2),
        strides=(stride0, stride1, stride2),
        offsets=(i0, i1, i2),
        block_shape=(B0, B1, B2),
        order=(2, 1, 0),
    )
    tl.store(out_block_ptr, value, boundary_check=(2,))


@triton.jit
def bc_b2_direct_2d_full(
    in_ptr,
    out_ptr,
    rows,
    cols,
    stride_row,
    stride_col,
    out_stride0,
    out_stride1,
    ROWBLOCK: tl.constexpr,
    COLBLOCK: tl.constexpr,
):
    col0 = tl.program_id(0) * COLBLOCK
    row0 = tl.program_id(1) * ROWBLOCK
    in_block_ptr = tl.make_block_ptr(
        base=in_ptr,
        shape=(cols, rows),
        strides=(stride_col, stride_row),
        offsets=(col0, row0),
        block_shape=(COLBLOCK, ROWBLOCK),
        order=(0, 1),
    )
    value = tl.load(in_block_ptr, boundary_check=(0, 1), padding_option="zero")

    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(cols, rows),
        strides=(out_stride0, out_stride1),
        offsets=(col0, row0),
        block_shape=(COLBLOCK, ROWBLOCK),
        order=(1, 0),
    )
    tl.store(out_block_ptr, value, boundary_check=(0, 1))


@triton.jit
def bc_b3_direct_3d_full(
    in_ptr,
    out_ptr,
    d0,
    d1,
    d2,
    stride0,
    stride1,
    stride2,
    out_stride0,
    out_stride1,
    out_stride2,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
):
    i0 = tl.program_id(0) * B0
    i1 = tl.program_id(1) * B1
    i2 = tl.program_id(2) * B2
    in_block_ptr = tl.make_block_ptr(
        base=in_ptr,
        shape=(d2, d0, d1),
        strides=(stride2, stride0, stride1),
        offsets=(i0, i1, i2),
        block_shape=(B0, B1, B2),
        order=(0, 2, 1),
    )
    value = tl.load(in_block_ptr, boundary_check=(0, 1, 2), padding_option="zero")

    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(d2, d0, d1),
        strides=(out_stride0, out_stride1, out_stride2),
        offsets=(i0, i1, i2),
        block_shape=(B0, B1, B2),
        order=(2, 1, 0),
    )
    tl.store(out_block_ptr, value, boundary_check=(0, 1, 2))


@triton.jit
def bc_c2_implicit_permute_2d_partial(
    in_ptr,
    out_ptr,
    rows,
    cols,
    stride_row,
    stride_col,
    out_stride0,
    out_stride1,
    ROWBLOCK: tl.constexpr,
    COLBLOCK: tl.constexpr,
):
    col0 = tl.program_id(0) * COLBLOCK
    row0 = tl.program_id(1) * ROWBLOCK
    in_block_ptr = tl.make_block_ptr(
        base=in_ptr,
        shape=(cols, rows),
        strides=(stride_col, stride_row),
        offsets=(col0, row0),
        block_shape=(COLBLOCK, ROWBLOCK),
        order=(0, 1),
    )
    value = tl.load(in_block_ptr, boundary_check=(0,), padding_option="zero")

    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(cols, rows),
        strides=(out_stride0, out_stride1),
        offsets=(col0, row0),
        block_shape=(COLBLOCK, ROWBLOCK),
        order=(1, 0),
    )
    tl.store(out_block_ptr, value, boundary_check=(0,))


@triton.jit
def bc_c3_implicit_permute_3d_full(
    in_ptr,
    out_ptr,
    d0,
    d1,
    d2,
    stride0,
    stride1,
    stride2,
    out_stride0,
    out_stride1,
    out_stride2,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
):
    i0 = tl.program_id(0) * B0
    i1 = tl.program_id(1) * B1
    i2 = tl.program_id(2) * B2
    in_block_ptr = tl.make_block_ptr(
        base=in_ptr,
        shape=(d2, d1, d0),
        strides=(stride2, stride1, stride0),
        offsets=(i0, i1, i2),
        block_shape=(B0, B1, B2),
        order=(0, 1, 2),
    )
    value = tl.load(in_block_ptr, boundary_check=(0, 1, 2), padding_option="zero")

    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(d2, d1, d0),
        strides=(out_stride0, out_stride1, out_stride2),
        offsets=(i0, i1, i2),
        block_shape=(B0, B1, B2),
        order=(2, 1, 0),
    )
    tl.store(out_block_ptr, value, boundary_check=(0, 1, 2))


@triton.jit
def bc_d2_direct_2d_advance_partial(
    in_ptr,
    out_ptr,
    rows,
    cols,
    stride_row,
    stride_col,
    advance_col,
    ROWBLOCK: tl.constexpr,
    COLBLOCK: tl.constexpr,
):
    row0 = tl.program_id(0) * ROWBLOCK
    in_block_ptr = tl.make_block_ptr(
        base=in_ptr,
        shape=(rows, cols),
        strides=(stride_row, stride_col),
        offsets=(row0, 0),
        block_shape=(ROWBLOCK, COLBLOCK),
        order=(1, 0),
    )
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(rows, cols),
        strides=(stride_row, stride_col),
        offsets=(row0, 0),
        block_shape=(ROWBLOCK, COLBLOCK),
        order=(1, 0),
    )
    in_block_ptr = tl.advance(in_block_ptr, (0, advance_col))
    out_block_ptr = tl.advance(out_block_ptr, (0, advance_col))
    value = tl.load(in_block_ptr, boundary_check=(1,), padding_option="zero")
    tl.store(out_block_ptr, value, boundary_check=(1,))


@triton.jit
def bc_e2_direct_2d_for_partial(
    in_ptr,
    out_ptr,
    rows,
    cols,
    stride_row,
    stride_col,
    start_col,
    step_col,
    num_steps,
    ROWBLOCK: tl.constexpr,
    COLBLOCK: tl.constexpr,
):
    row0 = tl.program_id(0) * ROWBLOCK
    in_block_ptr = tl.make_block_ptr(
        base=in_ptr,
        shape=(rows, cols),
        strides=(stride_row, stride_col),
        offsets=(row0, start_col),
        block_shape=(ROWBLOCK, COLBLOCK),
        order=(1, 0),
    )
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(rows, cols),
        strides=(stride_row, stride_col),
        offsets=(row0, start_col),
        block_shape=(ROWBLOCK, COLBLOCK),
        order=(1, 0),
    )
    for _ in range(num_steps):
        value = tl.load(in_block_ptr, boundary_check=(1,), padding_option="zero")
        tl.store(out_block_ptr, value, boundary_check=(1,))
        in_block_ptr = tl.advance(in_block_ptr, (0, step_col))
        out_block_ptr = tl.advance(out_block_ptr, (0, step_col))


def test_bc_a2_direct_2d_partial():
    rows, cols, row_block, col_block = SHAPE_2D
    src = make_input((rows, cols))
    out = torch.full_like(src, -1.0)

    grid = (ceil_div(rows, row_block), ceil_div(cols, col_block))
    bc_a2_direct_2d_partial[grid](
        src,
        out,
        rows,
        cols,
        src.stride(0),
        src.stride(1),
        ROWBLOCK=row_block,
        COLBLOCK=col_block,
    )

    assert_equal(out, src)


def test_bc_a3_direct_3d_partial():
    d0, d1, d2, b0, b1, b2 = SHAPE_3D_PARTIAL
    src = make_input((d0, d1, d2))
    out = torch.full_like(src, -1.0)

    grid = (ceil_div(d0, b0), ceil_div(d1, b1), ceil_div(d2, b2))
    bc_a3_direct_3d_partial[grid](
        src,
        out,
        d0,
        d1,
        d2,
        src.stride(0),
        src.stride(1),
        src.stride(2),
        B0=b0,
        B1=b1,
        B2=b2,
    )

    assert_equal(out, src)


def test_bc_b2_direct_2d_full():
    rows, cols, row_block, col_block = SHAPE_2D
    src = make_input((rows, cols))
    out = torch.full((cols, rows), -1.0, device="npu", dtype=src.dtype)

    grid = (ceil_div(cols, col_block), ceil_div(rows, row_block))
    bc_b2_direct_2d_full[grid](
        src,
        out,
        rows,
        cols,
        src.stride(0),
        src.stride(1),
        out.stride(0),
        out.stride(1),
        ROWBLOCK=row_block,
        COLBLOCK=col_block,
    )

    assert_equal(out, src.transpose(0, 1).contiguous())


def test_bc_b3_direct_3d_full():
    d0, d1, d2, b0, b1, b2 = SHAPE_3D_FULL
    src = make_input((d0, d1, d2))
    out = torch.full((d2, d0, d1), -1.0, device="npu", dtype=src.dtype)

    grid = (ceil_div(d2, b0), ceil_div(d0, b1), ceil_div(d1, b2))
    bc_b3_direct_3d_full[grid](
        src,
        out,
        d0,
        d1,
        d2,
        src.stride(0),
        src.stride(1),
        src.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        B0=b0,
        B1=b1,
        B2=b2,
    )

    assert_equal(out, src.permute(2, 0, 1).contiguous())


def test_bc_c2_implicit_permute_2d_partial():
    rows, cols, row_block, col_block = SHAPE_2D
    src = make_input((rows, cols))
    out = torch.full((cols, rows), -1.0, device="npu", dtype=src.dtype)

    grid = (ceil_div(cols, col_block), ceil_div(rows, row_block))
    bc_c2_implicit_permute_2d_partial[grid](
        src,
        out,
        rows,
        cols,
        src.stride(0),
        src.stride(1),
        out.stride(0),
        out.stride(1),
        ROWBLOCK=row_block,
        COLBLOCK=col_block,
    )

    assert_equal(out, src.transpose(0, 1).contiguous())


def test_bc_c3_implicit_permute_3d_full():
    d0, d1, d2, b0, b1, b2 = SHAPE_3D_FULL
    src = make_input((d0, d1, d2))
    out = torch.full((d2, d1, d0), -1.0, device="npu", dtype=src.dtype)

    grid = (ceil_div(d2, b0), ceil_div(d1, b1), ceil_div(d0, b2))
    bc_c3_implicit_permute_3d_full[grid](
        src,
        out,
        d0,
        d1,
        d2,
        src.stride(0),
        src.stride(1),
        src.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        B0=b0,
        B1=b1,
        B2=b2,
    )

    assert_equal(out, src.permute(2, 1, 0).contiguous())


def test_bc_d2_direct_2d_advance_partial():
    rows, cols, row_block, col_block, advance_col = SHAPE_2D_ADVANCE
    src = make_input((rows, cols))
    out = torch.full_like(src, -1.0)

    grid = (ceil_div(rows, row_block),)
    bc_d2_direct_2d_advance_partial[grid](
        src,
        out,
        rows,
        cols,
        src.stride(0),
        src.stride(1),
        advance_col,
        ROWBLOCK=row_block,
        COLBLOCK=col_block,
    )

    expected = make_partial_copy_expected(src, advance_col, col_block)
    assert_equal(out, expected)


def test_bc_e2_direct_2d_for_partial():
    rows, cols, row_block, col_block, start_col, step_col, num_steps = SHAPE_2D_LOOP
    src = make_input((rows, cols))
    out = torch.full_like(src, -1.0)

    grid = (ceil_div(rows, row_block),)
    bc_e2_direct_2d_for_partial[grid](
        src,
        out,
        rows,
        cols,
        src.stride(0),
        src.stride(1),
        start_col,
        step_col,
        num_steps,
        ROWBLOCK=row_block,
        COLBLOCK=col_block,
    )

    expected = make_partial_copy_expected(src, start_col, col_block, step_col, num_steps)
    assert_equal(out, expected)
