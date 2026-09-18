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
"""Regression test for the TritonToLinalg fix of negative strides.

getLastStrideOfReinterpretCastOp used to guard the static-stride fast path
with `> 0`, so a *negative* static stride of a memref::ReinterpretCastOp was
treated as a dynamic stride and resolved through the operand-value path. The
guard is now `!= ShapedType::kDynamic`, so any known static stride -
including negative values - is returned directly.

The kernel below is a 2D matmul whose B operand is loaded through an addptr
(`tl.load(ptr + offset)`) with a statically negative last-axis stride: the
host side passes the base pointer of a reversed view of B together with
`stride_bn = -1` as constexpr kernel arguments, so all strides fold into
static attributes on the ReinterpretCastOp. make_block_ptr is deliberately
not used: its loads are tagged GeneratedByMakeTensorPtrTAG and never reach
the lastStride logic in LoadStoreConverter.cpp.
"""

import torch
import torch_npu
import triton
import triton.language as tl


def _assert_close(actual, expected, *, atol=1e-3, rtol=1e-3):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@triton.jit
def _addptr_neg_stride2d_dot_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # a_tile[m, k] = A[m, k]
    a_off = offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    a_tile = tl.load(a_ptr + a_off)

    # b_tile[k, n] = B[k, N - 1 - n]: b_ptr points at the last column of B
    # and stride_bn is negative, walking the columns backwards
    b_off = offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    b_tile = tl.load(b_ptr + b_off)

    c_tile = tl.dot(a_tile, b_tile)

    c_off = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptr + c_off, c_tile)


def test_addptr_neg_stride2d_dot():
    M, N, K = 16, 16, 16
    a = torch.randn((M, K), dtype=torch.float16, device="npu")
    b = torch.randn((K, N), dtype=torch.float16, device="npu")
    c = torch.zeros((M, N), dtype=torch.float16, device="npu")

    # reversed view of B: base pointer at the last column, so the negative
    # stride walks the columns in reverse order
    flipped_b = b[:, b.shape[1] - 1:]
    fb_stride0 = b.stride(0)
    fb_stride1 = -b.stride(1)
    _addptr_neg_stride2d_dot_kernel[(1, )](a, flipped_b, c, M, N, K, a.stride(0), a.stride(1), fb_stride0, fb_stride1,
                                           c.stride(0), c.stride(1))
    expected = torch.matmul(a.to(torch.float32), torch.flip(b, [1]).to(torch.float32))
    _assert_close(c, expected.to(torch.float16))
