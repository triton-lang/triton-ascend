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

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

import torch
import torch_npu
import pytest
import test_common

# ---------------
# test native flip op (al.flip)
# ---------------


@triton.jit
def flip_kernel(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    y = al.flip(x, dim=0)
    tl.store(out_ptr + offs, y)


@pytest.mark.parametrize("dtype", ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'uint8'])
def test_flip(dtype):
    size = 64
    x = test_common.generate_tensor((size, ), dtype).npu()
    torch_ref = torch.flip(x, dims=[0])
    triton_res = torch.zeros(size, dtype=eval('torch.' + dtype)).npu()
    flip_kernel[(1, )](x, triton_res, size)
    assert (torch_ref == triton_res).all(), (torch_ref, triton_res)


# ---------------
# test tl.flip op
# ---------------
# `tl.flip` is lowered in the frontend as a bit-reversal idiom: reshape the
# 2^k-sized dim into k trailing 2s, xor-swap along each of them, then reshape
# back. TritonToLinalg recognizes that idiom and rewrites it into the native
# `ascend.flip`, which lowers through @triton_flip -> hfusion.flip -> hivm.vflip.
# Since hivm.vflip only flips the last (innermost) dim, only last-dim flips are
# rewritten; leading-dim flips stay on the bit-reversal path.


@triton.jit
def flip1d_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr, dim: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    y = tl.flip(x, dim=dim)
    tl.store(out_ptr + offs, y)


@triton.jit
def flip2d_kernel(x_ptr, out_ptr, ROWS: tl.constexpr, COLS: tl.constexpr, dim: tl.constexpr):
    r = tl.arange(0, ROWS)[:, None]
    c = tl.arange(0, COLS)[None, :]
    x = tl.load(x_ptr + r * COLS + c)
    y = tl.flip(x, dim=dim)
    tl.store(out_ptr + r * COLS + c, y)


def _assert_flip_lowered(h):
    # Numeric equality alone doesn't prove the rewrite fired: if the
    # bit-reversal pattern ever silently misses (e.g. upstream changes
    # tl.flip's lowering, or a canonicalizer folds the reduce/expand/broadcast/
    # xori chain), the backend falls back to the bit-reversal + expand_shape
    # path, which still produces a numerically correct result but reintroduces
    # the #1638 AlignBuffer overflow. Assert the lowered IR actually carries
    # the native flip (@triton_flip); a miss leaves no such symbol.
    code_str = h.asm["ttadapter"]
    assert "triton_flip" in code_str


def _assert_flip_not_lowered(h):
    # Leading-dim flips must NOT be rewritten to native flip: hivm.vflip only
    # supports the last dim, so a rewrite here would fail to compile. Lock in
    # that these stay on the bit-reversal path (which is still numerically
    # correct).
    code_str = h.asm["ttadapter"]
    assert "triton_flip" not in code_str


@pytest.mark.parametrize("size", [64, 8, 4])  # flip dim 2^6, 2^3, 2^2
@pytest.mark.parametrize("dtype", ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'uint8'])
def test_flip_1d(dtype, size):
    x = test_common.generate_tensor((size, ), dtype).npu()
    torch_ref = torch.flip(x, dims=[0])
    triton_res = torch.empty_like(x)
    h = flip1d_kernel[(1, )](x, triton_res, size, 0)
    assert (torch_ref == triton_res).all(), (torch_ref, triton_res)
    _assert_flip_lowered(h)


def _run_flip_2d(shape, dim, dtype):
    rows, cols = shape
    x = test_common.generate_tensor(shape, dtype).npu()
    torch_ref = torch.flip(x, dims=[dim])
    triton_res = torch.empty_like(x)
    h = flip2d_kernel[(1, )](x, triton_res, rows, cols, dim)
    assert (torch_ref == triton_res).all(), (torch_ref, triton_res)
    return h


@pytest.mark.parametrize("dtype", ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'uint8'])
@pytest.mark.parametrize("shape,dim", [((16, 64), 1),  # flip the trailing 2^6 dim (issue #1638 repro)
                                       ((64, 64), 1),  # square matrix, trailing dim
                                       ((4, 8), 1),  # trailing 2^3 dim -> [2, 2, 2]
                                       ((8, 4), 1),  # trailing 2^2 dim -> [2, 2] (smallest split, k=2)
                                       ])
def test_flip_2d_trailing(dtype, shape, dim):
    _assert_flip_lowered(_run_flip_2d(shape, dim, dtype))


@pytest.mark.parametrize("dtype", ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'uint8'])
@pytest.mark.parametrize("shape,dim", [((64, 16), 0),  # leading 2^6 dim -> vflip is last-dim only
                                       ((8, 32), 0),  # leading 2^3 dim -> [2, 2, 2]
                                       ((4, 4), 0),  # leading 2^2 dim -> [2, 2] (smallest split, k=2)
                                       ])
def test_flip_2d_leading(dtype, shape, dim):
    _assert_flip_not_lowered(_run_flip_2d(shape, dim, dtype))
