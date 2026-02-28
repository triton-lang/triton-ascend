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

import triton
import triton.language as tl
import test_common

import torch
import torch_npu


def copy(x):
    return x.clone()


@triton.jit
def copy_kernel(in_ptr, out_ptr, N: tl.constexpr, NUMEL):
    idx_block = tl.arange(0, N)
    is_valid = N <= NUMEL
    x = tl.load(in_ptr + idx_block, mask=idx_block < N)
    mask_i1 = is_valid[:, None] & (idx_block < N)[None, :]
    tl.store(out_ptr + idx_block[None, :], x[None, :], mask=mask_i1)


def test_complex_mask_copy():
    N = 1024
    x = torch.randn(N, dtype=torch.float32).npu()
    y = torch.empty_like(x).npu()
    copy_kernel[(1, )](x, y, N=N, NUMEL=N)
    torch.testing.assert_close(x, y)
