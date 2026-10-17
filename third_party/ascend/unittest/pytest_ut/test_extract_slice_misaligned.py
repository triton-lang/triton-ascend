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

import torch
import torch_npu

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

import pytest
import test_common


@triton.jit
def extract_slice_bf16_extf_kernel(x_ptr, out_ptr, N: tl.constexpr, OFF: tl.constexpr, SZ: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    x_f32 = al.cast(x, tl.float32)
    sub = al.extract_slice(x_f32, [OFF], [SZ], [1])
    tl.store(out_ptr + tl.arange(0, SZ), sub)


@pytest.mark.parametrize("off", [8, 16, 24])
def test_extract_slice_bf16_extf_misaligned(off):
    size, sz = 64, 16
    x = test_common.generate_tensor((size, ), 'bfloat16').npu()
    out = torch.zeros(sz, dtype=torch.float32).npu()
    extract_slice_bf16_extf_kernel[(1, )](x, out, size, off, sz)
    torch.testing.assert_close(out, x[off:off + sz].to(torch.float32))
