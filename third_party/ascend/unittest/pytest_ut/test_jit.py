# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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
import torch_npu  # noqa: F401  # Registers the "npu" device with PyTorch.

import triton
import triton.language as tl

device = "npu"


@triton.jit
def _impl(value=10):
    return value


def test_default():
    value = 5
    ret0 = torch.zeros(1, dtype=torch.int32, device=device)
    ret1 = torch.zeros(1, dtype=torch.int32, device=device)

    @triton.jit
    def _kernel(ret0, ret1, value=3):
        tl.store(ret0, _impl())
        tl.store(ret1, _impl(value))

    _kernel[(1, )](ret0, ret1, value)
    assert ret0.item() == 10
    assert ret1.item() == value

    _kernel[(1, )](ret0, ret1)
    assert ret0.item() == 10
    assert ret1.item() == 3


@triton.jit
def mul_jit_function(x, y):
    return x * y


@triton.jit
def apply_binary_op(x, combine_op):
    return combine_op(x, x)


def test_jit_function_arg():

    @triton.jit
    def square_kernel_jit_function(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        in_data = tl.load(in_ptr + offsets)
        out_data = apply_binary_op(in_data, mul_jit_function)
        tl.store(out_ptr + offsets, out_data)

    BLOCK_SIZE = 16
    x = torch.full((BLOCK_SIZE, ), 3.0, device=device)
    out = torch.empty((BLOCK_SIZE, ), device=device)
    expect = torch.full((BLOCK_SIZE, ), 9.0, dtype=x.dtype, device=device)

    square_kernel_jit_function[(1, )](x, out, BLOCK_SIZE)

    torch.testing.assert_close(out, expect)
