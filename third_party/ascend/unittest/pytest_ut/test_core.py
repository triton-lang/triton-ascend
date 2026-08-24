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


def test_constexpr_shape():

    @triton.jit
    def kernel(X):
        off = tl.arange(0, 128 + 128)
        tl.store(X + off, off)

    x = torch.empty((256, ), dtype=torch.int32, device=device)
    kernel[(1, )](x)
    torch.testing.assert_close(x, torch.arange(0, 256, dtype=torch.int32, device=device))


def test_constexpr_scalar_shape():

    @triton.jit
    def kernel(X, s):
        off = tl.arange(0, 256)
        val = off % (256 // s)
        tl.store(X + off, val)

    x = torch.empty((256, ), dtype=torch.int32, device=device)
    kernel[(1, )](x, 32)
    expected = torch.arange(0, 256, dtype=torch.int32, device=device) % 8
    torch.testing.assert_close(x, expected)


def test_dtype():

    @triton.jit
    def kernel(X):
        dtype_x: tl.constexpr = X.dtype.element_ty
        tl.static_assert(dtype_x == tl.int32)
        tl.static_assert(dtype_x == tl.constexpr(tl.int32))
        tl.static_assert(dtype_x == tl.int8 or (dtype_x == tl.int16 or dtype_x == tl.int32))

    X = torch.empty(1, dtype=torch.int32, device=device)
    kernel[(1, )](X)


def test_aliasing():

    @triton.jit
    def aliasing_kernel(buffer, buffer2):
        triton.language.store(buffer, 1)

    buffer = torch.zeros(1, device=device)
    aliasing_kernel[(1, )](buffer, buffer)
    assert buffer[0] == 1


def test_temp_var_in_loop():

    @triton.jit
    def temp_in_loop(Z, N: tl.constexpr, BLOCK: tl.constexpr):
        acc = tl.full((BLOCK, ), 0, dtype=tl.int32)
        for i in range(N):
            if i == 0:
                temp = tl.full((BLOCK, ), 2, dtype=tl.int32)
                acc = temp
            else:
                acc += tl.full((BLOCK, ), 1, dtype=tl.int32)
            # Reuse the temporary and verify that it does not create invalid IR.
            temp = tl.full((BLOCK, ), 1, dtype=tl.int32)
            acc += temp
        z = Z + tl.arange(0, BLOCK)
        tl.store(z, acc)

    N = 10
    BLOCK = 32
    out = torch.empty((BLOCK, ), dtype=torch.int32, device=device)
    temp_in_loop[(1, )](out, N, BLOCK)
    acc = torch.full((BLOCK, ), 0, dtype=torch.int32, device=device)
    for i in range(N):
        if i == 0:
            temp = torch.full((BLOCK, ), 2, dtype=torch.int32, device=device)
            acc = temp
        else:
            acc += torch.full((BLOCK, ), 1, dtype=torch.int32, device=device)
        temp = torch.full((BLOCK, ), 1, dtype=torch.int32, device=device)
        acc += temp
    assert (acc == out).all()


def test_static_range():

    @triton.jit
    def loop_kernel(Z, N: tl.constexpr, step: tl.constexpr):
        acc = 0
        for i in tl.static_range(0, N, step=step):
            acc += i
        tl.store(Z, acc)

    N = 100
    step = 7
    Out = torch.empty(1, dtype=torch.int32, device=device)
    loop_kernel[(1, )](Out, N, step)
    Acc = torch.tensor([0], dtype=torch.int32, device=device)
    for i in range(0, N, step):
        Acc += i
    assert (Out == Acc).all(), (Out, Acc)
