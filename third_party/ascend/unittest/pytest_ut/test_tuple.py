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

import pytest
import torch
import torch_npu  # noqa: F401  # Registers the "npu" device with PyTorch.

import triton
import triton.language as tl

device = "npu"


@triton.jit
def _tuple_ret(a, b):
    return a + b, a - b, a * b


def test_assign_return():

    @triton.jit
    def with_fn(X, Y, A, B, C):
        x = tl.load(X)
        y = tl.load(Y)
        a, b, c = _tuple_ret(x, y)
        tl.store(A, a)
        tl.store(B, b)
        tl.store(C, c)

    @triton.jit
    def without_fn(X, Y, A, B, C):
        x = tl.load(X)
        y = tl.load(Y)
        a, b, c = x + y, x - y, x * y
        tl.store(A, a)
        tl.store(B, b)
        tl.store(C, c)

    x = torch.tensor([1.3], device=device, dtype=torch.float32)
    y = torch.tensor([1.9], device=device, dtype=torch.float32)
    a_tri = torch.tensor([0], device=device, dtype=torch.float32)
    b_tri = torch.tensor([0], device=device, dtype=torch.float32)
    c_tri = torch.tensor([0], device=device, dtype=torch.float32)
    for kernel in [with_fn, without_fn]:
        kernel[(1, )](x, y, a_tri, b_tri, c_tri, num_warps=1)
        a_ref, b_ref, c_ref = x + y, x - y, x * y
        assert a_tri == a_ref
        assert b_tri == b_ref
        assert c_tri == c_ref


def test_eq():

    @triton.jit
    def fn(ret_ptrs):
        tl.store(ret_ptrs + 0, (1, 2) == (1, 2))
        tl.store(ret_ptrs + 1, (1, 2) == (1, 1))
        tl.store(ret_ptrs + 2, tl.tuple((1, 2)) == (1, 2))
        tl.store(ret_ptrs + 3, tl.tuple((1, 2)) == (1, 3))

    rets = torch.zeros((4, ), dtype=torch.int32, device=device)
    fn[(1, )](rets)
    assert rets[0].item() == 1
    assert rets[1].item() == 0
    assert rets[2].item() == 1
    assert rets[3].item() == 0


def test_add():

    @triton.jit
    def fn(ret_ptrs):
        tuple0 = ((0, 1)) + (2, 3)
        for i in tl.static_range(4):
            tl.store(ret_ptrs + i, tuple0[i])
        tuple1 = tl.tuple((4, 5)) + (6, 7)
        for i in tl.static_range(4):
            tl.store(ret_ptrs + 4 + i, tuple1[i])

    rets = torch.zeros((8, ), dtype=torch.int32, device=device)
    fn[(1, )](rets)
    torch.testing.assert_close(rets.cpu(), torch.arange(8, dtype=torch.int32))


def test_modifying_tuples():

    @triton.jit
    def set_tuple_value_at_idx():
        t = tl.tuple([5, 6, 7])
        t[0] = 0

    with pytest.raises(triton.CompilationError):
        set_tuple_value_at_idx[(1, )]()
