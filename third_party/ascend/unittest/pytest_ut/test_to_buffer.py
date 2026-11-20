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

import triton
import triton.language as tl
from triton.compiler import ASTSource
from triton.backends.ascend.utils import is_compile_on_910_95
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al

target = triton.runtime.driver.active.get_current_target()


@triton.jit
def to_buffer():
    a = tl.full((32, 2, 4), 0, dtype=tl.int64)
    a_buf = bl.to_buffer(a)
    b = tl.full((32, 2, 4), 0, dtype=tl.int64)
    b_buf = bl.to_buffer(b, al.ascend_address_space.UB)
    c = tl.full((32, 2, 4), 0, dtype=tl.int64)
    c_buf = bl.to_buffer(c, al.ascend_address_space.L1)
    d = tl.full((32, 2, 4), 0, dtype=tl.int64)
    d_buf = bl.to_buffer(d, al.ascend_address_space.L0A)
    e = tl.full((32, 2, 4), 0, dtype=tl.int64)
    e_buf = bl.to_buffer(e, al.ascend_address_space.L0B)
    f = tl.full((32, 2, 4), 0, dtype=tl.int64)
    f_buf = bl.to_buffer(f, al.ascend_address_space.L0C)


def test_to_buffer():
    src = ASTSource(
        fn=to_buffer,
        constexprs={},
        signature={},
    )
    triton.compile(src=src, target=target)


@triton.jit
def _to_buffer_after_if(X, Y, Out, Selector, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    y = tl.load(Y + offsets)
    if tl.load(Selector) != 0:
        selected = x
    else:
        selected = y
    # Keep the conversion live across a tensor-valued scf.if, as in the
    # ping/pong buffer selection used by manually scheduled attention kernels.
    src = bl.to_buffer(selected, al.ascend_address_space.UB)
    dst = bl.alloc(selected.dtype, [BLOCK], al.ascend_address_space.UB)
    al.copy(src, dst)
    tl.store(Out + offsets, bl.to_tensor(dst))


@pytest.mark.skipif(not is_compile_on_910_95(target.arch), reason="al.copy requires Ascend 950 (A5)")
@pytest.mark.parametrize("take_first", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_to_buffer_after_if(take_first, dtype):
    x = torch.arange(64, dtype=dtype, device="npu")
    y = -x - 1
    output = torch.empty_like(x)
    selector = torch.tensor(int(take_first), dtype=torch.int32, device="npu")
    kernel = _to_buffer_after_if[(1, )](x, y, output, selector, BLOCK=64)
    torch.testing.assert_close(output, x if take_first else y, rtol=0, atol=0)
    assert "bufferization.to_memref " in kernel.asm["ttadapter"]
    assert "bufferization.to_buffer " not in kernel.asm["ttadapter"]
