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

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _masked_atomic_add(x_ptr, mask_ptr, old_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = tl.load(mask_ptr + offsets) != 0
    old = tl.atomic_add(x_ptr + offsets, 1, mask=mask)
    tl.store(old_ptr + offsets, old)


def test_atomic_add_returns_zero_for_mask_holes():
    mask = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0], dtype=torch.int32)
    before = torch.arange(10, 10 + mask.numel(), dtype=torch.float32)
    x = before.npu()
    old = torch.full_like(x, -777)

    _masked_atomic_add[(1, )](x, mask.npu(), old, BLOCK=mask.numel())

    expected_old = torch.where(mask.bool(), before, 0)
    torch.testing.assert_close(old.cpu(), expected_old, rtol=0, atol=0)
    torch.testing.assert_close(x.cpu(), before + mask, rtol=0, atol=0)
