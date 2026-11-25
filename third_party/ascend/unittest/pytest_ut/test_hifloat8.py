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
from triton.backends.ascend.utils import is_compile_on_910_95


@triton.jit
def _fill_hifloat8(dst, BLOCK: tl.constexpr, VALUE: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(dst + offsets, tl.full((BLOCK, ), VALUE, tl.hifloat8))


@triton.jit
def _decode_hifloat8(src, dst, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(src + offsets)
    tl.store(dst + offsets, values.to(tl.float32))


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_hifloat8_full(value):
    if not is_compile_on_910_95(triton.runtime.driver.active.get_current_target().arch):
        pytest.skip("HiFloat8 requires Ascend A5")

    block = 256
    storage = torch.empty((block, ), dtype=torch.uint8, device="npu")
    output = torch.empty((block, ), dtype=torch.float32, device="npu")
    encoded = triton.reinterpret(storage, tl.hifloat8)
    _fill_hifloat8[(1, )](encoded, block, value, enable_dynamic_cv_pipeline=False)
    _decode_hifloat8[(1, )](encoded, output, block, enable_dynamic_cv_pipeline=False)

    torch.testing.assert_close(output.cpu(), torch.full((block, ), value), atol=0, rtol=0)
