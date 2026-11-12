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

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from triton.backends.ascend.utils import AscendArchGen, arch_generation


@triton.jit
def _downcast_fp8(X, Y, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + offsets)
    tl.store(Y + offsets, value.to(Y.dtype.element_ty, fp_downcast_rounding="rtne"))


@pytest.mark.parametrize("src_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dst_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_downcast_clamping(src_dtype, dst_dtype):
    arch = triton.runtime.driver.active.get_current_target().arch
    if arch_generation(arch) != AscendArchGen.GEN5:
        pytest.skip("FP8 downcast is only supported on A5")

    limit = torch.finfo(dst_dtype).max
    values = [limit + 100, -limit - 100, float("inf"), -float("inf"), float("nan"), 0, 1, -1]
    expected = torch.tensor([limit, -limit, limit, -limit, float("nan"), 0, 1, -1]).repeat(256)
    source = torch.tensor(values, dtype=src_dtype).repeat(256).to("npu")
    output = torch.empty(source.shape, dtype=dst_dtype, device="npu")

    _downcast_fp8[(2, )](source, output, BLOCK=1024)

    # Compare on CPU: CANN fill/isnan do not support FP8 reference tensors.
    actual = output.cpu().to(torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
