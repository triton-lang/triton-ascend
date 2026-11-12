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
    result = value.to(Y.dtype.element_ty, fp_downcast_rounding="rtne")
    tl.store(Y + offsets, result)


@pytest.mark.parametrize("src_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dst_dtype", [tl.float8e4nv, tl.float8e5])
def test_fp8_downcast_saturation(src_dtype, dst_dtype):
    arch = triton.runtime.driver.active.get_current_target().arch
    if arch_generation(arch) != AscendArchGen.GEN5:
        pytest.skip("FP8 conversion requires A5")

    # Fixed encodings check saturation, ties-to-even, subnormals and signed zero.
    if dst_dtype == tl.float8e4nv:
        cases = [(0.0, 0x00), (1.0, 0x38), (1.0625, 0x38), (1.1875, 0x3A), (2**-9, 0x01), (2**-10, 0x00), (448.0, 0x7E),
                 (464.0, 0x7E), (480.0, 0x7E), (65504.0, 0x7E), (float("inf"), 0x7E), (0.5, 0x30), (1.125, 0x39),
                 (432.0, 0x7E), (57344.0, 0x7E)]
        torch_dst_dtype = torch.float8_e4m3fn
    else:
        cases = [(0.0, 0x00), (1.0, 0x3C), (1.125, 0x3C), (1.375, 0x3E), (2**-16, 0x01), (2**-17, 0x00),
                 (57344.0, 0x7B), (61440.0, 0x7B), (65504.0, 0x7B), (float("inf"), 0x7B), (0.5, 0x38), (1.25, 0x3D),
                 (49152.0, 0x7A), (448.0, 0x5F), (512.0, 0x60)]
        torch_dst_dtype = torch.float8_e5m2
    cases.append((float("nan"), 0))
    signed_cases = [(value * sign, bits | sign_bit) for sign, sign_bit in [(1, 0), (-1, 0x80)] for value, bits in cases]
    source_cpu = torch.tensor([value for value, _ in signed_cases], dtype=src_dtype)
    source = source_cpu.to("npu")
    output = torch.empty(source.shape, dtype=torch.uint8, device="npu")
    _downcast_fp8[(1, )](source, triton.reinterpret(output, dst_dtype), source.numel())

    actual_bits = output.cpu()
    expected_bits = torch.tensor([bits for _, bits in signed_cases], dtype=torch.uint8)
    nan_mask = torch.isnan(source_cpu)
    assert torch.isnan(actual_bits.view(torch_dst_dtype).float()[nan_mask]).all()
    torch.testing.assert_close(actual_bits[~nan_mask], expected_bits[~nan_mask], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["max", "min", "inf", "-inf", "nan"])
@pytest.mark.parametrize("dst_dtype,torch_dst_dtype", [(tl.float8e4nv, torch.float8_e4m3fn),
                                                       (tl.float8e5, torch.float8_e5m2)])
@pytest.mark.parametrize("src_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fp8_downcast_clamping(src_dtype, dst_dtype, torch_dst_dtype, mode):
    arch = triton.runtime.driver.active.get_current_target().arch
    if arch_generation(arch) != AscendArchGen.GEN5:
        pytest.skip("FP8 conversion requires A5")

    # Match the community clamping test's values, shapes and RTNE conversion.
    limit = torch.finfo(torch_dst_dtype).max
    if mode in ("max", "min"):
        test_value = limit + 100.0
    elif mode in ("inf", "-inf"):
        test_value = float("inf")
    else:
        test_value = float("nan")
    expected = limit
    if mode in ("min", "-inf"):
        test_value *= -1
        expected *= -1

    block = 1024
    source = torch.full((2 * block, ), test_value, dtype=src_dtype, device="npu")
    output = torch.empty(source.shape, dtype=torch.uint8, device="npu")
    _downcast_fp8[(2, )](source, triton.reinterpret(output, dst_dtype), block)

    # CPU float32 checks do not require device FP8 isnan/full_like support.
    actual = output.cpu().view(torch_dst_dtype).float()
    if mode == "nan":
        assert torch.isnan(actual).all()
    else:
        torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=0, atol=0)
