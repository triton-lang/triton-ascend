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
def _fp32_downcast_rtz(X, Y, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    value = tl.load(X + offsets)
    result = value.to(Y.dtype.element_ty, fp_downcast_rounding="rtz")
    tl.store(Y + offsets, result)


@pytest.mark.parametrize(
    "dtype,cases",
    [
        (
            tl.float16, [(0x00000000, 0x0000),  # Zero.
                         (0x3F800000, 0x3C00),  # Exactly representable one.
                         (0x3F801001, 0x3C00),  # Just above a midpoint; RTNE rounds up.
                         (0x3F803000, 0x3C01),  # Odd midpoint; RTNE rounds up.
                         (0x33007089, 0x0000),  # Original community underflow failure.
                         (0x33800000, 0x0001),  # Smallest subnormal.
                         (0x387FFFFF, 0x03FF),  # Just below the smallest normal.
                         (0x477FE000, 0x7BFF),  # Largest finite value.
                         ]),
        (
            tl.bfloat16, [
                (0x00000000, 0x0000),
                (0x3F800000, 0x3F80),
                (0x3F808001, 0x3F80),
                (0x3F818000, 0x3F81),
                (0x3F80FFFF, 0x3F80),  # Just below the next representable value.
                (0x3F810000, 0x3F81),
                (0x00800000, 0x0080),  # Smallest normal.
                (0x7F7F0000, 0x7F7F),
            ]),
        (
            tl.float8e5, [
                (0x00000000, 0x00),
                (0x3F800000, 0x3C),
                (0x3F900001, 0x3C),
                (0x3FB00000, 0x3D),
                (0x37000001, 0x00),  # Underflow; RTNE produces a subnormal.
                (0x37800000, 0x01),
                (0x387FFFFF, 0x03),
                (0x47600000, 0x7B),
            ]),
    ],
    ids=["fp16", "bf16", "fp8e5m2"],
)
def test_fp32_downcast_rtz(dtype, cases):
    if dtype == tl.float8e5:
        arch = triton.runtime.driver.active.get_current_target().arch
        if arch_generation(arch) != AscendArchGen.GEN5:
            pytest.skip("FP32 to FP8 E5M2 RTZ conversion is only supported on A5")

    # Fixed IEEE encodings provide an oracle independent of the device cast
    # and PyTorch's default round-to-nearest conversion. Include both signs
    # and compare raw bits so that negative zero is checked too.
    sign_bit = 1 << (dtype.primitive_bitwidth - 1)
    signed_cases = [(src | (sign << 31), dst | (sign * sign_bit)) for sign in (0, 1) for src, dst in cases]
    source_bits = torch.tensor([src for src, _ in signed_cases], dtype=torch.int64).to(torch.int32).repeat(64)
    source = source_bits.view(torch.float32).to("npu")
    expected = torch.tensor([dst for _, dst in signed_cases], dtype=torch.int32).repeat(64)
    storage_dtype = torch.int8 if dtype.primitive_bitwidth == 8 else torch.int16
    output = torch.empty(source.shape, dtype=storage_dtype, device="npu")

    _fp32_downcast_rtz[(1, )](source, triton.reinterpret(output, dtype), BLOCK=source.numel())

    mask = (1 << dtype.primitive_bitwidth) - 1
    actual = output.cpu().to(torch.int32) & mask
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
