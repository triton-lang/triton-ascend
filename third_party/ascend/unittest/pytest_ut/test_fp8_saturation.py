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


@triton.jit
def _clamp_reshape_downcast_fp8(X, Y, LIMIT: tl.constexpr, PROPAGATE_NAN: tl.constexpr):
    offsets = tl.arange(0, 4)[:, None] * 8 + tl.arange(0, 8)[None, :]
    value = tl.load(X + offsets)
    clamped = tl.clamp(value, -LIMIT, LIMIT, propagate_nan=PROPAGATE_NAN)
    result = clamped.reshape((32, )).to(Y.dtype.element_ty, fp_downcast_rounding="rtne")
    tl.store(Y + tl.arange(0, 32), result)


@pytest.mark.parametrize("dst_dtype,torch_dst_dtype", [(tl.float8e4nv, torch.float8_e4m3fn),
                                                       (tl.float8e5, torch.float8_e5m2)])
@pytest.mark.parametrize("propagate_nan", [tl.PropagateNan.NONE, tl.PropagateNan.ALL])
def test_fp8_clamp_reshape_downcast(dst_dtype, torch_dst_dtype, propagate_nan):
    arch = triton.runtime.driver.active.get_current_target().arch
    if arch_generation(arch) != AscendArchGen.GEN5:
        pytest.skip("FP8 conversion requires A5")

    limit = torch.finfo(torch_dst_dtype).max
    values = [
        0.0, 1.0, 1.0625, 1.125, 1.1875, 2**-17, 2**-16, 2**-10, 2**-9, limit / 2, limit, limit + 1, 2 * limit, 65504.0,
        float("inf"),
        float("nan")
    ]
    source_cpu = torch.tensor([value * sign for sign in (1, -1) for value in values], dtype=torch.float32).reshape(4, 8)
    source = source_cpu.to("npu")
    output = torch.empty((source_cpu.numel(), ), dtype=torch.uint8, device="npu")
    _clamp_reshape_downcast_fp8[(1, )](source, triton.reinterpret(output, dst_dtype), limit, propagate_nan)

    # Match clamp's two NaN modes on CPU before rounding to FP8. No device FP8
    # operations are needed; exact byte checks also distinguish +0 from -0.
    lower = torch.tensor(-limit, dtype=source_cpu.dtype)
    upper = torch.tensor(limit, dtype=source_cpu.dtype)
    if propagate_nan == tl.PropagateNan.ALL:
        reference = torch.maximum(lower, torch.minimum(source_cpu, upper))
    else:
        reference = torch.fmax(lower, torch.fmin(source_cpu, upper))
    reference = reference.flatten()
    expected_bits = reference.to(torch_dst_dtype).view(torch.uint8)
    actual_bits = output.cpu()
    nan_mask = torch.isnan(reference)
    assert torch.isnan(actual_bits.view(torch_dst_dtype).float()[nan_mask]).all()
    torch.testing.assert_close(actual_bits[~nan_mask], expected_bits[~nan_mask], rtol=0, atol=0)


@triton.jit
def _bounded_downcast_fp8(X, Y, LIMIT: tl.constexpr, PROPAGATE_NAN: tl.constexpr, PATH: tl.constexpr):
    offsets = tl.arange(0, 4)[:, None] * 8 + tl.arange(0, 8)[None, :]
    value = tl.load(X + offsets)
    if PATH == "minmax":
        bounded = tl.maximum(tl.minimum(value, LIMIT, propagate_nan=PROPAGATE_NAN), -LIMIT, propagate_nan=PROPAGATE_NAN)
    elif PATH == "maxmin":
        bounded = tl.minimum(tl.maximum(value, -LIMIT, propagate_nan=PROPAGATE_NAN), LIMIT, propagate_nan=PROPAGATE_NAN)
    else:
        clamped = tl.clamp(value, -LIMIT, LIMIT, propagate_nan=PROPAGATE_NAN)
        if PATH == "select":
            narrow = tl.clamp(value, -LIMIT / 2, LIMIT / 2, propagate_nan=PROPAGATE_NAN)
            bounded = tl.where(offsets % 2 == 0, clamped, narrow)
        elif PATH == "abs":
            bounded = tl.abs(clamped)
        elif PATH == "neg":
            bounded = -clamped
        elif PATH == "trans":
            bounded = clamped.T
        elif PATH == "broadcast":
            bounded = tl.broadcast_to(clamped[:, None, :], (4, 2, 8))
        elif PATH == "minnum_nan_fallback":
            bounded = tl.minimum(clamped, 2 * LIMIT, propagate_nan=tl.PropagateNan.NONE)
        else:
            tl.static_assert(PATH == "maxnum_nan_fallback")
            bounded = tl.maximum(clamped, -2 * LIMIT, propagate_nan=tl.PropagateNan.NONE)
    OUT_BLOCK: tl.constexpr = 64 if PATH == "broadcast" else 32
    result = bounded.reshape((OUT_BLOCK, )).to(Y.dtype.element_ty, fp_downcast_rounding="rtne")
    tl.store(Y + tl.arange(0, OUT_BLOCK), result)


@pytest.mark.parametrize("dst_dtype,torch_dst_dtype", [(tl.float8e4nv, torch.float8_e4m3fn),
                                                       (tl.float8e5, torch.float8_e5m2)])
@pytest.mark.parametrize("path,propagate_nan",
                         [(path, propagate_nan)
                          for path in ("minmax", "maxmin", "select", "abs", "neg", "trans", "broadcast")
                          for propagate_nan in (tl.PropagateNan.NONE, tl.PropagateNan.ALL)] +
                         [("minnum_nan_fallback", tl.PropagateNan.ALL), ("maxnum_nan_fallback", tl.PropagateNan.ALL)])
def test_fp8_bounded_downcast(dst_dtype, torch_dst_dtype, path, propagate_nan):
    arch = triton.runtime.driver.active.get_current_target().arch
    if arch_generation(arch) != AscendArchGen.GEN5:
        pytest.skip("FP8 conversion requires A5")

    limit = torch.finfo(torch_dst_dtype).max
    values = [
        0.0, 1.0, 1.0625, 1.125, 1.1875, 2**-17, 2**-16, 2**-10, 2**-9, limit / 2, limit, limit + 1, 2 * limit, 65504.0,
        float("inf"),
        float("nan")
    ]
    source_cpu = torch.tensor([value * sign for sign in (1, -1) for value in values], dtype=torch.float32).reshape(4, 8)
    source = source_cpu.to("npu")
    output_size = 64 if path == "broadcast" else 32
    output = torch.empty((output_size, ), dtype=torch.uint8, device="npu")
    _bounded_downcast_fp8[(1, )](source, triton.reinterpret(output, dst_dtype), limit, propagate_nan, path)

    minimum = torch.minimum if propagate_nan == tl.PropagateNan.ALL else torch.fmin
    maximum = torch.maximum if propagate_nan == tl.PropagateNan.ALL else torch.fmax
    lower = torch.tensor(-limit, dtype=source_cpu.dtype)
    upper = torch.tensor(limit, dtype=source_cpu.dtype)
    reference = maximum(lower, minimum(source_cpu, upper))
    if path == "maxmin":
        reference = minimum(maximum(source_cpu, lower), upper)
    elif path == "select":
        narrow = maximum(lower / 2, minimum(source_cpu, upper / 2))
        condition = torch.arange(32).reshape(4, 8) % 2 == 0
        reference = torch.where(condition, reference, narrow)
    elif path == "abs":
        reference = reference.abs()
    elif path == "neg":
        reference = -reference
    elif path == "trans":
        reference = reference.T
    elif path == "broadcast":
        reference = reference[:, None, :].expand(4, 2, 8)
    elif path == "minnum_nan_fallback":
        reference = torch.fmin(reference, 2 * upper)
    elif path == "maxnum_nan_fallback":
        reference = torch.fmax(reference, 2 * lower)

    # A NaN from the clamp can select the out-of-range minnum/maxnum operand.
    # Ordered saturation must still clamp these values and preserve signed zero.
    reference = torch.where(reference > upper, upper, reference)
    reference = torch.where(reference < lower, lower, reference).flatten()
    expected_bits = reference.to(torch_dst_dtype).view(torch.uint8)
    actual_bits = output.cpu()
    nan_mask = torch.isnan(reference)
    assert torch.isnan(actual_bits.view(torch_dst_dtype).float()[nan_mask]).all()
    torch.testing.assert_close(actual_bits[~nan_mask], expected_bits[~nan_mask], rtol=0, atol=0)
