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

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401

import triton
import triton.language as tl


def _decode_reference(codes):
    """Evaluate sign, exponent and significand, independently of the lowering."""
    codes = np.asarray(codes, dtype=np.uint8)
    exponent = ((codes // 8) % 16).astype(np.int32)
    fraction = (codes % 8).astype(np.float64) / 8
    magnitude = np.where(exponent == 0, fraction * 2.0**-14, np.ldexp(1 + fraction, exponent - 15))
    return np.copysign(magnitude, np.where(codes >= 128, -1.0, 1.0))


def _encode_reference(values):
    """Choose the nearest value in the positive lattice; ties go away from zero.

    Match the upstream CUDA converter's actual PTX: narrowing saturates at
    1.75, including infinity and NaN. Decoding 0x7f still gives 1.875, despite
    the comment above that converter saying 1.750. The converter ignores the
    requested final downcast rounding mode.
    """
    values = np.asarray(values, dtype=np.float64)
    lattice = _decode_reference(np.arange(127, dtype=np.uint8))
    boundaries = (lattice[:-1] + lattice[1:]) / 2
    magnitude = np.minimum(np.nan_to_num(np.abs(values), nan=1.75, posinf=1.75), 1.75)
    codes = np.searchsorted(boundaries, magnitude, side="right").astype(np.uint8)
    return codes + np.signbit(values).astype(np.uint8) * 128


def _truncate_to_half_reference(values):
    """Numerically search the binary16 lattice for the value toward zero."""
    values = np.asarray(values, dtype=np.float64)
    positive_half = np.arange(0x7c00, dtype=np.uint16).view(np.float16).astype(np.float64)
    index = np.searchsorted(positive_half, np.abs(values), side="right") - 1
    magnitude = positive_half[np.clip(index, 0, positive_half.size - 1)]
    rounded = np.copysign(magnitude, values)
    return np.where(np.isnan(values), values, rounded)


def _decode_bfloat16_reference(bits):
    bits = np.asarray(bits, dtype=np.uint16)
    exponent = ((bits // 128) % 256).astype(np.int32)
    fraction = (bits % 128).astype(np.float64) / 128
    magnitude = np.where(exponent == 0, np.ldexp(fraction, -126), np.ldexp(1 + fraction, exponent - 127))
    magnitude = np.where(exponent == 255, np.where(fraction == 0, np.inf, np.nan), magnitude)
    return np.copysign(magnitude, np.where(bits >= 32768, -1.0, 1.0))


def _boundary_values(dtype):
    lattice = _decode_reference(np.arange(127, dtype=np.uint8)).astype(np.float32)
    midpoints = (lattice[:-1] + lattice[1:]) / 2
    # Immediately below a midpoint catches accidental fp32 -> fp16 RTNE:
    # that can round up to the midpoint before the custom ties-away conversion.
    positive = np.concatenate([
        lattice,
        np.nextafter(midpoints, np.float32(-np.inf)),
        midpoints,
        np.nextafter(midpoints, np.float32(np.inf)),
        np.array([1.875, 2.0, 65504, np.inf, np.nan], dtype=np.float32),
    ])
    values = np.concatenate([positive, -positive, np.array([0.0, -0.0, 2.0**-25], dtype=np.float32)])
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        return values.astype(dtype)


@triton.jit
def _convert(X, Y, N: tl.constexpr, DTYPE: tl.constexpr, ROUNDING: tl.constexpr, SCALAR: tl.constexpr,
             BLOCK: tl.constexpr):
    if SCALAR:
        offset = tl.program_id(0)
    else:
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + offset, offset < N, other=0)
    converted = value.to(DTYPE, fp_downcast_rounding=ROUNDING)
    tl.store(Y + offset, converted, offset < N)


def _launch(source, destination, size, dtype, rounding=None, scalar=False):
    _convert[(size if scalar else triton.cdiv(size, 256), )](source, destination, size, dtype, rounding, scalar, 256)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("layout", ["scalar", "block", "masked_tail"])
def test_fp8e4b15_decode_all_bit_patterns(dtype, layout):
    codes = np.arange(256, dtype=np.uint8)
    if layout == "masked_tail":
        codes = np.concatenate([codes, np.array([0x80, 0x7f, 0xff], dtype=np.uint8)])
    source = torch.from_numpy(codes).to("npu")
    output = torch.full((codes.size + 3, ), 13.0, dtype=getattr(torch, dtype), device="npu")
    _launch(triton.reinterpret(source, tl.float8e4b15), output, codes.size, getattr(tl, dtype),
            scalar=layout == "scalar")
    actual = output.cpu()
    expected = torch.from_numpy(_decode_reference(codes)).to(getattr(torch, dtype))
    # Compare bits to distinguish positive and negative zero as well.
    integer_dtype = torch.int32 if dtype == "float32" else torch.int16
    np.testing.assert_array_equal(actual[:codes.size].view(integer_dtype).numpy(), expected.view(integer_dtype).numpy())
    np.testing.assert_array_equal(actual[codes.size:].float().numpy(), np.full(3, 13.0, dtype=np.float32))


@pytest.mark.parametrize("rounding", [None, "rtne", "rtz"])
def test_fp8e4b15_encode_all_float16_bit_patterns(rounding):
    # Includes both zeros, every subnormal, every rounding tie, infinities,
    # and every signed NaN payload. No CPU floating-point conversion of input.
    values = np.arange(65536, dtype=np.uint16).view(np.float16)
    source = torch.from_numpy(values).to("npu")
    output = torch.full((values.size + 3, ), 0x55, dtype=torch.uint8, device="npu")
    _launch(source, triton.reinterpret(output, tl.float8e4b15), values.size, tl.float8e4b15, rounding)
    actual = output.cpu().numpy()
    np.testing.assert_array_equal(actual[:values.size], _encode_reference(values))
    np.testing.assert_array_equal(actual[values.size:], np.full(3, 0x55, dtype=np.uint8))


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("scalar", [False, True], ids=["masked_tail", "scalar"])
@pytest.mark.parametrize("rounding", [None, "rtne", "rtz"])
def test_fp8e4b15_encode_rounding_boundaries(dtype, scalar, rounding):
    values = _boundary_values(dtype)
    source = torch.from_numpy(values).to("npu")
    output = torch.full((values.size + 3, ), 0x55, dtype=torch.uint8, device="npu")
    _launch(source, triton.reinterpret(output, tl.float8e4b15), values.size, tl.float8e4b15, rounding, scalar)
    actual = output.cpu().numpy()
    reference_values = _truncate_to_half_reference(values) if dtype == "float32" else values
    expected = _encode_reference(reference_values)
    # Ascend's integer lowering preserves the source sign, including NaNs,
    # without an intermediate floating-point conversion that could erase it.
    np.testing.assert_array_equal(actual[:values.size], expected)
    np.testing.assert_array_equal(actual[values.size:], np.full(3, 0x55, dtype=np.uint8))


@pytest.mark.parametrize("rounding", [None, "rtne", "rtz"])
def test_fp8e4b15_encode_all_bfloat16_bit_patterns(rounding):
    bits = np.arange(65536, dtype=np.uint16)
    # Transfer raw storage without a floating-point operation that could
    # canonicalize NaN payloads or erase their original signs.
    source = torch.from_numpy(bits.view(np.int16)).to("npu")
    output = torch.full((bits.size + 3, ), 0x55, dtype=torch.uint8, device="npu")
    _launch(triton.reinterpret(source, tl.bfloat16), triton.reinterpret(output, tl.float8e4b15), bits.size,
            tl.float8e4b15, rounding)
    actual = output.cpu().numpy()
    expected = _encode_reference(_decode_bfloat16_reference(bits))
    np.testing.assert_array_equal(actual[:bits.size], expected)
    np.testing.assert_array_equal(actual[bits.size:], np.full(3, 0x55, dtype=np.uint8))


@pytest.mark.parametrize("dtype", ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"])
@pytest.mark.parametrize("scalar", [False, True], ids=["masked_tail", "scalar"])
def test_fp8e4b15_encode_integer_boundaries(dtype, scalar):
    if dtype == "bool":
        values = np.array([False, True, False, True, True, False, True])
    else:
        info = np.iinfo(dtype)
        if info.min < 0:
            values = np.array([info.min, info.min + 1, -2, -1, 0, 1, 2, info.max - 1, info.max], dtype=dtype)
        else:
            values = np.array([0, 1, 2, info.max // 2, info.max // 2 + 1, info.max - 1, info.max], dtype=dtype)
    storage = values
    if values.dtype.kind == "u" and values.dtype.itemsize > 1:
        # Torch NPU need not expose unsigned 16/32/64 tensor dtypes to exercise
        # Triton's unsigned loads and conversions from their actual bits.
        storage = values.view(f"int{values.dtype.itemsize * 8}")
    source = torch.from_numpy(storage).to("npu")
    if storage.dtype != values.dtype:
        source = triton.reinterpret(source, getattr(tl, dtype))
    output = torch.full((values.size + 3, ), 0x55, dtype=torch.uint8, device="npu")
    _launch(source, triton.reinterpret(output, tl.float8e4b15), values.size, tl.float8e4b15, scalar=scalar)
    actual = output.cpu().numpy()
    np.testing.assert_array_equal(actual[:values.size], _encode_reference(values))
    np.testing.assert_array_equal(actual[values.size:], np.full(3, 0x55, dtype=np.uint8))
