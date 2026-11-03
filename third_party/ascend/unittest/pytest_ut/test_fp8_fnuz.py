# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT
"""FNUZ byte storage and numerical conversion, with independent CPU oracles."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401

import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.ascend import utils

FORMATS = [(tl.float8e4b8, torch.float8_e4m3fnuz), (tl.float8e5b16, torch.float8_e5m2fnuz)]


def _boundaries(dtype):
    lattice = torch.arange(128, dtype=torch.uint8).view(dtype).float().numpy()
    upper = lattice[-1] + lattice[-1] - lattice[-2]
    midpoints = (lattice + np.r_[lattice[1:], upper]) / 2
    positive = np.r_[
        lattice,
        np.nextafter(midpoints, np.float32(-np.inf)), midpoints,
        np.nextafter(midpoints, np.float32(np.inf)),
        np.array([np.inf, np.nan, np.finfo(np.float32).
                  max, np.nextafter(np.float32(0), np.float32(1))], dtype=np.float32)]
    return np.r_[positive, -positive].astype(np.float32)


def _toward_zero_reference(source, dtype):
    values = source.float().numpy()
    lattice = torch.arange(128, dtype=torch.uint8).view(dtype).float().numpy()
    codes = np.clip(np.searchsorted(lattice, np.abs(values), side='right') - 1, 0, 127).astype(np.uint8)
    result = codes | np.where((codes != 0) & np.signbit(values), 128, 0).astype(np.uint8)
    return torch.from_numpy(np.where(np.isfinite(values), result, 0x80).astype(np.uint8))


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


@pytest.mark.parametrize('triton_dtype,torch_dtype', FORMATS)
@pytest.mark.parametrize('destination', ['float16', 'bfloat16', 'float32'])
@pytest.mark.parametrize('scalar', [False, True])
def test_fnuz_decode_all_bytes(triton_dtype, torch_dtype, destination, scalar):
    codes = torch.arange(256, dtype=torch.uint8)
    # A masked tail tests that a zero padding value uses the software encoding.
    codes = torch.cat([codes, torch.tensor([0, 0x80, 0xff], dtype=torch.uint8)])
    source = codes.to('npu')
    output = torch.full((codes.numel() + 3, ), 13., dtype=getattr(torch, destination), device='npu')
    _convert[(codes.numel() if scalar else triton.cdiv(codes.numel(), 256), )](triton.reinterpret(source,
                                                                                                  triton_dtype), output,
                                                                               codes.numel(), getattr(tl, destination),
                                                                               None, scalar, 256)
    actual = output.cpu()
    expected = codes.view(torch_dtype).to(getattr(torch, destination))
    torch.testing.assert_close(actual[:codes.numel()], expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(actual[codes.numel():], torch.full((3, ), 13., dtype=getattr(torch, destination)))


@pytest.mark.parametrize('triton_dtype,torch_dtype', FORMATS)
@pytest.mark.parametrize('source_dtype', ['float16', 'bfloat16', 'float32'])
@pytest.mark.parametrize('rounding', ['rtne', 'rtz'])
def test_fnuz_encode_boundaries(triton_dtype, torch_dtype, source_dtype, rounding):
    if source_dtype == 'float16':
        source = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.float16)
    elif source_dtype == 'bfloat16':
        source = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
    else:
        random_bits = np.random.default_rng(271).integers(0, 2**32, 65536, dtype=np.uint32)
        source = torch.from_numpy(np.r_[_boundaries(torch_dtype), random_bits.view(np.float32)])
    expected = (source.to(torch_dtype).view(torch.uint8) if rounding == 'rtne' else _toward_zero_reference(
        source, torch_dtype))
    output = torch.full((source.numel() + 3, ), 0x55, dtype=torch.uint8, device='npu')
    _convert[(triton.cdiv(source.numel(), 256), )](source.to('npu'), triton.reinterpret(output, triton_dtype),
                                                   source.numel(), triton_dtype, rounding, False, 256)
    actual = output.cpu()
    assert torch.equal(actual[:source.numel()], expected)
    assert torch.equal(actual[source.numel():], torch.full((3, ), 0x55, dtype=torch.uint8))


@triton.jit
def _copy_bytes(X, Y, N: tl.constexpr):
    offset = tl.program_id(0) * 256 + tl.arange(0, 256)
    value = tl.load(X + offset, offset < N, other=0)
    tl.store(Y + offset, value, offset < N)


@pytest.mark.parametrize('triton_dtype,torch_dtype', FORMATS)
def test_fnuz_load_store_preserves_bytes(triton_dtype, torch_dtype):
    codes = torch.arange(256, dtype=torch.uint8).repeat(2)[:511]
    source = codes.to('npu')
    output = torch.full((514, ), 0x55, dtype=torch.uint8, device='npu')
    _copy_bytes[(2, )](triton.reinterpret(source, triton_dtype), triton.reinterpret(output, triton_dtype), 511)
    assert torch.equal(output[:511].cpu(), codes)
    assert torch.equal(output[511:].cpu(), torch.full((3, ), 0x55, dtype=torch.uint8))


@triton.jit
def _constant_dot(Y, DTYPE: tl.constexpr):
    lhs = tl.full((16, 16), 1.5, DTYPE)
    rhs = tl.full((16, 16), -0.5, DTYPE)
    result = tl.dot(lhs, rhs)
    offset = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
    tl.store(Y + offset, result)


@pytest.mark.parametrize('triton_dtype,torch_dtype', FORMATS)
def test_fnuz_constants_dot_are_numeric(triton_dtype, torch_dtype):
    output = torch.empty((16, 16), dtype=torch.float32, device='npu')
    _constant_dot[(1, )](output, triton_dtype)
    assert torch.equal(output.cpu(), torch.full((16, 16), -12., dtype=torch.float32))


@pytest.mark.parametrize('triton_dtype,torch_dtype', FORMATS)
def test_fnuz_native_dtype_default_is_unchanged(triton_dtype, torch_dtype):
    context = ir.context()
    ir.load_dialects(context)
    builder = ir.builder(context)
    builder.options = SimpleNamespace(supported_fp8_dtypes=('fp8e4b8', 'fp8e5b16'))
    assert 'FNUZ' in str(triton_dtype.to_ir(builder))
    builder.options.software_fp8_dtypes = ('fp8e4b8', 'fp8e5b16')
    assert str(triton_dtype.to_ir(builder)) == 'i8'
    assert utils.convert_sigtype_to_int(str(triton_dtype)) == -1
