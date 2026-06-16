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

import test_common


# ---------------------------------------------------------------------------
# Reference: torch.cummin with reverse support
# ---------------------------------------------------------------------------
def torch_cummin(x, dim, reverse):
    if reverse:
        x = torch.flip(x, [dim])
    values = torch.cummin(x, dim=dim).values
    if reverse:
        values = torch.flip(values, [dim])
    return values


# ---------------------------------------------------------------------------
# Triton combine function  –  tl.minimum is recognized by the Ascend backend
# ScanConverter and lowered to the dedicated cummin template kernel.
# ---------------------------------------------------------------------------
@triton.jit
def _combine_min(a, b):
    return tl.minimum(a, b)


# ---------------------------------------------------------------------------
# Triton kernels (1-D / 2-D / 3-D)
# ---------------------------------------------------------------------------
@triton.jit
def _kernel_1d(
    out_ptr,
    in_ptr,
    reverse: tl.constexpr,
    N0: tl.constexpr,
    B0: tl.constexpr,
):
    tl.static_assert(N0 == B0)
    idx = tl.arange(0, B0)
    x = tl.load(in_ptr + idx)
    y = tl.associative_scan(x, axis=0, reverse=reverse, combine_fn=_combine_min)
    tl.store(out_ptr + idx, y)


@triton.jit
def _kernel_2d(
    out_ptr,
    in_ptr,
    dim: tl.constexpr,
    reverse: tl.constexpr,
    N0: tl.constexpr,
    N1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
):
    tl.static_assert(N0 == B0)
    tl.static_assert(N1 == B1)
    i0 = tl.arange(0, B0)
    i1 = tl.arange(0, B1)
    idx = i0[:, None] * N1 + i1[None, :]
    x = tl.load(in_ptr + idx)
    y = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=_combine_min)
    tl.store(out_ptr + idx, y)


@triton.jit
def _kernel_3d(
    out_ptr,
    in_ptr,
    dim: tl.constexpr,
    reverse: tl.constexpr,
    N0: tl.constexpr,
    N1: tl.constexpr,
    N2: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
):
    tl.static_assert(N0 == B0)
    tl.static_assert(N1 == B1)
    tl.static_assert(N2 == B2)
    i0 = tl.arange(0, B0)
    i1 = tl.arange(0, B1)
    i2 = tl.arange(0, B2)
    idx = i0[:, None, None] * N1 * N2 + i1[None, :, None] * N2 + i2[None, None, :]
    x = tl.load(in_ptr + idx)
    y = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=_combine_min)
    tl.store(out_ptr + idx, y)


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------
def triton_cummin(x, dim, reverse):
    out = torch.empty_like(x)
    ndim = x.ndim
    s = x.shape
    if ndim == 1:
        _kernel_1d[1, 1, 1](out, x, reverse, s[0], s[0])
    elif ndim == 2:
        _kernel_2d[1, 1, 1](out, x, dim, reverse, s[0], s[1], s[0], s[1])
    elif ndim == 3:
        _kernel_3d[1, 1, 1](out, x, dim, reverse, s[0], s[1], s[2], s[0], s[1], s[2])
    else:
        pytest.skip(f"Unsupported tensor dimension: {ndim}")
    return out


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------
_supported_dtypes = ['float32', 'float16', 'bfloat16', 'int32', 'int16', 'int8']

_shapes_2d = [(7, 23), (64, 8), (16, 128)]
_shapes_1d = [(128,), (37,)]
_shapes_3d = [(4, 8, 16), (3, 5, 7)]


# ---------------------------------------------------------------------------
# Tests – 1-D
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", _supported_dtypes)
@pytest.mark.parametrize("shape", _shapes_1d)
@pytest.mark.parametrize("reverse", [False, True])
def test_cummin_1d(dtype, shape, reverse):
    torch.manual_seed(0)
    x = test_common.generate_tensor(shape=shape, dtype=dtype).npu()
    triton_out = triton_cummin(x, dim=0, reverse=reverse)
    torch_dtype = eval('torch.' + dtype)
    ref_input = x.to(torch.float32) if torch_dtype in (torch.float16, torch.bfloat16) else x
    ref = torch_cummin(ref_input, dim=0, reverse=reverse).to(torch_dtype)
    test_common.validate_cmp(dtype, triton_out, ref)


# ---------------------------------------------------------------------------
# Tests – 2-D
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", _supported_dtypes)
@pytest.mark.parametrize("shape", _shapes_2d)
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_cummin_2d(dtype, shape, dim, reverse):
    torch.manual_seed(0)
    x = test_common.generate_tensor(shape=shape, dtype=dtype).npu()
    triton_out = triton_cummin(x, dim=dim, reverse=reverse)
    torch_dtype = eval('torch.' + dtype)
    ref_input = x.to(torch.float32) if torch_dtype in (torch.float16, torch.bfloat16) else x
    ref = torch_cummin(ref_input, dim=dim, reverse=reverse).to(torch_dtype)
    test_common.validate_cmp(dtype, triton_out, ref)


# ---------------------------------------------------------------------------
# Tests – 3-D
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", _supported_dtypes)
@pytest.mark.parametrize("shape", _shapes_3d)
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_cummin_3d(dtype, shape, dim, reverse):
    torch.manual_seed(0)
    x = test_common.generate_tensor(shape=shape, dtype=dtype).npu()
    triton_out = triton_cummin(x, dim=dim, reverse=reverse)
    torch_dtype = eval('torch.' + dtype)
    ref_input = x.to(torch.float32) if torch_dtype in (torch.float16, torch.bfloat16) else x
    ref = torch_cummin(ref_input, dim=dim, reverse=reverse).to(torch_dtype)
    test_common.validate_cmp(dtype, triton_out, ref)


# ---------------------------------------------------------------------------
# Tests – special float values (inf, -inf, nan)
# ---------------------------------------------------------------------------
_float_dtypes = ['float32', 'float16', 'bfloat16']


def _inject_special_values(x):
    """Randomly inject inf, -inf, and nan into ~10 % of the elements."""
    flat = x.flatten()
    n = flat.numel()
    num_special = max(1, n // 10)
    indices = torch.randperm(n)[:num_special]
    specials = torch.tensor(
        [float('inf'), float('-inf'), float('nan')],
        dtype=flat.dtype, device=flat.device,
    )
    flat[indices] = specials[torch.randint(0, 3, (num_special,))]
    return flat.view(x.shape)


@pytest.mark.parametrize("dtype", _float_dtypes)
@pytest.mark.parametrize("shape", [(7, 23), (64, 8)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_cummin_special_values(dtype, shape, dim, reverse):
    torch.manual_seed(42)
    torch_dtype = eval('torch.' + dtype)
    x = torch.randn(shape, dtype=torch.float32)
    x = _inject_special_values(x).to(torch_dtype).npu()
    triton_out = triton_cummin(x, dim=dim, reverse=reverse)
    ref_input = x.to(torch.float32)
    ref = torch_cummin(ref_input, dim=dim, reverse=reverse).to(torch_dtype)
    test_common.validate_cmp(dtype, triton_out, ref)
