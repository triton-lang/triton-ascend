# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT
"""Regression for StoreConverter preserving zero-stride write domains."""
import pytest
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def copy_zero_stride_kernel(src, dst, BLOCK: tl.constexpr):
    value = tl.load(src + tl.full((1, ), 0, tl.int32))
    tl.store(dst + tl.full((BLOCK, ), 0, tl.int32), value)


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('block', [1, 8, 16])
def test_copy_zero_stride(dtype, block):
    src = torch.tensor([13], dtype=dtype, device='npu')
    out = torch.full((1, ), -1, dtype=dtype, device='npu')
    copy_zero_stride_kernel[(1, )](src, out, BLOCK=block)
    torch.testing.assert_close(out, src, rtol=0, atol=0)


@triton.jit
def masked_alias_kernel(dst, ACTIVE: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    # Exactly one active lane, or none: the result has no write race.
    tl.store(dst + tl.full((BLOCK, ), 0, tl.int32), lane, lane == ACTIVE)


@pytest.mark.parametrize('active', [-1, 0, 3, 7])
def test_masked_alias_preserves_active_lane(active):
    out = torch.full((1, ), -1, dtype=torch.int32, device='npu')
    masked_alias_kernel[(1, )](out, ACTIVE=active, BLOCK=8)
    expected = torch.tensor([active], dtype=torch.int32, device='npu')
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@triton.jit
def two_axis_zero_stride_kernel(src, dst, M: tl.constexpr, N: tl.constexpr):
    value = tl.load(src + tl.full((1, 1), 0, tl.int32))
    tl.store(dst + tl.full((M, N), 0, tl.int32), value)


@pytest.mark.parametrize('shape', [(2, 8), (4, 16)])
def test_two_zero_stride_axes(shape):
    src = torch.tensor([3.5], dtype=torch.float32, device='npu')
    out = torch.full((1, ), -1, dtype=torch.float32, device='npu')
    two_axis_zero_stride_kernel[(1, )](src, out, M=shape[0], N=shape[1])
    torch.testing.assert_close(out, src, rtol=0, atol=0)


@triton.jit
def remainder_alias_kernel(dst, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    tl.store(dst + lane % 4, lane, lane < 4)


def test_remainder_linearization_with_zero_stride_axis():
    out = torch.full((4, ), -1, dtype=torch.int32, device='npu')
    remainder_alias_kernel[(1, )](out, BLOCK=8)
    torch.testing.assert_close(out, torch.arange(4, dtype=torch.int32, device='npu'), rtol=0, atol=0)
