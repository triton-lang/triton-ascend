# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Unsigned narrowing must preserve upper-bit checks and signed shifts."""
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _narrow_bits(X, T, O, N: tl.constexpr, SHIFT: tl.constexpr):
    i = tl.arange(0, N)
    x = tl.load(X + i)
    t = tl.load(T)
    u = x & 0xffffffff
    key = tl.where(x < 0, (~u) & 0xffffffff, u | 0x80000000)
    tl.store(O + i, key >> SHIFT)
    tl.store(O + N + i, (key >> SHIFT) == t)
    # This shift must NOT be changed to unsigned, nor narrowed before shifting.
    tl.store(O + 2 * N + i, x >> SHIFT)
    tl.store(O + 3 * N + i, x)


@pytest.mark.parametrize("shift", [0, 8, 24, 31])
@pytest.mark.parametrize("threshold", [-1, 0, 255, 0x1000000ff])
def test_unsigned_tensor_narrowing(shift, threshold):
    values = torch.tensor([
        -(1 << 63), -0x100000001, -1, 0, 1, 255, 0x7fffffff, 0x80000000, 0xffffffff, 0x100000000, 0x1000000ff,
        (1 << 63) - 1, -255, 0x123456789abcdef, -0x123456789abcdef, 42
    ], dtype=torch.int64, device="npu")
    target = torch.tensor([threshold], dtype=torch.int64, device=values.device)
    output = torch.empty((4, 16), dtype=torch.int64, device=values.device)
    _narrow_bits[(1, )](values, target, output, 16, shift, enable_graph_optimize=True)
    x = values.cpu()
    u = x & 0xffffffff
    key = torch.where(x < 0, (~u) & 0xffffffff, u | 0x80000000)
    expected = torch.stack((key >> shift, ((key >> shift) == threshold).long(), x >> shift, x))
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
