# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Regression for arithmetic on inactive rows introduced by program batching."""
import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

pytestmark = pytest.mark.backend("torch_npu")


@triton.jit
def _row_div_sum(x, divisors, out, WIDTH: tl.constexpr):
    row = tl.program_id(1)
    cols = tl.arange(0, WIDTH)
    values = tl.load(x + row * WIDTH + cols)
    divisor = tl.load(divisors + row)
    quotient = values // divisor
    tl.store(out + row, tl.sum(quotient, axis=0))


@pytest.mark.parametrize("rows", [63, 64, 65])
def test_integer_row_dynamic_divisor(rows):
    x = torch.full((rows, 16), 12, dtype=torch.int32, device="npu")
    divisors = torch.full((rows, ), 3, dtype=torch.int32, device="npu")
    out = torch.empty((rows, ), dtype=torch.int32, device="npu")
    compiled = _row_div_sum[(1, rows, 1)](x, divisors, out, WIDTH=16, compile_mode="simd", enable_graph_optimize=True)
    # Unknown divisors must not be padded with zero by integer-row batching.
    assert compiled.metadata.program_grid_transforms is None
    assert torch.equal(out.cpu(), torch.full((rows, ), 64, dtype=torch.int32))
