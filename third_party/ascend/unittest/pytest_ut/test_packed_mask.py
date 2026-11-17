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
"""
Packed Mask - Pytest Version
"""

import pytest
import torch
import torch_npu
import triton
import triton.language as tl

from triton_kernels.packed_mask import generate_bitmask, unpack_mask


@triton.jit
def packed_mask_kernel(score_ptr, packed_ptr, output_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr,
                       BLOCK_N: tl.constexpr):
    row_start = tl.program_id(0) * BLOCK_M
    col_start = tl.program_id(1) * BLOCK_N
    bits = unpack_mask(packed_ptr, row_start, col_start, N // 8, BLOCK_M, BLOCK_N)
    score_block = tl.make_block_ptr(score_ptr, (M, N), (N, 1), (row_start, col_start), (BLOCK_M, BLOCK_N), (1, 0))
    output_block = tl.make_block_ptr(output_ptr, (M, N), (N, 1), (row_start, col_start), (BLOCK_M, BLOCK_N), (1, 0))
    score = tl.load(score_block)
    allow = tl.full((BLOCK_M, BLOCK_N), 0.0, score.dtype)
    deny = tl.full((BLOCK_M, BLOCK_N), -8.0, score.dtype)
    penalty = tl.where(bits != 0, allow, deny)
    tl.store(output_block, score + penalty)


@pytest.mark.parametrize("block_n", [64, 128, 256, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("case", ["aligned", "tail", "short"])
def test_packed_mask(block_n, dtype, case):
    block_m = 32
    row_capacity, column_capacity = 2 * block_m, 2 * block_n
    if case == "aligned":
        rows, cols = row_capacity, column_capacity
    elif case == "tail":
        rows, cols = row_capacity - 3, column_capacity - 5
    else:
        rows, cols = 7, 11

    generator = torch.Generator().manual_seed(2026)
    mask = torch.randint(0, 2, (rows, cols), generator=generator).bool()
    # Set the first byte to 0x8D to check the bit order.
    mask[0, :8] = torch.tensor([True, False, True, True, False, False, False, True])
    packed = generate_bitmask(rows, cols, row_capacity, column_capacity, mask=mask.npu())

    dense = torch.zeros((row_capacity, column_capacity), dtype=torch.bool)
    dense[:rows, :cols] = mask
    packed_cpu = packed.cpu()
    assert packed_cpu.dtype == torch.uint8
    assert packed_cpu.shape == (row_capacity, column_capacity // 8)
    assert packed_cpu[0, 0].item() == 0x8D
    # Check each bit plane, including padded rows and columns.
    for bit in range(8):
        actual = (packed_cpu & (1 << bit)) != 0
        torch.testing.assert_close(actual, dense[:, bit::8])

    score_cpu = (torch.randint(0, 256, (row_capacity, column_capacity), generator=generator) / 256).to(dtype)
    score = score_cpu.npu()
    output_triton = torch.empty_like(score)
    grid = (row_capacity // block_m, column_capacity // block_n)
    packed_mask_kernel[grid](score, packed, output_triton, row_capacity, column_capacity, BLOCK_M=block_m,
                             BLOCK_N=block_n)

    output_torch = score_cpu + torch.where(dense, 0.0, -8.0).to(dtype)
    torch.testing.assert_close(output_triton.cpu(), output_torch, rtol=0, atol=0)
