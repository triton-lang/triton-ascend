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
Packed Mask
===========

Apply a packed boolean mask to a score tile on Ascend. Eight mask columns
share one byte, and each tile decodes its own mask before selecting a penalty.

Install the helpers from the repository root with
``python -m pip install ./python/triton_kernels``.
"""

# %%
# Compute one score tile per program.

import torch
import torch_npu
import triton
import triton.language as tl

from triton.backends.ascend.utils import is_register_based_arch
from triton_kernels.packed_mask import generate_bitmask, unpack_mask


@triton.jit
def packed_mask_kernel(score_ptr, packed_ptr, output_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr,
                       BLOCK_N: tl.constexpr, SHIFT_DTYPE: tl.constexpr):
    row_start = tl.program_id(0) * BLOCK_M
    col_start = tl.program_id(1) * BLOCK_N
    bits = unpack_mask(packed_ptr, row_start, col_start, N // 8, BLOCK_M, BLOCK_N, SHIFT_DTYPE)
    score_block = tl.make_block_ptr(score_ptr, (M, N), (N, 1), (row_start, col_start), (BLOCK_M, BLOCK_N), (1, 0))
    output_block = tl.make_block_ptr(output_ptr, (M, N), (N, 1), (row_start, col_start), (BLOCK_M, BLOCK_N), (1, 0))
    score = tl.load(score_block)
    allow = tl.full((BLOCK_M, BLOCK_N), 0.0, score.dtype)
    deny = tl.full((BLOCK_M, BLOCK_N), -8.0, score.dtype)
    penalty = tl.where(bits != 0, allow, deny)
    tl.store(output_block, score + penalty)


# %%
# Launch a 2D grid over the padded score tensor.


def apply_packed_mask(score, packed, block_m, block_n):
    m, n = score.shape
    output = torch.empty_like(score)
    grid = (m // block_m, n // block_n)
    arch = triton.runtime.driver.active.get_current_target().arch
    shift_dtype = tl.uint8 if is_register_based_arch(arch) else tl.int16
    packed_mask_kernel[grid](score, packed, output, m, n, BLOCK_M=block_m, BLOCK_N=block_n, SHIFT_DTYPE=shift_dtype)
    return output


# %%
# Pack a mask with a partial row and column tile and compare against PyTorch.


def test_packed_mask():
    block_m, block_n = 32, 128
    rows, cols = 29, 113

    # Replace this example with your own boolean mask.
    row = torch.arange(rows)[:, None]
    col = torch.arange(cols)[None, :]
    mask = (row + col) % 3 != 0

    # Pad to a full tile so every packed address the kernel reads is in bounds.
    packed = generate_bitmask(rows, cols, block_m, block_n, mask=mask.npu())
    score = torch.ones((block_m, block_n), dtype=torch.float32, device='npu')
    output_triton = apply_packed_mask(score, packed, block_m, block_n)

    dense_mask = torch.zeros((block_m, block_n), dtype=torch.bool)
    dense_mask[:rows, :cols] = mask
    output_torch = score.cpu() + torch.where(dense_mask, 0.0, -8.0)
    torch.testing.assert_close(output_triton.cpu(), output_torch, rtol=0, atol=0)


if __name__ == "__main__":
    test_packed_mask()
    print("======Packed Mask Test Passed!======")
