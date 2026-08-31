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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
getBoundarySizes() reconstructs the per-axis in-bounds size by decomposing the
flat block offset with the full-shape strides.  Two defects in that
decomposition are covered:

an axis that is *not* boundary-checked used to leave its
contribution in the flat offset, so a checked trailing axis computed its
boundary from the un-reduced value and clipped it to 0, silently dropping
the loaded/stored block;
"""

import torch
import torch_npu

import triton
import triton.language as tl


@triton.jit
def boundary_size_unchecked_axis_kernel(in_ptr, out_ptr, TOTAL_M, TOTAL_N, OFF_M, OFF_N, BLOCK_M: tl.constexpr,
                                        BLOCK_N: tl.constexpr):
    """The leading axis is *not* boundary-checked while the trailing axis
    is: its flat offset (OFF_M * TOTAL_N + OFF_N) must be reduced before the
    trailing-axis boundary is derived from it."""
    bptr = tl.make_block_ptr(
        base=in_ptr,
        shape=(TOTAL_M, TOTAL_N),
        strides=(TOTAL_N, 1),
        offsets=(OFF_M, OFF_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    val = tl.load(bptr, boundary_check=(1, ), padding_option="zero")

    out = tl.make_block_ptr(
        base=out_ptr,
        shape=(TOTAL_M, TOTAL_N),
        strides=(TOTAL_N, 1),
        offsets=(OFF_M, OFF_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(out, val, boundary_check=(1, ))


def _check_boundary(actual: torch.Tensor, expected: torch.Tensor, tag: str):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0,
        atol=0,
        msg=(f"{tag}: make_block_ptr boundary size diverged from the "
             "reference implementation"),
    )


def test_boundary_size_unchecked_axis():
    TOTAL_M, TOTAL_N = 8, 60
    BLOCK_M, BLOCK_N = 4, 32
    OFF_M, OFF_N = 2, 36

    x = torch.arange(TOTAL_M * TOTAL_N, dtype=torch.float32, device="npu").reshape(TOTAL_M, TOTAL_N)

    out_kernel = torch.zeros((TOTAL_M, TOTAL_N), dtype=torch.float32, device="npu")
    out_ref = torch.zeros_like(out_kernel)

    boundary_size_unchecked_axis_kernel[(1, )](x, out_kernel, TOTAL_M, TOTAL_N, OFF_M, OFF_N, BLOCK_M=BLOCK_M,
                                               BLOCK_N=BLOCK_N)
    torch.npu.synchronize()

    # 24 columns fit in bounds on the checked trailing axis; the 8 columns
    # past the tail stay zero (zero padding on the load, masked store).
    # With the un-reduced flat offset the boundary used to clip to 0 and the
    # whole block was dropped.
    out_ref[OFF_M:OFF_M + BLOCK_M, OFF_N:TOTAL_N] = \
        x[OFF_M:OFF_M + BLOCK_M, OFF_N:TOTAL_N]

    _check_boundary(out_kernel, out_ref, "unchecked_axis")
