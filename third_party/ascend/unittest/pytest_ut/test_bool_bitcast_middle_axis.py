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
"""End-to-end coverage for the bool pointer bitcast fix.

The kernel builds an [BLOCK_M, 1, BLOCK_N] pointer tile where only the outer
axis contributes an offset, so the size-1 middle axis has stride 0. For bool
tensors the frontend casts ptr<i1> to ptr<i8> before load/store, so the pointer
chain is `broadcast -> addptr -> bitcast -> load`.

With compile_mode="simd" this used to produce a memref descriptor carrying
`strides: [..., 16, 0]` and BiShengIR failed with
"[ConvertLinalgIRToBinary] error: strides must not be zero". simt_only hid the
bug because scalar SIMT accesses do not need non-zero strides.
"""

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def bool_middle_axis_load_kernel(in_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # [BLOCK_M, 1, BLOCK_N]: only the row axis carries an offset, so the
    # size-1 middle axis is a pure broadcast with stride 0.
    row = offs_m[:, None, None] * N
    offs = row + tl.arange(0, BLOCK_N)[None, None, :]

    mask = (offs_m[:, None, None] < M) & (tl.arange(0, BLOCK_N)[None, None, :] < N)

    # Bool load: the frontend casts ptr<i1> to ptr<i8> before tt.load.
    value = tl.load(in_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, value.to(tl.int32), mask=mask)


@pytest.mark.parametrize("compile_mode", ["simd", "simd_simt_template"])
@pytest.mark.parametrize("M,N", [(16, 16), (17, 13), (32, 64)])
def test_bool_middle_axis_load(compile_mode, M, N):
    torch.manual_seed(0)
    block_m = 16
    block_n = triton.next_power_of_2(N)

    host_in = torch.randint(0, 2, (M, 1, N), dtype=torch.int32).bool()
    in_tensor = host_in.npu()
    out = torch.zeros((M, 1, N), dtype=torch.int32, device="npu")

    grid = (triton.cdiv(M, block_m), )
    bool_middle_axis_load_kernel[grid](in_tensor, out, M, N, BLOCK_M=block_m, BLOCK_N=block_n,
                                       compile_mode=compile_mode)
    torch.npu.synchronize()

    torch.testing.assert_close(out.cpu(), host_in.to(torch.int32))


@triton.jit
def bool_middle_axis_store_kernel(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    row = offs_m[:, None, None] * N
    offs = row + tl.arange(0, BLOCK_N)[None, None, :]

    mask = (offs_m[:, None, None] < M) & (tl.arange(0, BLOCK_N)[None, None, :] < N)

    value = tl.load(x_ptr + offs, mask=mask, other=0)
    # Force a bool result so the frontend inserts the ptr<i1> -> ptr<i8> cast.
    result = value > 0
    tl.store(out_ptr + offs, result, mask=mask)


@pytest.mark.parametrize("compile_mode", ["simd", "simd_simt_template"])
@pytest.mark.parametrize("M,N", [(16, 16), (17, 13), (32, 64)])
def test_bool_middle_axis_store(compile_mode, M, N):
    torch.manual_seed(0)
    block_m = 16
    block_n = triton.next_power_of_2(N)

    x = torch.randint(-8, 8, (M, N), dtype=torch.int32, device="npu")
    out = torch.zeros((M, 1, N), dtype=torch.bool, device="npu")

    grid = (triton.cdiv(M, block_m), )
    bool_middle_axis_store_kernel[grid](x, out, M, N, BLOCK_M=block_m, BLOCK_N=block_n, compile_mode=compile_mode)
    torch.npu.synchronize()

    expected = (x > 0).reshape(M, 1, N)
    torch.testing.assert_close(out.cpu(), expected.cpu())
