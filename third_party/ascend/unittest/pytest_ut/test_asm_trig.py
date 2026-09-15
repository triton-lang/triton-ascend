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
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import test_common


@triton.jit
def triton_asm_sin(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.inline_asm_elementwise(
        asm="sin.approx.f32 $0, $1;",
        constraints="=r,r",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def triton_asm_cos(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.inline_asm_elementwise(
        asm="cos.approx.f32 $0, $1;",
        constraints="=r,r",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def triton_asm_tanh(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.inline_asm_elementwise(
        asm="tanh.approx.f32 $0, $1;",
        constraints="=r,r",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def triton_asm_atan(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.inline_asm_elementwise(
        asm="atan.approx.f32 $0, $1;",
        constraints="=r,r",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def triton_asm_cos_2d(
    x_ptr,
    output_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, BLOCK_N)
    offsets = row_offsets[:, None] * N + col_offsets[None, :]
    mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.inline_asm_elementwise(
        asm="cos.approx.f32 $0, $1;",
        constraints="=r,r",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(output_ptr + offsets, output, mask=mask)


TRIG_CASES = [
    (triton_asm_sin, torch.sin),
    (triton_asm_cos, torch.cos),
    (triton_asm_tanh, torch.tanh),
    (triton_asm_atan, torch.atan),
]


@pytest.mark.parametrize("kernel,ref_fn", TRIG_CASES)
@pytest.mark.parametrize("length,block_size", [
    (4096, 1024),
    (4096, 4096),
])
def test_trig_inline_asm_elementwise(kernel, ref_fn, length, block_size):
    dtype = 'float32'
    ncore = length // block_size
    x = test_common.generate_tensor((length, ), dtype).npu()
    res_ref = ref_fn(x.cpu()).npu()
    res_cal = torch.zeros((length, ), dtype=torch.float32).npu()
    kernel[(ncore, )](x, res_cal, length, BLOCK_SIZE=block_size)
    test_common.validate_cmp(dtype, res_cal, res_ref)


def test_cos_inline_asm_2d_large_tile():
    M, N = 64, 64
    block_m, block_n = 64, 64
    ncore = M // block_m
    x = test_common.generate_tensor((M, N), 'float32').npu()
    res_ref = torch.cos(x.cpu()).npu()
    res_cal = torch.zeros((M, N), dtype=torch.float32).npu()
    triton_asm_cos_2d[(ncore, )](x, res_cal, M, N, BLOCK_M=block_m, BLOCK_N=block_n)
    test_common.validate_cmp('float32', res_cal, res_ref)
