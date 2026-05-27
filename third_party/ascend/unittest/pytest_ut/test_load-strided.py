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

import triton
import triton.language as tl
import numpy as np
import torch
import pytest
import test_common

# eg: pytest -v test.py::test_add
#############################

def validate_strided_cmp(a, b, stride, dtype):
    # 超过2维时打平成2维: (d0, d1, ..., dn) -> (-1, dn)，保留尾轴不变
    if a.dim() > 2:
        a = a.reshape(-1, a.shape[-1])
    if b.dim() > 2:
        b = b.reshape(-1, b.shape[-1])

    # a[i, j] 应该等于 b[i, j * stride]
    # j 的取值范围: 0, 1, 2, ..., (last_dim // stride - 1)
    j_max = a.shape[-1] // stride
    a_sliced = a[:, :j_max]
    b_strided = b[:, 0::stride][:, :j_max]

    if dtype in ('float32', 'float16'):
        assert torch.allclose(a_sliced, b_strided, atol=1e-2, rtol=1e-3), \
            f"Strided comparison failed: max diff = {(a_sliced.float() - b_strided.float()).abs().max().item()}"
    else:
        assert torch.equal(a_sliced, b_strided), \
            f"Strided comparison failed: mismatch count = {(a_sliced != b_strided).sum().item()}"
        

@triton.jit
def triton_load_store(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr, STRIDE: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex * STRIDE
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tmp2 = tmp0
        tl.store(out_ptr0 + (xindex), tmp2, xmask)


# require: all data (4d and 5d) can be placed into but without ub overflow
@triton.jit
def triton_load_store_multi_d(
    in_ptr0, out_ptr0, 
    BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr, BLOCK_4: tl.constexpr,
    SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr, SHAPE_4: tl.constexpr,
    STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, STRIDE_3: tl.constexpr, STRIDE_4: tl.constexpr
):
    offsets = tl.program_id(0)

    offsets = offsets + tl.arange(0, BLOCK_0) * STRIDE_0
    masks = tl.arange(0, BLOCK_0) < SHAPE_0
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        masks = masks[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        masks = masks[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        masks = masks[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if BLOCK_4 > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        masks = masks[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)

    tmp_in = tl.load(in_ptr0 + offsets, masks)
    tmp_out = tmp_in
    tl.store(out_ptr0 + offsets, tmp_out, masks)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (2, 4096, 8), 2, 32768, 1024, 16],
                             ['float16', (2, 4096, 8), 2, 32768, 1024, 4],
                             ['int8', (2, 4096, 8), 2, 32768, 1024, 64],
                             ['float32', (8, 8, 4), 2, 128, 64, 6],
                             ['float16', (8, 8, 4), 2, 128, 64, 7],
                             ['int8', (8, 8, 4), 2, 128, 64,3],
                             ['int8', (8, 7, 4), 2, 128, 64,2],

                         ]
                         )
def test_load_store(param_list):
    dtype, shape, ncore, xblock, xblock_sub, stride  = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    y_ref = x0
    y_cal = test_common.generate_tensor(shape, dtype).npu()
    triton_load_store[(ncore, )](x0, y_cal, x0.numel(), xblock, xblock_sub, stride )
    validate_strided_cmp(y_cal, y_ref, stride, dtype)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (8, 4, 16, 16), 2],
                             ['float16', (8, 4, 16, 16),3],
                             ['int8', (8, 4, 16, 16),4],
                             ['float32', (8, 8, 4, 4),2],
                             ['float16', (8, 8, 4, 4),1],
                             ['int8', (8, 8, 4, 4),2],
                             ['float32', (3, 8, 2, 16, 16),3],
                             ['float16', (3, 8, 2, 16, 16),2],
                             ['int8', (9, 8, 8, 16, 16),2],
                             ['float32', (11, 8, 8, 4, 4),4],
                             ['float16', (11, 8, 8, 4, 4),1],
                             ['int8', (11, 8, 8, 4, 4),2],
                         ]
                         )
def test_load_store_multi_d(param_list):
    dtype, shape, stride  = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    y_expect = x0
    y_actual = test_common.generate_tensor(shape, dtype).npu()

    blocks = list(x0.size())
    shapes = list(x0.stride())
    strides = list(x0.stride())
    strides[-1] = stride 
    while len(blocks) < 5:
        blocks.append(1)
        shapes.append(1)
 
    y_cal = triton_load_store_multi_d[(1, )](x0, y_actual, *blocks, *shapes, *strides)
    validate_strided_cmp(y_cal, x0, stride, dtype)

param_list = ['float32', (32, ), 1, 32, 32, 2]
test_load_store(param_list)