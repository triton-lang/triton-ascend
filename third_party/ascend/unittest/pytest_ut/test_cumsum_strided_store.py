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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
import test_common


@triton.jit
def triton_cumsum(
    in_ptr0,
    out_ptr0,
    dim: tl.constexpr,
    reverse: tl.constexpr,
    in_L: tl.constexpr,
    in_M: tl.constexpr,
    out_M: tl.constexpr,
    stride: tl.constexpr,
):
    idx_l = tl.arange(0, in_L)
    idx_m = tl.arange(0, in_M)
    # GM 连续读入
    idx = idx_l[:, None] * in_M + idx_m[None, :]
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumsum(x, axis=dim, reverse=reverse)
    # 尾轴子视图 [..., ::stride] 写出，store stride 非对齐
    odx = idx_l[:, None] * out_M + idx_m[None, :] * stride
    tl.store(out_ptr0 + odx, ret)


@pytest.mark.parametrize('dtype', ['int16'])
@pytest.mark.parametrize('shape', [(128, 4), (9, 7)])
@pytest.mark.parametrize('stride', [2, 3])
def test_cumsum_strided_store(dtype, shape, stride):
    torch_dtype = eval('torch.' + dtype)
    from_shape = list(shape)
    from_shape[-1] = from_shape[-1] * stride

    x = test_common.generate_tensor(shape, dtype).npu()
    output = torch.zeros(from_shape, dtype=torch_dtype).npu()

    # 标杆：先升为 int64 累加避免溢出，再转回原 dtype
    ans = torch.cumsum(x.to(torch.int64), dim=0).to(torch_dtype)
    ref_output = output.clone()
    ref_output[..., ::stride] = ans

    triton_cumsum[(1, )](x, output, 0, False, shape[0], shape[1], from_shape[1], stride)

    test_common.validate_cmp(dtype, output, ref_output)
