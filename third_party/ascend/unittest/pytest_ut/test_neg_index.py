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
import triton
import triton.language as tl
import torch
import torch_npu
import test_common


@triton.jit
def triton_neg_index_load_kernel(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr, NEG_INDEX: tl.constexpr):
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset >= NEG_INDEX
    tmp = tl.load(in_ptr + ((-NEG_INDEX) + offset), mask, other=0.0)
    tl.store(out_ptr + offset, tmp)


def torch_neg_index_load(in_tensor, index):
    out = torch.zeros_like(in_tensor)
    n = in_tensor.numel()
    if 0 <= index < n:
        out[index:] = in_tensor[:n - index]
    return out


def test_neg_index_load():
    input_data = torch.arange(12, device="npu", dtype=torch.float32)
    out = torch.zeros_like(input_data)
    triton_neg_index_load_kernel[(1, )](input_data, out, 12, 6)
    ref = torch_neg_index_load(input_data, 6)
    test_common.validate_cmp("float32", out, ref)


@triton.jit
def triton_mixed_static_index_load_kernel(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr, NEG_A: tl.constexpr,
                                          POS_B: tl.constexpr):
    offset = tl.arange(0, BLOCK_SIZE)
    idx = offset + (-NEG_A) + POS_B
    mask = (idx >= 0) & (idx < BLOCK_SIZE)
    tmp = tl.load(in_ptr + idx, mask, other=0.0)
    tl.store(out_ptr + offset, tmp)


def torch_mixed_static_index_load(in_tensor, neg_a, pos_b):
    n = in_tensor.numel()
    shift = pos_b - neg_a
    out = torch.zeros_like(in_tensor)
    for i in range(n):
        j = i + shift
        if 0 <= j < n:
            out[i] = in_tensor[j]
    return out


@pytest.mark.parametrize('neg_a,pos_b', [(8, 2),  # linear shift -6 < 0
                                         (4, 10),  # linear shift +6 >= 0
                                         (5, 12),  # linear shift +7 >= 0
                                         ])
def test_mixed_static_index_load(neg_a, pos_b):
    n = 12
    input_data = torch.arange(n, device="npu", dtype=torch.float32)
    out = torch.zeros_like(input_data)
    triton_mixed_static_index_load_kernel[(1, )](input_data, out, n, neg_a, pos_b)
    ref = torch_mixed_static_index_load(input_data, neg_a, pos_b)
    test_common.validate_cmp("float32", out, ref)
