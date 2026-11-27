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

# Regression test for the BlockDataParser address-chain fixes:
#  - arith.trunci producers must be accepted by BlockDataParser::parse
#    (parseTruncI) instead of failing with "unsupported BlockData producer".
#  - unmasked single-element loads (tensor<1xT>) feeding an address chain must
#    be parsed as dynamic scalars instead of source-anchored UnstrucMemAcc
#    blocks, so that multiplying two such loads does not hit the
#    mulBlock double-source assertion.
# Extracted from triton_per_fused__to_copy_add_embedding_mul_na_1.py
# (inductor embedding pattern: trunci(load_a) * load_b as row index).

import triton
import triton.language as tl
import torch
import pytest
import test_common


def torch_indexed_load(a, b, data, c, n, numel):
    # Mirror the kernel semantics: a is narrowed to i32 first (trunci),
    # then multiplied by b and shifted by c to form the row index.
    a32 = a.to(torch.int32)
    row = int(((a32 * b).to(torch.int64) + c).item())
    assert 0 <= row < n
    return data[row * n:row * n + numel]


@triton.jit
def triton_trunci_indexed_load(idx_a_ptr, idx_b_ptr, in_ptr, out_ptr, xnumel, XBLOCK: tl.constexpr, N: tl.constexpr):
    x0 = tl.arange(0, XBLOCK)
    mask = x0 < xnumel
    tmp4 = tl.load(idx_a_ptr + tl.arange(0, 1))  # tensor<1xi64>
    tmp7 = tl.load(idx_b_ptr + tl.arange(0, 1))  # tensor<1xi32>
    # i64 -> i32 narrowing in the address chain (fix point a: arith.trunci).
    tmp6 = tmp4.to(tl.int32)
    # Two runtime load values multiplied, then extsi + constant offset.
    tmp9 = tmp6 * tmp7
    tmp10 = tmp9.to(tl.int64)
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp10 + tmp11
    tmp14 = tl.load(in_ptr + (x0 + N * tmp12), mask, other=0.0)
    tl.store(out_ptr + x0, tmp14, mask)


@pytest.mark.parametrize('param_list', [
    # dtype, idx_a, idx_b, rows, n, xblock
    ['float32', 1, 2, 4, 1024, 1024],
])
def test_case(param_list):
    dtype, idx_a, idx_b, rows, n, xblock = param_list
    a = torch.tensor([idx_a], dtype=torch.int64).npu()
    b = torch.tensor([idx_b], dtype=torch.int32).npu()
    data = test_common.generate_tensor((rows * n, ), dtype).npu()
    y_ref = torch_indexed_load(a, b, data, 1, n, xblock)
    y_cal = torch.zeros((xblock, ), dtype=eval('torch.' + dtype)).npu()
    triton_trunci_indexed_load[1, 1, 1](a, b, data, y_cal, xblock, xblock, n)
    test_common.validate_cmp(dtype, y_cal, y_ref)
