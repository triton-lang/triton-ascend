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
import test_common

import torch
import torch_npu


def standard_(x0, dtype):
    res, index = torch.max(x0, 0, keepdim=True)
    return res


@triton.jit
def triton_max_vector(in_ptr0, out_ptr0, N: tl.constexpr, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)

    if in_ptr0.dtype == tl.int8:
        padding = -128
    else:
        padding = -float('inf')

    x = tl.load(in_ptr0 + idx_block, mask=idx_block < N, other=padding)
    ret = tl.max(x, 0)
    tl.store(out_ptr0 + idx_block, ret, mask=idx_block < 1)


types = [
    (torch.float32, 'float32'),
    # (torch.float16,'float16'),  TODO : fix reduceConverter bug
    # (torch.bfloat16,'bfloat16'),  waiting for supporting or testing
    # (torch.int8,'int8'),  TODO : fix compiler bug
    # (torch.int16,'int16'),  waiting for supporting or testing
    # (torch.int32,'int32'),  waiting for supporting or testing
    # (torch.int64,'int64'),  waiting for supporting or testing
]

# if shape axis = 32/256 , then actual shape = axis/element_size()
shapes = [
    (3, 32),
    (-32, 32),
    (37, 64),
    (-256, 256),
    (781, 1024),
]

map_for_64_t = {37: 31}


@triton.jit
def triton_max_nan(in_ptr0, out_ptr0, NUMEL: tl.constexpr):
    idx_block = tl.arange(0, NUMEL)
    x = tl.load(in_ptr0 + idx_block)
    ret = tl.max(x, 0)
    tl.store(out_ptr0, ret)


# maxnum semantics: an axis whose elements are all NaN reduces to NaN, while
# an axis containing at least one non-NaN value reduces to the largest of
# those values (NaN is ignored).
@pytest.mark.parametrize('dtype, sigtype', [(torch.float32, 'float32')])
@pytest.mark.parametrize('nan_ratio', [1.0, 0.5])
def test_reduce_dim0_nan_max(dtype, sigtype, nan_ratio):
    numel = 64
    x0 = test_common.generate_tensor(shape=(numel, ), dtype=sigtype)
    nan_count = int(numel * nan_ratio)
    x0[:nan_count] = float('nan')
    if nan_count < numel:
        x0[nan_count:] = torch.arange(1, numel - nan_count + 1, dtype=dtype)

    print(f"reduce all-NaN ratio: {nan_ratio} {dtype} {sigtype}")

    # Expected value under maxnum semantics: max over the non-NaN elements,
    # or NaN when every element is NaN. torch.nanmax is not available in this
    # torch build, so filter the NaNs out manually.
    valid = x0[~torch.isnan(x0)]
    if valid.numel() == 0:
        ans = torch.full((1, ), float('nan'), dtype=dtype)
    else:
        ans = torch.max(valid, 0, keepdim=True).values
    x0 = x0.npu()

    output = torch.zeros((1, ), dtype=dtype).npu()
    triton_max_nan[1, 1, 1](x0, output, NUMEL=numel)

    test_common.validate_cmp(sigtype, output, ans)


@pytest.mark.skip(reason="randomly failed")
@pytest.mark.parametrize('dtype, sigtype', types)
@pytest.mark.parametrize('N, NUMEL', shapes)
def test_reduce_dim0_common(dtype, sigtype, N, NUMEL):
    N = (-N) // torch.tensor(0, dtype=dtype).element_size() if N < 0 else N

    if sigtype == 'int64':
        N = map_for_64_t[N] if N in map_for_64_t else N

    x0 = test_common.generate_tensor(shape=(N, ), dtype=sigtype)

    ans = standard_(x0, dtype)
    x0 = x0.npu()

    output = torch.zeros((1, ), dtype=dtype).npu()
    triton_max_vector[1, 1, 1](x0, output, N=N, NUMEL=NUMEL, debug=True)

    test_common.validate_cmp(sigtype, output, ans)
