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
import torch
import torch_npu
import pytest
import test_common


def torch_divsiop_select_analysis(offs_num, divnum, maxindex, index, query):
    idx = query[index]
    offs_m = torch.arange(offs_num, dtype=torch.int32).npu()
    query_pos = offs_m // divnum - idx
    mask = query_pos < maxindex

    tensor0 = torch.tensor(0, dtype=torch.int32).npu()
    tensor1 = torch.tensor(1, dtype=torch.int32).npu()
    result = torch.where(mask, tensor1, tensor0).npu()
    return result


@triton.jit
def divsiop_select_analysis_kernel1(index, out_ptr, query, offs_num: tl.constexpr, divnum: tl.constexpr,
                                    maxindex: tl.constexpr):
    idx = tl.load(query + index)
    offs_m = tl.arange(0, offs_num)

    query_pos = offs_m // divnum - idx

    mask = query_pos < maxindex
    query_mask = tl.where(mask, 1, 0).to(tl.int1)
    tl.store(out_ptr + tl.arange(0, offs_num), query_mask)


@triton.jit
def divsiop_select_analysis_kernel2(index, out_ptr, query, offs_num: tl.constexpr, divnum: tl.constexpr,
                                    maxindex: tl.constexpr):
    idx = tl.load(query + index)
    offs_m = tl.arange(0, offs_num)

    query_pos = -idx + offs_m // divnum
    mask = query_pos < maxindex

    query_mask = tl.where(mask, 1, 0).to(tl.int1)
    tl.store(out_ptr + tl.arange(0, offs_num), query_mask)


@pytest.mark.parametrize('param_list', [[16, 4, 2, index] for index in range(0, 4)])
def test_divsiop_select_analysis1(param_list):
    offs_num, divnum, maxindex, index = param_list
    query = torch.tensor(range(0, divnum)).npu()
    y_ref = torch_divsiop_select_analysis(offs_num, divnum, maxindex, index, query).npu()
    y_cal = torch.full((offs_num, ), 2, dtype=torch.int32).npu()
    compiled = divsiop_select_analysis_kernel1[(1, )](index, y_cal, query, offs_num, divnum, maxindex)
    test_common.validate_cmp('int32', y_cal, y_ref)
    _assert_continuous_mask(compiled)


@pytest.mark.parametrize('param_list', [[16, 4, 2, index] for index in range(0, 4)])
def test_divsiop_select_analysis2(param_list):
    offs_num, divnum, maxindex, index = param_list
    query = torch.tensor(range(0, divnum)).npu()
    y_ref = torch_divsiop_select_analysis(offs_num, divnum, maxindex, index, query).npu()
    y_cal = torch.full((offs_num, ), 2, dtype=torch.int32).npu()
    compiled = divsiop_select_analysis_kernel2[(1, )](index, y_cal, query, offs_num, divnum, maxindex)
    test_common.validate_cmp('int32', y_cal, y_ref)
    _assert_continuous_mask(compiled)


# `arith.divsi` truncates towards zero, so the reference must use truncating
# division: torch `//` floors and disagrees with the kernel for negative values.
def _trunc_div(value, divnum):
    return torch.div(value, divnum, rounding_mode="trunc")


def _trunc_mask_reference(offs_num, divnum, shift, pred, bound):
    offs = torch.arange(offs_num, dtype=torch.int32)
    q = _trunc_div(offs - shift, divnum)
    if pred == 0:
        mask = q < bound
    elif pred == 1:
        mask = q <= bound
    elif pred == 2:
        mask = q >= bound
    else:
        mask = q == bound
    return torch.where(mask, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32))


def _has_tensor_select(compiled):
    # The discrete fallback selects the complete data vector with a tensor
    # arith.select. The continuous path may still contain a scalar arith.select
    # that picks the quotient block edge when the bound is a runtime value.
    ttadapter = compiled.asm["ttadapter"]
    return any(": tensor<" in line for line in ttadapter.splitlines() if "arith.select" in line)


def _assert_continuous_mask(compiled):
    assert not _has_tensor_select(compiled), compiled.asm["ttadapter"]


@triton.jit
def divsiop_negative_constant_kernel(out_ptr, offs_num: tl.constexpr, divnum: tl.constexpr, shift: tl.constexpr,
                                     bound: tl.constexpr, pred: tl.constexpr):
    offs_m = tl.arange(0, offs_num)
    q = (offs_m - shift) // divnum
    if pred == 0:
        mask = q < bound
    elif pred == 1:
        mask = q <= bound
    elif pred == 2:
        mask = q >= bound
    else:
        mask = q == bound
    tl.store(out_ptr + offs_m, tl.where(mask, 1, 0))


@pytest.mark.parametrize('pred,bound', [(0, -1), (0, 0), (0, 2), (1, 1), (2, 1), (3, 0), (3, -1)])
def test_divsiop_negative_constant(pred, bound):
    offs_num, divnum, shift = 16, 4, 5
    y_ref = _trunc_mask_reference(offs_num, divnum, shift, pred, bound).npu()
    y_cal = torch.full((offs_num, ), 2, dtype=torch.int32).npu()
    compiled = divsiop_negative_constant_kernel[(1, )](y_cal, offs_num, divnum, shift, bound, pred)
    test_common.validate_cmp('int32', y_cal, y_ref)
    _assert_continuous_mask(compiled)


@triton.jit
def divsiop_negative_masked_load_kernel(x_ptr, out_ptr, offs_num: tl.constexpr, divnum: tl.constexpr,
                                        shift: tl.constexpr, bound: tl.constexpr):
    offs_m = tl.arange(0, offs_num)
    q = (offs_m - shift) // divnum
    x = tl.load(x_ptr + offs_m, mask=q < bound, other=0)
    tl.store(out_ptr + offs_m, x)


def test_divsiop_negative_masked_load():
    offs_num, divnum, shift, bound = 16, 4, 5, 2
    x = torch.arange(1, offs_num + 1, dtype=torch.int32).npu()
    mask = _trunc_mask_reference(offs_num, divnum, shift, 0, bound).bool()
    y_ref = torch.where(mask, x.cpu(), torch.tensor(0, dtype=torch.int32)).npu()
    y_cal = torch.full((offs_num, ), -1, dtype=torch.int32).npu()
    compiled = divsiop_negative_masked_load_kernel[(1, )](x, y_cal, offs_num, divnum, shift, bound)
    test_common.validate_cmp('int32', y_cal, y_ref)
    _assert_continuous_mask(compiled)


@triton.jit
def divsiop_negative_runtime_bound_kernel(nr_ptr, out_ptr, offs_num: tl.constexpr, divnum: tl.constexpr,
                                          shift: tl.constexpr):
    nr = tl.load(nr_ptr)
    offs_m = tl.arange(0, offs_num)
    q = (offs_m - shift) // divnum
    tl.store(out_ptr + offs_m, tl.where(q < nr, 1, 0))


@pytest.mark.parametrize('nr', [2, -1])
def test_divsiop_negative_runtime_bound(nr):
    offs_num, divnum, shift = 16, 4, 5
    nr_ptr = torch.tensor([nr], dtype=torch.int32).npu()
    y_ref = _trunc_mask_reference(offs_num, divnum, shift, 0, nr).npu()
    y_cal = torch.full((offs_num, ), 2, dtype=torch.int32).npu()
    compiled = divsiop_negative_runtime_bound_kernel[(1, )](nr_ptr, y_cal, offs_num, divnum, shift)
    test_common.validate_cmp('int32', y_cal, y_ref)
    # The runtime bound adds a scalar quotient-edge select but the mask stays an interval.
    _assert_continuous_mask(compiled)


@triton.jit
def divsiop_post_division_scalar_kernel(out_ptr, offs_num: tl.constexpr, divnum: tl.constexpr, shift: tl.constexpr,
                                        bound: tl.constexpr, post: tl.constexpr):
    offs_m = tl.arange(0, offs_num)
    q = (offs_m - shift) // divnum - post
    tl.store(out_ptr + offs_m, tl.where(q < bound, 1, 0))


@pytest.mark.parametrize('post', [0, 3])
def test_divsiop_post_division_scalar(post):
    offs_num, divnum, shift, bound = 16, 4, 5, 2
    offs = torch.arange(offs_num, dtype=torch.int32)
    q = _trunc_div(offs - shift, divnum) - post
    y_ref = torch.where(q < bound, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32)).npu()
    y_cal = torch.full((offs_num, ), 2, dtype=torch.int32).npu()
    compiled = divsiop_post_division_scalar_kernel[(1, )](y_cal, offs_num, divnum, shift, bound, post)
    test_common.validate_cmp('int32', y_cal, y_ref)
    # post == 0 must degenerate to the plain quotient mask; post == 3 shifts the bound.
    _assert_continuous_mask(compiled)


@triton.jit
def divsiop_post_division_runtime_scalar_kernel(post_ptr, out_ptr, offs_num: tl.constexpr, divnum: tl.constexpr,
                                                shift: tl.constexpr, bound: tl.constexpr):
    post = tl.load(post_ptr)
    offs_m = tl.arange(0, offs_num)
    q = (offs_m - shift) // divnum - post
    tl.store(out_ptr + offs_m, tl.where(q < bound, 1, 0))


@pytest.mark.parametrize('post', [0, 3])
def test_divsiop_post_division_runtime_scalar(post):
    offs_num, divnum, shift, bound = 16, 4, 5, 2
    post_ptr = torch.tensor([post], dtype=torch.int32).npu()
    offs = torch.arange(offs_num, dtype=torch.int32)
    q = _trunc_div(offs - shift, divnum) - post
    y_ref = torch.where(q < bound, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32)).npu()
    y_cal = torch.full((offs_num, ), 2, dtype=torch.int32).npu()
    compiled = divsiop_post_division_runtime_scalar_kernel[(1, )](post_ptr, y_cal, offs_num, divnum, shift, bound)
    test_common.validate_cmp('int32', y_cal, y_ref)
    _assert_continuous_mask(compiled)


@triton.jit
def divsiop_loop_quotient_kernel(out_ptr, offs_num: tl.constexpr, divnum: tl.constexpr, shift: tl.constexpr,
                                 bound: tl.constexpr, steps: tl.constexpr):
    offs_m = tl.arange(0, offs_num)
    q = (offs_m - shift) // divnum
    for i in range(0, steps):
        tl.store(out_ptr + offs_m + i * offs_num, tl.where(q < bound, 1, 0))
        q = q + 1


def test_divsiop_loop_carried_quotient():
    offs_num, divnum, shift, bound, steps = 16, 4, 5, 2, 4
    offs = torch.arange(offs_num, dtype=torch.int32)
    q = _trunc_div(offs - shift, divnum)
    rows = [
        torch.where(q + i < bound, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32))
        for i in range(steps)
    ]
    y_ref = torch.stack(rows).tolist()
    out = torch.full((steps * offs_num, ), 2, dtype=torch.int32).npu()
    compiled = divsiop_loop_quotient_kernel[(1, )](out, offs_num, divnum, shift, bound, steps)
    expected = torch.tensor(y_ref, dtype=torch.int32).npu()
    test_common.validate_cmp('int32', out.view(steps, offs_num), expected)
    _assert_continuous_mask(compiled)


@triton.jit
def divsiop_nested_division_kernel(out_ptr, offs_num: tl.constexpr, divnum: tl.constexpr, shift: tl.constexpr,
                                   bound: tl.constexpr):
    offs_m = tl.arange(0, offs_num)
    q = ((offs_m - shift) // divnum) // 2
    tl.store(out_ptr + offs_m, tl.where(q < bound, 1, 0))


def test_divsiop_nested_division():
    offs_num, divnum, shift, bound = 16, 4, 5, 1
    offs = torch.arange(offs_num, dtype=torch.int32)
    q = _trunc_div(_trunc_div(offs - shift, divnum), 2)
    y_ref = torch.where(q < bound, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32)).npu()
    y_cal = torch.full((offs_num, ), 2, dtype=torch.int32).npu()
    compiled = divsiop_nested_division_kernel[(1, )](y_cal, offs_num, divnum, shift, bound)
    test_common.validate_cmp('int32', y_cal, y_ref)
    _assert_continuous_mask(compiled)


# #2865 shape: `c + q // num_rows < total_blocks` with runtime `c`. With
# blocks_per_iter == 1 the quotient is uniformly 0, so the runtime scalar is
# never added to a non-zero quotient and the mask must stay an interval.
@triton.jit
def divsiop_issue2865_kernel(c_ptr, total_ptr, out_ptr, num_rows: tl.constexpr):
    c = tl.load(c_ptr)
    total = tl.load(total_ptr)
    q = tl.arange(0, num_rows)
    b = q // num_rows
    blk = c + b
    mask = blk < total
    tl.store(out_ptr + tl.arange(0, num_rows), tl.where(mask, 1, 0))


@pytest.mark.parametrize('c,total', [(0, 2), (2, 2), (2147483647, 2)])
def test_divsiop_issue2865_prefix_continuous(c, total):
    num_rows = 4
    offs = torch.arange(num_rows, dtype=torch.int32)
    b = _trunc_div(offs, num_rows)
    blk = torch.tensor(c, dtype=torch.int32) + b
    y_ref = torch.where(blk < total, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32)).npu()
    y_cal = torch.full((num_rows, ), 2, dtype=torch.int32).npu()
    c_ptr = torch.tensor([c], dtype=torch.int32).npu()
    total_ptr = torch.tensor([total], dtype=torch.int32).npu()
    compiled = divsiop_issue2865_kernel[(1, )](c_ptr, total_ptr, y_cal, num_rows)
    test_common.validate_cmp('int32', y_cal, y_ref)
    _assert_continuous_mask(compiled)


# #3043 shape: the numerator starts at `program_id * block`, so the division
# input bounds are runtime values. The mask must still become an interval;
# otherwise the load keeps the unbounded discrete form.
@triton.jit
def divsiop_issue3043_kernel(x_ptr, n_rows_ptr, out_ptr, start: tl.constexpr, width: tl.constexpr, block: tl.constexpr):
    pid = tl.program_id(0)
    n_rows = tl.load(n_rows_ptr)
    offs = pid * block + tl.arange(0, block)
    rows = start + offs // width
    vals = tl.load(x_ptr + rows, mask=rows < n_rows, other=0)
    tl.store(out_ptr + offs, vals)


@pytest.mark.parametrize('n_rows', [16, 0])
def test_divsiop_issue3043_prefix_continuous(n_rows):
    block, width, start = 16, 8, 1
    x = torch.arange(1, 64, dtype=torch.int32).npu()
    offs = torch.arange(block, dtype=torch.int32)
    rows = start + _trunc_div(offs, width)
    mask = rows < n_rows
    y_ref = torch.where(mask, x.cpu()[rows.long()], torch.tensor(0, dtype=torch.int32)).npu()
    y_cal = torch.full((block, ), 2, dtype=torch.int32).npu()
    n_rows_ptr = torch.tensor([n_rows], dtype=torch.int32).npu()
    compiled = divsiop_issue3043_kernel[(1, )](x, n_rows_ptr, y_cal, start, width, block)
    test_common.validate_cmp('int32', y_cal, y_ref)
    _assert_continuous_mask(compiled)
