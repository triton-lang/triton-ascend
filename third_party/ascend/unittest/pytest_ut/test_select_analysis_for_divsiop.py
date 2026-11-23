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
    elif pred == 3:
        mask = q == bound
    else:
        mask = q > bound
    return torch.where(mask, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32))


def _has_tensor_select(compiled):
    # The discrete fallback selects the complete data vector with a tensor
    # arith.select. The continuous path may still contain a scalar arith.select
    # that picks the quotient block edge when the bound is a runtime value.
    ttadapter = compiled.asm["ttadapter"]
    return any(": tensor<" in line for line in ttadapter.splitlines() if "arith.select" in line)


def _assert_continuous_mask(compiled):
    assert not _has_tensor_select(compiled), compiled.asm["ttadapter"]


def _assert_discrete_mask(compiled):
    assert _has_tensor_select(compiled), compiled.asm["ttadapter"]


# The bound is either a compile-time constant or loaded at runtime; the rest
# of the expression is identical, so both variants share one kernel.
@triton.jit
def divsiop_bound_kernel(bound_ptr, out_ptr, offs_num: tl.constexpr, divnum: tl.constexpr, shift: tl.constexpr,
                         bound: tl.constexpr, pred: tl.constexpr, runtime_bound: tl.constexpr):
    if runtime_bound:
        bound_val = tl.load(bound_ptr)
    else:
        bound_val = bound
    offs_m = tl.arange(0, offs_num)
    q = (offs_m - shift) // divnum
    if pred == 0:
        mask = q < bound_val
    elif pred == 1:
        mask = q <= bound_val
    elif pred == 2:
        mask = q >= bound_val
    elif pred == 3:
        mask = q == bound_val
    else:
        mask = q > bound_val
    tl.store(out_ptr + offs_m, tl.where(mask, 1, 0))


@pytest.mark.parametrize('pred,bound', [
    (0, -1), (0, 0), (0, 2),
    (1, 1),
    (2, 1),
    (3, 0), (3, -1),
    (4, -2147483648), (4, -1), (4, 0), (4, 2), (4, 2147483647),
])
def test_divsiop_bound_constant(pred, bound):
    offs_num, divnum, shift = 16, 4, 5
    bound_ptr = torch.tensor([bound], dtype=torch.int32).npu()
    y_ref = _trunc_mask_reference(offs_num, divnum, shift, pred, bound).npu()
    y_cal = torch.full((offs_num, ), 2, dtype=torch.int32).npu()
    compiled = divsiop_bound_kernel[(1, )](bound_ptr, y_cal, offs_num, divnum, shift, bound, pred, False)
    test_common.validate_cmp('int32', y_cal, y_ref)
    _assert_continuous_mask(compiled)


@pytest.mark.parametrize('bound', [2, -1])
def test_divsiop_bound_runtime(bound):
    offs_num, divnum, shift = 16, 4, 5
    bound_ptr = torch.tensor([bound], dtype=torch.int32).npu()
    y_ref = _trunc_mask_reference(offs_num, divnum, shift, 0, bound).npu()
    y_cal = torch.full((offs_num, ), 2, dtype=torch.int32).npu()
    compiled = divsiop_bound_kernel[(1, )](bound_ptr, y_cal, offs_num, divnum, shift, 0, 0, True)
    test_common.validate_cmp('int32', y_cal, y_ref)
    # The runtime bound adds a scalar quotient-edge select but the mask stays an interval.
    _assert_continuous_mask(compiled)


@triton.jit
def divsiop_masked_load_kernel(x_ptr, out_ptr, offs_num: tl.constexpr, divnum: tl.constexpr, shift: tl.constexpr,
                               bound: tl.constexpr):
    offs_m = tl.arange(0, offs_num)
    q = (offs_m - shift) // divnum
    x = tl.load(x_ptr + offs_m, mask=q < bound, other=0)
    tl.store(out_ptr + offs_m, x)


def test_divsiop_masked_load():
    offs_num, divnum, shift, bound = 16, 4, 5, 2
    x = torch.arange(1, offs_num + 1, dtype=torch.int32).npu()
    mask = _trunc_mask_reference(offs_num, divnum, shift, 0, bound).bool()
    y_ref = torch.where(mask, x.cpu(), torch.tensor(0, dtype=torch.int32)).npu()
    y_cal = torch.full((offs_num, ), -1, dtype=torch.int32).npu()
    compiled = divsiop_masked_load_kernel[(1, )](x, y_cal, offs_num, divnum, shift, bound)
    test_common.validate_cmp('int32', y_cal, y_ref)
    _assert_continuous_mask(compiled)


# The post-division scalar is either a compile-time constant or loaded at
# runtime; both variants share one kernel.
@triton.jit
def divsiop_post_division_scalar_kernel(post_ptr, out_ptr, offs_num: tl.constexpr, divnum: tl.constexpr,
                                        shift: tl.constexpr, bound: tl.constexpr, post: tl.constexpr,
                                        runtime_post: tl.constexpr):
    if runtime_post:
        post_val = tl.load(post_ptr)
    else:
        post_val = post
    offs_m = tl.arange(0, offs_num)
    q = (offs_m - shift) // divnum - post_val
    tl.store(out_ptr + offs_m, tl.where(q < bound, 1, 0))


@pytest.mark.parametrize('post,runtime_post', [(0, False), (3, False), (0, True), (3, True)])
def test_divsiop_post_division_scalar(post, runtime_post):
    offs_num, divnum, shift, bound = 16, 4, 5, 2
    post_ptr = torch.tensor([post], dtype=torch.int32).npu()
    offs = torch.arange(offs_num, dtype=torch.int32)
    q = _trunc_div(offs - shift, divnum) - post
    y_ref = torch.where(q < bound, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32)).npu()
    y_cal = torch.full((offs_num, ), 2, dtype=torch.int32).npu()
    compiled = divsiop_post_division_scalar_kernel[(1, )](post_ptr, y_cal, offs_num, divnum, shift, bound, post,
                                                          runtime_post)
    test_common.validate_cmp('int32', y_cal, y_ref)
    # post == 0 must degenerate to the plain quotient mask; post == 3 shifts the bound.
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


# Runtime scalar added to the quotient, compared with a runtime bound. Over
# `arange(0, num_rows)`, `// num_rows` is uniformly 0, so the scalar is never
# added to a non-zero quotient. `<`, `>=`, and `>` stay continuous.
@triton.jit
def divsiop_runtime_scalar_quotient_kernel(c_ptr, total_ptr, out_ptr, num_rows: tl.constexpr, pred: tl.constexpr):
    c = tl.load(c_ptr)
    total = tl.load(total_ptr)
    q = tl.arange(0, num_rows)
    b = q // num_rows
    blk = c + b
    if pred == 0:
        mask = blk < total
    elif pred == 1:
        mask = blk >= total
    else:
        mask = blk > total
    tl.store(out_ptr + tl.arange(0, num_rows), tl.where(mask, 1, 0))


@pytest.mark.parametrize('c,total,pred', [
    (0, 2, 0),
    (2, 2, 0),
    (2147483647, 2, 0),
    (0, 2, 1),
    (2, 2, 1),
    (2147483647, 2, 1),
    (0, 2, 2),
    (0, -1, 2),
    (0, 2147483647, 2),
])
def test_divsiop_runtime_scalar_quotient(c, total, pred):
    num_rows = 4
    offs = torch.arange(num_rows, dtype=torch.int32)
    b = _trunc_div(offs, num_rows)
    blk = torch.tensor(c, dtype=torch.int32) + b
    if pred == 0:
        mask = blk < total
    elif pred == 1:
        mask = blk >= total
    else:
        mask = blk > total
    y_ref = torch.where(mask, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32)).npu()
    y_cal = torch.full((num_rows, ), 2, dtype=torch.int32).npu()
    c_ptr = torch.tensor([c], dtype=torch.int32).npu()
    total_ptr = torch.tensor([total], dtype=torch.int32).npu()
    compiled = divsiop_runtime_scalar_quotient_kernel[(1, )](c_ptr, total_ptr, y_cal, num_rows, pred)
    test_common.validate_cmp('int32', y_cal, y_ref)
    _assert_continuous_mask(compiled)


# The numerator starts at `program_id * block`, so the division input bounds
# are runtime values. The mask must still become an interval; otherwise the
# load keeps the unbounded discrete form.
@triton.jit
def divsiop_runtime_extent_masked_load_kernel(x_ptr, n_rows_ptr, out_ptr, start: tl.constexpr, width: tl.constexpr,
                                              block: tl.constexpr):
    pid = tl.program_id(0)
    n_rows = tl.load(n_rows_ptr)
    offs = pid * block + tl.arange(0, block)
    rows = start + offs // width
    vals = tl.load(x_ptr + rows, mask=rows < n_rows, other=0)
    tl.store(out_ptr + offs, vals)


@pytest.mark.parametrize('n_rows', [16, 0])
def test_divsiop_runtime_extent_masked_load(n_rows):
    block, width, start = 16, 8, 1
    x = torch.arange(1, 64, dtype=torch.int32).npu()
    offs = torch.arange(block, dtype=torch.int32)
    rows = start + _trunc_div(offs, width)
    mask = rows < n_rows
    y_ref = torch.where(mask, x.cpu()[rows.long()], torch.tensor(0, dtype=torch.int32)).npu()
    y_cal = torch.full((block, ), 2, dtype=torch.int32).npu()
    n_rows_ptr = torch.tensor([n_rows], dtype=torch.int32).npu()
    compiled = divsiop_runtime_extent_masked_load_kernel[(1, )](x, n_rows_ptr, y_cal, start, width, block)
    test_common.validate_cmp('int32', y_cal, y_ref)
    _assert_continuous_mask(compiled)


# Re-dividing a quotient that has a non-zero post-division scalar is unsupported:
# trunc((q + C) / e) cannot be folded into the cumulative divisor, so the mask
# must fall back to the discrete path.
@triton.jit
def divsiop_redivide_shifted_quotient_kernel(out_ptr, offs_num: tl.constexpr, divnum: tl.constexpr, shift: tl.constexpr,
                                             post: tl.constexpr, divnum2: tl.constexpr, bound: tl.constexpr):
    offs_m = tl.arange(0, offs_num)
    q = ((offs_m - shift) // divnum - post) // divnum2
    tl.store(out_ptr + offs_m, tl.where(q < bound, 1, 0))


def test_divsiop_redivide_shifted_quotient_falls_back():
    offs_num, divnum, shift, post, divnum2, bound = 16, 4, 5, 1, 2, 1
    offs = torch.arange(offs_num, dtype=torch.int32)
    q = _trunc_div(_trunc_div(offs - shift, divnum) - post, divnum2)
    y_ref = torch.where(q < bound, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32)).npu()
    y_cal = torch.full((offs_num, ), 2, dtype=torch.int32).npu()
    compiled = divsiop_redivide_shifted_quotient_kernel[(1, )](y_cal, offs_num, divnum, shift, post, divnum2, bound)
    test_common.validate_cmp('int32', y_cal, y_ref)
    _assert_discrete_mask(compiled)


# Negative and runtime divisors are unsupported by continuous mask analysis.
# Both must preserve the original division semantics and use the discrete path.
@triton.jit
def divsiop_unsupported_divisor_kernel(div_ptr, out_ptr, offs_num: tl.constexpr, divnum: tl.constexpr,
                                       shift: tl.constexpr, bound: tl.constexpr, runtime_divisor: tl.constexpr):
    if runtime_divisor:
        divisor = tl.load(div_ptr)
    else:
        divisor = divnum
    offs_m = tl.arange(0, offs_num)
    q = (offs_m - shift) // divisor
    tl.store(out_ptr + offs_m, tl.where(q < bound, 1, 0))


@pytest.mark.parametrize('divnum,runtime_divisor', [(-4, False), (4, True)])
def test_divsiop_unsupported_divisor_falls_back(divnum, runtime_divisor):
    offs_num, shift, bound = 16, 5, 2
    div_ptr = torch.tensor([divnum], dtype=torch.int32).npu()
    offs = torch.arange(offs_num, dtype=torch.int32)
    q = _trunc_div(offs - shift, divnum)
    y_ref = torch.where(q < bound, torch.tensor(1, dtype=torch.int32), torch.tensor(0, dtype=torch.int32)).npu()
    y_cal = torch.full((offs_num, ), 2, dtype=torch.int32).npu()
    compiled = divsiop_unsupported_divisor_kernel[(1, )](div_ptr, y_cal, offs_num, divnum, shift, bound,
                                                         runtime_divisor)
    test_common.validate_cmp('int32', y_cal, y_ref)
    _assert_discrete_mask(compiled)
