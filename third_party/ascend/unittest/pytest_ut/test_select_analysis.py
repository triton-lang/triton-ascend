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

import triton
import triton.language as tl
import torch
import torch_npu
import pytest
import test_common


@triton.jit
def kernel_cal_select_mask_bool(
    Output_ptr,
    Indices_ptr,
    numel: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    indice = tl.load(Indices_ptr)

    true_tensor = tl.arange(0, BLOCK) < numel
    false_tensor = tl.arange(0, BLOCK) >= numel
    mask = offs < indice
    res = tl.where(mask, true_tensor, false_tensor)
    tl.store(Output_ptr + offs, res)


@triton.jit
def kernel_cal_select_mask(
    QK_ptr,
    Other_ptr,
    Output_ptr,
    Indices_ptr,
    stride_qk: tl.constexpr,
    numel: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.arange(0, BLOCK) * stride_qk
    cols = tl.arange(0, BLOCK)
    offs = rows[:, None] + cols[None, :]
    row_indices = tl.load(Indices_ptr)
    col_indices = tl.load(Indices_ptr + 1)

    qk_ub = tl.load(QK_ptr + offs)
    other = tl.load(Other_ptr + offs)
    mask_rows = rows < row_indices * stride_qk
    mask_cols = cols < col_indices

    res = tl.where(mask_rows[:, None] & mask_cols[None, :], qk_ub, other)
    tl.store(Output_ptr + offs, res)


def torch_cal_select_mask_bool(
    Indice: torch.Tensor,
    numel,
    BLOCK,
):
    offs = torch.arange(0, BLOCK)
    true_tensor = torch.arange(0, BLOCK) < numel
    false_tensor = torch.arange(0, BLOCK) >= numel
    mask = offs < Indice

    res = torch.where(mask, true_tensor, false_tensor)
    return res


def torch_cal_select_mask(
    QK: torch.Tensor,
    Other: torch.Tensor,
    Indices: torch.Tensor,
):
    row_limit_idx = Indices[0].item()
    col_limit_idx = Indices[1].item()
    Output = Other.clone()
    Output[:row_limit_idx, :col_limit_idx] = QK[:row_limit_idx, :col_limit_idx]
    return Output


@pytest.mark.parametrize('param_list', [['bool', 64, 63]])
def test_select_analysis_bool(param_list):
    dtype, SEQ_LEN, indice = param_list
    assert dtype == 'bool'
    qk_cal = torch.empty(SEQ_LEN).npu()
    indices = torch.tensor([indice]).npu()
    qk_ref = torch_cal_select_mask_bool(indice, SEQ_LEN, SEQ_LEN)
    kernel_cal_select_mask_bool[(1, )](qk_cal, indices, SEQ_LEN, SEQ_LEN)
    test_common.validate_cmp(dtype, qk_cal, qk_ref)


@pytest.mark.parametrize('param_list', [
    ['float16', 64, 63, 62],
    ['float32', 64, 63, 62],
])
def test_select_analysis(param_list):
    dtype, SEQ_LEN, indice_x, indice_y = param_list
    assert dtype != 'bool'
    qk = torch.rand([SEQ_LEN, SEQ_LEN], dtype=eval('torch.' + dtype), device='npu')
    qk_cal = torch.empty_like(qk).npu()
    other = torch.zeros_like(qk).npu()
    indices_tensor = torch.tensor([indice_x, indice_y]).npu()
    qk_ref = torch_cal_select_mask(qk, other, indices_tensor)
    kernel_cal_select_mask[(1, )](qk, other, qk_cal, indices_tensor, qk.stride(0), SEQ_LEN * SEQ_LEN, SEQ_LEN)
    test_common.validate_cmp(dtype, qk_cal, qk_ref)
