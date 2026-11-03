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

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.language.extra.tla as tla


@triton.jit
def tensor_view_padding_kernel(src, indices, dst, rows, cols, column_block, VIEW: tl.constexpr, PADDING: tl.constexpr):
    if VIEW == "partition":
        view = tla.make_partition_view(src, shape=[rows, cols], strides=[cols, 1], tile=[2, 8], padding_value=PADDING)
        value = tl.load(view, index=(0, column_block))
    elif VIEW == "strided":
        view = tla.make_strided_view(src, shape=[rows, cols], strides=[cols, 1], tile=[2, 8], traversal_strides=[2, 5],
                                     padding_value=PADDING)
        value = tl.load(view, index=(0, column_block))
    else:
        row_indices = tl.load(indices + tl.arange(0, 2))
        view = tla.make_gather_scatter_view(src, shape=[rows, cols], strides=[cols, 1], tile=[2, 8], sparse_dim=[0],
                                            padding_value=PADDING)
        value = tl.load(view, index=(row_indices, column_block))
    output = tla.make_partition_view(dst, shape=[2, 8], strides=[8, 1], tile=[2, 8])
    tl.store(output, value, index=(0, 0))


@pytest.mark.parametrize("view", ["partition", "strided", "gather"])
@pytest.mark.parametrize("padding", ["zero", "nan", "inf", "-inf"])
@pytest.mark.parametrize("column_block", [-1, 1, 3], ids=["left", "partial", "right"])
def test_tensor_view_padding(view, padding, column_block):
    # Exercise left/partial/right boundaries and an invalid sparse row.
    source = torch.arange(26, dtype=torch.float32).reshape(2, 13)
    row_indices = torch.tensor([-1, 1], dtype=torch.int32)
    source_npu, indices_npu = source.to("npu"), row_indices.to("npu")
    actual = torch.empty((2, 8), dtype=torch.float32, device="npu")
    tensor_view_padding_kernel[(1, )](source_npu, indices_npu, actual, 2, 13, column_block, VIEW=view, PADDING=padding)

    pad_value = {"zero": 0.0, "nan": float("nan"), "inf": float("inf"), "-inf": -float("inf")}[padding]
    expected = torch.full((2, 8), pad_value, dtype=torch.float32)
    origin = column_block * (5 if view == "strided" else 8)
    for i in range(2):
        row = int(row_indices[i]) if view == "gather" else i
        for j in range(8):
            column = origin + j
            if 0 <= row < 2 and 0 <= column < 13:
                expected[i, j] = source[row, column]
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)
