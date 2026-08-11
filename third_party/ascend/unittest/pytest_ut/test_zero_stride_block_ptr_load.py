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

import pytest
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def zero_stride_block_ptr_load_kernel(
    input_ptr,
    output_ptr,
    shape_m,
    shape_n,
    offset_m,
    offset_n,
    STRIDE_M: tl.constexpr,
    STRIDE_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    input_block = tl.make_block_ptr(
        base=input_ptr,
        shape=(shape_m, shape_n),
        strides=(STRIDE_M, STRIDE_N),
        offsets=(offset_m, offset_n),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    output_block = tl.make_block_ptr(
        base=output_ptr,
        shape=(BLOCK_M, BLOCK_N),
        strides=(BLOCK_N, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    value = tl.load(input_block, boundary_check=(0, 1), padding_option="zero")
    tl.store(output_block, value)


def _reference(values, shape, strides, offsets, block_shape):
    expected = torch.zeros(block_shape, dtype=values.dtype)
    for row in range(block_shape[0]):
        for col in range(block_shape[1]):
            logical_row = offsets[0] + row
            logical_col = offsets[1] + col
            if 0 <= logical_row < shape[0] and 0 <= logical_col < shape[1]:
                physical_offset = logical_row * strides[0] + logical_col * strides[1]
                expected[row, col] = values[physical_offset]
    return expected


@pytest.mark.parametrize(
    "strides,offsets,values",
    [
        ((0, 0), (4, 4), torch.tensor([3.5], dtype=torch.float32)),
        ((0, 1), (4, 3), torch.arange(5, dtype=torch.float32)),
        ((0, 0), (-2, 0), torch.tensor([7.0], dtype=torch.float32)),
    ],
)
def test_zero_stride_block_ptr_load(strides, offsets, values):
    shape = (6, 5)
    block_shape = (4, 4)
    input_tensor = values.npu()
    output = torch.empty(block_shape, dtype=values.dtype).npu()

    zero_stride_block_ptr_load_kernel[(1, )](
        input_tensor,
        output,
        shape[0],
        shape[1],
        offsets[0],
        offsets[1],
        STRIDE_M=strides[0],
        STRIDE_N=strides[1],
        BLOCK_M=block_shape[0],
        BLOCK_N=block_shape[1],
    )

    expected = _reference(values, shape, strides, offsets, block_shape)
    torch.testing.assert_close(output.cpu(), expected)
