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

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
import triton.language.extra.tla as tla


@tla.jit
def load_partition_tile(view, row, column):
    return tla.load(view, index=(row, column))


@tla.jit
def partition_view_load_store_kernel(src, dst, rows, cols, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    src_tiles = tla.make_partition_view(
        src,
        shape=[rows, cols],
        strides=[cols, 1],
        tile=[BLOCK_M, BLOCK_N],
    )
    dst_tiles = tla.make_partition_view(
        dst,
        shape=[rows, cols],
        strides=[cols, 1],
        tile=[BLOCK_M, BLOCK_N],
    )

    value = load_partition_tile(src_tiles, row_block, col_block)
    tla.store(dst_tiles, value, index=(row_block, col_block))


@tla.jit
def partition_view_padding_kernel(src, dst, src_elements, dst_elements, BLOCK_SIZE: tl.constexpr):
    block = tl.program_id(0)
    src_tiles = tla.make_partition_view(
        src,
        shape=[src_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
        padding_value="-inf",
    )
    dst_tiles = tla.make_partition_view(
        dst,
        shape=[dst_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
    )

    value = tla.load(src_tiles, index=(block, ))
    tla.store(dst_tiles, value, index=(block, ))


def test_tensor_view_partition_load_store():
    rows = 128
    cols = 128
    block_m = 64
    block_n = 64
    src = torch.rand((rows, cols), device="npu")
    actual = torch.zeros_like(src)

    grid = (triton.cdiv(rows, block_m), triton.cdiv(cols, block_n))
    partition_view_load_store_kernel[grid](src, actual, rows, cols, BLOCK_M=block_m, BLOCK_N=block_n)

    torch.testing.assert_close(actual, src)


def test_tensor_view_partition_padding():
    src_elements = 10
    block_size = 8
    dst_elements = 16
    src = torch.arange(src_elements, dtype=torch.float32, device="npu")
    actual = torch.empty(dst_elements, dtype=torch.float32, device="npu")

    partition_view_padding_kernel[(2, )](src, actual, src_elements, dst_elements, BLOCK_SIZE=block_size)

    expected = torch.full_like(actual, -torch.inf)
    expected[:src_elements] = src
    torch.testing.assert_close(actual, expected)


@tla.jit
def partition_view_physical_stride_kernel(src, dst, size, src_stride, BLOCK_SIZE: tl.constexpr):
    block = tl.program_id(0)
    source = tla.make_partition_view(src, shape=[size], strides=[src_stride], tile=[BLOCK_SIZE])
    output = tla.make_partition_view(dst, shape=[size], strides=[3], tile=[BLOCK_SIZE])
    value = tla.load(source, index=(block, ))
    tla.store(output, value, index=(block, ))


def test_tensor_view_partition_physical_stride():
    size = 17
    src = torch.arange(size * 3, dtype=torch.float32, device="npu")
    actual = torch.full_like(src, -17)
    partition_view_physical_stride_kernel[(triton.cdiv(size, 8), )](src, actual, size, 2, BLOCK_SIZE=8)
    expected = torch.full_like(src, -17)
    expected[::3] = src[:size * 2:2]
    torch.testing.assert_close(actual, expected)
