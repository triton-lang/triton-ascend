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
def gather_view_load_kernel(src, indices, dst, n_elements, BLOCK_SIZE: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    data_indices = tl.load(indices + offsets)
    src_gather = tla.make_gather_scatter_view(
        src,
        shape=[n_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
        sparse_dim=[0],
    )
    dst_tiles = tla.make_partition_view(
        dst,
        shape=[n_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
    )

    value = tla.load(src_gather, index=(data_indices, ))
    tla.store(dst_tiles, value, index=(block, ))


@tla.jit
def scatter_view_store_kernel(src, indices, dst, n_elements, BLOCK_SIZE: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    data_indices = tl.load(indices + offsets)
    src_tiles = tla.make_partition_view(
        src,
        shape=[n_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
    )
    dst_scatter = tla.make_gather_scatter_view(
        dst,
        shape=[n_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
        sparse_dim=[0],
    )

    value = tla.load(src_tiles, index=(block, ))
    tla.store(dst_scatter, value, index=(data_indices, ))


@tla.jit
def gather_view_padding_kernel(src, indices, dst, src_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    data_indices = tl.load(indices + offsets)
    src_gather = tla.make_gather_scatter_view(
        src,
        shape=[src_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
        sparse_dim=[0],
        padding_value="nan",
    )
    dst_tiles = tla.make_partition_view(
        dst,
        shape=[BLOCK_SIZE],
        strides=[1],
        tile=[BLOCK_SIZE],
    )

    value = tla.load(src_gather, index=(data_indices, ))
    tla.store(dst_tiles, value, index=(0, ))


@tla.jit
def scatter_view_bounds_kernel(src, indices, dst, dst_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    data_indices = tl.load(indices + offsets)
    src_tiles = tla.make_partition_view(
        src,
        shape=[BLOCK_SIZE],
        strides=[1],
        tile=[BLOCK_SIZE],
    )
    dst_scatter = tla.make_gather_scatter_view(
        dst,
        shape=[dst_elements],
        strides=[1],
        tile=[BLOCK_SIZE],
        sparse_dim=[0],
    )

    value = tla.load(src_tiles, index=(0, ))
    tla.store(dst_scatter, value, index=(data_indices, ))


def test_tensor_view_gather_load():
    n_elements = 1024
    block_size = 256
    src = torch.rand(n_elements, device="npu")
    indices = torch.randint(0, n_elements, (n_elements, ), dtype=torch.int32, device="npu")
    actual = torch.empty_like(src)

    gather_view_load_kernel[(triton.cdiv(n_elements, block_size), )](src, indices, actual, n_elements,
                                                                     BLOCK_SIZE=block_size)

    torch.testing.assert_close(actual, src[indices.long()])


def test_tensor_view_scatter_store():
    n_elements = 1024
    block_size = 256
    src = torch.rand(n_elements, device="npu")
    indices = torch.randperm(n_elements, device="npu").to(torch.int32)
    actual = torch.zeros_like(src)
    expected = torch.zeros_like(src)
    expected[indices.long()] = src

    scatter_view_store_kernel[(triton.cdiv(n_elements, block_size), )](src, indices, actual, n_elements,
                                                                       BLOCK_SIZE=block_size)

    torch.testing.assert_close(actual, expected)


def test_tensor_view_gather_padding():
    src = torch.arange(8, dtype=torch.float32, device="npu")
    indices = torch.tensor([0, 7, -1, 8], dtype=torch.int32, device="npu")
    actual = torch.empty(4, dtype=torch.float32, device="npu")

    gather_view_padding_kernel[(1, )](src, indices, actual, src.numel(), BLOCK_SIZE=4)

    expected = torch.tensor([0, 7, torch.nan, torch.nan], dtype=torch.float32, device="npu")
    torch.testing.assert_close(actual, expected, equal_nan=True)


def test_tensor_view_scatter_bounds():
    src = torch.tensor([11, 23, 45, 67], dtype=torch.float32, device="npu")
    indices = torch.tensor([-1, 0, 7, 8], dtype=torch.int32, device="npu")
    actual = torch.zeros(8, dtype=torch.float32, device="npu")

    scatter_view_bounds_kernel[(1, )](src, indices, actual, actual.numel(), BLOCK_SIZE=4)

    expected = torch.zeros_like(actual)
    expected[0] = src[1]
    expected[7] = src[2]
    torch.testing.assert_close(actual, expected)


@tla.jit
def cartesian_view_kernel(src, rows, cols, tile_output, scatter_output):
    offsets = tl.arange(0, 4)
    row_indices = tl.load(rows + offsets)
    col_indices = tl.load(cols + offsets)
    source = tla.make_gather_scatter_view(src, shape=[5, 7], strides=[7, 1], tile=[4, 4], sparse_dim=[0, 1],
                                          padding_value="inf")
    output = tla.make_partition_view(tile_output, shape=[4, 4], strides=[4, 1], tile=[4, 4])
    scatter = tla.make_gather_scatter_view(scatter_output, shape=[5, 7], strides=[7, 1], tile=[4, 4], sparse_dim=[0, 1])
    value = tla.load(source, index=(row_indices, col_indices))
    tla.store(output, value, index=(0, 0))
    tla.store(scatter, value, index=(row_indices, col_indices))


def test_tensor_view_cartesian_gather_scatter():
    src = torch.arange(35, dtype=torch.float32).reshape(5, 7)
    rows = torch.tensor([0, 4, -1, 5], dtype=torch.int32)
    cols = torch.tensor([6, 2, -1, 7], dtype=torch.int32)
    tile = torch.empty((4, 4), dtype=torch.float32, device="npu")
    scatter = torch.full((5, 7), -17, dtype=torch.float32, device="npu")
    cartesian_view_kernel[(1, )](src.to("npu"), rows.to("npu"), cols.to("npu"), tile, scatter)
    expected_tile = torch.full((4, 4), torch.inf)
    expected_scatter = torch.full((5, 7), -17, dtype=torch.float32)
    for i, row in enumerate(rows.tolist()):
        for j, col in enumerate(cols.tolist()):
            if 0 <= row < 5 and 0 <= col < 7:
                expected_tile[i, j] = src[row, col]
                expected_scatter[row, col] = src[row, col]
    torch.testing.assert_close(tile.cpu(), expected_tile)
    torch.testing.assert_close(scatter.cpu(), expected_scatter)
