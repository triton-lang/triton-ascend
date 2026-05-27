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

import math
import pytest
import torch
import triton
import triton.language as tl
import test_common


@pytest.mark.parametrize("dtype", ['float32', 'float16', 'bfloat16', 'int32', 'int64', 'int16', 'int8'])
@pytest.mark.parametrize("M_BLOCK,N_BLOCK", [(2, 16), (8, 16)])
def test_tensor_descriptor_load_store(dtype, M_BLOCK, N_BLOCK):

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
        in_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[M_BLOCK, N_BLOCK],
        )
        out_desc = tl.make_tensor_descriptor(
            out_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[M_BLOCK, N_BLOCK],
        )
        moffset = tl.program_id(0) * M_BLOCK
        noffset = tl.program_id(1) * N_BLOCK
        block = in_desc.load([moffset, noffset])
        out_desc.store([moffset, noffset], block)

    M, N = M_BLOCK * 2, N_BLOCK * 2
    inp = test_common.generate_tensor((M, N), dtype).npu()
    out = inp.new_empty((M, N))

    grid_m = M // M_BLOCK
    grid_n = N // N_BLOCK

    kernel[(grid_m, grid_n)](out, inp, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(inp, out)


@pytest.mark.parametrize("dtype", ['float32', 'float16', 'bfloat16', 'int32', 'int64', 'int16', 'int8'])
def test_tensor_descriptor_load_store3d(dtype):

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, K, stride_m, stride_n, stride_k, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr,
               K_BLOCK: tl.constexpr):
        in_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, N, K],
            strides=[stride_m, stride_n, stride_k],
            block_shape=[M_BLOCK, N_BLOCK, K_BLOCK],
        )
        out_desc = tl.make_tensor_descriptor(
            out_ptr,
            shape=[M, N, K],
            strides=[stride_m, stride_n, stride_k],
            block_shape=[M_BLOCK, N_BLOCK, K_BLOCK],
        )
        moffset = tl.program_id(0) * M_BLOCK
        noffset = tl.program_id(1) * N_BLOCK
        koffset = tl.program_id(2) * K_BLOCK
        block = in_desc.load([moffset, noffset, koffset])
        out_desc.store([moffset, noffset, koffset], block)

    M, N, K = 8, 16, 32
    inp = test_common.generate_tensor((M, N, K), dtype).npu()
    out = inp.new_empty((M, N, K))

    M_BLOCK = 2
    N_BLOCK = 4

    # 自动调整 K_BLOCK，保证最后一维 block 至少 16 字节
    dtype = getattr(inp, "dtype", None)
    itemsize = torch.tensor([], dtype=inp.dtype).element_size()
    min_k_block = max(16 // itemsize, 1)
    K_BLOCK = max(8, min_k_block)

    grid_m = M // M_BLOCK
    grid_n = N // N_BLOCK
    grid_k = K // K_BLOCK

    kernel[(grid_m, grid_n, grid_k)](out, inp, *inp.shape, *out.stride(), M_BLOCK, N_BLOCK, K_BLOCK)
    torch.testing.assert_close(inp.reshape(M * N * K), out.reshape(M * N * K))


# Exercise the functional load/store builtins once to ensure they map through.
@pytest.mark.parametrize("dtype", ["float32", "uint8"])
def test_tensor_descriptor_functional_interface(dtype):
    """Copies an entire tensor blockwise using the descriptor builtins."""

    @triton.jit
    def kernel(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
        in_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[M_BLOCK, N_BLOCK],
        )
        out_desc = tl.make_tensor_descriptor(
            out_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[M_BLOCK, N_BLOCK],
        )
        moffset = tl.program_id(0) * M_BLOCK
        noffset = tl.program_id(1) * N_BLOCK
        block = tl.load_tensor_descriptor(in_desc, [moffset, noffset])
        tl.store_tensor_descriptor(out_desc, [moffset, noffset], block)

    M, N = 32, 128
    inp = test_common.generate_tensor((M, N), dtype).npu()

    M_BLOCK = 8
    N_BLOCK = 32
    out = inp.new_empty((M, N))

    grid_m = M // M_BLOCK
    grid_n = N // N_BLOCK

    kernel[(grid_m, grid_n)](out, inp, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(inp, out)


@pytest.mark.parametrize("dtype_str", ["int32"])
@pytest.mark.parametrize("shape", [(128, 2, 4), (64, 2, 4), (32, 2, 4), (2, 4, 32), (2, 4, 2)])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("device", ["npu"])
def test_reduce_max(dtype_str, shape, axis, device):

    @triton.jit
    def kernel(
        In,
        Out,
        in_shape1: tl.constexpr,
        in_shape2: tl.constexpr,
        in_shape3: tl.constexpr,
        ou_shape1: tl.constexpr,
        ou_shape2: tl.constexpr,
        axis: tl.constexpr,
    ):
        in_desc = tl.make_tensor_descriptor(
            base=In,
            shape=[in_shape1 * in_shape2 * in_shape3],
            strides=[1],
            block_shape=[in_shape1 * in_shape2 * in_shape3],
        )
        out_desc = tl.make_tensor_descriptor(
            base=Out,
            shape=[ou_shape1 * ou_shape2],
            strides=[1],
            block_shape=[ou_shape1 * ou_shape2],
        )
        val = in_desc.load([0]).reshape(in_shape1, in_shape2, in_shape3)
        output = tl.max(val, axis=axis)
        out_desc.store([0], output.reshape(out_desc.block_shape))

    inp = torch.arange(math.prod(shape), dtype=getattr(torch, dtype_str), device=device).reshape(shape)
    expected, indices = torch.max(inp.to(torch.int64), dim=axis)
    expected = expected.to(torch.int32)
    actual = torch.zeros(expected.shape, dtype=getattr(torch, dtype_str), device=device)
    kernel[(1, )](inp, actual, *shape, *expected.shape, axis=axis)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("dtype", ['float32', 'float16', 'bfloat16'])
@pytest.mark.parametrize("padding", ["zero", "nan"])
def test_tensor_descriptor_padding(dtype, padding):

    @triton.jit
    def device_tma_load(in_ptr, out_ptr, IM, IN, YM, YN, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr,
                        padding: tl.constexpr):
        x_desc = tl.make_tensor_descriptor(in_ptr, shape=[IM, IN], strides=[IN, 1], block_shape=[M_BLOCK, N_BLOCK],
                                           padding_option=padding)
        out_desc = tl.make_tensor_descriptor(out_ptr, shape=[YM, YN], strides=[YN, 1], block_shape=[M_BLOCK, N_BLOCK])

        moffset = tl.program_id(0) * M_BLOCK
        noffset = tl.program_id(1) * N_BLOCK

        value = x_desc.load([moffset, noffset])

        out_desc.store([moffset, noffset], value)

    def alloc_fn(size: int, alignment: float, stream: float):
        return torch.ones(size, device="npu", dtype=torch.float32)

    triton.set_allocator(alloc_fn)

    IM, IN = 48, 48
    OM, ON = 64, 64
    M_BLOCK = 32
    N_BLOCK = 32
    torch_dtype = getattr(torch, dtype)
    input_tensor = torch.arange(IM * IN, device="npu", dtype=torch_dtype).reshape(IM, IN)
    out_device_tma = torch.zeros((OM, ON), device="npu", dtype=torch_dtype)
    grid = (triton.cdiv(OM, M_BLOCK), triton.cdiv(ON, N_BLOCK))
    device_tma_load[grid](input_tensor, out_device_tma, IM, IN, OM, ON, M_BLOCK, N_BLOCK, padding)
    expected = torch.zeros((OM, ON), device="npu", dtype=torch_dtype)
    expected[0:IN, 0:IM] = input_tensor
    if padding == "nan":
        expected[:, IN:ON] = float('nan')
        expected[IM:OM, :] = float('nan')

    torch.testing.assert_close(expected, out_device_tma, equal_nan=True)


@pytest.mark.parametrize("X, Y", [(128, 128), (64, 256)])
@pytest.mark.parametrize("BLOCK_X, BLOCK_Y", [(32, 32), (64, 128), (16, 128), (512, 16)])
@pytest.mark.parametrize("dtype", ['float32', 'float16', 'bfloat16', 'int32'])
@pytest.mark.parametrize("y", [0, 32, 48])
def test_tensor_descriptor_scatter(X, Y, BLOCK_X, BLOCK_Y, dtype, y):

    def torch_scatter_rows(input, idx, y, block_y, X, Y):
        out = torch.zeros((X, Y), dtype=input.dtype, device=input.device)
        for i, j in enumerate(idx):
            out[j][y:y + block_y] = input[i]
        return out

    @triton.jit
    def tensor_descriptor_scatter_rows_kernel(out_ptr, in_ptr, idx_ptr, y, X: tl.constexpr, Y: tl.constexpr,
                                              BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr):
        idx = tl.load(idx_ptr + tl.arange(0, BLOCK_X))
        data = tl.load(in_ptr + tl.arange(0, BLOCK_X)[:, None] * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :])
        desc = tl.make_tensor_descriptor(out_ptr, [X, Y], [Y, 1], [1, BLOCK_Y])
        desc.scatter(data, idx, y)

    device = 'npu'
    if BLOCK_X > X or y + BLOCK_Y > Y:
        pytest.skip()

    torch.manual_seed(42)
    torch_dtype = getattr(torch, dtype)
    input_tensor = torch.arange(BLOCK_X * BLOCK_Y, dtype=torch_dtype, device=device).reshape(BLOCK_X, BLOCK_Y)
    output = torch.zeros((X, Y), dtype=torch_dtype, device=device)

    idx = torch.randperm(BLOCK_X, dtype=torch.int32, device=device)

    def alloc_fn(size: int, align: int, stream):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)

    tensor_descriptor_scatter_rows_kernel[(1, )](output, input_tensor, idx, y, X, Y, BLOCK_X, BLOCK_Y)

    ref = torch_scatter_rows(input_tensor, idx, y, BLOCK_Y, X, Y)
    torch.testing.assert_close(ref, output, atol=0, rtol=0)


@pytest.mark.parametrize("X, Y", [(128, 128), (64, 256)])
@pytest.mark.parametrize("BLOCK_X, BLOCK_Y", [(32, 32), (64, 128), (16, 128), (512, 16)])
@pytest.mark.parametrize("dtype", ['float32', 'float16', 'bfloat16', 'int32', 'int16'])
@pytest.mark.parametrize("y", [0, 32, 48])
def test_tensor_descriptor_gather(X, Y, BLOCK_X, BLOCK_Y, dtype, y):

    @triton.jit
    def tensor_descriptor_gather_rows_kernel(out_ptr, in_ptr, idx_ptr, y, X: tl.constexpr, Y: tl.constexpr,
                                             BLOCK_X: tl.constexpr, BLOCK_Y: tl.constexpr):
        idx = tl.load(idx_ptr + tl.arange(0, BLOCK_X))
        desc = tl.make_tensor_descriptor(in_ptr, [X, Y], [Y, 1], [1, BLOCK_Y])
        out = desc.gather(idx, y)
        tl.store(out_ptr + tl.arange(0, BLOCK_X)[:, None] * BLOCK_Y + tl.arange(0, BLOCK_Y)[None, :], out)

    def torch_gather_rows(input, idx, y, block_y):
        return input[idx.long(), y:y + block_y]

    device = 'npu'
    if BLOCK_X > X or y + BLOCK_Y > Y:
        pytest.skip()

    torch.manual_seed(42)
    torch_dtype = getattr(torch, dtype)
    input_tensor = test_common.generate_tensor((X, Y), dtype).npu()
    output = torch.empty((BLOCK_X, BLOCK_Y), dtype=torch_dtype, device=device)

    idx = torch.randint(BLOCK_X, (BLOCK_X, ), dtype=torch.int32, device=device)

    def alloc_fn(size: int, align: int, steam):
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)

    tensor_descriptor_gather_rows_kernel[(1, )](output, input_tensor, idx, y, X, Y, BLOCK_X, BLOCK_Y)

    ref = torch_gather_rows(input_tensor, idx, y, BLOCK_Y)
    torch.testing.assert_close(ref, output, atol=0, rtol=0)
