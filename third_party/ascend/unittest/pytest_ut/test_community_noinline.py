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
"""Collection of upstream Triton `noinline` test cases for the Ascend backend.

The cases below are ported from triton-3.6.0
`python/test/unit/language/{test_core.py,test_tensor_descriptor.py}`. Ascend has
no device-side calling convention, so every private noinline callee is inlined
before lowering; the tests therefore also cover scalar returns, multi-value
returns, calls inside dynamic control flow and calls returning / receiving
tensor descriptors. The `test_line_info.py` noinline cases are not included
because they require a disassembler, which the NPU toolchain does not provide.
"""

import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

import test_common


# ---------------------------------------------------------------------------
# noinline basics: simple call, call graph, shared callee, dynamic control
# flow and multi-value return (test_core.py::test_noinline)
# ---------------------------------------------------------------------------


@triton.jit(noinline=True)
def noinline_simple_fn(x, y, Z):
    z = x + y
    tl.store(Z, z)


@triton.jit(noinline=True)
def noinline_call_graph_fn1(x):
    return x + 1


@triton.jit(noinline=True)
def noinline_call_graph_fn2(y):
    return y + 2


@triton.jit(noinline=True)
def noinline_call_graph_fn(x, y, Z):
    t0 = noinline_call_graph_fn1(x)
    t1 = noinline_call_graph_fn2(y)
    z = t0 + t1
    tl.store(Z, z)


@triton.jit(noinline=True)
def noinline_shared_fn(x, y, Z):
    offs = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
    z = tl.load(Z + offs)
    z = tl.dot(z, z) + x + y
    tl.store(Z + offs, z)


@triton.jit(noinline=True)
def noinline_dynamic_fn(x, y, Z):
    if x >= 1:
        x = noinline_call_graph_fn1(x)
    else:
        x = noinline_call_graph_fn2(x)
    if y >= 2:
        y = noinline_call_graph_fn2(y)
    else:
        y = noinline_call_graph_fn1(y)
    z = x + y
    tl.store(Z, z)


@triton.jit(noinline=True)
def noinline_call_multi_values_fn(x, y):
    return x + 1, y + 2


@triton.jit(noinline=True)
def noinline_multi_values_fn(x, y, Z):
    x, y = noinline_call_multi_values_fn(x, y)
    z = x + y
    tl.store(Z, z)


@triton.jit
def kernel_noinline_simple(X, Y, Z):
    x = tl.load(X)
    y = tl.load(Y)
    noinline_simple_fn(x, y, Z)


@triton.jit
def kernel_noinline_call_graph(X, Y, Z):
    x = tl.load(X)
    y = tl.load(Y)
    noinline_call_graph_fn(x, y, Z)


@triton.jit
def kernel_noinline_shared(X, Y, Z):
    x = tl.load(X)
    y = tl.load(Y)
    noinline_shared_fn(x, y, Z)


@triton.jit
def kernel_noinline_dynamic(X, Y, Z):
    x = tl.load(X)
    y = tl.load(Y)
    noinline_dynamic_fn(x, y, Z)


@triton.jit
def kernel_noinline_multi_values(X, Y, Z):
    x = tl.load(X)
    y = tl.load(Y)
    noinline_multi_values_fn(x, y, Z)


@pytest.mark.parametrize("mode", ["simple", "call_graph", "shared", "dynamic", "multi_values"])
def test_noinline(mode):
    kernel = {
        "simple": kernel_noinline_simple,
        "call_graph": kernel_noinline_call_graph,
        "shared": kernel_noinline_shared,
        "dynamic": kernel_noinline_dynamic,
        "multi_values": kernel_noinline_multi_values,
    }[mode]

    x = torch.tensor([1.0], device="npu", dtype=torch.float32)
    y = torch.tensor([2.0], device="npu", dtype=torch.float32)
    if mode == "shared":
        z = torch.ones((16, 16), device="npu", dtype=torch.float32)
    else:
        z = torch.tensor([0.0], device="npu", dtype=torch.float32)
    kernel[(1, )](x, y, z, num_warps=1)

    if mode == "simple":
        expected = x + y
    elif mode == "shared":
        expected = torch.full((16, 16), 16, device="npu", dtype=torch.float32) + x + y
    else:
        expected = x + 1 + y + 2
    test_common.validate_cmp("float32", z, expected)


# ---------------------------------------------------------------------------
# noinline call inside a dynamic `if` (test_core.py::test_if_call[jit_noinline])
# ---------------------------------------------------------------------------


@triton.jit(noinline=True)
def add_fn_noinline(x):
    return x + 1


@triton.jit
def kernel_if_call_noinline(Out):
    pid = tl.program_id(0)
    o = tl.load(Out)
    if pid == 0:
        a = o + 1
        add_fn_noinline(a)
        o = a
    tl.store(Out, o)


def test_if_call_noinline():
    out = torch.zeros((1, ), dtype=torch.int32, device="npu")
    kernel_if_call_noinline[(1, )](out)
    expected = torch.tensor([1], dtype=torch.int32)
    test_common.validate_cmp("int32", out.cpu(), expected)


# ---------------------------------------------------------------------------
# noinline call with a tensor argument is rejected
# (test_core.py::test_call[noinline], which expects an error upstream)
# ---------------------------------------------------------------------------


@triton.jit(noinline=True)
def val_multiplier_noinline(val, i):
    return val * i


@triton.jit
def vecmul_kernel(ptr, n_elements, rep):
    pid = tl.program_id(axis=0)
    offsets = pid * 128 + tl.arange(0, 128)
    mask = offsets < n_elements
    vec = tl.load(ptr + offsets, mask=mask)
    for i in range(1, rep):
        vec = val_multiplier_noinline(vec, i)
    tl.store(ptr + offsets, vec, mask=mask)


@triton.jit
def kernel_call_noinline(ptr, n_elements, num1, num2):
    vecmul_kernel(ptr, n_elements, num1)
    vecmul_kernel(ptr, n_elements, num2)


def test_call_noinline_with_tensor_arg_unsupported():
    """A noinline callee cannot take a tensor argument, mirroring upstream.

    Upstream `test_call` also expects the noinline variant to fail
    (`assert err_msg != ""`); the frontend rejects the call instead of
    silently inlining it.
    """
    size = 1024
    x = test_common.generate_tensor(shape=(size, ), dtype="float32").npu()
    with pytest.raises(Exception) as excinfo:
        kernel_call_noinline[(size // 128, )](x, size, 3, 5)
    assert "noinline" in str(excinfo.value)


# ---------------------------------------------------------------------------
# noinline returning / receiving tensor descriptors
# (test_tensor_descriptor.py::test_tensor_descriptor_{in_function,
#  return_value, argument})
# ---------------------------------------------------------------------------


def tensor_descriptor_alloc_fn(size: int, alignment: float, stream: float):
    return torch.empty(size, device="npu", dtype=torch.int8)


@triton.jit(noinline=True)
def tensor_descriptor_in_function_helper(out_ptr, in_ptr, M, N, M_BLOCK: tl.constexpr,
                                         N_BLOCK: tl.constexpr):
    in_desc = tl.make_tensor_descriptor(
        in_ptr,
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
    value = in_desc.load([moffset, noffset])
    out_desc.store([moffset, noffset], value.abs())


@triton.jit
def kernel_tensor_descriptor_in_function(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr,
                                         N_BLOCK: tl.constexpr):
    tensor_descriptor_in_function_helper(out_ptr, a_ptr, M, N, M_BLOCK, N_BLOCK)


def test_tensor_descriptor_in_function():
    triton.set_allocator(tensor_descriptor_alloc_fn)

    M, N = 32, 128
    M_BLOCK, N_BLOCK = 8, 32
    inp = torch.randn((M, N), device="npu")
    out = inp.new_empty((M, N))

    grid = (M // M_BLOCK, N // N_BLOCK)
    kernel_tensor_descriptor_in_function[grid](out, inp, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(inp.abs(), out)


@triton.jit(noinline=True)
def tensor_descriptor_return_helper(ptr, M, N, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr):
    return tl.make_tensor_descriptor(
        ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[M_BLOCK, N_BLOCK],
    )


@triton.jit
def kernel_tensor_descriptor_return_value(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr,
                                          N_BLOCK: tl.constexpr):
    in_desc = tensor_descriptor_return_helper(a_ptr, M, N, M_BLOCK, N_BLOCK)
    out_desc = tensor_descriptor_return_helper(out_ptr, M, N, M_BLOCK, N_BLOCK)
    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK
    value = in_desc.load([moffset, noffset])
    out_desc.store([moffset, noffset], value.abs())


def test_tensor_descriptor_return_value():
    triton.set_allocator(tensor_descriptor_alloc_fn)

    M, N = 32, 128
    M_BLOCK, N_BLOCK = 8, 32
    inp = torch.randn((M, N), device="npu")
    out = inp.new_zeros((M, N))

    grid = (M // M_BLOCK, N // N_BLOCK)
    kernel_tensor_descriptor_return_value[grid](out, inp, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(inp.abs(), out)


@triton.jit(noinline=True)
def tensor_descriptor_arg_helper(in_desc, out_desc, M_BLOCK: tl.constexpr,
                                 N_BLOCK: tl.constexpr):
    moffset = tl.program_id(0) * M_BLOCK
    noffset = tl.program_id(1) * N_BLOCK
    value = in_desc.load([moffset, noffset])
    out_desc.store([moffset, noffset], value.abs())


@triton.jit
def kernel_tensor_descriptor_argument(out_ptr, a_ptr, M, N, M_BLOCK: tl.constexpr,
                                      N_BLOCK: tl.constexpr):
    out_desc = tl.make_tensor_descriptor(out_ptr, shape=[M, N], strides=[N, 1],
                                        block_shape=[M_BLOCK, N_BLOCK])
    in_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, N], strides=[N, 1],
                                       block_shape=[M_BLOCK, N_BLOCK])
    tensor_descriptor_arg_helper(in_desc, out_desc, M_BLOCK, N_BLOCK)


def test_tensor_descriptor_argument():
    triton.set_allocator(tensor_descriptor_alloc_fn)

    M, N = 32, 128
    M_BLOCK, N_BLOCK = 8, 32
    inp = torch.randn((M, N), device="npu")
    out = inp.new_zeros((M, N))

    grid = (M // M_BLOCK, N // N_BLOCK)
    kernel_tensor_descriptor_argument[grid](out, inp, M, N, M_BLOCK, N_BLOCK)
    torch.testing.assert_close(inp.abs(), out)
