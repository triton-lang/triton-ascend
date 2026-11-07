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
import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.compiler import make_backend


@triton.jit
def _bool_mask_consumer(X, M, Y, N, OP: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    boundary = offsets < N
    loaded = tl.load(M + offsets, mask=boundary, other=False)
    if OP == "direct":
        mask = loaded
    elif OP == "and":
        mask = boundary & loaded
    elif OP == "reverse_and":
        mask = loaded & boundary
    elif OP == "or":
        mask = boundary & (loaded | False)
    elif OP == "xor":
        mask = boundary & (loaded ^ False)
    else:
        mask = boundary & ~loaded
    value = tl.load(X + offsets, mask=mask, other=0.0)
    tl.store(Y + offsets, value, mask=mask)


@triton.jit
def _load_dtype(M, Y, SCALAR: tl.constexpr, EXPECTED: tl.constexpr):
    if SCALAR:
        value = tl.load(M)
        tl.static_assert(value.dtype == EXPECTED)
        tl.store(Y, value)
    else:
        offsets = tl.arange(0, 32)
        value = tl.load(M + offsets)
        tl.static_assert(value.dtype == EXPECTED)
        tl.store(Y + offsets, value)


def _frontend(fn, signature, constexprs, arch="Ascend950PR_9579"):
    target = GPUTarget("npu", arch, 0)
    backend = make_backend(target)
    options = backend.parse_options({})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(fn, signature, constexprs=constexprs)
    module = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                            context)
    assert module.verify()
    return module


@pytest.mark.parametrize("arch", ["Ascend910B3", "Ascend950PR_9579"])
@pytest.mark.parametrize("op", ["direct", "and", "reverse_and", "or", "xor", "not"])
def test_bool_load_mask_frontend(arch, op):
    # Exercise actual TTIR verification, including load and store mask types.
    _frontend(_bool_mask_consumer,
              {"X": "*fp32", "M": "*i1", "Y": "*fp32", "N": "i32", "OP": "constexpr", "BLOCK": "constexpr"},
              {"OP": op, "BLOCK": 1024}, arch)


@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("dtype", [tl.int1, tl.int8])
def test_load_preserves_pointee_dtype(scalar, dtype):
    pointer_type = "*i1" if dtype == tl.int1 else "*i8"
    _frontend(_load_dtype, {"M": pointer_type, "Y": pointer_type, "SCALAR": "constexpr", "EXPECTED": "constexpr"},
              {"SCALAR": scalar, "EXPECTED": dtype})


def test_integer_load_does_not_become_valid_mask():
    with pytest.raises((RuntimeError, ValueError, triton.CompilationError)):
        _frontend(_bool_mask_consumer,
                  {"X": "*fp32", "M": "*i8", "Y": "*fp32", "N": "i32", "OP": "constexpr", "BLOCK": "constexpr"},
                  {"OP": "direct", "BLOCK": 1024})


@pytest.mark.functiontest
@pytest.mark.parametrize("size", [33, 1025])
@pytest.mark.parametrize("op", ["direct", "and", "or", "xor", "not"])
def test_bool_load_mask_npu(size, op):
    # Deterministic mixed masks and a sentinel test both masked loads and stores.
    x_cpu = torch.arange(size, dtype=torch.float32) + 0.25
    mask_cpu = torch.arange(size) % 3 != 1
    x = x_cpu.npu()
    mask = mask_cpu.npu()
    output = torch.full((size, ), -123.0, dtype=torch.float32, device=x.device)
    _bool_mask_consumer[(triton.cdiv(size, 1024), )](x, mask, output, size, OP=op, BLOCK=1024)
    expected_mask = ~mask_cpu if op == "not" else mask_cpu
    expected = torch.where(expected_mask, x_cpu, -123.0)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@triton.jit
def _bool_data(M, I, Y, OP: tl.constexpr):
    offsets = tl.arange(0, 32)
    value = tl.load(M + offsets)
    if OP == "invert":
        result = ~value
    elif OP == "mixed_add":
        result = value + tl.load(I + offsets)
    else:
        result = value.to(tl.float32)
    tl.store(Y + offsets, result)


@pytest.mark.functiontest
@pytest.mark.parametrize("op", ["invert", "mixed_add", "float_cast"])
def test_bool_data_consumers_npu(op):
    mask_cpu = torch.arange(32) % 3 != 1
    integer_cpu = (torch.arange(32) - 16).to(torch.int8)
    expected = {
        "invert": (~mask_cpu).to(torch.int8), "mixed_add": mask_cpu.to(torch.int8) + integer_cpu, "float_cast":
        mask_cpu.to(torch.float32)
    }[op]
    mask, integer = mask_cpu.npu(), integer_cpu.npu()
    output = torch.empty_like(expected, device=mask.device)
    _bool_data[(1, )](mask, integer, output, OP=op)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@triton.jit
def _index_select_bool(M, I, Y):
    indices = tl.load(I + tl.arange(0, 2))
    value = extension.index_select_simd(src=M, dim=0, index=indices, src_shape=(4, 32), src_offset=(-1, 0),
                                        read_shape=(-1, 32))
    tl.static_assert(value.dtype == M.dtype.element_ty)
    offsets = tl.arange(0, 2)[:, None] * 32 + tl.arange(0, 32)[None, :]
    # Store to int8 so an accidental bitwise inversion of pseudo-bool bytes is visible.
    tl.store(Y + offsets, ~value)


@pytest.mark.parametrize("pointer", ["*i1", "*i8"])
def test_index_select_preserves_dtype_frontend(pointer):
    _frontend(_index_select_bool, {"M": pointer, "I": "*i32", "Y": "*i8"}, {})


@pytest.mark.functiontest
@pytest.mark.parametrize("dtype", [torch.bool, torch.int8])
def test_index_select_bool_npu(dtype):
    raw = torch.arange(128).reshape(4, 32)
    source_cpu = (raw % 3 != 1) if dtype == torch.bool else (raw - 64).to(dtype)
    indices_cpu = torch.tensor([3, 1], dtype=torch.int32)
    source, indices = source_cpu.npu(), indices_cpu.npu()
    output = torch.empty((2, 32), dtype=torch.int8, device=source.device)
    _index_select_bool[(1, )](source, indices, output)
    expected = (~source_cpu[indices_cpu.long()]).to(torch.int8)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
