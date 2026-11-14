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
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.compiler import make_backend


@triton.jit
def _masked_load(X, M, Y, N, REVERSE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    boundary = offsets < N
    loaded = tl.load(M + offsets, mask=boundary, other=False)
    mask = loaded & boundary if REVERSE else boundary & loaded
    tl.static_assert(loaded.dtype == tl.int8)
    tl.static_assert(mask.dtype == tl.int8)
    value = tl.load(X + offsets, mask=mask, other=0.0)
    tl.static_assert(mask.dtype == tl.int8)
    tl.store(Y + offsets, value, mask=boundary)


def _frontend(mask_type, reverse, arch):
    target = GPUTarget("npu", arch, 0)
    backend = make_backend(target)
    options = backend.parse_options({})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    source = ASTSource(
        _masked_load,
        {"X": "*fp32", "M": mask_type, "Y": "*fp32", "N": "i32", "REVERSE": "constexpr", "BLOCK": "constexpr"},
        constexprs={"REVERSE": reverse, "BLOCK": 1024})
    return source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                          context)


@pytest.mark.parametrize("arch", ["Ascend910B3", "Ascend950PR_9579"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("mask_type", ["*i1", "*i8"])
def test_bool_and_mask_frontend(arch, reverse, mask_type):
    if mask_type == "*i1":
        assert _frontend(mask_type, reverse, arch).verify()
    else:
        with pytest.raises(RuntimeError, match="error encountered during parsing"):
            _frontend(mask_type, reverse, arch)


@pytest.mark.functiontest
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("size", [4, 510, 1024, 2048])
def test_bool_and_load_npu(size, reverse):
    x_cpu = torch.arange(size, dtype=torch.float32) + 0.25
    mask_cpu = torch.arange(size) % 3 != 1
    x, mask = x_cpu.npu(), mask_cpu.npu()
    output = torch.empty_like(x)
    _masked_load[(triton.cdiv(size, 1024), )](x, mask, output, size, REVERSE=reverse, BLOCK=1024)
    torch.testing.assert_close(output.cpu(), torch.where(mask_cpu, x_cpu, 0.0), rtol=0, atol=0)
