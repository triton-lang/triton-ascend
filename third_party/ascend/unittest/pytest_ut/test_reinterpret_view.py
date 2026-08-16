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

import os

import pytest
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al
from triton._C.libtriton import buffer_ir, ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import ASTSource
from triton.compiler.errors import CompilationError


os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

pytestmark = pytest.mark.backend("cpu")


class Options:
    num_warps = 4
    num_stages = 3
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False
    arch = "Ascend910_95"


def compile_kernel(kernel):
    src = ASTSource(kernel, {}, {})
    context = ir.context()
    ir.load_dialects(context)
    buffer_ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(
        kernel,
        src,
        context,
        Options(),
        {"create_address_space": al.semantic.create_address_space},
        {},
    )
    return str(module)


@triton.jit
def reinterpret_contiguous_prefix_kernel():
    lane = bl.alloc(
        tl.float32, [64, 128], al.ascend_address_space.UB
    )
    pv = lane.reinterpret_view([64, 64], [64, 1])
    pv.to_tensor()


@triton.jit
def reinterpret_zero_stride_kernel():
    lane = bl.alloc(
        tl.float32, [64, 128], al.ascend_address_space.UB
    )
    bl.reinterpret_view(lane, [64, 64], [64, 0])


@triton.jit
def reinterpret_out_of_bounds_kernel():
    lane = bl.alloc(
        tl.float32, [64, 128], al.ascend_address_space.UB
    )
    bl.reinterpret_view(lane, [64, 64], [64, 1], offset=4097)


def test_reinterpret_view_uses_one_physical_allocation():
    mlir = compile_kernel(reinterpret_contiguous_prefix_kernel)

    assert mlir.count("memref.alloc") == 1
    assert "memref.reinterpret_cast" in mlir
    assert "offset: [0], sizes: [64, 64], strides: [64, 1]" in mlir
    assert "memref<64x128xf32" in mlir
    assert "memref<64x64xf32, strided<[64, 1]" in mlir


def test_reinterpret_view_rejects_zero_stride():
    with pytest.raises(CompilationError, match=r"strides\[1\] must be positive"):
        compile_kernel(reinterpret_zero_stride_kernel)


def test_reinterpret_view_rejects_out_of_bounds_footprint():
    with pytest.raises(
        CompilationError,
        match="reinterpret_view footprint exceeds the source allocation",
    ):
        compile_kernel(reinterpret_out_of_bounds_kernel)
