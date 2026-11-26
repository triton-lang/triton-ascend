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

import inspect
import os
import re
from unittest.mock import Mock

import pytest
import triton
import torch
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al
from triton.language.extra.cann.extension import index_put, gather_out_to_ub, scatter_ub_to_out
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir, buffer_ir, ascend
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.compiler.errors import MLIRCompilationError
from triton.backends.ascend import _apply_ascend_patch

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

_apply_ascend_patch()


class Options:
    num_warps = 4
    num_stages = 3
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False
    arch = "Ascend910_9589"


def compile_kernel(kernel, signature, constants):
    """Helper to compile a kernel to MLIR."""
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    buffer_ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    ascend.load_dialects(context)
    module = ast_to_ttir(kernel, src, context, Options(), {"create_address_space": al.semantic.create_address_space},
                         {})
    return str(module)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("kernel_size, dilation, padding, expected_padding, output_length", [
    (1, 1, "same", (0, 0), 8),
    (3, 1, "same", (1, 1), 8),
    (4, 1, "same", (1, 2), 8),
    (3, 2, "same", (2, 2), 8),
    (4, 2, "same", (3, 3), 8),
    (4, 3, "same", (4, 5), 8),
    (3, 1, 0, 0, 6),
    (3, 1, 1, 1, 8),
    (4, 1, (1, 2), (1, 2), 8),
    (3, 1, "valid", 0, 6),
])
def test_conv1d_padding_frontend(monkeypatch, batched, kernel_size, dilation, padding, expected_padding, output_length):
    # Exercise the real frontend with constexpr tensor shapes, stopping before
    # IR construction so padding normalization is checked independently.
    input_shape = [2, 2, 8] if batched else [2, 8]
    input_tensor = tl.tensor(None, tl.block_type(tl.float16, input_shape))
    weight = tl.tensor(None, tl.block_type(tl.float16, [4, 2, kernel_size]))
    semantic_conv1d = Mock()
    monkeypatch.setattr(al.semantic, "conv1d", semantic_conv1d)
    semantic = object()

    al.conv1d(input_tensor, weight, padding=padding, dilation=dilation, _semantic=semantic)

    output_shape = [2, 4, output_length] if batched else [4, output_length]
    semantic_conv1d.assert_called_once_with(input_tensor, weight, None, 1, expected_padding, dilation, 1, output_shape,
                                            semantic)
    actual_padding = semantic_conv1d.call_args.args[4]
    padding_values = actual_padding if isinstance(actual_padding, tuple) else (actual_padding, )
    assert all(type(value) is int for value in padding_values)
    assert type(semantic_conv1d.call_args.args[7][-1]) is int


def test_conv1d_same_rejects_strided_convolution(monkeypatch):
    input_tensor = tl.tensor(None, tl.block_type(tl.float16, [2, 8]))
    weight = tl.tensor(None, tl.block_type(tl.float16, [4, 2, 3]))
    semantic_conv1d = Mock()
    monkeypatch.setattr(al.semantic, "conv1d", semantic_conv1d)

    with pytest.raises(ValueError, match="padding='same' is only supported when stride=1"):
        al.conv1d(input_tensor, weight, padding="same", stride=2, _semantic=object())

    semantic_conv1d.assert_not_called()


CONV1D_PADDING_SIZE_ERROR = "conv1d() got an unexpected keyword argument 'padding_size'; use 'padding' instead."


@pytest.mark.parametrize("padding_size,also_padding", [
    (0, False),
    (None, False),
    ((1, 2), False),
    (tl.constexpr(None), False),
    (0, True),
])
def test_conv1d_padding_size_reports_replacement(monkeypatch, padding_size, also_padding):
    x = tl.tensor(None, tl.block_type(tl.float16, [2, 8]))
    weight = tl.tensor(None, tl.block_type(tl.float16, [4, 2, 1]))
    semantic_conv1d = Mock()
    monkeypatch.setattr(al.semantic, "conv1d", semantic_conv1d)
    kwargs = {"padding_size": padding_size}
    if also_padding:
        kwargs["padding"] = 0

    with pytest.raises(TypeError) as error:
        al.conv1d(x, weight, **kwargs, _semantic=object())

    assert str(error.value) == CONV1D_PADDING_SIZE_ERROR
    semantic_conv1d.assert_not_called()


def test_conv1d_padding_signature_and_positional_call(monkeypatch):
    assert list(inspect.signature(
        al.conv1d).parameters) == ["input", "weight", "bias", "stride", "padding", "dilation", "groups", "_semantic"]
    x = tl.tensor(None, tl.block_type(tl.float16, [2, 8]))
    weight = tl.tensor(None, tl.block_type(tl.float16, [4, 2, 4]))
    semantic_conv1d = Mock()
    monkeypatch.setattr(al.semantic, "conv1d", semantic_conv1d)
    semantic = object()

    al.conv1d(x, weight, None, 1, (1, 2), 1, 1, _semantic=semantic)

    semantic_conv1d.assert_called_once_with(x, weight, None, 1, (1, 2), 1, 1, [4, 8], semantic)


def test_conv1d_other_unknown_keyword_remains_an_error(monkeypatch):
    semantic_conv1d = Mock()
    monkeypatch.setattr(al.semantic, "conv1d", semantic_conv1d)
    x = tl.tensor(None, tl.block_type(tl.float16, [2, 8]))
    weight = tl.tensor(None, tl.block_type(tl.float16, [4, 2, 1]))

    with pytest.raises(TypeError, match="unexpected keyword argument 'padding_sze'"):
        al.conv1d(x, weight, padding_sze=0, _semantic=object())

    semantic_conv1d.assert_not_called()


@triton.jit
def kernel_conv1d_legacy_padding():
    x = tl.full((2, 8), 1, tl.float16)
    weight = tl.full((4, 2, 1), 1, tl.float16)
    al.conv1d(x, weight, padding_size=0)


def test_conv1d_padding_size_jit_reports_replacement(monkeypatch):
    semantic_conv1d = Mock()
    monkeypatch.setattr(al.semantic, "conv1d", semantic_conv1d)

    with pytest.raises(triton.CompilationError, match=re.escape(CONV1D_PADDING_SIZE_ERROR)):
        compile_kernel(kernel_conv1d_legacy_padding, {}, {})

    semantic_conv1d.assert_not_called()


@triton.jit
def kernel_conv1d_padding(Y, K: tl.constexpr, PADDING: tl.constexpr):
    x = tl.full((2, 8), 1, tl.float16)
    weight = tl.full((4, 2, K), 1, tl.float16)
    result = al.conv1d(x, weight, padding=PADDING)
    tl.store(Y + tl.arange(0, 32), tl.reshape(result, (32, )))


@pytest.mark.parametrize("kernel_size,padding,padding_attr", [
    (1, 0, "padding = 0 : i64"),
    (4, (1, 2), "padding = array<i32: 1, 2>"),
])
def test_conv1d_padding_real_ttir(kernel_size, padding, padding_attr):
    module = compile_kernel(kernel_conv1d_padding, {"Y": "*fp16"}, {"K": kernel_size, "PADDING": padding})
    convolution = next(line for line in module.splitlines() if "ascend.conv1d" in line)
    assert padding_attr in convolution
    assert "-> tensor<4x8xf16>" in convolution


# Test cases for core.py
@triton.jit
def kernel_core_functions(M: tl.constexpr, N: tl.constexpr):
    # Test buffer allocation with different address spaces
    buf_ub = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)
    buf_l1 = bl.alloc(tl.float32, [M, N], al.ascend_address_space.L1)
    buf_l0a = bl.alloc(tl.float32, [M, N], al.ascend_address_space.L0A)
    buf_l0b = bl.alloc(tl.float32, [M, N], al.ascend_address_space.L0B)
    buf_l0c = bl.alloc(tl.float32, [M, N], al.ascend_address_space.L0C)

    # Test int64
    int64_val = al.int64(1234567890123456789)

    # Test sub_vec_id and sub_vec_num
    vec_id = al.sub_vec_id()
    vec_num = al.sub_vec_num()

    # Test debug_barrier
    al.debug_barrier(al.SYNC_IN_VF.VV_ALL)

    # Test fixpipe (simple usage)
    data = tl.full([M, N], 0.0, dtype=tl.float32)
    result_fixpipe = al.fixpipe(data, buf_ub)


# Test cases for math_ops.py
@triton.jit
def kernel_math_ops(M: tl.constexpr, N: tl.constexpr):
    # Test atan2
    x = tl.full([M, N], 1.0, dtype=tl.float32)
    y = tl.full([M, N], 1.0, dtype=tl.float32)
    result_atan2 = al.atan2(y, x)

    # Test isfinited
    result_isfinited = al.isfinited(x)

    # Test finitef
    result_finitef = al.finitef(x)


# Test cases for vec_ops.py
@triton.jit
def kernel_vec_ops(M: tl.constexpr, N: tl.constexpr):
    # Test insert_slice
    data = tl.full([M, N], 0.0, dtype=tl.float32)
    slice_data = tl.full([M // 2, N // 2], 1.0, dtype=tl.float32)
    result_insert = al.insert_slice(data, slice_data, [0, 0], [M // 2, N // 2], [1, 1])

    # Test extract_slice
    result_extract = al.extract_slice(data, [0, 0], [M // 2, N // 2], [1, 1])

    # Test get_element
    result_get = al.get_element(data, [0, 0])

    # Test cast
    data_int = tl.full([M, N], 1, dtype=tl.int32)
    result_cast = al.cast(data_int, tl.float32)


# Test cases for aux_ops.py
@triton.jit
def kernel_aux_ops(M: tl.constexpr, N: tl.constexpr):
    # Test parallel
    data = tl.full([M, N], 0.0, dtype=tl.float32)
    for _ in al.parallel(M):
        pass

    # Test compile_hint
    al.compile_hint(data, "core_mode")

    # Test multibuffer
    al.multibuffer(data, 2)


# Test cases for scope.py
@triton.jit
def kernel_scope(M: tl.constexpr, N: tl.constexpr):
    # Test scope with core_mode
    data = tl.full([M, N], 0.0, dtype=tl.float32)
    with al.scope(core_mode="vector"):
        result = data + 1.0


# Test cases for sync operations
@triton.jit
def kernel_sync_operations(M: tl.constexpr, N: tl.constexpr):
    # Test sync_block_all
    al.sync_block_all("all", 0)

    # Test sync_block_set and sync_block_wait
    al.sync_block_set("cube", "vector", 0)
    al.sync_block_wait("cube", "vector", 0)


# Test cases for custom_op.py
@triton.jit
def kernel_custom_op(M: tl.constexpr, N: tl.constexpr):
    # Test custom op functionality (placeholder)
    data = tl.full([M, N], 0.0, dtype=tl.float32)
    # Actual custom op usage would depend on specific ops


# Test function definitions
def test_core_functions():
    print("=" * 60)
    print("Test 1: Core Functions")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_core_functions,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


def test_math_ops():
    print("=" * 60)
    print("Test 2: Math Operations")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_math_ops,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


def test_vec_ops():
    print("=" * 60)
    print("Test 4: Vector Operations")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_vec_ops,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


def test_aux_ops():
    print("=" * 60)
    print("Test 5: Auxiliary Operations")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_aux_ops,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


def test_scope():
    print("=" * 60)
    print("Test 6: Scope Operations")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_scope,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


def test_sync_operations():
    print("=" * 60)
    print("Test 7: Sync Operations")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_sync_operations,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


def test_custom_op():
    print("=" * 60)
    print("Test 8: Custom Operations")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_custom_op,
        {},
        {"M": 16, "N": 16},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")


# ============== Main for manual testing ==============
if __name__ == "__main__":
    test_core_functions()
    test_math_ops()
    test_vec_ops()
    test_aux_ops()
    test_scope()
    test_sync_operations()
    test_custom_op()
