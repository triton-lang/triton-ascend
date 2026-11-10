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

import numpy as np
import pytest
import torch_npu  # noqa: F401

import triton
import triton.language as tl
from triton._internal_testing import to_numpy, to_triton
from triton.backends.ascend.utils import is_register_based_arch


@triton.jit
def reduce_with_indices_kernel(X, V, I, ROWS: tl.constexpr, COLS: tl.constexpr, OP: tl.constexpr, LAYOUT: tl.constexpr,
                               OUTPUTS: tl.constexpr, TIE_BREAK_LEFT: tl.constexpr):
    if LAYOUT == "vector":
        x = tl.load(X + tl.arange(0, COLS))
        axis: tl.constexpr = 0
        out_offsets = 0
    else:
        rows = tl.arange(0, ROWS)
        cols = tl.arange(0, COLS)
        x = tl.load(X + rows[:, None] * COLS + cols[None, :])
        if LAYOUT == "rows":
            axis: tl.constexpr = 1
            out_offsets = rows
        else:
            axis: tl.constexpr = 0
            out_offsets = cols
    if OP == "max":
        value, index = tl.max(x, axis=axis, return_indices=True, return_indices_tie_break_left=TIE_BREAK_LEFT)
    else:
        value, index = tl.min(x, axis=axis, return_indices=True, return_indices_tie_break_left=TIE_BREAK_LEFT)
    if OUTPUTS != "index":
        tl.store(V + out_offsets, value)
    if OUTPUTS != "value":
        tl.store(I + out_offsets, index)


def reduce_input(dtype, op, case):
    numpy_dtype = np.dtype("float32" if dtype == "bfloat16" else dtype)
    floating = np.issubdtype(numpy_dtype, np.floating)
    if case == "extremum":
        if floating:
            value = -np.inf if op == "max" else np.inf
        else:
            bounds = np.iinfo(numpy_dtype)
            value = bounds.min if op == "max" else bounds.max
        return np.full(64, value, dtype=numpy_dtype)

    if floating:
        base, best = (-768, -512) if op == "max" else (768, 512)
    elif dtype == "uint64":
        low, high = (1 << 63) + 512, (1 << 63) + 768
        base, best = (low, high) if op == "max" else (high, low)
    else:
        magnitude = (1 << 40) if numpy_dtype.itemsize == 8 else (768 if numpy_dtype.itemsize > 1 else 64)
        if np.issubdtype(numpy_dtype, np.signedinteger) and op == "max":
            base, best = -magnitude, -magnitude // 2
        elif op == "max":
            base, best = magnitude // 2, magnitude
        else:
            base, best = magnitude, magnitude // 2
    x = np.full(64, base, dtype=numpy_dtype)
    x[3] = x[17] = best
    return x


def check_reduce(x, dtype, op, layout, outputs, tie_break_left):
    if dtype in ("uint16", "uint32",
                 "uint64") and not is_register_based_arch(triton.runtime.driver.active.get_current_target().arch):
        pytest.skip("Wide unsigned reductions are unsupported by the A2/A3 lowering")

    axis = 1 if layout == "rows" else 0
    reference = (np.max if op == "max" else np.min)(x, axis=axis).reshape(-1)
    reference_index = (np.argmax if op == "max" else np.argmin)(x, axis=axis).reshape(-1)
    x_tri = to_triton(x, device="npu", dst_type=dtype)
    value = to_triton(np.zeros(reference.shape, dtype=x.dtype), device="npu", dst_type=dtype)
    index = to_triton(np.zeros(reference.shape, dtype=np.int32), device="npu")
    rows, cols = (1, x.size) if x.ndim == 1 else x.shape
    reduce_with_indices_kernel[(1, )](x_tri, value, index, rows, cols, op, layout, outputs, tie_break_left)
    if outputs != "value":
        actual_index = to_numpy(index).reshape(-1)
        assert np.all((actual_index >= 0) & (actual_index < x.shape[axis]))
        if tie_break_left:
            np.testing.assert_array_equal(actual_index, reference_index)
        else:
            # Fast tie-breaking can choose any occurrence of the extreme value.
            shape = list(x.shape)
            shape[axis] = 1
            selected = np.take_along_axis(x, actual_index.reshape(shape), axis=axis)
            np.testing.assert_array_equal(selected.reshape(-1), reference)
    if outputs != "index":
        np.testing.assert_array_equal(to_numpy(value).reshape(-1), reference)


@pytest.mark.parametrize("dtype", [
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "bfloat16",
    "float32",
])
@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("case", ["extremum", "finite"])
@pytest.mark.parametrize("outputs", ["value", "both"])
def test_reduce_with_indices_extrema(dtype, op, case, outputs):
    # Includes the original value-only i8/u8 compile failure and inputs that
    # expose an incorrectly reused 8-bit bound in a wider integer reduction.
    check_reduce(reduce_input(dtype, op, case), dtype, op, "vector", outputs, True)


@pytest.mark.parametrize("dtype", ["int8", "uint16", "float32", "bfloat16"])
@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("layout", ["vector", "rows", "columns"])
@pytest.mark.parametrize("tie_break_left", [True, False])
@pytest.mark.parametrize("outputs", ["index", "both"])
def test_reduce_with_indices_ties(dtype, op, layout, tie_break_left, outputs):
    # All values equal the initial extremum: a made-up initial index must never
    # escape. Check exact indices for the public leftmost-index contract.
    x = reduce_input(dtype, op, "extremum")
    if layout == "rows":
        x = np.broadcast_to(x, (4, 64)).copy()
    elif layout == "columns":
        x = np.broadcast_to(x[:, None], (64, 4)).copy()
    check_reduce(x, dtype, op, layout, outputs, tie_break_left)
