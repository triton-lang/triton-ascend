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

import re

import pytest
import torch
import torch_npu  # noqa: F401

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.backends.ascend.utils import is_compile_on_910_95


@triton.jit(do_not_specialize=["offset"])
def _make_range_slice_kernel(output, offset):
    values = tl.arange(16, 80)
    selected = al.extract_slice(values, [offset], [8], [3])
    tl.store(output + tl.arange(0, 8), selected)


def test_make_range_slice_retains_start_and_stride():
    offset = 4
    output = torch.empty(8, dtype=torch.int32, device="npu")
    compiled = _make_range_slice_kernel[(1, )](output, offset)

    # Exercise an actual range extraction, not a precomputed constant slice.
    ttir = compiled.asm["ttir"]
    assert re.search(r"tt.make_range .*end = 80 : i32, start = 16 : i32", ttir)
    assert re.search(r"tensor.extract_slice .*\[8\] \[3\].*tensor<64xi32> to tensor<8xi32>", ttir)
    expected = 16 + offset + torch.arange(8, dtype=torch.int32) * 3
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@triton.jit
def _indirect_load_kernel(source, output, n, row_stride, VOLATILE: tl.constexpr):
    rows = tl.arange(0, 4)
    cols = tl.arange(0, 8)
    offsets = rows[:, None] * row_stride + cols[None, :] % n
    values = tl.load(source + offsets, volatile=VOLATILE)
    tl.store(output + rows[:, None] * 8 + cols[None, :], values)


@pytest.mark.parametrize("volatile", [False, True])
def test_indirect_load_preserves_explicit_volatile(volatile):
    if not is_compile_on_910_95(triton.runtime.driver.active.get_current_target().arch):
        pytest.skip("requires the Ascend 950 indirect-load lowering")

    n, row_stride = 5, 16
    source_cpu = torch.arange(64, dtype=torch.float32)
    source = source_cpu.npu()
    output = torch.empty((4, 8), dtype=torch.float32, device="npu")
    compiled = _indirect_load_kernel[(1, )](source, output, n, row_stride, VOLATILE=volatile,
                                            compile_mode="simd_simt_template")

    offsets = torch.arange(4)[:, None] * row_stride + torch.arange(8)[None, :] % n
    torch.testing.assert_close(output.cpu(), source_cpu[offsets], rtol=0, atol=0)
    # Numerical equality alone cannot detect loss of the volatile property.
    calls = [
        line for line in compiled.asm["ttadapter"].splitlines()
        if re.search(r"\b(?:func\.)?call\s+@triton_indirect_load\w*\(", line)
    ]
    assert len(calls) == 1, compiled.asm["ttadapter"]
    expected = "true" if volatile else "false"
    assert re.search(rf"\bisVolatile\s*=\s*{expected}\b", calls[0]), calls[0]
