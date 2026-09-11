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
"""Regression test for the ``set_buffer_count`` module argument.

``ascend.passes.ttir.set_buffer_count`` takes the MLIR module as its first
argument.  Kernels that set ``intra_cache_num``, ``inter_cache_num`` or
``load_cache_num`` used to call it without that argument, so compilation failed
with::

    TypeError: set_buffer_count(): incompatible function arguments.

The kernel below is intentionally minimal: the only thing under test is that a
real compilation with each buffer-count option reaches the option handling in
``ttir_to_linalg`` without raising.
"""

import pytest
import torch
import torch_npu  # noqa: F401  # register the "npu" device

import triton
import triton.language as tl
from triton.backends.ascend.utils import is_compile_on_910_95

pytestmark = pytest.mark.skipif(
    not is_compile_on_910_95(),
    reason="buffer-count compile options are validated on Ascend 910_95 only",
)


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(out_ptr + offsets, x + y)


@pytest.mark.parametrize(
    "buffer_count_kwargs",
    [
        pytest.param({"intra_cache_num": 3}, id="intra_cache_num"),
        pytest.param({"inter_cache_num": 2}, id="inter_cache_num"),
        pytest.param({"load_cache_num": 1}, id="load_cache_num"),
    ],
)
def test_set_buffer_count_receives_module(monkeypatch, buffer_count_kwargs):
    # A cached compilation would skip the compiler path that contains the
    # regression, so force one real compilation for every parametrization.
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    size = 256
    x = torch.randn(size, device="npu")
    y = torch.randn(size, device="npu")
    out = torch.empty_like(x)

    _add_kernel[(1, )](x, y, out, BLOCK=size, **buffer_count_kwargs)
    torch.npu.synchronize()

    torch.testing.assert_close(out, x + y)
