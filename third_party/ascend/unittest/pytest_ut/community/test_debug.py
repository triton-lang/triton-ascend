# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
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

# Source: python/test/unit/test_debug.py at main-dev@396df6cb5b001314e36f22220be07a560de44664

import pytest
import torch
import triton.language as tl
import triton


def test_device_assert_barrier(monkeypatch, device):
    monkeypatch.setenv("TRITON_DEBUG", "1")
    triton.knobs.refresh_knobs()
    tensor = torch.zeros([16], dtype=torch.int32, device=device)

    @triton.jit
    def _kernel(in_ptr0):
        xindex = tl.arange(0, 8)
        tmp0 = tl.load(in_ptr0 + xindex)
        tl.device_assert(tmp0 < 1)

    _kernel[(1, )](tensor)
    getattr(torch, device).synchronize()


@pytest.mark.parametrize("cond", [False, True])
def test_static_assert(cond):

    @triton.jit
    def _kernel(COND: tl.constexpr):
        tl.static_assert(COND)

    if not cond:
        with pytest.raises(triton.compiler.errors.CompileTimeAssertionFailure):
            _kernel[(1, )](cond)
        return

    _kernel[(1, )](cond)


def _test_overflow(x, y, x_dtype, y_dtype, debug, should_overflow, tri_func, ref_func, device):
    x = torch.tensor([x], dtype=getattr(torch, x_dtype), device=device)
    y = torch.tensor([y], dtype=getattr(torch, y_dtype), device=device)
    z = torch.empty_like(x)
    if should_overflow and debug:
        with pytest.raises(RuntimeError) as exc_info:
            tri_func[(1, )](x, y, z, debug=debug)
            getattr(torch, device).synchronize()
        assert "device-side assert" in str(exc_info.value)
    else:
        tri_func[(1, )](x, y, z, debug=debug)
        getattr(torch, device).synchronize()
        assert int(z) == int(ref_func(x, y))


# integer overflow sanitization

# mul overflow

# sub overflow
