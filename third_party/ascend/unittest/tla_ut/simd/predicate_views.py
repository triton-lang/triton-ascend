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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Regression coverage for static, dynamic and nested predicate byte views."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, get_stages


@tla.jit
def predicate_views(X, O, LOAD_OFFSET, STORE_OFFSET, W: tl.constexpr, DIST: tl.constexpr, VIEW: tl.constexpr):
    x = tla.alloc_buffer((512, ), tl.uint8, space="ub")
    out = tla.alloc_buffer((512, ), tl.uint8, space="ub")
    tla.copy(tla.view(X, (512, )), x)
    tla.copy(tla.view(X, (512, )), out)
    if VIEW == "static":
        src = tla.subview(x, offsets=(64, ), sizes=(256, ))
        dst = tla.subview(out, offsets=(128, ), sizes=(256, ))
    elif VIEW == "dynamic":
        src = tla.subview(x, offsets=(LOAD_OFFSET, ), sizes=(256, ))
        dst = tla.subview(out, offsets=(STORE_OFFSET, ), sizes=(256, ))
    else:
        src_outer = tla.subview(x, offsets=(LOAD_OFFSET - 32, ), sizes=(320, ))
        dst_outer = tla.subview(out, offsets=(STORE_OFFSET - 32, ), sizes=(320, ))
        src = tla.subview(src_outer, offsets=(32, ), sizes=(256, ))
        dst = tla.subview(dst_outer, offsets=(32, ), sizes=(256, ))
    with tla.scope.simd():
        pred = tla.simd.pld(src[32], width=W, dist=DIST)
        tla.simd.pst(dst[64], pred)
    tla.copy(out, tla.view(O, (512, )))


PREDICATE_CASES = [(width, dist, view)
                   for width in (8, 16, 32)
                   for dist in ("NORM", "US", "DS")
                   for view in ("static", "dynamic", "nested")]


@pytest.mark.parametrize("case", PREDICATE_CASES)
def test_ir_predicate_view(case):
    width, dist, view = case
    module, _ = compile_offline(predicate_views, dict(X="*u8", O="*u8", LOAD_OFFSET="i32", STORE_OFFSET="i32"),
                                dict(W=width, DIST=dist, VIEW=view))
    assert module.verify()
    assert "llvm." not in str(module)


@pytest.mark.parametrize("case", PREDICATE_CASES)
@pytest.mark.skipif(not get_stages()[2], reason="runtime stage disabled")
def test_run_predicate_view(case):
    width, dist, view = case
    raw = torch.randint(0, 256, (512, ), generator=torch.Generator().manual_seed(331), dtype=torch.uint8)
    out = torch.empty_like(raw, device="npu")
    predicate_views[(1, )](raw.npu(), out, 64, 128, W=width, DIST=dist, VIEW=view)
    torch.npu.synchronize()
    bits = torch.arange(2048 // width)
    if dist == "US":
        bits //= 2
    elif dist == "DS":
        bits *= 2
    logical = (raw.to(torch.int64)[96 + bits // 8] >> (bits % 8)) & 1
    packed = (logical.reshape(-1, 8) << torch.arange(8)).sum(1).to(torch.uint8)
    expected = raw.clone()
    expected[192:192 + packed.numel()] = packed
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
