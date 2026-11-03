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
"""Native predicate operations at logical PB8/PB16/PB32 densities."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, compile_native_kernel


@tla.jit
def predicate_kernel(X, Y, O, W: tl.constexpr, OP: tl.constexpr, PART: tl.constexpr):
    N: tl.constexpr = 2048 // W
    OW: tl.constexpr = (W // 2 if OP == "ppack" else (W * 2 if OP == "punpack" else W))
    ON: tl.constexpr = 2048 // OW
    DT: tl.constexpr = tl.int8 if OW == 8 else (tl.int16 if OW == 16 else tl.int32)
    x = tla.alloc_buffer((32, ), tl.uint8, space="ub")
    y = tla.alloc_buffer((32, ), tl.uint8, space="ub")
    out = tla.alloc_buffer((2, ON), DT, space="ub")
    tla.copy(tla.view(X, (32, )), x)
    tla.copy(tla.view(Y, (32, )), y)
    with tla.scope.simd():
        a = tla.simd.pld(x[0], width=W)
        b = tla.simd.pld(y[0], width=W)
        if OP == "ppack":
            p = tla.simd.ppack(a, part=PART)
            q = tla.simd.ppack(b, part=PART)
        elif OP == "punpack":
            p = tla.simd.punpack(a, part=PART)
            q = tla.simd.punpack(b, part=PART)
        elif OP == "pintlv":
            p, q = tla.simd.pintlv(a, b)
        else:
            p, q = tla.simd.pdintlv(a, b)
        zero = tla.simd.vsplat(0, DT)
        one = tla.simd.vsplat(1, DT)
        tla.simd.vsts(out[0, 0], tla.simd.vsel(p, one, zero), dist="NORM_B8" if OW == 8 else
                      ("NORM_B16" if OW == 16 else "NORM_B32"))
        tla.simd.vsts(out[1, 0], tla.simd.vsel(q, one, zero), dist="NORM_B8" if OW == 8 else
                      ("NORM_B16" if OW == 16 else "NORM_B32"))
    tla.copy(out, tla.view(O, (2, ON)))


CASES = [(op, w, part)
         for op, widths in (("ppack", (16, 32)), ("punpack", (8, 16)))
         for w in widths
         for part in ("lower", "higher")]
CASES += [(op, w, "lower") for op in ("pintlv", "pdintlv") for w in (8, 16, 32)]


def config(case):
    op, w, part = case
    ow = w // 2 if op == "ppack" else w * 2 if op == "punpack" else w
    return dict(W=w, OP=op, PART=part), ow


def launch(case, pattern):
    kwargs, ow = config(case)
    op, w, part = case
    n = 2048 // w
    lane = torch.arange(256)
    if pattern == "mixed":
        a, b = (lane * 7 + 3) % 11 < 5, (lane * 3 + 1) % 13 < 6
    elif pattern == "edge":
        a, b = (lane == n // 2 - 1) | (lane == n // 2), (lane == 0) | (lane == n - 1)
    elif pattern == "ones":
        a, b = lane >= 0, lane >= 0
    else:
        a, b = lane < 0, lane < 0
    pack = lambda x: (x.to(torch.int64).reshape(-1, 8) << torch.arange(8)).sum(1).to(torch.uint8).npu()
    out = torch.empty((2, 2048 // ow), dtype={8: torch.int8, 16: torch.int16, 32: torch.int32}[ow], device="npu")
    compiled = predicate_kernel[(1, )](pack(a), pack(b), out, **kwargs)
    torch.npu.synchronize()
    a, b = a[:n], b[:n]
    if op == "ppack":
        z = torch.zeros(n, dtype=torch.bool)
        ref = torch.stack([torch.cat((x, z) if part == "lower" else (z, x)) for x in (a, b)])
    elif op == "punpack":
        start = 0 if part == "lower" else n // 2
        ref = torch.stack([x[start:start + n // 2] for x in (a, b)])
    elif op == "pintlv":
        ref = torch.stack((a, b), dim=1).flatten().reshape(2, n)
    else:
        joined = torch.cat((a, b))
        ref = torch.stack((joined[::2], joined[1::2]))
    return out.cpu(), ref.to(out.dtype), compiled


@pytest.mark.parametrize("case", CASES)
def test_ir(case):
    kwargs, ow = config(case)
    module, _ = compile_offline(predicate_kernel, dict(X="*u8", Y="*u8", O=f"*i{ow}"), kwargs)
    text = str(module)
    op, w, _ = case
    assert f'llvm.hivm.{op}' in text
    if op in ("pintlv", "pdintlv"):
        assert f'llvm.hivm.{op}.b{w}' in text
    assert "ave.hir.preg" not in text


@pytest.mark.parametrize("case", CASES)
def test_compile(case):
    kwargs, ow = config(case)
    compile_native_kernel(predicate_kernel, dict(X="*u8", Y="*u8", O=f"*i{ow}"), kwargs)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("pattern", ("mixed", "edge", "ones", "zero"))
def test_run(case, pattern):
    actual, expected, _ = launch(case, pattern)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
