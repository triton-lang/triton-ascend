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
from utils.tla_test_utils import compile_offline, lower_offline, compile_native_kernel, get_stages


@tla.jit
def predicate_kernel(X, Y, O, W: tl.constexpr, OP: tl.constexpr, PART: tl.constexpr):
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
    module, ctx = compile_offline(predicate_kernel, dict(X="*u8", Y="*u8", O=f"*i{ow}"), kwargs)
    text = str(module)
    op, w, _ = case
    ave_op = {'ppack': 'pack', 'punpack': 'unpack', 'pintlv': 'interleave', 'pdintlv': 'deinterleave'}[op]
    assert f'ave.hir.preg.{ave_op}' in text
    assert 'llvm.call_intrinsic' not in text
    assert 'ave.hir.preg.cast' not in text
    lowered = str(lower_offline(module, ctx))
    assert f'ave.hir.preg.{ave_op}' in lowered
    assert 'llvm.call_intrinsic' not in lowered


@pytest.mark.parametrize("case", CASES)
def test_compile(case):
    kwargs, ow = config(case)
    compile_native_kernel(predicate_kernel, dict(X="*u8", Y="*u8", O=f"*i{ow}"), kwargs)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("pattern", ("mixed", "edge", "ones", "zero"))
def test_run(case, pattern):
    actual, expected, _ = launch(case, pattern)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@tla.jit
def pack_generated_kernel(O, W: tl.constexpr, COUNT: tl.constexpr):
    N: tl.constexpr = 2048 // W
    DT: tl.constexpr = tl.int8 if W == 16 else tl.int16
    out = tla.alloc_buffer((2, 2 * N), DT, space="ub")
    with tla.scope.simd():
        p = tla.simd.pge(COUNT, N)
        lo = tla.simd.ppack(p, part=tla.simd.PART.LOWER)
        hi = tla.simd.ppack(p, part=tla.simd.PART.HIGHER)
        one = tla.simd.vsplat(1, DT)
        zero = tla.simd.vsplat(0, DT)
        tla.simd.vsts(out[0, 0], tla.simd.vsel(lo, one, zero), dist="NORM_B8" if W == 16 else "NORM_B16")
        tla.simd.vsts(out[1, 0], tla.simd.vsel(hi, one, zero), dist="NORM_B8" if W == 16 else "NORM_B16")
    tla.copy(out, tla.view(O, (2, 2 * N)))


@pytest.mark.parametrize("width,count", ((16, 8), (16, 128), (32, 8), (32, 64)))
def test_ir_generated_pack(width, count):
    module, ctx = compile_offline(pack_generated_kernel, dict(O=f"*i{width // 2}"), dict(W=width, COUNT=count))
    text = str(lower_offline(module, ctx))
    assert text.count("ave.hir.preg.pack") == 2
    assert "ave.hir.preg.cast" not in text


@pytest.mark.parametrize("width,count", ((16, 8), (16, 128), (32, 8), (32, 64)))
@pytest.mark.skipif(not get_stages()[2], reason="runtime stage disabled")
def test_run_generated_pack(width, count):
    # A shared pge producer must keep logical padding after canonicalization.
    n = 2048 // width
    out = torch.empty((2, 2 * n), dtype=torch.int8 if width == 16 else torch.int16, device="npu")
    pack_generated_kernel[(1, )](out, W=width, COUNT=count)
    torch.npu.synchronize()
    src = torch.arange(n) < count
    zero = torch.zeros(n, dtype=torch.bool)
    expected = torch.stack((torch.cat((src, zero)), torch.cat((zero, src)))).to(out.dtype)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


@tla.jit
def predicate_roundtrip_kernel(X, Y, O, COUNT, W: tl.constexpr, PART: tl.constexpr):
    N: tl.constexpr = 2048 // W
    DT: tl.constexpr = tl.int8 if W == 8 else (tl.int16 if W == 16 else tl.int32)
    x = tla.alloc_buffer((N, ), DT, space="ub")
    y = tla.alloc_buffer((N, ), DT, space="ub")
    out = tla.alloc_buffer((4, N), DT, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(Y, (N, )), y)
    with tla.scope.simd():
        a = tla.simd.vcmps(tla.simd.vlds(x[0]), COUNT, op=tla.simd.CMP.LT)
        b = tla.simd.vcmps(tla.simd.vlds(y[0]), 0, op=tla.simd.CMP.NE)
        lo, hi = tla.simd.pintlv(a, b)
        ra, rb = tla.simd.pdintlv(lo, hi)
        if W != 8:
            ra = tla.simd.punpack(tla.simd.ppack(ra, part=PART), part=PART)
            rb = tla.simd.punpack(tla.simd.ppack(rb, part=PART), part=PART)
        zero = tla.simd.vsplat(0, DT)
        one = tla.simd.vsplat(1, DT)
        # Consume the shared producers as well as both reconstructed results.
        tla.simd.vsts(out[0, 0], tla.simd.vsel(a, one, zero), dist="NORM_B8" if W == 8 else
                      ("NORM_B16" if W == 16 else "NORM_B32"))
        tla.simd.vsts(out[1, 0], tla.simd.vsel(b, one, zero), dist="NORM_B8" if W == 8 else
                      ("NORM_B16" if W == 16 else "NORM_B32"))
        tla.simd.vsts(out[2, 0], tla.simd.vsel(ra, one, zero), dist="NORM_B8" if W == 8 else
                      ("NORM_B16" if W == 16 else "NORM_B32"))
        tla.simd.vsts(out[3, 0], tla.simd.vsel(rb, one, zero), dist="NORM_B8" if W == 8 else
                      ("NORM_B16" if W == 16 else "NORM_B32"))
    tla.copy(out, tla.view(O, (4, N)))


@pytest.mark.parametrize("width,part", [(8, "lower")] + [(w, p) for w in (16, 32) for p in ("lower", "higher")])
@pytest.mark.parametrize("count", (-1, 1, 31, 63))
@pytest.mark.skipif(not get_stages()[2], reason="runtime stage disabled")
def test_run_roundtrip(width, part, count):
    n = 2048 // width
    dtype = {8: torch.int8, 16: torch.int16, 32: torch.int32}[width]
    lane = torch.arange(n)
    x = (lane % 64).to(dtype)
    y = ((lane * 7 + 3) % 11 < 5).to(dtype)
    out = torch.empty((4, n), dtype=dtype, device="npu")
    predicate_roundtrip_kernel[(1, )](x.npu(), y.npu(), out, count, W=width, PART=part)
    torch.npu.synchronize()
    expected = torch.stack((x < count, y != 0, x < count, y != 0)).to(dtype)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
