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
"""Indexed UB stores: masks, sentinels, view offsets and load-after-store."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, compile_native_kernel, lower_offline, get_stages


@tla.jit
def scatter_kernel(X, I, M, INIT, O, CHECK, OFFSET, DT: tl.constexpr, W: tl.constexpr, VIEW: tl.constexpr,
                   MASKED: tl.constexpr, READBACK: tl.constexpr, INNER: tl.constexpr):
    N: tl.constexpr = 2048 // W
    IT: tl.constexpr = tl.uint32 if W == 32 else tl.uint16
    MT: tl.constexpr = tl.int8 if W == 8 else (tl.int16 if W == 16 else tl.int32)
    x = tla.alloc_buffer((N, ), DT, space="ub")
    indices = tla.alloc_buffer((N, ), IT, space="ub")
    m = tla.alloc_buffer((N, ), MT, space="ub")
    out = tla.alloc_buffer((1024, ), DT, space="ub")
    check = tla.alloc_buffer((N, ), DT, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(I, (N, )), indices)
    tla.copy(tla.view(M, (N, )), m)
    tla.copy(tla.view(INIT, (1024, )), out)
    if VIEW == "static":
        target = tla.subview(out, offsets=(64, ), sizes=(960, ))
    elif VIEW == "dynamic":
        target = tla.subview(out, offsets=(OFFSET, ), sizes=(960, ))
    else:
        target = out
    with tla.scope.simd():
        data = tla.simd.vlds(x[0])
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        if not MASKED:
            mask = None
        if W == 8:
            idx = (tla.simd.vlds(indices[0]), tla.simd.vlds(indices[128]))
        else:
            idx = tla.simd.vlds(indices[0])
        if INNER:
            # Trace a view created inside the VF back to its captured base.
            target = tla.subview(target, offsets=(8, ), sizes=(952, ))
        tla.simd.vscatter(target[32], idx, data, mask)
        # Alias read after the native store must observe its memory effects.
        if READBACK:
            tla.simd.membar("VST_VLD")
            back = tla.simd.vgather(target[32], idx, mask)
        else:
            back = tla.simd.vsplat(0, DT)
        tla.simd.vsts(check[0], back, dist="NORM_B8" if W == 8 else ("NORM_B16" if W == 16 else "NORM_B32"))
    tla.copy(out, tla.view(O, (1024, )))
    tla.copy(check, tla.view(CHECK, (N, )))


TYPES = [("int8", "i8", 8), ("uint8", "u8", 8), ("int16", "i16", 16), ("uint16", "u16", 16), ("int32", "i32", 32),
         ("uint32", "u32", 32), ("float16", "fp16", 16), ("bfloat16", "bf16", 16), ("float32", "fp32", 32)]
CASES = [(typ, view) for typ in TYPES for view in ("none", "static", "dynamic")]


def constants(case, masked, readback=True, inner=False):
    (name, _, w), view = case
    return dict(DT=getattr(tl, name), W=w, VIEW=view, MASKED=masked, READBACK=readback, INNER=inner)


def launch(case, pattern, readback=True, inner=False):
    (name, _, w), view = case
    n = 2048 // w
    dtype = getattr(torch, name)
    lane = torch.arange(n)
    data = ((lane * 17 + 3) % 101).to(dtype)
    index = (lane * 3 + 5).flip(0)
    mask = {
        "unmasked": lane >= 0, "all": lane >= 0, "none": lane < 0, "tail": lane < n - 3, "mixed": (lane * 7 + 3) % 11
        < 5
    }[pattern]
    poisoned = torch.where(mask, index, 0xFFFF)
    init = torch.full((1024, ), 111, dtype=dtype)
    out = torch.empty_like(init, device="npu")
    check = torch.empty(n, dtype=dtype, device="npu")
    compiled = scatter_kernel[(1, )](data.npu(), poisoned.to(torch.uint32 if w == 32 else torch.uint16).npu(),
                                     mask.to({8: torch.int8, 16: torch.int16, 32: torch.int32}[w]).npu(), init.npu(),
                                     out, check, 64, **constants(case, pattern != "unmasked", readback, inner))
    torch.npu.synchronize()
    expected = init.to(torch.float64)
    expected[32 + (64 if view != "none" else 0) + (8 if inner else 0) + index[mask]] = data.to(torch.float64)[mask]
    check_ref = torch.where(mask, data.to(torch.float64), 0) if readback else torch.zeros(n, dtype=torch.float64)
    return out.cpu().to(torch.float64), expected, check.cpu().to(torch.float64), check_ref, compiled


@pytest.mark.parametrize("case", CASES)
def test_ir(case):
    (_, sig, w), _ = case
    module, ctx = compile_offline(
        scatter_kernel,
        dict(X=f"*{sig}", I="*u32" if w == 32 else "*u16", M=f"*i{w}", INIT=f"*{sig}", O=f"*{sig}", CHECK=f"*{sig}",
             OFFSET="i32"), constants(case, True))
    text = str(module)
    assert text.count("ave.hir.vscatter ") == 1
    assert "llvm.call_intrinsic" not in text
    assert "llvm.inttoptr" not in text
    assert "ave.hir.vintlv" not in text
    assert ("extra" in text) == (w == 8)
    lowered = str(lower_offline(module, ctx))
    assert "ave.hir.vscatter" in lowered
    assert module.verify()


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("readback", (False, True))
@pytest.mark.parametrize("inner", (False, True))
def test_compile(case, readback, inner):
    (_, sig, w), _ = case
    kernel = compile_native_kernel(
        scatter_kernel,
        dict(X=f"*{sig}", I="*u32" if w == 32 else "*u16", M=f"*i{w}", INIT=f"*{sig}", O=f"*{sig}", CHECK=f"*{sig}",
             OFFSET="i32"), constants(case, True, readback, inner))
    assert "ave.hir.vscatter" in kernel.asm["ttadapter"]


@pytest.mark.parametrize("case,inner",
                         [(case, False) for case in CASES] + [(case, True) for case in CASES if case[1] == "dynamic"])
@pytest.mark.parametrize("pattern", ("unmasked", "all", "none", "tail", "mixed"))
@pytest.mark.parametrize("readback", (False, True))
def test_run(case, inner, pattern, readback):
    actual, expected, check, check_ref, _ = launch(case, pattern, readback, inner)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(check, check_ref, rtol=0, atol=0)


@tla.jit
def two_scopes_kernel(O):
    out = tla.alloc_buffer((64, ), tl.int32, space="ub")
    with tla.scope.simd():
        idx = tla.simd.vsplat(0, tl.uint32)
        data = tla.simd.vsplat(7, tl.int32)
        tla.simd.vscatter(out[0], idx, data, tla.simd.pge(1, 64))
    with tla.scope.simd():
        idx = tla.simd.vsplat(1, tl.uint32)
        data = tla.simd.vsplat(9, tl.int32)
        tla.simd.vscatter(out[0], idx, data, tla.simd.pge(1, 64))
    tla.copy(out, tla.view(O, (64, )))


def test_ir_multiple_scopes():
    module, ctx = compile_offline(two_scopes_kernel, dict(O="*i32"), {})
    text = str(lower_offline(module, ctx))
    assert text.count("ave.hir.vscatter") == 2
    assert module.verify()


@tla.jit
def indexed_boundary_kernel(X, I, M, DATA, O, CHECK, W: tl.constexpr):
    N: tl.constexpr = 2048 // W
    DT: tl.constexpr = tl.int8 if W == 8 else tl.int16
    x = tla.alloc_buffer((65536, ), DT, space="ub")
    indices = tla.alloc_buffer((N, ), tl.uint16, space="ub")
    masks = tla.alloc_buffer((N, ), DT, space="ub")
    data = tla.alloc_buffer((N, ), DT, space="ub")
    check = tla.alloc_buffer((2, N), DT, space="ub")
    tla.copy(tla.view(X, (65536, )), x)
    tla.copy(tla.view(I, (N, )), indices)
    tla.copy(tla.view(M, (N, )), masks)
    tla.copy(tla.view(DATA, (N, )), data)
    with tla.scope.simd():
        if W == 8:
            idx = (tla.simd.vlds(indices[0]), tla.simd.vlds(indices[128]))
        else:
            idx = tla.simd.vlds(indices[0])
        mask = tla.simd.vcmps(tla.simd.vlds(masks[0]), 0, op=tla.simd.CMP.NE)
        before = tla.simd.vgather(x[0], idx, mask)
        tla.simd.membar("VLD_VST")
        tla.simd.vscatter(x[0], idx, tla.simd.vlds(data[0]), mask)
        tla.simd.membar("VST_VLD")
        after = tla.simd.vld(x[0], idx, mask)
        tla.simd.vsts(check[0, 0], before, dist="NORM_B8" if W == 8 else "NORM_B16")
        tla.simd.vsts(check[1, 0], after, dist="NORM_B8" if W == 8 else "NORM_B16")
    tla.copy(x, tla.view(O, (65536, )))
    tla.copy(check, tla.view(CHECK, (2, N)))


@pytest.mark.parametrize("width", (8, 16))
@pytest.mark.parametrize("active", ("all", "lower_edge", "upper_edge"))
@pytest.mark.skipif(not get_stages()[2], reason="runtime stage disabled")
def test_run_unsigned_index_boundary(width, active):
    n = 2048 // width
    dtype = torch.int8 if width == 8 else torch.int16
    lane = torch.arange(n)
    # Exercise unsigned i16 offsets on both sides of 32768, up to 65535.
    index = torch.cat((32768 - n // 2 + torch.arange(n // 2), 65536 - n // 2 + torch.arange(n // 2)))
    mask = lane >= 0 if active == "all" else lane == (n // 2 - 1 if active == "lower_edge" else n // 2)
    raw = ((torch.arange(65536) * 37 + 9) % 101).to(dtype)
    data = (-(lane % 97) - 1).to(dtype)
    out = torch.empty_like(raw, device="npu")
    check = torch.empty((2, n), dtype=dtype, device="npu")
    indexed_boundary_kernel[(1, )](raw.npu(), index.to(torch.uint16).npu(), mask.to(dtype).npu(), data.npu(), out,
                                   check, W=width)
    torch.npu.synchronize()
    expected = raw.clone()
    expected[index[mask]] = data[mask]
    expected_check = torch.stack((torch.where(mask, raw[index], 0), torch.where(mask, data, 0)))
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(check.cpu(), expected_check, rtol=0, atol=0)
