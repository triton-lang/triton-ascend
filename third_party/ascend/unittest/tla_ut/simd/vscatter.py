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
from utils.tla_test_utils import compile_offline, compile_native_kernel, lower_offline


@tla.jit
def scatter_kernel(X, I, M, INIT, O, CHECK, OFFSET, DT: tl.constexpr, W: tl.constexpr, VIEW: tl.constexpr,
                   MASKED: tl.constexpr, READBACK: tl.constexpr):
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


def constants(case, masked, readback=True):
    (name, _, w), view = case
    return dict(DT=getattr(tl, name), W=w, VIEW=view, MASKED=masked, READBACK=readback)


def launch(case, pattern, readback=True):
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
                                     out, check, 64, **constants(case, pattern != "unmasked", readback))
    torch.npu.synchronize()
    expected = init.to(torch.float64)
    expected[32 + (64 if view != "none" else 0) + index[mask]] = data.to(torch.float64)[mask]
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
    assert text.count('llvm.call_intrinsic "llvm.hivm.vscatter.') == (2 if w == 8 else 1)
    assert "ave.hir.vscatter" not in text
    assert "llvm.inttoptr" in text
    lowered = str(lower_offline(module, ctx))
    assert "func.func private @tla_native_simd" in lowered
    assert "#hivm.memory_effect<read_write>" in lowered
    assert "call @tla_native_simd" in lowered


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("readback", (False, True))
def test_compile(case, readback):
    (_, sig, w), _ = case
    kernel = compile_native_kernel(
        scatter_kernel,
        dict(X=f"*{sig}", I="*u32" if w == 32 else "*u16", M=f"*i{w}", INIT=f"*{sig}", O=f"*{sig}", CHECK=f"*{sig}",
             OFFSET="i32"), constants(case, True, readback))
    assert "#hivm.memory_effect<read_write>" in kernel.asm["ttadapter"]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("pattern", ("unmasked", "all", "none", "tail", "mixed"))
@pytest.mark.parametrize("readback", (False, True))
def test_run(case, pattern, readback):
    actual, expected, check, check_ref, _ = launch(case, pattern, readback)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(check, check_ref, rtol=0, atol=0)


@tla.jit
def two_scopes_kernel(O):
    out = tla.alloc_buffer((64,), tl.int32, space="ub")
    with tla.scope.simd():
        idx = tla.simd.vsplat(0, tl.uint32)
        data = tla.simd.vsplat(7, tl.int32)
        tla.simd.vscatter(out[0], idx, data, tla.simd.pge(1, 64))
    with tla.scope.simd():
        idx = tla.simd.vsplat(1, tl.uint32)
        data = tla.simd.vsplat(9, tl.int32)
        tla.simd.vscatter(out[0], idx, data, tla.simd.pge(1, 64))
    tla.copy(out, tla.view(O, (64,)))


def test_ir_multiple_scopes():
    module, ctx = compile_offline(two_scopes_kernel, dict(O="*i32"), {})
    text = str(lower_offline(module, ctx))
    assert text.count("func.func private @tla_native_simd") == 2
    assert text.count("#hivm.memory_effect<read_write>") == 2
    assert module.verify()
