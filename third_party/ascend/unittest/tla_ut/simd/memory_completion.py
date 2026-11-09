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
"""Reduction, dual loads, block gather and scatter against independent references."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, check_kernel_binary


@tla.jit
def memory_completion(X, I, M, D, O, DT: tl.constexpr, W: tl.constexpr, OP: tl.constexpr, BASE: tl.constexpr):
    N: tl.constexpr = 2048 // W
    x = tla.alloc_buffer((1024, ), DT, space="ub")
    i = tla.alloc_buffer((64, ), tl.uint32, space="ub")
    m = tla.alloc_buffer((N, ), tl.int16 if W == 16 else tl.int32, space="ub")
    dst = tla.alloc_buffer((1024, ), DT, space="ub")
    out = tla.alloc_buffer((2, N), DT, space="ub")
    tla.copy(tla.view(X, (1024, )), x)
    tla.copy(tla.view(I, (64, )), i)
    tla.copy(tla.view(M, (N, )), m)
    tla.copy(tla.view(D, (1024, )), dst)
    with tla.scope.simd():
        a = tla.simd.vlds(x[BASE])
        index = tla.simd.vlds(i[0])
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        if OP == "vcadd":
            r0 = tla.simd.vcadd(a, mask)
            r1 = r0
        elif OP == "vld2":
            r0, r1 = tla.simd.vld2(x[BASE], dist="DINTLV_B16" if W == 16 else "DINTLV_B32", off=16)
        elif OP == "vgatherb":
            r0 = tla.simd.vgatherb(x[BASE], index, mask)
            r1 = r0
        else:
            r0 = tla.simd.vlds(dst[BASE])
            tla.simd.membar("VLD_VST")
            tla.simd.vscatter(a, dst[BASE], index, mask)
            tla.simd.membar("VST_VLD")
            r1 = tla.simd.vlds(dst[BASE])
        tla.simd.vsts(out[0, 0], r0, dist="NORM_B16" if W == 16 else "NORM_B32")
        tla.simd.vsts(out[1, 0], r1, dist="NORM_B16" if W == 16 else "NORM_B32")
    tla.copy(out, tla.view(O, (2, N)))


CASES = [(op, name, sig, width)
         for op in ("vcadd", "vld2", "vgatherb", "vscatter")
         for name, sig, width in (("int32", "i32", 32), ("float32", "fp32", 32))]
CASES += [("vld2", "int16", "i16", 16), ("vld2", "float16", "fp16", 16)]
CASES += [(op, "uint32", "u32", 32) for op in ("vgatherb", "vscatter")]


def constants(case, base):
    op, name, _, width = case
    return dict(DT=getattr(tl, name), W=width, OP=op, BASE=base)


@pytest.mark.parametrize("case", CASES)
def test_ir(case):
    op, _, sig, width = case
    module, _ = compile_offline(memory_completion, dict(X=f"*{sig}", I="*u32", M=f"*i{width}", D=f"*{sig}",
                                                        O=f"*{sig}"), constants(case, 32))
    text = str(module)
    expected = {
        "vcadd": "ave.hir.reduction", "vld2": "ave.hir.vdintlv", "vgatherb": "ave.hir.vgatherb", "vscatter":
        "ave.hir.vscatter"
    }
    assert expected[op] in text
    assert "llvm." not in text


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("pattern", ("all", "none", "tail", "mixed"))
@pytest.mark.parametrize("base", (0, 64))
def test_run(case, pattern, base):
    op, name, _, width = case
    n = 2048 // width
    dtype = getattr(torch, name)
    raw = (torch.arange(1024) * 7 + 5) % 31
    if name != "uint32":
        raw -= 15
    data = raw.to(dtype)
    lane = torch.arange(n)
    masks = {"all": lane >= 0, "none": lane < 0, "tail": lane < n - 7, "mixed": lane % 3 == 1}
    mask = masks[pattern]
    index = (torch.arange(64) * 17 + 3) % 64
    if op == "vgatherb":
        # Blocks are deliberately nonadjacent, repeated, and not sorted.
        index[:8] = torch.tensor([384, 0, 128, 128, 64, 512, 32, 256])
    initial = torch.full((1024, ), 23, dtype=dtype)
    out = torch.empty((2, n), dtype=dtype, device="npu")
    kernel = memory_completion[(1, )](data.to("npu"), index.to(torch.uint32).to("npu"),
                                      mask.to(torch.int16 if width == 16 else torch.int32).to("npu"), initial.to("npu"),
                                      out, **constants(case, base))
    torch.npu.synchronize()
    check_kernel_binary(kernel)
    result = out.cpu().to(torch.float64)
    if op == "vcadd":
        expected = data[base:base + n].to(torch.float64)[mask].sum()
        torch.testing.assert_close(result[:, 0], expected.expand(2), rtol=0, atol=0)
        return
    if op == "vld2":
        values = data[base + 16:base + 16 + 2 * n].to(torch.float64)
        expected = torch.stack((values[::2], values[1::2]))
    elif op == "vgatherb":
        values = torch.cat([data[base + int(off) // 4:base + int(off) // 4 + 8] for off in index[:8]]).to(torch.float64)
        values = torch.where(mask, values, 0)
        expected = torch.stack((values, values))
    else:
        updated = initial[base:base + n].to(torch.float64).clone()
        updated[index[mask]] = data[base:base + n].to(torch.float64)[mask]
        expected = torch.stack((initial[base:base + n].to(torch.float64), updated))
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@tla.jit
def scatter_alias(INIT, I, O, CHECK, OFFSET, STEPS, ALIAS: tl.constexpr):
    buf = tla.alloc_buffer((256, ), tl.int32, space="ub")
    indices = tla.alloc_buffer((64, ), tl.uint32, space="ub")
    check = tla.alloc_buffer((2, 64), tl.int32, space="ub")
    tla.copy(tla.view(INIT, (256, )), buf)
    tla.copy(tla.view(I, (64, )), indices)
    if ALIAS == "captured":
        target = tla.subview(buf, offsets=(OFFSET, ), sizes=(128, ))
    with tla.scope.simd():
        if ALIAS == "inner":
            target = tla.subview(buf, offsets=(OFFSET, ), sizes=(128, ))
        idx = tla.simd.vlds(indices[0])
        before = tla.simd.vlds(buf[64])
        for step in range(STEPS):
            value = tla.simd.vadds(tla.simd.vlds(buf[64]), 1)
            tla.simd.membar("VLD_VST")
            if ALIAS == "root":
                tla.simd.vscatter(value, buf[64], idx)
            else:
                tla.simd.vscatter(value, target[0], idx)
            tla.simd.membar("VST_VLD")
        after = tla.simd.vlds(buf[64])
        tla.simd.vsts(check[0, 0], before)
        tla.simd.vsts(check[1, 0], after)
    tla.copy(buf, tla.view(O, (256, )))
    tla.copy(check, tla.view(CHECK, (2, 64)))


@pytest.mark.parametrize("alias", ("root", "inner", "captured"))
@pytest.mark.parametrize("steps", (1, 3))
def test_scatter_alias(alias, steps):
    data = torch.arange(256, dtype=torch.int32)
    idx = torch.arange(64, dtype=torch.int64).to(torch.uint32)
    out = torch.empty_like(data, device="npu")
    check = torch.empty((2, 64), dtype=torch.int32, device="npu")
    scatter_alias[(1, )](data.npu(), idx.npu(), out, check, 64, steps, ALIAS=alias)
    torch.npu.synchronize()
    expected = data.clone()
    expected[64:128] += steps
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(check.cpu(), torch.stack((data[64:128], expected[64:128])), rtol=0, atol=0)
