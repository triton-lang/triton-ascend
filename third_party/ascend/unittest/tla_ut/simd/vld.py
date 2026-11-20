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
"""Indexed UB loads: element offsets, full-width masks and two modes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, get_stages, check_kernel_binary


@tla.jit
def indexed_load(X, I, M, O, DT: tl.constexpr, W: tl.constexpr, BASE: tl.constexpr, MASKED: tl.constexpr = True):
    N: tl.constexpr = 2048 // W
    IT: tl.constexpr = tl.uint32 if W == 32 else tl.uint16
    MT: tl.constexpr = tl.int16 if W == 16 else tl.int32
    x = tla.alloc_buffer((1024, ), DT, space="ub")
    indices = tla.alloc_buffer((N, ), IT, space="ub")
    m = tla.alloc_buffer((N, ), MT, space="ub")
    o = tla.alloc_buffer((3, N), DT, space="ub")
    tla.copy(tla.view(X, (1024, )), x)
    tla.copy(tla.view(I, (N, )), indices)
    tla.copy(tla.view(M, (N, )), m)
    with tla.scope.simd():
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        if not MASKED:
            mask = None
        idx = tla.simd.vlds(indices[0])
        a = tla.simd.vgather(x[BASE], idx, mask)
        b = tla.simd.vld(x[BASE], idx, mask, mode=tla.simd.MODE.MERGING)
        if MASKED:
            c = tla.simd.vsel(mask, a, tla.simd.vsplat(7, DT))
        else:
            c = a
        tla.simd.vsts(o[0, 0], a, dist="NORM_B16" if W == 16 else "NORM_B32")
        tla.simd.vsts(o[1, 0], b, dist="NORM_B16" if W == 16 else "NORM_B32")
        tla.simd.vsts(o[2, 0], c, dist="NORM_B16" if W == 16 else "NORM_B32")
    tla.copy(o, tla.view(O, (3, N)))


TYPES = [("int16", "i16", 16), ("uint16", "u16", 16), ("int32", "i32", 32), ("uint32", "u32", 32),
         ("float16", "fp16", 16), ("bfloat16", "bf16", 16), ("float32", "fp32", 32)]
CASES = [(typ, base) for typ in TYPES for base in (0, 64)]


def constants(case):
    (name, _, w), base = case
    return dict(DT=getattr(tl, name), W=w, BASE=base)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("masked", (False, True))
def test_ir(case, masked):
    (_, signature, w), _ = case
    module, _ = compile_offline(indexed_load,
                                dict(X=f"*{signature}", I="*u32" if w == 32 else "*u16", M=f"*i{w}", O=f"*{signature}"),
                                dict(**constants(case), MASKED=masked))
    text = str(module)
    assert text.count("ave.hir.vgather") == 2
    assert "llvm." not in text
    assert text.count("ave.hir.vsel") == (1 if masked else 0)


def launch(case, pattern, masked=True):
    (name, _, w), base = case
    n = 2048 // w
    dtype = getattr(torch, name)
    raw = (torch.arange(1024, dtype=torch.int64) * 37 + 9) % 109 - 53
    if name.startswith("uint"):
        raw += 53
    data = raw.to(dtype)
    lane = torch.arange(n)
    masks = {"all": lane >= 0, "none": lane < 0, "tail": lane < n - 5, "mixed": (lane * 7 + 3) % 11 < 5}
    mask = masks[pattern] if masked else torch.ones(n, dtype=torch.bool)
    index = (lane * 17 + 13) % 800
    expected = raw[base + index].to(dtype).to(torch.float64)
    poisoned = torch.where(mask, index, 0xFFFF)
    out = torch.empty((3, n), dtype=dtype, device="npu")
    compiled = indexed_load[(1, )](data.to("npu"), poisoned.to(torch.uint32 if w == 32 else torch.uint16).to("npu"),
                                   mask.to({16: torch.int16, 32: torch.int32}[w]).to("npu"), out, **constants(case),
                                   MASKED=masked)
    torch.npu.synchronize()
    expected = torch.stack((torch.where(mask, expected, 0), torch.where(mask, expected,
                                                                        0), torch.where(mask, expected, 7))).to(dtype)
    return out.cpu().to(torch.float64), expected.to(torch.float64), compiled


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("masked", (False, True))
def test_compile(case, masked):
    _, _, compiled = launch(case, "all", masked=masked)
    check_kernel_binary(compiled)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("pattern", ("all", "none", "tail", "mixed"))
@pytest.mark.parametrize("masked", (False, True))
def test_run(case, pattern, masked):
    actual, expected, _ = launch(case, pattern, masked=masked)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    ir, compile_, run = get_stages()
    for case in CASES:
        for masked in (False, True):
            if ir:
                test_ir(case, masked)
            if compile_:
                test_compile(case, masked)
            if run:
                for pattern in ("all", "none", "tail", "mixed"):
                    test_run(case, pattern, masked)
    print("ALL PASS: vld/vgather")
