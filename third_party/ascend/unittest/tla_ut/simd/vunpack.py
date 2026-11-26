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
"""Integer widening: selectors, signedness and result-width predicates."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, get_stages, check_kernel_binary


@tla.jit
def unpack_kernel(X, M, O, SB: tl.constexpr, DB: tl.constexpr, SIGNED: tl.constexpr, PART: tl.constexpr,
                  PP: tl.constexpr, MASKED: tl.constexpr = True):
    S: tl.constexpr = 2048 // SB
    D: tl.constexpr = 2048 // DB
    ST: tl.constexpr = tl.int8 if SB == 8 else tl.int16
    DT: tl.constexpr = tl.int16 if DB == 16 else tl.int32
    x = tla.alloc_buffer((S, ), ST, space="ub")
    m = tla.alloc_buffer((D, ), DT, space="ub")
    out = tla.alloc_buffer((3, D), DT, space="ub")
    tla.copy(tla.view(X, (S, )), x)
    tla.copy(tla.view(M, (D, )), m)
    with tla.scope.simd():
        src = tla.simd.vlds(x[0])
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        if not MASKED:
            mask = None
        a = tla.simd.vunpack(src, DT, mask, part=PART, pp=PP, signed=SIGNED)
        b = tla.simd.vunpack(src, DT, mask, part=PART, pp=PP, signed=SIGNED, mode=tla.simd.MODE.MERGING)
        old = tla.simd.vdup(37, DT)
        if MASKED:
            c = tla.simd.vsel(mask, a, old)
        else:
            c = a
        tla.simd.vsts(out[0, 0], a, dist="NORM_B16" if DB == 16 else "NORM_B32")
        tla.simd.vsts(out[1, 0], b, dist="NORM_B16" if DB == 16 else "NORM_B32")
        tla.simd.vsts(out[2, 0], c, dist="NORM_B16" if DB == 16 else "NORM_B32")
    tla.copy(out, tla.view(O, (3, D)))


CASES = [(sb, db, signed, selector)
         for sb, db in ((8, 16), (16, 32), (8, 32))
         for signed in (False, True)
         for selector in range(db // sb)]


def constants(case):
    sb, db, signed, selector = case
    return dict(SB=sb, DB=db, SIGNED=signed, PART="part_odd" if selector == 1 and db // sb == 2 else "part_even",
                PP=f"pp{selector}" if db // sb == 4 else "pp0")


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("masked", (False, True))
def test_ir(case, masked):
    sb, db, signed, selector = case
    module, _ = compile_offline(unpack_kernel, dict(X=f"*i{sb}", M=f"*i{db}", O=f"*i{db}"),
                                dict(**constants(case), MASKED=masked))
    text = str(module)
    assert text.count("ave.hir.vextsi" if signed else "ave.hir.vextui") == 2
    assert text.count("ave.hir.vsel") == (3 if masked else 0)
    assert (f"pp{selector}" if db // sb == 4 else constants(case)["PART"]) in text
    assert "llvm." not in text
    assert f"vector<{2048 // db}xi{db}>" in text


def launch(case, pattern, masked=True):
    sb, db, signed, selector = case
    n, d = 2048 // sb, 2048 // db
    st, dt = (torch.int8 if sb == 8 else torch.int16), (torch.int16 if db == 16 else torch.int32)
    x = ((torch.arange(n, dtype=torch.int64) * 151 + 17) % (1 << sb) - (1 << (sb - 1))).to(st)
    lane = torch.arange(d)
    masks = {"all": lane >= 0, "none": lane < 0, "tail": lane < d - 3, "mixed": ((lane * 7 + 3) % 11) < 5}
    mask = masks[pattern]
    out = torch.empty((3, d), dtype=dt, device="npu")
    compiled = unpack_kernel[(1, )](x.to("npu"), mask.to(dt).to("npu"), out, **constants(case), MASKED=masked)
    torch.npu.synchronize()
    widened = x.to(torch.int64)
    if not signed:
        widened &= (1 << sb) - 1
    expected = widened[selector::db // sb]
    if not masked:
        mask = torch.ones_like(mask)
    ref = torch.stack((torch.where(mask, expected, 0), torch.where(mask, expected, 0), torch.where(mask, expected,
                                                                                                   37))).to(dt)
    return out.cpu(), ref, compiled


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
    print("ALL PASS: vunpack")
