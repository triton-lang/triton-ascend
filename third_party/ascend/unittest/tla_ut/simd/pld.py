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
"""Packed predicate loads: LSB-first NORM, repeated US and decimated DS."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, get_stages, check_kernel_binary


@tla.jit
def load_predicate(X, O, W: tl.constexpr, DIST: tl.constexpr, OFFSET: tl.constexpr):
    D: tl.constexpr = 2048 // W
    DT: tl.constexpr = tl.int8 if W == 8 else (tl.int16 if W == 16 else tl.int32)
    x = tla.alloc_buffer((256, ), tl.uint8, space="ub")
    o = tla.alloc_buffer((D, ), DT, space="ub")
    tla.copy(tla.view(X, (256, )), x)
    with tla.scope.simd():
        p = tla.simd.pld(x[OFFSET], width=W, dist=DIST)
        v = tla.simd.vsel(p, tla.simd.vdup(1, DT), tla.simd.vdup(0, DT))
        tla.simd.vsts(o[0], v, dist="NORM_B8" if W == 8 else ("NORM_B16" if W == 16 else "NORM_B32"))
    tla.copy(o, tla.view(O, (D, )))


CASES = [(w, dist, off) for w in (8, 16, 32) for dist in ("NORM", "US", "DS") for off in (0, 64)]


@pytest.mark.parametrize("case", CASES)
def test_ir(case):
    w, dist, offset = case
    module, _ = compile_offline(load_predicate, dict(X="*u8", O=f"*i{w}"), dict(W=w, DIST=dist, OFFSET=offset))
    text = str(module)
    assert "llvm." not in text
    assert module.verify()
    assert f"ave.hir.vload <{dist}>" in text
    assert f"#ave.func_dist_type<pb{w}>" in text
    assert f"vector<{2048 // w}xi1>" in text
    assert "memref.view" in text


def launch(case, pattern):
    w, dist, offset = case
    raw = ((torch.arange(256, dtype=torch.int64) * 73 + 41) % 256)
    if pattern == "zero":
        raw.zero_()
    if pattern == "ones":
        raw.fill_(255)
    dtype = {8: torch.int8, 16: torch.int16, 32: torch.int32}[w]
    out = torch.empty(2048 // w, dtype=dtype, device="npu")
    compiled = load_predicate[(1, )](raw.to(torch.uint8).to("npu"), out, w, dist, offset)
    torch.npu.synchronize()
    # Hardware probe: NORM uses bit i, US uses i//2, DS uses 2*i.
    bit = torch.arange(2048 // w)
    if dist == "US":
        bit //= 2
    if dist == "DS":
        bit *= 2
    expected = ((raw[offset + bit // 8] >> (bit % 8)) & 1).to(dtype)
    return out.cpu(), expected, compiled


@pytest.mark.parametrize("case", CASES)
def test_compile(case):
    _, _, compiled = launch(case, "mixed")
    check_kernel_binary(compiled)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("pattern", ("mixed", "zero", "ones"))
def test_run(case, pattern):
    actual, expected, _ = launch(case, pattern)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    ir, compile_, run = get_stages()
    for case in CASES:
        if ir:
            test_ir(case)
        if compile_:
            test_compile(case)
        if run:
            for pattern in ("mixed", "zero", "ones"):
                test_run(case, pattern)
    print("ALL PASS: pld")
