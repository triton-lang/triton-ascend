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
"""Predicate stores: packed bit order and untouched surrounding bytes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, get_stages, check_kernel_binary


@tla.jit
def store_predicate(X, INIT, O, W: tl.constexpr, OFFSET: tl.constexpr):
    D: tl.constexpr = 2048 // W
    DT: tl.constexpr = tl.int8 if W == 8 else (tl.int16 if W == 16 else tl.int32)
    x = tla.alloc_buffer((D, ), DT, space="ub")
    o = tla.alloc_buffer((128, ), tl.uint8, space="ub")
    tla.copy(tla.view(X, (D, )), x)
    tla.copy(tla.view(INIT, (128, )), o)
    with tla.scope.simd():
        p = tla.simd.vcmps(tla.simd.vlds(x[0]), 0, op=tla.simd.CMP.NE)
        tla.simd.pst(o[OFFSET], p)
    tla.copy(o, tla.view(O, (128, )))


CASES = [(w, off) for w in (8, 16, 32) for off in (0, 64)]


@pytest.mark.parametrize("case", CASES)
def test_ir(case):
    w, offset = case
    module, _ = compile_offline(store_predicate, dict(X=f"*i{w}", INIT="*u8", O="*u8"), dict(W=w, OFFSET=offset))
    text = str(module)
    assert "llvm." not in text
    assert module.verify()
    assert "ave.hir.masked_store <NORM_B8>" in text
    assert f"#ave.func_dist_type<pb{w}>" in text
    assert "memref.view" in text


def launch(case, pattern):
    w, offset = case
    lane = torch.arange(2048 // w)
    bits = ((lane * 7 + 3) % 11 < 5).to(torch.int64)
    if pattern == "zero":
        bits.zero_()
    if pattern == "ones":
        bits.fill_(1)
    dtype = {8: torch.int8, 16: torch.int16, 32: torch.int32}[w]
    init = torch.full((128, ), 0xA5, dtype=torch.uint8)
    out = torch.empty_like(init, device="npu")
    compiled = store_predicate[(1, )](bits.to(dtype).to("npu"), init.to("npu"), out, w, offset)
    torch.npu.synchronize()
    packed = (bits.reshape(-1, 8) << torch.arange(8)).sum(1).to(torch.uint8)
    expected = init.clone()
    expected[offset:offset + len(packed)] = packed
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
    print("ALL PASS: pst")
