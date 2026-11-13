"""Tests for AVE exclusive prefix mask count."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (compile_offline, get_stages, lower_offline, check_compile_artifacts)


@tla.jit
def vusqz_kernel(M, O, D: tl.constexpr, N: tl.constexpr, DIST: tl.constexpr):
    m = tla.alloc_buffer((N, ), D, space="ub")
    o = tla.alloc_buffer((N, ), D, space="ub")
    tla.copy(tla.view(M, (N, )), m)
    with tla.scope.simd():
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        tla.simd.vsts(o[0], tla.simd.vusqz(mask, D), dist=DIST)
    tla.copy(o, tla.view(O, (N, )))


CASES = [
    (torch.uint8, tl.uint8, "u8", 256, "NORM_B8"),
    (torch.uint16, tl.uint16, "u16", 128, "NORM_B16"),
    (torch.uint32, tl.uint32, "u32", 64, "NORM_B32"),
]


def test_ir():
    for _, dtype, short, n, dist in CASES:
        mod, ctx = compile_offline(vusqz_kernel, {"M": "*" + short, "O": "*" + short},
                                   {"D": dtype, "N": n, "DIST": dist})
        text = str(lower_offline(mod, ctx))
        assert "ave.hir.vusqz" in text
        assert "ave.hir.vslide" not in text
    print("IR PASS: vusqz")


def _launch():
    td, dtype, _, n, dist = CASES[0]
    mask = torch.ones(n, dtype=td)
    out = torch.empty(n, dtype=td, device="npu")
    vusqz_kernel[(1, )](mask.npu(), out, D=dtype, N=n, DIST=dist)
    torch.npu.synchronize()
    return out


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vusqz generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    for td, dtype, _, n, dist in CASES:
        for name, mask in [("zero", torch.zeros(n)), ("one", torch.ones(n)), ("alternating", torch.arange(n) % 2 == 0),
                           ("sparse", torch.arange(n) % 3 != 0)]:
            out = torch.empty(n, dtype=td, device="npu")
            vusqz_kernel[(1, )](mask.to(td).npu(), out, D=dtype, N=n, DIST=dist)
            torch.npu.synchronize()
            ref = mask.to(torch.int64).cumsum(0) - mask.to(torch.int64)
            torch.testing.assert_close(out.cpu().to(torch.int64), ref, rtol=0, atol=0)
            print("RUN PASS: vusqz", td, n, name)
    print("RUN PASS: vusqz all")


if __name__ == "__main__":
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
