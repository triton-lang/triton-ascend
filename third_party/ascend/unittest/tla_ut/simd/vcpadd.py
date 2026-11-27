"""Tests for AVE pairwise adjacent-lane sum."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (compile_offline, get_stages, lower_offline, check_compile_artifacts)


@tla.jit
def vcpadd_kernel(X, M, O, D: tl.constexpr, I: tl.constexpr, N: tl.constexpr, DIST: tl.constexpr, MODE: tl.constexpr):
    x = tla.alloc((N, ), D, space="ub")
    m = tla.alloc((N, ), I, space="ub")
    o = tla.alloc((N, ), D, space="ub")
    tla.load_gm_to_ub(tla.view(X, (N, )), x)
    tla.load_gm_to_ub(tla.view(M, (N, )), m)
    with tla.scope.simd():
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        tla.simd.vsts(o[0], tla.simd.vcpadd(tla.simd.vlds(x[0]), mask, mode=MODE), dist=DIST)
    tla.store_ub_to_gm(o, tla.view(O, (N, )))


CASES = [
    ("f16", torch.float16, torch.int16, tl.float16, tl.int16, "fp16", "i16", 128, "NORM_B16"),
    ("f32", torch.float32, torch.int32, tl.float32, tl.int32, "fp32", "i32", 64, "NORM_B32"),
]


def test_ir():
    for _, _, _, dt, mt, short, mshort, n, dist in CASES:
        mod, ctx = compile_offline(vcpadd_kernel, {"X": "*" + short, "M": "*" + mshort, "O": "*" + short},
                                   {"D": dt, "I": mt, "N": n, "DIST": dist, "MODE": tla.simd.MODE.ZEROING})
        text = str(lower_offline(mod, ctx))
        assert "ave.hir.vcpadd" in text
        assert "ave.hir.vsel" not in text
        with pytest.raises(Exception) as err:
            compile_offline(vcpadd_kernel, {"X": "*" + short, "M": "*" + mshort, "O": "*" + short},
                            {"D": dt, "I": mt, "N": n, "DIST": dist, "MODE": tla.simd.MODE.MERGING})
        assert "MODE.MERGING" in str(err.value)
    print("IR PASS: vcpadd")


def _launch():
    _, td, md, dt, mt, _, _, n, dist = CASES[0]
    base = torch.arange(n, dtype=td)
    mask = torch.ones(n, dtype=torch.bool)
    out = torch.empty(n, dtype=td, device="npu")
    vcpadd_kernel[(1, )](base.npu(), mask.to(md).npu(), out, D=dt, I=mt, N=n, DIST=dist, MODE=tla.simd.MODE.ZEROING)
    torch.npu.synchronize()
    return out


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vcpadd generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    for name, td, md, dt, mt, _, _, n, dist in CASES:
        base = torch.arange(n, dtype=td)
        for mask_name, mask in [("zero", torch.zeros(n, dtype=torch.bool)), ("one", torch.ones(n, dtype=torch.bool)),
                                ("alternating", torch.arange(n) % 2 == 0), ("sparse", torch.arange(n) % 3 != 0)]:
            out = torch.empty(n, dtype=td, device="npu")
            vcpadd_kernel[(1, )](base.npu(), mask.to(md).npu(), out, D=dt, I=mt, N=n, DIST=dist,
                                 MODE=tla.simd.MODE.ZEROING)
            torch.npu.synchronize()
            gated = torch.where(mask, base, torch.zeros_like(base))
            ref = torch.zeros(n, dtype=td)
            ref[:n // 2] = gated[0::2] + gated[1::2]
            torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
            print("RUN PASS: vcpadd", name, n, mask_name, "zero")
    print("RUN PASS: vcpadd all")


if __name__ == "__main__":
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
