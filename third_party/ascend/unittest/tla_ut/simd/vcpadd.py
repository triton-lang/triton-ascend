"""Tests for AVE pairwise adjacent-lane sum."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import compile_offline, get_stages, lower_offline


@tla.jit
def vcpadd_kernel(X, M, O, D: tl.constexpr, I: tl.constexpr, N: tl.constexpr, DIST: tl.constexpr, MODE: tl.constexpr):
    x = tla.alloc_buffer((N, ), D, space="ub")
    m = tla.alloc_buffer((N, ), I, space="ub")
    o = tla.alloc_buffer((N, ), D, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(M, (N, )), m)
    with tla.scope.simd():
        mask = tla.simd.vcmps(tla.simd.vlds(m[0]), 0, op=tla.simd.CMP.NE)
        tla.simd.vsts(o[0], tla.simd.vcpadd(tla.simd.vlds(x[0]), mask, mode=MODE), dist=DIST)
    tla.copy(o, tla.view(O, (N, )))


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
        mod, ctx = compile_offline(vcpadd_kernel, {"X": "*" + short, "M": "*" + mshort, "O": "*" + short},
                                   {"D": dt, "I": mt, "N": n, "DIST": dist, "MODE": tla.simd.MODE.MERGING})
        assert "ave.hir.vsel" in str(lower_offline(mod, ctx))
    print("IR PASS: vcpadd")


def test_run():
    for name, td, md, dt, mt, _, _, n, dist in CASES:
        base = torch.arange(n, dtype=td)
        for mask_name, mask in [("zero", torch.zeros(n, dtype=torch.bool)), ("one", torch.ones(n, dtype=torch.bool)),
                                ("alternating", torch.arange(n) % 2 == 0), ("sparse", torch.arange(n) % 3 != 0)]:
            out = torch.empty(n, dtype=td, device="npu")
            for mode_name, mode in [("zero", tla.simd.MODE.ZEROING), ("merge", tla.simd.MODE.MERGING)]:
                out = torch.empty(n, dtype=td, device="npu")
                vcpadd_kernel[(1, )](base.npu(), mask.to(md).npu(), out, D=dt, I=mt, N=n, DIST=dist, MODE=mode)
                torch.npu.synchronize()
                gated = torch.where(mask, base, torch.zeros_like(base))
                zero_ref = torch.zeros(n, dtype=td)
                zero_ref[:n // 2] = gated[0::2] + gated[1::2]
                ref = torch.where(mask, zero_ref, base) if mode is tla.simd.MODE.MERGING else zero_ref
                torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
                print("RUN PASS: vcpadd", name, n, mask_name, mode_name)
    print("RUN PASS: vcpadd all")


if __name__ == "__main__":
    do_ir, _, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_run:
        test_run()
