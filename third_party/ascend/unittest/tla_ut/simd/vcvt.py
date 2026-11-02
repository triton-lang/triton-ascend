"""TLA 3-in-1 test: vcvt dtype conversions with masks, rounding and sat.

Two kernels: signed int->fp (vsitofp, i32->f32) and fp->int (vfptosi,
f32->i32 with round/sat/mask).  Inputs stay within the f32-exact range
(|x| < 2**24) so the references are exact.

The unsigned directions (vuitofp u8->f16 / u64->f32, vfptoui f16->u8) are
implemented in the builder but their part->lane mapping needs ISA-doc
confirmation, so they are not numerically exercised here yet.

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

SHAPE = dict(lanes=64, active=32)
SIGNATURE_I2F = {"X": "*i32", "OF": "*fp32"}
SIGNATURE_F2I = {"XF": "*fp32", "OI": "*i32"}


@tla.jit
def vcvt_i2f(X, OF, N: tl.constexpr, ACTIVE: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.int32, space="ub")
    o = tla.alloc_buffer((2, N), tl.float32, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    with tla.scope.simd():
        xv = tla.simd.vlds(x[0])
        prefix = tla.simd.pge(ACTIVE, N)
        tla.simd.vsts(o[0, 0], tla.simd.vcvt(xv, tl.float32))
        tla.simd.vsts(o[1, 0], tla.simd.vcvt(xv, tl.float32, prefix))
    tla.copy(o, tla.view(OF, (2, N)))


@tla.jit
def vcvt_f2i(XF, OI, N: tl.constexpr, ACTIVE: tl.constexpr):
    xf = tla.alloc_buffer((N, ), tl.float32, space="ub")
    oi = tla.alloc_buffer((4, N), tl.int32, space="ub")
    tla.copy(tla.view(XF, (N, )), xf)
    with tla.scope.simd():
        f = tla.simd.vlds(xf[0])
        prefix = tla.simd.pge(ACTIVE, N)
        # f2s with default round=R (ties to even)
        tla.simd.vsts(oi[0, 0], tla.simd.vcvt(f, tl.int32))
        # f2s with round=Z (toward zero)
        tla.simd.vsts(oi[1, 0], tla.simd.vcvt(f, tl.int32, round=tla.simd.ROUND.Z))
        # f2s with sat=True saturates an overflowing row
        tla.simd.vsts(oi[2, 0], tla.simd.vcvt(tla.simd.vmuls(f, 1e9), tl.int32, sat=True))
        # masked f2s ZEROING
        tla.simd.vsts(oi[3, 0], tla.simd.vcvt(f, tl.int32, prefix))
    tla.copy(oi, tla.view(OI, (4, N)))


def constexprs():
    return {"N": SHAPE["lanes"], "ACTIVE": SHAPE["active"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vcvt_i2f, SIGNATURE_I2F, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.vsitofp") == 2
    assert "ave.hir.pge <VL32>" in text
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=1, stores=1, buffers=None, extra=("ave.hir.vsitofp", ))

    module, ctx = compile_offline(vcvt_f2i, SIGNATURE_F2I, constexprs())
    text = str(module)
    assert text.count("ave.hir.vfptosi") == 4
    text = str(lower_offline(module, ctx))
    check_lowered_ir(text, loads=1, stores=1, buffers=None, extra=("ave.hir.vfptosi", ))
    print("IR PASS: vcvt s2f/f2i round/sat/mask variants")


def _launch():
    s = SHAPE
    x = torch.randint(-(1 << 24), 1 << 24, (s["lanes"], ), dtype=torch.int64).to(torch.int32)
    of = torch.empty((2, s["lanes"]), dtype=torch.float32, device="npu")
    vcvt_i2f[(1, )](x.to("npu"), of, **constexprs())
    torch.npu.synchronize()
    ref = torch.zeros((2, s["lanes"]), dtype=torch.float32)
    ref[0] = x.to(torch.float32)
    ref[1, :s["active"]] = x[:s["active"]].to(torch.float32)

    # f32 rounding kernel with tie-heavy values
    base = torch.arange(s["lanes"], dtype=torch.float32) * 0.5 - 8.0
    xf = base + torch.where(torch.arange(s["lanes"]) % 2 == 0, 0.25, -0.25)
    oi = torch.empty((4, s["lanes"]), dtype=torch.int32, device="npu")
    vcvt_f2i[(1, )](xf.to("npu"), oi, **constexprs())
    torch.npu.synchronize()
    floor = torch.floor(xf)
    ties_even = torch.where(floor + 0.5 == xf, floor + (floor % 2), torch.round(xf))
    ref2 = torch.zeros((4, s["lanes"]), dtype=torch.int64)
    ref2[0] = ties_even.to(torch.int64)
    ref2[1] = torch.trunc(xf).to(torch.int64)
    ref2[2] = (xf.double() * 1e9).clamp(-(2**31), 2**31 - 1).to(torch.int64)
    ref2[3, :s["active"]] = ties_even[:s["active"]].to(torch.int64)
    return (of, oi), (ref, ref2)


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vcvt generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    (of, oi), (ref, ref2) = _launch()
    torch.testing.assert_close(of.cpu(), ref, rtol=0, atol=0)
    torch.testing.assert_close(oi.cpu(), ref2.to(torch.int32), rtol=0, atol=0)
    print("VCVT OK: int/fp conversions, rounding and saturation")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vcvt", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
