"""TLA 3-in-1 test: vmula/vmadd (acc + src0 * src1) masks and modes on f32.

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).
Rows: 0 full-mask, 1 prefix ZEROING, 2 all-false ZEROING, 3 prefix MERGING
(MERGING keeps acc on the inactive lanes, CANN's dst-preserving predicate).
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
SIGNATURE = {"X": "*fp32", "Y": "*fp32", "O": "*fp32"}
SCALAR = 2.0


@tla.jit
def vmula(X, Y, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.float32, space="ub")
    y = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o = tla.alloc_buffer((4, N), tl.float32, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(Y, (N, )), y)
    with tla.scope.simd():
        acc = tla.simd.vlds(x[0])
        a = tla.simd.vlds(y[0])
        b = tla.simd.vsplat(2.0, tl.float32)
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(o[0, 0], tla.simd.vmula(acc, a, b))
        tla.simd.vsts(o[1, 0], tla.simd.vmula(acc, a, b, prefix))
        tla.simd.vsts(o[2, 0], tla.simd.vmula(acc, a, b, none))
        tla.simd.vsts(o[3, 0], tla.simd.vmula(acc, a, b, mask=prefix, mode=tla.simd.MODE.MERGING))
    tla.copy(o, tla.view(O, (4, N)))


@tla.jit
def vmadd(X, Y, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.float32, space="ub")
    y = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o = tla.alloc_buffer((4, N), tl.float32, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(Y, (N, )), y)
    with tla.scope.simd():
        acc = tla.simd.vlds(x[0])
        a = tla.simd.vlds(y[0])
        b = tla.simd.vsplat(2.0, tl.float32)
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(o[0, 0], tla.simd.vmadd(acc, a, b))
        tla.simd.vsts(o[1, 0], tla.simd.vmadd(acc, a, b, prefix))
        tla.simd.vsts(o[2, 0], tla.simd.vmadd(acc, a, b, none))
        tla.simd.vsts(o[3, 0], tla.simd.vmadd(acc, a, b, mask=prefix, mode=tla.simd.MODE.MERGING))
    tla.copy(o, tla.view(O, (4, N)))


def constexprs():
    return {"N": SHAPE["lanes"], "ACTIVE": SHAPE["active"]}


def _reference(x, y):
    full = x + y * SCALAR
    ref = torch.zeros((4, SHAPE["lanes"]), dtype=torch.float32)
    ref[0] = full
    ref[1, :SHAPE["active"]] = full[:SHAPE["active"]]
    ref[3, :SHAPE["active"]] = full[:SHAPE["active"]]
    ref[3, SHAPE["active"]:] = x[SHAPE["active"]:]
    return ref


def test_ir(output_dir=None):
    module, ctx = compile_offline(vmula, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.vmula") == 4
    assert text.count("ave.hir.vsel") == 1, "only the MERGING row adds vsel"
    assert "ave.hir.pge <VL32>" in text
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=2, stores=1, buffers=None, extra=("ave.hir.vmula", ))

    module, ctx = compile_offline(vmadd, SIGNATURE, constexprs())
    text = str(module)
    assert text.count("ave.hir.vmula") == 4, "vmadd lowers to the same vmula meta-op"
    assert text.count("ave.hir.vsel") == 1
    print("IR PASS: vmula/vmadd full/ZEROING/all-false/MERGING")


def _launch():
    s = SHAPE
    x = torch.randn(s["lanes"], dtype=torch.float32)
    y = torch.randn(s["lanes"], dtype=torch.float32)
    out = torch.empty((4, s["lanes"]), dtype=torch.float32, device="npu")
    vmula[(1, )](x.to("npu"), y.to("npu"), out, **constexprs())
    torch.npu.synchronize()
    ref = _reference(x, y)
    out2 = torch.empty((4, s["lanes"]), dtype=torch.float32, device="npu")
    vmadd[(1, )](x.to("npu"), y.to("npu"), out2, **constexprs())
    torch.npu.synchronize()
    return out, out2, ref


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vmula/vmadd generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    out, out2, ref = _launch()
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out2.cpu(), ref, rtol=1e-6, atol=1e-6)
    print("VMULA/VMADD OK: full/ZEROING/all-false/MERGING")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vmula_vmadd", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
