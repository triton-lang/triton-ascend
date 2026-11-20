"""TLA 3-in-1 test: vcmps (vector-scalar compare) exact bits on u32/i32.

vcmps broadcasts the scalar then runs vcmp, so the IR asserts one broadcast
plus one vcmp per row.

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).
Rows: 0 u32 ULT (unsigned below), 1 i32 signed LT, 2 u32 prefix-masked.
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
SIGNATURE = {"X": "*u32", "Y": "*i32", "O": "*u32"}


@tla.jit
def vcmps(X, Y, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.uint32, space="ub")
    y = tla.alloc_buffer((N, ), tl.int32, space="ub")
    o = tla.alloc_buffer((3, N), tl.uint32, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(Y, (N, )), y)
    with tla.scope.simd():
        xv = tla.simd.vlds(x[0])
        yv = tla.simd.vlds(y[0])
        prefix = tla.simd.pge(ACTIVE, N)
        ones = tla.simd.vsplat(1, tl.uint32)
        zeros = tla.simd.vsplat(0, tl.uint32)
        tla.simd.vsts(o[0, 0], tla.simd.vsel(tla.simd.vcmps(xv, 0x80000000, op=tla.simd.CMP.ULT), ones, zeros))
        tla.simd.vsts(o[1, 0], tla.simd.vsel(tla.simd.vcmps(yv, -1, op=tla.simd.CMP.LT), ones, zeros))
        tla.simd.vsts(o[2, 0],
                      tla.simd.vsel(tla.simd.vcmps(xv, 0x80000000, mask=prefix, op=tla.simd.CMP.ULT), ones, zeros))
    tla.copy(o, tla.view(O, (3, N)))


def constexprs():
    return {"N": SHAPE["lanes"], "ACTIVE": SHAPE["active"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vcmps, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.vcmp") == 3
    assert text.count("ave.hir.broadcast") == 3, "vcmps broadcasts the scalar per row"
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=2, stores=1, buffers=None, extra=("ave.hir.vcmp", ))
    print("IR PASS: vcmps broadcast + vcmp composition")


def _launch():
    s = SHAPE
    x = torch.randint(0, 1 << 32, (s["lanes"], ), dtype=torch.int64)
    y = torch.randint(-(1 << 31), 1 << 31, (s["lanes"], ), dtype=torch.int64)
    out = torch.empty((3, s["lanes"]), dtype=torch.uint32, device="npu")
    vcmps[(1, )](x.to(torch.uint32).to("npu"), y.to(torch.int32).to("npu"), out, **constexprs())
    torch.npu.synchronize()
    r0 = (x < 0x80000000).to(torch.int64)
    r1 = (y < -1).to(torch.int64)
    r2 = torch.zeros(s["lanes"], dtype=torch.int64)
    r2[:s["active"]] = (x[:s["active"]] < 0x80000000).to(torch.int64)
    ref = torch.stack([r0, r1, r2])
    return out, ref.to(torch.uint32)


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vcmps generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    out, ref = _launch()
    torch.testing.assert_close(out.cpu().view(torch.int32), ref.view(torch.int32), rtol=0, atol=0)
    print("VMCPS OK: vector-scalar compare exact bits")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vcmps", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
