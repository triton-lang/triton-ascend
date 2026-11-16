"""TLA 3-in-1 test: pnot (predicate NOT) exact bits on i1 registers.

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).
Rows: 0 full-mask, 1 prefix-gated.
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
SIGNATURE = {"X": "*u32", "T1": "*u32", "O": "*u32"}


@tla.jit
def pnot(X, T1, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.uint32, space="ub")
    t1 = tla.alloc_buffer((N, ), tl.uint32, space="ub")
    o = tla.alloc_buffer((2, N), tl.uint32, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(T1, (N, )), t1)
    with tla.scope.simd():
        xv = tla.simd.vlds(x[0])
        a = tla.simd.vlds(t1[0])
        p1 = tla.simd.vcmp(xv, a, op=tla.simd.CMP.LT)
        prefix = tla.simd.pge(ACTIVE, N)
        ones = tla.simd.vsplat(1, tl.uint32)
        zeros = tla.simd.vsplat(0, tl.uint32)
        tla.simd.vsts(o[0, 0], tla.simd.vsel(tla.simd.pnot(p1), ones, zeros))
        tla.simd.vsts(o[1, 0], tla.simd.vsel(tla.simd.pnot(p1, prefix), ones, zeros))
    tla.copy(o, tla.view(O, (2, N)))


def constexprs():
    return {"N": SHAPE["lanes"], "ACTIVE": SHAPE["active"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(pnot, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.preg.not") == 2
    assert text.count("ave.hir.vsel") == 2, "one materializing vsel per row"
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=2, stores=1, buffers=None, extra=("ave.hir.preg.not", ))
    print("IR PASS: pnot full/prefix gating")


def _launch():
    s = SHAPE
    x = torch.randint(0, 1 << 32, (s["lanes"], ), dtype=torch.int64)
    t1 = torch.randint(0, 1 << 32, (s["lanes"], ), dtype=torch.int64)
    out = torch.empty((2, s["lanes"]), dtype=torch.uint32, device="npu")
    pnot[(1, )](x.to(torch.uint32).to("npu"), t1.to(torch.uint32).to("npu"), out, **constexprs())
    torch.npu.synchronize()
    p = ~(x < t1)
    ref = torch.zeros((2, s["lanes"]), dtype=torch.int64)
    ref[0] = p.to(torch.int64)
    ref[1, :s["active"]] = p[:s["active"]].to(torch.int64)
    return out, ref.to(torch.uint32)


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: pnot generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    out, ref = _launch()
    torch.testing.assert_close(out.cpu().view(torch.int32), ref.view(torch.int32), rtol=0, atol=0)
    print("PNOT OK: predicate NOT exact bits")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: pnot", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
