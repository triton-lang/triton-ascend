"""TLA 3-in-1 test: non-prefix predicate patterns as value-op masks.

The per-op suites cover full / prefix / all-false masks, but a prefix mask is
self-similar under several lane reinterpretations and can hide lane-mapping
mistakes.  This file drives the interleaved hardware pattern ``PAT.M4``
(every 4th lane true, validated bit-for-bit in pset.py) through value ops:

  0 vand(a, b, M4)                  ZEROING: inactive lanes read 0
  1 vand(a, b, M4, MERGING)         inactive lanes keep ``a``
  2 vmula(acc, x, y, M4)            ZEROING
  3 vmula(acc, x, y, M4, MERGING)   inactive lanes keep ``acc``

``M4`` over the 64-lane b32 register is lane ``i % 4 == 0``.

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

SHAPE = dict(lanes=64)
SIGNATURE = {"X": "*u32", "Y": "*u32", "A": "*fp32", "P": "*fp32", "O": "*u32", "F": "*fp32"}


@tla.jit
def pat_mask(X, Y, A, P, O, F, N: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.uint32, space="ub")
    y = tla.alloc_buffer((N, ), tl.uint32, space="ub")
    a = tla.alloc_buffer((N, ), tl.float32, space="ub")
    p = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o = tla.alloc_buffer((2, N), tl.uint32, space="ub")
    f = tla.alloc_buffer((2, N), tl.float32, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(Y, (N, )), y)
    tla.copy(tla.view(A, (N, )), a)
    tla.copy(tla.view(P, (N, )), p)
    with tla.scope.simd():
        xv = tla.simd.vlds(x[0])
        yv = tla.simd.vlds(y[0])
        m4 = tla.simd.pset(32, tla.simd.PAT.M4)
        tla.simd.vsts(o[0, 0], tla.simd.vand(xv, yv, m4))
        tla.simd.vsts(o[1, 0], tla.simd.vand(xv, yv, mask=m4, mode=tla.simd.MODE.MERGING))
        acc = tla.simd.vlds(a[0])
        prod = tla.simd.vlds(p[0])
        two = tla.simd.vsplat(2.0, tl.float32)
        tla.simd.vsts(f[0, 0], tla.simd.vmula(acc, prod, two, m4))
        tla.simd.vsts(f[1, 0], tla.simd.vmula(acc, prod, two, mask=m4, mode=tla.simd.MODE.MERGING))
    tla.copy(o, tla.view(O, (2, N)))
    tla.copy(f, tla.view(F, (2, N)))


def constexprs():
    return {"N": SHAPE["lanes"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(pat_mask, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.vand") == 2
    assert text.count("ave.hir.vmula") == 2
    assert text.count("ave.hir.vsel") == 2, "only the two MERGING rows add vsel"
    assert text.count("ave.hir.pge <M4>") == 1, "one shared M4 predicate"
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=4, stores=2, buffers=None, extra=("ave.hir.vand", "ave.hir.vmula"))
    print("IR PASS: PAT.M4 masks through vand/vmula")


def _launch():
    n = SHAPE["lanes"]
    xi = torch.randint(0, 1 << 32, (n, ), dtype=torch.int64)
    yi = torch.randint(0, 1 << 32, (n, ), dtype=torch.int64)
    a = torch.randn(n)
    p = torch.randn(n)
    out = torch.empty((2, n), dtype=torch.uint32, device="npu")
    f = torch.empty((2, n), dtype=torch.float32, device="npu")
    pat_mask[(1, )](xi.to(torch.uint32).npu(), yi.to(torch.uint32).npu(), a.npu(), p.npu(), out, f, **constexprs())
    torch.npu.synchronize()
    m4 = torch.arange(n) % 4 == 0
    full = (xi & yi) & 0xffffffff
    oref = torch.stack([
        torch.where(m4, full, torch.zeros((), dtype=torch.int64)),
        torch.where(m4, full, xi & 0xffffffff),
    ])
    fsum = a + p * 2.0
    fref = torch.stack([
        torch.where(m4, fsum, torch.zeros((), dtype=torch.float32)),
        torch.where(m4, fsum, a),
    ])
    return out.cpu(), oref.to(torch.uint32), f.cpu(), fref


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: pat_mask generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    out, oref, f, fref = _launch()
    torch.testing.assert_close(out.view(torch.int32), oref.view(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(f, fref, rtol=1e-6, atol=1e-6)
    print("PAT_MASK OK: interleaved M4 patterns, ZEROING and MERGING")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: pat_mask", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
