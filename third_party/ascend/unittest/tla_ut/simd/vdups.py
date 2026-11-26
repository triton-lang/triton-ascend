"""TLA 3-in-1 test: vdups is the masked ``ave.hir.broadcast`` alias of vdup.

Two identical row groups are emitted from one kernel so the device comparison
is a raw-bit equality check, not a tolerance test:

  0..3  vdups: all / prefix / none / prefix+MERGING
  4..7  vdup : same four calls, same masks

Both spellings must lower to the masked ``ave.hir.broadcast`` (never to
``ave.hir.scalar_broadcast``), and every inactive lane must read as ``+0``.
The masks come from a signed lane index, so the predicate is independent of the
payload dtype.

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

SHAPE = dict(lanes=64, active=32, rows=8)
SIGNATURE = {"X": "*fp32", "O": "*fp32"}


@tla.jit
def vdups(X, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    xb = tla.alloc_buffer((N, ), tl.float32, space="ub")
    ob = tla.alloc_buffer((8, N), tl.float32, space="ub")
    tla.copy(tla.view(X, (N, )), xb)
    with tla.scope.simd():
        x = tla.simd.vlds(xb[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(ob[0, 0], tla.simd.vdups(1.5, tl.float32))
        tla.simd.vsts(ob[1, 0], tla.simd.vdups(1.5, tl.float32, mask=prefix))
        tla.simd.vsts(ob[2, 0], tla.simd.vdups(1.5, tl.float32, mask=none))
        tla.simd.vsts(ob[3, 0], tla.simd.vdups(1.5, tl.float32, mask=prefix, mode=tla.simd.MODE.MERGING))
        tla.simd.vsts(ob[4, 0], tla.simd.vdup(1.5, tl.float32))
        tla.simd.vsts(ob[5, 0], tla.simd.vdup(1.5, tl.float32, mask=prefix))
        tla.simd.vsts(ob[6, 0], tla.simd.vdup(1.5, tl.float32, mask=none))
        tla.simd.vsts(ob[7, 0], tla.simd.vdup(1.5, tl.float32, mask=prefix, mode=tla.simd.MODE.MERGING))
    tla.copy(ob, tla.view(O, (8, N)))


def constexprs():
    return {"N": SHAPE["lanes"], "ACTIVE": SHAPE["active"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vdups, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.broadcast") == 8, "vdups and vdup share one lowering"
    # The two MERGING rows add a +0 ``scalar_broadcast`` merge source; the eight
    # broadcast ops above prove neither spelling degraded to vsplat.
    assert text.count("ave.hir.scalar_broadcast") == 2, "only the MERGING merge source is scalar_broadcast"
    assert text.count("ave.hir.vsel") == 2, "one MERGING row per spelling"
    assert "ave.hir.pge <VL32>" in text
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=1, stores=1, buffers=None, extra=("ave.hir.broadcast", ))
    print("IR PASS: vdups/vdup share ave.hir.broadcast with masked +0 merge")


def _launch():
    n = SHAPE["lanes"]
    x = torch.randn(n)
    idx = torch.arange(n)
    prefix = idx < SHAPE["active"]
    out = torch.empty((SHAPE["rows"], n), dtype=torch.float32, device="npu")
    vdups[(1, )](x.npu(), out, **constexprs())
    torch.npu.synchronize()
    full = torch.full((n, ), 1.5)
    ref = torch.stack([
        full,
        torch.where(prefix, full, torch.zeros_like(full)),
        torch.zeros_like(full),
        torch.where(prefix, full, torch.zeros_like(full)),
        full,
        torch.where(prefix, full, torch.zeros_like(full)),
        torch.zeros_like(full),
        torch.where(prefix, full, torch.zeros_like(full)),
    ])
    return out.cpu(), ref, prefix


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vdups generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    got, ref, prefix = _launch()
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    # The alias must be bit-identical to vdup on device for every mask/mode.
    torch.testing.assert_close(got[:4].view(torch.int32), got[4:].view(torch.int32), rtol=0, atol=0)
    assert bool((got[1][~prefix] == 0).all()) and bool((got[3][~prefix] == 0).all())
    assert bool((got[2] == 0).all()) and bool((got[6] == 0).all())
    print("VDUPS OK: bit-identical to vdup, masked broadcast zeroes inactive lanes")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vdups", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
