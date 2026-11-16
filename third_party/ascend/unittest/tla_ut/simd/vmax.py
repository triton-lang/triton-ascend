"""TLA 3-in-1 test: elementwise vmax, ALL/prefix/arbitrary/none masks, both modes.

One 64-lane f32 register per output row:

  0 vmax(a, b)                    default ALL mask, MODE_ZEROING
  1 vmax(a, b, prefix)            first ACTIVE lanes, MODE_ZEROING
  2 vmax(a, b, arb)               per-lane threshold from ``T`` (signed i32 index)
  3 vmax(a, b, none)              all-false predicate
  4 vmax(a, b, prefix, MERGING)   merge source is ``+0``
  5 vmax(a, b, arb, MERGING)

Inactive lanes read as ``+0`` in both modes; MERGING only differs in IR
(the masked AVE op plus a ``vsel`` against a ``+0`` merge source).  A second
kernel pins the unsigned path: ``u32`` operands must select ``ave.hir.vumax``,
so high-bit values must not compare as negative.

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).
Inputs are regenerated on every run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

SHAPE = dict(lanes=64, active=32, rows=6)
SIGNATURE = {"X": "*fp32", "Y": "*fp32", "T": "*i32", "O": "*fp32"}
U32_SIGNATURE = {"X": "*u32", "Y": "*u32", "T": "*i32", "O": "*u32"}


@tla.jit
def vmax(X, Y, T, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    xy = tla.alloc_buffer((2, N), tl.float32, space="ub")
    tb = tla.alloc_buffer((N, ), tl.int32, space="ub")
    ob = tla.alloc_buffer((6, N), tl.float32, space="ub")
    tla.copy(tla.view(X, (2, N)), xy)
    tla.copy(tla.view(T, (N, )), tb)
    with tla.scope.simd():
        a = tla.simd.vlds(xy[0, 0])
        b = tla.simd.vlds(xy[1, 0])
        t = tla.simd.vlds(tb[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        arb = tla.simd.vcmp(idx, t, op=tla.simd.CMP.LT)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(ob[0, 0], tla.simd.vmax(a, b, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[1, 0], tla.simd.vmax(a, b, mask=prefix, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[2, 0], tla.simd.vmax(a, b, mask=arb, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[3, 0], tla.simd.vmax(a, b, mask=none, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[4, 0], tla.simd.vmax(a, b, mask=prefix, mode=tla.simd.MODE.MERGING))
        tla.simd.vsts(ob[5, 0], tla.simd.vmax(a, b, mask=arb, mode=tla.simd.MODE.MERGING))
    tla.copy(ob, tla.view(O, (6, N)))


@tla.jit
def vmax_u32(X, Y, T, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    xy = tla.alloc_buffer((2, N), tl.uint32, space="ub")
    tb = tla.alloc_buffer((N, ), tl.int32, space="ub")
    ob = tla.alloc_buffer((6, N), tl.uint32, space="ub")
    tla.copy(tla.view(X, (2, N)), xy)
    tla.copy(tla.view(T, (N, )), tb)
    with tla.scope.simd():
        a = tla.simd.vlds(xy[0, 0])
        b = tla.simd.vlds(xy[1, 0])
        t = tla.simd.vlds(tb[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        arb = tla.simd.vcmp(idx, t, op=tla.simd.CMP.LT)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(ob[0, 0], tla.simd.vmax(a, b, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[1, 0], tla.simd.vmax(a, b, mask=prefix, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[2, 0], tla.simd.vmax(a, b, mask=arb, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[3, 0], tla.simd.vmax(a, b, mask=none, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[4, 0], tla.simd.vmax(a, b, mask=prefix, mode=tla.simd.MODE.MERGING))
        tla.simd.vsts(ob[5, 0], tla.simd.vmax(a, b, mask=arb, mode=tla.simd.MODE.MERGING))
    tla.copy(ob, tla.view(O, (6, N)))


def constexprs():
    return {"N": SHAPE["lanes"], "ACTIVE": SHAPE["active"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vmax, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.vmax") == 6
    assert text.count("ave.hir.vsel") == 2, "only the two MERGING rows merge a +0 source"
    assert "ave.hir.pge <VL32>" in text, "prefix mask must be VL32"
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=2, stores=1, buffers=None, extra=("ave.hir.vmax", ))

    module, ctx = compile_offline(vmax_u32, U32_SIGNATURE, constexprs())
    utext = str(module)
    assert "ave.hir.vumax" in utext and "ave.hir.vmax " not in utext, "u32 must pick vumax"
    utext = str(lower_offline(module, ctx))
    check_lowered_ir(utext, loads=2, stores=1, buffers=None, extra=("ave.hir.vumax", ))
    print("IR PASS: vmax f32 six masks/modes plus u32 vumax")


def _masks():
    n, t = SHAPE["lanes"], SHAPE["lanes"]
    idx = torch.arange(n)
    thr = (torch.arange(t) % 5) * 12
    return {"prefix": idx < SHAPE["active"], "arb": idx < thr, "none": torch.zeros(n, dtype=torch.bool)}, thr


def _apply(value, mask):
    if value.dtype in (torch.uint32, torch.uint64):
        iv = value.to(torch.int64)
        return torch.where(mask, iv, torch.zeros((), dtype=torch.int64)).to(value.dtype)
    return torch.where(mask, value, torch.zeros((), dtype=value.dtype))


def _launch():
    n = SHAPE["lanes"]
    a = torch.randn(n)
    b = torch.randn(n)
    masks, thr = _masks()
    out = torch.empty((SHAPE["rows"], n), dtype=torch.float32, device="npu")
    vmax[(1, )](torch.stack([a, b]).npu(), a.npu(), thr.to(torch.int32).npu(), out, **constexprs())
    torch.npu.synchronize()
    full = torch.maximum(a, b)
    ref = torch.stack([
        full,
        _apply(full, masks["prefix"]),
        _apply(full, masks["arb"]),
        _apply(full, masks["none"]),
        _apply(full, masks["prefix"]),
        _apply(full, masks["arb"]),
    ])
    ua = torch.randint(0, 1 << 32, (n, ), dtype=torch.int64)
    ub = torch.randint(0, 1 << 32, (n, ), dtype=torch.int64)
    uout = torch.empty((SHAPE["rows"], n), dtype=torch.uint32, device="npu")
    vmax_u32[(1, )](torch.stack([ua, ub]).to(torch.uint32).npu(), ua.to(torch.uint32).npu(), thr.to(torch.int32).npu(),
                    uout, **constexprs())
    torch.npu.synchronize()
    ufull = ua.clamp(max=0xFFFFFFFF).maximum(ub.clamp(max=0xFFFFFFFF))
    uref = torch.stack([
        ufull,
        _apply(ufull, masks["prefix"]),
        _apply(ufull, masks["arb"]),
        _apply(ufull, masks["none"]),
        _apply(ufull, masks["prefix"]),
        _apply(ufull, masks["arb"]),
    ])
    return out.cpu(), ref, uout.cpu(), uref.to(torch.uint32)


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vmax generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    got, ref, ugot, uref = _launch()
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    torch.testing.assert_close(ugot.view(torch.int32), uref.view(torch.int32), rtol=0, atol=0)
    print("VMAX OK: ALL/prefix/arbitrary/none, ZEROING/MERGING init0, u32 unsigned")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vmax", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
