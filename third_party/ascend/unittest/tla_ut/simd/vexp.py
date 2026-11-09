"""TLA 3-in-1 test: vexp on f32 and f16 with ALL/prefix/arbitrary/none masks.

Rows of ``O`` (one full register each: 64 f32 lanes, 128 f16 lanes):

  0 vexp(x)                    default ALL mask, MODE_ZEROING
  1 vexp(x, prefix)            first ACTIVE lanes, MODE_ZEROING
  2 vexp(x, arb)               per-lane threshold from ``T`` (signed lane index)
  3 vexp(x, none)              all-false predicate
  4 vexp(x, prefix, MERGING)   merge source is ``+0``
  5 vexp(x, arb, MERGING)

The f16 masks are built from a signed ``i16`` lane-index row, never from an
unsigned data compare: the mask is lane-shaped, so it must stay independent of
the payload dtype.  Inactive lanes read as ``+0`` in both modes.

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

SHAPE = dict(lanes=64, active=32, rows=6)
F16_LANES = 128
SIGNATURE = {"X": "*fp32", "T": "*i32", "O": "*fp32"}
F16_SIGNATURE = {"X": "*fp16", "I": "*i16", "T": "*i16", "O": "*fp16"}


@tla.jit
def vexp(X, T, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    xb = tla.alloc_buffer((N, ), tl.float32, space="ub")
    tb = tla.alloc_buffer((N, ), tl.int32, space="ub")
    ob = tla.alloc_buffer((6, N), tl.float32, space="ub")
    tla.copy(tla.view(X, (N, )), xb)
    tla.copy(tla.view(T, (N, )), tb)
    with tla.scope.simd():
        x = tla.simd.vlds(xb[0])
        t = tla.simd.vlds(tb[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        arb = tla.simd.vcmp(idx, t, op=tla.simd.CMP.LT)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(ob[0, 0], tla.simd.vexp(x, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[1, 0], tla.simd.vexp(x, mask=prefix, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[2, 0], tla.simd.vexp(x, mask=arb, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[3, 0], tla.simd.vexp(x, mask=none, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[4, 0], tla.simd.vexp(x, mask=prefix, mode=tla.simd.MODE.MERGING))
        tla.simd.vsts(ob[5, 0], tla.simd.vexp(x, mask=arb, mode=tla.simd.MODE.MERGING))
    tla.copy(ob, tla.view(O, (6, N)))


@tla.jit
def vexp_f16(X, I, T, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    xb = tla.alloc_buffer((N, ), tl.float16, space="ub")
    ib = tla.alloc_buffer((N, ), tl.int16, space="ub")
    tb = tla.alloc_buffer((N, ), tl.int16, space="ub")
    ob = tla.alloc_buffer((6, N), tl.float16, space="ub")
    tla.copy(tla.view(X, (N, )), xb)
    tla.copy(tla.view(I, (N, )), ib)
    tla.copy(tla.view(T, (N, )), tb)
    with tla.scope.simd():
        x = tla.simd.vlds(xb[0])
        idx = tla.simd.vlds(ib[0])
        t = tla.simd.vlds(tb[0])
        prefix = tla.simd.pge(ACTIVE, N)
        arb = tla.simd.vcmp(idx, t, op=tla.simd.CMP.LT)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(ob[0, 0], tla.simd.vexp(x, mode=tla.simd.MODE.ZEROING), dist="NORM_B16")
        tla.simd.vsts(ob[1, 0], tla.simd.vexp(x, mask=prefix, mode=tla.simd.MODE.ZEROING), dist="NORM_B16")
        tla.simd.vsts(ob[2, 0], tla.simd.vexp(x, mask=arb, mode=tla.simd.MODE.ZEROING), dist="NORM_B16")
        tla.simd.vsts(ob[3, 0], tla.simd.vexp(x, mask=none, mode=tla.simd.MODE.ZEROING), dist="NORM_B16")
        tla.simd.vsts(ob[4, 0], tla.simd.vexp(x, mask=prefix, mode=tla.simd.MODE.MERGING), dist="NORM_B16")
        tla.simd.vsts(ob[5, 0], tla.simd.vexp(x, mask=arb, mode=tla.simd.MODE.MERGING), dist="NORM_B16")
    tla.copy(ob, tla.view(O, (6, N)))


def constexprs(lanes=None, active=None):
    return {"N": SHAPE["lanes"] if lanes is None else lanes, "ACTIVE": SHAPE["active"] if active is None else active}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vexp, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.vexp") == 6
    assert text.count("ave.hir.vsel") == 2, "only the two MERGING rows add vsel"
    assert "ave.hir.pge <VL32>" in text
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=2, stores=1, buffers=None, extra=("ave.hir.vexp", ))

    module, ctx = compile_offline(vexp_f16, F16_SIGNATURE, constexprs(F16_LANES, F16_LANES // 2))
    htext = str(module)
    assert htext.count("ave.hir.vexp") == 6
    assert "ave.hir.pge <VL64>" in htext, "f16 prefix mask must be VL64 of 128 lanes"
    htext = str(lower_offline(module, ctx))
    check_lowered_ir(htext, loads=3, stores=1, buffers=None, extra=("ave.hir.vexp", ))
    print("IR PASS: vexp f32/f16 six masks/modes")


def _masks(n, active):
    idx = torch.arange(n)
    thr = (torch.arange(n) % 5) * (n // 5)
    return {"prefix": idx < active, "arb": idx < thr, "none": torch.zeros(n, dtype=torch.bool)}, thr


def _apply(value, mask):
    return torch.where(mask, value, torch.zeros((), dtype=value.dtype))


def _launch():
    n = SHAPE["lanes"]
    x = torch.randn(n) * 0.5
    masks, thr = _masks(n, SHAPE["active"])
    out = torch.empty((SHAPE["rows"], n), dtype=torch.float32, device="npu")
    vexp[(1, )](x.npu(), thr.to(torch.int32).npu(), out, **constexprs())
    torch.npu.synchronize()
    e = torch.exp(x)
    ref = torch.stack([
        e,
        _apply(e, masks["prefix"]),
        _apply(e, masks["arb"]),
        _apply(e, masks["none"]),
        _apply(e, masks["prefix"]),
        _apply(e, masks["arb"]),
    ])

    m = F16_LANES
    hx = (torch.randn(m) * 0.5).to(torch.float16)
    i16 = torch.arange(m, dtype=torch.int16)
    t16 = ((torch.arange(m) % 5) * (m // 5)).to(torch.int16)
    hmasks = {
        "prefix": torch.arange(m) < m // 2, "arb": torch.arange(m) < t16.to(torch.int64), "none":
        torch.zeros(m, dtype=torch.bool)
    }
    hout = torch.empty((SHAPE["rows"], m), dtype=torch.float16, device="npu")
    vexp_f16[(1, )](hx.npu(), i16.npu(), t16.npu(), hout, **constexprs(F16_LANES, m // 2))
    torch.npu.synchronize()
    he = torch.exp(hx.float()).to(torch.float16)
    href = torch.stack([
        he,
        _apply(he, hmasks["prefix"]),
        _apply(he, hmasks["arb"]),
        _apply(he, hmasks["none"]),
        _apply(he, hmasks["prefix"]),
        _apply(he, hmasks["arb"]),
    ])
    return (out.cpu(), ref, masks), (hout.cpu(), href, hmasks)


def _bitwise_zero(t):
    if t.dtype == torch.float32:
        return bool((t.view(torch.int32) == 0).all())
    return bool((t.view(torch.int16) == 0).all())


def _inactive_zero(got, masks):
    for row, name in ((1, "prefix"), (2, "arb"), (3, "none"), (4, "prefix"), (5, "arb")):
        keep = masks[name]
        inactive = got[row][~keep] if keep.any() else got[row]
        assert _bitwise_zero(inactive), f"row {row} ({name}) has a non-zero inactive lane"


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vexp generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    (got, ref, masks), (hgot, href, hmasks) = _launch()
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=0)
    _inactive_zero(got, masks)
    torch.testing.assert_close(hgot, href, rtol=5e-3, atol=0)
    _inactive_zero(hgot, hmasks)
    print("VEXP OK: f32/f16 ALL/prefix/arbitrary/none, ZEROING/MERGING init0")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vexp", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
