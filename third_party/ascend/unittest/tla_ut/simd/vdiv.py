"""TLA 3-in-1 test: vdiv, masks/modes, fp32 exact tier and unsigned zeroing.

f32 kernel rows (64 lanes each):

  0 vdiv(a, b)                    default ALL mask, MODE_ZEROING
  1 vdiv(a, b, prefix)            first ACTIVE lanes, MODE_ZEROING
  2 vdiv(a, b, arb)               per-lane threshold from ``T`` (signed i32 index)
  3 vdiv(a, b, none)              all-false predicate
  4 vdiv(a, b, prefix, MERGING)   merge source is ``+0``
  5 vdiv(a, b, arb, MERGING)

Two device paths that IR alone cannot settle are pinned here:

* ``precision='exact'`` selects ``ave.hir.vdivfhp`` (0ULP); the result is
  compared bit-for-bit against IEEE f32 division with the caller mask forwarded.
* unsigned 16/32-bit division under a caller-supplied ZEROING mask: the AVE
  software sequence does not zero inactive lanes by itself, so the frontend
  appends ``vsel(result, 0)``.  The device check requires every inactive lane to
  be bitwise ``+0``; MERGING reuses the builder-side ``+0`` merge instead.
  Every mask is built from signed lane indices (never an unsigned data compare).

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
U16_LANES = 128
EXACT_ROWS = 4
SIGNATURE = {"X": "*fp32", "Y": "*fp32", "T": "*i32", "O": "*fp32"}
EXACT_SIGNATURE = {"X": "*fp32", "Y": "*fp32", "O": "*fp32"}
U32_SIGNATURE = {"X": "*u32", "Y": "*u32", "T": "*i32", "O": "*u32"}
U16_SIGNATURE = {"X": "*u16", "Y": "*u16", "I": "*i16", "T": "*i16", "O": "*u16"}

# Exact-quotient pairs: the precise tier must match correctly rounded f32.
EXACT_A = [1.0, 3.0, 1e-30, 1.0, 7.0, -7.0, 1e20, 1e-20, 2.0, 5.0, 11.0, -3.0]
EXACT_B = [3.0, 7.0, 3e-30, 0.1, 3.0, 3.0, 3.0, 7e-21, 3.0, 7.0, 13.0, 7.0]


@tla.jit
def vdiv(X, Y, T, O, N: tl.constexpr, ACTIVE: tl.constexpr):
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
        tla.simd.vsts(ob[0, 0], tla.simd.vdiv(a, b, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[1, 0], tla.simd.vdiv(a, b, mask=prefix, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[2, 0], tla.simd.vdiv(a, b, mask=arb, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[3, 0], tla.simd.vdiv(a, b, mask=none, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[4, 0], tla.simd.vdiv(a, b, mask=prefix, mode=tla.simd.MODE.MERGING))
        tla.simd.vsts(ob[5, 0], tla.simd.vdiv(a, b, mask=arb, mode=tla.simd.MODE.MERGING))
    tla.copy(ob, tla.view(O, (6, N)))


@tla.jit
def vdiv_exact(X, Y, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    xy = tla.alloc_buffer((2, N), tl.float32, space="ub")
    ob = tla.alloc_buffer((4, N), tl.float32, space="ub")
    tla.copy(tla.view(X, (2, N)), xy)
    with tla.scope.simd():
        a = tla.simd.vlds(xy[0, 0])
        b = tla.simd.vlds(xy[1, 0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(ob[0, 0], tla.simd.vdiv(a, b, precision="exact"))
        tla.simd.vsts(ob[1, 0], tla.simd.vdiv(a, b, mask=prefix, precision="exact"))
        tla.simd.vsts(ob[2, 0], tla.simd.vdiv(a, b, mask=none, precision="exact"))
        tla.simd.vsts(ob[3, 0], tla.simd.vdiv(a, b, mask=prefix, precision="vdiv_0ulp_ftz_true"))
    tla.copy(ob, tla.view(O, (4, N)))


@tla.jit
def vdiv_u32(X, Y, T, O, N: tl.constexpr, ACTIVE: tl.constexpr):
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
        tla.simd.vsts(ob[0, 0], tla.simd.vdiv(a, b, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[1, 0], tla.simd.vdiv(a, b, mask=prefix, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[2, 0], tla.simd.vdiv(a, b, mask=arb, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[3, 0], tla.simd.vdiv(a, b, mask=none, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[4, 0], tla.simd.vdiv(a, b, mask=prefix, mode=tla.simd.MODE.MERGING))
        tla.simd.vsts(ob[5, 0], tla.simd.vdiv(a, b, mask=arb, mode=tla.simd.MODE.MERGING))
    tla.copy(ob, tla.view(O, (6, N)))


@tla.jit
def vdiv_u16(X, Y, I, T, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    xy = tla.alloc_buffer((2, N), tl.uint16, space="ub")
    ib = tla.alloc_buffer((N, ), tl.int16, space="ub")
    tb = tla.alloc_buffer((N, ), tl.int16, space="ub")
    ob = tla.alloc_buffer((6, N), tl.uint16, space="ub")
    tla.copy(tla.view(X, (2, N)), xy)
    tla.copy(tla.view(I, (N, )), ib)
    tla.copy(tla.view(T, (N, )), tb)
    with tla.scope.simd():
        a = tla.simd.vlds(xy[0, 0])
        b = tla.simd.vlds(xy[1, 0])
        idx = tla.simd.vlds(ib[0])
        t = tla.simd.vlds(tb[0])
        prefix = tla.simd.pge(ACTIVE, N)
        arb = tla.simd.vcmp(idx, t, op=tla.simd.CMP.LT)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(ob[0, 0], tla.simd.vdiv(a, b, mode=tla.simd.MODE.ZEROING), dist="NORM_B16")
        tla.simd.vsts(ob[1, 0], tla.simd.vdiv(a, b, mask=prefix, mode=tla.simd.MODE.ZEROING), dist="NORM_B16")
        tla.simd.vsts(ob[2, 0], tla.simd.vdiv(a, b, mask=arb, mode=tla.simd.MODE.ZEROING), dist="NORM_B16")
        tla.simd.vsts(ob[3, 0], tla.simd.vdiv(a, b, mask=none, mode=tla.simd.MODE.ZEROING), dist="NORM_B16")
        tla.simd.vsts(ob[4, 0], tla.simd.vdiv(a, b, mask=prefix, mode=tla.simd.MODE.MERGING), dist="NORM_B16")
        tla.simd.vsts(ob[5, 0], tla.simd.vdiv(a, b, mask=arb, mode=tla.simd.MODE.MERGING), dist="NORM_B16")
    tla.copy(ob, tla.view(O, (6, N)))


def constexprs(lanes=None, active=None):
    return {"N": SHAPE["lanes"] if lanes is None else lanes, "ACTIVE": SHAPE["active"] if active is None else active}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vdiv, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.vdiv ") == 6
    assert text.count("ave.hir.vsel") == 2, "f32 ZEROING stays bare; only MERGING adds vsel"
    assert "ave.hir.pge <VL32>" in text
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=2, stores=1, buffers=None, extra=("ave.hir.vdiv", ))

    module, ctx = compile_offline(vdiv_exact, EXACT_SIGNATURE, constexprs(), output_dir=output_dir)
    etext = str(module)
    assert etext.count("ave.hir.vdivfhp ") == 4, "exact/0ulp tiers must use vdivfhp"
    assert "ave.hir.vsel" not in etext, "exact tier forwards the mask without an extra select"
    etext = str(lower_offline(module, ctx))
    check_lowered_ir(etext, loads=1, stores=1, buffers=None, extra=("ave.hir.vdivfhp", ))

    module, ctx = compile_offline(vdiv_u32, U32_SIGNATURE, constexprs())
    utext = str(module)
    assert "cast_unsigned" in utext, "u32 division must select the unsigned cast"
    # 3 caller-masked ZEROING rows append vsel(result, 0); 2 MERGING rows merge.
    assert utext.count("ave.hir.vsel") == 5
    utext = str(lower_offline(module, ctx))
    check_lowered_ir(utext, loads=2, stores=1, buffers=None, extra=("ave.hir.vdiv", "cast_unsigned"))

    module, ctx = compile_offline(vdiv_u16, U16_SIGNATURE, constexprs(U16_LANES, U16_LANES // 2))
    htext = str(module)
    assert "cast_unsigned" in htext and htext.count("ave.hir.vsel") == 5
    assert "ave.hir.pge <VL64>" in htext
    htext = str(lower_offline(module, ctx))
    check_lowered_ir(htext, loads=3, stores=1, buffers=None, extra=("ave.hir.vdiv", "cast_unsigned"))
    print("IR PASS: vdiv f32 masks/modes, f32 exact vdivfhp, u32/u16 zeroing select")


def _masks():
    n = SHAPE["lanes"]
    idx = torch.arange(n)
    thr = (torch.arange(n) % 5) * 12
    return {"prefix": idx < SHAPE["active"], "arb": idx < thr, "none": torch.zeros(n, dtype=torch.bool)}, thr


def _apply(value, mask):
    if value.dtype in (torch.uint32, torch.uint64):
        iv = value.to(torch.int64)
        return torch.where(mask, iv, torch.zeros((), dtype=torch.int64)).to(value.dtype)
    return torch.where(mask, value, torch.zeros((), dtype=value.dtype))


def _launch():
    n = SHAPE["lanes"]
    a = torch.randn(n)
    b = torch.rand(n) * 2.0 + 0.5
    masks, thr = _masks()
    out = torch.empty((SHAPE["rows"], n), dtype=torch.float32, device="npu")
    vdiv[(1, )](torch.stack([a, b]).npu(), a.npu(), thr.to(torch.int32).npu(), out, **constexprs())
    torch.npu.synchronize()
    q = a / b
    ref = torch.stack([
        q,
        _apply(q, masks["prefix"]),
        _apply(q, masks["arb"]),
        _apply(q, masks["none"]),
        _apply(q, masks["prefix"]),
        _apply(q, masks["arb"]),
    ])

    ea = torch.tensor((EXACT_A * 8)[:n], dtype=torch.float32)
    eb = torch.tensor((EXACT_B * 8)[:n], dtype=torch.float32)
    eout = torch.empty((EXACT_ROWS, n), dtype=torch.float32, device="npu")
    vdiv_exact[(1, )](torch.stack([ea, eb]).npu(), ea.npu(), eout, **constexprs())
    torch.npu.synchronize()
    eq = ea / eb
    eref = torch.stack([
        eq,
        _apply(eq, masks["prefix"]),
        _apply(eq, masks["none"]),
        _apply(eq, masks["prefix"]),
    ])

    ua = torch.randint(0, 1 << 32, (n, ), dtype=torch.int64).to(torch.uint32)
    ub = torch.randint(1, 1000, (n, ), dtype=torch.int64).to(torch.uint32)
    uout = torch.empty((SHAPE["rows"], n), dtype=torch.uint32, device="npu")
    vdiv_u32[(1, )](torch.stack([ua, ub]).to(torch.uint32).npu(), ua.npu(), thr.to(torch.int32).npu(), uout,
                    **constexprs())
    torch.npu.synchronize()
    uq = ((ua.to(torch.int64) // ub.to(torch.int64)) & 0xFFFFFFFF).to(torch.uint32)
    uref = torch.stack([
        uq,
        _apply(uq, masks["prefix"]),
        _apply(uq, masks["arb"]),
        _apply(uq, masks["none"]),
        _apply(uq, masks["prefix"]),
        _apply(uq, masks["arb"]),
    ])

    m = U16_LANES
    i16 = torch.arange(m, dtype=torch.int16)
    t16 = ((torch.arange(m) % 3) * 20).to(torch.int16)
    masks16 = {
        "prefix": torch.arange(m) < m // 2,
        "arb": torch.arange(m) < t16.to(torch.int64),
        "none": torch.zeros(m, dtype=torch.bool),
    }
    ha = torch.randint(1, 60000, (m, ), dtype=torch.int64)
    hb = torch.randint(1, 60, (m, ), dtype=torch.int64)
    # The kernel signature is u16: the runtime dtype is inferred from the tensor,
    # so the host buffers must be uint16 (an int16 tensor would compile the signed
    # path and silently skip the unsigned-zeroing regression).
    hout = torch.empty((SHAPE["rows"], m), dtype=torch.uint16, device="npu")
    vdiv_u16[(1, )](torch.stack([ha, hb]).to(torch.uint16).npu(), ha.to(torch.uint16).npu(), i16.npu(), t16.npu(), hout,
                    **constexprs(U16_LANES, m // 2))
    torch.npu.synchronize()
    hq = ha // hb
    href = torch.stack([
        hq,
        _apply(hq, masks16["prefix"]),
        _apply(hq, masks16["arb"]),
        _apply(hq, masks16["none"]),
        _apply(hq, masks16["prefix"]),
        _apply(hq, masks16["arb"]),
    ])
    return {
        "f32": (out.cpu(), ref, masks),
        "exact": (eout.cpu(), eref, masks),
        "u32": (uout.cpu(), uref, masks),
        # Compare the u16 payloads as their int16 bit pattern (same bits, exact).
        "u16": (hout.cpu().view(torch.int16), href.to(torch.int16), masks16),
    }


def _bitwise_zero(t):
    """True when every element is the raw bit pattern of +0 (so -0.0 fails)."""
    if t.dtype == torch.float32:
        return bool((t.view(torch.int32) == 0).all())
    if t.dtype == torch.float16:
        return bool((t.view(torch.int16) == 0).all())
    return bool((t == 0).all())


def _inactive_zero(got, masks):
    """Every lane outside the mask must be bitwise +0 (not merely close)."""
    for row, name in ((1, "prefix"), (2, "arb"), (3, "none"), (4, "prefix"), (5, "arb")):
        keep = masks[name]
        inactive = got[row][~keep] if keep.any() else got[row]
        assert _bitwise_zero(inactive), f"row {row} ({name}) has a non-zero inactive lane"


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vdiv generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    runs = _launch()
    fgot, fref, fmasks = runs["f32"]
    torch.testing.assert_close(fgot, fref, rtol=1e-5, atol=1e-6)
    _inactive_zero(fgot, fmasks)
    for key in ("exact", "u32"):
        got, ref, masks = runs[key]
        torch.testing.assert_close(got.view(torch.int32), ref.view(torch.int32), rtol=0, atol=0)
        _inactive_zero(got, masks)
    hgot, href, hmasks = runs["u16"]
    torch.testing.assert_close(hgot, href, rtol=0, atol=0)
    _inactive_zero(hgot, hmasks)
    print("VDIV OK: masks/modes, f32 exact bits, u32/u16 masked zeroing")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vdiv", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
