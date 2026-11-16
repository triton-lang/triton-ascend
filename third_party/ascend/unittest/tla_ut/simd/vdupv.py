"""TLA 3-in-1 test: vdupv lane broadcast (LOWEST/HIGHEST), 32 and 64 bit.

f32 kernel, one 64-lane register per row:

   0/1  vdupv(x, LOWEST/HIGHEST)                  default ALL mask
   2/3  ... prefix mask                           first ACTIVE lanes
   4/5  ... arbitrary mask                        per-lane threshold from ``T``
   6/7  ... all-false mask                        every lane must be ``+0``
   8/9  ... prefix mask, MODE_MERGING             merge source is ``+0``

The wide kernel repeats LOWEST/HIGHEST for real 64-lane ``i64`` and ``u64``
registers with prefix/arbitrary/all-false masks.  Those masks are built from a
signed ``i32`` lane index compared with a per-lane threshold, so neither the
mask nor its compare depends on unsigned 64-bit data (no unsigned ``vcmp``).
``vdupv`` must use the real ``ave.hir.vector_broadcast`` 64-lane entry, not a
``vbitcast`` of a 32-lane register.

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

SHAPE = dict(lanes=64, active=32, rows=10)
WIDE_ROWS = 16
SIGNATURE = {"X": "*fp32", "T": "*i32", "O": "*fp32"}
WIDE_SIGNATURE = {"I": "*i64", "U": "*u64", "T": "*i32", "OI": "*i64", "OU": "*u64"}


@tla.jit
def vdupv(X, T, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    xb = tla.alloc_buffer((N, ), tl.float32, space="ub")
    tb = tla.alloc_buffer((N, ), tl.int32, space="ub")
    ob = tla.alloc_buffer((10, N), tl.float32, space="ub")
    tla.copy(tla.view(X, (N, )), xb)
    tla.copy(tla.view(T, (N, )), tb)
    with tla.scope.simd():
        x = tla.simd.vlds(xb[0])
        t = tla.simd.vlds(tb[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        arb = tla.simd.vcmp(idx, t, op=tla.simd.CMP.LT)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(ob[0, 0], tla.simd.vdupv(x, pos=tla.simd.POS.LOWEST))
        tla.simd.vsts(ob[1, 0], tla.simd.vdupv(x, pos=tla.simd.POS.HIGHEST))
        tla.simd.vsts(ob[2, 0], tla.simd.vdupv(x, mask=prefix, pos=tla.simd.POS.LOWEST))
        tla.simd.vsts(ob[3, 0], tla.simd.vdupv(x, mask=prefix, pos=tla.simd.POS.HIGHEST))
        tla.simd.vsts(ob[4, 0], tla.simd.vdupv(x, mask=arb, pos=tla.simd.POS.LOWEST))
        tla.simd.vsts(ob[5, 0], tla.simd.vdupv(x, mask=arb, pos=tla.simd.POS.HIGHEST))
        tla.simd.vsts(ob[6, 0], tla.simd.vdupv(x, mask=none, pos=tla.simd.POS.LOWEST))
        tla.simd.vsts(ob[7, 0], tla.simd.vdupv(x, mask=none, pos=tla.simd.POS.HIGHEST))
        tla.simd.vsts(ob[8, 0], tla.simd.vdupv(x, mask=prefix, pos=tla.simd.POS.LOWEST, mode=tla.simd.MODE.MERGING))
        tla.simd.vsts(ob[9, 0], tla.simd.vdupv(x, mask=prefix, pos=tla.simd.POS.HIGHEST, mode=tla.simd.MODE.MERGING))
    tla.copy(ob, tla.view(O, (10, N)))


@tla.jit
def vdupv_wide(I, U, T, OI, OU, N: tl.constexpr, ACTIVE: tl.constexpr):
    ib = tla.alloc_buffer((N, ), tl.int64, space="ub")
    ub = tla.alloc_buffer((N, ), tl.uint64, space="ub")
    tb = tla.alloc_buffer((N, ), tl.int32, space="ub")
    oi = tla.alloc_buffer((8, N), tl.int64, space="ub")
    ou = tla.alloc_buffer((8, N), tl.uint64, space="ub")
    tla.copy(tla.view(I, (N, )), ib)
    tla.copy(tla.view(U, (N, )), ub)
    tla.copy(tla.view(T, (N, )), tb)
    with tla.scope.simd():
        iv = tla.simd.vlds(ib[0])
        uv = tla.simd.vlds(ub[0])
        t = tla.simd.vlds(tb[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        # Signed i32 lane index vs per-lane threshold: no unsigned 64-bit compare.
        arb = tla.simd.vcmp(idx, t, op=tla.simd.CMP.LT)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(oi[0, 0], tla.simd.vdupv(iv, pos=tla.simd.POS.LOWEST), dist="NORM_B64")
        tla.simd.vsts(oi[1, 0], tla.simd.vdupv(iv, pos=tla.simd.POS.HIGHEST), dist="NORM_B64")
        tla.simd.vsts(oi[2, 0], tla.simd.vdupv(iv, mask=prefix, pos=tla.simd.POS.LOWEST), dist="NORM_B64")
        tla.simd.vsts(oi[3, 0], tla.simd.vdupv(iv, mask=prefix, pos=tla.simd.POS.HIGHEST), dist="NORM_B64")
        tla.simd.vsts(oi[4, 0], tla.simd.vdupv(iv, mask=arb, pos=tla.simd.POS.LOWEST), dist="NORM_B64")
        tla.simd.vsts(oi[5, 0], tla.simd.vdupv(iv, mask=arb, pos=tla.simd.POS.HIGHEST), dist="NORM_B64")
        tla.simd.vsts(oi[6, 0], tla.simd.vdupv(iv, mask=none, pos=tla.simd.POS.LOWEST), dist="NORM_B64")
        tla.simd.vsts(oi[7, 0], tla.simd.vdupv(iv, mask=none, pos=tla.simd.POS.HIGHEST), dist="NORM_B64")
        tla.simd.vsts(ou[0, 0], tla.simd.vdupv(uv, pos=tla.simd.POS.LOWEST), dist="NORM_B64")
        tla.simd.vsts(ou[1, 0], tla.simd.vdupv(uv, pos=tla.simd.POS.HIGHEST), dist="NORM_B64")
        tla.simd.vsts(ou[2, 0], tla.simd.vdupv(uv, mask=prefix, pos=tla.simd.POS.LOWEST), dist="NORM_B64")
        tla.simd.vsts(ou[3, 0], tla.simd.vdupv(uv, mask=prefix, pos=tla.simd.POS.HIGHEST), dist="NORM_B64")
        tla.simd.vsts(ou[4, 0], tla.simd.vdupv(uv, mask=arb, pos=tla.simd.POS.LOWEST), dist="NORM_B64")
        tla.simd.vsts(ou[5, 0], tla.simd.vdupv(uv, mask=arb, pos=tla.simd.POS.HIGHEST), dist="NORM_B64")
        tla.simd.vsts(ou[6, 0], tla.simd.vdupv(uv, mask=none, pos=tla.simd.POS.LOWEST), dist="NORM_B64")
        tla.simd.vsts(ou[7, 0], tla.simd.vdupv(uv, mask=none, pos=tla.simd.POS.HIGHEST), dist="NORM_B64")
    tla.copy(oi, tla.view(OI, (8, N)))
    tla.copy(ou, tla.view(OU, (8, N)))


def constexprs():
    return {"N": SHAPE["lanes"], "ACTIVE": SHAPE["active"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vdupv, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.vector_broadcast") == 10
    assert text.count("ave.hir.vsel") == 2, "only the two MERGING rows add vsel"
    assert "ave.hir.pge <VL32>" in text
    # The two MERGING rows add a +0 ``scalar_broadcast`` merge source; the ten
    # vector_broadcast ops above prove vdupv is a lane (not scalar) broadcast.
    assert text.count("ave.hir.scalar_broadcast") == 2, "only the MERGING merge source is scalar_broadcast"
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=2, stores=1, buffers=None, extra=("ave.hir.vector_broadcast", ))

    module, ctx = compile_offline(vdupv_wide, WIDE_SIGNATURE, constexprs())
    wtext = str(module)
    assert wtext.count("ave.hir.vector_broadcast") == 16
    assert "vector<64xi64>" in wtext, "i64/u64 must use real 64-lane registers"
    assert "vbitcast" not in wtext, "no 32-lane bitcast hack for the wide path"
    wtext = str(lower_offline(module, ctx))
    check_lowered_ir(wtext, loads=3, stores=2, buffers=None, extra=("ave.hir.vector_broadcast", "vector<64xi64>"))
    print("IR PASS: vdupv LOWEST/HIGHEST on f32 plus real 64-lane i64/u64")


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


def _rows(src, mask):
    low = torch.full_like(src, src[0])
    high = torch.full_like(src, src[-1])
    if mask is not None:
        low, high = _apply(low, mask), _apply(high, mask)
    return low, high


def _to_npu_u64(arr):
    return torch.from_numpy(np.ascontiguousarray(arr).view(np.int64)).view(torch.uint64).npu()


def _launch():
    n = SHAPE["lanes"]
    x = torch.randn(n)
    masks, thr = _masks()
    out = torch.empty((SHAPE["rows"], n), dtype=torch.float32, device="npu")
    vdupv[(1, )](x.npu(), thr.to(torch.int32).npu(), out, **constexprs())
    torch.npu.synchronize()
    rows = []
    for mask in (None, masks["prefix"], masks["arb"], masks["none"], masks["prefix"]):
        low, high = _rows(x, mask)
        rows += [low, high]
    ref = torch.stack(rows)

    iv = torch.tensor([((i * 7919) % (2**62)) * ((-1)**i) for i in range(n)], dtype=torch.int64)
    uv_np = np.array([(i * 2654435761 + 3 * 2**62) % 2**64 for i in range(n)], dtype=np.uint64)
    uv = torch.from_numpy(uv_np.view(np.int64)).view(torch.uint64)
    oi = torch.empty((8, n), dtype=torch.int64, device="npu")
    ou = torch.empty((8, n), dtype=torch.uint64, device="npu")
    vdupv_wide[(1, )](iv.npu(), _to_npu_u64(uv_np), thr.to(torch.int32).npu(), oi, ou, **constexprs())
    torch.npu.synchronize()
    irows, urows = [], []
    for mask in (None, masks["prefix"], masks["arb"], masks["none"]):
        for src, acc in ((iv, irows), (uv, urows)):
            low, high = _rows(src, mask)
            acc += [low, high]
    return (out.cpu(), ref, masks), (oi.cpu(), torch.stack(irows)), (ou.cpu(), torch.stack(urows))


def _bitwise_zero(t):
    if t.dtype == torch.float32:
        return bool((t.view(torch.int32) == 0).all())
    if t.dtype == torch.uint64:
        return bool((t.view(torch.int64) == 0).all())
    return bool((t == 0).all())


F32_ROW_MASKS = (None, None, "prefix", "prefix", "arb", "arb", "none", "none", "prefix", "prefix")
WIDE_ROW_MASKS = (None, None, "prefix", "prefix", "arb", "arb", "none", "none")


def _inactive_zero(got, masks, names):
    assert len(names) == got.shape[0]
    for row, name in enumerate(names):
        if name is None:
            continue
        keep = masks[name]
        inactive = got[row][~keep] if keep.any() else got[row]
        assert _bitwise_zero(inactive), f"row {row} ({name}) has a non-zero inactive lane"


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vdupv generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    (got, ref, masks), (igot, iref), (ugot, uref) = _launch()
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    _inactive_zero(got, masks, F32_ROW_MASKS)
    torch.testing.assert_close(igot, iref, rtol=0, atol=0)
    torch.testing.assert_close(ugot.view(torch.int64), uref.view(torch.int64), rtol=0, atol=0)
    _inactive_zero(igot, masks, WIDE_ROW_MASKS)
    _inactive_zero(ugot, masks, WIDE_ROW_MASKS)
    print("VDUPV OK: LOWEST/HIGHEST, masks/modes, 64-lane i64/u64 with signed-index masks")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vdupv", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
