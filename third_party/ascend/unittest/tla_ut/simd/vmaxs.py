"""TLA 3-in-1 test: vmaxs masks/modes plus exact 32/64-bit scalar bounds.

f32 kernel rows (64 lanes each):

  0 vmaxs(x, 0.0)                    default ALL mask, MODE_ZEROING
  1 vmaxs(x, 0.0, prefix)            first ACTIVE lanes, MODE_ZEROING
  2 vmaxs(x, 0.0, arb)               per-lane threshold from ``T`` (signed i32 index)
  3 vmaxs(x, 0.0, none)              all-false predicate
  4 vmaxs(x, 0.0, prefix, MERGING)   merge source is ``+0``
  5 vmaxs(x, 0.0, arb, MERGING)

Scalar-bound kernel (one register per row, three scalars per dtype):
``u32`` 0x80000000 / 0x80000001 / 0xFFFFFFFF, ``i64`` 2**53 / 2**53+1 / 2**63-1
and ``u64`` 2**63 / 2**63+1 / 2**64-1.  The constexpr path must emit exact
integer constants (no Python double round-trip): 2**53+1 stays exact and the
unsigned 64-bit values travel as two's-complement i64 bits.  Row 1 of each
group is prefix-masked, row 2 uses the all-false predicate.

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

SHAPE = dict(lanes=64, active=32, rows=6)
SIGNATURE = {"X": "*fp32", "T": "*i32", "O": "*fp32"}
BOUND_SIGNATURE = {"I": "*i64", "U": "*u64", "W": "*u32", "OI": "*i64", "OU": "*u64", "OW": "*u32"}
I_SCALARS = (2**53, 2**53 + 1, 2**63 - 1)
U_SCALARS = (2**63, 2**63 + 1, 2**64 - 1)
W_SCALARS = (0x80000000, 0x80000001, 0xFFFFFFFF)


@tla.jit
def vmaxs(X, T, O, N: tl.constexpr, ACTIVE: tl.constexpr):
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
        tla.simd.vsts(ob[0, 0], tla.simd.vmaxs(x, 0.0, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[1, 0], tla.simd.vmaxs(x, 0.0, mask=prefix, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[2, 0], tla.simd.vmaxs(x, 0.0, mask=arb, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[3, 0], tla.simd.vmaxs(x, 0.0, mask=none, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[4, 0], tla.simd.vmaxs(x, 0.0, mask=prefix, mode=tla.simd.MODE.MERGING))
        tla.simd.vsts(ob[5, 0], tla.simd.vmaxs(x, 0.0, mask=arb, mode=tla.simd.MODE.MERGING))
    tla.copy(ob, tla.view(O, (6, N)))


@tla.jit
def vmaxs_bounds(I, U, W, OI, OU, OW, N: tl.constexpr, ACTIVE: tl.constexpr):
    ib = tla.alloc_buffer((N, ), tl.int64, space="ub")
    ub = tla.alloc_buffer((N, ), tl.uint64, space="ub")
    wb = tla.alloc_buffer((N, ), tl.uint32, space="ub")
    oi = tla.alloc_buffer((3, N), tl.int64, space="ub")
    ou = tla.alloc_buffer((3, N), tl.uint64, space="ub")
    ow = tla.alloc_buffer((3, N), tl.uint32, space="ub")
    tla.copy(tla.view(I, (N, )), ib)
    tla.copy(tla.view(U, (N, )), ub)
    tla.copy(tla.view(W, (N, )), wb)
    with tla.scope.simd():
        iv = tla.simd.vlds(ib[0])
        uv = tla.simd.vlds(ub[0])
        wv = tla.simd.vlds(wb[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(oi[0, 0], tla.simd.vmaxs(iv, 2**53), dist="NORM_B64")
        tla.simd.vsts(oi[1, 0], tla.simd.vmaxs(iv, 2**53 + 1, mask=prefix), dist="NORM_B64")
        tla.simd.vsts(oi[2, 0], tla.simd.vmaxs(iv, 2**63 - 1, mask=none), dist="NORM_B64")
        tla.simd.vsts(ou[0, 0], tla.simd.vmaxs(uv, 2**63), dist="NORM_B64")
        tla.simd.vsts(ou[1, 0], tla.simd.vmaxs(uv, 2**63 + 1, mask=prefix), dist="NORM_B64")
        tla.simd.vsts(ou[2, 0], tla.simd.vmaxs(uv, 2**64 - 1, mask=none), dist="NORM_B64")
        tla.simd.vsts(ow[0, 0], tla.simd.vmaxs(wv, 0x80000000))
        tla.simd.vsts(ow[1, 0], tla.simd.vmaxs(wv, 0x80000001, mask=prefix))
        tla.simd.vsts(ow[2, 0], tla.simd.vmaxs(wv, 0xFFFFFFFF, mask=none))
    tla.copy(oi, tla.view(OI, (3, N)))
    tla.copy(ou, tla.view(OU, (3, N)))
    tla.copy(ow, tla.view(OW, (3, N)))


def constexprs():
    return {"N": SHAPE["lanes"], "ACTIVE": SHAPE["active"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vmaxs, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.vmaxs") == 6
    assert text.count("ave.hir.vsel") == 2, "only the two MERGING rows add vsel"
    assert "ave.hir.pge <VL32>" in text
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=2, stores=1, buffers=None, extra=("ave.hir.vmaxs", ))

    module, ctx = compile_offline(vmaxs_bounds, BOUND_SIGNATURE, constexprs())
    btext = str(module)
    assert btext.count("ave.hir.vmaxs") == 3, "signed i64 uses vmaxs"
    assert btext.count("ave.hir.vumaxs") == 6, "u64 and u32 must select vumaxs"
    assert "9007199254740993 : i64" in btext, "2**53+1 must stay an exact i64 constant"
    assert "-9223372036854775808 : i64" in btext, "u64 2**63 travels as two's-complement bits"
    assert "-1 : i64" in btext, "u64 2**64-1 travels as two's-complement bits"
    btext = str(lower_offline(module, ctx))
    check_lowered_ir(btext, loads=3, stores=3, buffers=None, extra=("ave.hir.vmaxs", "ave.hir.vumaxs"))
    print("IR PASS: vmaxs masks/modes plus exact i64/u64/u32 scalar bounds")


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


def _to_npu_u64(arr):
    return torch.from_numpy(np.ascontiguousarray(arr).view(np.int64)).view(torch.uint64).npu()


def _launch():
    n = SHAPE["lanes"]
    x = torch.randn(n) * 2.0
    masks, thr = _masks()
    out = torch.empty((SHAPE["rows"], n), dtype=torch.float32, device="npu")
    vmaxs[(1, )](x.npu(), thr.to(torch.int32).npu(), out, **constexprs())
    torch.npu.synchronize()
    v = torch.maximum(x, torch.zeros_like(x))
    ref = torch.stack([
        v,
        _apply(v, masks["prefix"]),
        _apply(v, masks["arb"]),
        _apply(v, masks["none"]),
        _apply(v, masks["prefix"]),
        _apply(v, masks["arb"]),
    ])

    iv = torch.tensor([(i * 7919) % (2**62) * ((-1)**i) for i in range(n)], dtype=torch.int64)
    uv_np = np.array([(i * 2654435761 + 2**62) % 2**64 for i in range(n)], dtype=np.uint64)
    wv = torch.tensor([(i * 2654435761) % (1 << 32) for i in range(n)], dtype=torch.int64).to(torch.uint32)

    oi = torch.empty((3, n), dtype=torch.int64, device="npu")
    ou = torch.empty((3, n), dtype=torch.uint64, device="npu")
    ow = torch.empty((3, n), dtype=torch.uint32, device="npu")
    vmaxs_bounds[(1, )](iv.npu(), _to_npu_u64(uv_np), wv.npu(), oi, ou, ow, **constexprs())
    torch.npu.synchronize()

    irows = []
    for k, s in enumerate(I_SCALARS):
        row = torch.maximum(iv, torch.tensor(s, dtype=torch.int64))
        irows.append(_apply(row, masks["prefix"]) if k == 1 else (_apply(row, masks["none"]) if k == 2 else row))
    urows = []
    for k, s in enumerate(U_SCALARS):
        row = np.maximum(uv_np, np.uint64(s))
        row = torch.from_numpy(row.view(np.int64)).view(torch.uint64)
        urows.append(_apply(row, masks["prefix"]) if k == 1 else (_apply(row, masks["none"]) if k == 2 else row))
    wrows = []
    for k, s in enumerate(W_SCALARS):
        row = torch.maximum(wv.to(torch.int64), torch.tensor(s, dtype=torch.int64)).to(torch.uint32)
        wrows.append(_apply(row, masks["prefix"]) if k == 1 else (_apply(row, masks["none"]) if k == 2 else row))
    return (out.cpu(), ref, masks), (oi.cpu(), torch.stack(irows), None), \
        (ou.cpu(), torch.stack(urows), None), (ow.cpu(), torch.stack(wrows), None)


def _bitwise_zero(t):
    if t.dtype == torch.float32:
        return bool((t.view(torch.int32) == 0).all())
    if t.dtype == torch.float64:
        return bool((t.view(torch.int64) == 0).all())
    if t.dtype == torch.uint64:
        return bool((t.view(torch.int64) == 0).all())
    if t.dtype == torch.int64:
        return bool((t == 0).all())
    return bool((t.view(torch.int32) == 0).all())


def _inactive_zero(got, masks):
    for row, name in ((1, "prefix"), (2, "arb"), (3, "none"), (4, "prefix"), (5, "arb")):
        keep = masks[name]
        inactive = got[row][~keep] if keep.any() else got[row]
        assert _bitwise_zero(inactive), f"row {row} ({name}) has a non-zero inactive lane"


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vmaxs generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    (got, ref, masks), (igot, iref, _), (ugot, uref, _), (wgot, wref, _) = _launch()
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    _inactive_zero(got, masks)
    torch.testing.assert_close(igot, iref, rtol=0, atol=0)
    torch.testing.assert_close(ugot.view(torch.int64), uref.view(torch.int64), rtol=0, atol=0)
    torch.testing.assert_close(wgot.view(torch.int32), wref.view(torch.int32), rtol=0, atol=0)
    # The masked bound rows must keep their inactive lanes at +0 as well.
    assert _bitwise_zero(igot[1][~masks["prefix"]]), "i64 prefix row leaked an inactive lane"
    assert _bitwise_zero(wgot[2]), "u32 all-false row must be +0"
    print("VMAXS OK: masks/modes plus exact i64/u64/u32 scalar bounds")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vmaxs", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
