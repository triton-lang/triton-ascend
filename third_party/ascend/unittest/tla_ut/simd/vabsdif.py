"""TLA 3-in-1 test: vabsdif with f32/f16, six masks, and a tail tile.

Rows 0..5 use one full register with default / prefix / arbitrary / all-false
masks plus ZEROING / MERGING. Rows 6.. are tiled registers; the last tile is
8 lanes and its masked store keeps the sentinel padding.

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from _simd_ext_common import MASK_ROWS, TAIL_BASE, MASK_ROWS_C, TAIL_BASE_C, masks, ref_rows, inactive_zero, padded_binary
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

SHAPE = dict(lanes=64, active=32, tiles=3, total=136)
SHAPE["tail"] = SHAPE["total"] - (SHAPE["tiles"] - 1) * SHAPE["lanes"]
SHAPE["rows"] = MASK_ROWS + SHAPE["tiles"]
SHAPE16 = dict(lanes=128, active=64, tiles=2, total=136)
SHAPE16["tail"] = SHAPE16["total"] - (SHAPE16["tiles"] - 1) * SHAPE16["lanes"]
SHAPE16["rows"] = MASK_ROWS + SHAPE16["tiles"]
SENTINEL_F = -123.0
SENTINEL_FC = tl.constexpr(SENTINEL_F)
SIGNATURE_F = {"X": "*fp32", "Y": "*fp32", "T": "*i32", "O": "*fp32"}
SIGNATURE_H = {"X": "*fp16", "Y": "*fp16", "I": "*i16", "T": "*i16", "O": "*fp16"}


@tla.jit
def vabsdif_f32(X, Y, T, O, N: tl.constexpr, ACTIVE: tl.constexpr, TILES: tl.constexpr, TAIL: tl.constexpr):
    xb = tla.alloc_buffer((TILES, N), tl.float32, space="ub")
    yb = tla.alloc_buffer((TILES, N), tl.float32, space="ub")
    tb = tla.alloc_buffer((N, ), tl.int32, space="ub")
    ob = tla.alloc_buffer((MASK_ROWS_C + TILES, N), tl.float32, space="ub")
    tla.copy(tla.view(X, (TILES, N)), xb)
    tla.copy(tla.view(Y, (TILES, N)), yb)
    tla.copy(tla.view(T, (N, )), tb)
    with tla.scope.simd():
        x = tla.simd.vlds(xb[0, 0])
        y = tla.simd.vlds(yb[0, 0])
        t = tla.simd.vlds(tb[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        arb = tla.simd.vcmp(idx, t, op=tla.simd.CMP.LT)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(ob[0, 0], tla.simd.vabsdif(x, y))
        tla.simd.vsts(ob[1, 0], tla.simd.vabsdif(x, y, mask=prefix))
        tla.simd.vsts(ob[2, 0], tla.simd.vabsdif(x, y, mask=arb))
        tla.simd.vsts(ob[3, 0], tla.simd.vabsdif(x, y, mask=none))
        tla.simd.vsts(ob[4, 0], tla.simd.vabsdif(x, y, mask=prefix, mode=tla.simd.MODE.MERGING))
        tla.simd.vsts(ob[5, 0], tla.simd.vabsdif(x, y, mask=arb, mode=tla.simd.MODE.MERGING))
        tla.simd.vsts(ob[TAIL_BASE_C + TILES - 1, 0], tla.simd.vsplat(SENTINEL_FC, tl.float32))
        for tile in tl.static_range(TILES):
            vx = tla.simd.vlds(xb[tile, 0])
            vy = tla.simd.vlds(yb[tile, 0])
            if tile == TILES - 1:
                tail_mask = tla.simd.pge(TAIL, N)
                tla.simd.vsts(ob[TAIL_BASE_C + tile, 0], tla.simd.vabsdif(vx, vy, mask=tail_mask), mask=tail_mask)
            else:
                tla.simd.vsts(ob[TAIL_BASE_C + tile, 0], tla.simd.vabsdif(vx, vy))
    tla.copy(ob, tla.view(O, (MASK_ROWS_C + TILES, N)))


@tla.jit
def vabsdif_f16(X, Y, I, T, O, N: tl.constexpr, ACTIVE: tl.constexpr, TILES: tl.constexpr, TAIL: tl.constexpr):
    xb = tla.alloc_buffer((TILES, N), tl.float16, space="ub")
    yb = tla.alloc_buffer((TILES, N), tl.float16, space="ub")
    ib = tla.alloc_buffer((N, ), tl.int16, space="ub")
    tb = tla.alloc_buffer((N, ), tl.int16, space="ub")
    ob = tla.alloc_buffer((MASK_ROWS_C + TILES, N), tl.float16, space="ub")
    tla.copy(tla.view(X, (TILES, N)), xb)
    tla.copy(tla.view(Y, (TILES, N)), yb)
    tla.copy(tla.view(I, (N, )), ib)
    tla.copy(tla.view(T, (N, )), tb)
    with tla.scope.simd():
        x = tla.simd.vlds(xb[0, 0])
        y = tla.simd.vlds(yb[0, 0])
        idx = tla.simd.vlds(ib[0])
        t = tla.simd.vlds(tb[0])
        prefix = tla.simd.pge(ACTIVE, N)
        arb = tla.simd.vcmp(idx, t, op=tla.simd.CMP.LT)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(ob[0, 0], tla.simd.vabsdif(x, y), dist="NORM_B16")
        tla.simd.vsts(ob[1, 0], tla.simd.vabsdif(x, y, mask=prefix), dist="NORM_B16")
        tla.simd.vsts(ob[2, 0], tla.simd.vabsdif(x, y, mask=arb), dist="NORM_B16")
        tla.simd.vsts(ob[3, 0], tla.simd.vabsdif(x, y, mask=none), dist="NORM_B16")
        tla.simd.vsts(ob[4, 0], tla.simd.vabsdif(x, y, mask=prefix, mode=tla.simd.MODE.MERGING), dist="NORM_B16")
        tla.simd.vsts(ob[5, 0], tla.simd.vabsdif(x, y, mask=arb, mode=tla.simd.MODE.MERGING), dist="NORM_B16")
        tla.simd.vsts(ob[TAIL_BASE_C + TILES - 1, 0], tla.simd.vsplat(SENTINEL_FC, tl.float16), dist="NORM_B16")
        for tile in tl.static_range(TILES):
            vx = tla.simd.vlds(xb[tile, 0])
            vy = tla.simd.vlds(yb[tile, 0])
            if tile == TILES - 1:
                tail_mask = tla.simd.pge(TAIL, N)
                tla.simd.vsts(ob[TAIL_BASE_C + tile, 0], tla.simd.vabsdif(vx, vy, mask=tail_mask), mask=tail_mask,
                              dist="NORM_B16")
            else:
                tla.simd.vsts(ob[TAIL_BASE_C + tile, 0], tla.simd.vabsdif(vx, vy), dist="NORM_B16")
    tla.copy(ob, tla.view(O, (MASK_ROWS_C + TILES, N)))


def constexprs(shape=None):
    s = SHAPE if shape is None else shape
    return {"N": s["lanes"], "ACTIVE": s["active"], "TILES": s["tiles"], "TAIL": s["tail"]}


def test_ir(output_dir=None):
    for fn, sig, shape, loads in ((vabsdif_f32, SIGNATURE_F, SHAPE, 3), (vabsdif_f16, SIGNATURE_H, SHAPE16, 4)):
        module, ctx = compile_offline(fn, sig, constexprs(shape), output_dir=output_dir)
        text = str(module)
        assert text.count("ave.hir.vabs_diff") == MASK_ROWS + shape["tiles"]
        assert text.count("ave.hir.vsel") == 2, "two MERGING rows add vsel"
        assert f"ave.hir.pge <VL{shape['active']}>" in text
        assert "ave.hir.pge <VL8>" in text, "tail predicate must select 8 lanes"
        text = str(lower_offline(module, ctx, output_dir=output_dir))
        check_lowered_ir(text, loads=loads, stores=1, buffers=None, extra=("ave.hir.vabs_diff", ))
    print("IR PASS: vabsdif f32/f16 six masks plus full and tail tiles")


def _launch_f32():
    s = SHAPE
    x = torch.randn((s["tiles"], s["lanes"])) * 2.0
    y = torch.randn((s["tiles"], s["lanes"])) * 2.0
    mask_set, thr = masks(s["lanes"], s["active"])
    out = torch.empty((s["rows"], s["lanes"]), dtype=torch.float32, device="npu")
    vabsdif_f32[(1, )](x.npu(), y.npu(), thr.to(torch.int32).npu(), out, **constexprs())
    torch.npu.synchronize()
    ref = torch.cat([
        ref_rows(torch.abs(x[0] - y[0]), mask_set),
        padded_binary(lambda a, b: torch.abs(a - b), x, y, s["tiles"], s["lanes"], s["total"], SENTINEL_F)
    ])
    return out.cpu(), ref, mask_set


def _launch_f16():
    s = SHAPE16
    x = (torch.randn((s["tiles"], s["lanes"])) * 2.0).to(torch.float16)
    y = (torch.randn((s["tiles"], s["lanes"])) * 2.0).to(torch.float16)
    idx = torch.arange(s["lanes"], dtype=torch.int16)
    thr = ((torch.arange(s["lanes"]) % 5) * (s["lanes"] // 5)).to(torch.int16)
    mask_set = {
        "prefix": torch.arange(s["lanes"]) < s["active"],
        "arb": torch.arange(s["lanes"]) < thr.to(torch.int64),
        "none": torch.zeros(s["lanes"], dtype=torch.bool),
    }
    out = torch.empty((s["rows"], s["lanes"]), dtype=torch.float16, device="npu")
    vabsdif_f16[(1, )](x.npu(), y.npu(), idx.npu(), thr.npu(), out, **constexprs(s))
    torch.npu.synchronize()

    def absdiff_half(a, b):
        return torch.abs(a.float() - b.float()).to(torch.float16)

    ref = torch.cat([
        ref_rows(absdiff_half(x[0], y[0]), mask_set),
        padded_binary(absdiff_half, x, y, s["tiles"], s["lanes"], s["total"], SENTINEL_F, dtype=torch.float16)
    ])
    return out.cpu(), ref, mask_set


def test_compile():
    _launch_f32()
    _launch_f16()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vabsdif generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    got, ref, mask_set = _launch_f32()
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)
    inactive_zero(got, mask_set)
    hgot, href, hmask_set = _launch_f16()
    torch.testing.assert_close(hgot, href, rtol=5e-3, atol=1e-3)
    inactive_zero(hgot, hmask_set)
    print("VABSDIF OK: f32/f16, six masks, full and tail tiles")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vabsdif", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
