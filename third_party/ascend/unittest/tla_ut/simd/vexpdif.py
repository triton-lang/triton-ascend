"""TLA 3-in-1 test: vexpdif f32, six masks, and a tail tile.

Rows 0..5 use one full register with default / prefix / arbitrary / all-false
masks plus ZEROING / MERGING. Rows 6.. are three tiled registers; the last tile
is 8/64 lanes and its masked store keeps the sentinel padding.

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
SENTINEL_F = -123.0
SENTINEL_FC = tl.constexpr(SENTINEL_F)
SIGNATURE = {"X": "*fp32", "Y": "*fp32", "T": "*i32", "O": "*fp32"}


@tla.jit
def vexpdif_f32(X, Y, T, O, N: tl.constexpr, ACTIVE: tl.constexpr, TILES: tl.constexpr, TAIL: tl.constexpr):
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
        tla.simd.vsts(ob[0, 0], tla.simd.vexpdif(x, y))
        tla.simd.vsts(ob[1, 0], tla.simd.vexpdif(x, y, mask=prefix))
        tla.simd.vsts(ob[2, 0], tla.simd.vexpdif(x, y, mask=arb))
        tla.simd.vsts(ob[3, 0], tla.simd.vexpdif(x, y, mask=none))
        tla.simd.vsts(ob[4, 0], tla.simd.vexpdif(x, y, mask=prefix, mode=tla.simd.MODE.MERGING))
        tla.simd.vsts(ob[5, 0], tla.simd.vexpdif(x, y, mask=arb, mode=tla.simd.MODE.MERGING))
        tla.simd.vsts(ob[TAIL_BASE_C + TILES - 1, 0], tla.simd.vsplat(SENTINEL_FC, tl.float32))
        for tile in tl.static_range(TILES):
            vx = tla.simd.vlds(xb[tile, 0])
            vy = tla.simd.vlds(yb[tile, 0])
            if tile == TILES - 1:
                tail_mask = tla.simd.pge(TAIL, N)
                tla.simd.vsts(ob[TAIL_BASE_C + tile, 0], tla.simd.vexpdif(vx, vy, mask=tail_mask), mask=tail_mask)
            else:
                tla.simd.vsts(ob[TAIL_BASE_C + tile, 0], tla.simd.vexpdif(vx, vy))
    tla.copy(ob, tla.view(O, (MASK_ROWS_C + TILES, N)))


def constexprs():
    return {"N": SHAPE["lanes"], "ACTIVE": SHAPE["active"], "TILES": SHAPE["tiles"], "TAIL": SHAPE["tail"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vexpdif_f32, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.vexpdif") == MASK_ROWS + SHAPE["tiles"]
    assert text.count("ave.hir.vsel") == 2, "two MERGING rows add vsel"
    assert "ave.hir.pge <VL32>" in text
    assert "ave.hir.pge <VL8>" in text, "tail predicate must select 8 lanes"
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=3, stores=1, buffers=None, extra=("ave.hir.vexpdif", ))
    print("IR PASS: vexpdif f32 six masks plus 2 full + 1 tail tile")


def _launch():
    s = SHAPE
    x = torch.randn((s["tiles"], s["lanes"])) * 0.5
    y = torch.randn((s["tiles"], s["lanes"])) * 0.5
    mask_set, thr = masks(s["lanes"], s["active"])
    out = torch.empty((s["rows"], s["lanes"]), dtype=torch.float32, device="npu")
    vexpdif_f32[(1, )](x.npu(), y.npu(), thr.to(torch.int32).npu(), out, **constexprs())
    torch.npu.synchronize()
    ref = torch.cat([
        ref_rows(torch.exp(x[0] - y[0]), mask_set),
        padded_binary(lambda a, b: torch.exp(a - b), x, y, s["tiles"], s["lanes"], s["total"], SENTINEL_F)
    ])
    return out.cpu(), ref, mask_set


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vexpdif generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    got, ref, mask_set = _launch()
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)
    inactive_zero(got, mask_set)
    print("VEXPDIF OK: f32, six masks, full and tail tiles")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vexpdif", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
