"""TLA 3-in-1 test: vabs with f32/i32, six masks, and a tail tile.

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
from _simd_ext_common import MASK_ROWS, TAIL_BASE, MASK_ROWS_C, TAIL_BASE_C, masks, ref_rows, inactive_zero, padded_unary
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

SHAPE = dict(lanes=64, active=32, tiles=3, total=136)
SHAPE["tail"] = SHAPE["total"] - (SHAPE["tiles"] - 1) * SHAPE["lanes"]
SHAPE["rows"] = MASK_ROWS + SHAPE["tiles"]
SENTINEL_F = -123.0
SENTINEL_I = 0x5A5A5A5A
SENTINEL_FC = tl.constexpr(SENTINEL_F)
SENTINEL_IC = tl.constexpr(SENTINEL_I)
SIGNATURE_F = {"X": "*fp32", "T": "*i32", "O": "*fp32"}
SIGNATURE_I = {"X": "*i32", "T": "*i32", "O": "*i32"}


@tla.jit
def vabs_f32(X, T, O, N: tl.constexpr, ACTIVE: tl.constexpr, TILES: tl.constexpr, TAIL: tl.constexpr):
    xb = tla.alloc_buffer((TILES, N), tl.float32, space="ub")
    tb = tla.alloc_buffer((N, ), tl.int32, space="ub")
    ob = tla.alloc_buffer((MASK_ROWS_C + TILES, N), tl.float32, space="ub")
    tla.copy(tla.view(X, (TILES, N)), xb)
    tla.copy(tla.view(T, (N, )), tb)
    with tla.scope.simd():
        x = tla.simd.vlds(xb[0, 0])
        t = tla.simd.vlds(tb[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        arb = tla.simd.vcmp(idx, t, op=tla.simd.CMP.LT)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(ob[0, 0], tla.simd.vabs(x))
        tla.simd.vsts(ob[1, 0], tla.simd.vabs(x, mask=prefix))
        tla.simd.vsts(ob[2, 0], tla.simd.vabs(x, mask=arb))
        tla.simd.vsts(ob[3, 0], tla.simd.vabs(x, mask=none))
        tla.simd.vsts(ob[4, 0], tla.simd.vabs(x, mask=prefix, mode=tla.simd.MODE.MERGING))
        tla.simd.vsts(ob[5, 0], tla.simd.vabs(x, mask=arb, mode=tla.simd.MODE.MERGING))
        tla.simd.vsts(ob[TAIL_BASE_C + TILES - 1, 0], tla.simd.vsplat(SENTINEL_FC, tl.float32))
        for tile in tl.static_range(TILES):
            v = tla.simd.vlds(xb[tile, 0])
            if tile == TILES - 1:
                tail_mask = tla.simd.pge(TAIL, N)
                tla.simd.vsts(ob[TAIL_BASE_C + tile, 0], tla.simd.vabs(v, mask=tail_mask), mask=tail_mask)
            else:
                tla.simd.vsts(ob[TAIL_BASE_C + tile, 0], tla.simd.vabs(v))
    tla.copy(ob, tla.view(O, (MASK_ROWS_C + TILES, N)))


@tla.jit
def vabs_i32(X, T, O, N: tl.constexpr, ACTIVE: tl.constexpr, TILES: tl.constexpr, TAIL: tl.constexpr):
    xb = tla.alloc_buffer((TILES, N), tl.int32, space="ub")
    tb = tla.alloc_buffer((N, ), tl.int32, space="ub")
    ob = tla.alloc_buffer((MASK_ROWS_C + TILES, N), tl.int32, space="ub")
    tla.copy(tla.view(X, (TILES, N)), xb)
    tla.copy(tla.view(T, (N, )), tb)
    with tla.scope.simd():
        x = tla.simd.vlds(xb[0, 0])
        t = tla.simd.vlds(tb[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        arb = tla.simd.vcmp(idx, t, op=tla.simd.CMP.LT)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(ob[0, 0], tla.simd.vabs(x))
        tla.simd.vsts(ob[1, 0], tla.simd.vabs(x, mask=prefix))
        tla.simd.vsts(ob[2, 0], tla.simd.vabs(x, mask=arb))
        tla.simd.vsts(ob[3, 0], tla.simd.vabs(x, mask=none))
        tla.simd.vsts(ob[4, 0], tla.simd.vabs(x, mask=prefix, mode=tla.simd.MODE.MERGING))
        tla.simd.vsts(ob[5, 0], tla.simd.vabs(x, mask=arb, mode=tla.simd.MODE.MERGING))
        tla.simd.vsts(ob[TAIL_BASE_C + TILES - 1, 0], tla.simd.vsplat(SENTINEL_IC, tl.int32))
        for tile in tl.static_range(TILES):
            v = tla.simd.vlds(xb[tile, 0])
            if tile == TILES - 1:
                tail_mask = tla.simd.pge(TAIL, N)
                tla.simd.vsts(ob[TAIL_BASE_C + tile, 0], tla.simd.vabs(v, mask=tail_mask), mask=tail_mask)
            else:
                tla.simd.vsts(ob[TAIL_BASE_C + tile, 0], tla.simd.vabs(v))
    tla.copy(ob, tla.view(O, (MASK_ROWS_C + TILES, N)))


def constexprs():
    return {"N": SHAPE["lanes"], "ACTIVE": SHAPE["active"], "TILES": SHAPE["tiles"], "TAIL": SHAPE["tail"]}


def test_ir(output_dir=None):
    for fn, sig in ((vabs_f32, SIGNATURE_F), (vabs_i32, SIGNATURE_I)):
        module, ctx = compile_offline(fn, sig, constexprs(), output_dir=output_dir)
        text = str(module)
        assert text.count("ave.hir.vabs") == MASK_ROWS + SHAPE["tiles"]
        assert text.count("ave.hir.vsel") == 2, "two MERGING rows add vsel"
        assert "ave.hir.pge <VL32>" in text
        assert "ave.hir.pge <VL8>" in text, "tail predicate must select 8 lanes"
        text = str(lower_offline(module, ctx, output_dir=output_dir))
        check_lowered_ir(text, loads=2, stores=1, buffers=None, extra=("ave.hir.vabs", ))
    print("IR PASS: vabs f32/i32 six masks plus 2 full + 1 tail tile")


def _launch_f32():
    s = SHAPE
    x = torch.randn((s["tiles"], s["lanes"])) * 2.0
    mask_set, thr = masks(s["lanes"], s["active"])
    out = torch.empty((s["rows"], s["lanes"]), dtype=torch.float32, device="npu")
    vabs_f32[(1, )](x.npu(), thr.to(torch.int32).npu(), out, **constexprs())
    torch.npu.synchronize()
    ref = torch.cat([
        ref_rows(torch.abs(x[0]), mask_set),
        padded_unary(torch.abs, x, s["tiles"], s["lanes"], s["total"], SENTINEL_F)
    ])
    return out.cpu(), ref, mask_set


def _launch_i32():
    s = SHAPE
    x = torch.randint(-(1 << 20), 1 << 20, (s["tiles"], s["lanes"]), dtype=torch.int32)
    mask_set, thr = masks(s["lanes"], s["active"])
    out = torch.empty((s["rows"], s["lanes"]), dtype=torch.int32, device="npu")
    vabs_i32[(1, )](x.npu(), thr.to(torch.int32).npu(), out, **constexprs())
    torch.npu.synchronize()
    ref = torch.cat([
        ref_rows(torch.abs(x[0]), mask_set),
        padded_unary(torch.abs, x, s["tiles"], s["lanes"], s["total"], SENTINEL_I, dtype=torch.int32)
    ])
    return out.cpu(), ref, mask_set


def test_compile():
    _launch_f32()
    _launch_i32()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vabs generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    for got, ref, mask_set in (_launch_f32(), _launch_i32()):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
        inactive_zero(got, mask_set)
    print("VABS OK: f32/i32, six masks, full and tail tiles")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vabs", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
