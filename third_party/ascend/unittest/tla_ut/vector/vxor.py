"""TLA 3-in-1 test: uint32 vxor exact bits, ZEROING, reductions and tail tile.

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).
Inputs are regenerated randomly on every run.
Rows 0/1 hold a single-tile full-mask and ZEROING result, rows 2/3 their
horizontal sums, and rows 4..7 a four-tile run whose last tile is 8/64 lanes.
The tail store is masked, so the tail row padding keeps its sentinel.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline,
                                 check_lowered_ir, check_compile_artifacts)

SHAPE = dict(lanes=64, active=32, tiles=4, total=200)
SHAPE["tail"] = SHAPE["total"] - (SHAPE["tiles"] - 1) * SHAPE["lanes"]
SHAPE["rows"] = 4 + SHAPE["tiles"]
SENTINEL = 0xDEADBEEF
SENTINEL_C = tl.constexpr(SENTINEL)
SIGNATURE = {name: "*u32" for name in ("X", "Y", "O")}


@tla.jit
def vxor(X, Y, O, N: tl.constexpr, ACTIVE: tl.constexpr,
          TILES: tl.constexpr, TAIL: tl.constexpr):
    x = tla.alloc_buffer((TILES, N), tl.uint32, space="ub")
    y = tla.alloc_buffer((TILES, N), tl.uint32, space="ub")
    o = tla.alloc_buffer((4 + TILES, N), tl.uint32, space="ub")
    tla.copy(tla.view(X, (TILES, N)), x)
    tla.copy(tla.view(Y, (TILES, N)), y)
    with tla.scope.simd():
        a = tla.simd.vlds(x[0, 0])
        b = tla.simd.vlds(y[0, 0])
        mask = tla.simd.pge(ACTIVE, N)
        full = tla.simd.vxor(a, b)
        zeroed = tla.simd.vxor(a, b, mask)
        tla.simd.vsts(o[0, 0], full)
        tla.simd.vsts(o[1, 0], zeroed)
        tla.simd.vsts(o[2, 0], tla.simd.vreduce(full, "add"))
        tla.simd.vsts(o[3, 0], tla.simd.vreduce(zeroed, "add"))
        # Seed only the tail row; full tiles overwrite their rows entirely.
        tla.simd.vsts(o[4 + TILES - 1, 0], tla.simd.vsplat(SENTINEL_C, tl.uint32))
        for t in tl.static_range(TILES):
            va = tla.simd.vlds(x[t, 0])
            vb = tla.simd.vlds(y[t, 0])
            if t == TILES - 1:
                tail_mask = tla.simd.pge(TAIL, N)
                tla.simd.vsts(o[4 + t, 0], tla.simd.vxor(va, vb, tail_mask),
                              mask=tail_mask)
            else:
                tla.simd.vsts(o[4 + t, 0], tla.simd.vxor(va, vb))
    tla.copy(o, tla.view(O, (4 + TILES, N)))


def constexprs():
    return {"N": SHAPE["lanes"], "ACTIVE": SHAPE["active"],
            "TILES": SHAPE["tiles"], "TAIL": SHAPE["tail"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vxor, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert SHAPE["total"] % SHAPE["lanes"] != 0 and 0 < SHAPE["tail"] < SHAPE["lanes"]
    assert text.count("ave.hir.vxor") == 2 + SHAPE["tiles"]
    assert text.count("ave.hir.vsel") == 2, "ZEROING row and tail row"
    assert text.count("ave.hir.reduction") == 2
    assert "ave.hir.pge <VL8>" in text, "tail predicate must select 8 lanes"
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=2, stores=1, buffers=3,
                     extra=("ave.hir.vxor", "ave.hir.vsel", "ave.hir.reduction"))
    print("IR PASS: vxor full/ZEROING/reduce plus 3 full + 1 tail tile")


def _launch():
    s = SHAPE
    padded = s["tiles"] * s["lanes"]
    xi = torch.randint(0, 1 << 32, (padded,), dtype=torch.int64)
    yi = torch.randint(0, 1 << 32, (padded,), dtype=torch.int64)
    out = torch.empty((s["rows"], s["lanes"]), dtype=torch.uint32, device="npu")
    vxor[(1,)](xi.to(torch.uint32).to("npu"), yi.to(torch.uint32).to("npu"),
               out, **constexprs())
    torch.npu.synchronize()

    x0, y0 = xi[:s["lanes"]], yi[:s["lanes"]]
    full = (x0 ^ y0) & 0xffffffff
    zeroed = full.clone()
    zeroed[s["active"]:] = 0
    ref = torch.zeros((s["rows"], s["lanes"]), dtype=torch.int64)
    ref[0], ref[1] = full, zeroed
    ref[2, 0] = full.sum() & 0xffffffff
    ref[3, 0] = zeroed.sum() & 0xffffffff
    tiled = torch.full((padded,), SENTINEL, dtype=torch.int64)
    tiled[:s["total"]] = (xi ^ yi)[:s["total"]] & 0xffffffff
    ref[4:] = tiled.reshape(s["tiles"], s["lanes"])
    return out, ref.to(torch.uint32)


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vxor generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    out, ref = _launch()
    got = out.cpu()
    torch.testing.assert_close(got[:2].view(torch.int32), ref[:2].view(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(got[2:4, 0], ref[2:4, 0], rtol=0, atol=0)
    torch.testing.assert_close(got[4:].view(torch.int32), ref[4:].view(torch.int32), rtol=0, atol=0)
    print("VXOR OK: exact bits, ZEROING, reductions and tail tile")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vxor", "ir" if do_ir else "",
          "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
