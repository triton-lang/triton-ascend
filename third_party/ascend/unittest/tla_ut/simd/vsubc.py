"""TLA 3-in-1 test: composed uint32/int32 vsubc and borrow-out.

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).
Inputs are regenerated randomly on every run.
Rows 0/1 hold full-mask and masked uint32 results, row 2 the equivalent int32
bitcast case, and the remaining rows a four-tile run whose last tile is 8/64
lanes.  The tail stores are masked, so the tail row padding keeps its sentinel.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

SHAPE = dict(lanes=64, active=32, tiles=4, total=200)
SHAPE["tail"] = SHAPE["total"] - (SHAPE["tiles"] - 1) * SHAPE["lanes"]
SHAPE["base_rows"] = 3
SHAPE["rows"] = SHAPE["base_rows"] + SHAPE["tiles"]
SENTINEL = 0xDEADBEEF
SENTINEL_C = tl.constexpr(SENTINEL)
SIGNATURE = {name: "*u32" for name in ("X", "Y", "D", "B")}


@tla.jit
def vsubc(X, Y, D, B, N: tl.constexpr, ACTIVE: tl.constexpr, TILES: tl.constexpr, TAIL: tl.constexpr,
          BASE_ROWS: tl.constexpr):
    x = tla.alloc_buffer((TILES, N), tl.uint32, space="ub")
    y = tla.alloc_buffer((TILES, N), tl.uint32, space="ub")
    d = tla.alloc_buffer((BASE_ROWS + TILES, N), tl.uint32, space="ub")
    b = tla.alloc_buffer((BASE_ROWS + TILES, N), tl.uint32, space="ub")
    tla.copy(tla.view(X, (TILES, N)), x)
    tla.copy(tla.view(Y, (TILES, N)), y)
    with tla.scope.simd():
        a = tla.simd.vlds(x[0, 0])
        c = tla.simd.vlds(y[0, 0])
        mask = tla.simd.pge(ACTIVE, N)
        one_u = tla.simd.vsplat(1, tl.uint32)
        zero_u = tla.simd.vsplat(0, tl.uint32)

        full, full_borrow = tla.simd.vsubc(a, c)
        zeroed, zeroed_borrow = tla.simd.vsubc(a, c, mask)
        tla.simd.vsts(d[0, 0], full)
        tla.simd.vsts(d[1, 0], zeroed)
        tla.simd.vsts(b[0, 0], tla.simd.vsel(full_borrow, one_u, zero_u))
        tla.simd.vsts(b[1, 0], tla.simd.vsel(zeroed_borrow, one_u, zero_u))

        signed = tla.simd.vbitcast(a, "int32")
        signed_subtrahend = tla.simd.vbitcast(c, "int32")
        signed_diff, signed_borrow = tla.simd.vsubc(signed, signed_subtrahend)
        one_i = tla.simd.vsplat(1, tl.int32)
        zero_i = tla.simd.vsplat(0, tl.int32)
        tla.simd.vsts(d[2, 0], tla.simd.vbitcast(signed_diff, "uint32"))
        tla.simd.vsts(b[2, 0], tla.simd.vsel(signed_borrow, one_i, zero_i))

        # Seed only the tail rows; full tiles overwrite their rows entirely.
        tla.simd.vsts(d[BASE_ROWS + TILES - 1, 0], tla.simd.vsplat(SENTINEL_C, tl.uint32))
        tla.simd.vsts(b[BASE_ROWS + TILES - 1, 0], tla.simd.vsplat(SENTINEL_C, tl.uint32))
        for t in tl.static_range(TILES):
            va = tla.simd.vlds(x[t, 0])
            vb = tla.simd.vlds(y[t, 0])
            if t == TILES - 1:
                tail_mask = tla.simd.pge(TAIL, N)
                tail_diff, tail_borrow = tla.simd.vsubc(va, vb, tail_mask)
                tla.simd.vsts(d[BASE_ROWS + t, 0], tail_diff, mask=tail_mask)
                tla.simd.vsts(b[BASE_ROWS + t, 0], tla.simd.vsel(tail_borrow, one_u, zero_u), mask=tail_mask)
            else:
                tile_diff, tile_borrow = tla.simd.vsubc(va, vb)
                tla.simd.vsts(d[BASE_ROWS + t, 0], tile_diff)
                tla.simd.vsts(b[BASE_ROWS + t, 0], tla.simd.vsel(tile_borrow, one_u, zero_u))
    tla.copy(d, tla.view(D, (BASE_ROWS + TILES, N)))
    tla.copy(b, tla.view(B, (BASE_ROWS + TILES, N)))


def constexprs():
    return {
        "N": SHAPE["lanes"],
        "ACTIVE": SHAPE["active"],
        "TILES": SHAPE["tiles"],
        "TAIL": SHAPE["tail"],
        "BASE_ROWS": SHAPE["base_rows"],
    }


def test_ir(output_dir=None):
    module, ctx = compile_offline(vsubc, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert SHAPE["total"] % SHAPE["lanes"] != 0 and 0 < SHAPE["tail"] < SHAPE["lanes"]
    assert text.count("ave.hir.vsub") == SHAPE["base_rows"] + SHAPE["tiles"]
    assert text.count("ave.hir.vcmp <ULT>") == SHAPE["base_rows"] + SHAPE["tiles"]
    assert "ave.hir.vsubc" not in text, "vsubc must be composed in the frontend"
    assert "ave.hir.pge <VL8>" in text, "tail predicate must select 8 lanes"
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=2, stores=2, buffers=None, extra=("ave.hir.vsub", "ave.hir.vcmp"))
    print("IR PASS: vsubc composed from vsub + unsigned vcmp")


def _launch():
    s = SHAPE
    padded = s["tiles"] * s["lanes"]
    xi = torch.randint(0, 1 << 32, (padded, ), dtype=torch.int64)
    yi = torch.randint(0, 1 << 32, (padded, ), dtype=torch.int64)
    d = torch.empty((s["rows"], s["lanes"]), dtype=torch.uint32, device="npu")
    b = torch.empty((s["rows"], s["lanes"]), dtype=torch.uint32, device="npu")
    vsubc[(1, )](xi.to(torch.uint32).to("npu"), yi.to(torch.uint32).to("npu"), d, b, **constexprs())
    torch.npu.synchronize()

    x0, y0 = xi[:s["lanes"]], yi[:s["lanes"]]
    full = (x0 - y0) & 0xffffffff
    zeroed = full.clone()
    zeroed[s["active"]:] = 0
    full_borrow = (x0 < y0).to(torch.int64)
    zeroed_borrow = full_borrow.clone()
    zeroed_borrow[s["active"]:] = 0

    dref = torch.zeros((s["rows"], s["lanes"]), dtype=torch.int64)
    bref = torch.zeros((s["rows"], s["lanes"]), dtype=torch.int64)
    dref[0], dref[1], dref[2] = full, zeroed, full
    bref[0], bref[1], bref[2] = full_borrow, zeroed_borrow, full_borrow

    tiled_d = torch.full((padded, ), SENTINEL, dtype=torch.int64)
    tiled_b = torch.full((padded, ), SENTINEL, dtype=torch.int64)
    tiled_d[:s["total"]] = (xi - yi)[:s["total"]] & 0xffffffff
    tiled_b[:s["total"]] = (xi < yi)[:s["total"]].to(torch.int64)
    dref[s["base_rows"]:] = tiled_d.reshape(s["tiles"], s["lanes"])
    bref[s["base_rows"]:] = tiled_b.reshape(s["tiles"], s["lanes"])
    return d, b, dref.to(torch.uint32), bref.to(torch.uint32)


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vsubc generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    d, b, dref, bref = _launch()
    torch.testing.assert_close(d.cpu().view(torch.int32), dref.cpu().view(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(b.cpu().view(torch.int32), bref.cpu().view(torch.int32), rtol=0, atol=0)
    print("VSUBC OK: diff, borrow, ZEROING and tail tile")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vsubc", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
