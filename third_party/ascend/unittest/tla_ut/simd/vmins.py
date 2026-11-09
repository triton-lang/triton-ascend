"""TLA 3-in-1 test: uint32 scalar min, ZEROING, reductions and tail tile.

High-bit inputs distinguish unsigned min from signed min. Rows 0/1 hold a
single-tile full-mask and ZEROING result, rows 2/3 their horizontal sums,
and rows 4..7 a four-tile run whose last tile is 8/64 lanes. The tail store
is masked, so the tail row padding keeps its sentinel.

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).
Inputs are regenerated randomly on every run.
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
SHAPE["rows"] = 4 + SHAPE["tiles"]
SENTINEL = 0xDEADBEEF
SENTINEL_C = tl.constexpr(SENTINEL)
SIGNATURE = {"X": "*u32", "O": "*u32", "limit": "u32"}


@tla.jit
def vmins(X, O, limit, N: tl.constexpr, ACTIVE: tl.constexpr, TILES: tl.constexpr, TAIL: tl.constexpr):
    x = tla.alloc_buffer((TILES, N), tl.uint32, space="ub")
    o = tla.alloc_buffer((4 + TILES, N), tl.uint32, space="ub")
    tla.copy(tla.view(X, (TILES, N)), x)
    with tla.scope.simd():
        a = tla.simd.vlds(x[0, 0])
        mask = tla.simd.pge(ACTIVE, N)
        limit = limit.to(tl.uint32)
        full = tla.simd.vmins(a, limit)
        zeroed = tla.simd.vmins(a, limit, mask)
        tla.simd.vsts(o[0, 0], full)
        tla.simd.vsts(o[1, 0], zeroed)
        tla.simd.vsts(o[2, 0], tla.simd.vreduce(full, "add"))
        tla.simd.vsts(o[3, 0], tla.simd.vreduce(zeroed, "add"))
        # Seed only the tail row; full tiles overwrite their rows entirely.
        tla.simd.vsts(o[4 + TILES - 1, 0], tla.simd.vsplat(SENTINEL_C, tl.uint32))
        for t in tl.static_range(TILES):
            va = tla.simd.vlds(x[t, 0])
            if t == TILES - 1:
                tail_mask = tla.simd.pge(TAIL, N)
                tla.simd.vsts(o[4 + t, 0], tla.simd.vmins(va, limit, tail_mask), mask=tail_mask)
            else:
                tla.simd.vsts(o[4 + t, 0], tla.simd.vmins(va, limit))
    tla.copy(o, tla.view(O, (4 + TILES, N)))


def constexprs():
    return {"N": SHAPE["lanes"], "ACTIVE": SHAPE["active"], "TILES": SHAPE["tiles"], "TAIL": SHAPE["tail"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vmins, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert SHAPE["total"] % SHAPE["lanes"] != 0 and 0 < SHAPE["tail"] < SHAPE["lanes"]
    assert text.count("ave.hir.vumins") == 2 + SHAPE["tiles"]
    assert "ave.hir.vsel" not in text, "ZEROING must stay a bare masked op"
    assert text.count("ave.hir.reduction") == 2
    assert "ave.hir.pge <VL8>" in text, "tail predicate must select 8 lanes"
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=1, stores=1, buffers=None, extra=("ave.hir.vumins", "ave.hir.reduction"))
    print("IR PASS: vmins full/ZEROING/reduce plus 3 full + 1 tail tile")


def _launch():
    s = SHAPE
    padded = s["tiles"] * s["lanes"]
    xi = torch.randint(0, 1 << 32, (padded, ), dtype=torch.int64)
    limit = int(torch.randint(0, 1 << 32, (1, )).item())
    out = torch.empty((s["rows"], s["lanes"]), dtype=torch.uint32, device="npu")
    vmins[(1, )](xi.to(torch.uint32).to("npu"), out, limit, **constexprs())
    torch.npu.synchronize()

    full = xi[:s["lanes"]].clamp(max=limit)
    zeroed = full.clone()
    zeroed[s["active"]:] = 0
    ref = torch.zeros((s["rows"], s["lanes"]), dtype=torch.int64)
    ref[0], ref[1] = full, zeroed
    ref[2, 0] = full.sum() & 0xffffffff
    ref[3, 0] = zeroed.sum() & 0xffffffff
    tiled = torch.full((padded, ), SENTINEL, dtype=torch.int64)
    tiled[:s["total"]] = xi[:s["total"]].clamp(max=limit)
    ref[4:] = tiled.reshape(s["tiles"], s["lanes"])
    return out, ref.to(torch.uint32)


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vmins generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    out, ref = _launch()
    got = out.cpu()
    torch.testing.assert_close(got[:2].view(torch.int32), ref[:2].view(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(got[2:4, 0], ref[2:4, 0], rtol=0, atol=0)
    torch.testing.assert_close(got[4:].view(torch.int32), ref[4:].view(torch.int32), rtol=0, atol=0)
    print("VMINS OK: unsigned min, ZEROING, vreduce sums and tail tile")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vmins", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
