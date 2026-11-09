"""TLA 3-in-1 test: vdup masked scalar broadcast (all masks, both modes).

One register per output row (64 f32 lanes):

  0 vdup(1.5, f32)                    default ALL mask, MODE_ZEROING
  1 vdup(1.5, f32, prefix)            first ACTIVE lanes, MODE_ZEROING
  2 vdup(1.5, f32, arb)               per-lane threshold from ``T`` (signed i32 index)
  3 vdup(1.5, f32, none)              all-false predicate
  4 vdup(1.5, f32, prefix, MERGING)   merge source is ``+0``
  5 vdup(1.5, f32, arb, MERGING)

``vdup`` must lower to the masked ``ave.hir.broadcast`` (vdups form), never to
the unmasked ``ave.hir.scalar_broadcast`` that ``vsplat`` emits.  A bf16 kernel
pins the 128-lane 16-bit register, and an IR-only kernel covers every
documented dtype spelling (``i8``/``u8``/``i32``/``u32``/``f16``/``f32``/
``bf16``/``fp8e4nv`` plus the canonical ``int*``/``uint*``/``fp*``/``float*``
aliases); i64/u64 stay rejected because that wrapper drops the predicate.

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
BF16_LANES = 128
BF16_ROWS = 3
SIGNATURE = {"T": "*i32", "O": "*fp32"}
BF16_SIGNATURE = {"O": "*bf16"}
ALIAS_SIGNATURE = {"O": "*u32"}
ALIAS_CASES = (
    ("i8", "vector<256xi8>"),
    ("u8", "vector<256xi8>"),
    ("i32", "vector<64xi32>"),
    ("u32", "vector<64xi32>"),
    ("f16", "vector<128xf16>"),
    ("f32", "vector<64xf32>"),
    ("bf16", "vector<128xbf16>"),
    ("fp8e4nv", "vector<256xf8E4M3FN>"),
    ("int32", "vector<64xi32>"),
    ("uint8", "vector<256xi8>"),
    ("fp16", "vector<128xf16>"),
    ("float32", "vector<64xf32>"),
)


@tla.jit
def vdup(T, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    tb = tla.alloc_buffer((N, ), tl.int32, space="ub")
    ob = tla.alloc_buffer((6, N), tl.float32, space="ub")
    tla.copy(tla.view(T, (N, )), tb)
    with tla.scope.simd():
        t = tla.simd.vlds(tb[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        arb = tla.simd.vcmp(idx, t, op=tla.simd.CMP.LT)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(ob[0, 0], tla.simd.vdup(1.5, tl.float32, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[1, 0], tla.simd.vdup(1.5, tl.float32, mask=prefix, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[2, 0], tla.simd.vdup(1.5, tl.float32, mask=arb, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[3, 0], tla.simd.vdup(1.5, tl.float32, mask=none, mode=tla.simd.MODE.ZEROING))
        tla.simd.vsts(ob[4, 0], tla.simd.vdup(1.5, tl.float32, mask=prefix, mode=tla.simd.MODE.MERGING))
        tla.simd.vsts(ob[5, 0], tla.simd.vdup(1.5, tl.float32, mask=arb, mode=tla.simd.MODE.MERGING))
    tla.copy(ob, tla.view(O, (6, N)))


@tla.jit
def vdup_bf16(O, N: tl.constexpr, ACTIVE: tl.constexpr):
    ob = tla.alloc_buffer((3, N), tl.bfloat16, space="ub")
    with tla.scope.simd():
        prefix = tla.simd.pge(ACTIVE, N)
        zero = tla.simd.vsplat(0.0, tl.bfloat16)
        none = tla.simd.vcmp(zero, zero, op=tla.simd.CMP.NE)
        tla.simd.vsts(ob[0, 0], tla.simd.vdup(1.5, tl.bfloat16), dist="NORM_B16")
        tla.simd.vsts(ob[1, 0], tla.simd.vdup(1.5, tl.bfloat16, mask=prefix), dist="NORM_B16")
        tla.simd.vsts(ob[2, 0], tla.simd.vdup(1.5, tl.bfloat16, mask=none), dist="NORM_B16")
    tla.copy(ob, tla.view(O, (3, N)))


@tla.jit
def vdup_alias(O, DT: tl.constexpr):
    """IR-only: dtype spelling -> canonical element type, still ``broadcast``."""
    ob = tla.alloc_buffer((64, ), tl.uint32, space="ub")
    with tla.scope.simd():
        d = tla.simd.vdup(1, DT)
        bc = tla.simd.vbitcast(d, "uint32")
        tla.simd.vsts(ob[0], bc, dist="NORM_B32")
    tla.copy(ob, tla.view(O, (64, )))


def constexprs(lanes=None, active=None):
    return {"N": SHAPE["lanes"] if lanes is None else lanes, "ACTIVE": SHAPE["active"] if active is None else active}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vdup, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.broadcast") == 6, "all six vdup rows use the masked broadcast"
    # The two MERGING rows each append a +0 ``scalar_broadcast`` merge source
    # (tla_builder.cc::applySimdMode), so scalar_broadcast is expected exactly
    # twice.  The six vdup ops themselves are asserted above via broadcast.
    assert text.count("ave.hir.scalar_broadcast") == 2, "only the MERGING merge source is scalar_broadcast"
    assert text.count("ave.hir.vsel") == 2, "only the two MERGING rows add vsel"
    assert "ave.hir.pge <VL32>" in text
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=1, stores=1, buffers=None, extra=("ave.hir.broadcast", ))

    module, ctx = compile_offline(vdup_bf16, BF16_SIGNATURE, constexprs(BF16_LANES, BF16_LANES // 2))
    btext = str(module)
    assert btext.count("ave.hir.broadcast") == 3
    assert "ave.hir.pge <VL64>" in btext
    btext = str(lower_offline(module, ctx))
    check_lowered_ir(btext, loads=0, stores=1, buffers=None, extra=("ave.hir.broadcast", "vector<128xbf16>"))

    for name, vec_ty in ALIAS_CASES:
        module, ctx = compile_offline(vdup_alias, ALIAS_SIGNATURE, {"DT": name})
        module = lower_offline(module, ctx)
        atext = str(module)
        assert "ave.hir.broadcast" in atext, name
        assert "ave.hir.scalar_broadcast" not in atext, name
        assert vec_ty in atext, (name, vec_ty)
    print("IR PASS: vdup masked broadcast, 128-lane bf16 and every dtype alias")


def _masks(n, active):
    idx = torch.arange(n)
    thr = (torch.arange(n) % 5) * (n // 5)
    return {"prefix": idx < active, "arb": idx < thr, "none": torch.zeros(n, dtype=torch.bool)}, thr


def _apply(value, mask):
    return torch.where(mask, value, torch.zeros((), dtype=value.dtype))


def _launch():
    n = SHAPE["lanes"]
    masks, thr = _masks(n, SHAPE["active"])
    out = torch.empty((SHAPE["rows"], n), dtype=torch.float32, device="npu")
    vdup[(1, )](thr.to(torch.int32).npu(), out, **constexprs())
    torch.npu.synchronize()
    full = torch.full((n, ), 1.5)
    ref = torch.stack([
        full,
        _apply(full, masks["prefix"]),
        _apply(full, masks["arb"]),
        _apply(full, masks["none"]),
        _apply(full, masks["prefix"]),
        _apply(full, masks["arb"]),
    ])

    m = BF16_LANES
    bmasks = {"prefix": torch.arange(m) < m // 2, "none": torch.zeros(m, dtype=torch.bool)}
    bout = torch.empty((BF16_ROWS, m), dtype=torch.bfloat16, device="npu")
    vdup_bf16[(1, )](bout, **constexprs(BF16_LANES, m // 2))
    torch.npu.synchronize()
    bfull = torch.full((m, ), 1.5, dtype=torch.bfloat16)
    bref = torch.stack([
        bfull,
        _apply(bfull, bmasks["prefix"]),
        _apply(bfull, bmasks["none"]),
    ])
    return (out.cpu(), ref, masks), (bout.cpu(), bref, bmasks)


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
    print("COMPILE PASS: vdup generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    (got, ref, masks), (bgot, bref, bmasks) = _launch()
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    _inactive_zero(got, masks)
    torch.testing.assert_close(bgot, bref, rtol=0, atol=0)
    assert _bitwise_zero(bgot[1][~bmasks["prefix"]]), "bf16 prefix row leaked an inactive lane"
    assert _bitwise_zero(bgot[2]), "bf16 all-false row must be +0"
    print("VDUP OK: f32/bf16 masked broadcast, ALL/prefix/arbitrary/none, both modes")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vdup", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
