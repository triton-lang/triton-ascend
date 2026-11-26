"""TLA 3-in-1 test: combined elementwise SIMD chain (end-to-end).

Semantics for one 64-lane f32 tile (row) of ``X``:

    zero = vdup(0.0)                        masked scalar broadcast, ALL mask
    half = vdup(0.5)                        masked scalar broadcast, ALL mask
    q    = vdiv(a, b)                       ave.hir.vdiv
    mx   = vmax(a, b)                       ave.hir.vmax
    mn   = vmin(a, b)                       ave.hir.vmin
    ms   = vmaxs(a, 0.0)                    ave.hir.vmaxs (scalar == f32)
    ex   = vexp(vmin(a, zero))              ave.hir.vexp (hardware SFU)
    acc  = q + mx + mn + ms + ex
    o    = vsel(pge(32, 64), acc, half)     lanes >= 32 -> 0.5

This file is the combined-chain regression only: it proves the six interfaces
compose in one SIMD scope and that the runtime launch matches the reference.
The per-interface contracts (mask matrix, MODE_ZEROING/MERGING merge source,
unsigned zeroing, exact tier, dtype aliases, 64-lane wide registers, scalar
bounds) live in ``vector/vdiv.py``, ``vexp.py``, ``vmax.py``, ``vmin.py``,
``vmaxs.py``, ``vdup.py``, ``vdups.py`` and ``vdupv.py``; the runtime-N tail /
multi-chunk / multi-core matrix lives in ``vector/simd_tail.py``.  Keeping that
detail out of this file avoids two copies of the same IR assertions.

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tla_ut/
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

# default shapes: rows (tmax), lane block, cores
SHAPE = dict(tmax=8, blk=64, cores=4)
SIGNATURE = {"X_ptr": "*fp32", "O_ptr": "*fp32", "num_tokens": "i32"}


@tla.jit
def simd_elementwise(X_ptr, O_ptr, num_tokens, TMAX: tl.constexpr, BLK: tl.constexpr, CORES: tl.constexpr):
    tl.static_assert(BLK == 64, "one full f32 register per tile")
    tl.static_assert(TMAX % 2 == 0, "one pair of rows per output row")
    xy_ub = tla.alloc_buffer((2, BLK), tl.float32, space="ub", versions=2)
    o_ub = tla.alloc_buffer((BLK, ), tl.float32, space="ub", versions=2)
    for p in range(tl.program_id(0), num_tokens // 2, CORES):
        xy = tla.view(X_ptr, (2, BLK), strides=(BLK, 1), offsets=(2 * p, 0))
        o = tla.view(O_ptr, (BLK, ), strides=(1, ), offsets=(p * BLK, ))
        tla.copy(xy, xy_ub)
        with tla.scope.simd():
            a = tla.simd.vlds(xy_ub[0, 0], dist="NORM")
            b = tla.simd.vlds(xy_ub[1, 0], dist="NORM")
            zero = tla.simd.vdup(0.0, tl.float32)
            half = tla.simd.vdup(0.5, tl.float32)
            q = tla.simd.vdiv(a, b)
            mx = tla.simd.vmax(a, b)
            mn = tla.simd.vmin(a, b)
            ms = tla.simd.vmaxs(a, 0.0)
            ex = tla.simd.vexp(tla.simd.vmin(a, zero))
            acc = tla.simd.vadd(tla.simd.vadd(q, mx), tla.simd.vadd(mn, ms))
            acc = tla.simd.vadd(acc, ex)
            res = tla.simd.vsel(tla.simd.pge(32, BLK), acc, half)
            tla.simd.vsts(o_ub[0], res, dist="NORM_B32")
        tla.copy(o_ub, o)


def constexprs():
    s = SHAPE
    return {"TMAX": s["tmax"], "BLK": s["blk"], "CORES": s["cores"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(simd_elementwise, SIGNATURE, constexprs(), output_dir=output_dir)
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(text, loads=1, stores=1, buffers=2)
    for probe in ("ave.hir.vdiv", "ave.hir.vmax", "ave.hir.vmin", "ave.hir.vmaxs", "ave.hir.vexp", "ave.hir.broadcast"):
        assert probe in text, f"missing {probe} in lowered IR"
    # vdup must be the masked broadcast, never the unmasked scalar_broadcast
    assert "ave.hir.scalar_broadcast" not in text, "vdup must not degrade to vsplat"
    # The combined chain ends in an explicit vsel, which is the only select here.
    assert text.count("ave.hir.vsel") == 1
    print("IR PASS: combined vdiv/vmax/vmin/vmaxs/vexp/vdup chain plus vsel")


def _launch():
    s = SHAPE
    # strictly positive inputs: b is the divisor, so no inf/NaN in the reference
    X = torch.rand(s["tmax"], s["blk"], dtype=torch.float32, device="npu") + 0.5
    O = torch.full((s["tmax"] // 2, s["blk"]), -7.0, dtype=torch.float32, device="npu")
    simd_elementwise[(s["cores"], )](X, O, s["tmax"], **constexprs())
    torch.npu.synchronize()
    x = X.cpu()
    a, b = x[0::2], x[1::2]
    zero = torch.zeros_like(a)
    acc = (a / b) + torch.maximum(a, b) + torch.minimum(a, b)
    acc = acc + torch.maximum(a, zero) + torch.exp(torch.minimum(a, zero))
    keep = (torch.arange(s["blk"]) < 32).to(torch.float32)
    ref = acc * keep + 0.5 * (1.0 - keep)
    return O.cpu(), ref


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: simd_elementwise generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    out, ref = _launch()
    ok = torch.allclose(out, ref, rtol=1e-5, atol=1e-5)
    if not ok:
        bad = (out - ref).abs() > 1e-5
        print("num_mismatch:", int(bad.sum()), "first:", bad.nonzero()[:8].tolist())
    print("SIMD_ELEMENTWISE", "OK" if ok else "MISMATCH", "tmax", SHAPE["tmax"])
    assert ok, "simd_elementwise e2e mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: simd_elementwise", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
