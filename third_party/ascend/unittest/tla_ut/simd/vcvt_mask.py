"""TLA 3-in-1 test: width-changing vcvt conversions under partial masks.

Pins the measured f16<->f32 lane contract on Ascend950PR:

* the predicate follows the **source** register's granularity in both
  directions: ``f16 -> f32`` takes the f16 register's 128-lane ``b16`` mask
  and gates the source lane each output reads, while ``f32 -> f16`` takes
  the f32 register's 64-lane ``b32`` mask and gates whole lane pairs;
* each f32 lane ``i`` pairs with the f16 lanes ``(2i, 2i+1)``: widening
  reads ``f16[2i]``/``f16[2i+1]`` (``part_even``/``part_odd``) into
  ``f32[i]``, and narrowing writes ``f16(x[i])`` into lane ``2i``/``2i+1``.

Under ``PAT.M4`` (every 4th slot) the widening ``part_even`` row therefore
survives on even outputs while ``part_odd`` converts nothing (its slots are
never 0 mod 4), and the narrowing rows survive on pairs with ``i % 4 == 0``.

Rows (each kernel: full/part_even, M4/part_even, full/part_odd, M4/part_odd).

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

F16_LANES = 128
F32_LANES = 64
SIGNATURE_W = {"X": "*fp16", "O": "*fp32"}
SIGNATURE_N = {"X": "*fp32", "O": "*fp16"}


@tla.jit
def widen(X, O, N: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.float16, space="ub")
    o = tla.alloc_buffer((4, N // 2), tl.float32, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    with tla.scope.simd():
        h = tla.simd.vlds(x[0])
        m4 = tla.simd.pset(16, tla.simd.PAT.M4)
        tla.simd.vsts(o[0, 0], tla.simd.vcvt(h, tl.float32))
        tla.simd.vsts(o[1, 0], tla.simd.vcvt(h, tl.float32, m4))
        tla.simd.vsts(o[2, 0], tla.simd.vcvt(h, tl.float32, part=tla.simd.PART.ODD))
        tla.simd.vsts(o[3, 0], tla.simd.vcvt(h, tl.float32, m4, part=tla.simd.PART.ODD))
    tla.copy(o, tla.view(O, (4, N // 2)))


@tla.jit
def narrow(X, O, N: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o = tla.alloc_buffer((4, N * 2), tl.float16, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    with tla.scope.simd():
        f = tla.simd.vlds(x[0])
        m4 = tla.simd.pset(32, tla.simd.PAT.M4)
        tla.simd.vsts(o[0, 0], tla.simd.vcvt(f, tl.float16), dist="NORM_B16")
        tla.simd.vsts(o[1, 0], tla.simd.vcvt(f, tl.float16, m4), dist="NORM_B16")
        tla.simd.vsts(o[2, 0], tla.simd.vcvt(f, tl.float16, part=tla.simd.PART.ODD), dist="NORM_B16")
        tla.simd.vsts(o[3, 0], tla.simd.vcvt(f, tl.float16, m4, part=tla.simd.PART.ODD), dist="NORM_B16")
    tla.copy(o, tla.view(O, (4, N * 2)))


def test_ir(output_dir=None):
    module, ctx = compile_offline(widen, SIGNATURE_W, {"N": F16_LANES}, output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.vextf") == 4, "the widening path lowers to vextf"
    assert text.count("ave.hir.pge <M4>") == 1, "one shared M4 predicate"
    assert "vector<128xi1>" in text, "the widening predicate is the f16 source register's 128-lane mask"
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=1, stores=1, buffers=None, extra=("ave.hir.vextf", ))

    module, ctx = compile_offline(narrow, SIGNATURE_N, {"N": F32_LANES}, output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.vtruncf") == 4, "the narrowing path lowers to vtruncf"
    assert text.count("ave.hir.pge <M4>") == 1, "one shared M4 predicate"
    assert "vector<64xi1>" in text, "the narrowing predicate is the f32 source register's 64-lane mask"
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=1, stores=1, buffers=None, extra=("ave.hir.vtruncf", ))
    print("IR PASS: f16<->f32 vcvt masks at the source register's granularity")


def _launch():
    x16 = (torch.arange(F16_LANES, dtype=torch.float32) * 0.5).to(torch.float16).float()
    wout = torch.empty((4, F32_LANES), dtype=torch.float32, device="npu")
    widen[(1, )](x16.to(torch.float16).npu(), wout, N=F16_LANES)
    torch.npu.synchronize()
    # Widen: f32[i] reads f16[2i]/f16[2i+1], gated by that source lane's slot.
    even, odd, zero = x16[0::2], x16[1::2], torch.zeros(F32_LANES)
    even_slot = torch.arange(F32_LANES) % 2 == 0
    wref = torch.stack([
        even,
        torch.where(even_slot, even, zero), odd, zero,  # (2i+1) % 4 != 0: part_odd converts nothing under M4
    ])

    x32 = torch.arange(F32_LANES, dtype=torch.float32) * 0.5
    nout = torch.empty((4, F16_LANES), dtype=torch.float16, device="npu")
    narrow[(1, )](x32.npu(), nout, N=F32_LANES)
    torch.npu.synchronize()
    # Narrow: f16(x[i]) lands in lane 2i/2i+1, gated as a whole b32 pair.
    full_even, full_odd = torch.zeros(F16_LANES), torch.zeros(F16_LANES)
    full_even[0::2], full_odd[1::2] = x32, x32
    m4_even, m4_odd = torch.zeros(F16_LANES), torch.zeros(F16_LANES)
    m4_even[0::8] = x32[0::4]
    m4_odd[1::8] = x32[0::4]
    nref = torch.stack([full_even, m4_even, full_odd, m4_odd]).to(torch.float16)
    return wout.cpu(), wref, nout.cpu(), nref


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vcvt_mask generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    wout, wref, nout, nref = _launch()
    torch.testing.assert_close(wout, wref, rtol=0, atol=0)
    torch.testing.assert_close(nout.view(torch.int16), nref.view(torch.int16), rtol=0, atol=0)
    print("VCVT_MASK OK: f16<->f32 pair lanes gated at the source register's granularity")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vcvt_mask", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
