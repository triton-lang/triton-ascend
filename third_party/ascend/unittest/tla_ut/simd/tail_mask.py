"""TLA 3-in-1 test: tail_mask (dynamic prefix predicate, TileLang update_mask).

tail_mask(value) is composed as vci(0) ULT broadcast(value), so the IR asserts
one vci, one broadcast and one vcmp per row.

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).
Rows: 0 value=20, 1 value=0 (all false), 2 value=64 (all true).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

SHAPE = dict(lanes=64)
SIGNATURE = {"O": "*u32"}


@tla.jit
def tail_mask(O, N: tl.constexpr):
    o = tla.alloc_buffer((3, N), tl.uint32, space="ub")
    with tla.scope.simd():
        ones = tla.simd.vsplat(1, tl.uint32)
        zeros = tla.simd.vsplat(0, tl.uint32)
        tla.simd.vsts(o[0, 0], tla.simd.vsel(tla.simd.tail_mask(20), ones, zeros))
        tla.simd.vsts(o[1, 0], tla.simd.vsel(tla.simd.tail_mask(0), ones, zeros))
        tla.simd.vsts(o[2, 0], tla.simd.vsel(tla.simd.tail_mask(64), ones, zeros))
    tla.copy(o, tla.view(O, (3, N)))


def constexprs():
    return {"N": SHAPE["lanes"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(tail_mask, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.vci") == 3
    assert text.count("ave.hir.broadcast") == 3
    assert text.count("ave.hir.vcmp") == 3
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=0, stores=1, buffers=None, extra=("ave.hir.vcmp", ))
    print("IR PASS: tail_mask vci+broadcast+vcmp composition")


def _launch():
    s = SHAPE
    out = torch.empty((3, s["lanes"]), dtype=torch.uint32, device="npu")
    tail_mask[(1, )](out, **constexprs())
    torch.npu.synchronize()
    ref = torch.zeros((3, s["lanes"]), dtype=torch.int64)
    ref[0, :20] = 1
    ref[2, :] = 1
    return out, ref.to(torch.uint32)


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: tail_mask generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    out, ref = _launch()
    torch.testing.assert_close(out.cpu().view(torch.int32), ref.view(torch.int32), rtol=0, atol=0)
    print("TAIL_MASK OK: dynamic prefix predicate exact bits")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: tail_mask", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
