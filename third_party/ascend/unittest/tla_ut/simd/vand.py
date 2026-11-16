"""TLA 3-in-1 test: vand full/ZEROING/all-false/MERGING masks on u32 exact bits.

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).
Rows: 0 full-mask, 1 prefix ZEROING, 2 all-false ZEROING, 3 prefix MERGING
(MERGING keeps the first source operand on the inactive lanes).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

SHAPE = dict(lanes=64, active=32)
SIGNATURE = {"X": "*u32", "Y": "*u32", "O": "*u32"}


@tla.jit
def vand(X, Y, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.uint32, space="ub")
    y = tla.alloc_buffer((N, ), tl.uint32, space="ub")
    o = tla.alloc_buffer((4, N), tl.uint32, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(Y, (N, )), y)
    with tla.scope.simd():
        a = tla.simd.vlds(x[0])
        b = tla.simd.vlds(y[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(o[0, 0], tla.simd.vand(a, b))
        tla.simd.vsts(o[1, 0], tla.simd.vand(a, b, prefix))
        tla.simd.vsts(o[2, 0], tla.simd.vand(a, b, none))
        tla.simd.vsts(o[3, 0], tla.simd.vand(a, b, mask=prefix, mode=tla.simd.MODE.MERGING))
    tla.copy(o, tla.view(O, (4, N)))


def constexprs():
    return {"N": SHAPE["lanes"], "ACTIVE": SHAPE["active"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vand, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.vand") == 4
    assert text.count("ave.hir.vsel") == 1, "only the MERGING row adds vsel"
    assert "ave.hir.pge <VL32>" in text
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=2, stores=1, buffers=None, extra=("ave.hir.vand", ))
    print("IR PASS: vand full/ZEROING/all-false/MERGING")


def _launch():
    s = SHAPE
    xi = torch.randint(0, 1 << 32, (s["lanes"], ), dtype=torch.int64)
    yi = torch.randint(0, 1 << 32, (s["lanes"], ), dtype=torch.int64)
    out = torch.empty((4, s["lanes"]), dtype=torch.uint32, device="npu")
    vand[(1, )](xi.to(torch.uint32).to("npu"), yi.to(torch.uint32).to("npu"), out, **constexprs())
    torch.npu.synchronize()
    full = (xi & yi) & 0xffffffff
    ref = torch.zeros((4, s["lanes"]), dtype=torch.int64)
    ref[0] = full
    ref[1, :s["active"]] = full[:s["active"]]
    ref[3, :s["active"]] = full[:s["active"]]
    ref[3, s["active"]:] = (xi & 0xffffffff)[s["active"]:]
    return out, ref.to(torch.uint32)


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vand generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    out, ref = _launch()
    got = out.cpu()
    torch.testing.assert_close(got.view(torch.int32), ref.view(torch.int32), rtol=0, atol=0)
    print("VAND OK: exact bits, ZEROING and MERGING masks")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vand", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
