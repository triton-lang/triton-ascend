"""TLA 3-in-1 test: vmull (widening multiply) exact bits on i32.

vmull(x, y) returns (low, high) halves of the full 64-bit signed product,
both carried in i32 registers.

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).
Rows: 0 full-mask low halves, 1 full-mask high halves,
     2 prefix-ZEROING low, 3 prefix-ZEROING high.
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
SIGNATURE = {"X": "*i32", "Y": "*i32", "O": "*i32"}


@tla.jit
def vmull(X, Y, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.int32, space="ub")
    y = tla.alloc_buffer((N, ), tl.int32, space="ub")
    o = tla.alloc_buffer((4, N), tl.int32, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    tla.copy(tla.view(Y, (N, )), y)
    with tla.scope.simd():
        a = tla.simd.vlds(x[0])
        b = tla.simd.vlds(y[0])
        prefix = tla.simd.pge(ACTIVE, N)
        lo, hi = tla.simd.vmull(a, b)
        tla.simd.vsts(o[0, 0], lo)
        tla.simd.vsts(o[1, 0], hi)
        lo_m, hi_m = tla.simd.vmull(a, b, prefix)
        tla.simd.vsts(o[2, 0], lo_m)
        tla.simd.vsts(o[3, 0], hi_m)
    tla.copy(o, tla.view(O, (4, N)))


def constexprs():
    return {"N": SHAPE["lanes"], "ACTIVE": SHAPE["active"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vmull, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.mull") == 2
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=2, stores=1, buffers=None, extra=("ave.hir.mull", ))
    print("IR PASS: vmull full/ZEROING halves")


def _launch():
    s = SHAPE
    x = torch.randint(-(1 << 30), 1 << 30, (s["lanes"], ), dtype=torch.int64)
    y = torch.randint(-(1 << 30), 1 << 30, (s["lanes"], ), dtype=torch.int64)
    out = torch.empty((4, s["lanes"]), dtype=torch.int32, device="npu")
    vmull[(1, )](x.to(torch.int32).to("npu"), y.to(torch.int32).to("npu"), out, **constexprs())
    torch.npu.synchronize()
    full = x * y
    lo = (full & 0xFFFFFFFF).to(torch.int64).to(torch.int32).to(torch.int64)
    hi = (full >> 32).to(torch.int32).to(torch.int64)
    ref = torch.zeros((4, s["lanes"]), dtype=torch.int64)
    ref[0] = lo
    ref[1] = hi
    ref[2, :s["active"]] = lo[:s["active"]]
    ref[3, :s["active"]] = hi[:s["active"]]
    return out, ref.to(torch.int32)


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vmull generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    out, ref = _launch()
    torch.testing.assert_close(out.cpu().view(torch.int32), ref.view(torch.int32), rtol=0, atol=0)
    print("VMULL OK: 64-bit product halves exact bits")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vmull", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
