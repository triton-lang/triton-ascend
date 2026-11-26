"""TLA 3-in-1 test: vmuls (vector-scalar multiply) masks and modes on f32.

Stages: ir (offline HIVM), compile (kernel artifacts), run (numerics).
Rows: 0 full-mask, 1 prefix ZEROING, 2 all-false ZEROING, 3 prefix MERGING
(MERGING keeps the source vector on the inactive lanes).
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
SIGNATURE = {"X": "*fp32", "O": "*fp32"}
SCALAR = 2.5


@tla.jit
def vmuls(X, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    x = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o = tla.alloc_buffer((4, N), tl.float32, space="ub")
    tla.copy(tla.view(X, (N, )), x)
    with tla.scope.simd():
        a = tla.simd.vlds(x[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        none = tla.simd.vcmp(idx, idx, op=tla.simd.CMP.NE)
        tla.simd.vsts(o[0, 0], tla.simd.vmuls(a, 2.5))
        tla.simd.vsts(o[1, 0], tla.simd.vmuls(a, 2.5, prefix))
        tla.simd.vsts(o[2, 0], tla.simd.vmuls(a, 2.5, none))
        tla.simd.vsts(o[3, 0], tla.simd.vmuls(a, 2.5, mask=prefix, mode=tla.simd.MODE.MERGING))
    tla.copy(o, tla.view(O, (4, N)))


def constexprs():
    return {"N": SHAPE["lanes"], "ACTIVE": SHAPE["active"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vmuls, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert text.count("ave.hir.vmuls") == 4
    assert text.count("ave.hir.vsel") == 1, "only the MERGING row adds vsel"
    assert "ave.hir.pge <VL32>" in text
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=1, stores=1, buffers=None, extra=("ave.hir.vmuls", ))
    print("IR PASS: vmuls full/ZEROING/all-false/MERGING")


def _launch():
    s = SHAPE
    x = torch.randn(s["lanes"], dtype=torch.float32)
    out = torch.empty((4, s["lanes"]), dtype=torch.float32, device="npu")
    vmuls[(1, )](x.to("npu"), out, **constexprs())
    torch.npu.synchronize()
    full = x * SCALAR
    ref = torch.zeros((4, s["lanes"]), dtype=torch.float32)
    ref[0] = full
    ref[1, :s["active"]] = full[:s["active"]]
    ref[3, :s["active"]] = full[:s["active"]]
    ref[3, s["active"]:] = x[s["active"]:]
    return out, ref


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vmuls generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    out, ref = _launch()
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-6, atol=1e-6)
    print("VMULS OK: full/ZEROING/all-false/MERGING")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vmuls", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
