"""TLA 3-in-1 test: E5M2 strided UB store, full and partial masks.

Stages: ir (offline HIVM), compile (kernel artifacts), run (raw bytes).
Inputs are regenerated randomly on every run.
No float conversion: preserve E5M2 NaNs/signed zeros and untouched sentinels.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

SHAPE = dict(lanes=256, rows=8, stride=64, active=32)
SIGNATURE = {"X": "*fp8e5", "O": "*fp8e5"}


@tla.jit
def vsstb(X, O, N: tl.constexpr, STRIDE: tl.constexpr, ACTIVE: tl.constexpr):
    src = tla.alloc_buffer((1, N), tl.float8e5, space="ub")
    dst = tla.alloc_buffer((16, STRIDE), tl.float8e5, space="ub")
    tla.copy(tla.view(X, (1, N)), src)
    tla.copy(tla.view(O, (16, STRIDE)), dst)
    with tla.scope.simd():
        v = tla.simd.vlds(src[0, 0])
        mask = tla.simd.pge(ACTIVE, N)
        tla.simd.vsstb(v, dst[0, N // 8], STRIDE)
        tla.simd.vsstb(v, dst[8, N // 8], STRIDE, mask)
    tla.copy(dst, tla.view(O, (16, STRIDE)))


def constexprs():
    return {"N": SHAPE["lanes"], "STRIDE": SHAPE["stride"], "ACTIVE": SHAPE["active"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vsstb, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert "f8E5M2" in text
    assert text.count("ave.hir.store_with_stride") == 2
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=2, stores=1, buffers=None, extra=("ave.hir.store_with_stride", ))
    print("IR PASS: vsstb E5M2 full/partial strided stores")


def _launch():
    s = SHAPE
    raw = torch.randint(0, 256, (s["lanes"], ), dtype=torch.int64).to(torch.uint8)
    initial = torch.full((16, s["stride"]), 0xA5, dtype=torch.uint8)
    out = initial.view(torch.float8_e5m2).to("npu")
    vsstb[(1, )](raw.view(torch.float8_e5m2).to("npu"), out, **constexprs())
    torch.npu.synchronize()
    ref = initial.clone()
    block = s["lanes"] // s["rows"]
    ref[:8, block:2 * block] = raw.reshape(8, block)
    for i in range(s["active"]):
        ref[8 + i // block, block + i % block] = raw[i]
    return out, ref


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vsstb generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    out, ref = _launch()
    torch.testing.assert_close(out.cpu().view(torch.uint8), ref, rtol=0, atol=0)
    print("VSSTB OK: exact E5M2 bytes, stride/mask and untouched sentinels")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vsstb", "ir" if do_ir else "", "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
