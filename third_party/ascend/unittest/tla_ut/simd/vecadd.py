"""TLA 3-in-1 test: simd.vadd on a tilelang-style pipelined vecadd.

Ported from ``tilelang/examples/ascend/example_vecadd.py`` (SIMD path): one
block copies a 4096-element tile GM -> UB, runs 64 vector-register adds of 64
f32 lanes, then stores UB -> GM. tilelang drives the same loop with
``T.simd.vld`` / ``T.simd.vadd`` / ``T.simd.vsts``.

Semantics: out = a + b, f32.

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline HIVM/AVE IR structure (no NPU, no torch)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (VECADD_SIMD OK)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

TILE = 4096  # elements per block (tilelang default tile)
LANES = 64  # f32 lanes in one 2048-bit register
N = TILE

SIGNATURE = {"A_ptr": "*fp32", "B_ptr": "*fp32", "O_ptr": "*fp32"}


@tla.jit
def vecadd(A_ptr, B_ptr, O_ptr, N: tl.constexpr):
    a_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    b_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    o_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    a = tla.view(A_ptr, (N, ), strides=(1, ), offsets=(0, ))
    b = tla.view(B_ptr, (N, ), strides=(1, ), offsets=(0, ))
    o = tla.view(O_ptr, (N, ), strides=(1, ), offsets=(0, ))

    tla.copy(a, a_ub)
    tla.copy(b, b_ub)
    with tla.scope.simd():
        for i in tl.static_range(N // 64):
            av = tla.simd.vlds(a_ub[i * 64], dist="NORM")
            bv = tla.simd.vlds(b_ub[i * 64], dist="NORM")
            tla.simd.vsts(o_ub[i * 64], tla.simd.vadd(av, bv), dist="NORM_B32")
    tla.copy(o_ub, o)


def constexprs():
    return {"N": N}


def test_ir(output_dir=None):
    module, ctx = compile_offline(vecadd, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=2, stores=1, buffers=None)
    tiles = N // LANES
    assert text.count("ave.hir.vadd") == tiles, text
    assert text.count("ave.hir.vload") == 2 * tiles, text
    assert text.count("ave.hir.masked_store") == tiles, text
    print(f"IR PASS: vecadd ({tiles} register adds over {N} elements)")


def _launch():
    import torch
    A = torch.randn(N, dtype=torch.float32, device="npu")
    B = torch.randn(N, dtype=torch.float32, device="npu")
    O = torch.full((N, ), -7.0, dtype=torch.float32, device="npu")
    vecadd[(1, )](A, B, O, **constexprs())
    torch.npu.synchronize()
    return A, B, O


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vecadd")


def test_run():
    import numpy as np
    A, B, O = _launch()
    ok = np.allclose(O.cpu().numpy(), A.cpu().numpy() + B.cpu().numpy(), atol=0.0, rtol=0.0)
    print("VECADD_SIMD", "OK" if ok else "MISMATCH")
    assert ok, "vecadd accuracy mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vecadd", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
