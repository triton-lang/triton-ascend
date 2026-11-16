"""TLA 3-in-1 test: simd.vcmp/vsel/vcmax on a tilelang-style per-token amax.

Ported from the amax stage of ``tilelang/examples/ascend/
example_per_token_cast_to_fp8.py``: for every row, take the vector absolute
value (``T.simd.vabs`` in tilelang; ``vcmp`` + ``vsel`` + ``vneg`` here) and
reduce it with ``T.simd.vcmax`` (``vcmax`` here), writing lane 0 per row.

Semantics: out[i] = max_j |x[i, j]|, f32, N = 64 (one register per row).

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline HIVM/AVE IR structure (no NPU, no torch)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (PER_TOKEN_AMAX OK)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

M, N = 64, 64

SIGNATURE = {"X_ptr": "*fp32", "OUT_ptr": "*fp32"}


@tla.jit
def per_token_amax(X_ptr, OUT_ptr, M: tl.constexpr, N: tl.constexpr):
    x_ub = tla.alloc_buffer((M, N), tl.float32, space="ub")
    amax_ub = tla.alloc_buffer((M, ), tl.float32, space="ub")
    x = tla.view(X_ptr, (M, N), strides=(N, 1))
    out = tla.view(OUT_ptr, (M, ), strides=(1, ), offsets=(0, ))

    tla.copy(x, x_ub)
    with tla.scope.simd():
        zero = tla.simd.vfull(0.0, tl.float32, N)
        for r in tl.static_range(M):
            v = tla.simd.vlds(x_ub[r, 0], dist="NORM")
            abs_v = tla.simd.vsel(tla.simd.vcmp(v, zero, op=tla.simd.CMP.GT), v, tla.simd.vneg(v))
            tla.simd.vsts(amax_ub[r], tla.simd.vcmax(abs_v), dist="ONEPT_B32")
    tla.copy(amax_ub, out)


def constexprs():
    return {"M": M, "N": N}


def test_ir(output_dir=None):
    module, ctx = compile_offline(per_token_amax, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=1, stores=1, buffers=None)
    assert text.count("ave.hir.vcmp") == M, text
    assert text.count("ave.hir.vsel") == M, text
    assert text.count("ave.hir.vneg") == M, text
    assert text.count("ave.hir.reduction <max>") == M, text
    print(f"IR PASS: per_token_amax ({M} rows of |max| via vcmp/vsel/vcmax)")


def _launch():
    import torch
    X = torch.randn(M, N, dtype=torch.float32, device="npu")
    OUT = torch.full((M, ), -7.0, dtype=torch.float32, device="npu")
    per_token_amax[(1, )](X, OUT, **constexprs())
    torch.npu.synchronize()
    return X, OUT


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: per_token_amax")


def test_run():
    import numpy as np
    X, OUT = _launch()
    ref = np.abs(X.cpu().numpy()).max(axis=1)
    ok = np.allclose(OUT.cpu().numpy(), ref, atol=0.0, rtol=0.0)
    print("PER_TOKEN_AMAX", "OK" if ok else "MISMATCH")
    assert ok, "per_token_amax accuracy mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: per_token_amax", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
