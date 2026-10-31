"""TLA 3-in-1 test: inter-core synchronization (cube -> vector handoff).

Real kernel, modelled on ``cube/gemm_bias.py`` (tilelang ascend gemm + vector
elementwise):

    C = A @ B                       (cube: nd2nz x2 -> mmadL1 -> fixpipe to GM)
    cube --block_set_flag--> flag 0 (raised only after C is in GM)
    vector --block_wait_flag--> GM -> UB, O = C + bias over every 64-lane tile
    vector -> O (GM)

The flag protects the real inter-core dependency: the vector stage must not
read C before the cube's fixpipe has written it to GM.

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline HIVM/AVE IR structure (no NPU, no torch)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (SYNC_INTER OK)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

M, N, K = 64, 64, 64
BIAS = 1.0
FLAG = 0

SIGNATURE = {"A_ptr": "*fp16", "B_ptr": "*fp16", "C_ptr": "*fp32", "O_ptr": "*fp32"}


@triton.jit
def sync_inter(A_ptr, B_ptr, C_ptr, O_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BIAS: tl.constexpr,
               FLAG: tl.constexpr):
    tl.static_assert(K % 16 == 0 and M % 16 == 0 and N % 16 == 0, "M/N/K must be multiples of 16 (cube tile)")
    tl.static_assert((M * N) % 64 == 0, "M*N must be a multiple of 64")

    a_l1 = tla.alloc_buffer((M, K), tl.float16, space="l1")
    b_l1 = tla.alloc_buffer((K, N), tl.float16, space="l1")
    c_l0c = tla.alloc_buffer((M, N), tl.float32, space="l0c")
    c_ub = tla.alloc_buffer((M, N), tl.float32, space="ub")

    a = tla.view(A_ptr, (M, K), strides=(K, 1))
    b = tla.view(B_ptr, (K, N), strides=(N, 1))
    c = tla.view(C_ptr, (M, N), strides=(N, 1))
    o = tla.view(O_ptr, (M, N), strides=(N, 1))

    with tla.scope.cube():
        tla.copy(a, a_l1)
        tla.copy(b, b_l1)
        tla.cube.mmad(a_l1, b_l1, c_l0c, clear_accum=True)
        tla.copy(c_l0c, c)
        # Raise the flag only after fixpipe has written C to GM.
        tla.sync.block_set_flag(0, "FIX", FLAG)

    # Vector side: wait for the cube result, then post-process the whole tile.
    tla.sync.block_wait_flag("MTE2", FLAG)
    tla.copy(c, c_ub)
    with tla.scope.simd():
        bias = tla.simd.vfull(BIAS, tl.float32, 64)
        for i in tl.static_range((M * N) // 64):
            row = i // (N // 64)
            col = (i % (N // 64)) * 64
            v = tla.simd.vlds(c_ub[row, col], dist="NORM")
            tla.simd.vsts(c_ub[row, col], tla.simd.vadd(v, bias), dist="NORM_B32")
    tla.copy(c_ub, o)


def constexprs():
    return {"M": M, "N": N, "K": K, "BIAS": BIAS, "FLAG": FLAG}


def test_ir(output_dir=None):
    module, ctx = compile_offline(sync_inter, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    check_lowered_ir(text, loads=1, stores=1, buffers=4, extra=("#hivm.tcore_type<CUBE>", 'mix_mode = "mix"'))
    assert text.count("hivm.hir.nd2nz") == 2, text
    assert text.count("hivm.hir.mmadL1") == 1, text
    assert text.count("hivm.hir.fixpipe") == 1, text
    assert text.count("ave.hir.vadd") == (M * N) // 64, text
    set_line = next(l.strip() for l in text.splitlines() if "hivm.hir.sync_block_set" in l)
    wait_line = next(l.strip() for l in text.splitlines() if "hivm.hir.sync_block_wait" in l)
    assert set_line.startswith("hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_FIX>] "
                               f"flag = {FLAG}"), set_line
    assert wait_line.startswith("hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE2>, <PIPE_MTE2>] "
                                f"flag = {FLAG}"), wait_line
    for line in (set_line, wait_line):
        assert "sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>" in line, line
    assert text.index("fixpipe") < text.index("sync_block_set"), \
        "the flag must follow the GM write"
    assert text.index("sync_block_wait") < text.index("hivm.hir.load"), \
        "the vector load must wait for the flag"
    print("IR PASS: sync_inter (gemm -> fixpipe -> flag -> vector bias -> O)")


def _launch():
    import torch
    A = torch.randn(M, K, dtype=torch.float16, device="npu")
    B = torch.randn(K, N, dtype=torch.float16, device="npu")
    C = torch.zeros(M, N, dtype=torch.float32, device="npu")
    O = torch.full((M, N), -7.0, dtype=torch.float32, device="npu")
    sync_inter[(1, )](A, B, C, O, enable_tla_mode=True, **constexprs())
    torch.npu.synchronize()
    return A, B, C, O


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: sync_inter")


def test_run():
    import numpy as np
    A, B, C, O = _launch()
    ref_c = (A.float() @ B.float()).cpu().numpy()
    got_c = C.cpu().numpy()
    got_o = O.cpu().numpy()
    ok_c = np.allclose(got_c, ref_c, atol=1e-2, rtol=1e-2)
    ok_o = np.allclose(got_o, ref_c + np.float32(BIAS), atol=1e-2, rtol=1e-2)
    print("SYNC_INTER_GEMM", "OK" if ok_c else "MISMATCH")
    print("SYNC_INTER_BIAS", "OK" if ok_o else "MISMATCH")
    assert ok_c and ok_o, "sync_inter accuracy mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: sync_inter", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
