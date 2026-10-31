"""TLA 3-in-1 test: gemm_bias (TLA direct port of tilelang ascend gemm + vector elementwise).

Semantics: C[M, N] = A[M, K] @ B[K, N] + bias  (vector stage adds a constant per 64-lane)

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline-compile the HIVM IR handed to npuir and assert
            (2 nd2nz / 1 mmadL1 / 1 fixpipe, 1 store, cbuf/cc/ub alloc,
            cube scope carries only hivm.tcore_type<CUBE> (not a VF),
            vector scope keeps the AIV VF markers, AIC->AIV sync flags)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (GEMM_BIAS OK; standard Triton
            reference gemm_bias_triton, same semantics/shape ->
            TRITON_GEMM_BIAS OK)

v1 scope: single block, one cube scope, one vector scope (sync wait) and
one simd scope; no pipelining / multi-core.
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

# default shapes: M, N, K (K must be a multiple of 16; M*N of 64)
SHAPE = dict(M=64, N=64, K=64, BIAS=1.0, FLAG=0)

# offline compile signature (pointer entry takes pointer args).
SIGNATURE = {"A_ptr": "*fp16", "B_ptr": "*fp16", "C_ptr": "*fp32"}


@triton.jit
def gemm_bias(A_ptr, B_ptr, C_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BIAS: tl.constexpr,
              FLAG: tl.constexpr):
    tl.static_assert(K % 16 == 0 and M % 16 == 0 and N % 16 == 0, "M/N/K must be multiples of 16 (cube tile)")
    tl.static_assert((M * N) % 64 == 0, "M*N must be a multiple of 64 (vector lanes)")
    a_l1 = tla.alloc_buffer((K // 16, M // 16, 16, 16), tl.float16, space="l1")
    b_l1 = tla.alloc_buffer((N // 16, K // 16, 16, 16), tl.float16, space="l1")
    c_l0c = tla.alloc_buffer((N // 16, M // 16, 16, 16), tl.float32, space="l0c")
    c_ub = tla.alloc_buffer((M, N), tl.float32, space="ub")
    a_blk = tla.view(A_ptr, (M, K), strides=(K, 1))
    b_blk = tla.view(B_ptr, (K, N), strides=(N, 1))
    c_blk = tla.view(C_ptr, (M, N), strides=(N, 1))
    with tla.scope.cube():
        tla.copy(a_blk, a_l1)  # hivm.hir.nd2nz (ND -> NZ)
        tla.copy(b_blk, b_l1)  # hivm.hir.nd2nz
        tla.cube.mmad(a_l1, b_l1, c_l0c, clear_accum=True)  # hivm.hir.mmadL1
        tla.copy(c_l0c, c_ub)  # hivm.hir.fixpipe (NZ -> ND into UB)
        # AIC: signal the handoff only after fixpipe has written c_ub.
        tla.sync.block_set_flag(0, "FIX", FLAG)
    with tla.scope.vector():
        # AIV: wait for the cube handoff before the SIMD stage reads c_ub.
        tla.sync.block_wait_flag("V", FLAG)
        with tla.scope.simd():
            bias = tla.simd.vfull(BIAS, tl.float32, 64)
            for i in tl.range((M * N) // 64):
                row = i // (N // 64)
                col = (i % (N // 64)) * 64
                v = tla.simd.vlds(c_ub[row, col])  # NORM, ND row-major read
                tla.simd.vsts(c_ub[row, col], tla.simd.vadd(v, bias))
        tla.copy(c_ub, c_blk)  # hivm.hir.store (MTE3 UB->GM)


def constexprs():
    s = SHAPE
    return {"M": s["M"], "N": s["N"], "K": s["K"], "BIAS": s["BIAS"], "FLAG": s["FLAG"]}


# ---- standard Triton reference (same semantics/shape) ----


@triton.jit
def gemm_bias_triton(A_ptr, B_ptr, C_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BIAS: tl.constexpr):
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    a = tl.load(A_ptr + rows[:, None] * K + cols[None, :])
    b = tl.load(B_ptr + cols[:, None] * K + rows[None, :])
    c = tl.sum(a[:, :, None] * b[None, :, :], axis=1) + BIAS
    tl.store(C_ptr + rows[:, None] * N + cols[None, :], c)


def test_ir(output_dir=None):
    module, ctx = compile_offline(gemm_bias, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert "!tt.ptr" in text, "route A entry must keep pointer args at AST stage"
    assert text.count("hivm.hir.pointer_cast") == 3, "tla.view(ptr) x3"
    assert text.count("memref.reinterpret_cast") == 3, "tla.view(ptr) x3"
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(
        text, loads=0, stores=1, buffers=4, extra=("#hivm.address_space<cbuf>", "#hivm.address_space<cc>",
                                                   "#hivm.tcore_type<CUBE>", "#hivm.func_core_type<AIV>"))
    assert text.count("hivm.hir.nd2nz") == 2, "GM->L1 x2"
    assert text.count("hivm.hir.mmadL1") == 1, "one gemm"
    assert text.count("hivm.hir.fixpipe") == 1, "L0C->UB"
    assert text.count("hivm.hir.sync_block_set") == 1, "AIC signals the UB handoff"
    assert text.count("hivm.hir.sync_block_wait") == 1, "AIV waits for the UB handoff"
    assert text.index("hivm.hir.fixpipe") < text.index("hivm.hir.sync_block_set"), \
        "the flag must follow the fixpipe write"
    assert text.index("hivm.hir.sync_block_wait") < text.index("ave.hir.vload"), \
        "the SIMD read must wait for the cube handoff"
    assert 'mix_mode = "mix"' in text, "cube+vector kernel must be tagged mix"
    # The AIC (cube) scope is not a vector function: hivm.vector_function may
    # appear only on the SIMD scope (exactly once), and the tla.memory_effect
    # attribute was dropped with the VF effect recording.
    assert text.count("hivm.vector_function") == 1, "only the SIMD scope is a VF"
    assert "hivm.memory_effect" not in text, "VF effect recording was removed"
    print("IR PASS: gemm_bias HIVM structure ok"
          " (2 nd2nz / 1 mmadL1 / 1 fixpipe / 1 sync set+wait / 1 store)")


def _launch():
    s = SHAPE
    A = torch.randn(s["M"], s["K"], dtype=torch.float16, device="npu")
    B = torch.randn(s["K"], s["N"], dtype=torch.float16, device="npu")
    C = torch.full((s["M"], s["N"]), 7.0, dtype=torch.float32, device="npu")
    gemm_bias[(1, )](A, B, C, **constexprs())
    torch.npu.synchronize()
    return C, (A.float() @ B.float() + s["BIAS"])


def _launch_triton():
    s = SHAPE
    A = torch.randn(s["M"], s["K"], dtype=torch.float16, device="npu")
    B = torch.randn(s["K"], s["N"], dtype=torch.float16, device="npu")
    C = torch.full((s["M"], s["N"]), 7.0, dtype=torch.float32, device="npu")
    gemm_bias_triton[(1, )](A, B, C, **constexprs())
    torch.npu.synchronize()
    return C, (A.float() @ B.float() + s["BIAS"])


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: gemm_bias generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    C, ref = _launch()
    oc, rc = C.cpu(), ref.cpu()
    print("C[:4,:4]  :", oc[:4, :4].tolist())
    print("ref[:4,:4]:", rc[:4, :4].tolist())
    ok = torch.allclose(oc, rc, atol=0.05)
    print("GEMM_BIAS", "OK" if ok else "MISMATCH", "M", SHAPE["M"], "N", SHAPE["N"], "K", SHAPE["K"])
    assert ok, "gemm_bias e2e mismatch"
    # standard Triton reference.
    Ct, rt = _launch_triton()
    oc_t, rc_t = Ct.cpu(), rt.cpu()
    bad_t = ((oc_t - rc_t).abs() > 0.05).nonzero().flatten()
    print("TRITON_GEMM_BIAS", "OK" if bad_t.numel() == 0 else "MISMATCH", "num_mismatch:", bad_t.numel())
    assert bad_t.numel() == 0, "standard triton gemm_bias_triton mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: gemm_bias", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
