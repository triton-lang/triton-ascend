"""TLA 3-in-1 test: gemm_basic (TLA direct port of tilelang ascend gemm).

Semantics: C[M, N] = A[M, K] @ B[K, N]  (fp16 in, f32 accumulate in L0C)

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline-compile the HIVM IR handed to npuir and assert
            (2 nd2nz / 1 mmadL1 / 1 fixpipe, cbuf/cc alloc, cube scope
            carries only hivm.tcore_type<CUBE> and is not a VF)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (GEMM_BASIC OK; standard Triton
            reference gemm_triton, same semantics/shape -> TRITON_GEMM_BASIC OK)

v1 scope: single block (M=N=K=64), clear_accum=True one-shot; no K-loop
pipelining / multi-core.
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

# default shapes: M, N, K (K must be a multiple of 16)
SHAPE = dict(M=64, N=64, K=64)

# offline compile signature (pointer entry takes pointer args).
SIGNATURE = {"A_ptr": "*fp16", "B_ptr": "*fp16", "C_ptr": "*fp32"}


@triton.jit
def gemm_basic(A_ptr, B_ptr, C_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    tl.static_assert(K % 16 == 0 and M % 16 == 0 and N % 16 == 0, "M/N/K must be multiples of 16 (cube tile)")
    a_l1 = tla.alloc_buffer((M, K), tl.float16, space="l1")
    b_l1 = tla.alloc_buffer((K, N), tl.float16, space="l1")
    c_l0c = tla.alloc_buffer((M, N), tl.float32, space="l0c")
    a_blk = tla.view(A_ptr, (M, K), strides=(K, 1))
    b_blk = tla.view(B_ptr, (K, N), strides=(N, 1))
    c_blk = tla.view(C_ptr, (M, N), strides=(N, 1))
    with tla.scope.cube():
        tla.copy(a_blk, a_l1)  # hivm.hir.nd2nz
        tla.copy(b_blk, b_l1)  # hivm.hir.nd2nz
        tla.cube.mmad(a_l1, b_l1, c_l0c, clear_accum=True)  # hivm.hir.mmadL1
        tla.copy(c_l0c, c_blk)  # hivm.hir.fixpipe (2D dst -> nz2nd)


def constexprs():
    s = SHAPE
    return {"M": s["M"], "N": s["N"], "K": s["K"]}


# ---- standard Triton reference (same semantics/shape) ----


@triton.jit
def gemm_triton(A_ptr, B_ptr, C_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    a = tl.load(A_ptr + rows[:, None] * K + cols[None, :])  # (M, K)
    b = tl.load(B_ptr + cols[:, None] * K + rows[None, :])  # (N, K) -> load B(K, N) transposed
    c = tl.sum(a[:, :, None] * b[None, :, :], axis=1)  # (M, N) f32 accumulate
    tl.store(C_ptr + rows[:, None] * N + cols[None, :], c)


def test_ir(output_dir=None):
    module, ctx = compile_offline(gemm_basic, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert "!tt.ptr" in text, "route A entry must keep pointer args at AST stage"
    assert text.count("hivm.hir.pointer_cast") == 3, "tla.view(ptr) x3"
    assert text.count("memref.reinterpret_cast") == 3, "tla.view(ptr) x3"
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(text, loads=0, stores=0, buffers=3,
                     extra=("#hivm.address_space<cbuf>", "#hivm.address_space<cc>", "#hivm.tcore_type<CUBE>"))
    assert text.count("hivm.hir.nd2nz") == 2, "GM->L1 x2"
    assert text.count("hivm.hir.mmadL1") == 1, "one gemm"
    assert text.count("hivm.hir.fixpipe") == 1, "L0C->GM"
    assert 'mix_mode = "mix"' in text, "kernel with cube work must be tagged mix"
    # Pure-cube kernel: AIC scope is not a vector function, so no VF markers
    # (hivm.vector_function / func_core_type / outline) may appear.
    assert text.count("hivm.vector_function") == 0, "cube scope must not be a VF"
    assert "hivm.memory_effect" not in text, "VF effect recording was removed"
    print("IR PASS: gemm_basic HIVM structure ok (2 nd2nz / 1 mmadL1 / 1 fixpipe)")


def _launch():
    s = SHAPE
    A = torch.randn(s["M"], s["K"], dtype=torch.float16, device="npu")
    B = torch.randn(s["K"], s["N"], dtype=torch.float16, device="npu")
    C = torch.full((s["M"], s["N"]), 7.0, dtype=torch.float32, device="npu")
    gemm_basic[(1, )](A, B, C, **constexprs())
    torch.npu.synchronize()
    return C, (A.float() @ B.float())


def _launch_triton():
    s = SHAPE
    A = torch.randn(s["M"], s["K"], dtype=torch.float16, device="npu")
    B = torch.randn(s["K"], s["N"], dtype=torch.float16, device="npu")
    C = torch.full((s["M"], s["N"]), 7.0, dtype=torch.float32, device="npu")
    gemm_triton[(1, )](A, B, C, **constexprs())
    torch.npu.synchronize()
    return C, (A.float() @ B.float())


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: gemm_basic generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    C, ref = _launch()
    oc, rc = C.cpu(), ref.cpu()
    print("C[:4,:4]  :", oc[:4, :4].tolist())
    print("ref[:4,:4]:", rc[:4, :4].tolist())
    ok = torch.allclose(oc, rc, atol=0.05)
    print("GEMM_BASIC", "OK" if ok else "MISMATCH", "M", SHAPE["M"], "N", SHAPE["N"], "K", SHAPE["K"])
    assert ok, "gemm_basic e2e mismatch"
    # standard Triton reference.
    Ct, rt = _launch_triton()
    oc_t, rc_t = Ct.cpu(), rt.cpu()
    bad_t = ((oc_t - rc_t).abs() > 0.05).nonzero().flatten()
    print("TRITON_GEMM_BASIC", "OK" if bad_t.numel() == 0 else "MISMATCH", "num_mismatch:", bad_t.numel())
    assert bad_t.numel() == 0, "standard triton gemm_triton mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: gemm_basic", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
