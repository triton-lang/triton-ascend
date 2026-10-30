"""TLA 3-in-1 test: single-tile flash-attention forward with direct UB->L1 P handoff.

Semantics: O[BM, N] = softmax(Q[BM, D] @ K[BN, D]^T / sqrt(D)) @ V[BN, N]

A5 SCENARIO: AIC computes S and O. AIV computes one softmax row per 64-lane
f32 SIMD register, then writes P into a group-major zN-friendly UB layout with
``ave.hir.store_with_stride`` (the vsstb path). The resulting UB buffer is copied
directly to L1; P does not round-trip through GM and no 3-D transpose template is
used. ``P_ptr`` remains in the public signature for interface parity.

Inter-core handoffs use CANN block flags:
  FLAG_S : AIC fixpipe wrote S to UB  -> AIV softmax reads S
  FLAG_P : AIV copied P to L1         -> AIC loads P as the MMAD input

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline HIVM/AVE IR structure (2 mmadL1 / 2 fixpipe / 3 nd2nz,
            64 strided stores, 1 UB->L1 copy, 2 sync set+wait)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (FLASH_ATTN OK vs torch reference)

v1 scope: single tile (BM x BN x D), no K-loop / causal mask.
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import torch
import triton.language as tl
import triton.language.extra.tla as tla
import triton.language.extra.cann.extension as cann

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

# BM query rows, BN key/value rows, D feature dim, N output cols.
SHAPE = dict(BM=64, BN=64, D=128, N=64, FLAG_S=0, FLAG_P=1)

SIGNATURE = {"Q_ptr": "*fp16", "K_ptr": "*fp16", "V_ptr": "*fp32", "P_ptr": "*fp32", "O_ptr": "*fp32"}


@tla.jit
def flash_attn(Q_ptr, K_ptr, V_ptr, P_ptr, O_ptr, BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr, N: tl.constexpr,
               SCALE: tl.constexpr, FLAG_S: tl.constexpr, FLAG_P: tl.constexpr):
    tl.static_assert(D % 16 == 0 and BM % 16 == 0 and BN % 16 == 0 and N % 16 == 0,
                     "BM/BN/D/N must be multiples of 16 (cube tile)")
    tl.static_assert(BN == 64, "v1: softmax row must fit one 64-lane f32 register")

    # AIC: S = Q @ K^T  (f16 cube inputs, f32 L0C).
    s_ub = tla.alloc_buffer((BM, BN), tl.float32, space="ub")
    q_blk = tla.view(Q_ptr, (BM, D), strides=(D, 1))
    k_blk = tla.view(K_ptr, (D, BN), strides=(BN, 1))  # contiguous K^T input
    with tla.scope.cube():
        q_l1 = tla.alloc_buffer((D // 16, BM // 16, 16, 16), tl.float16, space="l1")
        k_l1 = tla.alloc_buffer((BN // 16, D // 16, 16, 16), tl.float16, space="l1")
        s_l0c = tla.alloc_buffer((BN // 16, BM // 16, 16, 16), tl.float32, space="l0c")
        tla.copy(q_blk, q_l1)  # hivm.hir.nd2nz
        tla.copy(k_blk, k_l1)  # hivm.hir.nd2nz
        tla.cube.mmad(q_l1, k_l1, s_l0c, clear_accum=True)  # hivm.hir.mmadL1
        tla.copy(s_l0c, s_ub)  # hivm.hir.fixpipe (NZ -> ND into UB)
        cann.sync_block_set("cube", "vector", FLAG_S, cann.PIPE.PIPE_FIX, cann.PIPE.PIPE_FIX)

    # AIV: row-wise softmax of S in place (each row = one 64-lane f32 reg).
    # P no longer round-trips through GM; P_ptr remains for interface parity.
    p_l1 = tla.alloc_buffer((BN // 8, BM // 16, 16, 8), tl.float32, space="l1")
    with tla.scope.vector():
        # Physical zN destination: group-major with one 8-value row block.
        p_t = tla.alloc_buffer((BM, BN), tl.float32, space="ub")
        cann.sync_block_wait("cube", "vector", FLAG_S, cann.PIPE.PIPE_V, cann.PIPE.PIPE_V)
        with tla.scope.simd():
            one = tla.simd.vfull(1.0, tl.float32, 64)
            sc = tla.simd.vfull(SCALE, tl.float32, 64)
            for r in tl.static_range(BM):
                v = tla.simd.vmul(tla.simd.vlds(s_ub[r, 0]), sc)
                m = tla.simd.vreduce_max(v)  # row max, broadcast
                v = tla.simd.vadd(v, tla.simd.vneg(m))
                e = tla.simd.vexp(v)  # ave.hir.vexp
                l = tla.simd.vbroadcast(tla.simd.vreduce_sum(e), low=True)  # row sum
                p = tla.simd.vmul(e, tla.simd.vdiv(one, l))
                # p_t[g, r, l] = p[r, g * 8 + l].  A 64-lane f32 store
                # emits eight 8-lane blocks separated by BM * 8 elements.
                tla.simd.vsts_stride(p_t[0, r * 8], p, stride=BM * 8)
        # AIV: direct UB->L1 handoff of P, then signal the cube on the MTE3 pipe.
        # The 2-D UB and 4-D zN L1 buffers are contiguous and have equal extents.
        tla.copy(p_t, p_l1)  # hivm.hir.copy UB->L1 (MTE3)
        cann.sync_block_set("vector", "cube", FLAG_P, cann.PIPE.PIPE_MTE3, cann.PIPE.PIPE_MTE3)

    # AIC: O = P @ V (f32 inputs -> f32 L0C), fixpipe straight to GM.
    v_blk = tla.view(V_ptr, (BN, N), strides=(N, 1))
    o_blk = tla.view(O_ptr, (BM, N), strides=(N, 1))
    with tla.scope.cube():
        v_l1 = tla.alloc_buffer((N // 8, BN // 16, 16, 8), tl.float32, space="l1")
        o_l0c = tla.alloc_buffer((N // 16, BM // 16, 16, 16), tl.float32, space="l0c")
        cann.sync_block_wait("vector", "cube", FLAG_P, cann.PIPE.PIPE_MTE1, cann.PIPE.PIPE_MTE1)
        tla.copy(v_blk, v_l1)  # hivm.hir.nd2nz
        tla.cube.mmad(p_l1, v_l1, o_l0c, clear_accum=True)  # hivm.hir.mmadL1
        tla.copy(o_l0c, o_blk)  # hivm.hir.fixpipe (NZ -> ND into GM)


def constexprs():
    s = SHAPE
    return {
        "BM": s["BM"], "BN": s["BN"], "D": s["D"], "N": s["N"], "SCALE": 1.0 / s["D"]**0.5, "FLAG_S": s["FLAG_S"],
        "FLAG_P": s["FLAG_P"]
    }


# ---- torch reference (same semantics/shape) ----


def _ref_torch(Q, K, V):
    """Host reference: O = softmax(Q @ K^T / sqrt(D)) @ V."""
    s = SHAPE
    p = torch.softmax(Q.float() @ K.float().T * (1.0 / s["D"]**0.5), dim=-1)  # (BM, BN)
    return p @ V  # (BM, N)


def test_ir(output_dir=None):
    module, ctx = compile_offline(flash_attn, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert "!tt.ptr" in text, "route A entry must keep pointer args at AST stage"
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(
        text, loads=0, stores=0, buffers=8, extra=("#hivm.address_space<cbuf>", "#hivm.address_space<cc>",
                                                   "#hivm.tcore_type<CUBE>", "#hivm.func_core_type<AIV>"))
    assert text.count("hivm.hir.nd2nz") == 3, "GM->L1 x3 (q, k, v)"
    transpose_count = text.count("hivm.hir.vtranspose")
    strided_store_count = text.count("ave.hir.store_with_stride")
    vector_function_count = text.count("hivm.vector_function")
    print(f"V8 IR: vtranspose={transpose_count}, strided_store={strided_store_count}, "
          f"vector_function={vector_function_count}")
    assert transpose_count == 0, "v8 must not use UB transpose"
    assert strided_store_count == 64, "one strided store per softmax row"
    assert vector_function_count == 1, "softmax should be the only SIMD VF"
    assert text.count("hivm.hir.copy") == 1, "direct P UB->L1 copy"
    assert text.count("hivm.hir.mmadL1") == 2, "two gemms: S and O"
    assert text.count("hivm.hir.fixpipe") == 2, "L0C->UB/GM x2 (S, O)"
    assert text.count("hivm.hir.sync_block_set") == 2, "AIC/AIV signals x2"
    assert text.count("hivm.hir.sync_block_wait") == 2, "AIV/AIC waits x2"
    assert text.count("ave.hir.vexp") == SHAPE["BM"], "one exp per softmax row"
    assert text.count("ave.hir.vload") == SHAPE["BM"], "one row load per softmax row"
    assert text.count("hivm.hir.store") == 0, "P must not round-trip through GM"

    def nth(pat, k):
        """0-based index of the k-th occurrence (k is 1-based)."""
        pos = -1
        for _ in range(k):
            pos = text.index(pat, pos + 1)
        return pos

    assert text.index("hivm.hir.fixpipe") < text.index("hivm.hir.sync_block_set"), "S flag after fixpipe"
    assert text.index("hivm.hir.sync_block_wait") < text.index("ave.hir.vload"), "softmax waits for S"
    assert nth("hivm.hir.sync_block_set", 2) > nth("hivm.hir.copy", 1), "P flag after the UB->L1 copy"
    assert nth("hivm.hir.sync_block_wait", 2) < nth("hivm.hir.nd2nz", 3), "AIC waits before loading V and using P"
    assert 'mix_mode = "mix"' in text, "cube+vector kernel must be tagged mix"
    print("IR PASS: flash_attn UB->L1 HIVM structure ok"
          " (3 nd2nz / 2 mmadL1 / 2 fixpipe / 64 vsstb / 1 UB->L1 copy)")


def _launch():
    s = SHAPE
    Q = torch.randn(s["BM"], s["D"], dtype=torch.float16, device="npu")
    K = torch.randn(s["BN"], s["D"], dtype=torch.float16, device="npu")
    V = torch.randn(s["BN"], s["N"], dtype=torch.float32, device="npu")
    P = torch.full((s["BM"], s["BN"]), 0.0, dtype=torch.float32, device="npu")
    O = torch.full((s["BM"], s["N"]), 7.0, dtype=torch.float32, device="npu")
    flash_attn[(1, )](Q, K.T.contiguous(), V, P, O, **constexprs())
    torch.npu.synchronize()
    return Q, K, V, O


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: flash_attn artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    Q, K, V, O = _launch()
    Ot = _ref_torch(Q, K, V)
    oo, ot = O.cpu(), Ot.cpu()
    print("O[:4,:4]  :", oo[:4, :4].tolist())
    print("ref[:4,:4]:", ot[:4, :4].tolist())
    ok = torch.allclose(oo, ot, atol=1e-2, rtol=1e-2)
    print("FLASH_ATTN", "OK" if ok else "MISMATCH", "BM", SHAPE["BM"], "BN", SHAPE["BN"], "D", SHAPE["D"], "N",
          SHAPE["N"])
    assert ok, "flash_attn e2e mismatch vs torch reference"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: flash_attn", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
