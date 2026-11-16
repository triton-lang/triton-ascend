"""TLA 3-in-1 test: single-tile flash-attention forward.

Semantics: O[BM, N] = softmax(Q[BM, D] @ K[BN, D]^T / sqrt(D)) @ V[BN, N]

Inter-core (AIC <-> AIV) handoffs are synchronized by explicit FFTS flags
(``tla.sync.block_set_flag`` / ``block_wait_flag``); intra-core ordering
is left to the compiler.  Two flags, one per handoff:

  FLAG_S : AIC fixpipe wrote S to UB  -> AIV softmax reads S
  FLAG_P : AIV stored P to GM         -> AIC loads P back into L1

P must round-trip through GM because there is no UB -> L1 path yet; it is
kept f32 (softmax output) and V is fp32 so both mmad inputs share dtype.

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline HIVM/AVE IR structure (2 mmadL1 / 2 fixpipe / 4 nd2nz,
            2 sync set+wait, 1 store, exp per softmax row)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (FLASH_ATTN OK vs torch reference)

v1 scope: single tile (BM x BN x D), no K-loop / causal mask.
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import torch
import triton.language as tl
import triton.language.extra.tla as tla

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
        # AIC: signal the softmax handoff only after fixpipe has written S.
        tla.sync.block_set_flag(tla.sync.MODE.INTER, tla.sync.PIPE.F, FLAG_S)

    # AIV: row-wise softmax of S in place (each row = one 64-lane f32 reg).
    p_blk = tla.view(P_ptr, (BM, BN), strides=(BN, 1))
    with tla.scope.vector():
        p_ub = tla.alloc_buffer((BM, BN), tl.float32, space="ub")
        # AIV: wait for S, then softmax each row and store P (f32) to GM.
        tla.sync.block_wait_flag(tla.sync.MODE.INTER, tla.sync.PIPE.V, FLAG_S)
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
                tla.simd.vsts(p_ub[r, 0], p)
        tla.copy(p_ub, p_blk)  # hivm.hir.store (MTE3 UB->GM)
        # AIV: signal the P round-trip only after the MTE3 store landed.
        tla.sync.block_set_flag(tla.sync.MODE.INTER, tla.sync.PIPE.MTE3, FLAG_P)

    # AIC: O = P @ V (f32 inputs -> f32 L0C), fixpipe straight to GM.
    v_blk = tla.view(V_ptr, (BN, N), strides=(N, 1))
    o_blk = tla.view(O_ptr, (BM, N), strides=(N, 1))
    with tla.scope.cube():
        p_l1 = tla.alloc_buffer((BN // 8, BM // 16, 16, 8), tl.float32, space="l1")
        v_l1 = tla.alloc_buffer((N // 8, BN // 16, 16, 8), tl.float32, space="l1")
        o_l0c = tla.alloc_buffer((N // 16, BM // 16, 16, 16), tl.float32, space="l0c")
        # AIC: wait for P, then load it back into L1 and accumulate O.
        tla.sync.block_wait_flag(tla.sync.MODE.INTER, tla.sync.PIPE.MTE2, FLAG_P)
        tla.copy(p_blk, p_l1)  # hivm.hir.nd2nz
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
    assert text.count("hivm.hir.pointer_cast") == 5, "tla.view(ptr) x5"
    assert text.count("memref.reinterpret_cast") == 5, "tla.view(ptr) x5"
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(
        text, loads=0, stores=1, buffers=None, extra=("#hivm.address_space<cbuf>", "#hivm.address_space<cc>",
                                                      "#hivm.tcore_type<CUBE>", "#hivm.func_core_type<AIV>"))
    assert text.count("hivm.hir.nd2nz") == 4, "GM->L1 x4 (q, k, p, v)"
    assert text.count("hivm.hir.mmadL1") == 2, "two gemms: S and O"
    assert text.count("hivm.hir.fixpipe") == 2, "L0C->UB/GM x2 (S, O)"
    assert text.count("hivm.hir.sync_block_set") == 2, "AIC/AIV signals x2"
    assert text.count("hivm.hir.sync_block_wait") == 2, "AIV/AIC waits x2"
    assert text.count("ave.hir.vexp") == SHAPE["BM"], "one exp per softmax row"
    assert text.count("ave.hir.vload") == SHAPE["BM"], "one row load per softmax row"

    # Each handoff flag must follow its producer and precede its consumer.
    def nth(pat, k):
        """0-based index of the k-th occurrence (k is 1-based)."""
        pos = -1
        for _ in range(k):
            pos = text.index(pat, pos + 1)
        return pos

    assert text.index("hivm.hir.fixpipe") < text.index("hivm.hir.sync_block_set"), "S flag after fixpipe"
    assert text.index("hivm.hir.sync_block_wait") < text.index("ave.hir.vload"), "softmax waits for S"
    assert nth("hivm.hir.sync_block_set", 2) > nth("hivm.hir.store", 1), "P flag after the P store"
    assert nth("hivm.hir.sync_block_wait", 2) < nth("hivm.hir.nd2nz", 3), "AIC waits before reloading P"
    assert 'mix_mode = "mix"' in text, "cube+vector kernel must be tagged mix"
    assert text.count("hivm.vector_function") == 1, "only the softmax SIMD scope is a VF"
    print("IR PASS: flash_attn HIVM structure ok"
          " (4 nd2nz / 2 mmadL1 / 2 fixpipe / 2 sync set+wait / 1 store / 1 vexp per row)")


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
