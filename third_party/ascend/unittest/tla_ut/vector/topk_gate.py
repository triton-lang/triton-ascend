"""TLA 3-in-1 test: moe topk gate (batch TMAX, E=128, K=4; topk index +
non-finite detection).

    idx[r, 2k]   = column of k-th largest score (ties -> smallest col); idx[r, 2k+1] = 0
    flag[r]      = 1.0 if row has NaN/Inf else 0.0   (DBG-01 non-finite check)

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline HIVM IR assert (1 load / 2 store / 3 multibuffer / ave.vmax)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (TOPK_GATE OK)
"""
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import torch
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

# default shapes: tmax, e, k, cores
SHAPE = dict(tmax=8, e=128, k=4, cores=4)

SIGNATURE = {"SCR_ptr": "*fp32", "IDX_ptr": "*i32", "FLAG_ptr": "*fp32", "num_tokens": "i32"}


@tla.jit
def topk_gate(SCR_ptr, IDX_ptr, FLAG_ptr, num_tokens, TMAX: tl.constexpr, E: tl.constexpr, K: tl.constexpr,
              CORES: tl.constexpr):
    tl.static_assert(E > 0 and E <= 384 and K > 0 and K <= 32 and K <= E, "E in [1,384], K in [1, min(E,32)]")
    tl.static_assert(E % 64 == 0, "E must be a multiple of 64 until tail DMA is supported")
    tl.static_assert(TMAX > 0 and CORES > 0)
    # Route A: tile kernel pointer args directly via one tla.view(ptr) each.
    scores_ub = tla.alloc_buffer((((E + 63) // 64) * 64, ), tl.float32, space="ub", versions=6)
    # int64 output (i32-equivalent): low word = index, high word = 0.
    out_ub = tla.alloc_buffer((2 * K, ), tl.int32, space="ub", versions=6)
    flag_ub = tla.alloc_buffer((1, ), tl.float32, space="ub")
    for row in range(tl.program_id(0), num_tokens, CORES):
        if row < num_tokens:
            scr = tla.view(SCR_ptr, (((E + 63) // 64) * 64, ), strides=(1, ), offsets=(row * E, ))
            idx = tla.view(IDX_ptr, (2 * K, ), strides=(1, ), offsets=(row * 2 * K, ))
            flag = tla.view(FLAG_ptr, (1, ), strides=(1, ), offsets=(row, ))
            tla.copy(scr, scores_ub)
            with tla.scope.simd():
                neg_inf = tla.simd.vsplat(float("-inf"), tl.float32)
                zero32 = tla.simd.vsplat(0.0, tl.float32)
                one32 = tla.simd.vsplat(1.0, tl.float32)
                zero_i32 = tla.simd.vsplat(0, tl.int32)
                sentinel = tla.simd.vsplat(E, tl.int32)
                s_raw = ()
                idx_lane = ()
                pad_ok = ()
                for i in tl.static_range((E + 63) // 64):
                    s_raw = s_raw + (tla.simd.vlds(scores_ub[i * 64], dist="NORM"), )
                    idx_lane = idx_lane + (tla.simd.vci(i * 64), )
                    pad_ok = pad_ok + (tla.simd.vcmp(idx_lane[i], sentinel, op=tla.simd.CMP.LT), )
                # DBG-01: non-finite check (active lanes only).
                nf = ()
                for i in tl.static_range((E + 63) // 64):
                    d = tla.simd.vadd(s_raw[i], tla.simd.vneg(s_raw[i]))
                    nf_mask = tla.simd.vcmp(d, zero32, op=tla.simd.CMP.NE)
                    nf_raw = tla.simd.vsel(nf_mask, one32, zero32)
                    nf = nf + (tla.simd.vsel(pad_ok[i], nf_raw, zero32), )
                any_nf = tla.simd.vreduce_max(nf)
                tla.simd.vsts(flag_ub[0], any_nf, dist="ONEPT_B32")
                # padding lane -> -inf。
                sc = ()
                for i in tl.static_range((E + 63) // 64):
                    sc = sc + (tla.simd.vsel(pad_ok[i], s_raw[i], neg_inf), )
                for k in tl.static_range(K):
                    m = tla.simd.vreduce_max(sc)
                    cand = ()
                    for i in tl.static_range((E + 63) // 64):
                        cand = cand + (tla.simd.vsel(tla.simd.vcmp(sc[i], m, op=tla.simd.CMP.EQ), idx_lane[i],
                                                     sentinel), )
                    winner = tla.simd.vreduce_min(cand)
                    tla.simd.vsts(out_ub[2 * k], winner, dist="ONEPT_B32")
                    tla.simd.vsts(out_ub[2 * k + 1], zero_i32, dist="ONEPT_B32")
                    new_sc = ()
                    for i in tl.static_range((E + 63) // 64):
                        new_sc = new_sc + (tla.simd.vsel(tla.simd.vcmp(idx_lane[i], winner, op=tla.simd.CMP.EQ),
                                                         neg_inf, sc[i]), )
                    sc = new_sc
            tla.copy(out_ub, idx)
            tla.copy(flag_ub, flag)


def constexprs():
    s = SHAPE
    return {"TMAX": s["tmax"], "E": s["e"], "K": s["k"], "CORES": s["cores"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(topk_gate, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert "!tt.ptr" in text, "route A entry must keep pointer args at AST stage"
    assert text.count("hivm.hir.pointer_cast") == 3, "tla.view(ptr) x3"
    assert text.count("memref.reinterpret_cast") == 3, "tla.view(ptr) x3"
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(text, loads=1, stores=2, buffers=None, extra=("ave.hir.vmax", ))
    print("IR PASS: topk_gate HIVM structure ok"
          " (1 load / 2 store / 3 multibuffer / ave.vmax)")


def _launch():
    s = SHAPE
    SCR = torch.randn(s["tmax"], s["e"], dtype=torch.float32, device="npu")
    IDX = torch.full((s["tmax"], 2 * s["k"]), -1, dtype=torch.int32, device="npu")
    FLAG = torch.zeros(s["tmax"], dtype=torch.float32, device="npu")
    topk_gate[(s["cores"], )](SCR, IDX, FLAG, s["tmax"], **constexprs())
    torch.npu.synchronize()
    return SCR, IDX, FLAG


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: topk_gate generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    s = SHAPE
    SCR, IDX, FLAG = _launch()
    # numpy reference: iterative max with tie -> smallest index.
    sc = SCR.float().cpu().numpy().copy()
    ref_idx = np.full((s["tmax"], 2 * s["k"]), -1, dtype=np.int64)
    for r in range(s["tmax"]):
        for kk in range(s["k"]):
            mx = sc[r].max()
            winner = int(np.where(sc[r] == mx)[0].min())
            ref_idx[r, 2 * kk] = winner
            ref_idx[r, 2 * kk + 1] = 0
            sc[r, winner] = -np.inf
    idxc = IDX.cpu().numpy()
    bad = (idxc != ref_idx).nonzero()
    print("IDX[0,:8] :", idxc[0].tolist())
    print("ref[0,:8] :", ref_idx[0].tolist())
    print("num_mismatch:", len(bad[0]), "first rows:", [int(i) for i in bad[0][:8].tolist()])
    ok = bool(np.array_equal(idxc, ref_idx))
    flagc = FLAG.cpu().numpy()
    print("FLAG[:8]  :", flagc[:8].tolist(), "(expect all 0 for finite input)")
    ok = ok and bool(np.all(flagc == 0))
    print("TOPK_GATE", "OK" if ok else "MISMATCH", "tmax", s["tmax"], "E", s["e"], "K", s["k"])
    assert ok, "topk_gate view e2e mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: topk_gate", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
