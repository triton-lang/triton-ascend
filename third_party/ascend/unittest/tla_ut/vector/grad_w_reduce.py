"""TLA 3-in-1 test: engram grad_w_reduce.

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline HIVM IR assert (5 load / 2 store / 5 multibuffer)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (GRAD_W_REDUCE OK)
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import torch
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

# default shapes: h, hc, np, bw, versions, cores
SHAPE = dict(h=256, hc=2, np=2, bw=128, versions=1, cores=4)

SIGNATURE = {"P_ptr": "*fp32", "WH_ptr": "*bf16", "WE_ptr": "*bf16", "GH_ptr": "*fp32", "GE_ptr": "*fp32"}


@tla.jit
def grad_w_reduce(P_ptr, WH_ptr, WE_ptr, GH_ptr, GE_ptr, H: tl.constexpr, HC: tl.constexpr, NP: tl.constexpr,
                  BW: tl.constexpr, CORES: tl.constexpr, VERSIONS: tl.constexpr):
    tl.static_assert(H > 0 and H % BW == 0 and BW % 64 == 0,
                     "H must be a positive multiple of BW and BW a multiple of 64")
    tl.static_assert(NP > 0 and HC > 0 and CORES > 0)
    # Route A: tile kernel pointer args directly via one tla.view(ptr) each.
    # UB 5-buffer: partial (NP,1,BW) matches strided GM view; two read_write grads.
    p_ub = tla.alloc_buffer((NP, 1, BW), tl.float32, space="ub", versions=VERSIONS)
    wh_ub = tla.alloc_buffer((1, BW), tl.bfloat16, space="ub", versions=VERSIONS)
    we_ub = tla.alloc_buffer((1, BW), tl.bfloat16, space="ub", versions=VERSIONS)
    gh_ub = tla.alloc_buffer((1, BW), tl.float32, space="ub", versions=VERSIONS)
    ge_ub = tla.alloc_buffer((1, BW), tl.float32, space="ub", versions=VERSIONS)
    for task in range(tl.program_id(0), (H // BW) * HC, CORES):
        shard = task // HC
        head = task % HC
        off = shard * BW
        # strided source: (head, off:off+BW) across all NP partials in one move.
        p_blk = tla.view(P_ptr, (NP, 1, BW), strides=(HC * H, H, 1), offsets=(0, head, off))
        wh = tla.view(WH_ptr, (1, BW), strides=(H, 1), offsets=(head, off))
        we = tla.view(WE_ptr, (1, BW), strides=(H, 1), offsets=(head, off))
        gh = tla.view(GH_ptr, (1, BW), strides=(H, 1), offsets=(head, off))
        ge = tla.view(GE_ptr, (1, BW), strides=(H, 1), offsets=(head, off))
        tla.copy(p_blk, p_ub)
        tla.copy(wh, wh_ub)
        tla.copy(we, we_ub)
        tla.copy(gh, gh_ub)
        tla.copy(ge, ge_ub)
        with tla.scope.simd():
            # per register block: serially accumulate NP partials, fma back to gh/ge.
            for c in tl.static_range(BW // 64):
                acc = tla.simd.vsplat(0.0, tl.float32)
                for q in tl.static_range(NP):
                    pq = tla.simd.vlds(p_ub[q, 0, c * 64], dist="NORM")
                    acc = tla.simd.vadd(acc, pq)
                a = tla.simd.vlds(gh_ub[0, c * 64], dist="NORM")
                b = tla.simd.vlds(ge_ub[0, c * 64], dist="NORM")
                w_he = tla.simd.vextf(tla.simd.vlds(we_ub[0, c * 64], dist="UNPK_B16"), tl.float32, part="part_even")
                w_hh = tla.simd.vextf(tla.simd.vlds(wh_ub[0, c * 64], dist="UNPK_B16"), tl.float32, part="part_even")
                # simd.vfma(x, y, z) = x + y*z -> grad_new = grad_old + acc*w。
                tla.simd.vsts(gh_ub[0, c * 64], tla.simd.vfma(a, acc, w_he), dist="NORM_B32")
                tla.simd.vsts(ge_ub[0, c * 64], tla.simd.vfma(b, acc, w_hh), dist="NORM_B32")
        tla.copy(gh_ub, gh)
        tla.copy(ge_ub, ge)


def constexprs():
    s = SHAPE
    return {"H": s["h"], "HC": s["hc"], "NP": s["np"], "BW": s["bw"], "CORES": s["cores"], "VERSIONS": s["versions"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(grad_w_reduce, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert "!tt.ptr" in text, "route A entry must keep pointer args at AST stage"
    assert text.count("hivm.hir.pointer_cast") == 5, "tla.view(ptr) x5"
    assert text.count("memref.reinterpret_cast") == 5, "tla.view(ptr) x5"
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(text, loads=5, stores=2, buffers=5)
    print("IR PASS: grad_w_reduce HIVM structure ok (5 load / 2 store / 5 multibuffer)")


def _launch():
    s = SHAPE
    P = torch.randn(s["np"], s["hc"], s["h"], dtype=torch.float32, device="npu")
    WH = torch.randn(s["hc"], s["h"], dtype=torch.bfloat16, device="npu")
    WE = torch.randn(s["hc"], s["h"], dtype=torch.bfloat16, device="npu")
    GH = torch.randn(s["hc"], s["h"], dtype=torch.float32, device="npu")
    GE = torch.randn(s["hc"], s["h"], dtype=torch.float32, device="npu")
    gh_ref = GH.cpu().clone()
    ge_ref = GE.cpu().clone()
    grad_w_reduce[(s["cores"], )](P, WH, WE, GH, GE, **constexprs())
    torch.npu.synchronize()
    # reference: acc = sum_p P[:, head, off:off+BW]; grad_new = grad_old + acc * w
    acc = P.cpu().sum(dim=0)  # (hc, h)
    gh_ref += acc * WE.cpu().float()
    ge_ref += acc * WH.cpu().float()
    return GH, GE, gh_ref, ge_ref


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: grad_w_reduce generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    GH, GE, gh_ref, ge_ref = _launch()
    ghc, gec = GH.cpu(), GE.cpu()
    bad = (((ghc - gh_ref).abs() > 1e-2) | ((gec - ge_ref).abs() > 1e-2)).nonzero().flatten()
    print("GH[0,:8]  :", ghc[0, :8].tolist())
    print("ref[0,:8] :", gh_ref[0, :8].tolist())
    print("GE[0,:8]  :", gec[0, :8].tolist())
    print("ref[0,:8] :", ge_ref[0, :8].tolist())
    print("num_mismatch:", bad.numel(), bad[:8].tolist())
    ok = (torch.allclose(ghc, gh_ref, atol=1e-2) and torch.allclose(gec, ge_ref, atol=1e-2))
    print("GRAD_W_REDUCE", "OK" if ok else "MISMATCH", "H", SHAPE["h"], "HC", SHAPE["hc"], "NP", SHAPE["np"])
    assert ok, "grad_w_reduce view e2e mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: grad_w_reduce", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
