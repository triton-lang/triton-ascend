"""TLA 3-in-1 test: quant rmsnorm forward.

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline HIVM IR assert (2 load / 1 store / 4 multibuffer, membar alive)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (NORM_FORWARD OK)
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

# default shapes: tmax, hid, pad_h, nacc, eps, versions, cores
SHAPE = dict(tmax=8, hid=256, pad_h=256, nacc=4, eps=1e-5, versions=2, cores=4)

SIGNATURE = {"X_ptr": "*bf16", "W_ptr": "*bf16", "O_ptr": "*fp32", "num_tokens": "i32"}


@triton.jit
def rmsnorm(X_ptr, W_ptr, O_ptr, num_tokens, TMAX: tl.constexpr, HID: tl.constexpr, PAD_H: tl.constexpr,
            NACC: tl.constexpr, WITH_W: tl.constexpr, CORES: tl.constexpr, VERSIONS: tl.constexpr, EPS: tl.constexpr):
    tl.static_assert(HID > 0 and HID == PAD_H and PAD_H % 256 == 0,
                     "HID must equal PAD_H (no tail DMA); PAD_H multiple of 256")
    tl.static_assert(NACC == PAD_H // 64 and CORES > 0 and TMAX > 0, "NACC must equal PAD_H // 64")
    # Route A: tile kernel pointer args directly via one tla.view(ptr) each.
    x_ub = tla.alloc_buffer((1, PAD_H), tl.bfloat16, space="ub", versions=VERSIONS)
    out_ub = tla.alloc_buffer((1, PAD_H), tl.float32, space="ub", versions=VERSIONS)
    w_ub = tla.alloc_buffer((PAD_H, ), tl.bfloat16, space="ub")
    rstd_ub = tla.alloc_buffer((64, ), tl.float32, space="ub")
    for task in range(tl.program_id(0), num_tokens, CORES):
        x_blk = tla.view(X_ptr, (1, PAD_H), strides=(PAD_H, 1), offsets=(task, 0))
        o_blk = tla.view(O_ptr, (1, PAD_H), strides=(PAD_H, 1), offsets=(task, 0))
        tla.copy(x_blk, x_ub)
        if WITH_W:
            w_blk = tla.view(W_ptr, (PAD_H, ), strides=(1, ), offsets=(0, ))
            tla.copy(w_blk, w_ub)
        with tla.scope.simd():
            # UNPK_B16 reads 64 contiguous bf16 values into even sublanes.
            ss = tla.simd.vsplat(0.0, tl.float32)
            for c in tl.static_range(PAD_H // 64):
                a = tla.simd.vextf(tla.simd.vlds(x_ub[0, c * 64], dist="UNPK_B16"), tl.float32, part="part_even")
                ss = tla.simd.vadd(ss, tla.simd.vreduce_sum(tla.simd.vmul(a, a)))
            tla.simd.vsts(rstd_ub[0], ss, dist="ONEPT_B32")
            # SYNC-02: write-after-read barrier.
            tla.simd.membar("VST_VLD")
            v = tla.simd.vlds(rstd_ub[0], dist="BRC_B32")
            rstd_v = tla.math.compensated_rstd(v, HID, EPS)
            for c in tl.static_range(PAD_H // 64):
                x = tla.simd.vextf(tla.simd.vlds(x_ub[0, c * 64], dist="UNPK_B16"), tl.float32, part="part_even")
                y = tla.simd.vmul(x, rstd_v)
                if WITH_W:
                    w = tla.simd.vextf(tla.simd.vlds(w_ub[c * 64], dist="UNPK_B16"), tl.float32, part="part_even")
                    y = tla.simd.vmul(y, w)
                tla.simd.vsts(out_ub[0, c * 64], y, dist="NORM_B32")
        tla.copy(out_ub, o_blk)


def constexprs():
    s = SHAPE
    return {
        "TMAX": s["tmax"], "HID": s["hid"], "PAD_H": s["pad_h"], "NACC": s["nacc"], "WITH_W": True, "CORES": s["cores"],
        "VERSIONS": s["versions"], "EPS": s["eps"]
    }


def test_ir(output_dir=None):
    module, ctx = compile_offline(rmsnorm, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert "!tt.ptr" in text, "route A entry must keep pointer args at AST stage"
    assert text.count("hivm.hir.pointer_cast") == 3, "tla.view(ptr) x3"
    assert text.count("memref.reinterpret_cast") == 3, "tla.view(ptr) x3"
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(text, loads=2, stores=1, buffers=4, extra=("membar", ))
    print("IR PASS: norm_forward HIVM structure ok"
          " (2 load / 1 store / 4 multibuffer / membar alive)")


def _launch():
    s = SHAPE
    X = torch.randn(s["tmax"], s["hid"], dtype=torch.bfloat16, device="npu")
    W = torch.randn(s["hid"], dtype=torch.bfloat16, device="npu")
    O = torch.empty(s["tmax"], s["hid"], dtype=torch.float32, device="npu")
    rmsnorm[(s["cores"], )](X, W, O, s["tmax"], **constexprs())
    torch.npu.synchronize()
    xf = X.float()
    mean_sq = (xf * xf).mean(-1, keepdim=True) + s["eps"]
    rstd = mean_sq.rsqrt()
    ref = xf * rstd * W.float()
    return O, ref


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: norm_forward generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    O, ref = _launch()
    oc, rc = O.cpu(), ref.cpu()
    bad = ((oc - rc).abs() > 1e-3).nonzero().flatten()
    print("O[0,:8]  :", oc[0, :8].tolist())
    print("ref[0,:8]:", rc[0, :8].tolist())
    print("num_mismatch:", bad.numel(), bad[:8].tolist())
    ok = torch.allclose(oc, rc, atol=1e-3)
    print("NORM_FORWARD", "OK" if ok else "MISMATCH", "tmax", SHAPE["tmax"], "hid", SHAPE["hid"])
    assert ok, "norm_forward view e2e mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: norm_forward", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
