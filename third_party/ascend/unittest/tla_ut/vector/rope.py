"""TLA 3-in-1 test: RoPE (NeoX half-split, f32 / RD=128).

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline HIVM IR assert (2 load / 1 store / 2 multibuffer)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (ROPE OK)
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import torch
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

# default shapes: b, s, nh, rd, bt, off, versions, cores
SHAPE = dict(b=1, s=8, nh=4, rd=128, bt=4, off=0, versions=2, cores=4)

SIGNATURE = {"X_ptr": "*fp32", "CS_ptr": "*fp32"}


@tla.jit
def rope(X_ptr, CS_ptr, B: tl.constexpr, S: tl.constexpr, NH: tl.constexpr, RD: tl.constexpr, BT: tl.constexpr,
         OFF: tl.constexpr, CORES: tl.constexpr, VERSIONS: tl.constexpr):
    tl.static_assert(RD == 128 and B > 0 and S > 0 and NH > 0 and BT > 0,
                     "RD must be 128 (first config); positive dims")
    tl.static_assert(S % BT == 0, "main config has no tail block")
    tl.static_assert(CORES > 0 and OFF == 0, "OFF must be zero for the current (S, RD) CS buffer")
    # Route A: tile kernel pointer args directly via one tla.view(ptr) each.
    x_ub = tla.alloc_buffer((1, BT, NH, RD), tl.float32, space="ub", versions=VERSIONS)
    cs_ub = tla.alloc_buffer((BT, RD), tl.float32, space="ub", versions=VERSIONS)
    n_seq_blk = S // BT
    for task in range(tl.program_id(0), B * n_seq_blk, CORES):
        b = task // n_seq_blk
        tb = task % n_seq_blk
        x_blk = tla.view(X_ptr, (1, BT, NH, RD), strides=(S * NH * RD, NH * RD, RD, 1), offsets=(b, tb * BT, 0, 0))
        cs_blk = tla.view(CS_ptr, (BT, RD), strides=(RD, 1), offsets=(tb * BT + OFF, 0))
        tla.copy(x_blk, x_ub)  # in-place alias: write back to same x_blk
        tla.copy(cs_blk, cs_ub)
        with tla.scope.simd():  # x_ub: read_write, cs_ub: read
            for row in tl.static_range(BT):
                c = tla.simd.vlds(cs_ub[row, 0], dist="NORM")  # cos
                s = tla.simd.vlds(cs_ub[row, 64], dist="NORM")  # sin
                for head in tl.static_range(NH):
                    x0 = tla.simd.vlds(x_ub[0, row, head, 0], dist="NORM")
                    x1 = tla.simd.vlds(x_ub[0, row, head, 64], dist="NORM")
                    y0 = tla.simd.vfma(tla.simd.vmul(x1, tla.simd.vneg(s)), x0, c)
                    y1 = tla.simd.vfma(tla.simd.vmul(x1, c), x0, s)
                    tla.simd.vsts(x_ub[0, row, head, 0], y0, dist="NORM_B32")
                    tla.simd.vsts(x_ub[0, row, head, 64], y1, dist="NORM_B32")
        tla.copy(x_ub, x_blk)  # in-place write-back


def constexprs():
    s = SHAPE
    return {
        "B": s["b"], "S": s["s"], "NH": s["nh"], "RD": s["rd"], "BT": s["bt"], "OFF": s["off"], "CORES": s["cores"],
        "VERSIONS": s["versions"]
    }


def test_ir(output_dir=None):
    module, ctx = compile_offline(rope, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert "!tt.ptr" in text, "route A entry must keep pointer args at AST stage"
    assert text.count("hivm.hir.pointer_cast") == 2, "tla.view(ptr) x2"
    assert text.count("memref.reinterpret_cast") == 2, "tla.view(ptr) x2"
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(text, loads=2, stores=1, buffers=2)
    print("IR PASS: rope HIVM structure ok (2 load / 1 store / 2 multibuffer)")


def _launch():
    s = SHAPE
    X = torch.randn(s["b"], s["s"], s["nh"], s["rd"], dtype=torch.float32, device="npu")
    CS = torch.randn(s["s"], s["rd"], dtype=torch.float32, device="npu")
    x_orig = X.cpu().clone()  # X is updated in place by the kernel
    rope[(s["cores"], )](X, CS, **constexprs())
    torch.npu.synchronize()
    x0 = x_orig[..., :64]
    x1 = x_orig[..., 64:]
    cos = CS.cpu()[:, :64].view(1, s["s"], 1, 64)  # broadcast over NH
    sin = CS.cpu()[:, 64:].view(1, s["s"], 1, 64)
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    ref = torch.cat([y0, y1], dim=-1)
    return X, ref


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: rope generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    X, ref = _launch()
    oc = X.cpu()
    bad = ((oc - ref).abs() > 1e-3).nonzero().flatten()
    print("O[0,0,0,:8]  :", oc[0, 0, 0, :8].tolist())
    print("ref[0,0,0,:8]:", ref[0, 0, 0, :8].tolist())
    print("num_mismatch:", bad.numel(), bad[:8].tolist())
    ok = torch.allclose(oc, ref, atol=1e-3)
    print("ROPE", "OK" if ok else "MISMATCH", "B", SHAPE["b"], "S", SHAPE["s"], "NH", SHAPE["nh"])
    assert ok, "rope view e2e mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: rope", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
