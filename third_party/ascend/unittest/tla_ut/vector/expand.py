"""TLA 3-in-1 test: mhc expand forward.

Semantics: o[n, m, h] = x[n, h]  (broadcast along new MHC dim, bitwise copy)

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline HIVM IR assert (1 load / MHC=2 store / 2 multibuffer)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (EXPAND OK; standard Triton
            reference expand_triton -> TRITON_EXPAND OK)
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

# default shapes: tmax, h, mhc, h_blk, ns, cores
SHAPE = dict(tmax=8, h=256, mhc=2, h_blk=128, ns=2, cores=4)

SIGNATURE = {"X_ptr": "*bf16", "O_ptr": "*bf16", "num_tokens": "i32"}


@triton.jit
def expand_to_mhc_fwd(X_ptr, O_ptr, num_tokens, TMAX: tl.constexpr, H: tl.constexpr, MHC: tl.constexpr,
                      H_BLK: tl.constexpr, NS: tl.constexpr, CORES: tl.constexpr):
    tl.static_assert(H > 0 and H % H_BLK == 0 and H_BLK % 128 == 0,
                     "H must be a multiple of H_BLK and H_BLK a multiple of 128")
    tl.static_assert(MHC > 0 and TMAX > 0 and CORES > 0)
    # Route A: tile kernel pointer args directly via one tla.view(ptr) each.
    x_ub = tla.alloc_buffer((1, H_BLK), tl.bfloat16, space="ub", versions=NS)
    o_ub = tla.alloc_buffer((MHC, 1, H_BLK), tl.bfloat16, space="ub", versions=NS)
    for task in range(tl.program_id(0), num_tokens * (H // H_BLK), CORES):
        n = task // (H // H_BLK)
        hb = task % (H // H_BLK)
        # NOTE(pointer entry): ``strides`` are the *source buffer* strides.
        # They serve both as the offset-folding coefficients (linear =
        # sum(off_i * stride_i), so the shifted address is n*H + hb*H_BLK)
        # and as the final memref strides.  Passing the tile-internal
        # row-major strides (H_BLK, 1) here would fold offsets with the
        # wrong coefficient (n*H_BLK + hb*H_BLK) and read wrong data.
        x_row = tla.view(X_ptr, (1, H_BLK), strides=(H, 1), offsets=(n, hb * H_BLK))
        tla.copy(x_row, x_ub)
        with tla.scope.simd():
            for v in tl.static_range(H_BLK // 128):
                xb = tla.simd.vlds(x_ub[0, v * 128], dist="NORM")
                for m in tl.static_range(MHC):
                    tla.simd.vsts(o_ub[m, 0, v * 128], xb, dist="NORM_B16")
        # One UB->GM move per MHC row.  A single bulk move cannot describe
        # multi-row tiles here: pointer-entry strides double as the offset
        # folding coefficients AND the final memref strides, so the m-row
        # stride (H) would conflict with the n fold coefficient (MHC*H).
        for m in tl.static_range(MHC):
            o_ub_row = tla.view(o_ub, (1, 1, H_BLK), strides=(1, 1, 1), offsets=(m, 0, 0))
            o_row = tla.view(O_ptr, (1, 1, H_BLK), strides=(MHC * H, H, 1), offsets=(n, m, hb * H_BLK))
            tla.copy(o_ub_row, o_row)


def constexprs():
    s = SHAPE
    return {"TMAX": s["tmax"], "H": s["h"], "MHC": s["mhc"], "H_BLK": s["h_blk"], "NS": s["ns"], "CORES": s["cores"]}


# ---- standard Triton reference (same semantics/shape) ----


@triton.jit
def expand_triton(X_ptr, O_ptr, TMAX: tl.constexpr, H: tl.constexpr, MHC: tl.constexpr):
    pid = tl.program_id(0)
    for task in range(pid, TMAX * (H // 128), tl.num_programs(0)):
        n = task // (H // 128)
        hb = task % (H // 128)
        cols = hb * 128 + tl.arange(0, 128)
        x = tl.load(X_ptr + n * H + cols)
        for m in tl.static_range(MHC):
            tl.store(O_ptr + n * (MHC * H) + m * H + cols, x)


def test_ir(output_dir=None):
    module, ctx = compile_offline(expand_to_mhc_fwd, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert "!tt.ptr" in text, "route A entry must keep pointer args at AST stage"
    # one pointer entry for x_row + one per MHC-row o_row view.
    assert text.count("hivm.hir.pointer_cast") == 1 + SHAPE["mhc"], \
        "tla.view(ptr) x (1 x_row + mhc o_row)"
    assert text.count("memref.reinterpret_cast") == 1 + SHAPE["mhc"], \
        "tla.view(ptr) x (1 x_row + mhc o_row)"
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(text, loads=1, stores=SHAPE["mhc"], buffers=2)
    print("IR PASS: expand HIVM structure ok (1 load / 2 store / 2 multibuffer)")


def _launch():
    s = SHAPE
    X = torch.randn(s["tmax"], s["h"], dtype=torch.bfloat16, device="npu")
    O = torch.full((s["tmax"], s["mhc"], s["h"]), -1.0, dtype=torch.bfloat16, device="npu")
    expand_to_mhc_fwd[(s["cores"], )](X, O, s["tmax"], **constexprs())
    torch.npu.synchronize()
    ref = X.unsqueeze(1).expand(s["tmax"], s["mhc"], s["h"])
    return O, ref


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: expand generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    O, ref = _launch()
    oc, rc = O.cpu(), ref.cpu()
    bad = (oc != rc).nonzero().flatten()
    print("O[:4,:4]:", oc[0, 0, :4].tolist(), oc[0, 1, :4].tolist())
    print("X[:4]   :", rc[0, 0, :4].tolist())
    print("num_mismatch:", bad.numel(), bad[:8].tolist())
    ok = torch.allclose(oc, rc, atol=0.0)
    print("EXPAND", "OK" if ok else "MISMATCH", "tmax", SHAPE["tmax"], "mhc", SHAPE["mhc"])
    assert ok, "expand view e2e mismatch"
    # standard Triton reference.
    s = SHAPE
    Xt = torch.randn(s["tmax"], s["h"], dtype=torch.bfloat16, device="npu")
    Ot = torch.full((s["tmax"], s["mhc"], s["h"]), -1.0, dtype=torch.bfloat16, device="npu")
    expand_triton[(s["cores"], )](Xt, Ot, TMAX=s["tmax"], H=s["h"], MHC=s["mhc"])
    torch.npu.synchronize()
    rt = Xt.unsqueeze(1).expand(s["tmax"], s["mhc"], s["h"])
    bad_t = (Ot.cpu() != rt.cpu()).nonzero().flatten()
    print("TRITON_EXPAND", "OK" if bad_t.numel() == 0 else "MISMATCH", "num_mismatch:", bad_t.numel())
    assert bad_t.numel() == 0, "standard triton expand mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: expand", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
