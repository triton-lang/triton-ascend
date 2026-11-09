"""TLA 3-in-1 test: cast_back (quant branch A, plan B full-row copy).

Semantics: out[t, c] = f32(x_fp8[t, c]) * scale[t, c // 32]

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline-compile the HIVM IR handed to npuir and assert
            (2 load / 1 store / 3 multibuffer)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (CAST_BACK OK; standard Triton
            reference cast_back_triton, same semantics/shape ->
            TRITON_CAST_BACK OK)

Based on archive/e2e_cast_back_tla_v3.py (plan B): copy block merged to a
full row HID=256, one MTE2 load + one store per row; VF still processes in
64-lane groups.
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

# default shapes: tmax, hid, bm, bk, npc, cores, versions
SHAPE = dict(tmax=8, hid=256, bm=1, bk=64, npc=32, cores=4, versions=2)

# offline compile signature (pointer entry takes pointer args).
SIGNATURE = {"X_ptr": "*fp8e4nv", "SF_ptr": "*fp32", "O_ptr": "*fp32", "num_tokens": "i32"}


@tla.jit
def cast_back(X_ptr, SF_ptr, O_ptr, num_tokens, TMAX: tl.constexpr, HID: tl.constexpr, BM: tl.constexpr,
              BK: tl.constexpr, NPC: tl.constexpr, CORES: tl.constexpr, VERSIONS: tl.constexpr):
    tl.static_assert(HID > 0 and HID % 64 == 0, "HID must be a multiple of 64")
    tl.static_assert(BM == 1 and BK % 64 == 0 and NPC == 32,
                     "BM must be 1 until tail rows are supported; BK multiple of 64, NPC == 32")
    tl.static_assert(TMAX > 0 and CORES > 0)
    # Route A: tile kernel pointer args directly via one tla.view(ptr) each.
    # plan B: copy block = full row HID(256), one MTE2 move per row.
    x_ub = tla.alloc_buffer((BM, HID), tl.float8e4nv, space="ub", versions=VERSIONS)
    sf_ub = tla.alloc_buffer((BM, HID // NPC), tl.float32, space="ub", versions=VERSIONS)
    out_ub = tla.alloc_buffer((BM, HID), tl.float32, space="ub", versions=VERSIONS)
    for pt in range(tl.program_id(0), tl.cdiv(num_tokens, BM), CORES):
        x_blk = tla.view(X_ptr, (BM, HID), strides=(HID, 1), offsets=(pt * BM, 0))
        sf_blk = tla.view(SF_ptr, (BM, HID // NPC), strides=(HID // NPC, 1), offsets=(pt * BM, 0))
        o_blk = tla.view(O_ptr, (BM, HID), strides=(HID, 1), offsets=(pt * BM, 0))
        tla.copy(x_blk, x_ub)
        tla.copy(sf_blk, sf_ub)
        with tla.scope.simd():
            # group mask: first 32 lanes use s0, last 32 use s1.
            low_half = tla.simd.pge(32, 64)
            for c in tl.static_range(HID // 64):
                s0 = tla.simd.vlds(sf_ub[0, (c * 64) // NPC], dist="BRC_B32")
                s1 = tla.simd.vlds(sf_ub[0, (c * 64) // NPC + 1], dist="BRC_B32")
                scale = tla.simd.vsel(low_half, s0, s1)
                v = tla.simd.vlds(x_ub[0, c * 64], dist="UNPK4_B8")
                vf = tla.simd.vextf(v, tl.float32)
                tla.simd.vsts(out_ub[0, c * 64], tla.simd.vmul(vf, scale), dist="NORM_B32")
        tla.copy(out_ub, o_blk)


def constexprs():
    s = SHAPE
    return {
        "TMAX": s["tmax"], "HID": s["hid"], "BM": s["bm"], "BK": s["bk"], "NPC": s["npc"], "CORES": s["cores"],
        "VERSIONS": s["versions"]
    }


# ---- standard Triton reference (same semantics/shape) ----


@triton.jit
def cast_back_triton(X_ptr, SF_ptr, O_ptr, TMAX: tl.constexpr, HID: tl.constexpr):
    pid = tl.program_id(0)
    for task in range(pid, TMAX * (HID // 64), tl.num_programs(0)):
        pt = task // (HID // 64)
        ph = task % (HID // 64)
        cols = ph * 64 + tl.arange(0, 64)
        x = tl.load(X_ptr + pt * HID + cols)  # fp8e4nv
        sg = pt * (HID // 32) + cols // 32  # scale[t, c//32] flattened
        s = tl.load(SF_ptr + sg)
        tl.store(O_ptr + pt * HID + cols, x.to(tl.float32) * s)


def test_ir(output_dir=None):
    module, ctx = compile_offline(cast_back, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    # view (pointer entry) route: AST entry keeps pointers; tla.view(ptr)
    # materializes 3 GM memrefs via hivm.hir.pointer_cast.
    assert "!tt.ptr" in text, "route A entry must keep pointer args at AST stage"
    assert text.count("hivm.hir.pointer_cast") == 3, "tla.view(ptr) x3"
    assert text.count("memref.reinterpret_cast") == 3, "tla.view(ptr) x3"
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(text, loads=2, stores=1, buffers=3)
    print("IR PASS: cast_back HIVM structure ok (2 load / 1 store / 3 multibuffer)")


def _launch():
    s = SHAPE
    X = torch.randn(s["tmax"], s["hid"], dtype=torch.float32, device="npu").to(torch.float8_e4m3fn)
    SF = torch.randn(s["tmax"], s["hid"] // s["npc"], dtype=torch.float32, device="npu")
    O = torch.full((s["tmax"], s["hid"]), 7.0, dtype=torch.float32, device="npu")
    cast_back[(s["cores"], )](X, SF, O, s["tmax"], **constexprs())
    torch.npu.synchronize()
    scale = SF.repeat_interleave(s["npc"], dim=1)  # (tmax, hid)
    ref = X.float() * scale  # fp8 -> f32
    return O, ref


def _launch_triton():
    s = SHAPE
    X = torch.randn(s["tmax"], s["hid"], dtype=torch.float32, device="npu").to(torch.float8_e4m3fn)
    SF = torch.randn(s["tmax"], s["hid"] // s["npc"], dtype=torch.float32, device="npu")
    O = torch.full((s["tmax"], s["hid"]), 7.0, dtype=torch.float32, device="npu")
    cast_back_triton[(s["cores"], )](X, SF, O, TMAX=s["tmax"], HID=s["hid"])
    torch.npu.synchronize()
    scale = SF.repeat_interleave(s["npc"], dim=1)
    ref = X.float() * scale
    return O, ref


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: cast_back generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    O, ref = _launch()
    oc, rc = O.cpu(), ref.cpu()
    bad = ((oc - rc).abs() > 0.05).nonzero().flatten()
    print("O[:8]  :", oc[0, :8].tolist())
    print("ref[:8]:", rc[0, :8].tolist())
    print("num_mismatch:", bad.numel(), bad[:8].tolist())
    ok = torch.allclose(oc, rc, atol=0.05)
    print("CAST_BACK", "OK" if ok else "MISMATCH", "tmax", SHAPE["tmax"], "hid", SHAPE["hid"])
    assert ok, "cast_back view e2e mismatch"
    # standard Triton reference.
    Ot, rt = _launch_triton()
    oc_t, rc_t = Ot.cpu(), rt.cpu()
    bad_t = ((oc_t - rc_t).abs() > 0.05).nonzero().flatten()
    print("TRITON_CAST_BACK", "OK" if bad_t.numel() == 0 else "MISMATCH", "num_mismatch:", bad_t.numel())
    assert bad_t.numel() == 0, "standard triton cast_back mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: cast_back", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
