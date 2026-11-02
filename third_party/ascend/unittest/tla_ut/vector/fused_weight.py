"""TLA 3-in-1 test: fused_weight (engram per-token fused weight).

Semantics: o[i] = f32(h[i]) * f32(e[i]), tiled by head * 256-tile.

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline HIVM IR assert (2 load / 1 store / 3 multibuffer / pointer_cast x3)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (FUSED_WEIGHT OK)

Note: original e2e used `tla.copy(o, ou)` with reversed GM<->UB direction
(yields init values); this file fixes it to `tla.copy(ou, o)`.
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tests/
import torch
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

# default shapes: h, hc, cores, versions
SHAPE = dict(h=512, hc=2, cores=4, versions=3)

SIGNATURE = {"H_ptr": "*bf16", "E_ptr": "*bf16", "O_ptr": "*fp32"}


@tla.jit
def fused_weight(H_ptr, E_ptr, O_ptr, H: tl.constexpr, HC: tl.constexpr, CORES: tl.constexpr, VERSIONS: tl.constexpr):
    # Route A: tile kernel pointer args directly via one tla.view(ptr) each.
    hu = tla.alloc_buffer((256, ), tl.bfloat16, space="ub", versions=VERSIONS)
    eu = tla.alloc_buffer((256, ), tl.bfloat16, space="ub", versions=VERSIONS)
    ou = tla.alloc_buffer((256, ), tl.float32, space="ub", versions=VERSIONS)
    num_tiles = H // 256
    for task in range(tl.program_id(0), HC * num_tiles, CORES):
        head = task % HC
        tile = task // HC
        base = head * H + tile * 256
        h = tla.view(H_ptr, (256, ), strides=(1, ), offsets=(base, ))
        e = tla.view(E_ptr, (256, ), strides=(1, ), offsets=(base, ))
        o = tla.view(O_ptr, (256, ), strides=(1, ), offsets=(base, ))
        tla.copy(h, hu)  # GM->UB x2
        tla.copy(e, eu)
        with tla.scope.simd():  # VF 4x64 chunk
            for c in tl.static_range(4):
                a = tla.simd.vlds(hu[c * 64], dist="UNPK_B16")
                af = tla.simd.vextf(a, tl.float32, part="part_even")
                b = tla.simd.vlds(eu[c * 64], dist="UNPK_B16")
                bf = tla.simd.vextf(b, tl.float32, part="part_even")
                tla.simd.vsts(ou[c * 64], tla.simd.vmul(af, bf), dist="NORM_B32")
        tla.copy(ou, o)  # UB->GM (direction was reversed in original e2e; fixed)


def constexprs():
    s = SHAPE
    return {"H": s["h"], "HC": s["hc"], "CORES": s["cores"], "VERSIONS": s["versions"]}


def test_ir(output_dir=None):
    module, ctx = compile_offline(fused_weight, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(module)
    assert "!tt.ptr" in text, "route A entry must keep pointer args at AST stage"
    assert text.count("hivm.hir.pointer_cast") == 3, "tla.view(ptr) x3"
    assert text.count("memref.reinterpret_cast") == 3, "tla.view(ptr) x3"
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    check_lowered_ir(text, loads=2, stores=1, buffers=3)
    print("IR PASS: fused_weight HIVM structure ok (2 load / 1 store / 3 multibuffer)")


def _launch():
    s = SHAPE
    H = torch.randn(s["h"], dtype=torch.bfloat16, device="npu")
    E = torch.randn(s["h"], dtype=torch.bfloat16, device="npu")
    O = torch.empty(s["h"], dtype=torch.float32, device="npu")
    fused_weight[(s["cores"], )](H, E, O, **constexprs())
    torch.npu.synchronize()
    return H, E, O


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: fused_weight generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    H, E, O = _launch()
    oc, ref = O.cpu(), (H.float() * E.float()).cpu()
    bad = ((oc - ref).abs() > 1e-3).nonzero().flatten()
    print("O[:8]  :", oc[:8].tolist())
    print("ref[:8]:", ref[:8].tolist())
    print("num_mismatch:", bad.numel(), bad[:8].tolist())
    ok = torch.allclose(oc, ref, atol=1e-3)
    print("FUSED_WEIGHT", "OK" if ok else "MISMATCH", "H", SHAPE["h"], "HC", SHAPE["hc"])
    assert ok, "fused_weight view e2e mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: fused_weight", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
