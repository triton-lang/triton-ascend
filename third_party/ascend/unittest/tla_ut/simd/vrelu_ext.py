"""TLA 3-in-1 test: vlrelu and vprelu with four masks (ZEROING only)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import triton
import triton.language as tl
import triton.language.extra.tla as tla
from _simd_ext_common import masks, ref_rows
from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_lowered_ir, check_compile_artifacts)

ROWS = 4
ROWS_C = tl.constexpr(ROWS)
SHAPE = dict(lanes=64, active=32)
SHAPE_F16 = dict(lanes=128, active=64)
SIGNATURE_L = {"X": "*fp32", "T": "*i32", "O": "*fp32"}
SIGNATURE_P = {"X": "*fp32", "A": "*fp32", "T": "*i32", "O": "*fp32"}
SIGNATURE_L16 = {"X": "*fp16", "T": "*i32", "O": "*fp16"}
SIGNATURE_P16 = {"X": "*fp16", "A": "*fp16", "T": "*i32", "O": "*fp16"}
ALPHA = 0.25


@tla.jit
def vlrelu_f32(X, T, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    xb = tla.alloc_buffer((N, ), tl.float32, space="ub")
    tb = tla.alloc_buffer((N, ), tl.int32, space="ub")
    ob = tla.alloc_buffer((ROWS_C, N), tl.float32, space="ub")
    tla.copy(tla.view(X, (N, )), xb)
    tla.copy(tla.view(T, (N, )), tb)
    with tla.scope.simd():
        x = tla.simd.vlds(xb[0])
        t = tla.simd.vlds(tb[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        arb = tla.simd.vcmp(idx, t, op=tla.simd.CMP.LT)
        none = tla.simd.vcmp(x, x, op=tla.simd.CMP.NE)
        tla.simd.vsts(ob[0, 0], tla.simd.vlrelu(x, 0.25))
        tla.simd.vsts(ob[1, 0], tla.simd.vlrelu(x, 0.25, mask=prefix))
        tla.simd.vsts(ob[2, 0], tla.simd.vlrelu(x, 0.25, mask=arb))
        tla.simd.vsts(ob[3, 0], tla.simd.vlrelu(x, 0.25, mask=none))
    tla.copy(ob, tla.view(O, (ROWS_C, N)))


@tla.jit
def vprelu_f32(X, A, T, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    xb = tla.alloc_buffer((N, ), tl.float32, space="ub")
    ab = tla.alloc_buffer((N, ), tl.float32, space="ub")
    tb = tla.alloc_buffer((N, ), tl.int32, space="ub")
    ob = tla.alloc_buffer((ROWS_C, N), tl.float32, space="ub")
    tla.copy(tla.view(X, (N, )), xb)
    tla.copy(tla.view(A, (N, )), ab)
    tla.copy(tla.view(T, (N, )), tb)
    with tla.scope.simd():
        x = tla.simd.vlds(xb[0])
        a = tla.simd.vlds(ab[0])
        t = tla.simd.vlds(tb[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        arb = tla.simd.vcmp(idx, t, op=tla.simd.CMP.LT)
        none = tla.simd.vcmp(x, x, op=tla.simd.CMP.NE)
        tla.simd.vsts(ob[0, 0], tla.simd.vprelu(x, a))
        tla.simd.vsts(ob[1, 0], tla.simd.vprelu(x, a, mask=prefix))
        tla.simd.vsts(ob[2, 0], tla.simd.vprelu(x, a, mask=arb))
        tla.simd.vsts(ob[3, 0], tla.simd.vprelu(x, a, mask=none))
    tla.copy(ob, tla.view(O, (ROWS_C, N)))


@tla.jit
def vlrelu_f16(X, T, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    xb = tla.alloc_buffer((N, ), tl.float16, space="ub")
    tb = tla.alloc_buffer((N, ), tl.int32, space="ub")
    ob = tla.alloc_buffer((ROWS_C, N), tl.float16, space="ub")
    tla.copy(tla.view(X, (N, )), xb)
    tla.copy(tla.view(T, (N, )), tb)
    with tla.scope.simd():
        x = tla.simd.vlds(xb[0])
        t = tla.simd.vlds(tb[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        arb = tla.simd.vcmp(x, tla.simd.vsplat(0.0, tl.float16), op=tla.simd.CMP.LT)
        none = tla.simd.vcmp(x, x, op=tla.simd.CMP.NE)
        tla.simd.vsts(ob[0, 0], tla.simd.vlrelu(x, 0.25), dist="NORM_B16")
        tla.simd.vsts(ob[1, 0], tla.simd.vlrelu(x, 0.25, mask=prefix), dist="NORM_B16")
        tla.simd.vsts(ob[2, 0], tla.simd.vlrelu(x, 0.25, mask=arb), dist="NORM_B16")
        tla.simd.vsts(ob[3, 0], tla.simd.vlrelu(x, 0.25, mask=none), dist="NORM_B16")
    tla.copy(ob, tla.view(O, (ROWS_C, N)))


@tla.jit
def vprelu_f16(X, A, T, O, N: tl.constexpr, ACTIVE: tl.constexpr):
    xb = tla.alloc_buffer((N, ), tl.float16, space="ub")
    ab = tla.alloc_buffer((N, ), tl.float16, space="ub")
    tb = tla.alloc_buffer((N, ), tl.int32, space="ub")
    ob = tla.alloc_buffer((ROWS_C, N), tl.float16, space="ub")
    tla.copy(tla.view(X, (N, )), xb)
    tla.copy(tla.view(A, (N, )), ab)
    tla.copy(tla.view(T, (N, )), tb)
    with tla.scope.simd():
        x = tla.simd.vlds(xb[0])
        a = tla.simd.vlds(ab[0])
        t = tla.simd.vlds(tb[0])
        idx = tla.simd.vci(0)
        prefix = tla.simd.pge(ACTIVE, N)
        arb = tla.simd.vcmp(x, tla.simd.vsplat(0.0, tl.float16), op=tla.simd.CMP.LT)
        none = tla.simd.vcmp(x, x, op=tla.simd.CMP.NE)
        tla.simd.vsts(ob[0, 0], tla.simd.vprelu(x, a), dist="NORM_B16")
        tla.simd.vsts(ob[1, 0], tla.simd.vprelu(x, a, mask=prefix), dist="NORM_B16")
        tla.simd.vsts(ob[2, 0], tla.simd.vprelu(x, a, mask=arb), dist="NORM_B16")
        tla.simd.vsts(ob[3, 0], tla.simd.vprelu(x, a, mask=none), dist="NORM_B16")
    tla.copy(ob, tla.view(O, (ROWS_C, N)))


def constexprs(shape=SHAPE):
    return {"N": shape["lanes"], "ACTIVE": shape["active"]}


def test_ir(output_dir=None):
    cases = ((vlrelu_f32, SIGNATURE_L, SHAPE, "ave.hir.vlrelus"), (vprelu_f32, SIGNATURE_P, SHAPE, "ave.hir.vprelu"),
             (vlrelu_f16, SIGNATURE_L16, SHAPE_F16, "ave.hir.vlrelus"), (vprelu_f16, SIGNATURE_P16, SHAPE_F16,
                                                                         "ave.hir.vprelu"))
    for fn, sig, shape, op_name in cases:
        module, ctx = compile_offline(fn, sig, constexprs(shape), output_dir=output_dir)
        text = str(module)
        assert text.count(op_name) == ROWS
        assert f"ave.hir.pge <VL{shape['active']}>" in text
        text = str(lower_offline(module, ctx, output_dir=output_dir))
        check_lowered_ir(text, loads=3 if fn in (vprelu_f32, vprelu_f16) else 2, stores=1, buffers=None,
                         extra=(op_name, ))
    print("IR PASS: vlrelu/vprelu f16/f32 four masks")


def _launch():
    torch.manual_seed(2026)
    s = SHAPE
    s16 = SHAPE_F16
    x = torch.randn(s["lanes"], dtype=torch.float32) * 2.0
    a = torch.randn(s["lanes"], dtype=torch.float32)
    a[::5] = -0.5
    a[::7] = 0.0
    x16 = (torch.randn(s16["lanes"], dtype=torch.float32) * 2.0).to(torch.float16)
    a16 = torch.randn(s16["lanes"], dtype=torch.float32).to(torch.float16)
    a16[::5] = -0.5
    a16[::7] = 0.0
    mask_set, thr = masks(s["lanes"], s["active"])
    mask_set16, thr16 = masks(s16["lanes"], s16["active"])
    mask_set16["arb"] = x16 < 0
    out_l = torch.empty((ROWS, s["lanes"]), dtype=torch.float32, device="npu")
    out_p = torch.empty((ROWS, s["lanes"]), dtype=torch.float32, device="npu")
    out_l16 = torch.empty((ROWS, s16["lanes"]), dtype=torch.float16, device="npu")
    out_p16 = torch.empty((ROWS, s16["lanes"]), dtype=torch.float16, device="npu")
    vlrelu_f32[(1, )](x.npu(), thr.to(torch.int32).npu(), out_l, **constexprs())
    vprelu_f32[(1, )](x.npu(), a.npu(), thr.to(torch.int32).npu(), out_p, **constexprs())
    vlrelu_f16[(1, )](x16.npu(), thr16.to(torch.int32).npu(), out_l16, **constexprs(s16))
    vprelu_f16[(1, )](x16.npu(), a16.npu(), thr16.to(torch.int32).npu(), out_p16, **constexprs(s16))
    torch.npu.synchronize()
    alpha = torch.tensor(ALPHA, dtype=torch.float32)
    ref_l = torch.where(x >= 0, x, x * alpha)
    ref_p = torch.where(x >= 0, x, x * a)
    ref_l16 = torch.where(x16 >= 0, x16, x16 * torch.tensor(ALPHA, dtype=torch.float16))
    ref_p16 = torch.where(x16 >= 0, x16, x16 * a16)
    return ((out_l.cpu(), ref_rows(ref_l, mask_set)[:ROWS]), (out_p.cpu(), ref_rows(ref_p, mask_set)[:ROWS]),
            (out_l16.cpu(), ref_rows(ref_l16, mask_set16)[:ROWS]), (out_p16.cpu(), ref_rows(ref_p16,
                                                                                            mask_set16)[:ROWS]))


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: vlrelu/vprelu generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    for got, ref in _launch():
        tol = 1e-6 if got.dtype == torch.float32 else 1e-3
        torch.testing.assert_close(got, ref, rtol=tol, atol=tol)
    print("VLRELU/VPRELU OK: f16/f32, four masks, mixed slopes")


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: vlrelu/vprelu", "ir" if do_ir else "",
          "compile/run" if do_run else "compile" if do_compile else "")


if __name__ == "__main__":
    main()
