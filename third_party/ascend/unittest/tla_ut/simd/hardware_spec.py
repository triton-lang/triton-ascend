"""TLA 3-in-1 test: SIMD hardware-specification coverage.

One f32/i32/u32 kernel exercises all six SIMD interfaces with the hardware
specification boundaries:

  vadd    MODE.ZEROING (all lanes) and MODE.MERGING (mask pge(32, 64))
  vadds   vector-scalar add in both modes
  vaddc   int32 and uint32 two's-complement carry-out
  vcmax   signed f32 and unsigned u32 horizontal max (lane 0)
  vcmin   signed f32 and unsigned u32 horizontal min (lane 0)
  vcmp    EQ/NE/GT/GE/LT/LE plus unsigned UGT/UGE and auto-mapped ULT/ULE

A second, IR-only kernel pins the full-register lane counts of the 2048-bit
vector registers: 128 f16 lanes and 256 i8 lanes.

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline AVE IR structure, modes, compare kinds and lane counts
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (SIMD_SPEC OK)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_compile_artifacts)

N = 64  # f32/i32/u32 lanes in one 2048-bit register
N16 = 128  # f16 lanes
N8 = 256  # i8 lanes
SCALAR = 1.5
SCALAR_MERGE = -1.5

SIGNATURE = {
    "A_ptr": "*fp32",
    "B_ptr": "*fp32",
    "C_ptr": "*fp32",
    "C2_ptr": "*fp32",
    "S_ptr": "*fp32",
    "S2_ptr": "*fp32",
    "MX_ptr": "*fp32",
    "MN_ptr": "*fp32",
    "I_ptr": "*i32",
    "J_ptr": "*i32",
    "EQ_ptr": "*i32",
    "NE_ptr": "*i32",
    "GT_ptr": "*i32",
    "GE_ptr": "*i32",
    "LT_ptr": "*i32",
    "LE_ptr": "*i32",
    "ULT_ptr": "*i32",
    "UGT_ptr": "*i32",
    "UGE_ptr": "*i32",
    "ULE_ptr": "*i32",
    "SUM_ptr": "*i32",
    "CAR_ptr": "*i32",
    "USUM_ptr": "*i32",
    "UCAR_ptr": "*i32",
    "UMAX_ptr": "*i32",
    "UMIN_ptr": "*i32",
}

WIDTH_SIGNATURE = {
    "A16_ptr": "*fp16",
    "B16_ptr": "*fp16",
    "C16_ptr": "*fp16",
    "MX16_ptr": "*fp16",
    "MN16_ptr": "*fp16",
    "M16_ptr": "*fp16",
    "A8_ptr": "*i8",
    "B8_ptr": "*i8",
    "C8_ptr": "*i8",
    "MX8_ptr": "*i8",
    "MN8_ptr": "*i8",
}


@tla.jit
def hardware_spec(A_ptr, B_ptr, C_ptr, C2_ptr, S_ptr, S2_ptr, MX_ptr, MN_ptr, I_ptr, J_ptr, EQ_ptr, NE_ptr, GT_ptr,
                  GE_ptr, LT_ptr, LE_ptr, ULT_ptr, UGT_ptr, UGE_ptr, ULE_ptr, SUM_ptr, CAR_ptr, USUM_ptr, UCAR_ptr,
                  UMAX_ptr, UMIN_ptr, N: tl.constexpr, SCALAR: tl.constexpr, SCALAR_MERGE: tl.constexpr):
    a_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    b_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    c_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    c2_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    s_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    s2_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    mx_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    mn_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    i_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    j_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    eq_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    ne_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    gt_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    ge_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    lt_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    le_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    ult_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    ugt_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    uge_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    ule_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    sum_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    car_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    usum_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    ucar_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    umax_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    umin_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")

    a = tla.view(A_ptr, (N, ), strides=(1, ), offsets=(0, ))
    b = tla.view(B_ptr, (N, ), strides=(1, ), offsets=(0, ))
    i = tla.view(I_ptr, (N, ), strides=(1, ), offsets=(0, ))
    j = tla.view(J_ptr, (N, ), strides=(1, ), offsets=(0, ))

    tla.copy(a, a_ub)
    tla.copy(b, b_ub)
    tla.copy(i, i_ub)
    tla.copy(j, j_ub)

    with tla.scope.simd():
        av = tla.simd.vlds(a_ub[0], dist="NORM")
        bv = tla.simd.vlds(b_ub[0], dist="NORM")
        iv = tla.simd.vlds(i_ub[0], dist="NORM")
        jv = tla.simd.vlds(j_ub[0], dist="NORM")
        half = tla.simd.pge(32, N)
        one = tla.simd.vsplat(1.0, tl.int32)
        zero = tla.simd.vsplat(0.0, tl.int32)

        # vadd: zeroing and merging under the pge(32, 64) half mask.
        tla.simd.vsts(c_ub[0], tla.simd.vadd(av, bv), dist="NORM_B32")
        tla.simd.vsts(c2_ub[0], tla.simd.vadd(av, bv, mask=half, mode=tla.simd.MODE.MERGING), dist="NORM_B32")

        # vadds: both predicate modes and a negative scalar.
        tla.simd.vsts(s_ub[0], tla.simd.vadds(av, SCALAR), dist="NORM_B32")
        tla.simd.vsts(s2_ub[0], tla.simd.vadds(av, SCALAR_MERGE, mask=half, mode=tla.simd.MODE.MERGING),
                      dist="NORM_B32")

        # vcmax/vcmin: signed f32, result in lane 0.
        tla.simd.vsts(mx_ub[0], tla.simd.vcmax(av), dist="ONEPT_B32")
        tla.simd.vsts(mn_ub[0], tla.simd.vcmin(av), dist="ONEPT_B32")

        # vcmp: all six ordered compare kinds on f32.
        tla.simd.vsts(eq_ub[0], tla.simd.vsel(tla.simd.vcmp(av, bv, op=tla.simd.CMP.EQ), one, zero), dist="NORM_B32")
        tla.simd.vsts(ne_ub[0], tla.simd.vsel(tla.simd.vcmp(av, bv, op=tla.simd.CMP.NE), one, zero), dist="NORM_B32")
        tla.simd.vsts(gt_ub[0], tla.simd.vsel(tla.simd.vcmp(av, bv, op=tla.simd.CMP.GT), one, zero), dist="NORM_B32")
        tla.simd.vsts(ge_ub[0], tla.simd.vsel(tla.simd.vcmp(av, bv, op=tla.simd.CMP.GE), one, zero), dist="NORM_B32")
        tla.simd.vsts(lt_ub[0], tla.simd.vsel(tla.simd.vcmp(av, bv, op=tla.simd.CMP.LT), one, zero), dist="NORM_B32")
        tla.simd.vsts(le_ub[0], tla.simd.vsel(tla.simd.vcmp(av, bv, op=tla.simd.CMP.LE), one, zero), dist="NORM_B32")

        # vaddc: int32 sum + carry-out.
        sv, cv = tla.simd.vaddc(iv, jv)
        tla.simd.vsts(sum_ub[0], sv, dist="NORM_B32")
        tla.simd.vsts(car_ub[0], tla.simd.vsel(cv, one, zero), dist="NORM_B32")

        # Unsigned view: LT/LE auto-map to ULT/ULE, UGT/UGE stay explicit.
        iuv = tla.simd.vbitcast(iv, "uint32")
        juv = tla.simd.vbitcast(jv, "uint32")
        tla.simd.vsts(ult_ub[0], tla.simd.vsel(tla.simd.vcmp(iuv, juv, op=tla.simd.CMP.LT), one, zero), dist="NORM_B32")
        tla.simd.vsts(ule_ub[0], tla.simd.vsel(tla.simd.vcmp(iuv, juv, op=tla.simd.CMP.LE), one, zero), dist="NORM_B32")
        tla.simd.vsts(ugt_ub[0], tla.simd.vsel(tla.simd.vcmp(iuv, juv, op=tla.simd.CMP.UGT), one, zero),
                      dist="NORM_B32")
        tla.simd.vsts(uge_ub[0], tla.simd.vsel(tla.simd.vcmp(iuv, juv, op=tla.simd.CMP.UGE), one, zero),
                      dist="NORM_B32")

        # vaddc on uint32 and unsigned vcmax/vcmin (umax/umin kinds).
        usv, ucv = tla.simd.vaddc(iuv, juv)
        tla.simd.vsts(usum_ub[0], tla.simd.vbitcast(usv, "int32"), dist="NORM_B32")
        tla.simd.vsts(ucar_ub[0], tla.simd.vsel(ucv, one, zero), dist="NORM_B32")
        umax = tla.simd.vbitcast(tla.simd.vcmax(iuv), "int32")
        umin = tla.simd.vbitcast(tla.simd.vcmin(iuv), "int32")
        tla.simd.vsts(umax_ub[0], umax, dist="ONEPT_B32")
        tla.simd.vsts(umin_ub[0], umin, dist="ONEPT_B32")

    tla.copy(c_ub, tla.view(C_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(c2_ub, tla.view(C2_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(s_ub, tla.view(S_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(s2_ub, tla.view(S2_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(mx_ub, tla.view(MX_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(mn_ub, tla.view(MN_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(eq_ub, tla.view(EQ_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(ne_ub, tla.view(NE_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(gt_ub, tla.view(GT_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(ge_ub, tla.view(GE_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(lt_ub, tla.view(LT_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(le_ub, tla.view(LE_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(ult_ub, tla.view(ULT_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(ugt_ub, tla.view(UGT_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(uge_ub, tla.view(UGE_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(ule_ub, tla.view(ULE_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(sum_ub, tla.view(SUM_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(car_ub, tla.view(CAR_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(usum_ub, tla.view(USUM_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(ucar_ub, tla.view(UCAR_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(umax_ub, tla.view(UMAX_ptr, (N, ), strides=(1, ), offsets=(0, )))
    tla.copy(umin_ub, tla.view(UMIN_ptr, (N, ), strides=(1, ), offsets=(0, )))


@tla.jit
def simd_spec_widths(A16_ptr, B16_ptr, C16_ptr, MX16_ptr, MN16_ptr, M16_ptr, A8_ptr, B8_ptr, C8_ptr, MX8_ptr, MN8_ptr,
                     N16: tl.constexpr, N8: tl.constexpr):
    a16_ub = tla.alloc_buffer((N16, ), tl.float16, space="ub")
    b16_ub = tla.alloc_buffer((N16, ), tl.float16, space="ub")
    c16_ub = tla.alloc_buffer((N16, ), tl.float16, space="ub")
    mx16_ub = tla.alloc_buffer((N16, ), tl.float16, space="ub")
    mn16_ub = tla.alloc_buffer((N16, ), tl.float16, space="ub")
    m16_ub = tla.alloc_buffer((N16, ), tl.float16, space="ub")
    a8_ub = tla.alloc_buffer((N8, ), tl.int8, space="ub")
    b8_ub = tla.alloc_buffer((N8, ), tl.int8, space="ub")
    c8_ub = tla.alloc_buffer((N8, ), tl.int8, space="ub")
    mx8_ub = tla.alloc_buffer((N8, ), tl.int8, space="ub")
    mn8_ub = tla.alloc_buffer((N8, ), tl.int8, space="ub")

    tla.copy(tla.view(A16_ptr, (N16, ), strides=(1, ), offsets=(0, )), a16_ub)
    tla.copy(tla.view(B16_ptr, (N16, ), strides=(1, ), offsets=(0, )), b16_ub)
    tla.copy(tla.view(A8_ptr, (N8, ), strides=(1, ), offsets=(0, )), a8_ub)
    tla.copy(tla.view(B8_ptr, (N8, ), strides=(1, ), offsets=(0, )), b8_ub)

    with tla.scope.simd():
        a16v = tla.simd.vlds(a16_ub[0], dist="NORM")
        b16v = tla.simd.vlds(b16_ub[0], dist="NORM")
        a8v = tla.simd.vlds(a8_ub[0], dist="NORM")
        b8v = tla.simd.vlds(b8_ub[0], dist="NORM")

        tla.simd.vsts(c16_ub[0], tla.simd.vadd(a16v, b16v), dist="NORM_B16")
        tla.simd.vsts(c16_ub[0], tla.simd.vadds(a16v, 1.0), dist="NORM_B16")
        tla.simd.vsts(mx16_ub[0], tla.simd.vcmax(a16v), dist="ONEPT_B16")
        tla.simd.vsts(mn16_ub[0], tla.simd.vcmin(a16v), dist="ONEPT_B16")
        ge16 = tla.simd.vcmp(a16v, b16v, op=tla.simd.CMP.GE)
        tla.simd.vsts(m16_ub[0], tla.simd.vsel(ge16, a16v, b16v), dist="NORM_B16")

        tla.simd.vsts(c8_ub[0], tla.simd.vadd(a8v, b8v), dist="NORM_B8")
        tla.simd.vsts(mx8_ub[0], tla.simd.vcmax(a8v), dist="ONEPT_B8")
        tla.simd.vsts(mn8_ub[0], tla.simd.vcmin(a8v), dist="ONEPT_B8")

    tla.copy(c16_ub, tla.view(C16_ptr, (N16, ), strides=(1, ), offsets=(0, )))
    tla.copy(mx16_ub, tla.view(MX16_ptr, (N16, ), strides=(1, ), offsets=(0, )))
    tla.copy(mn16_ub, tla.view(MN16_ptr, (N16, ), strides=(1, ), offsets=(0, )))
    tla.copy(m16_ub, tla.view(M16_ptr, (N16, ), strides=(1, ), offsets=(0, )))
    tla.copy(c8_ub, tla.view(C8_ptr, (N8, ), strides=(1, ), offsets=(0, )))
    tla.copy(mx8_ub, tla.view(MX8_ptr, (N8, ), strides=(1, ), offsets=(0, )))
    tla.copy(mn8_ub, tla.view(MN8_ptr, (N8, ), strides=(1, ), offsets=(0, )))


def constexprs():
    return {"N": N, "SCALAR": SCALAR, "SCALAR_MERGE": SCALAR_MERGE}


def width_constexprs():
    return {"N16": N16, "N8": N8}


def test_ir(output_dir=None):
    module, ctx = compile_offline(hardware_spec, SIGNATURE, constexprs(), output_dir=output_dir)
    text = str(lower_offline(module, ctx, output_dir=output_dir))
    for op in ("ave.hir.vadd", "ave.hir.vadds", "ave.hir.reduction", "ave.hir.vcmp", "ave.hir.vsel"):
        assert op in text, f"missing {op}"

    # The i32/u32 vaddc pair aliases after bitcast, so CSE may fold one copy.
    assert text.count("ave.hir.vadd") >= 2, "vadd ZEROING + MERGING"
    assert text.count("ave.hir.vadds") == 2, "vadds ZEROING + MERGING"
    assert text.count("ave.hir.reduction") == 4, "vcmax/vcmin signed + unsigned"
    assert text.count("ave.hir.vcmp") >= 11, "6 ordered + 4 unsigned + carry"

    # Hardware compare kinds: all ordered and all unsigned kinds.
    for kind in ("EQ", "NE", "GT", "GE", "LT", "LE", "UGT", "UGE", "ULT", "ULE"):
        assert f"ave.hir.vcmp <{kind}>" in text, kind
    for kind in ("max", "min", "umax", "umin"):
        assert f"ave.hir.reduction <{kind}>" in text, kind

    # Full-register lane counts for narrower element types.
    width_dir = output_dir / "widths" if output_dir is not None else None
    module, ctx = compile_offline(simd_spec_widths, WIDTH_SIGNATURE, width_constexprs(), output_dir=width_dir)
    widths = str(lower_offline(module, ctx, output_dir=width_dir))
    assert f"vector<{N16}xf16>" in widths, "f16 must fill 128 lanes"
    assert f"vector<{N8}xi8>" in widths, "i8 must fill 256 lanes"
    for op in ("ave.hir.vadd", "ave.hir.vadds", "ave.hir.reduction", "ave.hir.vcmp"):
        assert op in widths, f"missing {op} in width coverage"
    print("IR PASS: SIMD hardware spec (6 interfaces, modes, compare kinds, "
          "64/128/256-lane widths)")


def _launch():
    import torch
    A = torch.randn(N, dtype=torch.float32, device="npu")
    B = torch.randn(N, dtype=torch.float32, device="npu")
    I = torch.randint(-2**31, 2**31, (N, ), dtype=torch.int32, device="npu")
    J = torch.randint(-2**31, 2**31, (N, ), dtype=torch.int32, device="npu")
    f32 = lambda: torch.zeros(N, dtype=torch.float32, device="npu")
    i32 = lambda: torch.zeros(N, dtype=torch.int32, device="npu")
    outputs = [f32() for _ in range(6)] + [i32() for _ in range(16)]
    # Kernel ABI: A, B, six f32 outputs, I, J, sixteen i32 outputs.
    hardware_spec[(1, )](A, B, *outputs[:6], I, J, *outputs[6:], **constexprs())
    torch.npu.synchronize()
    return (A, B, I, J, *outputs)


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: hardware_spec")


def test_run():
    import numpy as np
    (A, B, I, J, C, C2, S, S2, MX, MN, EQ, NE, GT, GE, LT, LE, ULT, UGT, UGE, ULE, SUM, CAR, USUM, UCAR, UMAX,
     UMIN) = _launch()
    a, b = A.cpu().numpy(), B.cpu().numpy()
    i, j = I.cpu().numpy(), J.cpu().numpy()
    iu, ju = i.astype(np.uint32), j.astype(np.uint32)
    ok = True

    ok &= np.allclose(C.cpu().numpy(), a + b, atol=1e-5)
    ref_c2 = a + b
    ref_c2[N // 2:] = 0.0
    ok &= np.allclose(C2.cpu().numpy(), ref_c2, atol=1e-5)
    ok &= np.allclose(S.cpu().numpy(), a + np.float32(SCALAR), atol=1e-5)
    ref_s2 = a + np.float32(SCALAR_MERGE)
    ref_s2[N // 2:] = 0.0
    ok &= np.allclose(S2.cpu().numpy(), ref_s2, atol=1e-5)
    ok &= np.isclose(MX.cpu().numpy()[0], a.max(), atol=1e-5)
    ok &= np.isclose(MN.cpu().numpy()[0], a.min(), atol=1e-5)

    for got, ref in (
        (EQ, a == b),
        (NE, a != b),
        (GT, a > b),
        (GE, a >= b),
        (LT, a < b),
        (LE, a <= b),
        (ULT, iu < ju),
        (UGT, iu > ju),
        (UGE, iu >= ju),
        (ULE, iu <= ju),
    ):
        ok &= np.array_equal(got.cpu().numpy(), ref.astype(np.int32))

    ref_sum = (iu + ju).astype(np.uint32)
    ref_carry = (((iu.astype(np.uint64) + ju.astype(np.uint64)) >> 32) & 1).astype(np.int32)
    ok &= np.array_equal(SUM.cpu().numpy(), ref_sum.view(np.int32))
    ok &= np.array_equal(CAR.cpu().numpy(), ref_carry)
    ok &= np.array_equal(USUM.cpu().numpy(), ref_sum.view(np.int32))
    ok &= np.array_equal(UCAR.cpu().numpy(), ref_carry)

    umax_got = np.asarray(UMAX.cpu().numpy()[0:1]).view(np.uint32)[0]
    umin_got = np.asarray(UMIN.cpu().numpy()[0:1]).view(np.uint32)[0]
    ok &= umax_got == iu.max()
    ok &= umin_got == iu.min()

    print("SIMD_SPEC", "OK" if ok else "MISMATCH")
    assert ok, "hardware_spec e2e mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: hardware_spec", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
