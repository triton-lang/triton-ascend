"""TLA 3-in-1 test: simd add/compare interfaces.

Covers the todo.md vector interfaces implemented on PR5:

  vadd    elementwise add, optional mask + MODE
  vadds   vector-scalar add, optional mask + MODE
  vaddc   int32/uint32 add with carry-out -> (sum, carry)
  vcmax   horizontal max reduction
  vcmin   horizontal min reduction
  vcmp    elementwise compare with CMP type + mask

Stages (env TLA_TEST_STAGE, default all with auto-degrade):
  ir      : offline HIVM/AVE IR assert (no NPU, no torch)
  compile : full compile, confirm .o/.npubin artifacts exist
  run     : real-device launch + accuracy (ADD_CMP OK)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # tla_ut/

import triton
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (get_stages, compile_offline, lower_offline, check_compile_artifacts)

N = 64  # f32/i32 vector-register lane count (2048 bits)

SIGNATURE = {
    "A_ptr": "*fp32",
    "B_ptr": "*fp32",
    "C_ptr": "*fp32",
    "S_ptr": "*fp32",
    "MX_ptr": "*fp32",
    "MN_ptr": "*fp32",
    "M_ptr": "*i32",
    "C2_ptr": "*fp32",
    "I_ptr": "*i32",
    "J_ptr": "*i32",
    "SUM_ptr": "*i32",
    "CAR_ptr": "*i32",
    "ULT_ptr": "*i32",
    "UMAX_ptr": "*i32",
    "UMIN_ptr": "*i32",
}


@tla.jit
def add_cmp(A_ptr, B_ptr, C_ptr, S_ptr, MX_ptr, MN_ptr, M_ptr, C2_ptr, I_ptr, J_ptr, SUM_ptr, CAR_ptr, ULT_ptr,
            UMAX_ptr, UMIN_ptr, N: tl.constexpr):
    a_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    b_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    c_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    s_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    mx_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    mn_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    m_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    c2_ub = tla.alloc_buffer((N, ), tl.float32, space="ub")
    i_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    j_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    sum_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    car_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    ult_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    umax_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")
    umin_ub = tla.alloc_buffer((N, ), tl.int32, space="ub")

    a = tla.view(A_ptr, (N, ), strides=(1, ), offsets=(0, ))
    b = tla.view(B_ptr, (N, ), strides=(1, ), offsets=(0, ))
    c = tla.view(C_ptr, (N, ), strides=(1, ), offsets=(0, ))
    s = tla.view(S_ptr, (N, ), strides=(1, ), offsets=(0, ))
    mx = tla.view(MX_ptr, (N, ), strides=(1, ), offsets=(0, ))
    mn = tla.view(MN_ptr, (N, ), strides=(1, ), offsets=(0, ))
    m = tla.view(M_ptr, (N, ), strides=(1, ), offsets=(0, ))
    c2 = tla.view(C2_ptr, (N, ), strides=(1, ), offsets=(0, ))
    i = tla.view(I_ptr, (N, ), strides=(1, ), offsets=(0, ))
    j = tla.view(J_ptr, (N, ), strides=(1, ), offsets=(0, ))
    total = tla.view(SUM_ptr, (N, ), strides=(1, ), offsets=(0, ))
    carry = tla.view(CAR_ptr, (N, ), strides=(1, ), offsets=(0, ))
    ult_out = tla.view(ULT_ptr, (N, ), strides=(1, ), offsets=(0, ))
    umax_out = tla.view(UMAX_ptr, (N, ), strides=(1, ), offsets=(0, ))
    umin_out = tla.view(UMIN_ptr, (N, ), strides=(1, ), offsets=(0, ))

    tla.copy(a, a_ub)
    tla.copy(b, b_ub)
    tla.copy(i, i_ub)
    tla.copy(j, j_ub)

    with tla.scope.simd():
        av = tla.simd.vlds(a_ub[0], dist="NORM")
        bv = tla.simd.vlds(b_ub[0], dist="NORM")
        iv = tla.simd.vlds(i_ub[0], dist="NORM")
        jv = tla.simd.vlds(j_ub[0], dist="NORM")

        # vadd: default zeroing and explicit merging under a half mask.
        tla.simd.vsts(c_ub[0], tla.simd.vadd(av, bv), dist="NORM_B32")
        half = tla.simd.pge(32, N)
        tla.simd.vsts(c2_ub[0], tla.simd.vadd(av, bv, mask=half, mode=tla.simd.MODE.MERGING), dist="NORM_B32")

        # vadds: vector + scalar.
        tla.simd.vsts(s_ub[0], tla.simd.vadds(av, 1.5), dist="NORM_B32")

        # vcmax / vcmin: horizontal reductions, result in lane 0.
        tla.simd.vsts(mx_ub[0], tla.simd.vcmax(av), dist="ONEPT_B32")
        tla.simd.vsts(mn_ub[0], tla.simd.vcmin(av), dist="ONEPT_B32")

        # vcmp: CMP enum + mask, materialize the predicate as i32 0/1.
        one = tla.simd.vsplat(1.0, tl.int32)
        zero = tla.simd.vsplat(0.0, tl.int32)
        gt = tla.simd.vcmp(av, bv, op=tla.simd.CMP.GT)
        tla.simd.vsts(m_ub[0], tla.simd.vsel(gt, one, zero), dist="NORM_B32")

        # vaddc: int32 sum + two's-complement carry-out.
        sv, cv = tla.simd.vaddc(iv, jv)
        tla.simd.vsts(sum_ub[0], sv, dist="NORM_B32")
        tla.simd.vsts(car_ub[0], tla.simd.vsel(cv, one, zero), dist="NORM_B32")

        # Unsigned view: ordered cmp maps to ULT and min/max use umin/umax.
        iuv = tla.simd.vbitcast(iv, "uint32")
        juv = tla.simd.vbitcast(jv, "uint32")
        ult = tla.simd.vcmp(iuv, juv, op=tla.simd.CMP.LT)
        tla.simd.vsts(ult_ub[0], tla.simd.vsel(ult, one, zero), dist="NORM_B32")
        umax = tla.simd.vbitcast(tla.simd.vcmax(iuv), "int32")
        umin = tla.simd.vbitcast(tla.simd.vcmin(iuv), "int32")
        tla.simd.vsts(umax_ub[0], umax, dist="ONEPT_B32")
        tla.simd.vsts(umin_ub[0], umin, dist="ONEPT_B32")

    tla.copy(c_ub, c)
    tla.copy(c2_ub, c2)
    tla.copy(s_ub, s)
    tla.copy(mx_ub, mx)
    tla.copy(mn_ub, mn)
    tla.copy(m_ub, m)
    tla.copy(sum_ub, total)
    tla.copy(car_ub, carry)
    tla.copy(ult_ub, ult_out)
    tla.copy(umax_ub, umax_out)
    tla.copy(umin_ub, umin_out)


def constexprs():
    return {"N": N}


def test_ir(output_dir=None):
    module, ctx = compile_offline(add_cmp, SIGNATURE, constexprs(), output_dir=output_dir)
    module = lower_offline(module, ctx, output_dir=output_dir)
    text = str(module)
    for op in ("ave.hir.vadd", "ave.hir.vadds", "ave.hir.reduction", "ave.hir.vcmp", "ave.hir.vsel"):
        assert op in text, f"missing {op}"
    assert text.count("ave.hir.reduction") == 4, "vcmax/vcmin + umax/umin"
    # vcmp (1) + vaddc carry (1) + unsigned LT (1)
    assert text.count("ave.hir.vcmp") >= 3, "vcmp + vaddc + unsigned"
    assert "ave.hir.vcmp <ULT>" in text, "unsigned LT should map to ULT"
    assert "ave.hir.reduction <umax>" in text, "unsigned max should be umax"
    assert "ave.hir.reduction <umin>" in text, "unsigned min should be umin"
    print("IR PASS: add_cmp AVE structure ok (vadd/vadds/vaddc/"
          "vcmax/vcmin/vcmp)")


def _launch():
    import torch
    A = torch.randn(N, dtype=torch.float32, device="npu")
    B = torch.randn(N, dtype=torch.float32, device="npu")
    # Large magnitudes force wrap-around carries on about half the lanes.
    I = torch.randint(-2**31, 2**31, (N, ), dtype=torch.int32, device="npu")
    J = torch.randint(-2**31, 2**31, (N, ), dtype=torch.int32, device="npu")
    C = torch.zeros(N, dtype=torch.float32, device="npu")
    S = torch.zeros(N, dtype=torch.float32, device="npu")
    MX = torch.zeros(N, dtype=torch.float32, device="npu")
    MN = torch.zeros(N, dtype=torch.float32, device="npu")
    C2 = torch.zeros(N, dtype=torch.float32, device="npu")
    m = torch.zeros(N, dtype=torch.int32, device="npu")
    total = torch.zeros(N, dtype=torch.int32, device="npu")
    carry = torch.zeros(N, dtype=torch.int32, device="npu")
    ult = torch.zeros(N, dtype=torch.int32, device="npu")
    umax = torch.zeros(N, dtype=torch.int32, device="npu")
    umin = torch.zeros(N, dtype=torch.int32, device="npu")
    add_cmp[(1, )](A, B, C, S, MX, MN, m, C2, I, J, total, carry, ult, umax, umin, **constexprs())
    torch.npu.synchronize()
    return (A, B, C, S, MX, MN, m, C2, I, J, total, carry, ult, umax, umin)


def test_compile():
    _launch()
    artifacts = check_compile_artifacts()
    assert artifacts, "no .o/.npubin artifact in TRITON_CACHE_DIR"
    print("COMPILE PASS: add_cmp generated artifacts", len(artifacts), "->", artifacts[0])


def test_run():
    import numpy as np
    (A, B, C, S, MX, MN, m, C2, I, J, total, carry, ult, umax, umin) = _launch()
    a, b = A.cpu().numpy(), B.cpu().numpy()
    i, j = I.cpu().numpy(), J.cpu().numpy().astype(np.int64)
    iu = I.cpu().numpy().astype(np.uint32)
    ok = True
    ok &= np.allclose(C.cpu().numpy(), a + b, atol=1e-5)
    ok &= np.allclose(S.cpu().numpy(), a + np.float32(1.5), atol=1e-5)
    ok &= np.isclose(MX.cpu().numpy()[0], a.max(), atol=1e-5)
    ok &= np.isclose(MN.cpu().numpy()[0], a.min(), atol=1e-5)
    ok &= np.array_equal(m.cpu().numpy(), (a > b).astype(np.int32))
    ref_c2 = a + b
    ref_c2[N // 2:] = 0.0  # MERGING merges +0 into masked-out lanes
    ok &= np.allclose(C2.cpu().numpy(), ref_c2, atol=1e-5)
    ref_sum = (i.astype(np.uint32) + j.astype(np.uint32)).astype(np.uint32)
    ref_carry = (((i.astype(np.uint32).astype(np.uint64) + j.astype(np.uint32).astype(np.uint64)) >> 32)
                 & 1).astype(np.int32)
    ok &= np.array_equal(total.cpu().numpy(), ref_sum.astype(np.int32))
    ok &= np.array_equal(carry.cpu().numpy(), ref_carry)
    ref_ult = (iu < j.astype(np.uint32)).astype(np.int32)
    ok &= np.array_equal(ult.cpu().numpy(), ref_ult)
    umax_got = np.asarray(umax.cpu().numpy()[0:1]).view(np.uint32)[0]
    umin_got = np.asarray(umin.cpu().numpy()[0:1]).view(np.uint32)[0]
    ok &= umax_got == iu.max()
    ok &= umin_got == iu.min()
    print("ADD_CMP", "OK" if ok else "MISMATCH")
    assert ok, "add_cmp e2e mismatch"


def main():
    do_ir, do_compile, do_run = get_stages()
    if do_ir:
        test_ir()
    if do_compile:
        test_compile()
    if do_run:
        test_run()
    print("ALL PASS: add_cmp", "ir" if do_ir else "", "compile" if do_compile else "", "run" if do_run else "")


if __name__ == "__main__":
    main()
